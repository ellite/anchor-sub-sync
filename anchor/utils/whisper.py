import gc
import sys
import time
import os
import torch
import whisperx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from .ui import make_ui_console, CaptureProgress
from .files import open_subtitle, backup_if_needed
from .alignment import GlobalAligner

console = Console()

def load_whisper_model(device, compute_type, language, model_size="large-v3"):
    # Safe console capture: duplicate stdout and wrap in a file object
    real_stdout_fd = os.dup(1)
    safe_file = os.fdopen(real_stdout_fd, "w")
    safe_console = Console(file=safe_file)

    model = None
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=safe_console,
            transient=True
        ) as progress:
            lang_str = language.upper() if language else "AUTO"
            task_id = progress.add_task(f"[cyan]Loading model ({lang_str})...", total=None)

            with CaptureProgress(progress, task_id):
                model = whisperx.load_model(model_size, device, compute_type=compute_type, language=language)

            safe_console.print("[dim]🤖 Model loaded.[/dim]")
    finally:
        try:
            if safe_console and getattr(safe_console, 'file', None):
                try:
                    safe_console.file.flush()
                except Exception:
                    pass
                try:
                    safe_console.file.close()
                except Exception:
                    pass
        except Exception:
            pass

    return model

def run_whisper_transcription(video_path, device, compute_type, batch_size, model, language=None):
    """Transcribes audio and aligns phonemes. Returns (whisper_data, detected_lang) or (None, None) on failure."""
    safe_console = Console(force_terminal=True)
    try:
        audio = whisperx.load_audio(str(video_path))
    except Exception as e:
        safe_console.print(f"[bold red]❌ Failed to load audio: {e}[/bold red]")
        return None, None

    result = None
    current_batch_size = batch_size
    is_windows = (os.name != 'posix')

    while current_batch_size >= 1:
        try:
            sys.stdout.flush()
            sys.stderr.flush()

            ui_console = make_ui_console()

            columns = [SpinnerColumn(), TextColumn("[progress.description]{task.description}")]
            if not is_windows:
                columns.append(BarColumn())
                columns.append(TextColumn("[progress.percentage]{task.percentage:>3.0f}%"))
            columns.append(TimeElapsedColumn())

            with Progress(
                *columns,
                console=ui_console,
                transient=True,
                refresh_per_second=10
            ) as progress:
                ui_console.print(f"[dim]🎤 Transcribing audio (Batch Size: {current_batch_size}, Compute: {compute_type})...[/dim]")
                task_id = progress.add_task(f"[cyan]Transcribing...", total=100 if not is_windows else None)
                progress.refresh()

                with CaptureProgress(progress, task_id, ui_console=ui_console):
                    result = model.transcribe(
                        audio,
                        batch_size=current_batch_size,
                        language=language,
                        print_progress=not is_windows,
                        combined_progress=False
                    )
            break

        except Exception as e:
            error_msg = str(e).lower()
            is_oom = any(x in error_msg for x in ["cuda", "out of memory", "alloc", "cudnn"])

            if current_batch_size == 1 or not is_oom:
                safe_console.print(f"[bold red]❌ Fatal Error: {e}[/bold red]")
                return None, None

            os.write(1, f"\033[93m⚠️ Batch size {current_batch_size} failed. Retrying with {current_batch_size // 2}...\033[0m\n".encode())
            if "cuda" in str(device):
                try:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                except: pass

            gc.collect()
            current_batch_size //= 2
            time.sleep(1)

    if not result:
        return None, None

    del audio
    gc.collect()
    if device == "cuda": torch.cuda.empty_cache()

    detected_lang = result.get("language", "unknown")
    console.print(f"[dim]📝 Transcription complete. [bold cyan]Detected language: {detected_lang.upper()}[/bold cyan][/dim]")

    # Align Phonemes
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True
    ) as progress:
        progress.add_task("[cyan] Aligning phonemes...", total=None)

        try:
            model_a, metadata = whisperx.load_align_model(language_code=detected_lang, device=device)
            audio_for_align = whisperx.load_audio(str(video_path))

            aligned_result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio_for_align,
                device,
                return_char_alignments=False,
            )
            segments = aligned_result["segments"]

            del model_a; del audio_for_align; gc.collect()
            if device == "cuda": torch.cuda.empty_cache()

        except Exception as e:
            console.print(f"[yellow]⚠️ Phoneme alignment failed ({e}). Using raw timestamps.[/yellow]")
            segments = result["segments"]

    console.print("[dim]📏 Phoneme alignment complete.[/dim]")

    whisper_data = [
        {'start': seg['start'], 'end': seg['end'], 'text': seg['text'], 'words': seg.get('words', [])}
        for seg in segments
    ]

    return whisper_data, detected_lang


def run_anchor_align_and_sync(sub_path, whisper_data, args=None):
    """Runs GlobalAligner on pre-computed whisper data and saves the synced subtitle."""
    console.print("[dim]🧮 Calculating sync offsets...[/dim]")
    original_subs = open_subtitle(sub_path)
    aligner = GlobalAligner(original_subs, whisper_data)
    synced_subs, rejected = aligner.run()

    if synced_subs is None:
        raise Exception("Zero matches found.")

    if args and getattr(args, "overwrite", False):
        backup_if_needed(sub_path, args)
        output_path = sub_path
        console.print(f"[dim]💾 Overwriting original subtitle: {output_path.name}[/dim]")
    else:
        output_path = sub_path.with_name(f"{sub_path.stem}.synced{sub_path.suffix}")

    synced_subs.save(str(output_path))

    return output_path, len(original_subs), rejected
