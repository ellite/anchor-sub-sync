import gc
import sys
import time
import os
import torch
import whisperx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from .ui import make_ui_console, CaptureProgress
from .files import open_subtitle
from .alignment import GlobalAligner

#for auto-detect-fix
import ffmpeg
import numpy as np
from collections import Counter

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

def run_anchor_sync(video_path, sub_path, device, compute_type, batch_size, model, language=None, args=None):
    safe_console = Console(force_terminal=True)
    try:
        audio = whisperx.load_audio(str(video_path))
    except Exception as e:
        safe_console.print(f"[bold red]❌ Failed to load audio: {e}[/bold red]")
        return None, 0, 0

    result = None 
    current_batch_size = batch_size
    is_windows = (os.name != 'posix')
    
    while current_batch_size >= 1:
        try:
            sys.stdout.flush()
            sys.stderr.flush()

            ui_console = make_ui_console()

            # DYNAMIC UI
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
                        print_progress=not is_windows,  # Linux/Mac=True, Windows=False
                        combined_progress=False
                    )
            break

        except Exception as e:
            error_msg = str(e).lower()
            is_oom = any(x in error_msg for x in ["cuda", "out of memory", "alloc", "cudnn"])
            
            if current_batch_size == 1 or not is_oom:
                safe_console.print(f"[bold red]❌ Fatal Error: {e}[/bold red]")
                return None, 0, 0

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
        return None, 0, 0

    del audio
    gc.collect()
    if device == "cuda": torch.cuda.empty_cache()
    
    console.print(f"[dim]📝 Transcription complete.[/dim]")

    # Align Phonemes
    # Use a spinner context to show activity during the CPU-heavy task
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True
    ) as progress:
        progress.add_task("[cyan] Aligning phonemes...", total=None)
        
        try:
            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
            audio_for_align = whisperx.load_audio(str(video_path))
            
            # Pass print_progress=False to disable the ugly tqdm bar and rely on clean Rich spinner instead.
            aligned_result = whisperx.align(
                result["segments"], 
                model_a, 
                metadata, 
                audio_for_align, 
                device, 
                return_char_alignments=False,
            )
            segments = aligned_result["segments"]
            
            # Cleanup
            del model_a; del audio_for_align; gc.collect(); 
            if device == "cuda": torch.cuda.empty_cache()
            
        except Exception as e:
            # If it fails, we catch it inside the spinner, print error, and continue
            console.print(f"[yellow]⚠️ Phoneme alignment failed ({e}). Using raw timestamps.[/yellow]")
            segments = result["segments"]

    console.print("[dim]📏 Phoneme alignment complete.[/dim]")

    # Prepare Data
    whisper_data = []
    for seg in segments:
        whisper_data.append({
            'start': seg['start'], 'end': seg['end'],
            'text': seg['text'], 'words': seg.get('words', [])
        })

    # Global Aligner
    console.print("[dim]🧮 Calculating sync offsets...[/dim]")
    original_subs = open_subtitle(sub_path)
    aligner = GlobalAligner(original_subs, whisper_data)
    synced_subs, rejected = aligner.run()
    
    if synced_subs is None:
        raise Exception("Zero matches found.")

    # Respect overwrite flag: when args.overwrite is True, overwrite the input file
    if args and getattr(args, "overwrite", False):
        output_path = sub_path
        console.print(f"[dim]💾 Overwriting original subtitle: {output_path.name}[/dim]")
    else:
        output_path = sub_path.with_name(f"{sub_path.stem}.synced{sub_path.suffix}")

    synced_subs.save(str(output_path))
    
    return output_path, len(original_subs), rejected
    

def detect_audio_language_whisper(video_path, device, compute_type):
    """
    Extracts three 30-second audio samples (at 30%, 50%, and 70%) and uses WhisperX
    to detect the language, using a majority vote to ensure high accuracy.
    """
    try:
        # Probe the video (silently) to find its total duration
        probe = ffmpeg.probe(str(video_path), v='error')
        duration = float(probe['format']['duration'])

        # Define our three jump points (30%, 50%, 70%)
        fractions = [0.3, 0.5, 0.7]
        detected_languages = []

        # Load the model ONCE outside the loop to save massive amounts of time
        model = whisperx.load_model("base", device, compute_type=compute_type, asr_options={"without_timestamps": True})

        # Loop through our three points in the video
        for frac in fractions:
            start_time = duration * frac

            try:
                # Extract 30 seconds, silencing FFmpeg
                out, _ = (
                    ffmpeg
                    .input(str(video_path), ss=start_time, t=30)
                    .output('-', format='s16le', acodec='pcm_s16le', ac=1, ar='16k')
                    .global_args('-loglevel', 'error')
                    .run(capture_stdout=True, capture_stderr=True)
                )

                # Convert the raw audio buffer
                audio = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

                # Analyze this specific clip
                result = model.transcribe(audio)

                # If speech was found, add it to our voting pool
                if "language" in result and result["language"] != "unknown":
                    detected_languages.append(result["language"])

            except Exception:
                # If one specific clip fails (e.g., end of file), just ignore it and try the next
                continue

        # --- THE VOTING SYSTEM ---
        if detected_languages:
            # Counter counts occurrences. most_common(1) returns e.g., [('en', 2)]
            # We then extract just the language code from that nested result.
            winner = Counter(detected_languages).most_common(1)[0][0]
            return winner
        else:
            # Absolute fallback if ALL three clips were purely silent
            return "en"

    except Exception as e:
        # Fallback if the entire file is unreadable
        return "en"
