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
from .alignment import GlobalAligner, drop_implausible_timing, quiet_library_logs

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

def repair_suspicious_segments(segments, video_path, model_name, device, compute_type, language, cpu_threads=0):
    """Re-transcribes zones where Whisper's output looks broken (same detector and repair as the
    transcribe task: long segments with little text, dragging or flat timing, run-on text).

    WhisperX batches 30 s VAD chunks; in loud scenes a chunk can come back with only its first
    sentence and a timestamp spanning the whole chunk. faster-whisper on the padded zone usually
    recovers the missing speech. Two passes (small then large padding), a repair is kept only when
    its quality score beats the original. Returns (segments, suspicious_zone_count, repaired_count).
    """
    from ..core.transcribe import transcribe as T

    segs = [dict(s) for s in segments]
    zone_count = repaired = 0
    fw = None   # one faster-whisper model for every zone, loaded when the first zone needs it
    for padding in (T.REPAIR_PADDING_PASS_1, T.REPAIR_PADDING_PASS_2):
        suspicious = [i for i, seg in enumerate(segs) if T.is_suspicious(seg, i, segs)]
        if not suspicious:
            break
        zones = T.merge_suspicious_zones(suspicious)
        if padding == T.REPAIR_PADDING_PASS_1:
            zone_count = len(zones)
        for z_start, z_end in reversed(zones):
            bad = segs[z_start:z_end + 1]
            original = [dict(s, text=T.clean_text(s.get("text", ""))) for s in bad if T.clean_text(s.get("text", ""))]
            if fw is None:
                fw = T.WhisperModel(model_name, device=device, compute_type=compute_type, cpu_threads=cpu_threads)
            fixed = T.repair_zone_best(model_name, device, compute_type, video_path, bad, padding, language, cpu_threads, whisper_model=fw)
            if T.zone_quality_score(fixed) > T.zone_quality_score(original) + 0.3:
                final = fixed
                repaired += 1
            else:
                final = original
            segs[z_start:z_end + 1] = final
            first, last = z_start, z_start + len(final) - 1
            segs = T.stitch_boundaries(segs, first, last)
            segs = T.dedupe_window(segs, first, last)
        segs = T.cleanup_redundancies(segs)
    if fw is not None:
        del fw
        gc.collect()
        if device == "cuda": torch.cuda.empty_cache()
    return segs, zone_count, repaired


def run_whisper_transcription(video_path, device, compute_type, batch_size, model, language=None, model_name=None, cpu_threads=0):
    """Transcribes audio and aligns phonemes. Returns (whisper_data, detected_lang) or (None, None) on failure.

    model_name (the Whisper model size) enables the repair pass for suspicious zones; without it that pass is skipped.
    """
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

    if model_name and result["segments"]:
        try:
            segments, zones, repaired = repair_suspicious_segments(
                result["segments"], video_path, model_name, device, compute_type, detected_lang, cpu_threads)
            result["segments"] = segments
            if zones:
                console.print(f"[dim]🔧 Repair pass: {zones} suspicious zone(s), {repaired} repaired.[/dim]")
        except Exception as e:
            console.print(f"[yellow]⚠️ Repair pass failed ({e}). Using the original transcription.[/yellow]")

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

            with quiet_library_logs():
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

    whisper_data, dropped = drop_implausible_timing(whisper_data)
    if dropped:
        console.print(f"[dim]🧹 Ignored {dropped} words with implausible timing.[/dim]")

    return whisper_data, detected_lang


def align_subtitles(subs, whisper_data, precise=False):
    """Runs GlobalAligner on an in-memory subtitle. Retimes the events of `subs` in place.

    Returns (synced_subs, rejected_count, anchor_count). The synced file may be re-sorted, but its events are
    the same objects as in `subs`, so callers holding references to them see the new timings.
    """
    console.print("[dim]🧮 Calculating sync offsets...[/dim]")
    aligner = GlobalAligner(subs, whisper_data, precise=precise)
    synced_subs, rejected = aligner.run()

    if synced_subs is None:
        raise Exception("Zero matches found.")

    return synced_subs, rejected, aligner.anchor_count


def check_subtitles(subs, whisper_data, precise=False, video_fps=None):
    """Measures how far `subs` is from the speech without changing it. Returns the dict from syncverdict.assess."""
    from . import alignment
    from .syncverdict import assess
    aligner = GlobalAligner(subs, whisper_data, precise=precise)
    was_quiet, alignment.console.quiet = alignment.console.quiet, True
    try:
        aligner.run()
    finally:
        alignment.console.quiet = was_quiet
    duration = max((w['end'] for s in whisper_data for w in s.get('words', []) if w.get('end') is not None), default=None)
    return assess(aligner.anchor_points, len(subs), duration, video_fps)


def run_anchor_align_and_sync(sub_path, whisper_data, args=None, precise=False):
    """Runs GlobalAligner on pre-computed whisper data and saves the synced subtitle."""
    original_subs = open_subtitle(sub_path)
    synced_subs, rejected, anchors = align_subtitles(original_subs, whisper_data, precise=precise)

    if args and getattr(args, "overwrite", False):
        backup_if_needed(sub_path, args)
        output_path = sub_path
        console.print(f"[dim]💾 Overwriting original subtitle: {output_path.name}[/dim]")
    else:
        output_path = sub_path.with_name(f"{sub_path.stem}.synced{sub_path.suffix}")

    synced_subs.save(str(output_path))

    return output_path, len(original_subs), rejected, anchors
