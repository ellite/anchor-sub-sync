import sys
import time
import gc
import copy
import torch
from pathlib import Path
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, BarColumn, TaskProgressColumn
from ...utils.files import get_files, find_best_video_match, select_video_fallback, open_subtitle, select_files_interactive, backup_if_needed, unique_path
from ...utils.mappings import get_language_code_for_nllb
from ...utils.languages import get_audio_language, get_subtitle_language, get_video_fps
from ...utils.whisper import run_whisper_transcription, run_anchor_align_and_sync, align_subtitles, load_whisper_model, check_subtitles
from ...utils import syncverdict
from ...utils import parakeet
from ..translation import translate_subtitle_nllb

ALT_TRANSLATIONS = 4   # extra translations kept per cue for matching (see translation.translate_subtitle_nllb)

# Constants
SUPPORTED_EXTENSIONS = {".srt", ".ass", ".vtt", ".sub"}

def _fix_policy(args, check, queue_len):
    """What a Sync Check does after reporting a file.

    'auto'   --check --fix (with or without -s): take the advice, never ask
    'report' --check on the command line (with or without -s): only report, never ask
    'batch'  several files from the menu: not interrupted, the summary shows what needs fixing
    'ask'    one file from the menu: offer the fixes
    None     not a check
    """
    if not check:
        return None
    if args.subtitle or getattr(args, "report_only", False):
        return "auto" if getattr(args, "fix", False) else "report"
    return "batch" if queue_len > 1 else "ask"


def _offer_fix(console, result):
    """After an interactive Sync Check: what to do about the finding. Returns 'framerate', 'audio' or None.

    The frame rate change is offered only when the drift matches a frame rate pair; Audio Sync is always offered unless the
    subtitle does not belong to this video (too few cues matched, so there is nothing to align).
    """
    verdict = result["verdict"]
    if verdict in ("in_sync", "not_comparable"):
        return None
    options = []
    if result.get("fps"):
        options.append(("framerate", f"Change frame rate from {result['fps'][0]} to {result['fps'][1]} fps (keeps the subtitle as it is)"))
    options.append(("audio", "Audio Sync (re-times every cue to the speech; reuses the transcription just made)"))
    console.print("\n[bold]What now?[/bold]")
    for i, (_, label) in enumerate(options, 1):
        console.print(f"   [bold cyan]{i}.[/bold cyan] {label}")
    console.print("   [bold cyan]n.[/bold cyan] Nothing, leave the subtitle as it is")
    default = "n" if verdict == "loose" else "1"
    choice = Prompt.ask("[bold]Choose[/bold]", choices=[str(i) for i in range(1, len(options) + 1)] + ["n"], default=default, show_choices=False)
    return None if choice == "n" else options[int(choice) - 1][0]


def _auto_fix_choice(console, result, target, whisper_data, precise, video_fps=None, translated=False):
    """Unattended `--check --fix`: takes the check's own advice. Returns 'framerate', 'audio' or None (nothing to do / cannot help).

    A subtitle in sync is skipped. A suggested frame rate change (the check only suggests one when a single pair fits) is still tried on
    a copy and checked again against the same transcription (about a second), and kept only when it brings the subtitle in sync;
    otherwise Audio Sync is used.
    """
    verdict = result["verdict"]
    if verdict in ("in_sync", "loose"):
        console.print("[green]✅ Nothing to fix. Skipped.[/green]")
        return None
    if verdict == "not_comparable":
        console.print("[yellow]⚠️ Cannot be fixed: the subtitle does not seem to belong to this video. Skipped.[/yellow]")
        return None
    if result.get("fps"):
        from ..framerate.framerate import convert_frame_rate, resolve_rate
        from_label, to_label = result["fps"]
        _, from_fps = resolve_rate(float(from_label))
        _, to_fps = resolve_rate(float(to_label))
        trial = copy.deepcopy(target)
        convert_frame_rate(trial, from_fps, to_fps)
        after = check_subtitles(trial, whisper_data, precise=precise, video_fps=video_fps, translated=translated)
        if after["verdict"] in ("in_sync", "loose"):
            console.print(f"[green]🎥 Frame rate {from_label} to {to_label} fps brings it in sync "
                          f"(checked again: average {after['median']:+.1f} s). Applying it.[/green]")
            return "framerate"
        console.print(f"[yellow]⚠️ Frame rate {from_label} to {to_label} fps would leave it "
                      f"{syncverdict.LABELS[after['verdict']][1].lower()}. Using Audio Sync instead.[/yellow]")
    else:
        console.print("[dim]No frame rate explains it. Using Audio Sync.[/dim]")
    return "audio"


def run_audiosync(args, device, model_size, compute_type, batch_size, translation_model, console, cpu_threads=0):
    """
    Main workflow for the Audio-based Sync (Whisper).
    """
    queue = []

    # BUILD SYNC QUEUE

    # PATH A: UNATTENDED MODE (-s provided)
    if args.subtitle:
        sub_path = Path(args.subtitle).resolve()
        
        if not sub_path.exists():
            console.print(f"[bold red]❌ Error:[/bold red] Subtitle file not found: [yellow]{sub_path}[/yellow]")
            sys.exit(1)

        # Determine Video Path
        video_file = None
        
        # Explicit Video Argument (-v) overrides everything
        if args.video:
            explicit_vid = Path(args.video).resolve()
            if explicit_vid.exists():
                video_file = explicit_vid
            else:
                console.print(f"[bold red]❌ Error:[/bold red] Video file not found: [yellow]{explicit_vid}[/yellow]")
                sys.exit(1)
        
        # Auto-detect using helper
        else:
            video_file = find_best_video_match(sub_path)
            
            if not video_file:
                console.print(f"[bold red]❌ Error:[/bold red] Could not auto-detect video for [cyan]{sub_path.name}[/cyan]")
                console.print(f"[dim]Please provide the video path explicitly using -v / --video[/dim]")
                sys.exit(1)

        console.print(f"[green]🚀 Unattended Mode:[/green] {'Checking' if getattr(args, 'check', False) else 'Syncing'} [cyan]{sub_path.name}[/cyan]")
        queue.append((sub_path, video_file))


    # PATH B: INTERACTIVE MODE
    else:
        # Get Subtitle Files
        subs = get_files(SUPPORTED_EXTENSIONS)
        if not subs:
            console.print("[bold red]❌ No subtitle files found in this folder![/]")
            return # Return instead of exit to allow going back to menu if needed

        # Launch TUI Picker
        selected_subs = select_files_interactive(subs)

        if not selected_subs:
            console.print("[yellow]No files selected. Returning to menu.[/yellow]")
            return

        # Match Videos
        for sub_file in selected_subs:
            # Try Auto-Match
            video_file = find_best_video_match(sub_file)
            
            # Fallback to Manual Selection (TUI)
            if not video_file:
                video_file = select_video_fallback(sub_file.name)
            
            if video_file:
                queue.append((sub_file, video_file))
            else:
                console.print(f"[dim red]Skipping {sub_file.name} (No video selected)[/dim red]")
    
    if not queue:
        console.print("[bold red]❌ No valid pairs to sync![/]")
        return

    # PROCESS QUEUE
    file_label = "file" if len(queue) == 1 else "files"
    check = bool(getattr(args, "check", False))      # Sync Check: measure and report, write nothing
    # --check on the command line (with or without -s) only reports, or with --fix acts on its own: it never asks. The menu entry
    # for a single file still offers the fixes.
    policy = _fix_policy(args, check, len(queue))
    hands_off = policy in ("auto", "report")          # exit code 3 tells a script when something is not in sync
    check_results = []
    check_entries = []          # for the batch summary: name, video name, result
    failed_names = []           # files that could not be checked or synced
    performed_sync = False      # a check that went on into an audio sync
    action = ("Batch Sync" if len(queue) > 1 else "Sync") if not check else ("Batch Sync Check" if len(queue) > 1 else "Sync Check")
    console.print(f"\n[bold green]🚀 Starting {action} ({len(queue)} {file_label})...[/bold green]")
    
    total_start = time.time()

    # Speech recognition engine. An explicit --asr parakeet is consent to install it; 'auto' never installs anything.
    asr = getattr(args, "asr", "auto") or "auto"
    parakeet_ok = False
    if asr == "parakeet":
        parakeet_ok = parakeet.ensure_ready(console, device, install_if_missing=True)
    elif asr == "auto":
        parakeet_ok = parakeet.is_installed()

    current_model = None
    loaded_lang_code = "UNSET"
    
    failed_count = 0

    def free_whisper():
        """Release the Whisper model so NLLB has VRAM to load into."""
        nonlocal current_model, loaded_lang_code
        current_model = None
        loaded_lang_code = "UNSET"
        gc.collect()
        if device == "cuda": torch.cuda.empty_cache()

    for i, (sub, vid) in enumerate(queue, 1):
        console.print(f"\n[bold reverse] Task {i}/{len(queue)} [/bold reverse] [cyan]{sub.name}[/cyan]")
        console.print(f"🎬 Video: [yellow]{vid.name}[/yellow]")
        
        # Detection
        meta_lang = get_audio_language(vid) 
        if meta_lang:
            console.print(f"[dim]🌐 Metadata language detected: [bold cyan]{meta_lang.upper()}[/bold cyan][/dim]")
        else:
            console.print("[dim]🌐 Language metadata missing. Using Auto-detect.[/dim]")

        sub_lang = get_subtitle_language(sub)
        console.print(f"[dim]📄 Subtitle language detected: [bold cyan]{sub_lang.upper()}[/bold cyan][/dim]")    

        needs_translation = False
        if meta_lang and sub_lang != "unknown" and meta_lang != sub_lang:
            console.print(f"[dim]⚠️ Mismatch detected: Audio is {meta_lang.upper()}, Subtitle is {sub_lang.upper()}. Needs translation.[/dim]")
            needs_translation = True

        # Translated "ghost" subtitle is synced in memory; each original event is paired with its
        # ghost event up front so timings can be copied back by identity, not by position.
        original_sub_object = None
        ghost_sub = None
        event_pairs = []
        
        # Translation
        if needs_translation:
            # Load the ORIGINAL content into memory now
            original_sub_object = open_subtitle(sub)

            free_whisper()
            
            nllb_source = get_language_code_for_nllb(sub_lang)
            nllb_target = get_language_code_for_nllb(meta_lang)
            
            status_msg = f"[bold dim] Translating subtitles from [cyan]{sub_lang.upper()}[/cyan] to [cyan]{meta_lang.upper()}[/cyan] using NLLB...[/]"
            
            with Progress(
                SpinnerColumn("dots"),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),      
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console,
                transient=True        
            ) as progress:
                task = progress.add_task("Translating", total=None)
                
                ghost_sub = translate_subtitle_nllb(
                    original_sub_object,
                    nllb_source,
                    nllb_target,
                    device=device,
                    model_id=translation_model,
                    compute_type=compute_type,
                    progress=progress,
                    task_id=task,
                    cpu_threads=cpu_threads,
                    alternatives=ALT_TRANSLATIONS
                )

            if ghost_sub is None:
                failed_count += 1
                failed_names.append(sub.name)
                continue

            console.print(f"[dim]🔄 Translation complete ({sub_lang.upper()} -> {meta_lang.upper()}).[/dim]")
            event_pairs = list(zip(original_sub_object, ghost_sub))

        # Determine Target Model
        target_model = model_size
        if meta_lang and meta_lang.lower() == "en":
            if target_model in {"tiny", "base", "small", "medium"}:
                target_model = f"{target_model}.en"

        # Which engine hears this file? Parakeet covers 25 languages; everything else, and any failure, uses Whisper.
        use_parakeet = False
        if asr != "whisper" and parakeet_ok:
            if parakeet.supports_language(meta_lang):
                use_parakeet = True
            elif meta_lang is None:
                use_parakeet = (asr == "parakeet")      # unknown language: only when explicitly asked for
            elif asr == "parakeet":
                console.print(f"[yellow]⚠️ Parakeet does not support {meta_lang.upper()}. Using Whisper for this file.[/yellow]")

        def ensure_whisper_loaded():
            nonlocal current_model, loaded_lang_code
            console.print(f"[dim]🎯 Target Model: [bold white]{target_model}[/bold white][/dim]")
            if current_model is None or loaded_lang_code != meta_lang or needs_translation:
                if current_model is not None:
                    console.print(f"[dim]🌐 Language changed ({loaded_lang_code} -> {meta_lang}). Switching model...[/dim]")
                    current_model = None
                    gc.collect()
                    if device == "cuda": torch.cuda.empty_cache()

                current_model = load_whisper_model(device, compute_type, meta_lang, target_model)
                loaded_lang_code = meta_lang
            else:
                console.print(f"[dim]♻️  Reusing cached model ({loaded_lang_code if loaded_lang_code else 'Auto'})...[/dim]")

        start_time = time.time()
        try:
            # Step 1: Transcribe
            parakeet_used = False
            whisper_data = detected_lang = media_duration = None
            if use_parakeet:
                try:
                    console.print("[dim]🦜 Transcribing with Parakeet...[/dim]")
                    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(),
                                  TaskProgressColumn(), TimeElapsedColumn(), console=console, transient=True) as progress:
                        task = progress.add_task("[cyan]Parakeet: extracting audio...", total=None)

                        def on_progress(stage, done, total):
                            if stage == "model":
                                progress.update(task, description="[cyan]Parakeet: loading the model...")
                            elif stage == "chunks" and total:
                                progress.update(task, description="[cyan]Parakeet: transcribing...", total=total, completed=done)

                        whisper_data, media_duration = parakeet.transcribe(vid, device, console, on_progress)
                    detected_lang = meta_lang or parakeet.detect_language(whisper_data) or (sub_lang if sub_lang != "unknown" else "en")
                    parakeet_used = True
                    console.print(f"[dim]📝 Parakeet transcription complete ({sum(len(s['words']) for s in whisper_data)} words).[/dim]")
                except Exception as e:
                    console.print(f"[yellow]⚠️ Parakeet failed ({e}). Using Whisper.[/yellow]")
            if not parakeet_used:
                ensure_whisper_loaded()
                whisper_data, detected_lang = run_whisper_transcription(vid, device, compute_type, batch_size, current_model, meta_lang, model_name=target_model, cpu_threads=cpu_threads)

            if whisper_data is None:
                failed_count += 1
                failed_names.append(sub.name)
                continue

            # Step 2: Auto-detect mismatch check (only when metadata was missing)
            if not meta_lang:
                console.print(f"[dim]🌐 Auto-detected audio language: [bold cyan]{detected_lang.upper()}[/bold cyan][/dim]")
                if sub_lang != "unknown" and detected_lang != sub_lang:
                    console.print(f"[dim]⚠️ Mismatch detected: Audio is {detected_lang.upper()}, Subtitle is {sub_lang.upper()}. Needs translation.[/dim]")
                    needs_translation = True
                    original_sub_object = open_subtitle(sub)

                    free_whisper()

                    nllb_source = get_language_code_for_nllb(sub_lang)
                    nllb_target = get_language_code_for_nllb(detected_lang)

                    status_msg = f"[bold dim] Translating subtitles from [cyan]{sub_lang.upper()}[/cyan] to [cyan]{detected_lang.upper()}[/cyan] using NLLB...[/]"

                    with Progress(
                        SpinnerColumn("dots"),
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(),
                        TaskProgressColumn(),
                        TimeElapsedColumn(),
                        console=console,
                        transient=True
                    ) as progress:
                        task = progress.add_task("Translating", total=None)

                        ghost_sub = translate_subtitle_nllb(
                            original_sub_object,
                            nllb_source,
                            nllb_target,
                            device=device,
                            model_id=translation_model,
                            compute_type=compute_type,
                            progress=progress,
                            task_id=task,
                            cpu_threads=cpu_threads,
                            alternatives=ALT_TRANSLATIONS
                        )

                    if ghost_sub is None:
                        raise RuntimeError("Translation failed")

                    console.print(f"[dim]🔄 Translation complete ({sub_lang.upper()} -> {detected_lang.upper()}).[/dim]")
                    event_pairs = list(zip(original_sub_object, ghost_sub))

            # Sync Check: compare and report. Unattended it stops there (nothing is written). Interactively the user may then
            # apply the suggested frame rate change, or carry on into the audio sync with the transcription already made.
            do_sync = not check
            if check:
                video_fps = get_video_fps(vid)
                target = ghost_sub if (needs_translation and original_sub_object) else open_subtitle(sub)
                result = check_subtitles(copy.deepcopy(target), whisper_data, precise=parakeet_used, video_fps=video_fps, translated=needs_translation)   # the aligner retimes what it is given
                check_results.append(result)
                entry = {"name": sub.name, "video": vid.name, "result": result, "action": ""}
                check_entries.append(entry)
                syncverdict.print_report(console, sub.name, result)
                console.print(f"[dim]Checked in {time.time() - start_time:.1f}s.[/dim]")
                if policy == "auto":
                    choice = _auto_fix_choice(console, result, target, whisper_data, parakeet_used, video_fps, translated=needs_translation)
                elif policy == "ask":
                    choice = _offer_fix(console, result)
                else:
                    choice = None            # report only / a batch is not interrupted: the summary at the end shows what needs fixing
                if choice == "framerate":
                    entry["action"] = f"frame rate {result['fps'][0]} to {result['fps'][1]} fps"
                    from ..framerate.framerate import _retime_file, resolve_rate
                    from_label, from_fps = resolve_rate(float(result["fps"][0]))
                    to_label, to_fps = resolve_rate(float(result["fps"][1]))
                    if not _retime_file(sub, from_fps, to_fps, from_label, to_label, args, console):
                        failed_count += 1
                    continue
                if choice == "audio":
                    entry["action"] = "Audio Sync"
                    do_sync = performed_sync = True
                    start_time = time.time()
                else:
                    console.print("[dim]No file was written.[/dim]")
                    continue

            # Step 2b: Parakeet drops some short shouted lines. Re-transcribe (faster-whisper, loaded once) only the
            # windows where the subtitle has cues Parakeet did not hear.
            # (Skipped while only checking: a few more anchors do not change a verdict, and the repair is most of the run time.)
            if parakeet_used and do_sync:
                try:
                    target_subs = ghost_sub if (needs_translation and original_sub_object) else open_subtitle(sub)
                    whisper_data, zones, recovered = parakeet.repair_missing_speech(
                        whisper_data, target_subs, vid, media_duration, meta_lang or detected_lang,
                        target_model, device, compute_type, cpu_threads, console)
                    if zones:
                        console.print(f"[dim]🔧 Gap repair: {zones} zone(s) around cues Parakeet did not hear, {recovered} words recovered.[/dim]")
                except Exception as e:
                    console.print(f"[yellow]⚠️ Gap repair skipped ({e}).[/yellow]")

            # Step 3: Align & Sync
            if needs_translation and original_sub_object:
                _, rejected, anchors = align_subtitles(ghost_sub, whisper_data, precise=parakeet_used, translated=True,
                                                       match_table=bool(getattr(args, 'match_table', False)))
                lines = len(original_sub_object)

                console.print("[dim]📥 Applying synced timestamps back to original subtitle...[/dim]")
                for orig_event, ghost_event in event_pairs:
                    orig_event.start = ghost_event.start
                    orig_event.end = ghost_event.end

                if args and getattr(args, "overwrite", False):
                    backup_if_needed(sub, args)
                    final_output_path = sub
                    console.print(f"[dim]💾 Overwriting original subtitle: {final_output_path.name}[/dim]")
                else:
                    final_output_path = unique_path(sub.with_suffix(".synced.srt"))

                original_sub_object.save(str(final_output_path))
                console.print(f"💾 Restored Original Content to: [underline]{final_output_path.name}[/underline]")
            else:
                final_output_path, lines, rejected, anchors = run_anchor_align_and_sync(sub, whisper_data, args, precise=parakeet_used)

            duration = time.time() - start_time

            console.print(f"[bold green]✨ Success![/bold green] ({duration:.1f}s)")
            console.print(f" 📝 Lines Processed: {lines}")
            console.print(f" ⚓ Anchors: {anchors} of {lines} lines")
            console.print(f" 🗑️ Outliers Rejected: {rejected}")

            if not needs_translation:
                console.print(f" 💾 Saved to: [underline]{final_output_path.name}[/underline]")

        except Exception as e:
            failed_count += 1
            failed_names.append(sub.name)
            console.print(f"[bold red]❌ Failed:[/bold red] {e}")

    # Cleanup at very end
    free_whisper()
    
    total_duration = time.time() - total_start
    
    # FINAL SUMMARY
    summary_color = "bold green" if failed_count == 0 else "bold yellow"
    summary_label = ("Batch Sync" if len(queue) > 1 else "Sync") + (" Check" if check and not performed_sync else "")
    summary_text = f"✨ {summary_label} Complete in {total_duration:.1f}s"
    
    if failed_count > 0:
        fail_label = "sync" if failed_count == 1 else "syncs"
        summary_text += f" with {failed_count} failed {fail_label}"
        summary_color = "bold red" 
        
    console.print(f"\n[{summary_color}]{summary_text}[/{summary_color}]")

    # After a batch check: what to fix, at a glance (a single check already printed its own report)
    if check and len(queue) > 1 and (check_entries or failed_names):
        syncverdict.print_summary(console, check_entries, failed_names)

    # A script can act on the result: exit code 3 when any checked subtitle is not in sync (unattended mode only).
    # With --fix the fixable ones were fixed, so only a subtitle that cannot be compared is left over.
    if hands_off and failed_count == 0:
        left = [r for r in check_results if r["verdict"] == "not_comparable"] if getattr(args, "fix", False) else \
               [r for r in check_results if r["verdict"] not in ("in_sync", "loose")]
        if left:
            sys.exit(3)

