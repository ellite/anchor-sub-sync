import sys
import time
import gc
import torch
import pysubs2
from pathlib import Path
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, BarColumn, TaskProgressColumn
from ...utils.files import get_files, find_best_video_match, select_video_fallback, open_subtitle, select_files_interactive, backup_if_needed
from ...utils.mappings import get_language_code_for_nllb
from ...utils.languages import get_audio_language, get_subtitle_language
from ...utils.whisper import run_whisper_transcription, run_anchor_align_and_sync, load_whisper_model
from ..translation import translate_subtitle_nllb

# Constants
SUPPORTED_EXTENSIONS = {".srt", ".ass", ".vtt", ".sub"}

def run_audiosync(args, device, model_size, compute_type, batch_size, translation_model, console):
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

        console.print(f"[green]🚀 Unattended Mode:[/green] Syncing [cyan]{sub_path.name}[/cyan]")
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
    action = "Batch Sync" if len(queue) > 1 else "Sync"
    console.print(f"\n[bold green]🚀 Starting {action} ({len(queue)} {file_label})...[/bold green]")
    
    total_start = time.time()
    
    current_model = None
    loaded_lang_code = "UNSET"
    
    failed_count = 0

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

        # Default: Sync the original file path
        sub_input_for_sync = sub      
        
        # Variables for cleanup later
        original_sub_object = None    
        ghost_file_path = None        
        
        # Translation
        if needs_translation:
            # Load the ORIGINAL content into memory now
            original_sub_object = open_subtitle(sub)
            
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
                    progress=progress,
                    task_id=task
                )
            
            console.print(f"[dim]🔄 Translation complete ({sub_lang.upper()} -> {meta_lang.upper()}).[/dim]")

            # Save Ghost to a TEMP FILE
            ghost_file_path = sub.with_suffix(f".tmp.{meta_lang}.srt")
            ghost_sub.save(str(ghost_file_path))
            
            # Point the sync engine to the translated temp file
            sub_input_for_sync = ghost_file_path
            console.print(f"[dim]👻 Created temporary sync target: {ghost_file_path.name}[/dim]")

        # Determine Target Model
        target_model = model_size
        if meta_lang and meta_lang.lower() == "en":
            if target_model in {"tiny", "base", "small", "medium"}:
                target_model = f"{target_model}.en"

        console.print(f"[dim]🎯 Target Model: [bold white]{target_model}[/bold white][/dim]")

        # Load/Reload Whisper Model
        if current_model is None or loaded_lang_code != meta_lang or needs_translation:
            if current_model is not None:
                console.print(f"[dim]🌐 Language changed ({loaded_lang_code} -> {meta_lang}). Switching model...[/dim]")
                del current_model
                gc.collect()
                if device == "cuda": torch.cuda.empty_cache()

            current_model = load_whisper_model(device, compute_type, meta_lang, target_model)
            loaded_lang_code = meta_lang
        else:
            console.print(f"[dim]♻️  Reusing cached model ({loaded_lang_code if loaded_lang_code else 'Auto'})...[/dim]")

        start_time = time.time()
        try:
            # Step 1: Transcribe
            whisper_data, detected_lang = run_whisper_transcription(vid, device, compute_type, batch_size, current_model, meta_lang)

            if whisper_data is None:
                failed_count += 1
                continue

            # Step 2: Auto-detect mismatch check (only when metadata was missing)
            if not meta_lang:
                console.print(f"[dim]🌐 Auto-detected audio language: [bold cyan]{detected_lang.upper()}[/bold cyan][/dim]")
                if sub_lang != "unknown" and detected_lang != sub_lang:
                    console.print(f"[dim]⚠️ Mismatch detected: Audio is {detected_lang.upper()}, Subtitle is {sub_lang.upper()}. Needs translation.[/dim]")
                    needs_translation = True
                    original_sub_object = open_subtitle(sub)

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
                            progress=progress,
                            task_id=task
                        )

                    console.print(f"[dim]🔄 Translation complete ({sub_lang.upper()} -> {detected_lang.upper()}).[/dim]")

                    ghost_file_path = sub.with_suffix(f".tmp.{detected_lang}.srt")
                    ghost_sub.save(str(ghost_file_path))
                    sub_input_for_sync = ghost_file_path
                    console.print(f"[dim]👻 Created temporary sync target: {ghost_file_path.name}[/dim]")

            # Step 3: Align & Sync
            out_path, lines, rejected = run_anchor_align_and_sync(sub_input_for_sync, whisper_data, args)

            # Restoration Logic
            final_output_path = out_path

            if needs_translation and original_sub_object:
                console.print("[dim]📥 Applying synced timestamps back to original subtitle...[/dim]")

                synced_ghost = pysubs2.load(str(out_path))

                for orig_event, ghost_event in zip(original_sub_object, synced_ghost):
                    orig_event.start = ghost_event.start
                    orig_event.end = ghost_event.end

                if args and getattr(args, "overwrite", False):
                    backup_if_needed(sub, args)
                    final_output_path = sub
                    console.print(f"[dim]💾 Overwriting original subtitle: {final_output_path.name}[/dim]")
                else:
                    final_output_path = sub.with_suffix(".synced.srt")

                original_sub_object.save(str(final_output_path))
                console.print(f"💾 Restored Original Content to: [underline]{final_output_path.name}[/underline]")

                try:
                    if ghost_file_path and ghost_file_path.exists():
                        ghost_file_path.unlink()
                    if out_path.exists() and out_path != final_output_path:
                        out_path.unlink()
                except Exception:
                    pass

            duration = time.time() - start_time

            console.print(f"[bold green]✨ Success![/bold green] ({duration:.1f}s)")
            console.print(f" 📝 Lines Processed: {lines}")
            console.print(f" 🗑️ Outliers Rejected: {rejected}")

            if not needs_translation:
                console.print(f" 💾 Saved to: [underline]{final_output_path.name}[/underline]")

        except Exception as e:
            failed_count += 1
            console.print(f"[bold red]❌ Failed:[/bold red] {e}")

    # Cleanup at very end
    if current_model:
        del current_model
    
    total_duration = time.time() - total_start
    
    # FINAL SUMMARY
    summary_color = "bold green" if failed_count == 0 else "bold yellow"
    summary_label = "Batch Sync" if len(queue) > 1 else "Sync"
    summary_text = f"✨ {summary_label} Complete in {total_duration:.1f}s"
    
    if failed_count > 0:
        fail_label = "sync" if failed_count == 1 else "syncs"
        summary_text += f" with {failed_count} failed {fail_label}"
        summary_color = "bold red" 
        
    console.print(f"\n[{summary_color}]{summary_text}[/{summary_color}]")

