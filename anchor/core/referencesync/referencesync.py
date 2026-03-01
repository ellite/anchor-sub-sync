import sys
import time
import pysubs2
from pathlib import Path
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, BarColumn, TaskProgressColumn

from ...utils.files import get_files, open_subtitle, select_files_interactive
from ...utils.mappings import get_language_code_for_nllb
from ...utils.languages import get_subtitle_language
from ..translation import translate_subtitle_nllb
from ...utils.whisper import GlobalAligner

SUPPORTED_EXTENSIONS = {".srt", ".ass", ".vtt", ".sub"}

def run_referencesync(args, device, translation_model, console):
    """
    Main workflow for Reference-based Sync.
    Maps an out-of-sync target subtitle to a perfectly synced reference subtitle.
    """
    queue = []

    # ==========================================
    # 1. BUILD SYNC QUEUE
    # ==========================================

    # PATH A: UNATTENDED MODE (-s and -r provided)
    if args.subtitle and args.reference:
        target_path = Path(args.subtitle).resolve()
        ref_path = Path(args.reference).resolve()
        
        if not target_path.exists():
            console.print(f"[bold red]❌ Error:[/bold red] Target subtitle not found: [yellow]{target_path}[/yellow]")
            sys.exit(1)
        if not ref_path.exists():
            console.print(f"[bold red]❌ Error:[/bold red] Reference subtitle not found: [yellow]{ref_path}[/yellow]")
            sys.exit(1)

        console.print(f"[green]🚀 Unattended Mode:[/green] Syncing [cyan]{target_path.name}[/cyan] to [green]{ref_path.name}[/green]")
        queue.append((target_path, ref_path))

    # PATH B: INTERACTIVE MODE
    else:
        subs = get_files(SUPPORTED_EXTENSIONS)
        if len(subs) < 2:
            console.print("[bold red]❌ You need at least TWO subtitle files in this folder for a reference sync![/]")
            return

        # Prompt 1: The Target
        console.print("\n[bold cyan]🎯 Step 1: Select TARGET Subtitle(s) to be fixed[/bold cyan]")
        selected_targets = select_files_interactive(subs, header_lines=["Select the subtitle file you want to sync (the 'target')."], multi_select=False)
        if not selected_targets:
            console.print("[yellow]No target files selected. Returning to menu.[/yellow]")
            return

        # Prompt 2: The Reference
        # Filter out the selected target so it doesn't accidentally get picked as the reference
        remaining_subs = [s for s in subs if s not in selected_targets]
        if not remaining_subs:
            console.print("[bold red]❌ No other subtitles left to act as a reference![/]")
            return

        console.print("\n[bold green]📑 Step 2: Select the perfectly timed REFERENCE Subtitle[/bold green]")
        selected_refs = select_files_interactive(remaining_subs, header_lines=["Select the perfectly synced subtitle file to use as reference for syncing."], multi_select=False)
        if not selected_refs:
            console.print("[yellow]No reference file selected. Returning to menu.[/yellow]")
            return
        
        reference_sub = selected_refs[0] 

        for target in selected_targets:
            queue.append((target, reference_sub))

    if not queue:
        return

    # ==========================================
    # 2. PROCESS QUEUE
    # ==========================================
    file_label = "file" if len(queue) == 1 else "files"
    action = "Batch Reference Sync" if len(queue) > 1 else "Reference Sync"
    console.print(f"\n[bold green]🚀 Starting {action} ({len(queue)} {file_label})...[/bold green]")
    
    total_start = time.time()
    failed_count = 0

    for i, (target_sub, ref_sub) in enumerate(queue, 1):
        console.print(f"\n[bold reverse] Task {i}/{len(queue)} [/bold reverse] [cyan]{target_sub.name}[/cyan]")
        console.print(f"📑 Reference: [green]{ref_sub.name}[/green]")
        
        # Detection
        target_lang = get_subtitle_language(target_sub)
        ref_lang = get_subtitle_language(ref_sub)
        
        console.print(f"[dim]🎯 Target language: [bold cyan]{target_lang.upper()}[/bold cyan][/dim]")    
        console.print(f"[dim]📑 Reference language: [bold green]{ref_lang.upper()}[/bold green][/dim]")    

        needs_translation = False
        if target_lang != "unknown" and ref_lang != "unknown" and target_lang != ref_lang:
            console.print(f"[dim]⚠️ Mismatch detected. Translating Target ({target_lang.upper()}) to match Reference ({ref_lang.upper()}).[/dim]")
            needs_translation = True

        sub_input_for_sync = target_sub      
        original_sub_object = None    
        ghost_file_path = None        
        
        # Translation (if needed)
        if needs_translation:
            original_sub_object = open_subtitle(target_sub)
            
            nllb_source = get_language_code_for_nllb(target_lang)
            nllb_target = get_language_code_for_nllb(ref_lang)
            
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
            
            console.print(f"[dim]🔄 Translation complete ({target_lang.upper()} -> {ref_lang.upper()}).[/dim]")

            # Save Ghost to a TEMP FILE
            ghost_file_path = target_sub.with_suffix(f".tmp.{ref_lang}.srt")
            ghost_sub.save(str(ghost_file_path))
            
            sub_input_for_sync = ghost_file_path
            console.print(f"[dim]👻 Created temporary sync target: {ghost_file_path.name}[/dim]")

        # ==========================================
        # 3. ALIGNMENT LOGIC
        # ==========================================
        start_time = time.time()
        try:
            # 1. Load the target text and the perfect reference text
            target_subs_obj = pysubs2.load(str(sub_input_for_sync))
            ref_subs_obj = pysubs2.load(str(ref_sub))

            # 2. Trick the engine: Convert the reference subtitle into pseudo-Whisper data
            pseudo_whisper = []
            for event in ref_subs_obj:
                pseudo_whisper.append({
                    "text": event.text,
                    "start": event.start / 1000.0,
                    "end": event.end / 1000.0
                })

            # 3. Run Global Aligner to map the target text to the reference timings
            console.print("[dim]🧠 Routing text to Global Aligner...[/dim]")
            aligner = GlobalAligner(target_subs_obj, pseudo_whisper)
            synced_subs_obj, rejected = aligner.run()

            if not synced_subs_obj:
                raise Exception("Alignment failed to find enough matching text.")

            lines = len(synced_subs_obj)

            # 4. Save the locally synced output to a temporary path
            out_path = sub_input_for_sync.with_suffix(".aligned.tmp.srt")
            synced_subs_obj.save(str(out_path))
            
            # Restoration Logic (Applies translated timestamps back to original)
            final_output_path = out_path 
            
            if needs_translation and original_sub_object:
                console.print("[dim]📥 Applying synced timestamps back to original subtitle...[/dim]")
                synced_ghost = pysubs2.load(str(out_path))
                
                for orig_event, ghost_event in zip(original_sub_object, synced_ghost):
                    orig_event.start = ghost_event.start
                    orig_event.end = ghost_event.end
                
                if args and getattr(args, "overwrite", False):
                    final_output_path = target_sub
                    console.print(f"[dim]💾 Overwriting original subtitle: {final_output_path.name}[/dim]")
                else:
                    final_output_path = target_sub.with_suffix(".synced.srt")

                original_sub_object.save(str(final_output_path))
                console.print(f"💾 Restored Original Content to: [underline]{final_output_path.name}[/underline]")
                
                # Cleanup Temp Files
                try:
                    if ghost_file_path and ghost_file_path.exists():
                        ghost_file_path.unlink()
                    if out_path.exists() and out_path != final_output_path and out_path != target_sub:
                        out_path.unlink() 
                except Exception:
                    pass 

            duration = time.time() - start_time
            
            console.print(f"[bold green]✨ Success![/bold green] ({duration:.1f}s)")
            console.print(f" 📝 Lines Processed: {lines}")
            console.print(f" 🗑️ Outliers Rejected: {rejected}")
            
            if not needs_translation:
                # If it wasn't translated, handle the final save for the native language file
                if args and getattr(args, "overwrite", False):
                    final_output_path = target_sub
                    console.print(f"[dim]💾 Overwriting original subtitle: {final_output_path.name}[/dim]")
                else:
                    final_output_path = target_sub.with_suffix(".synced.srt")
                
                synced_subs_obj.save(str(final_output_path))
                console.print(f"💾 Saved aligned subtitle to: [underline]{final_output_path.name}[/underline]")
                
                # Cleanup the temp alignment file
                if out_path.exists() and out_path != final_output_path:
                    out_path.unlink()

        except Exception as e:
            failed_count += 1
            console.print(f"[bold red]❌ Failed:[/bold red] {e}")

    # FINAL SUMMARY
    total_duration = time.time() - total_start
    summary_color = "bold green" if failed_count == 0 else "bold yellow"
    summary_text = f"✨ {action} Complete in {total_duration:.1f}s"
    
    if failed_count > 0:
        summary_text += f" with {failed_count} failed syncs"
        summary_color = "bold red" 
        
    console.print(f"\n[{summary_color}]{summary_text}[/{summary_color}]")