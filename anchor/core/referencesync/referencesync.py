import sys
import time
import pysubs2
from pathlib import Path
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, BarColumn, TaskProgressColumn

from ...utils.files import get_files, open_subtitle, backup_if_needed
from ...utils.pairing import pick_pairs
from ...utils.mappings import get_language_code_for_nllb
from ...utils.languages import get_subtitle_language
from ..translation import translate_subtitle_nllb
from ...utils.whisper import GlobalAligner

SUPPORTED_EXTENSIONS = {".srt", ".ass", ".vtt", ".sub"}

def run_referencesync(args, device, translation_model, compute_type, console, cpu_threads=0):
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
        sub_files = get_files(SUPPORTED_EXTENSIONS)
        if not sub_files:
            console.print("[bold red]❌ No subtitle files found in the current directory.[/bold red]")
            return

        if len(sub_files) < 2:
            console.print("[bold red]❌ Reference sync needs at least two subtitles in the folder (a target and a reference).[/bold red]")
            return

        console.print("\n[bold cyan]🎯 Pick the TARGET subtitles (left) and their REFERENCES (right)[/bold cyan]")
        console.print("[dim]Each pick is numbered in the order you make it: pair 1 is the first target with the first reference, and so on. "
                      "[A] selects a whole side in listing order, which pairs a season in two key presses.[/dim]")
        pairs = pick_pairs(sub_files, sub_files)
        if not pairs:
            console.print("[yellow]No pairs selected. Returning to menu.[/yellow]")
            return
        queue.extend(pairs)

        table = Table(box=None, show_header=True, header_style="bold", padding=(0, 1))
        table.add_column("#", justify="right", style="dim")
        table.add_column("Target", style="cyan", overflow="fold")
        table.add_column("Reference", style="green", overflow="fold")
        for n, (target_path, ref_path) in enumerate(queue, 1):
            table.add_row(str(n), target_path.name, ref_path.name)
        console.print(table)
        console.print(f"\n[bold green]✅ Queue built with {len(queue)} pair{'s' if len(queue) != 1 else ''}. Starting sync...[/bold green]\n")

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

        original_sub_object = None
        ghost_sub = None
        event_pairs = []
        
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
                    compute_type=compute_type,
                    progress=progress,
                    task_id=task,
                    cpu_threads=cpu_threads
                )
            
            if ghost_sub is None:
                failed_count += 1
                continue

            console.print(f"[dim]🔄 Translation complete ({target_lang.upper()} -> {ref_lang.upper()}).[/dim]")
            event_pairs = list(zip(original_sub_object, ghost_sub))

        # ==========================================
        # 3. ALIGNMENT LOGIC
        # ==========================================
        start_time = time.time()
        try:
            # 1. Load the target text (the in-memory ghost when translated) and the reference
            target_subs_obj = ghost_sub if needs_translation else open_subtitle(target_sub)
            ref_subs_obj = open_subtitle(ref_sub)

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
            aligner = GlobalAligner(target_subs_obj, pseudo_whisper, reference=True)
            synced_subs_obj, rejected = aligner.run()

            if not synced_subs_obj:
                raise Exception("Alignment failed to find enough matching text.")

            lines = len(synced_subs_obj)

            if needs_translation:
                # Copy timings from the synced ghost events back onto the original events
                console.print("[dim]📥 Applying synced timestamps back to original subtitle...[/dim]")
                for orig_event, ghost_event in event_pairs:
                    orig_event.start = ghost_event.start
                    orig_event.end = ghost_event.end
                result_subs = original_sub_object
            else:
                result_subs = synced_subs_obj

            if args and getattr(args, "overwrite", False):
                backup_if_needed(target_sub, args)
                final_output_path = target_sub
                console.print(f"[dim]💾 Overwriting original subtitle: {final_output_path.name}[/dim]")
            else:
                final_output_path = target_sub.with_suffix(".synced.srt")

            result_subs.save(str(final_output_path))

            duration = time.time() - start_time

            console.print(f"[bold green]✨ Success![/bold green] ({duration:.1f}s)")
            console.print(f" 📝 Lines Processed: {lines}")
            console.print(f" ⚓ Anchors: {aligner.anchor_count} of {lines} lines")
            console.print(f" 🗑️ Outliers Rejected: {rejected}")
            console.print(f"💾 Saved to: [underline]{final_output_path.name}[/underline]")

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
