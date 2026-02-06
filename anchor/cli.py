import sys

from . import pytorch_compat
pytorch_compat.apply_patches()

import time
import argparse
import re
import gc
import torch
import pysubs2
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
from .hardware import get_compute_device
from .utils import open_subtitle, parse_range_selection, get_audio_language, check_dependencies, get_subtitle_language, get_language_code_for_nllb
from .core import run_anchor_sync, load_whisper_model
from .translation import translate_subtitle_nllb
from . import __version__

console = Console()

SUPPORTED_EXTENSIONS = {".srt", ".ass", ".vtt", ".sub"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".webm"}

def get_files(extensions):
    return sorted([f for f in Path.cwd().iterdir() if f.suffix.lower() in extensions], key=lambda f: f.name)

def select_video_fallback(sub_filename):
    """
    If auto-match fails, ask the user to pick a video file.
    """
    console.print(f"\n[bold red]⚠️  Could not auto-match video for:[/bold red] {sub_filename}")
    console.print("Please select the media file manually:")
    
    videos = get_files(VIDEO_EXTENSIONS)
    if not videos:
        console.print("[red]No video files found in directory![/red]")
        return None

    table = Table(show_header=False, box=None)
    for idx, f in enumerate(videos, 1):
        table.add_row(f"[bold cyan]{idx}.[/bold cyan]", f.name)
    console.print(table)
    
    while True:
        selection = Prompt.ask("Select Video", default="1")
        if selection.isdigit():
            idx = int(selection) - 1
            if 0 <= idx < len(videos):
                return videos[idx]
        console.print("[red]Invalid selection.[/red]")

def find_best_video_match(sub_path):
    """
    Smart matching: Handles language codes (Movie.en.srt -> Movie.mp4)
    """
    # 1. Get the 'stem' (filename without .srt)
    clean_name = sub_path.stem

    # 2. Iteratively strip trailing language codes or markers
    # Handles endings like: .en, .eng, .pt-BR, .synced, .sync, .hi (in any order at the end)
    token_re = re.compile(r'(?:\.(?:[a-z]{2,3}(?:-[a-z]{2})?|synced|sync|hi|ai))$', flags=re.IGNORECASE)
    while True:
        new_name = token_re.sub('', clean_name)
        if new_name == clean_name:
            break
        clean_name = new_name

    # 3. Look for exact match with video extensions
    for ext in VIDEO_EXTENSIONS:
        candidate = sub_path.with_name(clean_name + ext)
        if candidate.exists():
            return candidate

    # 4. Fuzzy Match: If video filename contains the clean subtitle name
    videos = get_files(VIDEO_EXTENSIONS)
    for vid in videos:
        if clean_name.lower() in vid.name.lower():
            return vid
            
    return None

def main():
    parser = argparse.ArgumentParser(description="Anchor Subtitle Sync")
    parser.add_argument(
        "-m", "--model", 
        type=str, 
        help="Force a specific model size (e.g., tiny, base, small, medium, large-v3)",
        default=None
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        help="Force a specific batch size (overrides automatic selection)",
        default=None
    )
    parser.add_argument(
        "-t", "--translation-model",
        type=str,
        help="Force a specific translation model (overrides automatic selection)",
        default=None
    )
    args = parser.parse_args()

    try:
        console.clear()
        console.print(f"[bold blue]🎬 Anchor Subtitle Sync {__version__}[/bold blue]\n")

        # 1. Hardware Check
        device, compute_type, batch_size, model_size, translation_model = get_compute_device(force_model=args.model, force_batch=args.batch_size, force_translation_model=args.translation_model)
        console.print(f"[dim]Engine configured for: [bold white]{device}[/bold white] (model: {model_size}, precision: {compute_type}, batch size: {batch_size}, translation model: {translation_model})[/dim]\n")

        # 2. Dependency Check
        if not check_dependencies():
            sys.exit(1)

        # 3. List Subtitles
        subs = get_files(SUPPORTED_EXTENSIONS)
        if not subs:
            console.print("[bold red]❌ No subtitle files found in this folder![/]")
            sys.exit(1)

        table = Table(title="Available Subtitles", show_header=True, header_style="bold magenta", title_justify="left")
        table.add_column("#", style="dim", width=4)
        table.add_column("Filename", style="cyan")

        for idx, f in enumerate(subs, 1):
            table.add_row(str(idx), f.name)
        console.print(table)

        # 4. Select Range
        console.print("\n[dim]Examples: [bold cyan]1-3[/bold cyan] (Range), [bold cyan]1,4,6[/bold cyan] (Specific), or [bold cyan]Enter[/bold cyan] for ALL.[/dim]")
        
        selection_str = Prompt.ask("Select files to sync", default="all", show_default=False)
        
        # Handle "All" (Default) vs Specific Selection
        if selection_str.lower() == "all" or selection_str.strip() == "":
            selected_indices = range(len(subs))
        else:
            selected_indices = parse_range_selection(selection_str, len(subs))

        if not selected_indices:
            console.print("[yellow]No valid files selected. Exiting.[/yellow]")
            sys.exit(0)

        queue = []
        
        # 5. Pair Subtitles with Videos
        for idx in selected_indices:
            sub_file = subs[idx]
            video_file = find_best_video_match(sub_file)
            
            if not video_file:
                video_file = select_video_fallback(sub_file.name)
                
            if video_file:
                queue.append((sub_file, video_file))
        
        if not queue:
            sys.exit(0)

        # 6. Process Queue
        file_label = "file" if len(queue) == 1 else "files"
        action = "Batch Sync" if len(queue) > 1 else "Sync"
        console.print(f"\n[bold green]🚀 Starting {action} ({len(queue)} {file_label})...[/bold green]")
        
        total_start = time.time()
        
        # --- SMART MODEL CACHING STATE ---
        current_model = None
        loaded_lang_code = "UNSET"
        
        # --- TRACK FAILURES ---
        failed_count = 0
        # ----------------------

        for i, (sub, vid) in enumerate(queue, 1):
            console.print(f"\n[bold reverse] Task {i}/{len(queue)} [/bold reverse] [cyan]{sub.name}[/cyan]")
            console.print(f"🎬 Video: [yellow]{vid.name}[/yellow]")
            
            # 1. Detection
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

            # --- PREPARE INPUT FOR SYNC ---
            # Default: Sync the original file path
            sub_input_for_sync = sub      
            
            # Variables for cleanup later
            original_sub_object = None    
            ghost_file_path = None        
            
            # 2. Translation
            if needs_translation:
                # Load the ORIGINAL content into memory now
                original_sub_object = open_subtitle(sub)
                
                nllb_source = get_language_code_for_nllb(sub_lang)
                nllb_target = get_language_code_for_nllb(meta_lang)
                
                status_msg = f"[bold dim]🔄 Translating subtitles from [cyan]{sub_lang.upper()}[/cyan] to [cyan]{meta_lang.upper()}[/cyan] using NLLB...[/]"
                
                with console.status(status_msg, spinner="dots"):
                    ghost_sub = translate_subtitle_nllb(
                        original_sub_object, 
                        nllb_source, 
                        nllb_target, 
                        device=device, 
                        model_id=translation_model
                    )
                
                console.print(f"[dim]✅ Translation complete ({sub_lang.upper()} -> {meta_lang.upper()}).[/dim]")

                # Save Ghost to a TEMP FILE so we can pass a PATH to the sync engine
                ghost_file_path = sub.with_suffix(f".tmp.{meta_lang}.srt")
                ghost_sub.save(str(ghost_file_path))
                
                # Point the sync engine to the translated temp file
                sub_input_for_sync = ghost_file_path
                console.print(f"[dim]👻 Created temporary sync target: {ghost_file_path.name}[/dim]")

            # 3. Determine Target Model
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

            # 4. Run Sync
            start_time = time.time()
            try:
                # Pass the PATH (Original or Ghost Temp Path)
                out_path, lines, rejected = run_anchor_sync(vid, sub_input_for_sync, device, compute_type, batch_size, current_model, meta_lang)
                
                # 5. Restoration Logic
                final_output_path = out_path # Default
                
                if needs_translation and original_sub_object:
                    console.print("[dim]📥 Applying synced timestamps back to original subtitle...[/dim]")
                    
                    # Load the file Whisper just synced (translated)
                    synced_ghost = pysubs2.load(str(out_path))
                    
                    # Transfer timestamps to Original Object
                    for orig_event, ghost_event in zip(original_sub_object, synced_ghost):
                        orig_event.start = ghost_event.start
                        orig_event.end = ghost_event.end
                    
                    # Determine final name
                    final_output_path = sub.with_suffix(".synced.srt")
                    original_sub_object.save(str(final_output_path))
                    console.print(f"💾 Restored Original Content to: [underline]{final_output_path.name}[/underline]")
                    
                    # Cleanup Temp Files (The translated files)
                    try:
                        if ghost_file_path and ghost_file_path.exists():
                            ghost_file_path.unlink()
                        if out_path.exists() and out_path != final_output_path:
                            out_path.unlink() 
                    except Exception:
                        pass 

                duration = time.time() - start_time
                
                console.print(f"[bold green]✨ Success![/bold green] ({duration:.1f}s)")
                console.print(f"  📝 Lines Processed: {lines}")
                console.print(f"  🗑️ Outliers Rejected: {rejected}")
                
                if not needs_translation:
                    console.print(f"  💾 Saved to: [underline]{final_output_path.name}[/underline]")
                
            except Exception as e:
                failed_count += 1
                console.print(f"[bold red]❌ Failed:[/bold red] {e}")

        # Cleanup at very end
        if current_model:
            del current_model
        
        total_duration = time.time() - total_start
        
        # --- FINAL SUMMARY LOGIC ---
        summary_color = "bold green" if failed_count == 0 else "bold yellow"
        summary_label = "Batch Sync" if len(queue) > 1 else "Sync"
        summary_text = f"✨ {summary_label} Complete in {total_duration:.1f}s"
        
        if failed_count > 0:
            fail_label = "sync" if failed_count == 1 else "syncs"
            summary_text += f" with {failed_count} failed {fail_label}"
            summary_color = "bold red" # Turn red if there were errors
            
        console.print(f"\n[{summary_color}]{summary_text}[/{summary_color}]")

    except KeyboardInterrupt:
        console.print("\n[bold red]✖  Aborted by user.[/bold red]")
        sys.exit(130)
        
    except Exception as e:
        console.print(f"\n[bold red]💥 An unexpected error occurred:[/bold red] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()