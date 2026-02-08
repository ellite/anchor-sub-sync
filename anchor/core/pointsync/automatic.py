import difflib
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from .common import get_filtered_lines, apply_linear_correction
from ...utils.languages import get_subtitle_language
from ...utils.mappings import get_language_code_for_nllb
from ..translation import translate_subtitle_nllb
from ...utils.files import open_subtitle

console = Console()

def run_auto_linear_sync(target_file, reference_file, device="cpu", model_id="JustFrederik/nllb-200-distilled-600M-ct2-int8", console=console):
    """
    Automatically finds start/end matches and applies linear correction.
    """
    
    # Load Files
    try:
        console.print("[dim]⏳ Loading subtitles...[/dim]")
        sub_target = open_subtitle(target_file)
        sub_ref = open_subtitle(reference_file)
    except Exception as e:
        console.print(f"[bold red]❌ Error loading files:[/bold red] {e}")
        return

    # Language Check & Translation
    lang_target = get_subtitle_language(target_file)
    lang_ref = get_subtitle_language(reference_file)
    
    # Work on a copy to not modify the actual target object yet
    search_sub = sub_target 
    
    if lang_target != lang_ref and lang_target != "unknown":
        console.print(f"[yellow]⚠️ Language mismatch: Target ({lang_target.upper()}) vs Ref ({lang_ref.upper()})[/yellow]")
        
        # reuse your translation logic
        nllb_src = get_language_code_for_nllb(lang_target)
        nllb_tgt = get_language_code_for_nllb(lang_ref)
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), TimeElapsedColumn(), transient=True) as progress:
            progress.add_task("Translating...", total=None)
            search_sub = translate_subtitle_nllb(sub_target, nllb_src, nllb_tgt, device=device, model_id=model_id)

        console.print(f"[dim]🔄 Translation complete ({lang_target.upper()} -> {lang_ref.upper()})[/dim]")

    # Get Candidates
    t_start = get_filtered_lines(search_sub, "start", limit=100)
    r_start = get_filtered_lines(sub_ref, "start", limit=100)
    
    t_end = get_filtered_lines(search_sub, "end", limit=100)
    r_end = get_filtered_lines(sub_ref, "end", limit=100)

    # Find Best Matches
    console.print("[dim]🔍 Searching for synchronization points...[/dim]")
    
    # Start: Standard forward search
    start_match = find_best_match(t_start, r_start, "Start", reverse=False)
    
    # End: REVERSE search (Find best match closest to the end)
    end_match = find_best_match(t_end, r_end, "End", reverse=True)

    if not start_match or not end_match:
        console.print("[bold red]❌ Could not find reliable sync points automatically.[/bold red]")
        console.print("[dim]Try using Manual Sync mode instead.[/dim]")
        return

    p1_t, p1_r, text1 = start_match
    p2_t, p2_r, text2 = end_match

    # End must be after Start
    # If the file is very short, text1 might equal text2. We need to prevent that.
    if p2_t <= p1_t:
        console.print("[bold red]❌ Logic Error: End point found before (or at) Start point.[/bold red]")
        console.print("[dim]The file might be too short for auto-sync, or the algorithm picked the same line twice.[/dim]")
        return

    # Apply Correction
    console.print(f"\n[bold green]✅ Points Locked![/bold green]")
    console.print(f"   🔹 Start Match: [cyan]'{text1}'[/cyan]")
    console.print(f"   🔹 End Match:   [cyan]'{text2}'[/cyan]")

    try:
        # Apply correction to the ORIGINAL sub_target
        m, c = apply_linear_correction(sub_target, p1_t, p1_r, p2_t, p2_r)
        
        console.print(f"\n[bold green]⚡ Applying Linear Correction...[/bold green]")
        console.print(f"   📐 Speed Factor: [cyan]{m:.6f}[/cyan]")
        console.print(f"   ⏱️  Offset:       [cyan]{c:.2f} ms[/cyan]")

        # 6. Save
        output_path = target_file.with_suffix(".synced.srt")
        sub_target.save(str(output_path))
        console.print(f"💾 Saved to: [underline]{output_path.name}[/underline]")

    except Exception as e:
        console.print(f"[bold red]❌ Error applying sync:[/bold red] {e}")


def find_best_match(target_lines, ref_lines, label, reverse=False):
    """
    Finds the pair of lines with the highest fuzzy string similarity.
    Args:
        reverse (bool): If True, iterates target_lines from the end (Bottom-Up).
    Returns: (target_timestamp, ref_timestamp, matching_text) or None
    """
    best_score = 0.0
    best_pair = None
    
    # Threshold: Matches below 0.85 (85%) are probably wrong
    MIN_SCORE = 0.85 

    # Prepare Iterator: Normal or Reversed
    # Iterate Target lines. Ref lines are always searched completely for a match.
    iterator = reversed(target_lines) if reverse else target_lines

    for t_idx, t_time, t_text in iterator:
        for r_idx, r_time, r_text in ref_lines:
            
            # Optimization: Skip if lengths differ drastically
            if abs(len(t_text) - len(r_text)) > 10: continue

            ratio = difflib.SequenceMatcher(None, t_text.lower(), r_text.lower()).ratio()
            
            # Strict Greater Than (>) ensures that if we find an equal score later/earlier, 
            # Stick with the first one found in THIS direction.
            if ratio > best_score:
                best_score = ratio
                best_pair = (t_time, r_time, r_text) # Use ref text as the "clean" label

    if best_score >= MIN_SCORE:
        return best_pair
    
    return None