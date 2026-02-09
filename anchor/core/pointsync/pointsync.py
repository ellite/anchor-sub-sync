from rich.console import Console
from ...utils.files import get_files, select_files_interactive
from ..pointsync.manual import run_manual_sync_tui
from ..pointsync.automatic import run_auto_linear_sync
from pathlib import Path

SUPPORTED_EXTENSIONS = {".srt", ".ass", ".vtt", ".sub"}

console = Console()

def run_pointsync(args, mode, device="cpu", translation_model="JustFrederik/nllb-200-distilled-600M-ct2-int8", console=Console()):
    """
    Main workflow for Point Sync (Manual or Auto-Linear).
    Args:
        mode (str): "auto" or "manual"
    """
    
    # IF -s / --subtitle and -r / --reference are provided, we can skip straight to execution
    if args.subtitle and args.reference:
        target_file = Path(args.subtitle)
        reference_file = Path(args.reference)
        
        console.print(f"[bold green]🚀 Starting Point Sync ({mode.upper()})[/bold green]")
        console.print(f"   📝 Target (To Fix):  [cyan]{target_file}[/cyan]")
        console.print(f"   ⏱️ Reference:       [green]{reference_file}[/green]\n")


        run_auto_linear_sync(target_file, reference_file, device, translation_model, console, args)
        
        return

    # Get all available subtitles
    subs = get_files(SUPPORTED_EXTENSIONS)
    if not subs:
        console.print("[bold red]❌ No subtitle files found in this folder![/bold red]")
        return

    # Select TARGET (The file to fix)
    # Create a custom header for the picker to guide the user
    target_header = [
        "🔹 PHASE 1: SELECT TARGET",
        "Select the subtitle file you want to FIX (Sync/Adjust).",
        ""
    ]
    
    target_selection = select_files_interactive(subs, header_lines=target_header, multi_select=False)
    
    if not target_selection:
        console.print("[yellow]No target selected. returning to menu.[/yellow]")
        return
    
    # Point Sync typically works on 1 pair. If user picked multiple, we just take the first one.
    target_file = target_selection[0]
    
    # Select REFERENCE (The good file)
    # Filter out the target file so it won't sync it against itself
    ref_candidates = [f for f in subs if f != target_file]
    
    if not ref_candidates:
        console.print("[bold red]❌ No other subtitles found to use as reference![/bold red]")
        return

    ref_header = [
        "🔹 PHASE 2: SELECT REFERENCE",
        f"Target: {target_file.name}",
        "Select the subtitle with PERFECT TIMING to use as a reference.",
        ""
    ]

    ref_selection = select_files_interactive(ref_candidates, header_lines=ref_header, multi_select=False)
    
    if not ref_selection:
        console.print("[yellow]No reference selected. returning to menu.[/yellow]")
        return

    reference_file = ref_selection[0]

    # EXECUTE
    console.print(f"\n[bold green]🚀 Starting Point Sync ({mode.upper()})[/bold green]")
    console.print(f"   📝 Target (To Fix):  [cyan]{target_file.name}[/cyan]")
    console.print(f"   ⏱️  Reference:       [green]{reference_file.name}[/green]")

    if mode == "manual":
        # Launch the Dual-Pane TUI
        run_manual_sync_tui(target_file, reference_file, console, args)
        
    elif mode == "auto":
        # Launch the Auto Linear Sync flow
        run_auto_linear_sync(target_file, reference_file, device, translation_model, console, args)