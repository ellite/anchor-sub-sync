from ...utils.files import get_files, select_files_interactive, _run_curses_picker
from ...utils.files import open_subtitle
from .operations import remove_sdh, strip_tags, remove_empty_cues, fix_overlaps, remove_watermarks, fix_capitalization

def run_clean_fix(args, console):
    console.print("\n[bold cyan]✂️  Running Clean & Fix Task[/bold cyan]\n")

    # Pick subtitle files
    subs = get_files({".srt", ".ass", ".vtt", ".sub"})
    if not subs:
        console.print(" [yellow]No subtitle files found in the current directory.[/yellow]")
        return

    selected_files = select_files_interactive(
        subs, 
        header_lines=["Select subtitle file(s) to Clean & Fix (Space to select, Enter to confirm)"],
        multi_select=True
    )
    
    if not selected_files:
        console.print(" [dim]No files selected. Aborting.[/dim]")
        return

    # Define the available cleaning operations
    # The first element is the internal ID, the second is the display string
    operations_map = [
        ("sdh", "Remove SDH (sound descriptions & [speaker labels])"),
        ("tags", "Strip tags & styles (remove HTML/ASS formatting)"),
        ("empty", "Remove empty cues (delete blank entries)"),
        ("overlaps", "Fix overlaps (merge or trim overlapping timings)"),
        ("watermarks", "Remove watermarks (URLs & promo tags) [dim]- English/Generic[/dim]"),
        ("caps", "Fix capitalization (Sentence case) [dim]- English only[/dim]")
    ]

    # Extract just the display strings for the UI picker
    ui_options = [op[1] for op in operations_map]

    # Select which clean/fix operations to perform
    selected_indices = _run_curses_picker(
        ui_options,
        title="Select Cleaning Operations to Apply (Space to toggle, Enter to confirm)",
        multi_select=True
    )

    if not selected_indices:
        console.print(" [dim]No operations selected. Aborting.[/dim]")
        return

    # Map the selected indices back to internal operation IDs
    selected_operations = [operations_map[i][0] for i in selected_indices]

    total_files = len(selected_files)
    total_ops = len(selected_operations)
    
    console.print(f"\n[bold green]🚀 Starting Cleanup ({total_files} file{'s' if total_files > 1 else ''}, {total_ops} operation{'s' if total_ops > 1 else ''})...[/bold green]")

    # Processing Loop
    for task_idx, sub_path in enumerate(selected_files, 1):
        console.print(f"\n[black on white] Task {task_idx}/{total_files} [/black on white] [bold cyan]{sub_path.name}[/bold cyan]")
        
        # Load the subtitle file into memory
        try:
            edit_subs = open_subtitle(sub_path, keep_html_tags=True, keep_unknown_html_tags=True)
        except Exception as e:
            console.print(f" ❌ [bold red]Failed to load subtitle:[/bold red] {e}")
            continue

        # Chain the operations together on the 'edit_subs' object
        for op in selected_operations:
            if op == "sdh":
                console.print(" 🧹 Removing SDH...")
                edit_subs = remove_sdh(edit_subs)
                
            elif op == "tags":
                console.print(" 🏷️ Stripping tags...")
                edit_subs = strip_tags(edit_subs)
                
            elif op == "empty":
                console.print(" 👻 Removing empty cues...")
                edit_subs = remove_empty_cues(edit_subs)
                
            elif op == "overlaps":
                console.print(" ⏱️ Fixing overlaps...")
                edit_subs = fix_overlaps(edit_subs)
                
            elif op == "watermarks":
                console.print(" 🕵️ Hunting watermarks...")
                edit_subs = remove_watermarks(edit_subs)
                
            elif op == "caps":
                console.print(" 🔠 Fixing capitalization...")
                edit_subs = fix_capitalization(edit_subs)

        # Determine output path
        overwrite = getattr(args, 'overwrite', False)
        
        if overwrite:
            output_path = sub_path
        else:
            output_path = sub_path.with_name(f"{sub_path.stem}.cleaned{sub_path.suffix}")
            
        # Save the cleaned subtitle back to disk
        edit_subs.save(str(output_path))
        
        console.print(f" 💾 Saved to: [u]{output_path.name}[/u]")

    console.print("\n[bold green]🧽 Clean & Fix Complete![/bold green]")