from pathlib import Path
from ...utils.selections import select_target_format
from ...utils.files import open_subtitle
from ...utils.files import select_files_interactive
from .image_subtitles import run_ocr_engine

def run_convert(args, device, console):
    console.print("\n[bold blue]🔄 Convert Mode[/bold blue]\n")

    # Gather ALL allowed subtitle files (Text + Image formats)
    text_exts = {".srt", ".vtt", ".ass", ".ssa"}
    image_exts = {".sup", ".idx", ".sub", ".mks"}
    
    allowed_exts = text_exts.union(image_exts)
    
    available_files = [
        f for f in Path(".").iterdir() 
        if f.is_file() and f.suffix.lower() in allowed_exts
    ]

    if not available_files:
        console.print("[yellow]No subtitle files (text or image) found in the current directory.[/yellow]")
        return

    # Select files to convert using your interactive menu
    header = ["[bold cyan]Select subtitle files to convert (Space to toggle, Enter to confirm):[/bold cyan]"]
    selected_files = select_files_interactive(
        available_files, 
        header_lines=header, 
        multi_select=True
    )
    
    if not selected_files:
        console.print("[yellow]No files selected. Exiting.[/yellow]")
        return
        
    # Just a safety net in case multi_select=True returns a single item when 1 is picked
    if not isinstance(selected_files, list):
        selected_files = [selected_files]

    # Select target format
    target_ext = select_target_format()
    if not target_ext:
        return

    total_files = len(selected_files)
    console.print(f"\n[bold green]🚀 Starting Conversion to {target_ext.upper()} ({total_files} files)...[/bold green]\n")

    # Processing Loop
    for idx, file_path in enumerate(selected_files, 1):
        console.print(f" [black on white] File {idx}/{total_files} [/black on white] [bold cyan]{file_path.name}[/bold cyan]")
        
        file_ext = file_path.suffix.lower()
        
        if file_ext == target_ext:
            console.print(f"   [dim]Skipping (Already {target_ext.upper()})[/dim]")
            continue

        try:
            # --- BRANCH A: IMAGE SUBTITLE (OCR Prep) ---
            if file_ext in image_exts:
                console.print(f"   [yellow]⚠️ Image-based subtitle detected ({file_ext.upper()}). Routing to OCR engine...[/yellow]")
                
                # TODO: This is where we will plug in the OCR function!
                run_ocr_engine(file_path, target_ext, console, device)
                
                continue

            # --- BRANCH B: TEXT SUBTITLE (Standard Conversion) ---
            # Load the file using fallback encodings and tag protection
            subs = open_subtitle(file_path, keep_html_tags=True, keep_unknown_html_tags=True)
            
            # Construct the new filename
            output_path = file_path.with_suffix(target_ext)
            
            # Save it! pysubs2 automatically formats the output based on the extension
            subs.save(str(output_path))
            
            console.print(f"   [bold green]✓[/bold green] Saved as: {output_path.name}")
            
        except Exception as e:
            console.print(f"   [bold red]❌ Failed to convert:[/bold red] {e}")

    console.print("\n[bold green]✨ Conversion complete![/bold green]")