import time
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from ...utils.files import select_files_interactive, open_subtitle, get_files
from ...utils.languages import get_subtitle_language
from ...utils.mappings import get_language_code_for_nllb
from ..translation import translate_subtitle_nllb

SUPPORTED_EXTENSIONS = {".srt", ".ass", ".vtt", ".sub"}

def run_translation(args, device, translation_model, console: Console):
    if args.subtitle and args.language:
         """
        Unattended workflow for translating subtitles.
        """
         selected_files = [args.subtitle]
         target_lang_input = args.language
    else:
        """
        Interactive workflow for translating subtitles.
        """
        console.print("\n[bold yellow]🌐 Subtitle Translation[/bold yellow]")

        selected_files = select_files_interactive(get_files(SUPPORTED_EXTENSIONS), header_lines=["[dim]Select subtitle files to translate:[/dim]"])

        if not selected_files:
            console.print("[yellow]No files selected. Exiting.[/yellow]")
            return

        console.print("\n[bold cyan]Target Language[/bold cyan]")
        console.print("[dim]Enter the 2-letter language code (e.g., 'en', 'pt', 'es', 'fr', 'de')[/dim]")
        
        target_lang_input = Prompt.ask("Target Language Code", default="en")
    
    # Verify/Map to NLLB code
    nllb_target = get_language_code_for_nllb(target_lang_input)
    if not nllb_target:
        console.print(f"[bold red]❌ Invalid or unsupported language code: {target_lang_input}[/bold red]")
        return

    console.print(f"[green]Target set to: {nllb_target}[/green]\n")

    # Processing Loop
    success_count = 0

    if len(selected_files) > 1:
        console.print(f"[green]🌐 Starting batch translation of {len(selected_files)} files...[/green]\n")
    else:
        console.print(f"[green]🌐 Starting translation...[/green]\n")
    
    for file_path in selected_files:
        path = Path(file_path)
        console.print(f"[bold]Translating: {path.name}[/bold]")

        try:
            # Auto-detect Source Language
            detected_lang = get_subtitle_language(path)
            nllb_source = get_language_code_for_nllb(detected_lang)

            if detected_lang == "unknown" or not nllb_source:
                # Fallback: Ask user if detection fails
                nllb_source = Prompt.ask(
                    f"❓ Could not detect language for [cyan]{path.name}[/cyan]. Enter Source Code", 
                    default="en"
                )
                nllb_source = get_language_code_for_nllb(nllb_source)

            console.print(f"🔹 Source: [cyan]{nllb_source}[/cyan] ➔ Target: [cyan]{nllb_target}[/cyan]")
            console.print(f"🧠 Model:  [dim]{translation_model}[/dim]")

            # Load Subtitle
            sub = open_subtitle(path)
            
            # Run Translation (with progress bar)
            start_time = time.time()
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),      
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                # Start with total=None, but the function will update it immediately
                task = progress.add_task("Translating...", total=None)
                
                translated_sub = translate_subtitle_nllb(
                    sub, 
                    nllb_source, 
                    nllb_target, 
                    device=device, 
                    model_id=translation_model,
                    progress=progress,
                    task_id=task
                )
                
            duration = time.time() - start_time
            
            if translated_sub:
                new_stem = path.stem
                if f".{detected_lang}" in new_stem:
                     new_stem = new_stem.replace(f".{detected_lang}", f".{target_lang_input}.ai")
                else:
                    new_stem = f"{new_stem}.{target_lang_input}"

                output_path = path.with_name(f"{new_stem}{path.suffix}")
                
                translated_sub.save(str(output_path))
                console.print(f"[bold green]✅ Done in {duration:.1f}s![/bold green] Saved to: [underline]{output_path.name}[/underline]")
                success_count += 1
            else:
                console.print("[bold red]❌ Translation returned empty result.[/bold red]")

        except Exception as e:
            console.print(f"[bold red]❌ Error processing file:[/bold red] {e}")

    if len(selected_files) > 1:
        console.print(f"\n[bold green]🎉 Batch  translation complete! ({success_count}/{len(selected_files)} files)[/bold green]")
    else:
        console.print(f"\n[bold green]🎉 Translation complete![/bold green]")