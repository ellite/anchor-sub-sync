import time
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from ...utils.files import select_files_interactive, open_subtitle, get_files, resolve_output
from ...utils.languages import get_subtitle_language
from ...utils.mappings import get_language_code_for_nllb, ISO_639_MAPPING
from ..translation import translate_subtitle_nllb

SUPPORTED_EXTENSIONS = {".srt", ".ass", ".vtt", ".sub"}


def translated_stem(stem: str, source_lang: str, target_lang: str) -> str:
    """File name (without extension) of a machine translation: always marked `.ai`.

    The source's own language token (`.pt`, `.por`, `.pt-BR`, matched exactly, last one wins) is replaced by `<target>.ai`;
    a name without one gets `.<target>.ai` appended. A source that is already `.ai` does not get a second one.
    """
    parts = stem.split(".")
    for i in range(len(parts) - 1, 0, -1):
        base = parts[i].lower().split("-")[0].split("_")[0]
        if ISO_639_MAPPING.get(base, base) == (source_lang or "").lower():
            rest = [p for p in parts[i + 1:] if p.lower() != "ai"]
            return ".".join(parts[:i] + [target_lang, "ai"] + rest)
    return ".".join([p for p in parts if p.lower() != "ai"] + [target_lang, "ai"])

def run_translation(args, device, translation_model, compute_type, console: Console, cpu_threads=0):
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
                    compute_type=compute_type,
                    progress=progress,
                    task_id=task,
                    cpu_threads=cpu_threads
                )
                
            duration = time.time() - start_time
            
            if translated_sub:
                new_stem = translated_stem(path.stem, detected_lang, target_lang_input)

                output_path = resolve_output(path.with_name(f"{new_stem}{path.suffix}"), args)
                
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