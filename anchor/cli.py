import sys

from . import pytorch_compat
pytorch_compat.apply_patches()

from rich.console import Console
from .hardware import get_compute_device
from .utils.args import parse_arguments
from .utils.selections import select_container_mode, select_run_mode, select_pointsync_mode
from .utils.config import load_config
from .core.audiosync.audiosync import run_audiosync
from .core.pointsync.pointsync import run_pointsync
from .core.translate.translate import run_translation
from .core.transcribe.transcribe import run_transcription
from .core.container.container import run_container_tasks
from .core.clean.clean import run_clean_fix
from .core.burn.burn import run_burn
from .core.convert.convert import run_convert
from .core.download.download import run_download
from . import __version__

console = Console()

SUPPORTED_EXTENSIONS = {".srt", ".ass", ".vtt", ".sub"}

def main():
    args = parse_arguments()

    try:
        console.clear()
        console.print(f"[bold blue]⚓ Anchor Subtitle Sync {__version__}[/bold blue]\n")

        config = load_config()
        hw_overrides = config.get("hardware_overrides", {})

        final_audio_model = args.audio_model or hw_overrides.get("audio_model")
        final_batch_size = args.batch_size or hw_overrides.get("batch_size")
        final_translation_model = args.translation_model or hw_overrides.get("translation_model")

        # Hardware Check
        device, compute_type, batch_size, model_size, translation_model = get_compute_device(force_model=final_audio_model, force_batch=final_batch_size, force_translation_model=final_translation_model)
        console.print(f"[dim]Engine configured for: [bold white]{device}[/bold white] (model: {model_size}, precision: {compute_type}, batch size: {batch_size}, translation model: {translation_model})[/dim]\n")

        # Check if it should run in unattended mode
        # If -s / --subtitle is porvided, it will run in unattended mode.
        if args.subtitle:
            # If -v / --video is provided together with -s, it's a audio sync
            if args.video:
                run_audiosync(args, device, model_size, compute_type, batch_size, translation_model, console)
            # If -r / --reference is provided together with -s, it's a point sync    
            elif args.reference:
                run_pointsync(args, "auto", device, translation_model, console)
            # if -l / --language is provided together with -s, it's a translation 
            elif args.language:
                run_translation(args, device, translation_model, console)
            # If only -s is provided, it will run unattended mode and try to auto-match the video file
            else:
                run_audiosync(args, device, model_size, compute_type, batch_size, translation_model, console)
        elif args.download:
            run_download(args, config, console)        
        elif args.video:
            # If only -v / --video is provided, it will run transcription in unattended mode
            run_transcription(args, device, model_size, compute_type, console)

        else:
            # Interactive mode
            mode = select_run_mode()
            if (mode == "audio"):
                run_audiosync(args, device, model_size, compute_type, batch_size, translation_model, console)
            elif (mode == "point"):
                point_mode = select_pointsync_mode()
                run_pointsync(args, point_mode, device, translation_model, console)
            elif (mode == "translate"):
                run_translation(args, device, translation_model, console)
            elif (mode == "transcribe"):
                run_transcription(args, device, model_size, compute_type, console)
            elif (mode == "container"):
                container_mode = select_container_mode()
                run_container_tasks(args, container_mode, console)
            elif (mode == "burn"):
                run_burn(args, device, console)
            elif (mode == "clean_fix"):
                run_clean_fix(args, console)
            elif mode == "convert":
                run_convert(args, device, console)
            elif mode == "download":
                run_download(args, config, console)

    except KeyboardInterrupt:
        console.print("\n[bold red]✖  Aborted by user.[/bold red]")
        sys.exit(130)
        
    except Exception as e:
        console.print(f"\n[bold red]💥 An unexpected error occurred:[/bold red] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()