import sys
from datetime import date

from . import pytorch_compat
pytorch_compat.apply_patches()

from rich.console import Console
from .hardware import detect_hardware, apply_overrides, hardware_signature, HARDWARE_CACHE_KEYS
from .utils.args import parse_arguments
from .utils.selections import select_container_mode, select_run_mode, select_pointsync_mode
from .utils.config import load_config, save_config
from .api.api import run_apimode
from .core.audiosync.audiosync import run_audiosync
from .core.pointsync.pointsync import run_pointsync
from .core.translate.translate import run_translation
from .core.transcribe.transcribe import run_transcription
from .core.container.container import run_container_tasks
from .core.referencesync.referencesync import run_referencesync
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
        # The detection step (esp. CUDA init) is slow, so the raw profile is cached in the
        # config and reused while the host fingerprint matches. --check-hardware forces a
        # refresh; --cpu always detects fresh and is never cached (it's a one-off profile).
        hw_cache = config.get("hardware", {})
        signature = hardware_signature()
        cache_valid = (
            not args.check_hardware
            and not args.cpu
            and hw_cache.get("signature") == signature
            and all(k in hw_cache for k in HARDWARE_CACHE_KEYS)
        )

        if cache_valid:
            detected = tuple(hw_cache[k] for k in HARDWARE_CACHE_KEYS)
            console.print("[dim]Loaded cached hardware profile (run with --check-hardware to refresh).[/dim]")
        else:
            detected = detect_hardware(force_cpu=args.cpu)
            if not args.cpu:
                config["hardware"] = {
                    "signature": signature,
                    **dict(zip(HARDWARE_CACHE_KEYS, detected)),
                    "last_checked": date.today().isoformat(),
                }
                save_config(config)

        device, compute_type, batch_size, model_size, translation_model, cpu_threads = apply_overrides(
            detected,
            force_model=final_audio_model,
            force_batch=final_batch_size,
            force_translation_model=final_translation_model,
        )
        console.print(f"[dim]Engine configured for: [bold white]{device}[/bold white] (model: {model_size}, precision: {compute_type}, batch size: {batch_size}, translation model: {translation_model})[/dim]\n")

        # Check if it should run in unattended mode
        # If -s / --subtitle is porvided, it will run in unattended mode.
        if args.api:
            run_apimode(args, device, model_size, compute_type, batch_size, translation_model, console, config, cpu_threads)
        elif args.subtitle:
            # If -v / --video is provided together with -s, it's a audio sync
            if args.video:
                run_audiosync(args, device, model_size, compute_type, batch_size, translation_model, console, cpu_threads)
            # If -r / --reference is provided together with -s, it's a reference sync    
            elif args.reference:
                run_referencesync(args, device, translation_model, compute_type, console, cpu_threads)
            # if -l / --language is provided together with -s, it's a translation 
            elif args.language:
                run_translation(args, device, translation_model, compute_type, console, cpu_threads)
            # If only -s is provided, it will run unattended mode and try to auto-match the video file
            else:
                run_audiosync(args, device, model_size, compute_type, batch_size, translation_model, console, cpu_threads)
        elif args.download:
            run_download(args, config, console)        
        elif args.video:
            # If only -v / --video is provided, it will run transcription in unattended mode
            run_transcription(args, device, model_size, compute_type, console, cpu_threads)

        else:
            # Interactive mode
            mode = select_run_mode()
            if (mode == "audio"):
                run_audiosync(args, device, model_size, compute_type, batch_size, translation_model, console, cpu_threads)
            elif (mode == "reference"):
                run_referencesync(args, device, translation_model, compute_type, console, cpu_threads)
            elif (mode == "point"):
                point_mode = select_pointsync_mode()
                run_pointsync(args, point_mode, device, translation_model, console, cpu_threads)
            elif (mode == "translate"):
                run_translation(args, device, translation_model, compute_type, console, cpu_threads)
            elif (mode == "transcribe"):
                run_transcription(args, device, model_size, compute_type, console, cpu_threads)
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