import sys
from datetime import date

from rich.console import Console
from .utils.args import parse_arguments
from .utils.selections import select_container_mode, select_run_mode, select_pointsync_mode
from .utils.config import load_config, save_config
from . import __version__

console = Console()

SUPPORTED_EXTENSIONS = {".srt", ".ass", ".vtt", ".sub"}

# Tasks that never touch torch / whisperx / ctranslate2. These skip engine setup
# entirely so they don't pay the ~2s ML import cost.
LIGHT_TASKS = {"download", "container", "burn", "clean_fix", "convert"}


def _resolve_task(args):
    """Maps CLI args to a task name, or returns None for interactive selection."""
    if args.api:
        return "api"
    if args.subtitle:
        if args.video:
            return "audio"
        if args.reference:
            return "reference"
        if args.language:
            return "translate"
        return "audio"
    if args.download:
        return "download"
    if args.video:
        return "transcribe"
    return None


def _cached_device(config):
    """Best-effort device string from the cached hardware profile, without importing torch.

    Used by light tasks (ffmpeg hw-accel, OCR) that benefit from knowing the GPU but
    don't need full detection. Falls back to 'cpu' until a heavy run (or --check-hardware)
    populates the cache.
    """
    return config.get("hardware", {}).get("device") or "cpu"


def _print_hardware_summary(config):
    """One-line reminder of the cached hardware profile (no torch import)."""
    hw = config.get("hardware", {})
    if not hw:
        console.print(
            "[dim]Hardware: not profiled yet — detected on the first sync/transcribe/translate "
            "run, or now with [white]--check-hardware[/white].[/dim]\n"
        )
        return
    label = hw.get("label") or hw.get("device", "unknown")
    console.print(
        f"[dim]Hardware: [white]{label}[/white]  ·  {hw.get('model_size')} / {hw.get('compute_type')} / "
        f"batch {hw.get('batch_size')}  ·  cached {hw.get('last_checked', '?')} "
        f"([white]--check-hardware[/white] to refresh)[/dim]\n"
    )


def _detect_and_cache(args, config):
    """Runs hardware detection and (unless --cpu) persists it to the config cache. Imports torch."""
    from .hardware import HARDWARE_CACHE_KEYS, detect_hardware, hardware_signature

    detected = detect_hardware(force_cpu=args.cpu)
    if not args.cpu:
        config["hardware"] = {
            "signature": hardware_signature(),
            **dict(zip(HARDWARE_CACHE_KEYS, detected)),
            "last_checked": date.today().isoformat(),
        }
        save_config(config)
    return detected


def _resolve_engine(args, config):
    """Returns the hardware profile (cached or freshly detected) with overrides applied."""
    from .hardware import HARDWARE_CACHE_KEYS, apply_overrides, hardware_signature

    hw_overrides = config.get("hardware_overrides", {})
    forces = dict(
        force_model=args.audio_model or hw_overrides.get("audio_model"),
        force_batch=args.batch_size or hw_overrides.get("batch_size"),
        force_translation_model=args.translation_model or hw_overrides.get("translation_model"),
    )

    # Detection (esp. CUDA init) is slow, so the raw profile is cached in the config and
    # reused while the host fingerprint matches. A --check-hardware refresh already ran in
    # main() by this point; --cpu always detects fresh and is never cached (one-off profile).
    hw_cache = config.get("hardware", {})
    cache_valid = (
        not args.cpu
        and hw_cache.get("signature") == hardware_signature()
        and all(k in hw_cache for k in HARDWARE_CACHE_KEYS)
    )

    detected = tuple(hw_cache[k] for k in HARDWARE_CACHE_KEYS) if cache_valid else _detect_and_cache(args, config)
    return apply_overrides(detected, **forces)


def _run_light_task(task, args, config):
    """Dispatch for tasks that don't need the ML engine."""
    if task == "download":
        from .core.download.download import run_download
        run_download(args, config, console)
    elif task == "container":
        from .core.container.container import run_container_tasks
        run_container_tasks(args, select_container_mode(), console)
    elif task == "burn":
        from .core.burn.burn import run_burn
        run_burn(args, _cached_device(config), console)
    elif task == "clean_fix":
        from .core.clean.clean import run_clean_fix
        run_clean_fix(args, console)
    elif task == "convert":
        from .core.convert.convert import run_convert
        run_convert(args, _cached_device(config), console)


def _run_heavy_task(task, args, config):
    """Dispatch for tasks that need torch + a resolved hardware engine."""
    from . import pytorch_compat
    pytorch_compat.apply_patches()

    device, compute_type, batch_size, model_size, translation_model, cpu_threads, _label = _resolve_engine(args, config)
    console.print(
        f"[dim]Engine configured for: [bold white]{device}[/bold white] "
        f"(model: {model_size}, precision: {compute_type}, batch size: {batch_size}, "
        f"translation model: {translation_model})[/dim]\n"
    )

    if task == "api":
        from .api.api import run_apimode
        run_apimode(args, device, model_size, compute_type, batch_size, translation_model, console, config, cpu_threads)
    elif task == "audio":
        from .core.audiosync.audiosync import run_audiosync
        run_audiosync(args, device, model_size, compute_type, batch_size, translation_model, console, cpu_threads)
    elif task == "reference":
        from .core.referencesync.referencesync import run_referencesync
        run_referencesync(args, device, translation_model, compute_type, console, cpu_threads)
    elif task == "translate":
        from .core.translate.translate import run_translation
        run_translation(args, device, translation_model, compute_type, console, cpu_threads)
    elif task == "transcribe":
        from .core.transcribe.transcribe import run_transcription
        run_transcription(args, device, model_size, compute_type, console, cpu_threads)
    elif task == "point":
        from .core.pointsync.pointsync import run_pointsync
        run_pointsync(args, select_pointsync_mode(), device, translation_model, console, cpu_threads)


def main():
    args = parse_arguments()

    try:
        console.clear()
        console.print(f"[bold blue]⚓ Anchor Subtitle Sync {__version__}[/bold blue]\n")

        config = load_config()

        # Explicit refresh request: re-detect now so the user sees the result immediately.
        if args.check_hardware:
            console.print("[bold]Re-checking hardware…[/bold]")
            _detect_and_cache(args, config)

        _print_hardware_summary(config)

        # Unattended tasks come from args; interactive mode asks (before any ML import).
        task = _resolve_task(args) or select_run_mode()

        if task in LIGHT_TASKS:
            _run_light_task(task, args, config)
        elif task:
            _run_heavy_task(task, args, config)

    except KeyboardInterrupt:
        console.print("\n[bold red]✖  Aborted by user.[/bold red]")
        sys.exit(130)

    except Exception as e:
        console.print(f"\n[bold red]💥 An unexpected error occurred:[/bold red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
