import argparse
from anchor import __version__

def parse_arguments():
    """
    Handles CLI argument parsing for Anchor.
    Returns:
        Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Anchor Subtitle Sync")

    parser.add_argument(
        "-V", "--version", 
        action="version", 
        version=f"%(prog)s {__version__}",
        help="Show program's version number and exit."
    )
    
    # Model Configuration
    parser.add_argument(
        "-a", "--audio-model", 
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

    parser.add_argument(
        "--asr",
        choices=["auto", "whisper", "parakeet"],
        default="auto",
        help=("Speech recognition engine for audio sync. 'parakeet' (NVIDIA Parakeet, 25 European languages) is faster and "
              "places cues more accurately; it is installed on first use into ~/.anchor/parakeet-venv. 'auto' (default) uses "
              "Parakeet only when it is already installed and the audio language is supported, otherwise Whisper. "
              "'whisper' never uses Parakeet.")
    )

    # Sync Options
    parser.add_argument(
        "-o", "--overwrite",
        action="store_true",
        help="Overwrite synced files without adding .synced suffix.",
        default=False
    )
    parser.add_argument(
        "-B", "--backup",
        action="store_true",
        help="When used with --overwrite, creates a .bak copy of the original file before overwriting.",
        default=False
    )

    # Automation / Files
    parser.add_argument(
        "-r", "--reference",
        type=str,
        help="For unattended sync, provide reference subtitle file path for point sync.",
        default=None
    )
    parser.add_argument(
        "-s", "--subtitle",
        type=str,
        help="Runs unattended sync on a single subtitle file (provide path to .srt, .ass, etc.)",
        default=None
    )
    parser.add_argument(
        "-v", "--video",
        type=str,
        help="For unattended sync, provide path to the video file if the script fails to auto-match.",
        default=None
    )
    parser.add_argument(
        "-l", "--language",
        type=str,
        help="For unattended mode, provide the target language code (e.g. 'en', 'pt', 'fr') for translation or download.",
        default=None
    )
    parser.add_argument(
        "-d", "--download",
        action="store_true",
        help="For unattended mode, automatically download subtitles. Provide -v with the video file path, or anchor downloads for all videos in the directory.",
        default=False
    )
    
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check whether a subtitle is in sync with its video (use with -s, and -v if the video is not found automatically). "
             "Nothing is written; exit code 3 when the subtitle is not in sync.",
        default=False
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="With --check: act on the advice without asking. Subtitles in sync are skipped, a suggested frame rate change is applied "
             "(and verified, falling back to Audio Sync if it does not bring the subtitle in sync), otherwise Audio Sync is performed.",
        default=False
    )
    parser.add_argument(
        "--from-fps",
        type=float,
        metavar="FPS",
        help="For unattended frame rate change (use with -s and --to-fps): the frame rate the subtitle was made for, e.g. 23.976.",
        default=None
    )
    parser.add_argument(
        "--to-fps",
        type=float,
        metavar="FPS",
        help="For unattended frame rate change (use with -s and --from-fps): the frame rate of the video, e.g. 25.",
        default=None
    )

    parser.add_argument(
        "--missing",
        action="store_true",
        help="Skip video files (or languages) that already have a matching subtitle on disk.",
        default=False
    )

    # Hardware
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU execution, bypassing GPU detection.",
        default=False
    )
    parser.add_argument(
        "--check-hardware",
        action="store_true",
        help="Re-run hardware detection, refresh the cached profile in the config, then continue.",
        default=False
    )

    # API Mode
    parser.add_argument("--api",
        action="store_true",
        help="Enable API mode.",
        default=False
    )
    
    return parser.parse_args()