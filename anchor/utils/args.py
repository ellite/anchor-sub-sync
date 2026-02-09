import argparse

def parse_arguments():
    """
    Handles CLI argument parsing for Anchor.
    Returns:
        Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Anchor Subtitle Sync")
    
    # Model Configuration
    parser.add_argument(
        "-m", "--model", 
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

    # Sync Options
    parser.add_argument(
        "-o", "--overwrite",
        action="store_true",
        help="Overwrite synced files without adding .synced suffix",
        default=False
    )

    # Automation / Files
    parser.add_argument(
        "-r", "--reference",
        type=str,
        help="For unattended sync, provide reference subtitle file path for point sync",
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
        help="For unattended sync, provide path to the video file if the script fails to auto-match",
        default=None
    )
    
    return parser.parse_args()