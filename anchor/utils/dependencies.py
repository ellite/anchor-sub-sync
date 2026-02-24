import shutil

def get_system_dependencies() -> dict:
    """
    Scans the system for required external tools (not installed via pip).
    Returns a dictionary mapping the tool name to a boolean (True if found).
    """
    return {
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "ffprobe": shutil.which("ffprobe") is not None,
        "tesseract": shutil.which("tesseract") is not None,
    }