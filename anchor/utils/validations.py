import shutil
from rich.console import Console

console = Console()

def check_dependencies():
    if not shutil.which("ffmpeg"):
        console.print("\n[bold red]❌ Critical Error: FFmpeg is missing![/bold red]")
        return False
    if not shutil.which("ffprobe"):
        console.print("\n[bold yellow]⚠️  Performance Warning: ffprobe not found[/bold yellow]")
    return True