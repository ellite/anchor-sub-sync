from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from rich import box

console = Console()

def select_run_mode():
    """
    Displays the Main Menu inline with better styling.
    """
    
    console.print("\n[bold cyan]⚡ Select task[/bold cyan]")
    menu = Table(box=False, show_header=False, padding=(0, 1))
    menu.add_column("Opt", style="bold cyan", width=3, justify="right")
    menu.add_column("Icon", width=2, justify="center")
    menu.add_column("Name", style="bold white")
    menu.add_column("Desc", style="dim white")

    menu.add_row(
        "1.", 
        "🔊", 
        "Audio Sync", 
        "(Automatic Sync via Whisper)"
    )

    menu.add_row(
        "2.", 
        "📍", 
        "Point Sync", 
        "(Sync via reference Subtitle)"
    )

    menu.add_row(
        "3.",
        "🌐",
        "Translate",
        "(Translate subtitle text to another language)"
    )

    menu.add_row(
        "4.",
        "📝",
        "Transcribe",
        "(Generate subtitles from video/audio)"
    )

    menu.add_row(
        "5.",
        "📦",
        "Container Tasks",
        "(Extract, Embed, or Strip subtitles from media)"
    )

    # Burn subtitle to vide
    menu.add_row(
        "6.",
        "🔥",
        "Burn-in",
        "(Permanently burn subtitles into video)"
    )

    console.print(menu)
    console.print("")

    choice = Prompt.ask(
        "[bold]Select Mode[/bold]", 
        choices=["1", "2", "3", "4", "5", "6"], 
        default="1",
        show_choices=False,
        show_default=True
    )

    if choice == "1":
        return "audio"
    elif choice == "2":
        return "point"
    elif choice == "3":
        return "translate"
    elif choice == "4":
        return "transcribe"
    elif choice == "5":
        return "container"
    elif choice == "6":
        return "burn"
    
    return None


def select_container_mode():
    console.print("\n[bold cyan]📦 Select Container Task[/bold cyan]")
    menu = Table(box=False, show_header=False, padding=(0, 1))
    menu.add_column("Opt", style="bold cyan", width=3, justify="right")
    menu.add_column("Icon", width=2, justify="center")
    menu.add_column("Name", style="bold white")
    menu.add_column("Desc", style="dim white")

    menu.add_row(
        "1.", 
        "🧲", 
        "Extract", 
        "(Extract embedded subtitles from media)"
    )

    menu.add_row(
        "2.", 
        "🧩", 
        "Embed", 
        "(Embed external subtitles into media)"
    )

    menu.add_row(
        "3.",
        "🧹",
        "Strip",
        "(Remove embedded subtitles from media)"
    )

    console.print(menu)
    console.print("")

    choice = Prompt.ask(
        "[bold]Select Task[/bold]", 
        choices=["1", "2", "3"], 
        default="1",
        show_choices=False,
        show_default=True
    )

    if choice == "1":
        return "extract"
    elif choice == "2":
        return "embed"
    elif choice == "3":
        return "strip"
    
    return None

def select_pointsync_mode():
    r"""
    Displays the option for Point Sync mode (Auto vs Manual)
    """
    
    console.print("\n[bold cyan]⚡ Point Sync Method[/bold cyan]")
    
    # Same layout as main menu for consistency
    menu = Table(box=None, show_header=False, padding=(0, 1), pad_edge=False)
    menu.add_column("Icon", style="bold cyan", width=3, justify="right")
    menu.add_column("Opt", width=2, justify="center")
    menu.add_column("Name", style="bold white")
    menu.add_column("Desc", style="dim white")

    # Option 1: Automatic Linear
    menu.add_row(
        "1.", 
        "🤖", 
        "Auto-Linear", 
        "(Finds start/end matches in reference file)"
    )

    # Option 2: Manual
    menu.add_row(
        "2.", 
        "✋", 
        "Manual-Pick", 
        "(You select matching lines visually)"
    )

    console.print(menu)
    console.print("") # Spacer

    choice = Prompt.ask(
        "[bold]Select Method[/bold]", 
        choices=["1", "2"], 
        default="1",
        show_choices=False,
        show_default=True
    )

    if choice == "1":
        return "auto"
    elif choice == "2":
        return "manual"
    
    return None