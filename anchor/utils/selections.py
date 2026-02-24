from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from rich import box
from .dependencies import get_system_dependencies

console = Console()

def select_run_mode():
    """
    Displays the Main Menu inline with dynamic dependency checking.
    """
    console.print("\n[bold cyan]⚡ Select task[/bold cyan]")
    
    menu = Table(box=False, show_header=False, padding=(0, 1))
    menu.add_column("Opt", width=3, justify="right")
    menu.add_column("Icon", width=2, justify="center")
    menu.add_column("Name")
    menu.add_column("Desc")
    menu.add_column("Warning", style="bold red")

    sys_deps = get_system_dependencies()

    tasks = [
        {"id": "1", "icon": "🔊", "name": "Audio Sync", "desc": "Automatic Sync via Whisper", "req": ["ffmpeg"]},
        {"id": "2", "icon": "📍", "name": "Point Sync", "desc": "Sync via reference Subtitle", "req": []},
        {"id": "3", "icon": "🌐", "name": "Translate", "desc": "Translate subtitle text to another language", "req": []},
        {"id": "4", "icon": "📝", "name": "Transcribe", "desc": "Generate subtitles from video/audio", "req": ["ffmpeg"]},
        {"id": "5", "icon": "📦", "name": "Container Tasks", "desc": "Extract, Embed, or Strip subtitles from media", "req": ["ffmpeg", "ffprobe"]},
        {"id": "6", "icon": "🔥", "name": "Burn-in", "desc": "Permanently burn subtitles into video", "req": ["ffmpeg"]},
        {"id": "7", "icon": "🧽", "name": "Clean & Fix", "desc": "Repair and clean subtitle files", "req": []},
    ]

    valid_choices = []
    default_choice = None

    for t in tasks:
        # Check which required tools are missing for this specific task
        missing = [d for d in t["req"] if not sys_deps.get(d, False)]
        
        if not missing:
            # All good! Add to valid choices and style normally
            valid_choices.append(t["id"])
            if default_choice is None:
                default_choice = t["id"] # Set default to the first available option
                
            menu.add_row(
                f"[bold cyan]{t['id']}.[/bold cyan]", 
                t["icon"], 
                f"[bold white]{t['name']}[/bold white]", 
                f"[dim white]({t['desc']})[/dim white]"
            )
        else:
            # Missing deps! Gray out and add warning
            missing_str = ", ".join(missing)
            menu.add_row(
                f"[dim]{t['id']}.[/dim]", 
                f"[dim]{t['icon']}[/dim]", 
                f"[dim]{t['name']}[/dim]", 
                f"[dim]({t['desc']})[/dim]",
                f"[bold red]Missing: {missing_str}[/bold red]"
            )

    console.print(menu)
    console.print("")

    choice = Prompt.ask(
        "[bold]Select Mode[/bold]", 
        choices=valid_choices, 
        default=default_choice,
        show_choices=False,
        show_default=True
    )

    mapping = {
        "1": "audio",
        "2": "point",
        "3": "translate",
        "4": "transcribe",
        "5": "container",
        "6": "burn",
        "7": "clean_fix",
    }
    
    return mapping.get(choice)


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