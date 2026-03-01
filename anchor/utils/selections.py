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
        {"id": "2", "icon": "📑", "name": "Reference Sync", "desc": "Automatic Sync using a perfectly timed reference subtitle", "req": []},
        {"id": "3", "icon": "📍", "name": "Point Sync", "desc": "Sync via reference Subtitle", "req": []},
        {"id": "4", "icon": "🌐", "name": "Translate", "desc": "Translate subtitle text to another language", "req": []},
        {"id": "5", "icon": "📝", "name": "Transcribe", "desc": "Generate subtitles from video/audio", "req": ["ffmpeg"]},
        {"id": "6", "icon": "📦", "name": "Container Tasks", "desc": "Extract, Embed, or Strip subtitles from media", "req": ["ffmpeg", "ffprobe"]},
        {"id": "7", "icon": "🔥", "name": "Burn-in", "desc": "Permanently burn subtitles into video", "req": ["ffmpeg"]},
        {"id": "8", "icon": "🧽", "name": "Clean & Fix", "desc": "Repair and clean subtitle files", "req": []},
        {"id": "9", "icon": "🔄", "name": "Convert", "desc": "Convert between subtitle formats", "req": []},
        {"id": "10", "icon": "📥", "name": "Download", "desc": "Automatically find and download matching subtitles", "req": []},
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
                f"[dim white]({t['desc']})[/dim white]",
                ""
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
        "2": "reference",
        "3": "point",
        "4": "translate",
        "5": "transcribe",
        "6": "container",
        "7": "burn",
        "8": "clean_fix",
        "9": "convert",
        "10": "download",
    }
    
    return mapping.get(choice)


def select_container_mode():
    console.print("\n[bold cyan]📦 Select Container Task[/bold cyan]")
    
    menu = Table(box=False, show_header=False, padding=(0, 1))
    menu.add_column("Opt", width=3, justify="right")
    menu.add_column("Icon", width=2, justify="center")
    menu.add_column("Name")
    menu.add_column("Desc")
    menu.add_column("Warning", style="bold red")

    sys_deps = get_system_dependencies()

    tasks = [
        {"id": "1", "icon": "🧲", "name": "Extract", "desc": "(Extract embedded subtitles from media)", "req": ["ffmpeg", "ffprobe"]},
        {"id": "2", "icon": "🧩", "name": "Embed", "desc": "(Embed external subtitles into media)", "req": ["ffmpeg", "ffprobe"]},
        {"id": "3", "icon": "🧹", "name": "Strip", "desc": "(Remove embedded subtitles from media)", "req": ["ffmpeg", "ffprobe"]},
    ]

    valid_choices = []
    default_choice = None

    for t in tasks:
        missing = [d for d in t["req"] if not sys_deps.get(d, False)]
        
        if not missing:
            valid_choices.append(t["id"])
            if default_choice is None:
                default_choice = t["id"]
                
            # Pass an empty string for the Warning column
            menu.add_row(
                f"[bold cyan]{t['id']}.[/bold cyan]", 
                t["icon"], 
                f"[bold white]{t['name']}[/bold white]", 
                f"[dim white]{t['desc']}[/dim white]",
                "" 
            )
        else:
            missing_str = ", ".join(missing)
            menu.add_row(
                f"[dim]{t['id']}.[/dim]", 
                f"[dim]{t['icon']}[/dim]", 
                f"[dim]{t['name']}[/dim]", 
                f"[dim]{t['desc']}[/dim]",
                f"Missing: {missing_str}"
            )

    console.print(menu)
    console.print("")

    choice = Prompt.ask(
        "[bold]Select Task[/bold]", 
        choices=valid_choices, 
        default=default_choice,
        show_choices=False,
        show_default=True
    )

    mapping = {
        "1": "extract",
        "2": "embed",
        "3": "strip",
    }
    
    return mapping.get(choice)


def select_pointsync_mode():
    """
    Displays the option for Point Sync mode (Auto vs Manual)
    """
    console.print("\n[bold cyan]⚡ Point Sync Method[/bold cyan]")
    
    menu = Table(box=False, show_header=False, padding=(0, 1))
    menu.add_column("Opt", width=3, justify="right")
    menu.add_column("Icon", width=2, justify="center")
    menu.add_column("Name")
    menu.add_column("Desc")
    menu.add_column("Warning", style="bold red")

    sys_deps = get_system_dependencies()

    tasks = [
        {"id": "1", "icon": "🤖", "name": "Auto-Linear", "desc": "(Finds start/end matches in reference file)", "req": []},
        {"id": "2", "icon": "✋", "name": "Manual-Pick", "desc": "(You select matching lines visually)", "req": []},
    ]

    valid_choices = []
    default_choice = None

    for t in tasks:
        missing = [d for d in t["req"] if not sys_deps.get(d, False)]
        
        if not missing:
            valid_choices.append(t["id"])
            if default_choice is None:
                default_choice = t["id"]
                
            menu.add_row(
                f"[bold cyan]{t['id']}.[/bold cyan]", 
                t["icon"], 
                f"[bold white]{t['name']}[/bold white]", 
                f"[dim white]{t['desc']}[/dim white]",
                ""
            )
        else:
            missing_str = ", ".join(missing)
            menu.add_row(
                f"[dim]{t['id']}.[/dim]", 
                f"[dim]{t['icon']}[/dim]", 
                f"[dim]{t['name']}[/dim]", 
                f"[dim]{t['desc']}[/dim]",
                f"Missing: {missing_str}"
            )

    console.print(menu)
    console.print("")

    choice = Prompt.ask(
        "[bold]Select Method[/bold]", 
        choices=valid_choices, 
        default=default_choice,
        show_choices=False,
        show_default=True
    )

    mapping = {
        "1": "auto",
        "2": "manual",
    }
    
    return mapping.get(choice)


def select_target_format():
    """
    Displays the Target Format menu for Subtitle Conversion.
    """
    console.print("\n[bold cyan]🎯 Select Target Format[/bold cyan]")
    
    menu = Table(box=False, show_header=False, padding=(0, 1))
    menu.add_column("Opt", width=3, justify="right")
    menu.add_column("Icon", width=2, justify="center")
    menu.add_column("Name")
    menu.add_column("Desc")
    menu.add_column("Warning", style="bold red")

    sys_deps = get_system_dependencies()

    tasks = [
        {"id": "1", "icon": "📝", "name": "SRT", "desc": "(SubRip - Broadest compatibility)", "req": []},
        {"id": "2", "icon": "🌐", "name": "VTT", "desc": "(WebVTT - Standard for web players)", "req": []},
        {"id": "3", "icon": "🎨", "name": "ASS", "desc": "(Advanced SubStation Alpha - Rich styling)", "req": []},
    ]

    valid_choices = []
    default_choice = None

    for t in tasks:
        missing = [d for d in t["req"] if not sys_deps.get(d, False)]
        if not missing:
            valid_choices.append(t["id"])
            if default_choice is None:
                default_choice = t["id"]
            menu.add_row(
                f"[bold cyan]{t['id']}.[/bold cyan]", t["icon"], 
                f"[bold white]{t['name']}[/bold white]", f"[dim white]{t['desc']}[/dim white]", ""
            )
        else:
            missing_str = ", ".join(missing)
            menu.add_row(
                f"[dim]{t['id']}.[/dim]", f"[dim]{t['icon']}[/dim]", 
                f"[dim]{t['name']}[/dim]", f"[dim]{t['desc']}[/dim]", f"Missing: {missing_str}"
            )

    console.print(menu)
    console.print("")

    choice = Prompt.ask(
        "[bold]Select Format[/bold]", 
        choices=valid_choices, 
        default=default_choice,
        show_choices=False,
        show_default=True
    )

    mapping = {"1": ".srt", "2": ".vtt", "3": ".ass"}
    return mapping.get(choice)


def get_subtitle_mode(console):
    """
    Displays the option for Subtitle Download mode (Auto vs Manual)
    """
    console.print("\n[bold cyan]📥 Subtitle Download Mode[/bold cyan]")
    
    menu = Table(box=False, show_header=False, padding=(0, 1))
    menu.add_column("Opt", width=3, justify="right")
    menu.add_column("Icon", width=2, justify="center")
    menu.add_column("Name")
    menu.add_column("Desc")
    menu.add_column("Warning", style="bold red")

    sys_deps = get_system_dependencies()

    tasks = [
        {"id": "1", "icon": "🤖", "name": "Auto-Match", "desc": "(Automatically find and download the highest scoring subtitles)", "req": []},
        {"id": "2", "icon": "✋", "name": "Manual-Pick", "desc": "(View a list of top results and select them visually)", "req": []},
    ]

    valid_choices = []
    default_choice = None

    for t in tasks:
        missing = [d for d in t["req"] if not sys_deps.get(d, False)]
        
        if not missing:
            valid_choices.append(t["id"])
            if default_choice is None:
                default_choice = t["id"]
                
            menu.add_row(
                f"[bold cyan]{t['id']}.[/bold cyan]", 
                t["icon"], 
                f"[bold white]{t['name']}[/bold white]", 
                f"[dim white]{t['desc']}[/dim white]",
                ""
            )
        else:
            missing_str = ", ".join(missing)
            menu.add_row(
                f"[dim]{t['id']}.[/dim]", 
                f"[dim]{t['icon']}[/dim]", 
                f"[dim]{t['name']}[/dim]", 
                f"[dim]{t['desc']}[/dim]",
                f"Missing: {missing_str}"
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
        "1": "auto",
        "2": "manual",
    }
    
    return mapping.get(choice)