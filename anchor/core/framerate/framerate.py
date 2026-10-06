import curses

from ...utils.files import get_files, select_files_interactive, open_subtitle, backup_if_needed
from ..pointsync.manual import _dual_pane_picker

SUPPORTED_EXTENSIONS = {".srt", ".ass", ".vtt", ".sub"}

# Video frame rates offered in the picker. The NTSC rates are exact fractions (24000/1001 ...), which is what video
# files really use; the label is what people call them.
FRAME_RATES = [
    ("23.976", 24000 / 1001),
    ("24", 24.0),
    ("25", 25.0),
    ("29.97", 30000 / 1001),
    ("30", 30.0),
    ("50", 50.0),
    ("59.94", 60000 / 1001),
    ("60", 60.0),
]


def convert_frame_rate(subs, from_fps, to_fps):
    """Retimes every event of `subs` in place from `from_fps` to `to_fps`. Returns the factor applied.

    `from_fps` is the frame rate the subtitle was timed for; `to_fps` is the frame rate of the video it must match.
    The same frames take `from_fps / to_fps` as long: a subtitle for a 25 fps video shown on a 23.976 fps video
    must be stretched by 25 / 23.976, so for that conversion the factor is 25 / 23.976 (greater than 1).
    """
    factor = from_fps / to_fps
    for event in subs:
        event.start = int(round(event.start * factor))
        event.end = int(round(event.end * factor))
    return factor


def _timestamp(ms):
    total = ms / 1000.0
    return f"{int(total // 3600):d}:{int(total % 3600 // 60):02d}:{total % 60:06.3f}"


def run_framerate(args, console):
    """Interactive workflow: pick subtitle file(s), then pick the From and To frame rates in the dual-pane view."""
    console.print("\n[bold magenta]🎥  Change Frame Rate[/bold magenta]")

    files = get_files(SUPPORTED_EXTENSIONS)
    if not files:
        console.print("[bold red]❌ No subtitle files found in this folder![/bold red]")
        return

    selected = select_files_interactive(files, header_lines=["🎥  CHANGE FRAME RATE", "Select the subtitle file(s) to retime."])
    if not selected:
        console.print("[yellow]No files selected. Returning to menu.[/yellow]")
        return

    rates = [(i, fps, f"{label} fps") for i, (label, fps) in enumerate(FRAME_RATES)]
    from_fps, to_fps = curses.wrapper(
        _dual_pane_picker,
        rates,
        rates,
        "CHANGE FRAME RATE",
        headers=("FROM (frame rate the subtitle was made for)", "TO (frame rate of your video)"),
        row_text=lambda item: item[2],
        instructions="[Arrows] Move/Switch  [Space] Select  [Enter] Confirm  [Q] Cancel",
    )
    if from_fps is None:
        console.print("[yellow]Cancelled.[/yellow]")
        return

    names = {fps: label for label, fps in FRAME_RATES}
    from_label, to_label = names[from_fps], names[to_fps]
    if from_fps == to_fps:
        console.print(f"[yellow]From and To are both {from_label} fps. Nothing to change.[/yellow]")
        return

    console.print(f"[dim]Retiming {len(selected)} file(s) from {from_label} fps to {to_label} fps[/dim]")

    failed = 0
    for path in selected:
        try:
            subs = open_subtitle(path)
            if not len(subs):
                console.print(f"[yellow]⚠️ {path.name} has no cues. Skipped.[/yellow]")
                continue
            last_before = subs[-1].start
            factor = convert_frame_rate(subs, from_fps, to_fps)

            if args and getattr(args, "overwrite", False):
                backup_if_needed(path, args)
                output = path
            else:
                output = path.with_name(f"{path.stem}.synced{path.suffix}")
            subs.save(str(output))

            console.print(f"\n[bold green]✅ {path.name}[/bold green]")
            console.print(f"   📐 Factor: [cyan]{factor:.6f}[/cyan] ({from_label} → {to_label} fps)")
            console.print(f"   ⏱️  Last cue starts: [cyan]{_timestamp(last_before)}[/cyan] → [cyan]{_timestamp(subs[-1].start)}[/cyan]")
            console.print(f"   💾 Saved to: [underline]{output.name}[/underline]")
        except Exception as e:
            failed += 1
            console.print(f"[bold red]❌ Failed on {path.name}:[/bold red] {e}")

    if failed:
        console.print(f"\n[bold yellow]Done with {failed} failure(s).[/bold yellow]")
    else:
        console.print("\n[bold green]✨ Frame rate change complete.[/bold green]")
