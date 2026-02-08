import curses
from rich.console import Console
from .common import get_filtered_lines, apply_linear_correction
from ...utils.files import open_subtitle

console = Console()

def run_manual_sync_tui(target_file, reference_file, console=console):
    """
    Orchestrates the Manual Point Sync flow:
    """
    
    # Load Files
    try:
        console.print("[dim]⏳ Loading subtitles...[/dim]")
        sub_target = open_subtitle(target_file)
        sub_ref = open_subtitle(reference_file)
    except Exception as e:
        console.print(f"[bold red]❌ Error loading files:[/bold red] {e}")
        return

    # Pick Start Point
    # Grab the first 100 dialogue lines for the "Start" context
    target_start_lines = get_filtered_lines(sub_target, section="start", limit=100)
    ref_start_lines = get_filtered_lines(sub_ref, section="start", limit=100)

    if not target_start_lines or not ref_start_lines:
        console.print("[red]❌ Not enough dialogue lines found to sync start points.[/red]")
        return

    p1_target, p1_ref = curses.wrapper(
        _dual_pane_picker, 
        target_start_lines, 
        ref_start_lines, 
        "PHASE 1: SYNC START (First 100 lines)"
    )

    if p1_target is None:
        console.print("[yellow]Sync cancelled by user.[/yellow]")
        return

    # Pick End Point
    # Grab the last 100 dialogue lines for the "End" context
    target_end_lines = get_filtered_lines(sub_target, section="end", limit=100)
    ref_end_lines = get_filtered_lines(sub_ref, section="end", limit=100)

    p2_target, p2_ref = curses.wrapper(
        _dual_pane_picker, 
        target_end_lines, 
        ref_end_lines, 
        "PHASE 2: SYNC END (Last 100 lines)"
    )

    if p2_target is None:
        console.print("[yellow]Sync cancelled by user.[/yellow]")
        return

    # Calculate Linear Correction (y = mx + c)
    # m = slope (speed factor), c = intercept (offset)
    try:
        m, c = apply_linear_correction(sub_target, p1_target, p1_ref, p2_target, p2_ref)
    except ZeroDivisionError:
        console.print("[bold red]❌ Error:[/bold red] Start and End points are identical! Cannot calculate slope.")
        return
    

    console.print(f"\n[bold green]⚡ Applying Linear Correction...[/bold green]")
    console.print(f"   📐 Speed Factor (Stretch): [cyan]{m:.6f}[/cyan]")
    console.print(f"   ⏱️  Offset (Shift):        [cyan]{c:.2f} ms[/cyan]")

    # Save
    output_path = target_file.with_suffix(".synced.srt")
    sub_target.save(str(output_path))
    
    console.print(f"\n[bold green]✅ Sync Complete![/bold green]")
    console.print(f"💾 Saved to: [underline]{output_path.name}[/underline]")

#  CURSES TUI ENGINE

def _dual_pane_picker(stdscr, left_data, right_data, title):
    """
    Dual-pane TUI for picking a matching pair of lines.
    Returns: (left_timestamp, right_timestamp) or (None, None)
    """
    # Setup
    curses.curs_set(0)
    stdscr.clear()
    curses.start_color()
    curses.use_default_colors()
    
    # Palette
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_CYAN)  # Active Cursor
    curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_GREEN) # Locked Selection
    curses.init_pair(3, curses.COLOR_WHITE, -1)                 # Normal Text
    curses.init_pair(4, curses.COLOR_YELLOW, -1)                # Main Title
    curses.init_pair(5, curses.COLOR_CYAN, -1)                  # Column Headers (NEW)
    
    # State
    l_idx = 0
    r_idx = 0
    active_col = 0  # 0=Left, 1=Right
    
    l_sel = None    # Locked Index Left
    r_sel = None    # Locked Index Right
    
    # Viewport Offsets (for scrolling)
    l_offset = 0
    r_offset = 0

    while True:
        stdscr.clear()
        height, width = stdscr.getmaxyx()
        
        # Draw Main Title in Yellow
        stdscr.addstr(0, 0, f"{title}", curses.color_pair(4) | curses.A_BOLD)
        
        # Instructions
        instr = "[Arrows] Move/Switch  [Space] Select  [Enter] Confirm Pair  [Q] Cancel"
        stdscr.addstr(1, 0, instr, curses.color_pair(3) | curses.A_DIM)
        
        col_w = (width // 2) - 2
        
        header_str = f"{'TARGET (Bad)':<{col_w}} | {'REFERENCE (Good)':<{col_w}}"
        stdscr.addstr(3, 0, header_str, curses.color_pair(5) | curses.A_BOLD)
        stdscr.addstr(4, 0, "-" * (width-1), curses.color_pair(3) | curses.A_DIM)
        list_h = height - 6
        if list_h < 1: list_h = 1

        # Scroll Logic Left
        if l_idx < l_offset: l_offset = l_idx
        elif l_idx >= l_offset + list_h: l_offset = l_idx - list_h + 1
        
        # Scroll Logic Right
        if r_idx < r_offset: r_offset = r_idx
        elif r_idx >= r_offset + list_h: r_offset = r_idx - list_h + 1

        for i in range(list_h):
            row_y = i + 5
            
            # Left Column
            idx = l_offset + i
            if idx < len(left_data):
                orig_idx, time_ms, text = left_data[idx]
                ts = f"[{time_ms//60000:02d}:{(time_ms%60000)//1000:02d}]"
                display_text = f"{ts} {text}"
                
                # Truncate
                if len(display_text) > col_w - 4:
                    display_text = display_text[:col_w - 7] + "..."
                
                style = curses.color_pair(3)
                prefix = "[ ] "
                
                if l_sel == idx:
                    style = curses.color_pair(2)
                    prefix = "[x] "
                
                if active_col == 0 and l_idx == idx:
                    style = curses.color_pair(1) # Cursor overrides color
                
                stdscr.addstr(row_y, 0, f"{prefix}{display_text:<{col_w-4}}", style)

            stdscr.addstr(row_y, col_w, " | ", curses.color_pair(3) | curses.A_DIM)

            # Right Column
            idx = r_offset + i
            if idx < len(right_data):
                orig_idx, time_ms, text = right_data[idx]
                ts = f"[{time_ms//60000:02d}:{(time_ms%60000)//1000:02d}]"
                display_text = f"{ts} {text}"
                
                if len(display_text) > col_w - 4:
                    display_text = display_text[:col_w - 7] + "..."

                style = curses.color_pair(3)
                prefix = "[ ] "

                if r_sel == idx:
                    style = curses.color_pair(2)
                    prefix = "[x] "

                if active_col == 1 and r_idx == idx:
                    style = curses.color_pair(1)

                stdscr.addstr(row_y, col_w + 3, f"{prefix}{display_text:<{col_w-4}}", style)

        stdscr.refresh()

        key = stdscr.getch()

        if key == curses.KEY_UP:
            if active_col == 0: l_idx = max(0, l_idx - 1)
            else: r_idx = max(0, r_idx - 1)
            
        elif key == curses.KEY_DOWN:
            if active_col == 0: l_idx = min(len(left_data) - 1, l_idx + 1)
            else: r_idx = min(len(right_data) - 1, r_idx + 1)
            
        elif key == curses.KEY_LEFT:
            active_col = 0
            
        elif key == curses.KEY_RIGHT:
            active_col = 1
            
        elif key == ord(' '):
            if active_col == 0: l_sel = l_idx
            else: r_sel = r_idx
            
        elif key == 10: # Enter
            if l_sel is not None and r_sel is not None:
                # Return Timestamps
                return left_data[l_sel][1], right_data[r_sel][1]
                
        elif key == 27 or key == ord('q'): # Esc/Q
            return None, None