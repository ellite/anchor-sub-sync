import re
import curses
import pysubs2
from rich.console import Console
from pathlib import Path

console = Console()

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".webm"}

def get_files(extensions):
    return sorted([f for f in Path.cwd().iterdir() if f.suffix.lower() in extensions], key=lambda f: f.name)

def find_best_video_match(sub_path):
    """
    Smart matching: Handles language codes (Movie.en.srt -> Movie.mp4)
    """
    clean_name = sub_path.stem
    token_re = re.compile(r'(?:\.(?:[a-z]{2,3}(?:-[a-z]{2})?|synced|sync|hi|ai))$', flags=re.IGNORECASE)
    
    while True:
        new_name = token_re.sub('', clean_name)
        if new_name == clean_name:
            break
        clean_name = new_name

    for ext in VIDEO_EXTENSIONS:
        candidate = sub_path.with_name(clean_name + ext)
        if candidate.exists():
            return candidate

    videos = get_files(VIDEO_EXTENSIONS)
    for vid in videos:
        if clean_name.lower() in vid.name.lower():
            return vid
            
    return None

def open_subtitle(path: Path, **kwargs) -> pysubs2.SSAFile:
    """
    Attempts to load a subtitle file using various common encodings.
    """
    encodings = [
        "utf-8", "utf-8-sig", "cp1252",
        "cp1250", "cp1251", "gb18030",       
        "big5", "shift_jis", "cp949",         
        "cp874", "cp1258", "cp1257",        
        "cp1256", "cp1255", "cp1254",        
        "cp1253", "latin-1", "iso-8859-1",
        "iso-8859-2", "iso-8859-5", "iso-8859-15"
    ]
    
    path_str = str(path)
    
    for enc in encodings:
        try:
            return pysubs2.load(path_str, encoding=enc, **kwargs)
        except Exception as e:
            last_error = e
            continue
            
    console.print(f"[bold red]❌ Failed to open {path.name}. Last error: {last_error}[/bold red]")
    raise ValueError(f"Could not open {path.name}")

#  TUI FILE PICKER (Curses)
def select_files_interactive(files, header_lines=None, multi_select=True):
    """
    Multi-select picker for subtitles.
    Returns a list of selected Path objects.
    """
    if not files:
        return []
    
    options = [f.name for f in files]
    indices = _run_curses_picker(options, title="Select Subtitles", multi_select=multi_select, header_lines=header_lines)
    return [files[i] for i in indices]

def select_video_fallback(sub_filename, header_lines=None):
    """
    Single-select picker for video files.
    Returns a single Path object or None.
    """
    videos = get_files(VIDEO_EXTENSIONS)
    if not videos:
        console.print("[red]No video files found![/red]")
        return None

    options = [f.name for f in videos]
    indices = _run_curses_picker(
        options, 
        title=f"Select Video for: {sub_filename}", 
        multi_select=False,
        header_lines=header_lines
    )
    
    if indices:
        return videos[indices[0]]
    return None

def _run_curses_picker(options, title="Select Files", multi_select=True, header_lines=None):
    """
    Wrapper to handle the curses lifecycle safely.
    """
    return curses.wrapper(_picker_loop, options, title, multi_select, header_lines)

def _picker_loop(stdscr, options, title, multi_select, header_lines):
    # 1. Setup
    curses.curs_set(0) # Hide cursor
    stdscr.clear()
    curses.start_color()
    curses.use_default_colors() # Use terminal transparency if available

    # PALETTE
    # Pair 1: Selected Item (Cyan text)
    curses.init_pair(1, curses.COLOR_CYAN, -1)
    # Pair 2: Main Title (Green)
    curses.init_pair(2, curses.COLOR_GREEN, -1)
    # Pair 3: Highlight Bar (Black Text on Cyan BG)
    curses.init_pair(3, curses.COLOR_BLACK, curses.COLOR_CYAN)
    # Pair 4: Header Info (Magenta)
    curses.init_pair(4, curses.COLOR_MAGENTA, -1)
    # Pair 5: Dim/Instruction Text (White)
    curses.init_pair(5, curses.COLOR_WHITE, -1)

    current_row = 0
    selected_indices = set()
    offset = 0

    while True:
        stdscr.clear()
        height, width = stdscr.getmaxyx()
        
        start_y = 0

        # DRAW APP HEADER (If passed)
        if header_lines:
            for line in header_lines:
                safe_line = line[:width-1]
                # First line bold/magenta, others dim
                style = curses.color_pair(4) | curses.A_BOLD if start_y == 0 else curses.color_pair(5) | curses.A_DIM
                stdscr.addstr(start_y, 0, safe_line, style)
                start_y += 1
            start_y += 1 # Spacer after App Header

        # DRAW MENU TITLE & INSTRUCTIONS
        # Title
        stdscr.addstr(start_y, 0, f"📂 {title}", curses.color_pair(2) | curses.A_BOLD)
        start_y += 1
        
        # Instructions (Dimmed)
        if multi_select:
            instr = "   [↑/↓] Navigate   [Space] Toggle Select   [TAB] Select All/None   [Enter] Confirm"
        else:
            instr = "   [↑/↓] Navigate   [Space] Select   [Enter] Confirm"
        stdscr.addstr(start_y, 0, instr, curses.color_pair(5) | curses.A_DIM)
        start_y += 1
        
        # Spacer (The gap you requested)
        start_y += 1 

        # DRAW FILE LIST
        list_height = height - start_y - 1
        if list_height < 1: list_height = 1

        # Scroll Logic
        if current_row < offset:
            offset = current_row
        elif current_row >= offset + list_height:
            offset = current_row - list_height + 1

        for i in range(list_height):
            idx = offset + i
            if idx >= len(options):
                break
            
            y_pos = start_y + i
            
            is_hovered = (idx == current_row)
            is_checked = (idx in selected_indices)
            
            # Checkbox Icon
            checkbox = "[x]" if is_checked else "[ ]"
            if not multi_select:
                checkbox = "(*)" if is_checked else "( )"
                if not multi_select and is_hovered: 
                    checkbox = "(*)" # Visual feedback for single select

            row_text = f" {checkbox} {options[idx]}"
            
            # Truncate
            if len(row_text) > width - 1:
                row_text = row_text[:width-4] + "..."

            # DRAWING STYLES
            if is_hovered:
                # Highlight Bar: Black text on Cyan Background
                stdscr.attron(curses.color_pair(3))
                stdscr.addstr(y_pos, 0, row_text)
                stdscr.addstr(y_pos, len(row_text), " " * (width - len(row_text) - 1)) # Fill line
                stdscr.attroff(curses.color_pair(3))
            else:
                # Normal Rows
                if is_checked:
                    # Selected but not hovered: Bold Cyan
                    stdscr.addstr(y_pos, 0, row_text, curses.color_pair(1) | curses.A_BOLD)
                else:
                    # Unselected: DIM WHITE (Solves the "too white" issue)
                    stdscr.addstr(y_pos, 0, row_text, curses.color_pair(5) | curses.A_DIM)

        stdscr.refresh()

        # INPUT HANDLING
        key = stdscr.getch()

        if key == curses.KEY_UP:
            current_row = max(0, current_row - 1)
        elif key == curses.KEY_DOWN:
            current_row = min(len(options) - 1, current_row + 1)
        elif key == ord(' '):
            if multi_select:
                if current_row in selected_indices:
                    selected_indices.remove(current_row)
                else:
                    selected_indices.add(current_row)
            else:
                return [current_row] # Single select confirms immediately
        elif multi_select and key == 9: # TAB for Select All/None
            if len(selected_indices) == len(options):
                selected_indices.clear()
            else:
                selected_indices = set(range(len(options)))     
        elif key == 10: # Enter
            if selected_indices:
                # Return explicitly checked items
                return sorted(list(selected_indices))
            else:
                # If nothing checked, return the hovered item as a fallback
                return [current_row]
        elif key == 27 or key == ord('q'): # Esc/Q
            return []