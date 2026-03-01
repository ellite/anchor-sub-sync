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

def select_languages_interactive(langs, header_lines=None):
    """
    Multi-select picker for languages.
    Returns a list of selected language strings.
    """
    if len(langs) <= 1:
        return langs
        
    indices = _run_curses_picker(
        options=langs, 
        title="Select Languages to Search", 
        multi_select=True, 
        header_lines=header_lines
    )
    
    return [langs[i] for i in indices]

#  TUI FILE PICKER (Curses)
def select_files_interactive(files, header_lines=None, multi_select=True):
    """
    Multi-select picker for subtitles.
    Returns a list of selected Path objects.
    """
    if not files:
        return []
    
    options = [f.name for f in files]
    indices = _run_curses_picker(options, title="Select", multi_select=multi_select, header_lines=header_lines)
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

    # PALETTE (Original)
    curses.init_pair(1, curses.COLOR_CYAN, -1)
    curses.init_pair(2, curses.COLOR_GREEN, -1)
    curses.init_pair(3, curses.COLOR_BLACK, curses.COLOR_CYAN)
    curses.init_pair(4, curses.COLOR_MAGENTA, -1)
    curses.init_pair(5, curses.COLOR_WHITE, -1)
    
    # PALETTE (New Syntax Highlighting Colors)
    curses.init_pair(10, curses.COLOR_GREEN, -1)
    curses.init_pair(11, curses.COLOR_RED, -1)
    curses.init_pair(12, curses.COLOR_YELLOW, -1)
    curses.init_pair(13, curses.COLOR_BLUE, -1)
    curses.init_pair(14, curses.COLOR_WHITE, -1)
    curses.init_pair(15, curses.COLOR_MAGENTA, -1)

    # Map the Rich tags to our new Curses pairs
    COLORS = {
        '[green]': curses.color_pair(10) | curses.A_BOLD,
        '[red]': curses.color_pair(11) | curses.A_BOLD,
        '[yellow]': curses.color_pair(12) | curses.A_BOLD,
        '[blue]': curses.color_pair(13) | curses.A_BOLD,
        '[white]': curses.color_pair(14) | curses.A_NORMAL,
        '[magenta]': curses.color_pair(15) | curses.A_BOLD,
    }

    current_row = 0
    selected_indices = set()
    offset = 0

    while True:
        stdscr.clear()
        height, width = stdscr.getmaxyx()
        
        start_y = 0

        # DRAW APP HEADER
        if header_lines:
            for line in header_lines:
                safe_line = line[:width-1]
                style = curses.color_pair(4) | curses.A_BOLD if start_y == 0 else curses.color_pair(5) | curses.A_DIM
                stdscr.addstr(start_y, 0, safe_line, style)
                start_y += 1
            start_y += 1 

        # DRAW MENU TITLE & INSTRUCTIONS
        stdscr.addstr(start_y, 0, f"📋 {title}", curses.color_pair(2) | curses.A_BOLD)
        start_y += 1
        
        if multi_select:
            instr = "   [↑/↓] Navigate   [Space] Toggle Select   [TAB] Select All/None   [Enter] Confirm   [Q/Esc] Skip"
        else:
            instr = "   [↑/↓] Navigate   [Space] Select   [Enter] Confirm   [Q/Esc] Skip"
        stdscr.addstr(start_y, 0, instr, curses.color_pair(5) | curses.A_DIM)
        start_y += 1
        start_y += 1 

        # DRAW FILE LIST
        list_height = height - start_y - 1
        if list_height < 1: list_height = 1

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
                    checkbox = "(*)"

            row_raw = f" {checkbox} {options[idx]}"
            
            # Strip tags to calculate the "true" string length
            row_clean = re.sub(r'\[/?(?:green|red|yellow|blue|white|magenta)\]', '', row_raw)

            # DRAWING STYLES
            if is_hovered:
                # Highlight Bar: Strip colors so the cyan block stays readable
                trunc_clean = row_clean[:width-4] + "..." if len(row_clean) > width - 1 else row_clean
                
                stdscr.attron(curses.color_pair(3))
                stdscr.addstr(y_pos, 0, trunc_clean)
                stdscr.addstr(y_pos, len(trunc_clean), " " * (width - len(trunc_clean) - 1))
                stdscr.attroff(curses.color_pair(3))
            else:
                # Normal Rows: Parse the string chunk by chunk and paint the colors
                parts = re.split(r'(\[/?(?:green|red|yellow|blue|white|magenta)\])', row_raw)
                base_style = curses.color_pair(1) | curses.A_BOLD if is_checked else curses.color_pair(5) | curses.A_DIM
                current_style = base_style
                
                current_x = 0
                for part in parts:
                    if not part: continue
                    
                    if part in COLORS:
                        current_style = COLORS[part]
                    elif part.startswith('[/'):
                        current_style = base_style
                    else:
                        # Print the actual text chunk safely
                        space_left = width - current_x - 1
                        if space_left <= 0:
                            break
                        
                        chunk = part
                        if len(chunk) > space_left:
                            chunk = chunk[:space_left-3] + "..." if space_left > 3 else chunk[:space_left]
                            
                        try:
                            stdscr.addstr(y_pos, current_x, chunk, current_style)
                        except curses.error:
                            pass
                        
                        current_x += len(chunk)

        stdscr.refresh()

        # INPUT HANDLING
        key = stdscr.getch()

        if key == curses.KEY_UP:
            # Wraps to the bottom if going up from the top
            current_row = (current_row - 1) % len(options)
        elif key == curses.KEY_DOWN:
            # Wraps to the top if going down from the bottom
            current_row = (current_row + 1) % len(options)
        elif key == ord(' '):
            if multi_select:
                if current_row in selected_indices:
                    selected_indices.remove(current_row)
                else:
                    selected_indices.add(current_row)
            else:
                return [current_row] 
        elif multi_select and key == 9: 
            if len(selected_indices) == len(options):
                selected_indices.clear()
            else:
                selected_indices = set(range(len(options)))     
        elif key == 10: 
            if selected_indices:
                return sorted(list(selected_indices))
            else:
                return [current_row]
        elif key == 27 or key == ord('q'): 
            return []