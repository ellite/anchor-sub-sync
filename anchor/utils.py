import os
import sys
import re
import threading
import shutil
import subprocess
import pysubs2
import logging
from rich.console import Console
from pathlib import Path

# --- Cross-Platform Imports ---
if os.name == 'posix':
    import pty
    import fcntl
else:
    pty = None
    fcntl = None
    termios = None

console = Console()

ISO_639_MAPPING = {
    'eng': 'en', 'spa': 'es', 'fra': 'fr', 'deu': 'de', 'ita': 'it',
    'por': 'pt', 'jpn': 'ja', 'zho': 'zh', 'chi': 'zh', 'rus': 'ru',
    'kor': 'ko', 'nld': 'nl', 'swe': 'sv', 'nor': 'no', 'dan': 'da',
    'fin': 'fi', 'tur': 'tr', 'pol': 'pl', 'ukr': 'uk', 'ara': 'ar',
    'hin': 'hi', 'vie': 'vi', 'tha': 'th', 'ind': 'id'
}

def check_dependencies():
    if not shutil.which("ffmpeg"):
        console.print("\n[bold red]❌ Critical Error: FFmpeg is missing![/bold red]")
        return False
    if not shutil.which("ffprobe"):
        console.print("\n[bold yellow]⚠️  Performance Warning: ffprobe not found[/bold yellow]")
    return True

def format_timestamp(ms: int) -> str:
    seconds = ms / 1000.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02}:{m:02}:{s:06.3f}".replace('.', ',')

def clean_text(text: str) -> str:
    t = re.sub(r'\{.*?\}', '', text) 
    t = re.sub(r'<.*?>', '', t)
    t = re.sub(r'\\N', ' ', t)
    t = re.sub(r'[^\w\s]', '', t)
    return t.lower().strip()

def parse_range_selection(selection_str: str, max_items: int) -> list:
    if not selection_str.strip():
        return list(range(max_items)) 
    selected_idxs = set()
    parts = selection_str.split(",")
    for part in parts:
        part = part.strip()
        if "-" in part:
            try:
                start_s, end_s = part.split("-", 1)
                start = int(start_s)
                end = int(end_s)
                step = 1 if start <= end else -1
                for i in range(start, end + step, step):
                    if 0 < i <= max_items: selected_idxs.add(i - 1)
            except ValueError: continue
        elif part.isdigit():
            val = int(part)
            if 0 < val <= max_items: selected_idxs.add(val - 1)
    return sorted(list(selected_idxs))

def get_audio_language(video_path: Path):
    if shutil.which("ffprobe") is None: return None
    try:
        abs_path = str(Path(video_path).resolve())
        cmd = ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream_tags=language", "-of", "default=noprint_wrappers=1:nokey=1", abs_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        raw_lang = result.stdout.strip().lower()
        if not raw_lang or raw_lang == "und": return None
        if len(raw_lang) == 2: return raw_lang
        return ISO_639_MAPPING.get(raw_lang, None)
    except Exception: return None

class CaptureProgress:
    """
    Hybrid Capture:
    - Linux: Uses PTY + sys.stdout replacement (Required for Bar).
             Includes 'Logger Detach' logic to prevent crashes.
    - Windows: Does nothing.
    """
    def __init__(self, progress, task_id, ui_console=None):
        self.progress = progress
        self.task_id = task_id
        self.ui_console = ui_console or progress.console
        self.master_fd = None
        self.slave_fd = None
        self.reader_thread = None
        self.stop_event = threading.Event()
        self.is_windows = (os.name != 'posix')

    def __enter__(self):
        if self.is_windows: return self

        # Flush before hijacking
        sys.stdout.flush()
        sys.stderr.flush()

        self.orig_stdout_fd = os.dup(1)
        self.orig_stderr_fd = os.dup(2)
        self.saved_stdout = sys.stdout
        self.saved_stderr = sys.stderr

        self.master_fd, self.slave_fd = pty.openpty()
        fl = fcntl.fcntl(self.master_fd, fcntl.F_GETFL)
        fcntl.fcntl(self.master_fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
        
        # 1. Create File Object for Python
        self.pty_file = os.fdopen(self.slave_fd, "w", buffering=1)
        
        # 2. Replace Python Objects
        sys.stdout = self.pty_file
        sys.stderr = self.pty_file

        # 3. Replace OS Descriptors
        os.dup2(self.slave_fd, 1)
        os.dup2(self.slave_fd, 2)
        
        self.read_fd = self.master_fd
        self.reader_thread = threading.Thread(target=self._reader, daemon=True)
        self.reader_thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.is_windows: return

        # 1. Restore OS Descriptors
        try:
            os.dup2(self.orig_stdout_fd, 1)
            os.dup2(self.orig_stderr_fd, 2)
            os.close(self.orig_stdout_fd)
            os.close(self.orig_stderr_fd)
        except Exception: pass

        # 2. Restore Python Objects
        sys.stdout = self.saved_stdout
        sys.stderr = self.saved_stderr

        # 3. CRITICAL: Detach any Loggers pointing to our dying PTY
        root_logger = logging.getLogger()
        for h in root_logger.handlers[:]:
            if getattr(h, 'stream', None) == self.pty_file:
                root_logger.removeHandler(h)
        
        # Also check whisper specific loggers just in case
        for name in logging.root.manager.loggerDict:
            logger = logging.getLogger(name)
            for h in logger.handlers[:]:
                if getattr(h, 'stream', None) == self.pty_file:
                    logger.removeHandler(h)

        # 4. Now safe to close PTY
        self.stop_event.set()
        try: self.pty_file.close() 
        except: pass
        try: os.close(self.master_fd) 
        except: pass

        if self.reader_thread:
            self.reader_thread.join(timeout=1.0)

    def _reader(self):
        buffer = ""
        while not self.stop_event.is_set():
            try:
                data = os.read(self.read_fd, 4096)
                if not data: break 
                text = data.decode(errors="replace")
                buffer += text
                while "\n" in buffer or "\r" in buffer:
                    if "\n" in buffer: line, buffer = buffer.split("\n", 1)
                    else: line, buffer = buffer.split("\r", 1)
                    self._parse_line(line)
            except (OSError, IOError): 
                import time; time.sleep(0.01)
                continue
            except Exception: break

    def _parse_line(self, line):
        clean = line.strip()
        if not clean: return
        
        match = re.search(r"(\d+(?:\.\d+)?)\s*%", clean)
        if match:
            try:
                self.progress.update(self.task_id, completed=float(match.group(1)))
                return
            except: pass

        lower = clean.lower()
        if any(x in lower for x in ["performing voice", "model was trained", "bad things", "please wait", "symlinks"]): return

        if "error" in lower or "traceback" in lower:
            self.ui_console.print(f"[red]{clean}[/red]")

def load_subs_safely(file_path):
    encodings = ['utf-8', 'windows-1252', 'latin-1', 'utf-16']
    for enc in encodings:
        try: return pysubs2.load(str(file_path), encoding=enc)
        except: continue
    raise Exception(f"Could not decode subtitle file {file_path}.")

def make_ui_console():
    if os.name == 'posix':
        try: return Console(file=open("/dev/tty", "w", buffering=1), force_terminal=True)
        except: pass
    return Console(file=sys.__stdout__, force_terminal=True)