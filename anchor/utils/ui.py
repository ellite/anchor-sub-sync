import os
import sys
import re
import threading
import logging
from rich.console import Console

# Cross-Platform Imports
if os.name == 'posix':
    import pty
    import fcntl
else:
    pty = None
    fcntl = None
    termios = None

console = Console()

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
        
        # Create File Object for Python
        self.pty_file = os.fdopen(self.slave_fd, "w", buffering=1)
        
        # Replace Python Objects
        sys.stdout = self.pty_file
        sys.stderr = self.pty_file

        # Replace OS Descriptors
        os.dup2(self.slave_fd, 1)
        os.dup2(self.slave_fd, 2)
        
        self.read_fd = self.master_fd
        self.reader_thread = threading.Thread(target=self._reader, daemon=True)
        self.reader_thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.is_windows: return

        # Restore OS Descriptors
        try:
            os.dup2(self.orig_stdout_fd, 1)
            os.dup2(self.orig_stderr_fd, 2)
            os.close(self.orig_stdout_fd)
            os.close(self.orig_stderr_fd)
        except Exception: pass

        # Restore Python Objects
        sys.stdout = self.saved_stdout
        sys.stderr = self.saved_stderr

        # CRITICAL: Detach any Loggers pointing to our dying PTY
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

        # Now safe to close PTY
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

def make_ui_console():
    if os.name == 'posix':
        try: return Console(file=open("/dev/tty", "w", buffering=1), force_terminal=True)
        except: pass
    return Console(file=sys.__stdout__, force_terminal=True)