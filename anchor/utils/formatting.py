import re
from rich.console import Console

console = Console()

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