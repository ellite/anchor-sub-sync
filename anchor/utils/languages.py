import shutil
import subprocess
from langdetect import detect, LangDetectException
from pathlib import Path
from .files import open_subtitle
from .mappings import ISO_639_MAPPING

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

def get_subtitle_language(sub_path: Path) -> str:
    """
    Detects the language of a subtitle file by analyzing its text content.
    Returns the ISO 639-1 code (e.g., 'en', 'pt', 'es').
    """
    try:
        # Use safe loader
        subs = open_subtitle(sub_path)
        
        # Collect text samples
        sample_text = []
        char_count = 0
        
        for event in subs:
            text = event.plaintext.strip()
            
            # Skip empty lines, numbers, or very short generic sounds
            if not text or len(text) < 2 or text.isnumeric():
                continue
                
            sample_text.append(text)
            char_count += len(text)
            
            if char_count > 1000:
                break
        
        if not sample_text:
            return "unknown"

        full_sample = " ".join(sample_text)
        return detect(full_sample)

    except (LangDetectException, ValueError, Exception) as e:
        return "unknown"