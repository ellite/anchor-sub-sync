import shutil
import subprocess
from collections import Counter
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

def get_video_fps(video_path: Path):
    """Frame rate of the first video stream (ffprobe avg_frame_rate, e.g. 24000/1001 -> 23.976...), or None."""
    if shutil.which("ffprobe") is None: return None
    try:
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=avg_frame_rate", "-of",
               "default=noprint_wrappers=1:nokey=1", str(Path(video_path).resolve())]
        raw = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip().splitlines()[0]
        num, _, den = raw.partition("/")
        fps = float(num) / float(den or 1)
        return fps if fps > 0 else None
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

SAMPLE_RATE = 16000
SAMPLE_POINTS = (0.15, 0.50, 0.85)   # where in the file the language is sampled (fraction of its length)
SAMPLE_SECONDS = 30                  # Whisper's detection window
MIN_SAMPLED_SECONDS = 120            # shorter audio: one detection at the start is enough
TIE_BREAK_POINT = 0.50               # when all samples disagree, the middle of the file is the likeliest to hold speech


def detect_language_by_sampling(model, audio, warn=None):
    """Audio language from three 30 s samples (15 %, 50 %, 85 %) instead of only the first 30 s, majority vote.

    Whisper detects from the first 30 s it is given, so a silent logo or a long musical intro can decide the language of the whole
    file. `model` is a whisperx pipeline (`model.detect_language(audio_slice)` returns a code: one encoder pass, no VAD, no decoding).
    A sample that fails is reported through `warn(message)` and skipped. When all samples differ the middle one wins. Returns None for
    audio under 2 minutes or when no sample gave an answer; the caller then lets Whisper detect on its own.
    """
    total = len(audio)
    window = SAMPLE_SECONDS * SAMPLE_RATE
    if total < MIN_SAMPLED_SECONDS * SAMPLE_RATE:
        return None
    guesses = []
    for point in SAMPLE_POINTS:
        start = min(int(point * total), total - window)    # the last sample is a full window, not a truncated tail
        try:
            code = model.detect_language(audio[start:start + window])
        except Exception as e:
            if warn:
                warn(f"Language sample at {int(point * 100)}% failed: {e}")
            continue
        if code and code != "unknown":
            guesses.append((point, code))
    if not guesses:
        return None
    (top, count), = Counter(code for _, code in guesses).most_common(1)
    if count == 1 and len(guesses) > 1:
        return next((code for point, code in guesses if point == TIE_BREAK_POINT), guesses[0][1])
    return top
