import os
import gc
import re
import math
import time
import copy
import torch
import logging
import subprocess
import unicodedata
import tempfile
import whisperx
import numpy as np
from difflib import SequenceMatcher
from pathlib import Path
from faster_whisper import WhisperModel
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn, 
    TimeElapsedColumn, TaskProgressColumn
)

# --- LOGGING SUPPRESSION ---
logging.getLogger("speechbrain").setLevel(logging.ERROR)
logging.getLogger("pyannote").setLevel(logging.ERROR)
logging.getLogger("whisperx").setLevel(logging.ERROR)
logging.getLogger("faster_whisper").setLevel(logging.ERROR)

from ...utils.files import select_files_interactive, get_files
from ...utils.languages import get_audio_language

# ==========================================
# ⚙️ CONFIGURATION CONSTANTS
# ==========================================
MAX_LINE_WIDTH = 42
MIN_DURATION = 0.5
MAX_DURATION = 7.0
MIN_GAP = 0.05

# Repair Settings
REPAIR_PADDING_PASS_1 = 0.5
REPAIR_PADDING_PASS_2 = 5.0
SUSPICIOUS_DURATION = 6.0
SUSPICIOUS_LENGTH = 60
MAX_MERGE_GAP = 2

# Sync / Alignment Constants
SCENE_GAP_SEC = 5.0
OUTLIER_THRESHOLD_SEC = 1.5

# ==========================================
# 🧹 UTILITIES & CLEANING
# ==========================================

def get_duration(path):
    try:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path)
        ]
        return float(subprocess.check_output(cmd).strip())
    except:
        return 0.0

def extract_audio_chunk(path, start, duration, denoise=True):
    temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    af_chain = None
    if denoise:
        af_chain = "afftdn=nf=-25,highpass=f=80,lowpass=f=12000,dynaudnorm"

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-t", f"{duration:.3f}",
        "-i", str(path),
        "-vn", "-ac", "1", "-ar", "16000",
        "-acodec", "pcm_s16le",
    ]
    if af_chain:
        cmd += ["-af", af_chain]
    cmd += [temp, "-loglevel", "error"]
    subprocess.run(cmd, check=True)
    return temp

def clean_text(t):
    if t is None: return ""
    t = unicodedata.normalize("NFC", t)
    
    # Basic replacements
    t = t.replace("â€“", "-").replace("â€”", "-").replace(">>", "").replace("<<", "")
    t = re.sub(r"^\.\.\.", "", t)
    
    # Whitespace and Brackets
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\[.*?\]", "", t)
    t = re.sub(r"\(.*?\)", "", t)

    # Nuclear check (if only punctuation remains)
    if re.fullmatch(r"^[.,?!;:\s]+$", t or ""):
        return ""

    # Fix Punctuation Spacing
    t = re.sub(r"\s+([?!.,])", r"\1", t)
    t = re.sub(r"([?!,])(?=\S)", r"\1 ", t)
    t = re.sub(r"\.(?=[^\s.])", r". ", t)
    
    # Fix Bracket Spacing
    t = re.sub(r"\s*\(\s*", " (", t)
    t = re.sub(r"\s*\[\s*", " [", t)
    t = re.sub(r"''", "'", t)

    t = re.sub(r'\s+([.,!?;:])', r'\1', t)

    # Hallucination check
    hallucinations = [
        "thank you for watching", "thanks for watching", "don't forget to subscribe",
        "please subscribe", "subtitles by", "captioned by"
    ]   
    clean_lower = t.lower().strip().replace(".", "")
    for h in hallucinations:
        if h in clean_lower and len(clean_lower) < 30:
            return ""

    return t.strip()

def normalize_for_dedupe(text):
    return re.sub(r"[^\w\s]", "", (text or "")).lower().strip()

def break_lines(t):
    """
    Balanced line breaker. 
    Splits text into two lines of roughly equal length if it exceeds MAX_LINE_WIDTH.
    """
    if len(t) <= MAX_LINE_WIDTH:
        return t
    
    words = t.split()
    if not words:
        return ""
        
    # If it's just one massive word, don't break it
    if len(words) == 1:
        return t

    # Find the split point closest to the middle relative to character count
    best_split = -1
    best_diff = float('inf')
    total_chars = len(t)
    current_chars = 0
    
    for i, w in enumerate(words[:-1]): # Can't split after the last word
        current_chars += len(w)
        # Calculate balance: abs(len(first_half) - len(second_half))
        # first_half = current_chars + i (spaces)
        # second_half = total_chars - first_half
        
        len_first = current_chars + i
        len_second = total_chars - len_first - 1 # -1 for the split space
        
        diff = abs(len_first - len_second)
        
        if diff < best_diff:
            best_diff = diff
            best_split = i
            
    # Construct the two lines
    line1 = " ".join(words[:best_split+1])
    line2 = " ".join(words[best_split+1:])
    
    return f"{line1}\n{line2}"

def format_timestamp(seconds: float):
    if seconds is None: return "00:00:00,000"
    milliseconds = round((seconds - math.floor(seconds)) * 1000)
    if milliseconds >= 1000:
        seconds += 1
        milliseconds = 0
    hours = math.floor(seconds / 3600)
    seconds %= 3600
    minutes = math.floor(seconds / 60)
    seconds %= 60
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{int(milliseconds):03d}"

def write_srt(segments, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for i, sub in enumerate(segments, start=1):
            start = format_timestamp(sub['start'])
            end = format_timestamp(sub['end'])
            text = break_lines(sub.get('text', ''))
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

# ==========================================
# ⏱️ TIMING LOGIC (Snap & Zip)
# ==========================================

def snap_segments_to_words(segments, console):
    """
    Uses WhisperX word timestamps to shrink segment boundaries to exact speech.
    """
    if not segments: return []
    snapped = []
    
    for seg in segments:
        if not seg: continue
        
        # Keep original text/structure
        new_s = dict(seg)
        
        # Check for word-level timings
        words = new_s.get("words", [])
        
        # Filter for valid words (some might lack start/end if unaligned)
        valid_words = [w for w in words if w.get("start") is not None and w.get("end") is not None]
        
        if valid_words:
            # SNAP! Update segment bounds to match first/last word
            new_s["start"] = valid_words[0]["start"]
            new_s["end"] = valid_words[-1]["end"]
        
        # Ensure types are floats
        new_s["start"] = float(new_s["start"])
        new_s["end"] = float(new_s["end"])
        
        snapped.append(new_s)
        
    return snapped

def run_zipper_fix(segments):
    """
    Resolves overlaps and ensures minimum gaps/durations.
    """
    if not segments: return []
    
    # Clean & Sort
    clean_segs = []
    for s in segments:
        txt = clean_text(s.get("text", ""))
        if txt:
            new_s = dict(s)
            new_s["text"] = txt
            clean_segs.append(new_s)
            
    clean_segs.sort(key=lambda x: x["start"])
    
    # The Zipper Loop
    processed = []
    if clean_segs:
        processed.append(clean_segs[0])
        
    for i in range(1, len(clean_segs)):
        prev = processed[-1]
        curr = clean_segs[i]
        
        required_start = prev["end"] + MIN_GAP
        
        if required_start > curr["start"]:
            # OVERLAP! Shrink prev to fit
            new_prev_end = curr["start"] - MIN_GAP
            prev_duration = new_prev_end - prev["start"]
            
            if prev_duration < MIN_DURATION:
                # If shrinking kills prev, push curr instead
                prev["end"] = prev["start"] + MIN_DURATION
                curr["start"] = prev["end"] + MIN_GAP
                if curr["end"] < curr["start"] + MIN_DURATION:
                     curr["end"] = curr["start"] + max((curr["end"] - curr["start"]), MIN_DURATION)
            else:
                prev["end"] = new_prev_end
        
        # Max Duration Clamp (Safety)
        dur = curr["end"] - curr["start"]
        if dur > MAX_DURATION:
            new_start = curr["end"] - MAX_DURATION

            # Don't violate previous segment + gap
            min_start = prev["end"] + MIN_GAP
            if new_start < min_start:
                new_start = min_start

            curr["start"] = new_start

            # Ensure minimum duration still holds (rare edge case)
            if curr["end"] < curr["start"] + MIN_DURATION:
                curr["end"] = curr["start"] + MIN_DURATION

        processed.append(curr)
        
    return processed

# ==========================================
# 🔧 MERGE-STITCHING HELPERS
# ==========================================

def word_tokens_for_overlap(text):
    return re.findall(r"[A-Za-z0-9']+", (text or "").lower())

def overlap_words(prev_text, curr_text, max_n=8):
    a = word_tokens_for_overlap(prev_text)
    b = word_tokens_for_overlap(curr_text)
    max_n = min(max_n, len(a), len(b))
    for n in range(max_n, 1, -1):
        if a[-n:] == b[:n]:
            return n
    return 0

def trim_prefix_words_preserve_whitespace(original_text, n_words):
    if n_words <= 0: return (original_text or "").strip()
    parts = (original_text or "").split()
    if not parts: return ""
    norm = [re.sub(r"[^\w']+", "", p.lower()) for p in parts]
    removed = 0
    cut_idx = 0
    for i, w in enumerate(norm):
        cut_idx = i + 1
        if w:
            removed += 1
            if removed >= n_words: break
    remaining = " ".join(parts[cut_idx:]).strip()
    return remaining

def is_fuzzy_duplicate(prev_text, curr_text, threshold=0.86):
    a = normalize_for_dedupe(prev_text)
    b = normalize_for_dedupe(curr_text)
    if not a or not b: return False
    return SequenceMatcher(None, a, b).ratio() >= threshold

def stitch_pair(prev_seg, curr_seg, time_gap_limit=2.0, max_overlap_words=10, fuzzy_dup_threshold=0.88):
    if not prev_seg or not curr_seg: return curr_seg
    
    gap = float(curr_seg["start"]) - float(prev_seg["end"])
    prev_text_raw = (prev_seg.get("text", "") or "").strip()
    curr_text_raw = (curr_seg.get("text", "") or "").strip()
    if not curr_text_raw: return None

    prev_norm = normalize_for_dedupe(prev_text_raw)
    curr_norm = normalize_for_dedupe(curr_text_raw)

    # Full Fuzzy Duplicate
    if gap < time_gap_limit and is_fuzzy_duplicate(prev_text_raw, curr_text_raw, threshold=fuzzy_dup_threshold):
        return None

    # Trailing Echo Check (Catches "park" vs "park's" typo)
    # Checks if the current line is just a fuzzy repetition of the tail end of the previous line
    if gap < time_gap_limit and len(curr_norm) >= 15 and len(prev_norm) >= len(curr_norm):
        compare_tail = prev_norm[-len(curr_norm)-5:] # Grab the end of the previous line
        if SequenceMatcher(None, curr_norm, compare_tail).ratio() >= 0.85:
            return None # It's a stutter, drop it.

    # Do not allow the complex word-splicing below to alter native dialogue.
    if prev_seg.get("_orig_id") is not None and curr_seg.get("_orig_id") is not None:
        return curr_seg

    # Complex Splicing (Only runs on AI repaired segments)
    prev_tok = word_tokens_for_overlap(prev_text_raw)
    curr_tok = word_tokens_for_overlap(curr_text_raw)

    if gap < time_gap_limit and len(prev_tok) >= 3 and len(curr_tok) >= len(prev_tok):
        if curr_tok[:len(prev_tok)] == prev_tok:
            out = dict(curr_seg)
            out["_drop_prev"] = True
            return out

    if gap < time_gap_limit and len(prev_norm) >= 10:
        if prev_norm in curr_norm and len(curr_norm) > len(prev_norm) + 10:
            out = dict(curr_seg)
            out["_drop_prev"] = True
            return out

    n = overlap_words(prev_text_raw, curr_text_raw, max_n=max_overlap_words)
    if n > 0:
        trimmed = trim_prefix_words_preserve_whitespace(curr_text_raw, n)
        trimmed = clean_text(trimmed)
        if not trimmed: return None
        out = dict(curr_seg)
        out["text"] = trimmed
        return out

    return curr_seg

def stitch_boundaries(full_segments, insert_start_idx, insert_end_idx):
    if not full_segments: return full_segments
    left_idx = insert_start_idx - 1
    if 0 <= left_idx < len(full_segments) and 0 <= insert_start_idx < len(full_segments):
        prev_seg = full_segments[left_idx]
        first_seg = full_segments[insert_start_idx]
        stitched = stitch_pair(prev_seg, first_seg)
        if stitched is None:
            del full_segments[insert_start_idx]
            insert_end_idx -= 1
        else:
            full_segments[insert_start_idx] = stitched

    next_idx = insert_end_idx + 1
    if 0 <= insert_end_idx < len(full_segments) and 0 <= next_idx < len(full_segments):
        last_seg = full_segments[insert_end_idx]
        next_seg = full_segments[next_idx]
        stitched_next = stitch_pair(last_seg, next_seg)
        if stitched_next is None:
            del full_segments[next_idx]
        else:
            full_segments[next_idx] = stitched_next
    return full_segments

def dedupe_window(segs, i0, i1, max_lookback=8, max_time=8.0):
    if not segs: return segs
    start_idx = max(0, i0 - max_lookback)
    end_idx = min(len(segs) - 1, i1 + max_lookback)
    out = segs[:start_idx]
    window = segs[start_idx:end_idx + 1]

    for seg in window:
        t = normalize_for_dedupe(seg.get("text", ""))
        if not t: continue
        keep_this = True
        replaced_prev = False
        
        for j in range(len(out) - 1, max(-1, len(out) - max_lookback - 1), -1):
            prev = out[j]
            gap = float(seg["start"]) - float(prev["end"])
            
            if gap > max_time: break
            
            is_immediate = (j == len(out) - 1)
            candidate = stitch_pair(prev, seg)
            
            if candidate is None:
                if is_immediate:
                    keep_this = False
                    break
                else:
                    pass 

            elif isinstance(candidate, dict) and candidate.get("_drop_prev", False):
                if is_immediate:
                    candidate = dict(candidate)
                    candidate.pop("_drop_prev", None)
                    candidate["start"] = min(float(candidate["start"]), float(prev["start"]))
                    out[j] = candidate
                    keep_this = False
                    replaced_prev = True
                    break
                else:
                    pass

            elif candidate is not seg:
                if is_immediate:
                    seg = candidate

            if is_fuzzy_duplicate(prev.get("text", ""), seg.get("text", ""), threshold=0.88):
                if is_immediate:
                    keep_this = False
                    break

        if replaced_prev: continue
        if keep_this: out.append(seg)
    
    out.extend(segs[end_idx + 1:])
    return out

def cleanup_redundancies(segments):
    if not segments: return []
    cleaned = []
    for seg in segments:
        if not cleaned:
            cleaned.append(seg)
            continue
        prev = cleaned[-1]
        stitched = stitch_pair(prev, seg)
        if stitched is None: continue
        if isinstance(stitched, dict) and stitched.pop("_drop_prev", False):
            cleaned[-1] = stitched
            continue
        cleaned.append(stitched)
    return cleaned

def filter_low_speech_confidence(raw_segments, no_speech_prob=0.6, min_avg_logprob=-1.0):
    out = []
    for s in raw_segments:
        ns = s.get("no_speech_prob", None)
        lp = s.get("avg_logprob", None)

        if ns is not None and ns >= no_speech_prob:
            continue
        if lp is not None and lp <= min_avg_logprob:
            continue

        out.append(s)
    return out

# ==========================================
# 🛠️ REPAIR & DETECTION LOGIC
# ==========================================

def is_suspicious(seg, index=None, segments=None):
    start = float(seg["start"])
    end = float(seg["end"])
    text = (seg.get("text", "") or "").strip()
    duration = end - start
    words = re.findall(r"\w+", text)
    end_punct = sum(text.count(x) for x in ".?!")
    
    if len(words) >= 14 and end_punct == 0: return True
    if duration > SUSPICIOUS_DURATION or len(text) > SUSPICIOUS_LENGTH:
        if text.count(".") + text.count("?") + text.count("!") < 1: return True
    if text and text[0].islower() and duration > 2.0:
        if end_punct == 0 and (len(words) >= 10 or len(text) > 50 or (len(text) / max(duration, 0.001)) > 24):
            return True
    if duration > 1.5 and (len(text) / duration) > 35 and sum(text.count(x) for x in ".?!") == 0:
        return True
    if end_punct == 0 and 2.0 <= duration <= 5.0 and 4 <= len(words) <= 12:
        lengths = [len(w) for w in words]
        avg_len = sum(lengths) / max(len(lengths), 1)
        short_ratio = sum(1 for L in lengths if L <= 3) / max(len(lengths), 1)
        cps = len(text) / max(duration, 0.001)
        if avg_len <= 4.5 and short_ratio >= 0.5 and cps <= 18.0:
            return True

    char_len = len(re.sub(r"\s+", "", text))
    wps = len(words) / max(duration, 0.001)
    cps = len(text) / max(duration, 0.001)

    # Very long duration but tiny amount of content
    if duration >= 5.5 and (len(words) <= 6 or char_len <= 22):
        return True
        
    # Extremely long duration with low word density (Catching Whisper "hangs")
    if duration >= 7.0 and wps < 1.0:
        return True

    # Generally "too slow": low speaking density
    if duration >= 4.5 and wps <= 0.9 and cps <= 6.0:
        return True     

    if index is not None and segments is not None:
        def is_flat(d):
            return d >= 1.0 and abs(d - round(d)) <= 0.05

        if is_flat(duration):
            # Check previous
            if index > 0:
                p_dur = float(segments[index-1]["end"]) - float(segments[index-1]["start"])
                if is_flat(p_dur): return True
            # Check next
            if index < len(segments) - 1:
                n_dur = float(segments[index+1]["end"]) - float(segments[index+1]["start"])
                if is_flat(n_dur): return True
       
    return False

def merge_suspicious_zones(indices):
    if not indices: return []
    zones = []
    current_start = indices[0]
    current_end = indices[0]
    for i in range(1, len(indices)):
        idx = indices[i]
        if idx - current_end <= MAX_MERGE_GAP + 1:
            current_end = idx
        else:
            zones.append((current_start, current_end))
            current_start = idx
            current_end = idx
    zones.append((current_start, current_end))
    return zones

def zone_quality_score(segs):
    if not segs: return -1e9
    score = 0.0
    for s in segs:
        text = (s.get("text", "") or "").strip()
        dur = max(0.001, float(s["end"]) - float(s["start"]))
        words = re.findall(r"\w+", text)
        end_punct = sum(text.count(x) for x in ".?!")
        cps = len(text) / dur

        # Penalize very slow segments with little content, as they are likely hallucinations or mis-segmented zones.
        wps = len(words) / dur
        char_len = len(re.sub(r"\s+", "", text))  # ignore spaces

        if dur >= 5.5 and (len(words) <= 6 or char_len <= 22):
            score -= 2.0
            
        # Heavy penalty for dragging segments
        if dur >= 7.0 and wps < 1.0:
            score -= 2.5
            
        if dur >= 4.5 and wps <= 0.9 and cps <= 6.0:
            score -= 1.2

        if dur >= 1.0 and abs(dur - round(dur)) <= 0.05:
            score -= 1.5    

        score += 0.7 * end_punct
        if len(words) >= 14 and end_punct == 0: score -= 2.5
        if len(text) > 120: score -= 3.0
        elif len(text) > 90: score -= 1.5
        elif len(text) > 60: score -= 0.8
        if cps > 28: score -= 1.2
        elif cps > 22: score -= 0.6
        if 0.7 <= dur <= 4.0: score += 0.2
        has_upper = any(ch.isupper() for ch in text)
        avg_word_len = (sum(len(w) for w in words) / len(words)) if words else 0.0
        if (end_punct == 0 and not has_upper and 1.5 <= dur <= 5.0 and 
            4 <= len(words) <= 12 and cps <= 18.0 and avg_word_len <= 4.5):
            score -= 0.6
    return score


def repair_zone_best(model, device, compute_type, audio_path, zone_segments, repair_padding, language):
    core_start = float(zone_segments[0]["start"])
    core_end = float(zone_segments[-1]["end"])
    start = max(0.0, core_start - repair_padding)
    end = core_end + repair_padding
    duration = end - start
    temp_wav = extract_audio_chunk(audio_path, start, duration)
    
    try:
        # Optimized Attempts: Standard -> Context -> Force Listen
        attempts = [
            dict(beam_size=10, vad_filter=True, condition_on_previous_text=False, temperature=0.0),
            dict(beam_size=10, vad_filter=True, condition_on_previous_text=True, temperature=0.0),
            dict(beam_size=10, vad_filter=False, condition_on_previous_text=False, temperature=0.0),
        ]
        
        best = None
        best_score = -1e9
        
        # Re-using the model instance is faster than reloading it, 
        # but clear internal state if possible. 
        # It uses 'transcribe' which resets state, passing 'm' is fine.
        m = WhisperModel(model, device=device, compute_type=compute_type)
        
        for i, cfg in enumerate(attempts):
            try:
                segs, _ = m.transcribe(
                    temp_wav, language=language, task="transcribe", word_timestamps=True, **cfg
                )
                
                repaired = []
                for s in segs:
                    txt = clean_text(s.text)
                    if not txt:
                        continue

                    seg = {
                        "start": start + float(s.start),
                        "end": start + float(s.end),
                        "text": txt,
                    }
                    # Strict clipping to avoid "bleeding" into neighbors
                    if seg["end"] <= core_start: continue
                    if seg["start"] >= core_end: continue
                    
                    seg["text"] = clean_text(seg["text"])
                    if seg["text"]: repaired.append(seg)
                
                sc = zone_quality_score(repaired)
                
                # Always keep the best one found so far
                if sc > best_score:
                    best_score = sc
                    best = repaired

                # ⏱️ EARLY EXIT: If we found a "Good Enough" repair, stop trying!
                # A score > 1.5 usually means valid punctuation and good density.
                #if best_score > 1.5:
                #    break
                    
            except Exception as e:
                # If one config fails, try the next
                continue
                
        # Cleanup
        del m
        gc.collect()
        if device == "cuda": torch.cuda.empty_cache()
        
        return best if best is not None else zone_segments
    finally:
        if os.path.exists(temp_wav): os.remove(temp_wav)

# ==========================================
# ⏱️ GLOBAL ALIGNER
# ==========================================

class GlobalAligner:
    """
    Treats the transcribed text as a 'Reference' and the WhisperX words as 'Truth'.
    Uses global drift correction and block smoothing.
    """
    def __init__(self, segments, whisper_data, console=None):
        self.subs = segments 
        self.whisper = whisper_data
        self.console = console

    def _tokenize_subs(self):
        sub_words = []
        for idx, sub in enumerate(self.subs):
            text = clean_text(sub.get("text", ""))
            words = text.split()
            for i, w in enumerate(words):
                if w.strip():
                    sub_words.append({"word": w.strip(), "sub_idx": idx})
        return sub_words

    def _tokenize_whisper(self):
        whisper_words = []
        for seg in self.whisper:
            if 'words' in seg and seg['words']:
                for w in seg['words']:
                    if 'start' in w:
                        whisper_words.append({"word": clean_text(w['word']), "start": w['start']})
            else:
                text = clean_text(seg.get('text', ""))
                words = text.split()
                if not words: continue
                duration = seg['end'] - seg['start']
                wd = duration / len(words)
                for i, w in enumerate(words):
                    whisper_words.append({"word": w.strip(), "start": seg['start'] + i*wd})
        return whisper_words

    def smooth_offsets_by_block(self, anchors):
        if not anchors: return []
        
        scenes = []
        current_scene = [anchors[0]]
        
        for i in range(1, len(anchors)):
            prev = anchors[i-1]
            curr = anchors[i]
            
            if (curr['orig_start'] - prev['orig_start']) > SCENE_GAP_SEC:
                scenes.append(current_scene)
                current_scene = []
            current_scene.append(curr)
        scenes.append(current_scene)
        
        smoothed = []
        for scene in scenes:
            drifts = [(a['raw_match_time'] - a['orig_start']) for a in scene]
            avg_drift = np.median(drifts)
            for a in scene:
                a['final_start'] = a['orig_start'] + avg_drift
                smoothed.append(a)
        return smoothed

    def enforce_strict_spacing(self, subs):
        subs.sort(key=lambda x: x["start"])
        fix_count = 0
        min_dur_sec = MIN_DURATION 
        gap_sec = MIN_GAP 

        processed = []
        if subs: processed.append(subs[0])

        for i in range(1, len(subs)):
            prev = processed[-1]
            curr = subs[i]
            
            required_start = prev["end"] + gap_sec
            
            if required_start > curr["start"]:
                new_prev_end = curr["start"] - gap_sec
                prev_duration = new_prev_end - prev["start"]
                
                if prev_duration < min_dur_sec:
                    prev["end"] = prev["start"] + min_dur_sec
                    curr["start"] = prev["end"] + gap_sec
                    curr_dur = curr["end"] - curr["start"]
                    if curr["end"] < curr["start"] + min_dur_sec:
                         curr["end"] = curr["start"] + max(curr_dur, min_dur_sec)
                else:
                    prev["end"] = new_prev_end
                fix_count += 1
            
            # Duration clamp - align to END time
            if (curr["end"] - curr["start"]) > MAX_DURATION:
                curr["start"] = curr["end"] - MAX_DURATION

            processed.append(curr)
        
        return processed

    def run(self):
        sub_tokens = self._tokenize_subs()
        wh_tokens = self._tokenize_whisper()
        
        sub_strs = [x['word'] for x in sub_tokens]
        wh_strs = [x['word'] for x in wh_tokens]

        matcher = SequenceMatcher(None, sub_strs, wh_strs, autojunk=False)
        matches = matcher.get_matching_blocks()
        
        sub_matches = {i: [] for i in range(len(self.subs))}
        for match in matches:
            for i in range(match.size):
                sub_token = sub_tokens[match.a + i]
                wh_time = wh_tokens[match.b + i]['start']
                sub_matches[sub_token['sub_idx']].append(wh_time)

        candidates = []
        for idx in range(len(self.subs)):
            if sub_matches[idx]:
                sub = self.subs[idx]
                match_start = sub_matches[idx][0]
                drift = match_start - sub['start']
                candidates.append({
                    'idx': idx, 'orig_start': sub['start'], 
                    'raw_match_time': match_start, 'drift': drift
                })

        if not candidates: return self.subs

        raw_anchors = []
        window_size = 10 
        
        for i, cand in enumerate(candidates):
            start_i = max(0, i - window_size)
            end_i = min(len(candidates), i + window_size + 1)
            neighbors = [n['drift'] for n in candidates[start_i:end_i]]
            
            if abs(cand['drift'] - np.median(neighbors)) <= OUTLIER_THRESHOLD_SEC:
                raw_anchors.append(cand)

        anchors = self.smooth_offsets_by_block(raw_anchors)
        
        new_subs = []
        xp = [a['orig_start'] for a in anchors]
        fp = [a['final_start'] for a in anchors]
        
        for i, sub in enumerate(self.subs):
            orig = sub['start']
            dur = sub['end'] - sub['start']
            new_sub = dict(sub)
            
            if len(anchors) > 0:
                if i < anchors[0]['idx']:
                    shift = anchors[0]['final_start'] - anchors[0]['orig_start']
                    new_start = orig + shift
                elif i > anchors[-1]['idx']:
                    shift = anchors[-1]['final_start'] - anchors[-1]['orig_start']
                    new_start = orig + shift
                else:
                    new_start = np.interp(orig, xp, fp)
            else:
                new_start = orig
            
            new_sub['start'] = max(0.0, new_start)
            new_sub['end'] = max(0.0, new_start + dur)
            new_subs.append(new_sub)

        return self.enforce_strict_spacing(new_subs)        

# ==========================================
# 🚀  MAIN RUNNER
# ==========================================

def get_progress(console):
    """Returns a consistent Progress bar instance matching Audio Sync style."""
    return Progress(
        SpinnerColumn("dots"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True
    )

def run_transcription(args, device, model_size, compute_type, console: Console):
    console.print("\n[bold green]📝  Video/Audio Transcription[/bold green]")
    
    # Support multiple selected files: build list of input paths
    inputs = []
    if args.video:
        inputs = [Path(args.video)]
    else:
        selected = select_files_interactive(
            get_files({".mp4", ".mkv", ".avi", ".mov", ".mp3", ".wav", ".flac"}), 
            header_lines=["[dim]Select media file to transcribe:[/dim]"]
        )
        if not selected: return
        inputs = [Path(p) for p in selected]

    for input_path in inputs:
        if not input_path.exists():
            console.print(f"[bold red]❌ File not found:[/bold red] {input_path}")
            continue

        # --- Language Pre-Check ---
        detected_lang = None
        if hasattr(args, 'language') and args.language:
            detected_lang = args.language
            console.print(f"\n   [dim]🌍 Language (Forced): {detected_lang.upper()}[/dim]")
        else:
            try:
                detected_lang = get_audio_language(input_path)
                if detected_lang == "unknown": 
                    detected_lang = None
                else:
                    console.print(f"\n   [dim]🌍 Language (Detected): {detected_lang.upper()}[/dim]")
            except: 
                detected_lang = None

        lang_suffix = detected_lang if detected_lang else "auto"
        
        if args.overwrite:
            output_path = input_path.with_name(f"{input_path.stem}.{lang_suffix}.srt")
        else:
            output_path = input_path.with_name(f"{input_path.stem}.{lang_suffix}.ai.srt")

        console.print(f"   📥 Input:  [cyan]{input_path.name}[/cyan]")
        console.print(f"   💾 Output: [cyan]{output_path.name}[/cyan]")
        console.print(f"   🧠 Model:  [dim]{model_size}[/dim]")

        start_time = time.time()

        try:
            # --- TRANSCRIBE ---
            with get_progress(console) as p:
                p.add_task("Step 1/5: Transcribing...", total=None)
                
                model = WhisperModel(model_size, device=device, compute_type=compute_type)
                segs, info = model.transcribe(
                    str(input_path), beam_size=5, vad_filter=False, 
                    condition_on_previous_text=False, language=detected_lang
                )
                raw_segments = []
                for i, s in enumerate(segs):
                    raw_segments.append({
                        "start": float(s.start),
                        "end": float(s.end),
                        "text": s.text,
                        "no_speech_prob": getattr(s, "no_speech_prob", None),
                        "avg_logprob": getattr(s, "avg_logprob", None),
                        "_orig_id": i
                    })

                # Drop likely hallucinations / music
                filtered = []
                for seg in raw_segments:
                    ns = seg.get("no_speech_prob")
                    lp = seg.get("avg_logprob")

                    if ns is not None and ns >= 0.6:
                        continue
                    if lp is not None and lp <= -1.0:
                        continue

                    filtered.append(seg)

                raw_segments = filtered
                actual_lang = info.language
                
                del model
                gc.collect()
                if device == "cuda": torch.cuda.empty_cache()
            
            console.print(f"[dim]📝 Transcription complete.[/dim]")

            # --- REPAIR PASS 1 ---
            with get_progress(console) as p:
                task_repair = p.add_task("Step 2/5: Scanning Zones...", total=None)
                
                suspicious_indices = [i for i, seg in enumerate(raw_segments) if is_suspicious(seg, i, raw_segments)]
                repaired_count = 0
                repaired_pass_1 = copy.deepcopy(raw_segments)
                
                if suspicious_indices:
                    zones = merge_suspicious_zones(suspicious_indices)
                    p.update(task_repair, total=len(zones), description=f"Step 2/5: Repairing (Pass 1) - {len(zones)} Zones")
                    
                    for z_start, z_end in reversed(zones):
                        bad_slice = repaired_pass_1[z_start: z_end + 1]
                        
                        original_clean = []
                        for s in bad_slice:
                            txt = clean_text(s.get("text", ""))
                            if txt:
                                ss = dict(s)
                                ss["text"] = txt
                                original_clean.append(ss)
                        orig_score = zone_quality_score(original_clean)
                        
                        repaired_block = repair_zone_best(
                            model_size, device, compute_type, input_path, bad_slice, REPAIR_PADDING_PASS_1, actual_lang
                        )
                        rep_score = zone_quality_score(repaired_block)

                        if rep_score > orig_score + 0.3:
                            repaired_count += 1
                            final_block = repaired_block
                        else:
                            final_block = original_clean
                        
                        repaired_pass_1[z_start: z_end + 1] = final_block
                        insert_start = z_start
                        insert_end = z_start + len(final_block) - 1
                        repaired_pass_1 = stitch_boundaries(repaired_pass_1, insert_start, insert_end)
                        repaired_pass_1 = dedupe_window(repaired_pass_1, insert_start, insert_end)
                        
                        p.advance(task_repair)
                    
                    deduped_1 = cleanup_redundancies(repaired_pass_1)
                    console.print(f"[dim]🔧 Repaired Pass 1.[/dim] (Fixed {repaired_count} of {len(zones)} zones)")
                else:
                    deduped_1 = raw_segments
                    console.print(f"[dim]🔧 Repaired Pass 1[/dim] (No issues found)")  

            # --- REPAIR PASS 2 ---
            with get_progress(console) as p:
                task_repair_2 = p.add_task("Step 3/5: Scanning Zones...", total=None)
                
                suspicious_indices_2 = [i for i, seg in enumerate(deduped_1) if is_suspicious(seg, i, deduped_1)]
                repaired_pass_2 = copy.deepcopy(deduped_1)
                repaired_count = 0
                
                if suspicious_indices_2:
                    zones_2 = merge_suspicious_zones(suspicious_indices_2)
                    p.update(task_repair_2, total=len(zones_2), description=f"Step 3/5: Repairing (Pass 2) - {len(zones_2)} Zones")
                    
                    for z_start, z_end in reversed(zones_2):
                        bad_slice = repaired_pass_2[z_start: z_end + 1]
                        
                        original_clean = []
                        for s in bad_slice:
                            txt = clean_text(s.get("text", ""))
                            if txt:
                                ss = dict(s)
                                ss["text"] = txt
                                original_clean.append(ss)
                        orig_score = zone_quality_score(original_clean)
                        
                        repaired_block = repair_zone_best(
                            model_size, device, compute_type, input_path, bad_slice, REPAIR_PADDING_PASS_2, actual_lang
                        )
                        rep_score = zone_quality_score(repaired_block)

                        if rep_score > orig_score + 0.3:
                            repaired_count += 1
                            final_block = repaired_block
                        else:
                            final_block = original_clean
                        
                        repaired_pass_2[z_start: z_end + 1] = final_block
                        insert_start = z_start
                        insert_end = z_start + len(final_block) - 1
                        repaired_pass_2 = stitch_boundaries(repaired_pass_2, insert_start, insert_end)
                        repaired_pass_2 = dedupe_window(repaired_pass_2, insert_start, insert_end)
                        
                        p.advance(task_repair_2)
                    
                    deduped_2 = cleanup_redundancies(repaired_pass_2)
                    console.print(f"[dim]🔧 Repaired Pass 2.[/dim] (Fixed {repaired_count} of {len(zones_2)} zones)")
                else:
                    deduped_2 = deduped_1
                    console.print(f"[dim]🔧 Repaired Pass 2[/dim] (No issues found)") 

            # --- ALIGNMENT ---
            with get_progress(console) as p:
                p.add_task("Step 4/5: Aligning (WhisperX)...", total=None)
                
                clean_segments = []
                for seg in deduped_2:
                    cleaned_text = clean_text(seg.get("text", ""))
                    if cleaned_text:
                        out = dict(seg)
                        out["text"] = cleaned_text
                        clean_segments.append(out)
                
                model_a, metadata = whisperx.load_align_model(language_code=actual_lang, device=device)
                audio = whisperx.load_audio(str(input_path))
                result = whisperx.align(clean_segments, model_a, metadata, audio, device, return_char_alignments=False)
                aligned_segments = result["segments"]
                
                del model_a
                gc.collect()
                if device == "cuda": torch.cuda.empty_cache()
            
            console.print("[dim]🔬 Phoneme alignment complete.[/dim]")

            # --- SAVE ---
            with get_progress(console) as p:
                p.add_task("Step 5/5: Global Alignment & Save...", total=None)
                
                # Treat 'aligned_segments' (which has text + raw timing + word timings)
                # as the "Truth" source for word timings, but use GlobalAligner to
                # smooth the block pacing.
                aligner = GlobalAligner(aligned_segments, aligned_segments, console)
                final_segments = aligner.run()
                
                write_srt(final_segments, output_path)
            
            console.print("[dim]💾 Cleaned up and Saved.[/dim]")

        except Exception as e:
            console.print(f"\n[bold red]❌ Transcription failed for {input_path.name}:[/bold red] {e}")
            import traceback
            traceback.print_exc()
            continue

        duration = time.time() - start_time
        console.print(f"\n[bold green]✅ Done in {duration:.1f}s![/bold green] Saved to: [underline]{output_path.name}[/underline]")