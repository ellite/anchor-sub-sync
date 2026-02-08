import pysubs2
import difflib
import numpy as np
from rich.console import Console
from .formatting import clean_text

# ================= CONSTANTS =================
SCENE_GAP_SEC = 5.0          
MIN_DURATION_MS = 600        
GAP_MS = 50                  
OUTLIER_THRESHOLD_SEC = 1.5  
# =============================================

console = Console()

def smooth_offsets_by_block(anchors):
    if not anchors: return []
    console.print("[dim]   ⚖️ Applying Block Averaging (Smoothing)...[/dim]")
    
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

def enforce_strict_spacing(subs):
    console.print("[dim]   🧹 Running Zipper (Overlap Cleanup)...[/dim]")
    subs.sort()
    fix_count = 0
    
    for i in range(1, len(subs)):
        prev = subs[i-1]
        curr = subs[i]
        required_start = prev.end + GAP_MS
        
        if required_start > curr.start:
            new_prev_end = curr.start - GAP_MS
            prev_duration = new_prev_end - prev.start
            
            if prev_duration < MIN_DURATION_MS:
                prev.end = prev.start + MIN_DURATION_MS
                curr.start = prev.end + GAP_MS
                curr_dur = curr.end - curr.start
                curr.end = curr.start + curr_dur
            else:
                prev.end = new_prev_end
            fix_count += 1
            
    console.print(f"[dim]      ➡️ 🔧 Resolved {fix_count} overlaps.[/dim]")
    return subs

class GlobalAligner:
    def __init__(self, original_subs, whisper_data):
        self.subs = original_subs
        self.whisper = whisper_data
        
    def _tokenize_subs(self):
        sub_words = []
        for idx, sub in enumerate(self.subs):
            text = clean_text(sub.text) 
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
                text = clean_text(seg['text'])
                words = text.split()
                if not words: continue
                duration = seg['end'] - seg['start']
                wd = duration / len(words)
                for i, w in enumerate(words):
                    whisper_words.append({"word": w.strip(), "start": seg['start'] + i*wd})
        return whisper_words

    def run(self):
        console.print("[dim]   🧩 Tokenizing data...[/dim]")
        sub_tokens = self._tokenize_subs()
        wh_tokens = self._tokenize_whisper()
        
        sub_strs = [x['word'] for x in sub_tokens]
        wh_strs = [x['word'] for x in wh_tokens]

        console.print(f"[dim]   📐 Global Alignment ({len(sub_strs)} vs {len(wh_strs)} words)...[/dim]")
        matcher = difflib.SequenceMatcher(None, sub_strs, wh_strs, autojunk=False)
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
                drift = match_start - (sub.start / 1000.0)
                candidates.append({
                    'idx': idx, 'orig_start': sub.start/1000.0, 
                    'raw_match_time': match_start, 'drift': drift
                })

        if not candidates: return None, 0

        console.print("[dim]   🔍 Applying Rolling Window Drift Filter...[/dim]")
        raw_anchors = []
        rejected_count = 0
        window_size = 10 
        
        for i, cand in enumerate(candidates):
            start_i = max(0, i - window_size)
            end_i = min(len(candidates), i + window_size + 1)
            neighbors = [n['drift'] for n in candidates[start_i:end_i]]
            
            if abs(cand['drift'] - np.median(neighbors)) > OUTLIER_THRESHOLD_SEC:
                rejected_count += 1
            else:
                raw_anchors.append(cand)

        console.print(f"[dim]   ⚓️ Valid Anchors: {len(raw_anchors)} (Rejected {rejected_count} outliers)[/dim]")
        
        anchors = smooth_offsets_by_block(raw_anchors)
        
        console.print("[dim]   🔨 Reconstructing Timeline (Interpolation)...[/dim]")
        new_subs = pysubs2.SSAFile()
        new_subs.info = self.subs.info
        
        xp = [a['orig_start'] for a in anchors]
        fp = [a['final_start'] for a in anchors]
        
        for i, sub in enumerate(self.subs):
            orig = sub.start / 1000.0
            dur = (sub.end - sub.start) / 1000.0
            
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
            
            sub.start = int(max(0, new_start) * 1000)
            sub.end = int(max(0, new_start + dur) * 1000)
            new_subs.append(sub)

        new_subs = enforce_strict_spacing(new_subs)
        return new_subs, rejected_count