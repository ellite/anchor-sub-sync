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
STRAY_MATCH_SEC = 0.5        # reference mode: matched words implying starts this close agree
WEAK_DRIFT_TOL_SEC = 5.0      # reference mode: max distance of a weak match from the strong anchors' drift curve
STRONG_MATCH_WORDS = 4       # reference mode: a cue with at least this many matched words ...
STRONG_MATCH_RATIO = 0.6     # ... covering this share of its words skips the drift outlier filter
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

def anchor_exactly(anchors):
    """Reference mode: an anchor takes the reference time as measured. There is nothing to average,
    because the reference timestamps are exact (unlike Whisper's, which wobble by a few tens of ms)."""
    for a in anchors:
        a['final_start'] = a['raw_match_time']
    return anchors

def enforce_strict_spacing(subs, gap_ms=GAP_MS, min_duration_ms=MIN_DURATION_MS):
    console.print("[dim]   🧹 Running Zipper (Overlap Cleanup)...[/dim]")
    subs.sort()
    fix_count = 0
    
    for i in range(1, len(subs)):
        prev = subs[i-1]
        curr = subs[i]
        required_start = prev.end + gap_ms
        
        if required_start > curr.start:
            new_prev_end = curr.start - gap_ms
            prev_duration = new_prev_end - prev.start
            
            if prev_duration < min_duration_ms:
                prev.end = prev.start + min_duration_ms
                curr.start = prev.end + gap_ms
                curr_dur = curr.end - curr.start
                curr.end = curr.start + curr_dur
            else:
                prev.end = new_prev_end
            fix_count += 1
            
    console.print(f"[dim]      ➡️ 🔧 Resolved {fix_count} overlaps.[/dim]")
    return subs

class GlobalAligner:
    def __init__(self, original_subs, whisper_data, reference=False):
        """
        reference=False: `whisper_data` is Whisper output (word times estimate speech onset).
        reference=True:  `whisper_data` is a trusted subtitle's events ({'text','start','end'}).
                         Its cue boundaries are exact, so an anchored cue inherits the reference
                         start (and end, when its last word closes the reference cue) untouched: no
                         smoothing, no zipper. Only cues without a match are interpolated.
        """
        self.subs = original_subs
        self.whisper = whisper_data
        self.reference = reference
        self.anchor_count = 0  # valid anchors found by the last run()

    def _match_start(self, matches):
        """Start time of a subtitle cue from its matched words, as (position in cue, token) pairs."""
        if not self.reference:
            return matches[0][1]['start']

        # Reference cue boundaries are exact: if a match sits at the head of its reference cue,
        # use the cue start; otherwise back off by the words that precede it in our own cue.
        def estimate(pos, tok):
            if tok['pos'] <= pos:
                return tok['ev_start']
            return tok['start'] - pos * tok['rate']

        estimates = [estimate(pos, tok) for pos, tok in matches]

        def cluster_end(i):
            """Last index of the run of consecutive estimates agreeing within STRAY_MATCH_SEC."""
            while i + 1 < len(estimates) and abs(estimates[i + 1] - estimates[i]) <= STRAY_MATCH_SEC:
                i += 1
            return i

        # A lone common word ("the", "a") can pair with an earlier occurrence in the previous
        # reference cue. Skip a leading match that disagrees with a cluster of two or more later
        # ones. A cue that spans two reference cues still starts at its first cluster.
        i = 0
        while True:
            end = cluster_end(i)
            if end == i and end + 1 < len(estimates) and cluster_end(end + 1) > end + 1:
                i += 1
            else:
                break
        return estimates[i]

    @staticmethod
    def _is_strong(matched, cue_words):
        """A cue is confidently placed when enough of its words were found, in order, in the reference."""
        return matched >= STRONG_MATCH_WORDS and matched >= STRONG_MATCH_RATIO * cue_words

    def _match_end(self, matches, cue_words):
        """Reference mode: the reference cue end when our cue's last matched word is also the last
        word of its reference cue (same number of words still to come), else None (keep duration)."""
        if not self.reference:
            return None
        pos, tok = matches[-1]
        if cue_words - 1 - pos == tok['n'] - 1 - tok['pos']:
            return tok['ev_end']
        return None

    def _tokenize_subs(self):
        sub_words = []
        for idx, sub in enumerate(self.subs):
            text = clean_text(sub.text) 
            words = text.split()
            for i, w in enumerate(words):
                if w.strip():
                    sub_words.append({"word": w.strip(), "sub_idx": idx, "pos": i, "n": len(words)})
        return sub_words

    def _tokenize_whisper(self):
        whisper_words = []
        for seg in self.whisper:
            if 'words' in seg and seg['words']:
                for w in seg['words']:
                    if 'start' in w:
                        whisper_words.append({"word": clean_text(w['word']), "start": w['start'], "ev_start": seg['start'], "ev_end": seg['end'], "n": 0, "pos": 0, "rate": 0.0})
            else:
                text = clean_text(seg['text'])
                words = text.split()
                if not words: continue
                duration = seg['end'] - seg['start']
                wd = duration / len(words)
                for i, w in enumerate(words):
                    whisper_words.append({"word": w.strip(), "start": seg['start'] + i*wd, "ev_start": seg['start'], "ev_end": seg['end'], "n": len(words), "pos": i, "rate": wd})
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
        cue_words = {}
        for match in matches:
            for i in range(match.size):
                sub_token = sub_tokens[match.a + i]
                cue_words[sub_token['sub_idx']] = sub_token['n']
                sub_matches[sub_token['sub_idx']].append((sub_token['pos'], wh_tokens[match.b + i]))

        candidates = []
        for idx in range(len(self.subs)):
            if sub_matches[idx]:
                sub = self.subs[idx]
                match_start = self._match_start(sub_matches[idx])
                drift = match_start - (sub.start / 1000.0)
                candidates.append({
                    'idx': idx, 'orig_start': sub.start/1000.0, 
                    'raw_match_time': match_start, 'drift': drift,
                    'end_time': self._match_end(sub_matches[idx], cue_words[idx]),
                    'strong': self.reference and self._is_strong(len(sub_matches[idx]), cue_words[idx])
                })

        if not candidates: return None, 0

        console.print("[dim]   🔍 Applying Rolling Window Drift Filter...[/dim]")
        raw_anchors = []
        rejected_count = 0
        window_size = 10 

        # Reference mode: weak matches are judged against the drift curve of the strong anchors,
        # not against their neighbours' median, because two subtitle releases rarely differ by a
        # uniform offset (it can move by seconds from one cue to the next).
        strong = [c for c in candidates if c['strong']]
        trusted = len(strong) >= 2 and WEAK_DRIFT_TOL_SEC is not None
        if trusted:
            strong_x = [c['orig_start'] for c in strong]
            strong_drift = [c['drift'] for c in strong]
        
        for i, cand in enumerate(candidates):
            start_i = max(0, i - window_size)
            end_i = min(len(candidates), i + window_size + 1)
            neighbors = [n['drift'] for n in candidates[start_i:end_i]]
            
            # Reference mode trusts a strong match (many of the cue's words found in order): the
            # drift filter exists for false matches, and a real offset can change faster than it allows.
            if cand['strong']:
                raw_anchors.append(cand)
            elif trusted:
                if abs(cand['drift'] - np.interp(cand['orig_start'], strong_x, strong_drift)) > WEAK_DRIFT_TOL_SEC:
                    rejected_count += 1
                else:
                    raw_anchors.append(cand)
            elif abs(cand['drift'] - np.median(neighbors)) > OUTLIER_THRESHOLD_SEC:
                rejected_count += 1
            else:
                raw_anchors.append(cand)

        console.print(f"[dim]   ⚓️ Valid Anchors: {len(raw_anchors)} (Rejected {rejected_count} outliers)[/dim]")
        self.anchor_count = len(raw_anchors)
        
        anchors = anchor_exactly(raw_anchors) if self.reference else smooth_offsets_by_block(raw_anchors)
        
        console.print("[dim]   🔨 Reconstructing Timeline (Interpolation)...[/dim]")
        new_subs = pysubs2.SSAFile()
        new_subs.info = self.subs.info
        
        xp = [a['orig_start'] for a in anchors]
        fp = [a['final_start'] for a in anchors]
        anchor_by_idx = {a['idx']: a for a in anchors} if self.reference else {}
        to_ms = (lambda t: int(round(max(0, t) * 1000))) if self.reference else (lambda t: int(max(0, t) * 1000))
        
        for i, sub in enumerate(self.subs):
            orig = sub.start / 1000.0
            dur = (sub.end - sub.start) / 1000.0
            anchor = anchor_by_idx.get(i)
            
            if anchor is not None:
                new_start = anchor['final_start']
            elif len(anchors) > 0:
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
            
            new_end = new_start + dur
            if anchor is not None and anchor['end_time'] is not None and anchor['end_time'] > new_start:
                new_end = anchor['end_time']

            sub.start = to_ms(new_start)
            sub.end = to_ms(new_end)
            new_subs.append(sub)

        if self.reference:
            # Reference timings are valid as they are (the reference may use tighter gaps than our
            # 50 ms); only resolve real overlaps, never push an anchored cue.
            new_subs = enforce_strict_spacing(new_subs, gap_ms=0, min_duration_ms=0)
        else:
            new_subs = enforce_strict_spacing(new_subs)
        return new_subs, rejected_count