import contextlib
import difflib
import itertools
import logging

import pysubs2
import numpy as np
from rich.console import Console
from .formatting import clean_text

# ================= CONSTANTS =================
SCENE_GAP_SEC = 5.0          
MIN_DURATION_MS = 600        
GAP_MS = 50                  
OUTLIER_THRESHOLD_SEC = 1.5
STRAY_MATCH_SEC = 0.5        # reference mode: matched words implying starts this close agree
EDGE_FIT_ANCHORS = 10        # audio mode: anchors at each end used to read the drift trend there (0 = constant shift)
EDGE_MIN_ANCHORS = 4         # fewer anchors than this give no trustworthy trend
EDGE_MIN_SPAN_SEC = 20.0     # anchor pairs closer than this give a noisy slope
BLOCK_TREND_MIN_ANCHORS = 5  # audio mode: a block needs this many anchors to follow a drift trend inside it ...
BLOCK_TREND_MIN_SPAN_SEC = 20.0  # ... and to span at least this long; shorter blocks keep a single median shift
BLOCK_TREND_MIN_PAIR_SEC = 5.0   # anchor pairs closer than this give a noisy slope
BLOCK_TREND_MAX_SLOPE = 0.1  # |drift change| per second of media inside a block
EDGE_MAX_SLOPE = 0.1         # |drift change| per second of media, i.e. 10%
EDGE_MAX_EXTRA_SEC = 6.0     # most the trend may add on top of the edge anchor's shift
SLOW_SEGMENT_SEC = 5.5       # Whisper timing filter: a segment this long ...
SLOW_SEGMENT_MAX_WPS = 0.6   # ... carrying fewer words per second than this is a timestamp collapse, not speech
FAST_SEGMENT_MIN_WORDS = 3   # a segment with at least this many words ...
FAST_SEGMENT_MAX_WPS = 8.0   # ... faster than this cannot be real speech
DENSE_WINDOW_SEC = 3.0       # more words than DENSE_MAX_WPS per second inside this window ...
DENSE_MAX_WPS = 7.5          # ... is also a collapse (Whisper squeezing a long scene into a few seconds)
AUDIO_STRONG_ANCHORS = True   # audio mode: strong matches skip the neighbour-median drift filter too
AUDIO_WEAK_DRIFT_TOL_SEC = 1.5  # audio mode: max distance of a weak match from the strong anchors' drift curve
WEAK_DRIFT_TOL_SEC = 5.0      # reference mode: max distance of a weak match from the strong anchors' drift curve
STRONG_MATCH_WORDS = 4       # reference mode: a cue with at least this many matched words ...
STRONG_MATCH_RATIO = 0.6     # ... covering this share of its words skips the drift outlier filter
# =============================================

console = Console()


@contextlib.contextmanager
def quiet_library_logs():
    """Hides the log lines libraries print while we call them.

    WhisperX logs a warning for every segment its aligner cannot time ("backtrack failed, resorting to original") and
    falls back to the segment's own timing; the sync does not need to show that. Logging is switched off only for the
    duration of the block, then restored.
    """
    previous = logging.root.manager.disable
    logging.disable(logging.WARNING)
    try:
        yield
    finally:
        logging.disable(previous)

def smooth_offsets_by_block(anchors):
    """Audio mode: gives each anchor the shift of its block ("scene": anchors less than SCENE_GAP_SEC apart).

    Whisper times wobble by a few tenths of a second from word to word, so anchors are not used one by
    one. A short block gets the median shift of its anchors. A long block (BLOCK_TREND_MIN_ANCHORS anchors
    over BLOCK_TREND_MIN_SPAN_SEC) follows a straight line through its anchors' shifts (median of pairwise
    slopes, robust to a few bad anchors), because the real drift keeps changing inside a long block and one
    median would push the start of the block late and the end early.
    """
    if not anchors: return []
    console.print("[dim]   ⚖️ Applying Block Smoothing (median / trend)...[/dim]")

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
        xs = np.array([a['orig_start'] for a in scene])
        drifts = np.array([a['raw_match_time'] - a['orig_start'] for a in scene])
        median_drift = float(np.median(drifts))

        line = None
        if len(scene) >= BLOCK_TREND_MIN_ANCHORS and xs[-1] - xs[0] >= BLOCK_TREND_MIN_SPAN_SEC:
            slopes = [(drifts[j] - drifts[i]) / (xs[j] - xs[i])
                      for i, j in itertools.combinations(range(len(scene)), 2)
                      if xs[j] - xs[i] >= BLOCK_TREND_MIN_PAIR_SEC]
            if slopes:
                slope = float(np.clip(np.median(slopes), -BLOCK_TREND_MAX_SLOPE, BLOCK_TREND_MAX_SLOPE))
                line = (slope, float(np.median(drifts - slope * xs)))

        for a in scene:
            shift = line[0] * a['orig_start'] + line[1] if line else median_drift
            a['final_start'] = a['orig_start'] + shift
            smoothed.append(a)
    return smoothed

def drop_implausible_timing(whisper_data):
    """Remove Whisper output whose timing cannot be real speech, so it never becomes an anchor.

    WhisperX sometimes collapses a long, loud scene: one segment spanning ~25 s for 6 words, followed
    by every later line squeezed into ~2 s. Those words match subtitle cues at the wrong time. Dropped:
    segments with a speaking rate below SLOW_SEGMENT_MAX_WPS (when long) or above FAST_SEGMENT_MAX_WPS,
    and words inside any DENSE_WINDOW_SEC window denser than DENSE_MAX_WPS.
    Returns (filtered_data, dropped_word_count). The input is not modified.
    """
    def n_words(seg):
        ws = seg.get('words') or []
        return len(ws) if ws else len(clean_text(seg.get('text', '')).split())

    # 1) segment-level rate rules
    kept = []
    dropped = 0
    for seg in whisper_data:
        n = n_words(seg)
        dur = max(float(seg['end']) - float(seg['start']), 1e-3)
        wps = n / dur
        if (dur >= SLOW_SEGMENT_SEC and wps < SLOW_SEGMENT_MAX_WPS) or \
           (n >= FAST_SEGMENT_MIN_WORDS and wps > FAST_SEGMENT_MAX_WPS):
            dropped += n
        else:
            kept.append(seg)

    # 2) word-level density: words inside an over-dense window
    starts = sorted(w['start'] for seg in kept for w in (seg.get('words') or []) if 'start' in w)
    limit = DENSE_MAX_WPS * DENSE_WINDOW_SEC
    dense = set()
    lo = hi = 0
    half = DENSE_WINDOW_SEC / 2
    for i, t in enumerate(starts):
        while starts[lo] < t - half: lo += 1
        while hi < len(starts) and starts[hi] <= t + half: hi += 1
        if hi - lo > limit:
            dense.add(t)
    if not dense:
        return kept, dropped

    out = []
    for seg in kept:
        words = seg.get('words') or []
        if not words:
            out.append(seg)
            continue
        good = [w for w in words if w.get('start') not in dense]
        dropped += len(words) - len(good)
        if good:
            new = dict(seg)
            new['words'] = good
            out.append(new)
    return out, dropped

def edge_drift_slope(edge_anchors):
    """Drift trend (seconds of drift per second of media) read from the anchors at one end of the
    timeline: the median slope between anchor pairs, which tolerates a few bad anchors. 0.0 when
    there is not enough spread to trust it."""
    if EDGE_FIT_ANCHORS <= 0 or len(edge_anchors) < EDGE_MIN_ANCHORS:
        return 0.0
    pts = [(a['orig_start'], a['raw_match_time'] - a['orig_start']) for a in edge_anchors]
    slopes = [(d2 - d1) / (x2 - x1) for (x1, d1), (x2, d2) in itertools.combinations(pts, 2)
              if x2 - x1 >= EDGE_MIN_SPAN_SEC]
    if not slopes:
        return 0.0
    return float(np.clip(np.median(slopes), -EDGE_MAX_SLOPE, EDGE_MAX_SLOPE))

def anchor_exactly(anchors):
    """Reference mode: an anchor takes the reference time as measured. There is nothing to average,
    because the reference timestamps are exact (unlike Whisper's, which wobble by a few tens of ms)."""
    for a in anchors:
        a['final_start'] = a['raw_match_time']
    return anchors

def enforce_strict_spacing(subs, gap_ms=GAP_MS, min_duration_ms=MIN_DURATION_MS):
    console.print("[dim]   🧹 Running Zipper (Overlap and Gap Cleanup)...[/dim]")
    subs.sort()
    overlaps = tight_gaps = 0
    
    for i in range(1, len(subs)):
        prev = subs[i-1]
        curr = subs[i]
        required_start = prev.end + gap_ms
        
        if required_start > curr.start:
            # A real overlap (the previous cue runs into this one) is not the same as two cues that merely sit
            # closer than the minimum gap, which is normal for tightly timed subtitles (30 ms gaps are common).
            if prev.end > curr.start:
                overlaps += 1
            else:
                tight_gaps += 1
            new_prev_end = curr.start - gap_ms
            prev_duration = new_prev_end - prev.start
            
            if prev_duration < min_duration_ms:
                prev.end = prev.start + min_duration_ms
                curr.start = prev.end + gap_ms
                curr_dur = curr.end - curr.start
                curr.end = curr.start + curr_dur
            else:
                prev.end = new_prev_end
            
    message = f"Resolved {overlaps} overlap{'s' if overlaps != 1 else ''}"
    if gap_ms > 0:
        message += f", trimmed {tight_gaps} cue end{'s' if tight_gaps != 1 else ''} to keep a {gap_ms} ms gap"
    console.print(f"[dim]      ➡️ 🔧 {message}.[/dim]")
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
        self.anchored_idx = set()  # indices of the cues that anchored in the last run()

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

    @staticmethod
    def _edge_extra(slope, distance):
        """Shift added past the end anchor: the trend times the distance to it, capped."""
        return float(np.clip(slope * distance, -EDGE_MAX_EXTRA_SEC, EDGE_MAX_EXTRA_SEC))

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
                    'strong': (self.reference or AUDIO_STRONG_ANCHORS) and self._is_strong(len(sub_matches[idx]), cue_words[idx])
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
        weak_tol = WEAK_DRIFT_TOL_SEC if self.reference else AUDIO_WEAK_DRIFT_TOL_SEC
        trusted = len(strong) >= 2 and weak_tol is not None
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
                if abs(cand['drift'] - np.interp(cand['orig_start'], strong_x, strong_drift)) > weak_tol:
                    rejected_count += 1
                else:
                    raw_anchors.append(cand)
            elif abs(cand['drift'] - np.median(neighbors)) > OUTLIER_THRESHOLD_SEC:
                rejected_count += 1
            else:
                raw_anchors.append(cand)

        console.print(f"[dim]   ⚓️ Valid Anchors: {len(raw_anchors)} (Rejected {rejected_count} outliers)[/dim]")
        self.anchor_count = len(raw_anchors)
        self.anchored_idx = {a['idx'] for a in raw_anchors}
        
        anchors = anchor_exactly(raw_anchors) if self.reference else smooth_offsets_by_block(raw_anchors)
        
        console.print("[dim]   🔨 Reconstructing Timeline (Interpolation)...[/dim]")
        new_subs = pysubs2.SSAFile()
        new_subs.info = self.subs.info
        
        xp = [a['orig_start'] for a in anchors]
        fp = [a['final_start'] for a in anchors]
        # Beyond the first/last anchor the shift follows the local drift trend instead of staying
        # constant. A constant shift is wrong when the drift is still changing there (audio mode only:
        # reference anchors are dense and exact).
        head_slope = tail_slope = 0.0
        if not self.reference:
            head_slope = edge_drift_slope(raw_anchors[:EDGE_FIT_ANCHORS])
            tail_slope = edge_drift_slope(raw_anchors[-EDGE_FIT_ANCHORS:])
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
                    new_start = orig + shift + self._edge_extra(head_slope, orig - anchors[0]['orig_start'])
                elif i > anchors[-1]['idx']:
                    shift = anchors[-1]['final_start'] - anchors[-1]['orig_start']
                    new_start = orig + shift + self._edge_extra(tail_slope, orig - anchors[-1]['orig_start'])
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