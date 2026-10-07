"""Builders for synthetic subtitles and recognised speech with a known, exact answer."""
import pysubs2

from anchor.utils import alignment


def cue_text(i, words=5):
    """Distinct words per cue, so the global matcher can only pair a cue with its own speech."""
    return " ".join(f"w{i:03d}{chr(97 + k)}x" for k in range(words))


def make_subs(cues):
    """cues: [(start_sec, end_sec, text)] -> SSAFile."""
    subs = pysubs2.SSAFile()
    for start, end, text in cues:
        subs.append(pysubs2.SSAEvent(start=round(start * 1000), end=round(end * 1000), text=text))
    return subs


def spoken(text, start, pace=0.3, only=None):
    """A recognised segment: `text` spoken from `start`, one word every `pace` s. `only` keeps just those word indices
    (the others were not heard)."""
    words, t = [], start
    for w in text.split():
        words.append({"word": w, "start": t, "end": t + pace * 0.8})
        t += pace
    if only is not None:
        words = [words[i] for i in only]
    return {"start": words[0]["start"], "end": words[-1]["end"], "text": " ".join(w["word"] for w in words), "words": words}


def timeline(n=30, spacing=4.0, first=10.0, duration=2.5, words=5):
    """n cues, `spacing` s apart. Returns [(start, end, text)]."""
    return [(first + i * spacing, first + i * spacing + duration, cue_text(i, words)) for i in range(n)]


def speech_for(cues, drift, skip=(), only=None):
    """Recognised speech for `cues` where cue i is spoken at its start + drift(start). `skip`: cue indices never spoken.
    `only`: {cue index: [word indices heard]}."""
    only = only or {}
    return [spoken(text, start + drift(start), only=only.get(i)) for i, (start, _, text) in enumerate(cues) if i not in skip]


def align(cues, speech, precise=False, reference=False):
    """Runs the real GlobalAligner. Returns (subs, aligner); `subs` events are retimed in place, so subs[i] is cue i."""
    subs = make_subs(cues)
    aligner = alignment.GlobalAligner(subs, speech, reference=reference, precise=precise)
    result, rejected = aligner.run()
    return subs, aligner


def anchor(orig, drift, strong=True):
    """An anchor dict as GlobalAligner builds it, for the smoothing functions."""
    return {"orig_start": orig, "raw_match_time": orig + drift, "strong": strong}
