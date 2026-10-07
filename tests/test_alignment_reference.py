"""Reference sync: a trusted subtitle supplies exact timestamps (utils/alignment.py with reference=True).

See AGENTS.md, 'Reference sync'. The reference's times are exact, so none of audio mode's smoothing applies.
"""
import pytest

from anchor.utils import alignment
from helpers import align, cue_text, make_subs


def ref_cue(text, start, end):
    """A reference cue as referencesync.py feeds it to the aligner: text and times, no word timestamps."""
    return {"text": text, "start": start, "end": end}


def targets_and_reference(n, shifts, spacing=4.0, dur=2.5):
    """n target cues; cue i's reference sits `shifts(i)` seconds later."""
    cues = [(10 + i * spacing, 10 + i * spacing + dur, cue_text(i)) for i in range(n)]
    ref = [ref_cue(cue_text(i), cues[i][0] + shifts(i), cues[i][0] + shifts(i) + dur) for i in range(n)]
    return cues, ref


def test_an_anchored_cue_inherits_the_exact_reference_start_to_the_millisecond():
    """Anchors take the reference time untouched (rounded, not truncated): 12.3456 s -> 12346 ms."""
    cues = [(3.0, 5.5, cue_text(0))]
    subs, _ = align(cues, [ref_cue(cue_text(0), 12.3456, 14.3456)], reference=True)
    assert subs[0].start == 12346


def test_the_cue_end_comes_from_the_reference_only_when_the_last_words_line_up():
    """_match_end: if our cue ends where the reference cue ends, use its end; if our cue has extra words, keep our duration."""
    same = cue_text(0, 5)
    longer = cue_text(1, 5) + " extra1x extra2x"
    cues = [(3.0, 5.5, same), (10.0, 12.5, longer)]
    ref = [ref_cue(same, 20.0, 23.3333), ref_cue(cue_text(1, 5), 30.0, 33.3333)]
    subs, _ = align(cues, ref, reference=True)
    assert subs[0].end == 23333                                # exact reference end
    assert subs[1].end - subs[1].start == 2500                 # unmatched trailing words: duration kept


def test_a_strong_match_is_never_rejected_for_its_drift():
    """STRONG_MATCH_WORDS / RATIO: two releases of an episode can differ by seconds from one cue to the next. A cue with
    >= 4 matched words covering >= 60% of it skips the outlier filter (it once rejected a 9-of-9 match)."""
    cues, ref = targets_and_reference(20, lambda i: 2.0 if i < 10 else 9.0)
    _, aligner = align(cues, ref, reference=True)
    assert aligner.anchor_count == 20


def test_a_weak_match_is_judged_against_the_strong_drift_curve_with_five_seconds_of_slack():
    """WEAK_DRIFT_TOL_SEC = 5.0, not the neighbours' median: it threw away correct short matches ('yeah', 'thanks')."""
    cues, ref = targets_and_reference(12, lambda i: 1.0)
    for idx, off, expected in ((3, 4.0, True), (7, 6.0, False)):
        data = list(ref)
        # cue idx now has only its first word in the reference, `off` s from the curve
        data[idx] = ref_cue(cue_text(idx, 5).split()[0], cues[idx][0] + 1.0 + off, cues[idx][0] + 1.0 + off + 0.3)
        _, aligner = align(cues, data, reference=True)
        assert (idx in aligner.anchored_idx) == expected, f"weak match {off} s from the curve"


def test_a_lone_stray_leading_word_is_ignored_when_later_words_agree_with_each_other():
    """STRAY_MATCH_SEC: a common word ('the') can pair with an earlier occurrence in the previous reference cue."""
    aligner = alignment.GlobalAligner(make_subs([]), [], reference=True)
    far = {"pos": 0, "ev_start": 10.0, "start": 10.0, "rate": 0.3}
    near = {"pos": 1, "ev_start": 20.0, "start": 20.3, "rate": 0.3}
    near2 = {"pos": 2, "ev_start": 20.0, "start": 20.6, "rate": 0.3}
    assert aligner._match_start([(0, far), (1, near), (2, near2)]) == 20.0
    # but if nothing agrees, the first estimate stands
    assert aligner._match_start([(0, far), (1, near)]) == 10.0


def test_reference_timings_are_not_pushed_apart_by_the_audio_mode_spacing_rules():
    """The reference may time cues 20 ms apart or shorter than 600 ms; anchored cues must come out exactly as the reference."""
    cues = [(3.0, 3.4, cue_text(0)), (3.42, 3.9, cue_text(1)), (10.0, 12.0, cue_text(2)), (14.0, 16.0, cue_text(3))]
    ref = [ref_cue(cue_text(0), 20.0, 20.3), ref_cue(cue_text(1), 20.32, 20.9), ref_cue(cue_text(2), 30.0, 32.0), ref_cue(cue_text(3), 34.0, 36.0)]
    subs, _ = align(cues, ref, reference=True)
    assert (subs[0].start, subs[0].end, subs[1].start) == (20000, 20300, 20320)      # 300 ms long, 20 ms gap: untouched


def test_anchors_are_exact_not_smoothed_in_reference_mode():
    cues, ref = targets_and_reference(12, lambda i: 1.0 + (0.7 if i % 2 else 0.0))
    subs, _ = align(cues, ref, reference=True)
    for i, (start, _, _) in enumerate(cues):
        assert subs[i].start == round((start + 1.0 + (0.7 if i % 2 else 0.0)) * 1000)
