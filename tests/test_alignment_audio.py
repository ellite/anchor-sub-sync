"""Audio sync: how cues are matched to recognised speech and placed (utils/alignment.py, audio mode).

Every test names the choice it protects. See AGENTS.md, 'Global alignment' steps 1-7.
"""
import pytest

from anchor.utils import alignment
from helpers import align, cue_text, make_subs, spoken, speech_for, timeline

TOL = 0.05   # seconds


def starts(subs):
    return [e.start / 1000.0 for e in subs]


# ---------------------------------------------------------------- the basics

def test_a_constant_offset_is_removed():
    """The whole point: every cue moves to where its words are spoken."""
    cues = timeline(30)
    subs, _ = align(cues, speech_for(cues, lambda t: 2.5))
    for (orig, _, _), new in zip(cues, starts(subs)):
        assert new == pytest.approx(orig + 2.5, abs=TOL)


def test_matching_is_by_text_so_a_huge_offset_still_works():
    """Words are matched globally, never by time: a subtitle 90 s out must align like one 1 s out (Sync Check relies on this)."""
    cues = timeline(30)
    subs, aligner = align(cues, speech_for(cues, lambda t: 90.0))
    assert aligner.anchor_count == 30
    assert starts(subs)[10] == pytest.approx(cues[10][0] + 90.0, abs=TOL)


def test_a_drift_that_grows_through_the_file_is_followed():
    """A frame rate mismatch grows steadily; each cue must get the drift of its own place in the file, not one average."""
    cues = timeline(40)
    subs, _ = align(cues, speech_for(cues, lambda t: 1.0 + 0.02 * t))
    for (orig, _, _), new in zip(cues, starts(subs)):
        assert new == pytest.approx(orig + 1.0 + 0.02 * orig, abs=0.1)


def test_a_cue_starts_at_its_first_matched_word_not_where_the_cue_was():
    cues = timeline(10)
    speech = speech_for(cues, lambda t: 1.0)
    subs, aligner = align(cues, speech)
    assert subs[4].start / 1000.0 == pytest.approx(cues[4][0] + 1.0, abs=TOL)


# ---------------------------------------------------------------- which matches are trusted

def test_a_strong_match_is_trusted_even_when_its_drift_differs_from_its_neighbours():
    """AUDIO_STRONG_ANCHORS: two releases never differ by a uniform offset. A cue with >= 4 matched words (>= 60% of it)
    is an anchor without the neighbour-median test; that test used to reject correct cues and keep a wrong one-word match."""
    cues = timeline(20)

    def drift(t):
        return 3.5 if t == cues[10][0] else 1.0

    _, aligner = align(cues, speech_for(cues, drift))
    assert 10 in aligner.anchored_idx


def test_a_weak_match_far_from_the_strong_drift_curve_is_rejected():
    """AUDIO_WEAK_DRIFT_TOL_SEC = 1.0: a one-word match 1.3 s from where the strong anchors say the cue should be is
    rejected. (At 1.5 s it dragged its neighbour a second late on the Zoo York block.)"""
    cues = timeline(20)
    speech = speech_for(cues, lambda t: 1.0, only={11: [0]})
    speech[11] = spoken(cue_text(11), cues[11][0] + 1.0 + 1.3, only=[0])
    _, aligner = align(cues, speech)
    assert 11 not in aligner.anchored_idx


def test_a_weak_match_near_the_strong_drift_curve_is_kept():
    cues = timeline(20)
    speech = speech_for(cues, lambda t: 1.0)
    speech[11] = spoken(cue_text(11), cues[11][0] + 1.0 + 0.5, only=[0])
    _, aligner = align(cues, speech)
    assert 11 in aligner.anchored_idx


def test_a_cue_nobody_spoke_is_placed_between_its_neighbours_and_keeps_its_duration():
    """Unmatched cues are interpolated; durations are never rescaled (end = new start + original duration)."""
    cues = timeline(20)
    subs, aligner = align(cues, speech_for(cues, lambda t: 2.5, skip={7}))
    assert 7 not in aligner.anchored_idx
    assert subs[7].start / 1000.0 == pytest.approx(cues[7][0] + 2.5, abs=TOL)
    assert subs[7].end - subs[7].start == 2500


# ---------------------------------------------------------------- spelling and leading words

def test_a_name_spelled_differently_by_the_recogniser_still_anchors():
    """_snap_to_script: 'Vanetti' heard for 'Venetti' (>= 5 letters, similarity >= 0.85) is respelled, so the cue anchors
    instead of being interpolated."""
    cues = timeline(20)
    cues[9] = (cues[9][0], cues[9][1], "Venetti.")
    speech = speech_for(cues, lambda t: 1.0)
    speech[9] = spoken("Vanetti.", cues[9][0] + 1.0)
    _, aligner = align(cues, speech)
    assert 9 in aligner.anchored_idx


def test_look_alike_words_are_not_merged():
    """SNAP_MIN_SIMILARITY = 0.85 and SNAP_MIN_LETTERS = 5: 'three' must not become 'these' (0.8 merged real words),
    and short words ('than' / 'that') are never touched."""
    script = [{"word": w} for w in ("these", "would", "than", "ahead", "venetti")]
    heard = [{"word": w} for w in ("three", "could", "that", "head", "vanetti")]
    changed = alignment.GlobalAligner._snap_to_script(script, heard)
    # 'head' / 'ahead' are 0.89 alike: only the 5-letter minimum keeps them apart
    assert [t["word"] for t in heard] == ["three", "could", "that", "head", "venetti"]
    assert changed == 1


def test_leading_unmatched_words_move_the_start_back_at_the_cues_own_pace():
    """_match_start: the words before the first matched one were spoken earlier. Back off by the pace of the cue's own
    matched words, at most LEADING_WORD_MAX words."""
    aligner = alignment.GlobalAligner(make_subs([]), [])
    tok = lambda t: {"start": t}
    # words 2, 3, 4 matched, 0.4 s apart: two words missing before them
    assert aligner._match_start([(2, tok(10.0)), (3, tok(10.4)), (4, tok(10.8))]) == pytest.approx(10.0 - 2 * 0.4)
    # a lone matched word falls back to LEADING_WORD_SEC per missing word
    assert aligner._match_start([(1, tok(10.0))]) == pytest.approx(10.0 - 0.3)
    # never more than three words back
    assert aligner._match_start([(6, tok(10.0))]) == pytest.approx(10.0 - 3 * 0.3)
    # a cue whose first word matched starts exactly there
    assert aligner._match_start([(0, tok(10.0))]) == 10.0


def test_a_cue_with_unmatched_opening_words_never_starts_later_than_the_strong_drift_curve():
    """'Tape. / Meat roll.' matched only on 'roll' and started a second late. Its drift is clamped to the curve of the
    strong anchors (see anchor_points: the drift recorded for the cue)."""
    cues = timeline(21, words=5)
    cues[20] = (cues[20][0], cues[20][1], "alpha11x bravo22x charlie33x")
    speech = speech_for(cues[:20], lambda t: 1.0)
    speech.append(spoken("alpha11x bravo22x charlie33x", cues[20][0] + 1.0, only=[2]))
    speech[-1] = spoken("charlie33x", cues[20][0] + 1.9)           # only the last word was heard, late
    _, aligner = align(cues, speech)
    drift = {round(x, 3): d for x, d, _ in aligner.anchor_points}[round(cues[20][0], 3)]
    assert drift == pytest.approx(1.0, abs=0.01), "the cue was not held to the drift curve of the strong anchors"


# ---------------------------------------------------------------- the edges of the file

def test_cues_after_the_last_anchor_follow_the_drift_trend_not_a_constant_shift():
    """edge_drift_slope: where the drift is still changing, a constant shift left the first cues of a clip 2-3 s late."""
    cues = timeline(30)
    subs, _ = align(cues, speech_for(cues, lambda t: 1.0 + 0.01 * t, skip={25, 26, 27, 28, 29}))
    orig = cues[27][0]
    assert subs[27].start / 1000.0 == pytest.approx(orig + 1.0 + 0.01 * orig, abs=0.05)


def test_the_edge_trend_needs_enough_anchors_and_spread():
    pts = lambda xs, slope: [{"orig_start": x, "raw_match_time": x + 1.0 + slope * x} for x in xs]
    assert alignment.edge_drift_slope(pts([0, 30, 60, 90], 0.01)) == pytest.approx(0.01)
    assert alignment.edge_drift_slope(pts([0, 30, 60], 0.01)) == 0.0            # fewer than EDGE_MIN_ANCHORS
    assert alignment.edge_drift_slope(pts([0, 4, 8, 12], 0.01)) == 0.0          # closer than EDGE_MIN_SPAN_SEC
    assert alignment.edge_drift_slope(pts([0, 30, 60, 90], 0.5)) == 0.1            # never more than 10% of the media time


def test_the_edge_trend_adds_at_most_a_few_seconds():
    assert alignment.GlobalAligner._edge_extra(0.1, 1000.0) == 6.0
    assert alignment.GlobalAligner._edge_extra(-0.1, 1000.0) == -6.0


# ---------------------------------------------------------------- whisper timing filter

def test_speech_with_impossible_timing_is_dropped_before_matching():
    """drop_implausible_timing: WhisperX can squeeze a loud scene into a few seconds, or stretch 6 words over 25 s."""
    slow = {"start": 0.0, "end": 25.0, "text": "a b c d e f", "words": [{"word": w, "start": i * 4.0, "end": i * 4.0 + 1} for i, w in enumerate("abcdef")]}
    fast = {"start": 40.0, "end": 41.0, "text": "x", "words": [{"word": f"x{i}", "start": 40.0 + i * 0.1, "end": 40.1 + i * 0.1} for i in range(10)]}
    normal = spoken("one two three four five", 60.0)
    kept, dropped = alignment.drop_implausible_timing([slow, fast, normal])
    assert kept == [normal]
    assert dropped == 16


def test_the_timing_filter_leaves_its_input_alone():
    data = [spoken("one two three four five", 10.0)]
    before = [dict(s) for s in data]
    kept, dropped = alignment.drop_implausible_timing(data)
    assert data == before and dropped == 0 and kept == data


# ---------------------------------------------------------------- smoothing (audio mode)

def test_whisper_smoothing_gives_a_short_block_one_median_shift():
    anchors = [{"orig_start": x, "raw_match_time": x + d, "strong": True} for x, d in ((0, 1.0), (3, 2.0), (6, 3.0))]
    out = alignment.smooth_offsets_by_block(anchors, precise=False)
    assert [round(a["final_start"] - a["orig_start"], 3) for a in out] == [2.0, 2.0, 2.0]


def test_whisper_smoothing_follows_a_straight_line_through_a_long_block():
    """BLOCK_TREND: in a long block the true drift keeps changing; one median made the start late and the end early."""
    xs = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    anchors = [{"orig_start": x, "raw_match_time": x + 0.5 + 0.05 * x, "strong": True} for x in xs]
    out = alignment.smooth_offsets_by_block(anchors, precise=False)
    for a in out:
        assert a["final_start"] - a["orig_start"] == pytest.approx(0.5 + 0.05 * a["orig_start"], abs=0.01)


def test_a_gap_over_five_seconds_starts_a_new_block():
    """SCENE_GAP_SEC: scenes get their own shift."""
    anchors = [{"orig_start": x, "raw_match_time": x + d, "strong": True} for x, d in ((0, 1.0), (2, 1.0), (4, 1.0), (30, 5.0), (32, 5.0), (34, 5.0))]
    out = alignment.smooth_offsets_by_block(anchors, precise=False)
    assert [round(a["final_start"] - a["orig_start"], 3) for a in out] == [1.0, 1.0, 1.0, 5.0, 5.0, 5.0]


def test_parakeet_smoothing_lets_a_strong_anchor_keep_its_own_time():
    """precise=True (Parakeet word times are accurate): a strong anchor starts exactly where its own match says. Smoothing
    whole blocks erased real cue-to-cue offsets of up to 0.7 s and made cues look 'hit or miss'."""
    anchors = [
        {"orig_start": 0.0, "raw_match_time": 1.0, "strong": True},
        {"orig_start": 4.0, "raw_match_time": 4.0 + 1.7, "strong": True},
        {"orig_start": 8.0, "raw_match_time": 8.0 + 0.8, "strong": True},
    ]
    out = alignment.smooth_offsets_by_block(anchors, precise=True)
    assert [round(a["final_start"] - a["orig_start"], 3) for a in out] == [1.0, 1.7, 0.8]


def test_parakeet_smoothing_gives_a_weak_anchor_the_median_of_its_neighbours():
    anchors = [
        {"orig_start": 0.0, "raw_match_time": 1.0, "strong": True},
        {"orig_start": 4.0, "raw_match_time": 4.0 + 2.6, "strong": False},     # a stray single-word match
        {"orig_start": 8.0, "raw_match_time": 8.0 + 1.1, "strong": True},
    ]
    out = alignment.smooth_offsets_by_block(anchors, precise=True)
    assert out[1]["final_start"] - out[1]["orig_start"] == pytest.approx(1.1)


# ---------------------------------------------------------------- the zipper

def _events(*spans):
    return make_subs([(a / 1000.0, b / 1000.0, "x") for a, b in spans])


def test_the_zipper_trims_an_overlap_and_keeps_a_gap():
    subs = alignment.enforce_strict_spacing(_events((0, 1000), (900, 2000)))
    assert subs[0].end == 900 - 50                      # 50 ms between cues
    subs = alignment.enforce_strict_spacing(_events((0, 1000), (1020, 2000)))
    assert subs[0].end == 1020 - 50


def test_the_zipper_never_shortens_a_cue_below_the_minimum_and_pushes_the_next_one_instead():
    subs = alignment.enforce_strict_spacing(_events((0, 700), (300, 2000)))
    assert subs[0].end - subs[0].start == 600           # never shorter than 600 ms
    assert subs[1].start == subs[0].end + 50


def test_reference_mode_zipper_only_resolves_real_overlaps():
    """gap_ms=0, min_duration_ms=0: a reference may use tighter gaps than 50 ms, and an anchored cue is never pushed."""
    subs = alignment.enforce_strict_spacing(_events((0, 1000), (1020, 2000), (1900, 2500)), gap_ms=0, min_duration_ms=0)
    assert subs[0].end == 1000 and subs[0].start == 0           # a 20 ms gap is left alone
    assert subs[1].end == 1900                                   # the real overlap is trimmed to the next start


def test_the_zipper_reports_real_overlaps_and_trimmed_gaps_separately():
    """'Resolved N overlaps' used to count tight gaps too, which alarmed users about files that were fine."""
    alignment.console.quiet = False
    with alignment.console.capture() as captured:
        alignment.enforce_strict_spacing(_events((0, 1000), (900, 2000), (2020, 3000)))
    text = captured.get()
    assert "Resolved 1 overlap," in text
    assert "trimmed 1 cue end" in text


# ---------------------------------------------------------------- exact thresholds (changing one should be a decision)

def test_scenes_split_at_a_gap_of_more_than_five_seconds():
    mk = lambda gap: [{"orig_start": x, "raw_match_time": x + d, "strong": True}
                      for x, d in ((0, 1.0), (2, 1.0), (4, 1.0), (4 + gap, 5.0), (6 + gap, 5.0), (8 + gap, 5.0))]
    together = alignment.smooth_offsets_by_block(mk(4.5), precise=False)
    apart = alignment.smooth_offsets_by_block(mk(5.5), precise=False)
    assert {round(a["final_start"] - a["orig_start"], 3) for a in together} == {3.0}       # one block: median of 1 and 5
    assert {round(a["final_start"] - a["orig_start"], 3) for a in apart} == {1.0, 5.0}


def test_strong_means_four_matched_words_covering_sixty_percent():
    strong = alignment.GlobalAligner._is_strong
    assert strong(4, 6) and strong(4, 4) and strong(10, 15)
    assert not strong(3, 3)          # too few words, however complete
    assert not strong(4, 7)          # 57%


def test_without_strong_anchors_a_match_is_judged_against_its_neighbours_within_1_5_seconds():
    """OUTLIER_THRESHOLD_SEC: the fallback when fewer than two strong anchors exist."""
    cues = timeline(14)
    speech = speech_for(cues, lambda t: 1.0, only={i: [0, 1] for i in range(14)})     # 2 of 5 words: all weak
    speech[4] = spoken(cue_text(4), cues[4][0] + 1.0 + 2.0, only=[0, 1])
    speech[8] = spoken(cue_text(8), cues[8][0] + 1.0 + 1.2, only=[0, 1])
    _, aligner = align(cues, speech)
    assert 4 not in aligner.anchored_idx and 8 in aligner.anchored_idx


def test_the_timing_filter_thresholds():
    def seg(words, duration, start=0.0):
        step = duration / words
        return {"start": start, "end": start + duration, "text": "x",
                "words": [{"word": f"w{i}", "start": start + i * step, "end": start + (i + 1) * step} for i in range(words)]}
    keep = lambda s: alignment.drop_implausible_timing([s])[0] == [s]
    assert keep(seg(2, 5.0))             # slow but shorter than 5.5 s: kept
    assert not keep(seg(2, 6.0))         # slow and 5.5 s or longer: a timestamp collapse
    assert keep(seg(7, 1.0))             # 7 words/s
    assert not keep(seg(9, 1.0))         # 9 words/s cannot be speech
