"""Sync Check: how the drift of anchored cues becomes a verdict (utils/syncverdict.py).

Sign convention: drift = where the speech is minus where the subtitle puts the cue. Positive: the subtitle is early.
See AGENTS.md, 'Sync check'. The frame rate rule is deliberate: suggest a pair only when exactly one fits.
"""
import numpy as np
import pytest

from anchor.utils import syncverdict
from anchor.utils.syncverdict import assess

N, SPAN, CUES = 240, 2400.0, 260


def points(drift, noise=0.25, seed=7, n=N):
    rng = np.random.default_rng(seed)
    xs = np.linspace(30.0, SPAN, n)
    return [(float(x), float(drift(x) + rng.normal(0, noise)), True) for x in xs]


def test_a_subtitle_with_small_drift_everywhere_is_in_sync():
    r = assess(points(lambda x: 0.1), CUES)
    assert r["verdict"] == "in_sync"


def test_a_steady_shift_is_a_constant_offset_and_its_size_is_reported():
    r = assess(points(lambda x: 3.1), CUES)
    assert r["verdict"] == "offset"
    assert r["median"] == pytest.approx(3.1, abs=0.1)


def test_the_in_sync_limit_is_six_tenths_of_a_second():
    """SYNC_MEDIAN_SEC: people time cues 0.2-1 s off the speech; a real offset starts beyond that."""
    assert assess(points(lambda x: 0.5, noise=0.05), CUES)["verdict"] == "in_sync"
    assert assess(points(lambda x: 0.7, noise=0.05), CUES)["verdict"] == "offset"


def test_a_drift_that_grows_steadily_is_drifting_and_names_the_one_frame_rate_pair_that_explains_it():
    """A subtitle made for 25 fps on a 23.976 fps video runs early by 4.27%: the fix is Change Frame Rate 25 -> 23.976."""
    r = assess(points(lambda x: 1.0 + 0.0427 * x), CUES, video_fps=24000 / 1001)
    assert r["verdict"] == "drifting"
    assert r["fps"] == ("25", "23.976")


def test_a_frame_rate_pair_is_not_suggested_when_two_pairs_fit_equally_well():
    """23.976 -> 25 and 24 -> 25 differ by 0.1%; a measured slope is only good to about 0.2%. Guessing wrong leaves seconds
    of drift on a full episode, so nothing is suggested and both candidates are named."""
    r = assess(points(lambda x: 1.0 - 0.0427 * x), CUES, video_fps=25.0)
    assert r["verdict"] == "drifting"
    assert r["fps"] is None
    assert set(r["fps_alternatives"]) == {"23.976 to 25", "24 to 25"}
    assert "Audio Sync" in r["advice"] and "cannot tell" in r["advice"]


def test_the_videos_own_frame_rate_removes_ambiguity():
    """The subtitle's target rate must equal the video's, so 23.976 -> 24 stops being confusable with 29.97 -> 30."""
    slope = -(1 - (24000 / 1001) / 24)
    assert assess(points(lambda x: slope * x, noise=0.1), CUES)["fps"] is None
    assert assess(points(lambda x: slope * x, noise=0.1), CUES, video_fps=24.0)["fps"] == ("23.976", "24")
    # matching noise and human timing bend a measured slope by about 0.0003 per second: that must not lose the pair
    assert assess(points(lambda x: (slope - 0.0003) * x, noise=0.1), CUES, video_fps=24.0)["fps"] == ("23.976", "24")


def test_a_drift_no_frame_rate_explains_gets_no_suggestion():
    r = assess(points(lambda x: 0.03 * x), CUES, video_fps=24000 / 1001)
    assert r["verdict"] == "drifting" and r["fps"] is None


def test_a_jump_in_the_offset_is_steps_not_drift():
    """Another cut, removed scenes or a half-synced file: the offset changes in jumps, not steadily."""
    r = assess(points(lambda x: 0.1 if x < 1200 else 6.1), CUES)
    assert r["verdict"] == "steps"
    assert len(r["steps"]) == 1 and r["steps"][0][0] == pytest.approx(1200, abs=60)


def test_a_wobble_that_is_not_a_straight_line_is_out_of_sync_even_without_one_big_jump():
    """STEP_RESIDUAL_SEC: when 90% of cues are more than 1.5 s from the best straight line, no single offset or drift fixes it."""
    r = assess(points(lambda x: 2.5 * np.sin(2 * np.pi * x / 900.0), noise=0.2), CUES)
    assert r["verdict"] == "steps" and r["steps"] == []


def test_a_steady_drift_is_not_mistaken_for_steps():
    """Block medians wander by about 1 s around a steep line; only jumps of STEP_JUMP_SEC (3 s) are steps."""
    r = assess(points(lambda x: 1.0 + 0.047 * x, noise=0.6), CUES)
    assert r["verdict"] == "drifting" and r["steps"] == []


def test_a_subtitle_that_barely_matches_is_not_comparable():
    """Too few cues found in the speech: another episode, another language or a damaged file. Nothing is offered."""
    assert assess(points(lambda x: 0.0, n=5), CUES)["verdict"] == "not_comparable"
    assert assess(points(lambda x: 0.0, n=40), 400)["verdict"] == "not_comparable"          # 10% < MIN_COVERAGE


def test_centred_but_loose_timing_is_mostly_in_sync_not_a_failure():
    r = assess(points(lambda x: 0.1, noise=0.7), CUES)
    assert r["verdict"] == "loose"


def test_comparable_needs_eight_anchors_and_a_quarter_of_the_cues():
    assert assess(points(lambda x: 0.0, n=7), 20)["verdict"] == "not_comparable"
    assert assess(points(lambda x: 0.0, n=8), 20)["verdict"] == "in_sync"
    assert assess(points(lambda x: 0.0, n=48), 200)["verdict"] == "not_comparable"        # 24%
    assert assess(points(lambda x: 0.0, n=52), 200)["verdict"] == "in_sync"               # 26%


def test_a_drift_of_one_second_over_the_file_is_the_line_between_offset_and_drifting():
    assert assess(points(lambda x: 0.2 + 0.0003 * x, noise=0.02), CUES)["verdict"] == "in_sync"    # 0.7 s over the file
    assert assess(points(lambda x: 0.2 + 0.0006 * x, noise=0.02), CUES)["verdict"] == "drifting"   # 1.4 s


def test_assess_does_not_modify_its_input():
    pts = points(lambda x: 1.0)
    before = list(pts)
    assess(pts, CUES)
    assert pts == before


def test_every_verdict_has_a_label_for_the_report():
    for verdict in ("in_sync", "loose", "offset", "drifting", "steps", "not_comparable"):
        assert verdict in syncverdict.LABELS
