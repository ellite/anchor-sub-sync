"""Change Frame Rate (core/framerate/framerate.py). See AGENTS.md, 'Other modules'."""
import argparse

import pytest
from rich.console import Console

from anchor.core.framerate import framerate
from helpers import make_subs


def retime(path, from_label, to_label, overwrite=False):
    _, from_fps = framerate.resolve_rate(float(from_label))
    _, to_fps = framerate.resolve_rate(float(to_label))
    args = argparse.Namespace(overwrite=overwrite, backup=False)
    return framerate._retime_file(path, from_fps, to_fps, from_label, to_label, args, Console(quiet=True))


def test_the_factor_is_from_over_to_so_23976_to_24_pulls_cues_earlier():
    """Verified against Subtitle Edit to within 1 ms on every cue: a cue at 2400 s moves to 2397.6 s."""
    subs = make_subs([(2400.0, 2402.0, "x")])
    factor = framerate.convert_frame_rate(subs, 24000 / 1001, 24.0)
    assert factor == pytest.approx(0.999001, abs=1e-6)
    assert subs[0].start == 2397602


def test_a_subtitle_for_a_faster_rate_is_stretched_for_a_slower_video():
    subs = make_subs([(1000.0, 1002.0, "x")])
    assert framerate.convert_frame_rate(subs, 25.0, 24000 / 1001) == pytest.approx(25 / (24000 / 1001))
    assert subs[0].start > 1000 * 1000


def test_ntsc_rates_snap_to_the_exact_fractions_video_files_use():
    assert framerate.resolve_rate(23.98)[1] == 24000 / 1001
    assert framerate.resolve_rate(29.97)[1] == 30000 / 1001
    assert framerate.resolve_rate(59.94)[1] == 60000 / 1001
    assert framerate.resolve_rate(24)[1] == 24.0
    assert framerate.resolve_rate(25.5)[1] == 25.5


def test_the_output_is_named_after_the_target_rate_and_never_replaces_an_earlier_one(tmp_path):
    """Never '.synced': a frame rate change must not overwrite an audio or reference sync result."""
    src = tmp_path / "Movie.en.srt"
    make_subs([(10.0, 12.0, "hello")]).save(str(src))
    original = src.read_text()
    assert retime(src, "25", "24")
    first = (tmp_path / "Movie.en.24fps.srt")
    first.write_text(first.read_text() + "\n; hand edit\n")            # the user touched the result
    edited = first.read_text()
    assert retime(src, "25", "24")
    assert first.read_text() == edited                                   # the second run did not replace it
    assert (tmp_path / "Movie.en.24fps.1.srt").exists()
    assert src.read_text() == original                                   # nor the source


def test_dash_o_replaces_the_original_instead(tmp_path):
    src = tmp_path / "Movie.en.srt"
    make_subs([(10.0, 12.0, "hello")]).save(str(src))
    before = src.read_text()
    assert retime(src, "25", "24", overwrite=True)
    assert src.read_text() != before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["Movie.en.srt"]


def test_microdvd_sub_files_are_refused_because_they_store_frame_numbers(tmp_path):
    """A .sub already follows the video's frame rate; stretching the frame numbers would correct twice."""
    src = tmp_path / "Movie.sub"
    src.write_text("{1}{1}23.976\n{100}{200}Hello\n")
    assert retime(src, "25", "24") is False
    assert [p.name for p in tmp_path.iterdir()] == ["Movie.sub"]


def test_only_times_change_never_the_text():
    subs = make_subs([(1.0, 2.0, "keep me"), (3.0, 4.0, "and me")])
    framerate.convert_frame_rate(subs, 25.0, 24.0)
    assert [e.text for e in subs] == ["keep me", "and me"]
