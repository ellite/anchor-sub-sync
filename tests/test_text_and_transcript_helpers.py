"""Text cleaning and the recognised-speech helpers (utils/formatting.py, utils/parakeet.py, transcribe task)."""
import pytest

from anchor.utils import parakeet
from anchor.utils.formatting import clean_text


def test_clean_text_reduces_a_cue_to_comparable_words():
    """The matcher compares these words with the recogniser's: tags, ASS overrides, line breaks and punctuation must not matter."""
    assert clean_text("{\\an8}<i>Oh, you STILL haven't\\Nfound a place?</i>") == "oh you still havent found a place"


def test_parakeet_tokens_become_words_a_leading_space_starts_a_new_word():
    words = parakeet.words_from_tokens([" Hel", "lo", " world."], [1.0, 1.1, 1.5])
    assert [w["word"] for w in words] == ["Hello", "world."]
    assert words[0]["start"] == 1.0 and words[0]["end"] == pytest.approx(1.1 + 0.08)


def test_a_pause_or_a_sentence_end_closes_a_segment():
    w = lambda text, start: {"word": text, "start": start, "end": start + 0.2}
    segments = parakeet.segments_from_words([w("one", 0.0), w("two.", 0.3), w("three", 0.6), w("four", 2.0)])
    assert [s["text"] for s in segments] == ["one two.", "three", "four"]


def test_the_silence_and_pause_limits_are_exact():
    gap = lambda g: parakeet._silent_stretches([{"start": 0.0, "end": 10.0}, {"start": 10.0 + g, "end": 30.0}], 30.0)
    assert gap(3.9) == [] and gap(4.1) == [(10.0, 14.1)]
    w = lambda start: {"word": "a", "start": start, "end": start + 0.2}
    assert len(parakeet.segments_from_words([w(0.0), w(0.2 + 0.7)])) == 1       # a 0.7 s pause stays in the segment
    assert len(parakeet.segments_from_words([w(0.0), w(0.2 + 0.9)])) == 2


def test_silent_stretches_must_be_longer_than_four_seconds():
    segments = [{"start": 10.0, "end": 12.0}, {"start": 15.0, "end": 16.0}, {"start": 21.0, "end": 22.0}]
    # head 0-10 and 16-21 and 22-30 are long; 12-15 (3 s) is not
    assert parakeet._silent_stretches(segments, 30.0) == [(0.0, 10.0), (16.0, 21.0), (22.0, 30.0)]


def test_only_cues_inside_a_silent_stretch_are_repaired_not_text_mismatches():
    """A cue that stays unanchored although Parakeet heard speech around it is a text mismatch; re-transcribing it would
    not help and costs seconds per zone."""
    busy = [{"start": 0.0, "end": 100.0}]
    assert parakeet._missing_speech_zones(busy, [{"start": 40.0, "end": 42.0, "words": 5}], 100.0) == []
    gap = [{"start": 0.0, "end": 10.0}, {"start": 60.0, "end": 100.0}]
    zones = parakeet._missing_speech_zones(gap, [{"start": 30.0, "end": 32.0, "words": 5}], 100.0)
    assert zones == [(18.0, 44.0)]                       # the cue's position +- ZONE_REACH_SEC, inside the silence


def test_a_zone_never_reaches_beyond_the_silent_stretch():
    gap = [{"start": 0.0, "end": 10.0}, {"start": 20.0, "end": 100.0}]
    zones = parakeet._missing_speech_zones(gap, [{"start": 14.0, "end": 15.0, "words": 3}], 100.0)
    assert zones == [(10.0, 20.0)]


def test_zones_less_than_four_seconds_apart_are_merged_and_farther_ones_are_not():
    """One faster-whisper pass per zone costs seconds; two zones with a few seconds of speech between them are one pass."""
    cue = lambda a: {"start": a, "end": a + 1.0, "words": 3}
    near = [{"start": 30.0, "end": 33.0}]                                    # 3 s of speech between the two silences
    assert parakeet._missing_speech_zones(near, [cue(28.0), cue(34.0)], 100.0) == [(16.0, 47.0)]
    far = [{"start": 30.0, "end": 35.0}]                                     # 5 s of speech
    assert len(parakeet._missing_speech_zones(far, [cue(28.0), cue(36.0)], 100.0)) == 2


def test_a_cue_slightly_outside_a_silent_stretch_still_counts_but_not_far_outside():
    """CUE_REACH_SEC: a cue's expected position is only an estimate (the offset is unknown)."""
    busy_after = [{"start": 30.0, "end": 100.0}]                             # silence 0-30
    assert parakeet._missing_speech_zones(busy_after, [{"start": 32.0, "end": 33.0, "words": 3}], 100.0) == [(20.0, 30.0)]
    assert parakeet._missing_speech_zones(busy_after, [{"start": 34.5, "end": 35.5, "words": 3}], 100.0) == []


def test_stretched_leading_words_are_trimmed_but_one_word_is_always_kept():
    """WhisperX stretched a hallucinated 'The' over 4.4 s and turned a 5 s announcement into a 14.6 s cue from 0.0 s."""
    pytest.importorskip("whisperx")
    from anchor.core.transcribe.transcribe import trim_stretched_leading_words
    seg = {"start": 0.0, "words": [{"word": "The", "start": 0.0, "end": 4.4}, {"word": "Attention", "start": 9.8, "end": 10.3},
                                   {"word": "please", "start": 10.4, "end": 10.8}]}
    assert trim_stretched_leading_words([seg]) == 1
    assert [w["word"] for w in seg["words"]] == ["Attention", "please"]
    ok = {"start": 0.0, "words": [{"word": "Oh", "start": 0.0, "end": 1.4}, {"word": "no", "start": 1.5, "end": 1.9}]}
    assert trim_stretched_leading_words([ok]) == 0                                # 1.4 s is not stretched (limit 1.5 s)
    assert seg["start"] == pytest.approx(9.8 - 0.3)
    lone = {"start": 0.0, "words": [{"word": "Hmm", "start": 0.0, "end": 6.0}]}
    trim_stretched_leading_words([lone])
    assert len(lone["words"]) == 1
