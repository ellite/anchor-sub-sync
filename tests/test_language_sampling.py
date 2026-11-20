import numpy as np

from anchor.utils.languages import detect_language_by_sampling

RATE = 16000


class FakeModel:
    """Answers per call from a script; an Exception in the script is raised. Records each slice it was given."""
    def __init__(self, answers):
        self.answers = list(answers)
        self.slices = []

    def detect_language(self, audio):
        self.slices.append(len(audio))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def audio(seconds):
    return np.zeros(int(seconds * RATE), dtype=np.float32)


def test_majority_wins_over_a_silent_start():
    assert detect_language_by_sampling(FakeModel(["th", "sv", "sv"]), audio(600)) == "sv"


def test_all_three_different_takes_the_middle_sample():
    assert detect_language_by_sampling(FakeModel(["th", "sv", "de"]), audio(600)) == "sv"


def test_a_failed_sample_is_reported_and_the_others_still_vote():
    warnings = []
    result = detect_language_by_sampling(FakeModel([RuntimeError("CUDA out of memory"), "sv", "sv"]), audio(600), warn=warnings.append)
    assert result == "sv"
    assert len(warnings) == 1 and "15%" in warnings[0] and "out of memory" in warnings[0]


def test_two_answers_that_differ_fall_to_the_middle_one():
    assert detect_language_by_sampling(FakeModel([RuntimeError("x"), "sv", "de"]), audio(600)) == "sv"


def test_no_answer_returns_none():
    warnings = []
    model = FakeModel([RuntimeError("x"), "unknown", None])
    assert detect_language_by_sampling(model, audio(600), warn=warnings.append) is None
    assert len(warnings) == 1


def test_short_audio_is_left_to_whisper():
    model = FakeModel([])
    assert detect_language_by_sampling(model, audio(119)) is None
    assert model.slices == []


def test_exactly_two_minutes_is_sampled():
    assert detect_language_by_sampling(FakeModel(["sv", "sv", "sv"]), audio(120)) == "sv"


def test_every_sample_is_a_full_30_seconds_even_at_the_end_of_the_file():
    model = FakeModel(["sv", "sv", "sv"])
    detect_language_by_sampling(model, audio(121))
    assert model.slices == [30 * RATE] * 3


def test_samples_sit_at_15_50_85_percent():
    seen = []

    class Marked:
        def detect_language(self, audio):
            seen.append(float(audio[0]))
            return "sv"

    clip = np.arange(1000 * RATE, dtype=np.float32)   # each value is its own sample index
    detect_language_by_sampling(Marked(), clip)
    assert seen == [150 * RATE, 500 * RATE, 850 * RATE]
