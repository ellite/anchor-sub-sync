import pysubs2
from anchor.utils.alignment import GlobalAligner


def speech(words, start):
    return [{'text': ' '.join(words), 'start': start, 'end': start + 0.4 * len(words),
             'words': [{'word': w, 'start': start + 0.4 * i, 'end': start + 0.4 * i + 0.3} for i, w in enumerate(words)]}]


def subs_of(texts, step=4000):
    s = pysubs2.SSAFile()
    for i, t in enumerate(texts):
        s.append(pysubs2.SSAEvent(start=i * step, end=i * step + 2000, text=t))
    return s


SPOKEN = ["we need backup at the north exit now", "split up and search the building quickly", "did you hear that noise outside"]


def aligned(alternatives):
    data = []
    for i, line in enumerate(SPOKEN):
        data += speech(line.split(), i * 4.0)
    subs = subs_of(["we need backup at the north exit now", "separate yourself", "did you hear that noise outside"])
    if alternatives:
        subs.alternatives = alternatives
    aligner = GlobalAligner(subs, data)
    aligner.run()
    return aligner


def test_cue_matches_better_with_another_translation():
    aligner = aligned([[SPOKEN[0], "split up and search the building", SPOKEN[2]], [SPOKEN[0], "get out of the way", SPOKEN[2]]])
    assert aligner.variant == {1: 1}
    assert 1 in aligner.anchored_idx


def test_without_alternatives_the_cue_stays_unmatched():
    aligner = aligned(None)
    assert aligner.variant == {}
    assert 1 not in aligner.anchored_idx


def test_alternative_must_match_more_words_than_the_main_text():
    aligner = aligned([[SPOKEN[0], "separate yourself", SPOKEN[2]]])
    assert aligner.variant == {}
