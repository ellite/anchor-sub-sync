from anchor.utils.alignment import GlobalAligner


def test_translated_strong_match_is_more_forgiving():
    # "Do you hear that" matched 3 of 4 words; "I need backup on the north exit" 4 of 7
    assert not GlobalAligner._is_strong(3, 4)
    assert not GlobalAligner._is_strong(4, 7)
    assert GlobalAligner._is_strong(3, 4, 3, 0.5)
    assert GlobalAligner._is_strong(4, 7, 3, 0.5)
    assert not GlobalAligner._is_strong(2, 5, 3, 0.5)
