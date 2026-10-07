"""Reference sync pairing picker: the selection rules (utils/pairing.py, PairingState). See AGENTS.md, 'Reference sync'.

The curses drawing is not tested here; what matters is which target ends up with which reference.
"""
import pytest

from anchor.utils.pairing import PairingState, natural_key, shorten

NAMES = [f"Show.S01E{n:02d}.{lang}.srt" for n in (1, 2, 3) for lang in ("en", "pt")]


def state():
    return PairingState(NAMES, NAMES)


def test_picks_are_numbered_in_the_order_they_were_made_and_pair_by_number():
    """Pair N is the Nth left pick with the Nth right pick: that is the whole contract of the picker."""
    s = state()
    s.toggle(0, 2); s.toggle(0, 0)                 # left: item 2 is 1, item 0 is 2
    s.toggle(1, 5); s.toggle(1, 1)                 # right: item 5 is 1, item 1 is 2
    assert s.number(0, 2) == 1 and s.number(0, 0) == 2
    assert s.pairs() == [(2, 5), (0, 1)]


def test_unpicking_renumbers_the_later_picks():
    s = state()
    for i in (3, 1, 4):
        s.toggle(0, i)
    s.toggle(0, 3)
    assert (s.number(0, 1), s.number(0, 4)) == (1, 2)


def test_a_filter_narrows_the_pane_and_select_all_picks_only_what_is_shown_in_listing_order():
    """A season folder holds both languages; '/' '.pt.' then A must pick only the .pt files."""
    s = state()
    s.set_filter(0, ".PT.")                       # case-insensitive
    assert [NAMES[i] for i in s.visible(0)] == [n for n in NAMES if ".pt." in n]
    s.select_all(0)
    assert [NAMES[i] for i in s.picks[0]] == [n for n in NAMES if ".pt." in n]


def test_select_all_again_unpicks_what_is_shown_but_keeps_other_picks():
    s = state()
    s.toggle(0, 0)                                 # an .en file picked by hand
    s.set_filter(0, ".pt."); s.select_all(0); s.select_all(0)
    assert s.picks[0] == [0]


def test_confirming_needs_matching_counts_and_at_least_one_pair():
    s = state()
    assert s.problem()
    s.toggle(0, 1); s.toggle(0, 3)
    s.toggle(1, 0)
    assert "counts must match" in s.problem()
    s.toggle(1, 2)
    assert s.problem() is None


def test_a_file_cannot_be_both_target_and_reference_of_one_pair():
    s = state()
    s.toggle(0, 1); s.toggle(1, 1)
    assert "both target and reference" in s.problem()


def test_the_cursor_works_on_the_shown_list_not_the_whole_list():
    s = state()
    s.set_filter(0, ".pt.")
    s.move(1)
    s.toggle()
    assert NAMES[s.picks[0][0]] == "Show.S01E02.pt.srt"


def test_episodes_sort_naturally_and_long_names_keep_both_ends():
    assert sorted(["E10", "E2", "E1"], key=natural_key) == ["E1", "E2", "E10"]
    cut = shorten("CSI NY S02E03 Zoo York 720p Web-DL x265-OFT.en.srt", 30)
    assert len(cut) == 30 and cut.startswith("CSI NY") and cut.endswith("x265-OFT.en.srt"[-12:])
