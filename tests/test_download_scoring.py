from anchor.core.download.scoring import calculate_score, titles_match


def test_initialism_matches_full_name():
    assert titles_match("CSI NY", "CSI New York")
    assert titles_match("CSI New York", "CSI NY")


def test_other_shows_do_not_match():
    assert not titles_match("CSI NY", "CSI Miami")
    assert not titles_match("CSI NY", "CSI Vegas")
    assert not titles_match("The Office", "Office Space")


def test_articles_and_near_spellings_still_match():
    assert titles_match("The Mentalist", "Mentalist")
    assert titles_match("Marvels Agents of SHIELD", "Marvel's Agents of S.H.I.E.L.D.")


def parsed(name):
    from anchor.utils.parsers import parse_video_filename
    return parse_video_filename(name)


def test_zoo_york_release_is_not_rated_wrong_show():
    sub = {"filename": "CSI.New.York.S02E03.Zoo.York.DVDRip", "releases": []}
    assert calculate_score(parsed("CSI NY S02E03 Zoo York 720p Web-DL x265-OFT.mkv"), sub, ["pt"]) > 0


def test_wrong_episode_is_still_rejected():
    sub = {"filename": "CSI.New.York.S02E04.Some.Other.DVDRip", "releases": []}
    assert calculate_score(parsed("CSI NY S02E03 Zoo York 720p Web-DL x265-OFT.mkv"), sub, ["pt"]) == -100


# ---------------------------------------------------------------- picking one episode out of a season pack

from anchor.core.download.providers.subdl import _pick_episode_from_pack

NUMBERED = [f"{i:02d} {t}.srt" for i, t in enumerate(
    ["Summer in the City", "Grand Murder at Central Station", "Zoo York", "Corporate Warriors", "Manhattan Manunt (2)"], 1)]
TAGGED = [f"CSI.NY.S02E{i:02d}.720p.WEB-DL.srt" for i in range(1, 25)]


def test_name_starting_with_the_episode_number():
    """The reported bug: 'CSI NY Season 2' pack named '01 Summer in the City.srt'... gave episode 1 for every episode."""
    assert _pick_episode_from_pack(NUMBERED, "1") == NUMBERED[0]
    assert _pick_episode_from_pack(NUMBERED, "2") == NUMBERED[1]
    assert _pick_episode_from_pack(NUMBERED, "3") == NUMBERED[2]
    assert _pick_episode_from_pack(NUMBERED, "5") == NUMBERED[4]


def test_season_number_is_not_taken_for_the_episode():
    """S02E01 contains '02': episode 2 must not pick the first file."""
    assert _pick_episode_from_pack(TAGGED, "2", "2") == "CSI.NY.S02E02.720p.WEB-DL.srt"
    assert _pick_episode_from_pack(TAGGED, "10", "2") == "CSI.NY.S02E10.720p.WEB-DL.srt"
    assert _pick_episode_from_pack(TAGGED, "1", "2") == "CSI.NY.S02E01.720p.WEB-DL.srt"


def test_other_naming_styles():
    assert _pick_episode_from_pack(["Show 2x01.srt", "Show 2x02.srt", "Show 2x03.srt"], "3", "2") == "Show 2x03.srt"
    assert _pick_episode_from_pack(["Show.E01.srt", "Show.E02.srt"], "2") == "Show.E02.srt"


def test_a_pack_with_two_seasons_picks_the_right_one():
    names = ["Show.S01E02.srt", "Show.S02E02.srt"]
    assert _pick_episode_from_pack(names, "2", "2") == "Show.S02E02.srt"


def test_unknown_episode_falls_back_to_the_first_file():
    assert _pick_episode_from_pack(NUMBERED, "9") == NUMBERED[0]
