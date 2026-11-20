import pytest

from anchor.utils.languages import check_whisper_language, align_model_missing


@pytest.mark.parametrize("given, expected", [("en", "en"), ("EN", "en"), (" sr ", "sr"), ("yue", "yue"), ("haw", "haw")])
def test_a_single_whisper_language_is_accepted_and_normalised(given, expected):
    assert check_whisper_language(given) == (expected, None)


@pytest.mark.parametrize("given", ["en,fr,uk,ru", "en fr", "en;fr", "en/fr", "en, fr"])
def test_a_list_of_languages_is_refused_with_the_reason(given):
    code, message = check_whisper_language(given)
    assert code is None and "one language per file" in message and given in message


@pytest.mark.parametrize("given", ["xx", "english", "pt-br", "eng", "123"])
def test_an_unknown_code_is_refused_and_the_valid_ones_are_listed(given):
    code, message = check_whisper_language(given)
    assert code is None and "not a language Whisper knows" in message and "yue" in message


@pytest.mark.parametrize("given", ["", "   ", None])
def test_an_empty_language_is_refused(given):
    assert check_whisper_language(given)[0] is None


def test_a_language_without_a_default_align_model_is_reported_missing():
    defaults = {"en", "pt", "fr"}
    assert align_model_missing("sr", defaults) is True
    assert align_model_missing("SR", defaults) is True
    assert align_model_missing("en", defaults) is False
    assert align_model_missing("EN", defaults) is False


def test_a_custom_align_model_makes_any_language_usable():
    assert align_model_missing("sr", {"en"}, "dnikolic/wav2vec2-xlsr-530-serbian-colab") is False
