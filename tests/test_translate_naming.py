import pytest

from anchor.core.translate.translate import translated_stem


@pytest.mark.parametrize("stem, source, target, expected", [
    ("Movie.pt", "pt", "es", "Movie.es.ai"),                  # the language token is replaced
    ("Movie.es", "es", "pt", "Movie.pt.ai"),
    ("Movie", "pt", "es", "Movie.es.ai"),                     # no token: appended, still marked .ai (the reported bug)
    ("Movie.por", "pt", "es", "Movie.es.ai"),                 # 3-letter code
    ("Movie.pt-BR", "pt", "es", "Movie.es.ai"),               # region tag
    ("Movie.pt.hi", "pt", "es", "Movie.es.ai.hi"),            # trailing tags kept
    ("Movie.pt.ai", "pt", "es", "Movie.es.ai"),               # already an AI translation: no second .ai
    ("Movie.en", "pt", "es", "Movie.en.es.ai"),               # the tag is another language than the detected one: kept, appended
    ("Show.est.Cast", "es", "pt", "Show.est.Cast.pt.ai"),     # a substring of a word is not a language token
    ("Show.Pt.Two.pt", "pt", "es", "Show.Pt.Two.es.ai"),      # the last matching token wins
])
def test_translated_name(stem, source, target, expected):
    assert translated_stem(stem, source, target) == expected
