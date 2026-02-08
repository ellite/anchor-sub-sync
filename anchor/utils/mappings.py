ISO_639_MAPPING = {
    'eng': 'en', 'spa': 'es', 'fra': 'fr', 'deu': 'de', 'ita': 'it',
    'por': 'pt', 'jpn': 'ja', 'zho': 'zh', 'chi': 'zh', 'rus': 'ru',
    'kor': 'ko', 'nld': 'nl', 'swe': 'sv', 'nor': 'no', 'dan': 'da',
    'fin': 'fi', 'tur': 'tr', 'pol': 'pl', 'ukr': 'uk', 'ara': 'ar',
    'hin': 'hi', 'vie': 'vi', 'tha': 'th', 'ind': 'id'
}

def get_language_code_for_nllb(iso_code: str) -> str:
    """
    Maps ISO 639-1 (2-letter) codes to NLLB/FLORES-200 codes.
    Default fallback is 'eng_Latn' if the language is not found.
    """
    code = iso_code.strip().lower()

    iso_to_nllb = {
        "af": "afr_Latn", "am": "amh_Ethi", "ar": "arb_Arab", "az": "azj_Latn",
        "be": "bel_Cyrl", "bg": "bul_Cyrl", "bn": "ben_Beng", "bs": "bos_Latn",
        "ca": "cat_Latn", "cs": "ces_Latn", "cy": "cym_Latn", "da": "dan_Latn",
        "de": "deu_Latn", "el": "ell_Grek", "en": "eng_Latn", "es": "spa_Latn",
        "et": "est_Latn", "eu": "eus_Latn", "fa": "pes_Arab", "fi": "fin_Latn",
        "fr": "fra_Latn", "gl": "glg_Latn", "gu": "guj_Gujr", "he": "heb_Hebr",
        "hi": "hin_Deva", "hr": "hrv_Latn", "hu": "hun_Latn", "hy": "hye_Armn",
        "id": "ind_Latn", "is": "isl_Latn", "it": "ita_Latn", "ja": "jpn_Jpan",
        "jv": "jav_Latn", "ka": "kat_Geor", "kk": "kaz_Cyrl", "km": "khm_Khmr",
        "kn": "kan_Knda", "ko": "kor_Hang", "lo": "lao_Laoo", "lt": "lit_Latn",
        "lv": "lvs_Latn", "mk": "mkd_Cyrl", "ml": "mal_Mlym", "mn": "khk_Cyrl",
        "mr": "mar_Deva", "ms": "zsm_Latn", "my": "mya_Mymr", "nl": "nld_Latn",
        "no": "nob_Latn", "or": "ory_Orya", "pa": "pan_Guru", "pl": "pol_Latn",
        "ps": "pbt_Arab", "pt": "por_Latn", "ro": "ron_Latn", "ru": "rus_Cyrl",
        "sd": "snd_Arab", "si": "sin_Sinh", "sk": "slk_Latn", "sl": "slv_Latn",
        "so": "som_Latn", "sq": "als_Latn", "sr": "srp_Cyrl", "sv": "swe_Latn",
        "sw": "swh_Latn", "ta": "tam_Taml", "te": "tel_Telu", "th": "tha_Thai",
        "tl": "tgl_Latn", "tr": "tur_Latn", "uk": "ukr_Cyrl", "ur": "urd_Arab",
        "uz": "uzn_Latn", "vi": "vie_Latn", "yo": "yor_Latn", "zh": "zho_Hans",
        "zh-cn": "zho_Hans", "zh-tw": "zho_Hant", "zu": "zul_Latn"
    }

    return iso_to_nllb.get(code, "eng_Latn")