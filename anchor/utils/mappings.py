ISO_639_MAPPING = {
    # Most Common
    'eng': 'en', 'spa': 'es', 'fra': 'fr', 'fre': 'fr', 'deu': 'de', 'ger': 'de',
    'ita': 'it', 'por': 'pt', 'jpn': 'ja', 'zho': 'zh', 'chi': 'zh', 'rus': 'ru',
    'kor': 'ko', 'nld': 'nl', 'dut': 'nl', 'swe': 'sv', 'nor': 'no', 'dan': 'da',
    'fin': 'fi', 'tur': 'tr', 'pol': 'pl', 'ukr': 'uk', 'ara': 'ar', 'hin': 'hi',
    'vie': 'vi', 'tha': 'th', 'ind': 'id',

    # Expanded / NLLB Coverage
    'afr': 'af', 'amh': 'am', 'aze': 'az', 'bel': 'be', 'bul': 'bg', 'ben': 'bn',
    'bos': 'bs', 'cat': 'ca', 'ces': 'cs', 'cze': 'cs', 'cym': 'cy', 'wel': 'cy',
    'ell': 'el', 'gre': 'el', 'est': 'et', 'eus': 'eu', 'baq': 'eu', 'fas': 'fa',
    'per': 'fa', 'glg': 'gl', 'guj': 'gu', 'heb': 'he', 'hrv': 'hr', 'hun': 'hu',
    'hye': 'hy', 'arm': 'hy', 'isl': 'is', 'ice': 'is', 'jav': 'jv', 'kat': 'ka',
    'geo': 'ka', 'kaz': 'kk', 'khm': 'km', 'kan': 'kn', 'lao': 'lo', 'lit': 'lt',
    'lav': 'lv', 'mkd': 'mk', 'mac': 'mk', 'mal': 'ml', 'mon': 'mn', 'mar': 'mr',
    'msa': 'ms', 'may': 'ms', 'mya': 'my', 'bur': 'my', 'ori': 'or', 'pan': 'pa',
    'pus': 'ps', 'ron': 'ro', 'rum': 'ro', 'snd': 'sd', 'sin': 'si', 'slk': 'sk',
    'slo': 'sk', 'slv': 'sl', 'som': 'so', 'sqi': 'sq', 'alb': 'sq', 'srp': 'sr',
    'swa': 'sw', 'tam': 'ta', 'tel': 'te', 'tgl': 'tl', 'urd': 'ur', 'uzb': 'uz',
    'yor': 'yo', 'zul': 'zu'
}

def normalize_language_code(lang_code: str) -> str:
    """
    Converts 3-letter ISO 639-2 container codes to 2-letter ISO 639-1 codes.
    Returns the original code if not found in the mapping.
    """
    if not lang_code: return "und"
    lang_code = lang_code.lower().strip()
    return ISO_639_MAPPING.get(lang_code, lang_code)

def get_iso_639_2_code(lang_code: str) -> str:
    """Converts 2-letter ISO 639-1 codes to 3-letter ISO 639-2 container codes."""
    if not lang_code: return "und"
    lang_code = lang_code.lower().strip()
    
    # If it's already a 3-letter code, return it
    if len(lang_code) == 3: return lang_code
    
    # Safe reverse lookup for common languages
    reverse_map = {
        'en': 'eng', 'es': 'spa', 'fr': 'fra', 'de': 'deu', 'it': 'ita',
        'pt': 'por', 'ja': 'jpn', 'zh': 'zho', 'ru': 'rus', 'ko': 'kor',
        'nl': 'nld', 'sv': 'swe', 'no': 'nor', 'da': 'dan', 'fi': 'fin',
        'tr': 'tur', 'pl': 'pol', 'uk': 'ukr', 'ar': 'ara', 'hi': 'hin',
        'vi': 'vie', 'th': 'tha', 'id': 'ind', 'cs': 'ces', 'el': 'ell',
        'he': 'heb', 'ro': 'ron', 'sk': 'slk', 'bg': 'bul', 'hr': 'hrv',
        'hu': 'hun', 'sr': 'srp', 'ca': 'cat', 'is': 'isl', 'fa': 'fas',
        'et': 'est', 'lv': 'lav', 'lt': 'lit', 'sl': 'slv', 'ur': 'urd'
    }
    
    return reverse_map.get(lang_code, "und")    

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