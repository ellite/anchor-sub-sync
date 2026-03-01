import re
import requests
import zipfile
import io
import unicodedata
from pathlib import Path
from anchor import __version__

SUBDL_SEARCH_URL = "https://api.subdl.com/api/v1/subtitles"
SUBDL_DOWNLOAD_BASE = "https://dl.subdl.com"
USER_AGENT = "Anchor-Sub-Sync " + __version__

def _normalize_title(title: str) -> str:
    t = re.sub(r'[\._]', ' ', title.lower())
    return re.sub(r'[^\w\s]', '', t).strip()

def sanitize_filename(text: str) -> str:
    """Crushes weird Unicode/Fullwidth fonts back to standard ASCII."""
    if not text:
        return ""
    # NFKC converts compatibility characters (like ＷＥＢ) to their standard equivalents (WEB)
    return unicodedata.normalize('NFKC', text)

def _pick_best_show(results: list, parsed_data: dict) -> dict | None:
    """
    Picks the best show from SubDL's disambiguation results list.
    Prefers exact title + year match, then title only, then first result.
    """
    target_title = _normalize_title(parsed_data.get("title", ""))
    target_year = str(parsed_data.get("year", ""))

    # Pass 1: exact title + year
    for show in results:
        show_title = _normalize_title(show.get("name", ""))
        show_year = str(show.get("year") or "")
        if show_title == target_title and (not target_year or show_year == target_year):
            return show

    # Pass 2: title only
    for show in results:
        if _normalize_title(show.get("name", "")) == target_title:
            return show

    # Pass 3: last resort
    return results[0] if results else None


# SubDL sometimes returns full language names instead of 2-letter codes
_LANG_NAME_TO_CODE = {
    "english": "en", "portuguese": "pt", "brazilian portuguese": "pt-BR",
    "spanish": "es", "french": "fr", "german": "de", "italian": "it",
    "dutch": "nl", "russian": "ru", "arabic": "ar", "chinese": "zh",
    "japanese": "ja", "korean": "ko", "turkish": "tr", "polish": "pl",
    "swedish": "sv", "norwegian": "no", "danish": "da", "finnish": "fi",
    "czech": "cs", "hungarian": "hu", "romanian": "ro", "greek": "el",
    "hebrew": "he", "croatian": "hr", "serbian": "sr", "slovak": "sk",
    "slovenian": "sl", "bulgarian": "bg", "ukrainian": "uk",
}

def _normalize_lang_code(raw: str) -> str:
    """Converts 'English' or 'EN' to 'en' for pipeline consistency."""
    cleaned = raw.strip().lower()
    if cleaned == "br":
        return "pt-br"
    if len(cleaned) == 2:
        return cleaned
    return _LANG_NAME_TO_CODE.get(cleaned, cleaned[:2])


def _build_params(parsed_data: dict, langs_upper: str, api_key: str) -> dict:
    """Builds the base query params shared between first and retry requests."""
    content_type = "tv" if parsed_data.get("season") else "movie"

    params = {
        "api_key": api_key,
        "languages": langs_upper,
        "type": content_type,
        "releases": 1,
        "subs_per_page": 30,
    }

    if parsed_data.get("year"):
        params["year"] = parsed_data.get("year")
    if parsed_data.get("season"):
        params["season_number"] = parsed_data.get("season")
    if parsed_data.get("episode"):
        params["episode_number"] = parsed_data.get("episode")

    return params


def _normalize_results(raw_subs: list) -> list:
    """Converts raw SubDL subtitle dicts into the normalized pipeline shape."""
    normalized = []

    for sub in raw_subs:
        url_path = sub.get("url", "")
        if not url_path:
            continue

        releases = sub.get("releases", [])
        if isinstance(releases, str):
            releases = [releases] if releases else []

        release_name = sub.get("release_name", "")
        if release_name and release_name not in releases:
            releases.insert(0, release_name)

        filename = (releases[0] + ".srt") if releases else (sub.get("name", "Unknown") + ".srt")

        lang_code = _normalize_lang_code(sub.get("language", "en"))
        is_hi = bool(sub.get("hi", 0))

        normalized.append({
            "provider": "SubDL",
            "id": url_path,           # Full path used directly for download
            "filename": filename,
            "language": lang_code,
            "releases": releases,
            "download_url": f"{SUBDL_DOWNLOAD_BASE}{url_path}",
            "hash_match": False,      # SubDL has no hash search
            "score": 0,
            "_is_hi": is_hi,          # Direct HI flag, avoids regex in scoring
        })

    return normalized

def search_subdl(parsed_data: dict, language: str, api_key: str) -> list:
    """
    Searches SubDL using text metadata.
    `language` should be a single language code (e.g. "en" or "PT").
    Handles SubDL's disambiguation flow automatically:
        - First request searches by film_name
        - If SubDL returns shows but no subtitles, picks the best sd_id and retries (single language)
    Returns a normalized list of subtitle dicts matching the OpenSubtitles shape.
    """
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }

    langs_upper = ",".join(l.strip().upper() for l in language.split(",") if l.strip())
    params = _build_params(parsed_data, langs_upper, api_key)
    params["film_name"] = parsed_data.get("title", "")

    try:
        response = requests.get(
            SUBDL_SEARCH_URL,
            headers=headers,
            params=params,
            timeout=10,
        )

        if response.status_code != 200:
            return []

        data = response.json()

        if not data.get("status"):
            return []

        raw_subs = data.get("subtitles", [])

        if not raw_subs and data.get("results"):
            best_match = _pick_best_show(data["results"], parsed_data)

            if best_match:
                params.pop("film_name", None)
                params.pop("languages", None)
                params["sd_id"] = best_match["sd_id"]

                # Loop per-language since sd_id endpoint only accepts one at a time
                for lang in langs_upper.split(","):
                    params["language"] = lang.strip()  # Note: singular "language" key

                    try:
                        response2 = requests.get(
                            SUBDL_SEARCH_URL,
                            headers=headers,
                            params=params,
                            timeout=10,
                        )

                        if response2.status_code == 200:
                            data2 = response2.json()
                            if data2.get("status"):
                                raw_subs += data2.get("subtitles", [])
                    except Exception as e:
                        print(f"   [red]❌ SubDL retry [{lang}] Exception: {e}[/red]")

    except Exception as e:
        print(f"   [red]❌ SubDL Search Exception: {e}[/red]")
        return []
    
    normalized = _normalize_results(raw_subs)
    requested_langs = [l.strip().lower() for l in language.split(',')]
    filtered_results = []
    
    for sub in normalized:
        sub_lang = sub["language"].lower()
        
        if sub_lang in requested_langs:
            sub["filename"] = sanitize_filename(sub.get("filename", ""))
            if "releases" in sub:
                sub["releases"] = [sanitize_filename(r) for r in sub["releases"]]
                
            filtered_results.append(sub)
            
    return filtered_results


def download_subdl(url_path: str, target_video_path: Path, custom_suffix: str = ".srt", episode: str = None) -> bool:
    """
    Downloads a subtitle ZIP from SubDL, extracts the first .srt inside,
    and saves it next to the video with the given custom_suffix.
    url_path is the value stored in sub["id"], e.g. "/subtitle/3197651-3213944.zip"
    """
    headers = {
        "User-Agent": USER_AGENT,
    }

    download_url = f"{SUBDL_DOWNLOAD_BASE}{url_path}"

    try:
        response = requests.get(download_url, headers=headers, timeout=15)

        if response.status_code != 200:
            return False

        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            srt_names = [
                name for name in zf.namelist()
                if name.lower().endswith((".srt", ".sub"))
            ]
            if not srt_names:
                return False

            # --- EPISODE PICKER ---
            chosen = _pick_episode_from_pack(srt_names, episode)
            srt_content = zf.read(chosen)

        final_path = target_video_path.with_suffix(custom_suffix)
        with open(final_path, "wb") as f:
            f.write(srt_content)

        return True

    except Exception as e:
        print(f"   [red]❌ SubDL Download Exception: {e}[/red]")

    return False


def _pick_episode_from_pack(srt_names: list, episode: str) -> str:
    """
    Given a list of .srt filenames from a ZIP, returns the best match
    for the target episode number. Falls back to srt_names[0] if no match.
    """
    if not episode or len(srt_names) == 1:
        return srt_names[0]

    ep_num = str(episode).zfill(2)  # "1" -> "01"

    # Match patterns like S01E01, E01, _01_, .01.
    ep_pattern = re.compile(
        rf'[Ee]{ep_num}|[^0-9]{ep_num}[^0-9]',
    )

    for name in srt_names:
        if ep_pattern.search(name):
            return name

    # Fallback
    return srt_names[0]
