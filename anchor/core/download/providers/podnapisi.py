import requests
import zipfile
import io
from pathlib import Path
from anchor import __version__

PODNAPISI_BASE_URL = "https://www.podnapisi.net"
USER_AGENT = "Anchor-Sub-Sync " + __version__

# Podnapisi uses 2-letter language codes (ISO 639-1), same as OpenSubtitles
# e.g. "en", "pt", "es", "fr", "de"

def search_podnapisi(parsed_data: dict, file_hash: str, language: str, ) -> list:
    """
    Searches Podnapisi using both Hash and Text Metadata.
    Language param is a comma-separated string like "en,pt,es".
    Returns a normalized list of subtitle dicts, matching the OpenSubtitles shape.
    """
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }

    # Podnapisi takes one language per request; split and run per-lang
    langs = [l.strip() for l in language.split(",") if l.strip()]

    all_results = []
    hash_match_found = False

    for lang in langs:
        results_for_lang = []
        lang_hash_match = False

        # --- ATTEMPT 1: HASH SEARCH ---
        if file_hash:
            params_hash = {
                "moviehash": file_hash,
                "language": lang,
            }
            try:
                response = requests.get(
                    f"{PODNAPISI_BASE_URL}/subtitles/search/",
                    headers=headers,
                    params=params_hash,
                    timeout=10,
                )
                if response.status_code == 200:
                    data = response.json()
                    hits = data.get("subtitles", [])
                    if hits:
                        results_for_lang = hits
                        lang_hash_match = True
            except Exception:
                pass

        # --- ATTEMPT 2: TEXT FALLBACK ---
        if not results_for_lang:
            params_text = {
                "keywords": parsed_data.get("title", ""),
                "language": lang,
            }
            if parsed_data.get("year"):
                params_text["year"] = parsed_data.get("year")
            if parsed_data.get("season"):
                params_text["season"] = parsed_data.get("season")
            if parsed_data.get("episode"):
                params_text["episode"] = parsed_data.get("episode")

            try:
                response = requests.get(
                    f"{PODNAPISI_BASE_URL}/subtitles/search/",
                    headers=headers,
                    params=params_text,
                    timeout=10,
                )
                if response.status_code == 200:
                    data = response.json()
                    results_for_lang = data.get("subtitles", [])
            except Exception:
                pass

        # --- NORMALIZE RESULTS ---
        for sub in results_for_lang:
            sub_id = sub.get("id") or sub.get("pid")
            if not sub_id:
                continue

            # Podnapisi gives a "releases" list directly
            releases = sub.get("releases", [])

            # Build a display filename: prefer first release tag, else use title
            if releases:
                filename = releases[0] + ".srt"
            else:
                title = sub.get("title", "Unknown")
                filename = f"{title}.srt"

            all_results.append({
                "provider": "Podnapisi",
                "id": sub_id,
                "filename": filename,
                "language": sub.get("language", lang),
                "releases": releases,
                "download_url": f"{PODNAPISI_BASE_URL}/subtitles/{sub_id}/download",
                "hash_match": lang_hash_match,
                "score": 0,
            })

    return all_results


def download_podnapisi(sub_id: str, target_video_path: Path, custom_suffix: str = ".srt") -> bool:
    """
    Downloads a subtitle ZIP from Podnapisi, extracts the first .srt inside,
    and saves it next to the video with the given custom_suffix.
    """
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/zip, application/octet-stream, */*",
    }

    download_url = f"{PODNAPISI_BASE_URL}/subtitles/{sub_id}/download"

    try:
        response = requests.get(download_url, headers=headers, timeout=15)

        if response.status_code != 200:
            return False

        # The download is always a .zip containing one or more subtitle files
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            # Find the first .srt, .sub, or .ass inside the archive
            srt_names = [
                name for name in zf.namelist()
                if name.lower().endswith((".srt", ".sub", ".ass"))
            ]
            if not srt_names:
                return False

            srt_content = zf.read(srt_names[0])

        final_path = target_video_path.with_suffix(custom_suffix)
        with open(final_path, "wb") as f:
            f.write(srt_content)

        return True

    except Exception as e:
        print(f"   [red]❌ Podnapisi Download Exception: {e}[/red]")

    return False
