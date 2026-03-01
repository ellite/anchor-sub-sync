import requests
import urllib.parse
from pathlib import Path
from anchor import __version__

GESTDOWN_BASE_URL = "https://api.gestdown.info"
USER_AGENT = "Anchor-Sub-Sync " + __version__

def search_addic7ed(parsed_data: dict, file_hash: str, language: str) -> list:
    title = parsed_data.get("title")
    season = parsed_data.get("season")
    episode = parsed_data.get("episode")

    # Must be a TV show
    if not title or season is None or episode is None:
        return []

    # THE INTEGER FIX: Force them to ints so the API gets /2/8, not /02/08
    try:
        s_num = int(season)
        e_num = int(episode)
    except (ValueError, TypeError):
        return []

    clean_title = title.replace(".", " ").strip().lower()
    safe_title = urllib.parse.quote(clean_title)
    
    langs = [l.strip() for l in language.split(",") if l.strip()]
    all_results = []
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json"
    }

    for lang in langs:
        url = f"{GESTDOWN_BASE_URL}/subtitles/find/{lang}/{safe_title}/{s_num}/{e_num}"

        try:
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                hits = data.get("matchingSubtitles", [])

                for sub in hits:
                    sub_id = sub.get("subtitleId")
                    if not sub_id:
                        continue
                        
                    release_tag = sub.get("version", "Unknown")
                    
                    # Create a nice, clean filename based on the parser data
                    display_title = clean_title.title().replace(" ", ".")
                    filename = f"{display_title}.S{s_num:02d}E{e_num:02d}.{release_tag}.srt"
                    
                    # Grab everything to feed the scoring engine
                    releases = [release_tag]
                    if sub.get("release"):
                        releases.append(sub.get("release"))
                    releases.extend(sub.get("qualities", []))

                    all_results.append({
                        "provider": "Addic7ed",
                        "id": sub_id,
                        "filename": filename,
                        "language": lang,
                        "releases": releases,
                        "download_url": f"{GESTDOWN_BASE_URL}{sub.get('downloadUri')}",
                        "hash_match": False,
                        "_is_hi": sub.get("hearingImpaired", False),
                        "score": 0,
                    })
                    
            elif response.status_code == 429:
                print(f"   [yellow]⚠️ Addic7ed Rate Limit Hit (429) for {lang}. Skipping...[/yellow]")
            elif response.status_code == 404:
                # API returns 404 if no subtitles are found for that exact show/season combo
                pass 
            else:
                print(f"   [red]❌ Addic7ed API Error {response.status_code} for {lang}[/red]")
                
        except Exception as e:
            print(f"   [red]❌ Addic7ed Search Exception: {e}[/red]")

    return all_results

def download_addic7ed(sub_id: str, target_video_path: Path, custom_suffix: str = ".srt") -> bool:
    """
    Hits the Gestdown download endpoint using the subtitle ID.
    Defensively handles both raw SRT text streams and JSON-wrapped responses.
    """
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json"
    }

    # Construct the full download URL using the ID!
    download_url = f"{GESTDOWN_BASE_URL}/subtitles/download/{sub_id}"

    try:
        response = requests.get(download_url, headers=headers, timeout=15)

        if response.status_code == 200:
            final_path = target_video_path.with_suffix(custom_suffix)
            content_type = response.headers.get("Content-Type", "").lower()
            
            with open(final_path, "wb") as f:
                if "application/json" in content_type:
                    # If Gestdown honors the JSON header, the SRT text is inside a dictionary
                    data = response.json()
                    srt_text = data.get("content", data.get("text", ""))
                    f.write(srt_text.encode("utf-8"))
                else:
                    # If it ignores the header and dumps the raw SRT bytes
                    f.write(response.content)
                    
            return True
            
        else:
            print(f"   [red]❌ Addic7ed Download Error: API returned {response.status_code}[/red]")
            
    except Exception as e:
        print(f"   [red]❌ Addic7ed Download Exception: {e}[/red]")

    return False