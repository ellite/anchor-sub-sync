import requests
from pathlib import Path
from anchor import __version__


OS_BASE_URL = "https://api.opensubtitles.com/api/v1"
USER_AGENT = "Anchor-Sub-Sync " + __version__

def get_os_token(api_key: str, username: str, password: str) -> str:
    """Authenticates with OpenSubtitles to get a Bearer token for downloads."""
    headers = {
        "Api-Key": api_key,
        "User-Agent": USER_AGENT,
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    payload = {"username": username, "password": password}
    
    try:
        response = requests.post(f"{OS_BASE_URL}/login", headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            return response.json().get("token", "")
    except Exception as e:
        print(f"   [red]❌ OpenSubtitles Login Failed: {e}[/red]")
    return ""

def search_opensubtitles(parsed_data: dict, file_hash: str, language: str, api_key: str) -> list:
    """Searches OpenSubtitles using both Hash and Text Metadata."""
    headers = {
        "Api-Key": api_key,
        "User-Agent": USER_AGENT,
        "Accept": "application/json"
    }
    
    results = []
    hash_match_found = False
    
    # --- ATTEMPT 1: EXACT VIDEO HASH ---
    if file_hash:
        params_hash = {"languages": language, "moviehash": file_hash}
        try:
            response = requests.get(f"{OS_BASE_URL}/subtitles", headers=headers, params=params_hash, timeout=10)
            if response.status_code == 200:
                results = response.json().get("data", [])
                if results:
                    hash_match_found = True  
        except Exception:
            pass

    # --- ATTEMPT 2: TEXT FALLBACK ---
    if not results:
        params_text = {
            "languages": language,
            "query": parsed_data.get("title", "")
        }
        if parsed_data.get("year"): params_text["year"] = parsed_data.get("year")
        if parsed_data.get("season"): params_text["season_number"] = parsed_data.get("season")
        if parsed_data.get("episode"): params_text["episode_number"] = parsed_data.get("episode")
        
        try:
            response = requests.get(f"{OS_BASE_URL}/subtitles", headers=headers, params=params_text, timeout=10)
            if response.status_code == 200:
                results = response.json().get("data", [])
        except Exception:
            pass

    # --- NORMALIZE RESULTS ---
    normalized_results = []
    for sub in results:
        attrs = sub.get("attributes", {})
        
        files = attrs.get("files", [])
        if not files:
            continue
            
        file_info = files[0]
        file_id = file_info.get("file_id")
        filename = file_info.get("file_name", "Unknown.srt")
        
        # OS provides the exact release string natively!
        release_name = attrs.get("release", "")
        releases = [release_name] if release_name else []

        normalized_results.append({
            "provider": "OpenSubs",
            "id": file_id,
            "filename": filename,
            "language": attrs.get("language", "en"),
            "releases": releases,
            "download_url": "",
            "hash_match": hash_match_found,
            "score": 0 
        })
        
    return normalized_results

def download_opensubtitles(file_id: int, target_video_path: Path, api_key: str, token: str, custom_suffix: str = ".srt") -> bool:
    """Requests a download link from OS, then downloads and saves the .srt file."""
    headers = {
        "Api-Key": api_key,
        "Authorization": f"Bearer {token}",
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    
    payload = {"file_id": file_id}
    
    try:
        link_resp = requests.post(f"{OS_BASE_URL}/download", headers=headers, json=payload, timeout=10)
        
        if link_resp.status_code != 200:
            return False
            
        download_url = link_resp.json().get("link")
        if not download_url:
            return False
            
        file_resp = requests.get(download_url, timeout=15)
        if file_resp.status_code == 200:
            
            # Apply dynamically built suffix (e.g., .en.hi.srt)
            final_path = target_video_path.with_suffix(custom_suffix) 
            
            with open(final_path, "wb") as f:
                f.write(file_resp.content)
            return True
            
    except Exception as e:
        print(f"   [red]❌ OS Download Exception: {e}[/red]")
        
    return False