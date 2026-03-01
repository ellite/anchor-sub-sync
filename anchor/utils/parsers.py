import re
from pathlib import Path

def parse_video_filename(file_path: Path | str) -> dict:
    """
    Rips a standard scene release filename into its core scoring components.
    """
    filename = Path(file_path).stem if isinstance(file_path, Path) else file_path
    if isinstance(file_path, str) and "." in file_path:
        filename = file_path.rsplit('.', 1)[0]
        
    parsed = {
        "title": "", 
        "year": "", 
        "season": "",
        "episode": "",
        "source": "", 
        "network": "", 
        "resolution": "", 
        "group": ""
    }

    # --- 1. TITLE, YEAR, SEASON, EPISODE ---
    # First try S01E05 format
    tv_match = re.search(r'(?i)S(?P<season>\d{1,2})E(?P<episode>\d{1,2})', filename)
    if not tv_match:
        # Fallback to 01x05 format
        tv_match = re.search(r'(?i)\b(?P<season>\d{1,2})x(?P<episode>\d{1,2})\b', filename)

    if tv_match:
        parsed["season"] = tv_match.group('season').lstrip('0') or '0'
        parsed["episode"] = tv_match.group('episode').lstrip('0') or '0'
        title_raw = filename[:tv_match.start()]
        parsed["title"] = re.sub(r'[\._\-]', ' ', title_raw).strip()

    else:
        # --- Season-only pack (S01 with no episode) ---
        season_match = re.search(r'(?i)\bS(?P<season>\d{1,2})\b', filename)
        if season_match:
            parsed["season"] = season_match.group('season').lstrip('0') or '0'
            # Season pack
            title_raw = filename[:season_match.start()]
            parsed["title"] = re.sub(r'[\._\-]', ' ', title_raw).strip()

        else:
            year_match = re.search(r'(?P<year>(?:19|20)\d{2})', filename)
            if year_match:
                parsed["year"] = year_match.group('year')
                title_raw = filename[:year_match.start()]
                parsed["title"] = re.sub(r'[\._\-]', ' ', title_raw).strip()
            else:
                parsed["title"] = re.sub(r'[\._\-]', ' ', filename).strip()
            
    # --- 2. SOURCE ---
    source_match = re.search(r'(?i)(WEB-DL|WEBRip|WEB|BluRay|BDRip|BRRip|HDTV|DVDRip)', filename)
    if source_match:
        parsed["source"] = source_match.group(1).upper()
        
    # --- 3. NETWORK ---
    network_match = re.search(r'(?i)\b(AMZN|NF|HMAX|MAX|DSNP|ATVP|HULU|PCOK|PEACOCK|APPLE|ATV|VIU|VIKI)\b', filename)
    if network_match:
        parsed["network"] = network_match.group(1).upper()
        
    # --- 4. RESOLUTION ---
    res_match = re.search(r'(?i)(2160p|1080p|720p|480p|4K)', filename)
    if res_match:
        parsed["resolution"] = res_match.group(1).lower()
        
    # --- 5. RELEASE GROUP ---
    group_match = re.search(r'-([a-zA-Z0-9]+)$', filename)
    if group_match:
        parsed["group"] = group_match.group(1)

    parsed["raw_filename"] = filename    
        
    return parsed