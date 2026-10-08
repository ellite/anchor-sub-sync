import re
import difflib
from ...utils.parsers import parse_video_filename

def normalize_title(title: str) -> str:
    t = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', title)
    t = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', ' ', t)
    t = re.sub(r'[\._]', ' ', t.lower())
    return re.sub(r'[^\w\s]', '', t).strip()

def strip_articles(title: str) -> str:
    """Removes leading articles for a base comparison."""
    return re.sub(r'^(the|a|an)\s+', '', title).strip()

def _fold_initialisms(tokens_a, tokens_b):
    """Rewrites runs of words in `tokens_b` as the short word of `tokens_a` that is their initials ("new york" -> "ny")."""
    out = list(tokens_b)
    for short in {t for t in tokens_a if 2 <= len(t) <= 4}:
        n = len(short)
        for i in range(len(out) - n + 1):
            if out[i:i + n] != [short] and all(len(w) > 1 for w in out[i:i + n]) and "".join(w[0] for w in out[i:i + n]) == short:
                out[i:i + n] = [short]
                break
    return out


def titles_match(a: str, b: str) -> bool:
    """Same show under different names: articles, near spellings, and initialisms ("CSI NY" = "CSI New York")."""
    a, b = normalize_title(a), normalize_title(b)
    base_a, base_b = strip_articles(a), strip_articles(b)
    if a == b or base_a == base_b:
        return True
    if not (base_a and base_b):
        return False
    if difflib.SequenceMatcher(None, base_a, base_b).ratio() > 0.80:
        return True
    ta, tb = base_a.split(), base_b.split()
    return _fold_initialisms(ta, tb) == ta or _fold_initialisms(tb, ta) == tb


def calculate_score(target_parsed: dict, sub_dict: dict, target_langs_list: list, prefer_sdh: bool = False, prefer_forced: bool = False) -> int:
    sub_parsed = parse_video_filename(sub_dict.get("filename", ""))
    
    # --- 1. Is it the correct show/movie? ---
    # A. Check the title
    is_correct_show = titles_match(target_parsed.get("title", ""), sub_parsed.get("title", ""))
            
    # B. Strict TV Check
    t_season = target_parsed.get("season")
    t_ep = target_parsed.get("episode")
    s_season = sub_parsed.get("season")
    s_ep = sub_parsed.get("episode")

    if t_season and s_season and t_season != s_season:
        is_correct_show = False
    elif t_ep and s_ep and t_ep != s_ep:
        is_correct_show = False

    # C. Strict Year Check (Crucial for Movies)
    t_year = target_parsed.get("year")
    s_year = sub_parsed.get("year")
    
    if t_year and s_year and t_year != s_year:
        is_correct_show = False

    # IF IT IS THE WRONG SHOW, INSTANTLY FAIL IT!
    if not is_correct_show:
        return -100   

    # --- 2. BASE SCORE ---
    is_hash_match = sub_dict.get("hash_match")
    score = 100 if is_hash_match else 10
        
    # --- 3. ADDITIVE CRITERIA ---
    combined_text = (sub_dict.get("filename", "") + " " + " ".join(sub_dict.get("releases", []))).lower()

    # Edition / Cut Match (EXTENDED, UNRATED, DIRECTOR'S CUT)
    target_raw = target_parsed.get("raw_filename", "").lower()
    editions = ['extended', 'unrated', 'director', 'remastered']
    
    for ed in editions:
        t_has_ed = ed in target_raw
        s_has_ed = ed in combined_text
        
        if t_has_ed and s_has_ed:
            score += 15
        elif t_has_ed != s_has_ed and not is_hash_match:
            # Heavy penalty if one is extended and the other isn't (guaranteed desync)
            # Ignore this penalty ONLY if provider guarantees a perfect video hash match.
            score -= 20

    # Source Match
    p_source = target_parsed.get("source", "").upper()
    s_source = sub_parsed.get("source", "").upper()
    if p_source and s_source:
        if p_source == s_source:
            score += 20  
        elif "WEB" in p_source and "WEB" in s_source:
            score += 15  
            
    # Network Match
    if target_parsed.get("network") and target_parsed.get("network") == sub_parsed.get("network"):
        score += 20
            
    # Group Match
    p_group = target_parsed.get("group", "").lower()
    s_group = sub_parsed.get("group", "").lower()
    group_matched = False
    
    if p_group and s_group and p_group == s_group:
        group_matched = True
    elif p_group:
        for release_tag in sub_dict.get("releases", []):
            if p_group in release_tag.lower():
                group_matched = True
                break
                
    if group_matched:
        score += 10
        
    # --- 4. PREFERENCES (+5 for match, -10 for mismatch) ---
    
    # SDH check
    is_sdh = bool(re.search(r'\b(hi|sdh|cc)\b', combined_text)) or sub_dict.get("_is_hi", False)
    if prefer_sdh:
        if is_sdh:
            score += 5
        else:
            score -= 10
    else:
        if not is_sdh:
            score += 5
        else:
            score -= 10

    # Forced check
    is_forced = bool(re.search(r'\b(forced|foreign)\b', combined_text))
    if prefer_forced:
        if is_forced:
            score += 5
        else:
            score -= 10
    else:
        if not is_forced:
            score += 5
        else:
            score -= 10

    return score