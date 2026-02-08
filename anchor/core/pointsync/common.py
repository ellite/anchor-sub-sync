from ...utils.formatting import clean_text

def get_filtered_lines(subs, section="all", limit=100):
    """
    Extracts clean dialogue lines with timestamps.
    Args:
        section: "start", "end", or "all"
        limit: Max lines to return for start/end sections
    Returns: list of tuples (original_index, start_ms, text_preview)
    """
    lines = []
    for i, event in enumerate(subs):
        if event.is_comment: continue
        
        text = clean_text(event.text)
        
        # Heuristics to skip bad sync points (music, short words)
        if not text or len(text) < 3: continue
        if text.startswith('♪') or text.startswith('['): continue
        
        lines.append((i, event.start, text))

    if section == "start":
        return lines[:limit]
    elif section == "end":
        return lines[-limit:]
    
    return lines

def apply_linear_correction(target_sub, p1_target, p1_ref, p2_target, p2_ref):
    """
    Calculates m (slope) and c (offset) and applies them to target_sub in place.
    Returns: (m, c)
    """
    # Prevent division by zero
    if p2_target == p1_target:
        raise ValueError("Start and End points on target are identical.")

    m = (p2_ref - p1_ref) / (p2_target - p1_target)
    c = p1_ref - m * p1_target

    for line in target_sub:
        line.start = int(m * line.start + c)
        line.end = int(m * line.end + c)
        
    return m, c