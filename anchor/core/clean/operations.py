import re

def remove_sdh(sub):
    """
    Removes SDH elements based on heuristics.
    """
    bracket_pattern = re.compile(r'\[.*?\]|\(.*?\)|\{(?!\\).*?\}|\?.*?\?')
    speaker_pattern = re.compile(r'^((?:<[^>]+>|\{\\[^}]+\})*)\s*[A-Z0-9\s\.\-]+:\s*')
    tag_pattern = re.compile(r'<[^>]+>|\{\\[^}]+\}')
    
    # --- MUSIC PATTERN SETUP ---
    music_symbols = ['♪', '♫', '♩', '♬', '♭', '♮', '♯', '𝄞', '𝄢', '#']
    music_class = f"[{''.join(music_symbols)}]"
    music_pair_pattern = re.compile(f"{music_class}.*?{music_class}")
    music_trailing_pattern = re.compile(f"{music_class}.*?(?= - |$)")

    # --- PRE-SCAN: Protect ALL-CAPS subtitle files ---
    total_cues = len(sub)
    upper_count = 0
    for cue in sub:
        bare = re.sub(tag_pattern, '', cue.text).strip()
        if bare.isupper() and re.search(r'[A-Z]', bare):
            upper_count += 1
            
    # If more than 30% of the file is purely uppercase, it's not SDH.
    # Disale the uppercase nuke to avoid destroying legitimate ALL-CAPS files
    disable_upper_nuke = total_cues > 0 and (upper_count / total_cues) > 0.30

    valid_cues = []

    for cue in sub:
        original_text = cue.text
        
        # Auto-detect the newline style (pysubs2 uses \N, others use \n)
        nl = r'\N' if r'\N' in original_text else '\n'
        
        cleaned_text = re.sub(bracket_pattern, '', original_text)
        lines = re.split(r'\\N|\n', cleaned_text)
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Strip speaker label but keep leading tags
            line = re.sub(speaker_pattern, r'\1', line).strip()

            # Wipe out paired symbols and their lyrics
            line = re.sub(music_pair_pattern, '', line)
            
            # Wipe out unclosed symbols and trailing lyrics
            line = re.sub(music_trailing_pattern, '', line).strip()
            
            # Check the "bare" text
            bare_text = re.sub(tag_pattern, '', line).strip()
            
            # Only drop uppercase lines if the file isn't an ALL-CAPS file
            if not disable_upper_nuke and bare_text.isupper() and re.search(r'[A-Z]', bare_text):
                continue
                
            if bare_text.startswith('-') and not re.search(r'[a-zA-Z0-9]', bare_text):
                continue
                
            if bare_text:
                cleaned_lines.append(line)
                
        # Rejoin with the correct newline character
        final_text = nl.join(cleaned_lines).strip()
        cue.text = final_text
        
        # ONLY KEEP THE CUE IF IT STILL HAS VISIBLE TEXT
        # If removing SDH completely emptied the subtitle block or just left a single period, it gets dropped
        bare_final = re.sub(tag_pattern, '', final_text).strip()
        if bare_final and bare_final not in ['.', ',', '?', '-', '..']:
            valid_cues.append(cue)
            
    # Replace the old subtitle list
    sub[:] = valid_cues
    return sub

def strip_tags(sub):
    """
    Strips all HTML tags (e.g., <i>, <font>) and ASS formatting tags (e.g., {\\an8}).
    """
    # Matches HTML tags <...> and ASS tags {\...}
    tag_pattern = re.compile(r'<[^>]+>|\{\\[^}]+\}')
    
    for cue in sub:
        # 1. Strip all tags from the text
        cleaned_text = re.sub(tag_pattern, '', cue.text)
        
        # 2. Clean up any weird spacing left behind
        nl = r'\N' if r'\N' in cleaned_text else '\n'
        lines = re.split(r'\\N|\n', cleaned_text)
        
        # Keep lines that still have text, stripping outer spaces
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        
        # Rejoin with the correct newline
        cue.text = nl.join(cleaned_lines)
        
    return sub

def remove_empty_cues(sub):
    """
    Removes subtitle blocks that contain no visible text.
    Safely identifies cues that are purely whitespace, empty tags, or standalone newlines.
    """
    # Pattern to temporarily strip HTML and ASS tags for the "empty" check
    tag_pattern = re.compile(r'<[^>]+>|\{\\[^}]+\}')
    
    valid_cues = []
    
    for cue in sub:
        # Strip tags in memory
        bare_text = re.sub(tag_pattern, '', cue.text)
        
        # Strip literal pysubs2 newlines (\N), standard newlines (\n), and all spaces
        bare_text = bare_text.replace(r'\N', '').replace('\n', '').strip()
        
        # If there is actual visible text left, keep the original cue (tags included)
        if bare_text and bare_text not in ['.', ',', '?', '-', '..']:
            valid_cues.append(cue)
            
    # Replace the subtitle object's internal list with our scrubbed list
    sub[:] = valid_cues
    
    return sub

def fix_overlaps(sub):
    """
    Fixes overlapping display times by trimming the end time of the preceding cue 
    to match the start time of the incoming cue.
    """
    # Sort the subtitles chronologically by start time first!
    # pysubs2 has a built-in sort() method that handles this perfectly.
    sub.sort()
    
    valid_cues = []
    
    # Iterate through all cues except the very last one
    for i in range(len(sub) - 1):
        current_cue = sub[i]
        next_cue = sub[i + 1]
        
        #  Check for the overlap collision
        if current_cue.end > next_cue.start:
            # Trim the current cue to end exactly when the next one begins
            current_cue.end = next_cue.start
            
        # Safety Check: Did the trim destroy the cue?
        # If two cues started at the exact same millisecond, the first one 
        # just got its duration reduced to 0. Drop it.
        if current_cue.end > current_cue.start:
            valid_cues.append(current_cue)
            
    # (Since it has no "next_cue", it can't possibly overlap forward)
    if len(sub) > 0:
        last_cue = sub[-1]
        if last_cue.end > last_cue.start:
            valid_cues.append(last_cue)
            
    # Replace the subtitle timeline with our fixed one
    sub[:] = valid_cues
    
    return sub

def remove_watermarks(sub):
    """
    Removes promotional watermarks, credits, and URLs using a weighted confidence score.
    A cue must score >= 100 points to be safely deleted.
    """
    tag_pattern = re.compile(r'<[^>]+>|\{\\[^}]+\}')
    
    # Hard Promo: Characters practically never say these.
    hard_verb_pattern = re.compile(r'(?i)(?:sync(?:ed)?|sub(?:title)?s?|subbed|encod(?:ed)?)\s+(?:by|from)')
    # Soft Promo: Could technically be dialogue ("It was translated by the UN.")
    soft_verb_pattern = re.compile(r'(?i)(?:download(?:ed)?|correct(?:ed)?|translat(?:ed)?)\s+(?:by|from)')
    
    # URLs and Domains
    url_pattern = re.compile(r'(?i)(?:www\.|https?://|\.com|\.org|\.net|\.tv|\.ro|\.co)')
    # Known Subtitle keywords
    url_kw_pattern = re.compile(r'(?i)(?:sub|sync|addic7ed|yts|ganool|opensub|podnapisi|tvsubtitles|subscene|titlovi)')
    
    # Scene Release Metadata
    metadata_pattern = re.compile(r'(?i)(?:s\d{2}e\d{2}|1080p|720p|480p|x264|x265|bluray|web-?rip|hdtv|web-?dl|rip)')
    
    valid_cues = []
    total_cues = len(sub)
    
    for i, cue in enumerate(sub):
        # Check the bare text so watermarks can't hide inside <font> tags
        bare_text = re.sub(tag_pattern, '', cue.text).strip()
        
        # If it's already empty, keep it (the remove_empty_cues function handles blanks)
        if not bare_text or bare_text in ['.', ',', '?', '-', '..']:
            continue
            
        score = 0
        
        # --- APPLY HEURISTIC SCORING ---
        
        # The Boundary (First 10 or Last 10 cues)
        if i < 10 or i > (total_cues - 11):
            score += 50
            
        # Promo Verbs
        if re.search(hard_verb_pattern, bare_text):
            score += 100
        elif re.search(soft_verb_pattern, bare_text):
            score += 80
            
        # URLs
        if re.search(url_pattern, bare_text):
            score += 40
            # If the URL contains a subtitle keyword (e.g., opensubtitles.org)
            if re.search(url_kw_pattern, bare_text):
                score += 60
                
        # Release Metadata (e.g., S02E01)
        if re.search(metadata_pattern, bare_text):
            score += 60
            
        # Decorative ASCII
        # Looks for lines starting and ending with non-alphanumeric chars
        if re.search(r'^[^a-zA-Z0-9]+.*[^a-zA-Z0-9]+$', bare_text) and len(bare_text) > 4:
            score += 20

        if score >= 100:
            # Threshold met! Do NOT append it to valid_cues
            continue
            
        valid_cues.append(cue)
        
    sub[:] = valid_cues
    return sub

def fix_capitalization(sub):
    """
    Converts ALL-CAPS subtitle lines to Sentence case.
    Safely ignores formatting tags and attempts to preserve English 'I' pronouns.
    """
    # Regex to capture tags to split the string without losing them
    tag_pattern = re.compile(r'(<[^>]+>|\{\\[^}]+\})')

    for cue in sub:
        original_text = cue.text
        nl = r'\N' if r'\N' in original_text else '\n'
        lines = re.split(r'\\N|\n', original_text)
        cleaned_lines = []
        
        for line in lines:
            # Check the "bare text" to see if the line is entirely uppercase
            bare_text = re.sub(tag_pattern, '', line).strip()
            
            # Only apply fix if the line is 100% uppercase (and contains letters)
            if bare_text.isupper() and re.search(r'[A-Z]', bare_text):
                
                # Split the line into a list of [text, tag, text, tag, text...]
                parts = re.split(tag_pattern, line)
                capitalize_next = True # Start by capitalizing the first letter of the line
                
                for i, part in enumerate(parts):
                    # If this chunk is a tag, leave it completely alone
                    if re.match(tag_pattern, part):
                        continue
                        
                    # Lowercase the actual text chunk
                    text = part.lower()
                    
                    # Process character by character to handle sentence boundaries
                    new_text = []
                    for char in text:
                        if char.isalpha():
                            if capitalize_next:
                                new_text.append(char.upper())
                                capitalize_next = False
                            else:
                                new_text.append(char)
                        else:
                            new_text.append(char)
                            # If punctuation, trigger a capitalization for the next letter
                            if char in '.!?':
                                capitalize_next = True
                                
                    text = "".join(new_text)
                    
                    # English "I" pronoun fix (catches i, i'm, i'll, i've, i'd)
                    # \b ensures we don't accidentally capitalize the 'i' in 'alien'
                    text = re.sub(r"\b(i)(['’]?(?:m|ll|ve|d)?)\b", lambda m: "I" + m.group(2), text)
                    
                    # Update the chunk
                    parts[i] = text
                    
                # Rejoin the perfectly cased text and tags
                cleaned_lines.append("".join(parts))
            else:
                # If it wasn't ALL-CAPS, leave it exactly as it was
                cleaned_lines.append(line)
                
        # Rejoin multi-line cues
        cue.text = nl.join(cleaned_lines)
        
    return sub

