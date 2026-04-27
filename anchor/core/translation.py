import ctranslate2
import transformers
import pysubs2
import gc
import torch
import copy
import re
from typing import List, Tuple
from rich.console import Console
from rich.progress import Progress, TaskID
from huggingface_hub import snapshot_download
from huggingface_hub.utils import disable_progress_bars

disable_progress_bars()

console = Console()

# Use the distilled tokenizer as it is compatible with NLLB and lightweight
TOKENIZER_ID = "facebook/nllb-200-distilled-600M"

# ---------- General Helpers ----------

def merge_lines_if_needed(lines):
    merged = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if i == len(lines) - 1:
            merged.append(line)
            break
        next_line = lines[i + 1].strip()
        if (not re.search(r"[.!?…]$", line)) and (not next_line.startswith("-")):
            merged_line = line + " " + next_line
            merged.append(merged_line)
            i += 2  
        else:
            merged.append(line)
            i += 1
    return merged

def is_all_upper(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    return all(c.isupper() for c in letters)

# ---------- Scoring Logic ----------

def calculate_suspicion_score(src_text: str, tgt_text: str) -> float:
    score = 0.0
    src_clean = src_text.strip()
    tgt_clean = tgt_text.strip()
    
    if not src_clean: return 0.0
    if not tgt_clean: return 100.0  
        
    src_len = max(1, len(src_clean))
    tgt_len = max(1, len(tgt_clean))
    ratio = tgt_len / src_len
    
    # Truncation / Hallucination (Length Ratio)
    if ratio < 0.35: score += 50.0  
    elif ratio < 0.5: score += 20.0  
    elif ratio > 2.5: score += 30.0  
        
    # Vocabulary Density (Catches AI stuttering loops)
    tgt_words = re.findall(r'\b\w+\b', tgt_clean.lower())
    num_tgt_words = len(tgt_words)
    if num_tgt_words > 4:
        unique_words = len(set(tgt_words))
        density = unique_words / num_tgt_words
        if density < 0.4:
            score += 40.0  
            
    # Formatting Mismatches
    if "-" not in src_clean and tgt_clean.startswith("-"):
        score += 15.0  
        
    src_terminators = len(re.findall(r'[.!?]+(?:\s|$)', src_clean))
    tgt_terminators = len(re.findall(r'[.!?]+(?:\s|$)', tgt_clean))
    if tgt_terminators > max(1, src_terminators + 1):
        score += 20.0  
        
    return score

# ---------- Guardrail Fixes ----------

def strip_hallucinated_dialogue(src_text: str, tgt_text: str) -> str:
    src_clean = src_text.strip()
    tgt_clean = tgt_text.strip()
    if "-" not in src_clean and tgt_clean.startswith("- "):
        parts = [p.strip() for p in tgt_clean.split("- ") if p.strip()]
        if parts: tgt_clean = parts[0]
    src_terminators = len(re.findall(r'[.!?]+(?:\s|$)', src_clean))
    tgt_terminators = len(re.findall(r'[.!?]+(?:\s|$)', tgt_clean))
    if tgt_terminators > max(1, src_terminators):
        match = re.search(r'([.!?]+(?:\s|$))', tgt_clean)
        if match:
            cutoff_idx = match.end()
            tgt_clean = tgt_clean[:cutoff_idx].strip()
    return tgt_clean

def crush_stutter_loops(src_text: str, tgt_text: str) -> str:
    src_lines = src_text.split('\n')
    tgt_lines = tgt_text.split('\n')
    processed_lines = []
    for i, t_line in enumerate(tgt_lines):
        t_clean = t_line.strip()
        tgt_words = re.findall(r'\w+', t_clean)
        if not tgt_words:
            processed_lines.append(t_line)
            continue
        tgt_lower = [w.lower() for w in tgt_words]
        if len(tgt_lower) > 1 and len(set(tgt_lower)) == 1:
            s_line = src_lines[i] if i < len(src_lines) else src_text
            src_words = re.findall(r'\w+', s_line)
            if len(tgt_lower) > len(src_words):
                valid_count = max(1, len(src_words))
                base_word = tgt_words[0]
                if valid_count == 1: rebuilt = base_word
                else: rebuilt = base_word + ", " + ", ".join([base_word.lower()] * (valid_count - 1))
                prefix = "- " if t_clean.startswith("-") else ""
                match = re.search(r'([.!?]+)$', t_clean)
                suffix = match.group(1) if match else "."
                processed_lines.append(f"{prefix}{rebuilt}{suffix}")
                continue
        processed_lines.append(t_line)
    return '\n'.join(processed_lines) 

def protect_isolated_names(src_text: str, tgt_text: str) -> str:
    src_lines = src_text.split('\n')
    tgt_lines = tgt_text.split('\n')
    processed_lines = []
    universal_yes_no = {
        "yes", "no", "yeah", "yep", "nope", "yeap", "nah", "yup", "sim", "não", "nao", "sí", "si", "sì", "oui", "non",
        "ja", "nein", "nee", "da", "nu", "nej", "nei", "kyllä", "kylla", "ei", "evet", "hayır", "hayir", "ναι", "όχι", "οχι",
        "tak", "nie", "ano", "igen", "nem", "نعم", "لا", "כן", "net", "nyet", "ne", "да", "нет", "niet",
        "是", "对", "不", "不是", "不是的", "はい", "うん", "いいえ", "ううん", "네", "응", "아니요", "아니", "ok", "okay", "kk",
    }
    for i, t_line in enumerate(tgt_lines):
        s_line = src_lines[i] if i < len(src_lines) else ""
        src_words = re.findall(r'\b\w+\b', s_line)
        tgt_words = re.findall(r'\b\w+\b', t_line)
        if len(src_words) == 1 and len(tgt_words) == 1:
            s_word = src_words[0].lower()
            t_word = tgt_words[0].lower()
            if t_word in universal_yes_no and s_word not in universal_yes_no:
                match = re.search(r'([.!?]+)$', t_line.strip())
                punct = match.group(1) if match else ""
                prefix = "- " if t_line.strip().startswith("-") else ""
                processed_lines.append(f"{prefix}{src_words[0]}{punct}")
                continue
        processed_lines.append(t_line)
    return '\n'.join(processed_lines)   

def enforce_short_answers(src_text: str, tgt_text: str) -> str:
    src_lines = src_text.split('\n')
    tgt_lines = tgt_text.split('\n')
    processed_lines = []
    for i, t_line in enumerate(tgt_lines):
        s_line = src_lines[i] if i < len(src_lines) else ""
        src_words = re.findall(r'\w+', s_line)
        if 0 < len(src_words) <= 2:
            if "," not in s_line and "," in t_line:
                match = re.search(r'^([^,]+)', t_line)
                if match:
                    base = match.group(1).strip()
                    punct_match = re.search(r'([.!?]+)$', t_line.strip())
                    punct = punct_match.group(1) if punct_match else "."
                    processed_lines.append(f"{base}{punct}")
                    continue
        processed_lines.append(t_line)
    return '\n'.join(processed_lines)

def clean_typography(text: str) -> str:
    """Strips erroneous spaces before punctuation (common when translating from French)."""
    return re.sub(r'\s+([.!?:,;])', r'\1', text)

def format_subtitle_block(lines, max_length=42) -> str:
    joined_text = " ".join(l.strip() for l in lines if l.strip())
    joined_text = re.sub(r'\s+', ' ', joined_text) 
    if not joined_text: return ""
        
    dialogue_match = re.match(r'^(-\s*.*?)\s+(-\s*.*)$', joined_text)
    if dialogue_match:
        line1 = dialogue_match.group(1).strip()
        line2 = dialogue_match.group(2).strip()
        return f"{line1}\n{line2}"
        
    if len(joined_text) <= max_length:
        return joined_text
        
    words = joined_text.split(" ")
    if len(words) <= 1: return joined_text
        
    best_split = len(joined_text)
    min_diff = len(joined_text)
    current_length = 0
    
    for i in range(len(words) - 1):
        current_length += len(words[i]) + 1
        line1_len = current_length
        line2_len = len(joined_text) - current_length
        diff = abs(line1_len - line2_len)
        if diff < min_diff:
            min_diff = diff
            best_split = current_length

    line1 = joined_text[:best_split].strip()
    line2 = joined_text[best_split:].strip()
    return f"{line1}\n{line2}"

# ---------- CT2 Generation Wrapper ----------

def _translate_ct2_batch(batch, tokenizer, translator, forced_bos, beam_size=4, length_penalty=1.0, repetition_penalty=1.2, no_repeat_ngram_size=3):
    source = tokenizer(batch)["input_ids"]
    source_tokens = [tokenizer.convert_ids_to_tokens(s) for s in source]
    
    results = translator.translate_batch(
        source_tokens,
        target_prefix=[[forced_bos]] * len(batch),
        beam_size=beam_size,
        max_decoding_length=256,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        length_penalty=length_penalty
    )
    
    target_tokens = [x.hypotheses[0] for x in results]
    return tokenizer.batch_decode(
        [tokenizer.convert_tokens_to_ids(x) for x in target_tokens], 
        skip_special_tokens=True
    )


# Main CLI Translation Function (For Subtitle Files)

def translate_subtitle_nllb(
    sub: pysubs2.SSAFile, 
    source_code: str, 
    target_code: str, 
    device="cpu", 
    model_id="JustFrederik/nllb-200-distilled-600M-ct2-float16",
    progress: Progress = None,
    task_id: TaskID = None
) -> pysubs2.SSAFile:
    
    metas = [] 
    all_lines = []

    for i, event in enumerate(sub):
        if event.is_comment: continue
        
        raw_text = event.text.strip()
        if not raw_text: continue
            
        raw_lines = re.split(r"\\N|\n", raw_text)
        lines = merge_lines_if_needed(raw_lines)
        
        if lines:
            metas.append((i, len(lines)))
            all_lines.extend(lines)

    if not all_lines:
        return sub

    if progress and task_id is not None:
        progress.update(task_id, total=len(all_lines))

    translator = None
    tokenizer = None

    try:
        if progress and task_id is not None:
            progress.update(task_id, description="Loading Model...", advance=0)
            
        model_path = snapshot_download(repo_id=model_id)
        
        if progress and task_id is not None:
            progress.update(task_id, description="Translating...", advance=0)
            
        translator = ctranslate2.Translator(
            model_path, 
            device=device,
            compute_type="auto"
        )
        
        tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER_ID)
        tokenizer.src_lang = source_code
        tokenizer.tgt_lang = target_code
        forced_bos = target_code 

        # Main Translation
        batch_size = 32 
        translated_lines = []
        
        for i in range(0, len(all_lines), batch_size):
            batch = all_lines[i : i + batch_size]
            
            model_batch = []
            valid_indices = []
            batch_out = [""] * len(batch)
            
            for j, line in enumerate(batch):
                if not any(c.isalpha() for c in line):
                    batch_out[j] = line
                else:
                    valid_indices.append(j)
                    if is_all_upper(line):
                        model_batch.append(line.capitalize()) 
                    else:
                        model_batch.append(line)

            # Only translate the lines that actually made it into model_batch
            if model_batch:
                decoded_batch = _translate_ct2_batch(
                    model_batch, tokenizer, translator, forced_bos,
                    beam_size=4, length_penalty=1.0, repetition_penalty=1.2, no_repeat_ngram_size=3
                )
                
                # Map the translated text back into the blank template
                for valid_idx, trans in zip(valid_indices, decoded_batch):
                    batch_out[valid_idx] = trans
                    
            translated_lines.extend(batch_out)

            if progress and task_id is not None:
                progress.advance(task_id, advance=len(batch))

        # Suspicion Scoring
        fallback_threshold = 20.0
        fallback_indices = []
        fallback_sources = []
        
        for idx, (src, tgt) in enumerate(zip(all_lines, translated_lines)):
            if calculate_suspicion_score(src, tgt) >= fallback_threshold:
                fallback_indices.append(idx)
                fallback_sources.append(src.capitalize() if is_all_upper(src) else src)
                
        # Fallback Duel & Surgeon Pass
        if fallback_indices:
            if progress:
                progress.console.print(f"   [dim]🔍 Found {len(fallback_indices)} suspicious lines. Running fallback duel...[/dim]")
                
            fallback_out_lines = []
            duel_task_id = progress.add_task("   [dim]Fallback duel...[/dim]", total=len(fallback_sources)) if progress else None
            for i in range(0, len(fallback_sources), batch_size):
                batch = fallback_sources[i:i+batch_size]

                # Brute-Force Literal Settings
                decoded_batch = _translate_ct2_batch(
                    batch, tokenizer, translator, forced_bos,
                    beam_size=2, length_penalty=2.0, repetition_penalty=1.0, no_repeat_ngram_size=0
                )
                fallback_out_lines.extend(decoded_batch)
                if progress and duel_task_id is not None:
                    progress.advance(duel_task_id, advance=len(batch))
            if progress and duel_task_id is not None:
                progress.remove_task(duel_task_id)
                
            fixes_applied = 0
            for i, original_idx in enumerate(fallback_indices):
                src = all_lines[original_idx]
                old_tgt = translated_lines[original_idx]
                greedy_tgt = fallback_out_lines[i]
                
                # Keep track of all the attempts and their scores
                candidates = [
                    (old_tgt, calculate_suspicion_score(src, old_tgt)),
                    (greedy_tgt, calculate_suspicion_score(src, greedy_tgt))
                ]
                
                # SURGEON PASS
                chunks = [c.strip() for c in re.split(r'(?<=[.!?])\s+', src) if c.strip()]
                
                if len(chunks) > 1:
                    chunk_batch = [c.capitalize() if is_all_upper(c) else c for c in chunks]
                    
                    chunk_results = _translate_ct2_batch(
                        chunk_batch, tokenizer, translator, forced_bos,
                        beam_size=4, length_penalty=1.0, repetition_penalty=1.2, no_repeat_ngram_size=3
                    )
                    
                    # Clean the chunks of hallucinated hyphens BEFORE gluing them together!
                    cleaned_chunks = []
                    for c_src, c_tgt in zip(chunks, chunk_results):
                        c_tgt = strip_hallucinated_dialogue(c_src, c_tgt)
                        cleaned_chunks.append(c_tgt)
                    
                    surgeon_tgt = " ".join(cleaned_chunks)
                    candidates.append((surgeon_tgt, calculate_suspicion_score(src, surgeon_tgt)))

                # Find the candidate with the lowest suspicion score
                best_tgt, best_score = min(candidates, key=lambda x: x[1])
                
                # If one of the fallbacks beat the original, apply it!
                if best_tgt != old_tgt:
                    translated_lines[original_idx] = best_tgt
                    fixes_applied += 1
            
            if progress and fixes_applied > 0:
                progress.console.print(f"   [green]✨ Successfully repaired {fixes_applied} lines.[/green]")

        # Deterministic Guardrail Fixes
        for idx, (src, tgt) in enumerate(zip(all_lines, translated_lines)):
            clean_tgt = crush_stutter_loops(src, tgt)
            clean_tgt = enforce_short_answers(src, clean_tgt)
            clean_tgt = strip_hallucinated_dialogue(src, clean_tgt)
            clean_tgt = protect_isolated_names(src, clean_tgt)
            clean_tgt = clean_typography(clean_tgt)
            
            if is_all_upper(src):
                translated_lines[idx] = clean_tgt.upper()
            else:
                translated_lines[idx] = clean_tgt

        # Rebuild Blocks
        ghost_sub = copy.deepcopy(sub)
        cursor = 0
        
        for event_idx, line_count in metas:
            event_lines_translated = translated_lines[cursor : cursor + line_count]
            cursor += line_count

            # Apply the professional 2-line balancing
            block_text = format_subtitle_block(event_lines_translated)

            # PySubs2 requires '\N' for line breaks internally, so we swap standard newlines
            ghost_sub[event_idx].text = block_text.replace('\n', '\\N')
            
        return ghost_sub

    except Exception as e:
        console.print(f"\n[bold red]❌ Translation Failed:[/bold red] {e}")
        return None

    finally:
        if translator: del translator
        if tokenizer: del tokenizer
        gc.collect()
        if device == "cuda": torch.cuda.empty_cache()

# API Functions (For Raw Text Arrays)

def load_model(model_id: str, device: str = "cpu", compute_type: str = "auto") -> Tuple[transformers.PreTrainedTokenizer, ctranslate2.Translator]:
    """
    Loads the NLLB tokenizer and CTranslate2 model into VRAM persistently.
    Designed for the API watchdog so it doesn't constantly reload from disk.
    """
    model_path = snapshot_download(repo_id=model_id)
    
    translator = ctranslate2.Translator(
        model_path, 
        device=device,
        compute_type=compute_type
    )
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER_ID)
    
    return tokenizer, translator

def translate_lines_nllb(
    lines: List[str], 
    source_lang: str, 
    target_lang: str, 
    tokenizer: transformers.PreTrainedTokenizer, 
    translator: ctranslate2.Translator, 
    batch_size: int = 32
) -> List[str]:
    """
    Translates a raw list of strings, applying the same guardrails, fallback 
    duels, and formatting fixes as the subtitle pipeline.
    """
    if not lines:
        return []
        
    # Configure language tokens for NLLB
    tokenizer.src_lang = source_lang
    tokenizer.tgt_lang = target_lang
    forced_bos = target_lang 

    translated_lines = []
    
    # 1. Main Batch Translation
    for i in range(0, len(lines), batch_size):
        batch = lines[i : i + batch_size]
        
        model_batch = []
        valid_indices = []
        batch_out = [""] * len(batch)
        
        for j, line in enumerate(batch):
            # No letters? Don't translate it.
            if not any(c.isalpha() for c in line):
                batch_out[j] = line
            else:
                valid_indices.append(j)
                if is_all_upper(line):
                    model_batch.append(line.capitalize()) 
                else:
                    model_batch.append(line)

        if model_batch:
            decoded_batch = _translate_ct2_batch(
                model_batch, tokenizer, translator, forced_bos,
                beam_size=4, length_penalty=1.0, repetition_penalty=1.2, no_repeat_ngram_size=3
            )
            
            for valid_idx, trans in zip(valid_indices, decoded_batch):
                batch_out[valid_idx] = trans
                
        translated_lines.extend(batch_out)

    # 2. Suspicion Scoring
    fallback_threshold = 20.0
    fallback_indices = []
    fallback_sources = []
    
    for idx, (src, tgt) in enumerate(zip(lines, translated_lines)):
        if calculate_suspicion_score(src, tgt) >= fallback_threshold:
            fallback_indices.append(idx)
            fallback_sources.append(src.capitalize() if is_all_upper(src) else src)
            
    # 3. Fallback Duel & Surgeon Pass
    if fallback_indices:
        fallback_out_lines = []
        for i in range(0, len(fallback_sources), batch_size):
            batch = fallback_sources[i:i+batch_size]
            
            # Brute-Force Literal Settings
            decoded_batch = _translate_ct2_batch(
                batch, tokenizer, translator, forced_bos,
                beam_size=2, length_penalty=2.0, repetition_penalty=1.0, no_repeat_ngram_size=0
            )
            fallback_out_lines.extend(decoded_batch)
            
        for i, original_idx in enumerate(fallback_indices):
            src = lines[original_idx]
            old_tgt = translated_lines[original_idx]
            greedy_tgt = fallback_out_lines[i]
            
            candidates = [
                (old_tgt, calculate_suspicion_score(src, old_tgt)),
                (greedy_tgt, calculate_suspicion_score(src, greedy_tgt))
            ]
            
            # SURGEON PASS
            chunks = [c.strip() for c in re.split(r'(?<=[.!?])\s+', src) if c.strip()]
            
            if len(chunks) > 1:
                chunk_batch = [c.capitalize() if is_all_upper(c) else c for c in chunks]
                chunk_results = _translate_ct2_batch(
                    chunk_batch, tokenizer, translator, forced_bos,
                    beam_size=4, length_penalty=1.0, repetition_penalty=1.2, no_repeat_ngram_size=3
                )
                
                cleaned_chunks = []
                for c_src, c_tgt in zip(chunks, chunk_results):
                    c_tgt = strip_hallucinated_dialogue(c_src, c_tgt)
                    cleaned_chunks.append(c_tgt)
                
                surgeon_tgt = " ".join(cleaned_chunks)
                candidates.append((surgeon_tgt, calculate_suspicion_score(src, surgeon_tgt)))

            # Find the candidate with the lowest suspicion score
            best_tgt, best_score = min(candidates, key=lambda x: x[1])
            if best_tgt != old_tgt:
                translated_lines[original_idx] = best_tgt

    # 4. Deterministic Guardrail Fixes
    for idx, (src, tgt) in enumerate(zip(lines, translated_lines)):
        clean_tgt = crush_stutter_loops(src, tgt)
        clean_tgt = enforce_short_answers(src, clean_tgt)
        clean_tgt = strip_hallucinated_dialogue(src, clean_tgt)
        clean_tgt = protect_isolated_names(src, clean_tgt)
        clean_tgt = clean_typography(clean_tgt)
        
        if is_all_upper(src):
            translated_lines[idx] = clean_tgt.upper()
        else:
            translated_lines[idx] = clean_tgt

    return translated_lines