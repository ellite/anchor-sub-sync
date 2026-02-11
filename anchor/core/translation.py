import ctranslate2
import transformers
import pysubs2
import gc
import torch
import copy
import re
from rich.console import Console
from rich.progress import Progress, TaskID
from huggingface_hub import snapshot_download
from huggingface_hub.utils import disable_progress_bars

disable_progress_bars()

console = Console()

# Use the distilled tokenizer as it is compatible with NLLB and lightweight
TOKENIZER_ID = "facebook/nllb-200-distilled-600M"

# ---------- Helpers ----------

def merge_lines_if_needed(lines):
    """
    Merge lines according to the rule:
    If line[i] does NOT end with punctuation and next line does NOT start with '-',
    merge them into one line.
    """
    merged = []
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # last line -> nothing to merge
        if i == len(lines) - 1:
            merged.append(line)
            break
        
        next_line = lines[i + 1].strip()

        # condition for merging:
        if (not re.search(r"[.!?…]$", line)) and (not next_line.startswith("-")):
            merged_line = line + " " + next_line
            merged.append(merged_line)
            i += 2  # skip next line
        else:
            merged.append(line)
            i += 1

    return merged

def fix_long_lines(text, max_length=42):
    """
    Splits long lines into two lines, trying to find the best 
    split point near the middle (similar to Subtitle Edit).
    """
    if len(text) <= max_length or "\n" in text:
        return text

    # Find all possible split points (spaces)
    words = text.split(" ")
    if len(words) == 1:
        return text  # Can't split a single long word

    best_split = len(text)
    min_diff = len(text)
    
    # Iterate through words to find a split point closest to the middle
    current_length = 0
    for i in range(len(words) - 1):
        current_length += len(words[i]) + 1
        # Calculate how balanced the two lines would be
        line1 = current_length
        line2 = len(text) - current_length
        diff = abs(line1 - line2)
        
        if diff < min_diff:
            min_diff = diff
            best_split = current_length

    # Perform the split
    line1 = text[:best_split].strip()
    line2 = text[best_split:].strip()
    
    return f"{line1}\n{line2}"


def is_all_upper(text: str) -> bool:
    """Return True if the text contains at least one letter and all letters are uppercase."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    return all(c.isupper() for c in letters)

# ---------- Main Translation Function ----------

def translate_subtitle_nllb(
    sub: pysubs2.SSAFile, 
    source_code: str, 
    target_code: str, 
    device="cpu", 
    model_id="JustFrederik/nllb-200-distilled-600M-ct2-int8",
    progress: Progress = None,
    task_id: TaskID = None
) -> pysubs2.SSAFile:
    
    # --- 1. PRE-PROCESSING ---
    # Flatten the structure into a list of lines to translate,
    # but keep a 'meta' map to know how to put them back into events.
    
    # Struct: (event_index, number_of_lines_in_this_event)
    metas = [] 
    all_lines = []

    for i, event in enumerate(sub):
        if event.is_comment: continue
        
        # Split by SSA newline (\N) or literal newline (\n) to get raw lines
        raw_text = event.text.strip()
        if not raw_text:
            continue
            
        raw_lines = re.split(r"\\N|\n", raw_text)
        lines = merge_lines_if_needed(raw_lines)
        
        if lines:
            metas.append((i, len(lines)))
            all_lines.extend(lines)

    if not all_lines:
        return sub

    # Update Progress Total
    if progress and task_id is not None:
        progress.update(task_id, total=len(all_lines))

    # --- 2. MODEL LOADING ---
    translator = None
    tokenizer = None

    try:
        # Download model
        model_path = snapshot_download(repo_id=model_id)
        
        # Initialize CTranslate2
        translator = ctranslate2.Translator(
            model_path, 
            device=device,
            compute_type="int8"
        )
        
        # Initialize Tokenizer (Exact setup as your script)
        tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER_ID)
        
        # Explicitly set lang attributes (just like your script)
        tokenizer.src_lang = source_code
        tokenizer.tgt_lang = target_code
        
        forced_bos = target_code 

        # --- 3. BATCH TRANSLATION ---
        batch_size = 32 # Your script used 16, but 32 is standard. Change to 16 if you want exact match.
        translated_lines = []
        
        for i in range(0, len(all_lines), batch_size):
            batch = all_lines[i : i + batch_size]
            
            # Tokenize
            source = tokenizer(batch)["input_ids"]
            source_tokens = [tokenizer.convert_ids_to_tokens(s) for s in source]
            
            # Translate (CTranslate2)
            results = translator.translate_batch(
                source_tokens,
                target_prefix=[[forced_bos]] * len(batch)
            )
            
            # Decode
            target_tokens = [x.hypotheses[0] for x in results]
            decoded_batch = tokenizer.batch_decode(
                [tokenizer.convert_tokens_to_ids(x) for x in target_tokens], 
                skip_special_tokens=True
            )
            
            translated_lines.extend(decoded_batch)

            if progress and task_id is not None:
                progress.advance(task_id, advance=len(batch))

        # --- 4. RECONSTRUCTION (Post-processing) ---
        ghost_sub = copy.deepcopy(sub)
        
        cursor = 0
        for event_idx, line_count in metas:
            # Slice the translated lines belonging to this event
            event_lines_translated = translated_lines[cursor : cursor + line_count]
            orig_lines = all_lines[cursor : cursor + line_count]
            cursor += line_count

            processed_lines = []
            for orig, trans in zip(orig_lines, event_lines_translated):
                if is_all_upper(orig):
                    trans = trans.upper()
                processed_lines.append(fix_long_lines(trans))

            # Join back. pysubs2 uses \N for breaks.
            ghost_sub[event_idx].text = "\\N".join(processed_lines)
            
        return ghost_sub

    except Exception as e:
        console.print(f"\n[bold red]❌ Translation Failed:[/bold red] {e}")
        return None

    finally:
        if translator: del translator
        if tokenizer: del tokenizer
        gc.collect()
        if device == "cuda": torch.cuda.empty_cache()