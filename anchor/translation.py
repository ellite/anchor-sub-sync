import ctranslate2
import transformers
import pysubs2
import gc
import torch
import copy
from rich.console import Console
from huggingface_hub import snapshot_download
from huggingface_hub.utils import disable_progress_bars

disable_progress_bars()

console = Console()

TOKENIZER_ID = "facebook/nllb-200-distilled-600M"

def translate_subtitle_nllb(
    sub: pysubs2.SSAFile, 
    source_code: str, 
    target_code: str, 
    device="cuda", 
    model_id="OpenNMT/nllb-200-distilled-600M-ct2-int8"
) -> pysubs2.SSAFile:
    
    lines = []
    indices = []
    
    for i, event in enumerate(sub):
        text = event.plaintext.strip()
        if text:
            lines.append(text)
            indices.append(i)

    if not lines:
        return sub

    translator = None
    tokenizer = None

    try:
        # Download silently (progress bars disabled above)
        model_path = snapshot_download(repo_id=model_id)
        
        translator = ctranslate2.Translator(
            model_path, 
            device=device,
            compute_type="int8"
        )
        
        tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER_ID)

        batch_size = 32
        translated_lines = []
        
        for i in range(0, len(lines), batch_size):
            batch = lines[i : i + batch_size]
            
            # Tokenize
            source = tokenizer(batch)["input_ids"]
            source_tokens = [tokenizer.convert_ids_to_tokens(s) for s in source]
            
            results = translator.translate_batch(
                source_tokens,
                target_prefix=[[target_code]] * len(batch)
            )
            
            target_tokens = [x.hypotheses[0] for x in results]
            decoded_batch = tokenizer.batch_decode(
                [tokenizer.convert_tokens_to_ids(x) for x in target_tokens], 
                skip_special_tokens=True
            )
            
            translated_lines.extend(decoded_batch)

        ghost_sub = copy.deepcopy(sub)
        
        for idx, trans_text in zip(indices, translated_lines):
            ghost_sub[idx].text = trans_text
            
        return ghost_sub

    except Exception as e:
        console.print(f"[bold red]❌ Translation Failed:[/bold red] {e}")
        return sub

    finally:
        if translator: del translator
        if tokenizer: del tokenizer
        gc.collect()
        if device == "cuda": torch.cuda.empty_cache()