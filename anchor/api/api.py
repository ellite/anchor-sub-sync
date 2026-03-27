import time
import asyncio
import gc
import torch
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import List, Union

from ..utils.mappings import get_language_code_for_nllb
from ..core.translation import load_model, translate_lines_nllb 

app = FastAPI(title="Anchor Subtitle API (LibreTranslate Mode)")

# THE GLOBAL STATE
class EngineState:
    translator = None
    tokenizer = None
    last_active = 0.0
    config = {}
    load_lock = asyncio.Lock() 

state = EngineState()

class TranslateRequest(BaseModel):
    source: Union[List[str], str]
    src_lang: str
    tgt_lang: str

# THE BACKGROUND WATCHDOG
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(idle_watchdog())

async def idle_watchdog():
    while True:
        await asyncio.sleep(5)
        
        if getattr(state, 'translator', None) is not None:
            timeout_limit = state.config.get('idle_timeout_seconds', 60)
            
            if (time.time() - state.last_active) > timeout_limit:
                print(f"\n[dim]💤 {timeout_limit}-second idle timeout reached. Unloading model...[/dim]")
                
                # Prevent any new requests from trying to use it while we delete it
                async with state.load_lock:
                    try:
                        state.translator.unload_model()
                    except:
                        pass
                    
                    del state.translator
                    del state.tokenizer
                    state.translator = None
                    state.tokenizer = None
                    
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                print("✅ VRAM freed. API is back in idle mode.\n")

# THE FRONT DOOR (The Endpoints)
@app.get("/")
async def health_check():
    return {"status": "Anchor API is awake and listening!"}

@app.get("/languages")
async def get_languages():
    return [
        {"code": "eng_Latn", "name": "English"},
        {"code": "por_Latn", "name": "Portuguese"},
        {"code": "spa_Latn", "name": "Spanish"},
        {"code": "fra_Latn", "name": "French"},
        {"code": "deu_Latn", "name": "German"},
        {"code": "ita_Latn", "name": "Italian"},
        {"code": "nld_Latn", "name": "Dutch"},
        {"code": "rus_Cyrl", "name": "Russian"},
        {"code": "jpn_Jpan", "name": "Japanese"},
        {"code": "zho_Hans", "name": "Chinese"},
    ]

@app.post("/translate")
async def translate_endpoint(req: TranslateRequest, raw_request: Request):
    # FLORES codes come in directly from nllb-serve mode — use them as-is
    nllb_src = req.src_lang if "_" in req.src_lang else get_language_code_for_nllb(req.src_lang)
    nllb_tgt = req.tgt_lang if "_" in req.tgt_lang else get_language_code_for_nllb(req.tgt_lang)

    if not nllb_src:
        raise HTTPException(status_code=400, detail=f"Unsupported source language: {req.src_lang}")
    if not nllb_tgt:
        raise HTTPException(status_code=400, detail=f"Unsupported target language: {req.tgt_lang}")

    if state.translator is None or state.tokenizer is None:
        async with state.load_lock:
            # Check again INSIDE the lock. 
            # (If Request 1 just finished loading it, Request 2 will see it's now loaded and skip this)
            if state.translator is None or state.tokenizer is None:
                model_name = state.config.get('translation_model', 'Unknown Model')
                device = state.config.get('device', 'cuda')
                compute_type = state.config.get('compute_type', 'int8')
                
                print(f"\n⚡ Waking up model: {model_name}...")
                
                state.tokenizer, state.translator = load_model(
                    model_id=model_name, 
                    device=device, 
                    compute_type=compute_type
                )
                print("✅ Model loaded into VRAM!")

    state.last_active = time.time()
    start_time = time.time()
    
    texts_to_translate = req.source if isinstance(req.source, list) else [req.source]
    
    try:
        translated_texts = translate_lines_nllb(
            lines=texts_to_translate,
            source_lang=nllb_src,
            target_lang=nllb_tgt,
            tokenizer=state.tokenizer,
            translator=state.translator,
            batch_size=state.config.get('batch_size', 8)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")
    
    duration = time.time() - start_time
    print(f"➔ Translated {len(texts_to_translate)} lines in {duration:.2f}s")
    
    response_data = [{"translation": text} for text in translated_texts]
    return response_data

# THE LAUNCHER
def run_apimode(args, device, model_size, compute_type, batch_size, translation_model, console, config):
    host = config['api_server']['host']
    port = config['api_server']['port']
    timeout = config['api_server']['idle_timeout_seconds']
    
    console.print("[bold green]🔌 Starting Anchor API Mode (LibreTranslate Clone)...[/bold green]")
    console.print(f"⚓ Listening on [underline cyan]http://{host}:{port}[/underline cyan]")
    console.print(f"[dim]Idle mode active. VRAM is currently free. (Timeout: {timeout}s)[/dim]\n")
    
    state.config = {
        "device": device,
        "compute_type": compute_type,
        "batch_size": batch_size,
        "translation_model": translation_model,
        "idle_timeout_seconds": timeout
    }
    
    uvicorn.run(app, host=host, port=port, log_level="warning")