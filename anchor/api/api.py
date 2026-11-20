import os
import sys
import json
import time
import uuid
import asyncio
import gc
import tempfile
import shutil
import subprocess
import threading
import wave
import torch
import uvicorn
from fastapi import FastAPI, Request, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Union, Optional
from pathlib import Path

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
    whisper_model = None            # kept between /transcribe-segment calls, freed when idle
    whisper_key = None              # (model name, language) the loaded model was made for
    whisper_last = 0.0
    whisper_lock = threading.Lock()

state = EngineState()

class TranslateRequest(BaseModel):
    source: Union[List[str], str]
    src_lang: str
    tgt_lang: str

class SegmentRequest(BaseModel):
    video: str
    start: float = 0.0                  # seconds into the video
    seconds: float                      # length of the window
    language: Optional[str] = None      # audio language if known; None lets Whisper detect it


class SyncRequest(BaseModel):
    video: str
    subtitle: str
    overwrite: bool = False             # True replaces the subtitle in place; False saves X.synced.srt next to it


# Audio sync jobs. One runs at a time, in a child process (the same code path as `anchor -s … -v …`),
# so a sync never shares the GPU with a loaded translation model.
JOBS = {}                 # job_id -> dict
JOB_ORDER = []            # oldest first, trimmed to MAX_JOBS
MAX_JOBS = 200
sync_queue = None         # asyncio.Queue, created at startup
BACKGROUND_TASKS = set()  # strong references to the worker and the watchdog


def _require_token(request: Request):
    token = state.config.get("token") or ""
    if not token:
        return
    given = request.headers.get("authorization", "")
    if given != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="Missing or wrong API token")


def _allowed(path: Path) -> bool:
    """True when the real path of `path` is inside one of the api_server.allowed_dirs."""
    roots = [Path(d).expanduser().resolve() for d in state.config.get("allowed_dirs") or [] if d]
    real = path.resolve()
    return any(real == r or r in real.parents for r in roots)


def _public(job):
    return {k: v for k, v in job.items() if not k.startswith("_")}


async def _unload_translator():
    """Free the translation model's VRAM before a sync needs the GPU."""
    async with state.load_lock:
        if state.translator is not None:
            try:
                state.translator.unload_model()
            except Exception:
                pass
            state.translator = None
            state.tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


async def _run_job(job):
    job["state"] = "running"
    job["started"] = time.time()
    await _unload_translator()
    state.last_active = time.time()

    fd, result_file = tempfile.mkstemp(prefix="anchor-sync-", suffix=".json")
    os.close(fd)
    args = [sys.executable, "-c", "import sys; from anchor.cli import main; sys.exit(main())",
            "-s", job["subtitle"], "-v", job["video"]]
    if job["overwrite"]:
        args.append("-o")
    env = dict(os.environ, ANCHOR_RESULT_FILE=result_file, PYTHONUNBUFFERED="1",
               PYTHONPATH=str(Path(__file__).resolve().parents[1]) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    tail = []
    try:
        proc = await asyncio.create_subprocess_exec(
            *args, env=env, stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        job["_proc"] = proc
        async for raw in proc.stdout:
            line = raw.decode("utf-8", "replace").rstrip()
            if line.strip():
                tail.append(line)
                del tail[:-40]
                job["progress"] = line.strip()[-200:]
        code = await proc.wait()
        results = []
        try:
            with open(result_file, encoding="utf-8") as f:
                results = json.load(f)
        except (OSError, ValueError):
            pass
        ok = [r for r in results if r.get("ok")]
        if code == 0 and ok:
            job["state"] = "done"
            job["result"] = ok[0]
        else:
            job["state"] = "error"
            failed_line = next((l.split("❌", 1)[1].strip() for l in reversed(tail) if "❌" in l), None)
            job["error"] = (next((r.get("error") for r in results if not r.get("ok")), None)
                            or failed_line or f"anchor exited with code {code}")
            job["log"] = tail[-15:]
    except Exception as e:
        job["state"] = "error"
        job["error"] = str(e)
    finally:
        job["finished"] = time.time()
        job.pop("_proc", None)
        try:
            os.unlink(result_file)
        except OSError:
            pass


async def _sync_worker():
    while True:
        job_id = await sync_queue.get()
        job = JOBS.get(job_id)
        if job and job["state"] == "queued":
            await _run_job(job)


# THE BACKGROUND WATCHDOG
@app.on_event("startup")
async def startup_event():
    global sync_queue
    sync_queue = asyncio.Queue()
    # asyncio keeps only a weak reference to a task: hold these or the watchdog can be garbage collected
    # and the models then stay in VRAM for good.
    BACKGROUND_TASKS.add(asyncio.create_task(_sync_worker()))
    BACKGROUND_TASKS.add(asyncio.create_task(idle_watchdog()))

async def idle_watchdog():
    while True:
        await asyncio.sleep(5)
        try:
            await _idle_check()
        except Exception as exc:        # one failed unload must not end the watchdog
            print(f"⚠️ Idle watchdog error (will retry): {exc}")


async def _idle_check():
    if state.whisper_model is not None:
        if (time.time() - state.whisper_last) > state.config.get('idle_timeout_seconds', 60):
            if state.whisper_lock.acquire(blocking=False):
                try:
                    _free_whisper()
                finally:
                    state.whisper_lock.release()

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
                cpu_threads = state.config.get('cpu_threads', 0)

                print(f"\n⚡ Waking up model: {model_name}...")

                state.tokenizer, state.translator = load_model(
                    model_id=model_name,
                    device=device,
                    compute_type=compute_type,
                    cpu_threads=cpu_threads
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

def _free_whisper():
    state.whisper_model = None
    state.whisper_key = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _transcribe_segment(video: str, start: float, seconds: float, language: Optional[str]) -> dict:
    """Whisper on one window of a video, the way an audio sync does it: transcribe, repair the zones
    that look broken, then align the words to the audio. Times are relative to the window.

    Blocking (GPU work); run it in a thread. The model stays loaded for the next window.
    """
    from ..utils.whisper import load_whisper_model, run_whisper_transcription

    cfg = state.config
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is not installed on the Anchor host")
    lang = (language or "").strip().lower() or None
    target = cfg.get("model_size") or "large-v3"
    if lang == "en" and target in {"tiny", "base", "small", "medium"}:
        target = f"{target}.en"
    began = time.time()
    with state.whisper_lock:
        fd, wav = tempfile.mkstemp(prefix="anchor-seg-", suffix=".wav")
        os.close(fd)
        try:
            subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-y", "-ss", f"{max(start, 0):.3f}",
                            "-t", f"{seconds:.3f}", "-i", video, "-vn", "-ac", "1", "-ar", "16000",
                            "-acodec", "pcm_s16le", wav], check=True, capture_output=True, timeout=180)
            with wave.open(wav, "rb") as fh:
                audio_sec = fh.getnframes() / float(fh.getframerate() or 16000)
            if audio_sec <= 0:
                raise RuntimeError("no audio in this window")
            key = (target, lang)
            loaded_now = state.whisper_model is None or state.whisper_key != key
            if loaded_now:
                _free_whisper()
                state.whisper_model = load_whisper_model(cfg["device"], cfg["compute_type"], lang, target)
                state.whisper_key = key
            data, detected = run_whisper_transcription(
                wav, cfg["device"], cfg["compute_type"], cfg["batch_size"], state.whisper_model, lang,
                model_name=target, cpu_threads=cfg.get("cpu_threads", 0))
        except subprocess.CalledProcessError as exc:
            raise RuntimeError("ffmpeg could not read the audio: " + (exc.stderr or b"").decode("utf-8", "replace")[-200:])
        finally:
            state.whisper_last = time.time()
            try:
                os.unlink(wav)
            except OSError:
                pass
    if data is None:
        raise RuntimeError("Whisper could not transcribe this window")
    words = []
    for seg in data:
        for w in seg.get("words") or []:
            text = (w.get("word") or "").strip()
            if text and w.get("start") is not None and w.get("end") is not None:
                words.append({"w": text, "s": float(w["start"]), "e": float(w["end"])})
    return {"words": words, "audio_sec": audio_sec, "language": detected, "model": target,
            "device": cfg["device"], "model_loaded": loaded_now, "repair": True,
            "seconds_taken": round(time.time() - began, 2)}


@app.post("/transcribe-segment", dependencies=[Depends(_require_token)])
async def transcribe_segment_endpoint(req: SegmentRequest):
    """Word timings for one window of a video (for a sync check). Same Whisper pipeline as /sync, no subtitle."""
    if not state.config.get("allowed_dirs"):
        raise HTTPException(status_code=403, detail="api_server.allowed_dirs is empty in ~/.anchor/config.json; /transcribe-segment is disabled")
    video = Path(req.video)
    if not video.is_absolute():
        raise HTTPException(status_code=400, detail="video must be an absolute path")
    if not _allowed(video):
        raise HTTPException(status_code=403, detail="video is outside api_server.allowed_dirs")
    if not video.is_file():
        raise HTTPException(status_code=404, detail=f"video not found: {video}")
    if not (1 <= req.seconds <= 600) or req.start < 0:
        raise HTTPException(status_code=400, detail="seconds must be between 1 and 600 and start not negative")
    if any(j.get("state") == "running" for j in JOBS.values()):
        raise HTTPException(status_code=409, detail="an audio sync is running on the GPU; try again when it ends")
    await _unload_translator()
    state.last_active = time.time()
    try:
        return await asyncio.get_running_loop().run_in_executor(
            None, _transcribe_segment, str(video), req.start, req.seconds, req.language)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.post("/sync", dependencies=[Depends(_require_token)])
async def sync_endpoint(req: SyncRequest):
    if not state.config.get("allowed_dirs"):
        raise HTTPException(status_code=403, detail="api_server.allowed_dirs is empty in ~/.anchor/config.json; /sync is disabled")
    video = Path(req.video)
    sub = Path(req.subtitle)
    for label, p in (("video", video), ("subtitle", sub)):
        if not p.is_absolute():
            raise HTTPException(status_code=400, detail=f"{label} must be an absolute path")
        if not _allowed(p):
            raise HTTPException(status_code=403, detail=f"{label} is outside api_server.allowed_dirs")
        if not p.is_file():
            raise HTTPException(status_code=404, detail=f"{label} not found: {p}")

    job_id = uuid.uuid4().hex[:12]
    job = {"id": job_id, "state": "queued", "video": str(video), "subtitle": str(sub),
           "overwrite": req.overwrite, "created": time.time(), "progress": "", "result": None, "error": None}
    JOBS[job_id] = job
    JOB_ORDER.append(job_id)
    while len(JOB_ORDER) > MAX_JOBS:
        old = JOB_ORDER.pop(0)
        if JOBS.get(old, {}).get("state") in ("done", "error"):
            JOBS.pop(old, None)
        else:
            JOB_ORDER.append(old)
            break
    await sync_queue.put(job_id)
    return _public(job)


@app.get("/jobs/{job_id}", dependencies=[Depends(_require_token)])
async def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job")
    return _public(job)


# THE LAUNCHER
def run_apimode(args, device, model_size, compute_type, batch_size, translation_model, console, config, cpu_threads=0):
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
        "model_size": model_size,
        "translation_model": translation_model,
        "cpu_threads": cpu_threads,
        "idle_timeout_seconds": timeout,
        "token": config['api_server'].get('token', ''),
        "allowed_dirs": config['api_server'].get('allowed_dirs', [])
    }
    
    uvicorn.run(app, host=host, port=port, log_level="warning")