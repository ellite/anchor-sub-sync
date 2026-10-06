"""Optional Parakeet speech recognition engine for audio sync (`--asr parakeet|auto|whisper`).

NVIDIA Parakeet TDT 0.6B v3 gives word timestamps about 2.5x faster than WhisperX and placed cues more accurately in
our tests, but it covers only 25 European languages and drops short shouted lines in loud scenes. Anchor therefore
keeps Whisper for everything else and uses Parakeet only to time subtitles that already exist.

Parakeet runs in its own virtual environment (~/.anchor/parakeet-venv, override with ANCHOR_PARAKEET_VENV) so that
onnxruntime and its CUDA libraries never touch the torch / whisperx installation. `--asr parakeet` creates it on
first use; `--asr auto` only uses it when it already exists.
"""
import bisect
import collections
import copy
import gc
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .formatting import clean_text

# Languages Parakeet TDT 0.6B v3 supports (ISO 639-1).
SUPPORTED_LANGUAGES = {
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu", "it",
    "lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk",
}

ENV_VENV = "ANCHOR_PARAKEET_VENV"
MODEL_CACHE = "models--istupakov--parakeet-tdt-0.6b-v3-onnx"

COMMON_PACKAGES = ["onnx-asr[hub]==0.12.0", "soundfile", "numpy"]
CPU_PACKAGES = ["onnxruntime==1.23.1"]
# onnxruntime >= 1.30 refuses the Hugging Face cache layout for the model's external data file; 1.23.x works.
GPU_PACKAGES = [
    "onnxruntime-gpu==1.23.2", "nvidia-cudnn-cu12", "nvidia-cublas-cu12", "nvidia-cuda-runtime-cu12",
    "nvidia-cufft-cu12", "nvidia-curand-cu12", "nvidia-cuda-nvrtc-cu12",
]
INSTALL_SIZE_GPU = "about 3.4 GB"
INSTALL_SIZE_CPU = "about 0.3 GB"
MODEL_SIZE = "about 3 GB"

SEGMENT_GAP_SEC = 0.8          # a pause longer than this ends a segment
MIN_SILENCE_SEC = 4.0          # a stretch with no Parakeet words at least this long may hide dropped speech
CUE_REACH_SEC = 3.0            # a cue's expected position is only an estimate: it counts as inside a stretch within this
ZONE_REACH_SEC = 12.0          # re-transcribe at most this far either side of the cue inside a stretch
ZONE_MERGE_SEC = 4.0           # repair zones closer than this are merged


# ------------------------------------------------------------------ environment

def venv_dir():
    return Path(os.environ.get(ENV_VENV) or Path.home() / ".anchor" / "parakeet-venv")


def venv_python():
    base = venv_dir()
    return base / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _marker():
    return venv_dir() / "anchor-parakeet.json"


def is_installed():
    return venv_python().exists() and _marker().exists()


def _installed_is_gpu():
    try:
        return bool(json.loads(_marker().read_text()).get("gpu"))
    except Exception:
        return False


def supports_language(lang):
    return bool(lang) and lang.lower() in SUPPORTED_LANGUAGES


def wants_gpu(device):
    """The GPU build is NVIDIA CUDA on Linux. Windows would need a system CUDA/cuDNN install, ROCm and Apple use the CPU build."""
    if device != "cuda" or platform.system() != "Linux":
        return False
    try:
        import torch
        return not torch.version.hip
    except Exception:
        return False


def install(console, gpu):
    """Creates the isolated environment and installs Parakeet's dependencies into it. Raises RuntimeError on failure."""
    vdir = venv_dir()
    packages = COMMON_PACKAGES + (GPU_PACKAGES if gpu else CPU_PACKAGES)
    size = INSTALL_SIZE_GPU if gpu else INSTALL_SIZE_CPU
    console.print(f"[bold cyan]🦜 Installing Parakeet[/bold cyan] ({'GPU' if gpu else 'CPU'} build, {size} of packages "
                  f"into [underline]{vdir}[/underline]; the model, {MODEL_SIZE}, downloads on first use)")
    if vdir.exists():
        shutil.rmtree(vdir)
    try:
        subprocess.run([sys.executable, "-m", "venv", str(vdir)], check=True, capture_output=True, text=True)
        with console.status("[dim]Installing packages (this can take a few minutes)...[/dim]"):
            subprocess.run([str(venv_python()), "-m", "pip", "install", "--disable-pip-version-check", "-q", *packages],
                           check=True, capture_output=True, text=True)
        _marker().write_text(json.dumps({"gpu": gpu, "packages": packages}))
    except subprocess.CalledProcessError as e:
        shutil.rmtree(vdir, ignore_errors=True)
        tail = (e.stderr or e.stdout or "").strip().splitlines()[-3:]
        raise RuntimeError("Parakeet install failed: " + " | ".join(tail))
    console.print("[green]✅ Parakeet installed.[/green]")


def ensure_ready(console, device, install_if_missing):
    """True when Parakeet can run. Installs it first when allowed. Never raises: failure means 'use Whisper'."""
    gpu = wants_gpu(device)
    try:
        if is_installed():
            if install_if_missing and gpu and not _installed_is_gpu():
                install(console, gpu=True)      # a CPU build is installed but a CUDA GPU is available
            return True
        if not install_if_missing:
            return False
        install(console, gpu)
        return True
    except Exception as e:
        console.print(f"[yellow]⚠️ Parakeet is not available ({e}). Using Whisper.[/yellow]")
        return False


def _gpu_library_path():
    """The pip-installed CUDA libraries inside the environment, for LD_LIBRARY_PATH (Linux)."""
    dirs = sorted(str(p) for p in venv_dir().glob("lib/python*/site-packages/nvidia/*/lib"))
    return ":".join(dirs)


def _model_cached():
    return (Path.home() / ".cache" / "huggingface" / "hub" / MODEL_CACHE).exists()


# ------------------------------------------------------------------ transcription

def words_from_tokens(tokens, timestamps, step=0.08):
    """Parakeet emits sub-word tokens with start times; a token that starts with a space begins a new word."""
    words = []
    for token, stamp in zip(tokens, timestamps):
        if token.startswith(" ") or not words:
            words.append({"word": token.strip(), "start": float(stamp), "end": float(stamp) + step})
        else:
            words[-1]["word"] += token
            words[-1]["end"] = float(stamp) + step
    return [w for w in words if w["word"]]


def segments_from_words(words):
    """Groups words into sentence-like segments (a pause or sentence end closes one) in the format the aligner expects."""
    groups, current = [], []
    for w in words:
        if current and (w["start"] - current[-1]["end"] > SEGMENT_GAP_SEC or current[-1]["word"][-1:] in ".?!"):
            groups.append(current)
            current = []
        current.append(w)
    if current:
        groups.append(current)
    return [{"start": g[0]["start"], "end": g[-1]["end"], "text": " ".join(w["word"] for w in g), "words": g} for g in groups]


def transcribe(video_path, device, console, on_progress=None):
    """Transcribes a video with Parakeet. Returns (segments, duration_seconds). Raises RuntimeError on failure.

    `on_progress(stage, done, total)` is called as work advances: stage is "audio" (extracting), "model" (loading) or
    "chunks" (done of total windows of audio transcribed).
    """
    notify = on_progress or (lambda *a: None)
    gpu = wants_gpu(device) and _installed_is_gpu()
    work = Path(tempfile.mkdtemp(prefix="anchor-parakeet-"))
    try:
        wav, out = work / "audio.wav", work / "out.json"
        notify("audio", 0, None)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000",
                        "-acodec", "pcm_s16le", str(wav)], check=True, capture_output=True)
        env = os.environ.copy()
        if gpu:
            libs = _gpu_library_path()
            env["LD_LIBRARY_PATH"] = libs + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        first_run = not _model_cached()
        if first_run:
            console.print(f"[dim]🦜 First run: downloading the Parakeet model ({MODEL_SIZE})...[/dim]")
        worker = Path(__file__).with_name("parakeet_worker.py")
        notify("model", 0, None)
        # stderr is shown as it is on the first run (the model download prints its own progress) and otherwise
        # merged into stdout, so a noisy worker can never fill a pipe nobody is reading.
        proc = subprocess.Popen([str(venv_python()), "-I", str(worker), str(wav), str(out), "gpu" if gpu else "cpu"],
                                env=env, stdout=subprocess.PIPE, stderr=None if first_run else subprocess.STDOUT, text=True)
        tail = collections.deque(maxlen=5)
        for line in proc.stdout:
            parts = line.split()
            if len(parts) == 3 and parts[0] == "CHUNKS":
                notify("chunks", int(parts[1]), int(parts[2]))
            elif line.strip():
                tail.append(line.strip())
        if proc.wait() != 0 or not out.exists():
            raise RuntimeError("Parakeet failed" + (": " + " | ".join(list(tail)[-3:]) if tail else ""))
        data = json.loads(out.read_text(encoding="utf-8"))
        return segments_from_words(words_from_tokens(data["tokens"], data["timestamps"])), data["duration"]
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"audio extraction failed: {(e.stderr or b'').decode(errors='ignore')[-200:]}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def detect_language(segments):
    """Language of a transcript, for files with no audio language tag (None when it cannot tell)."""
    try:
        from langdetect import detect
        text = " ".join(s["text"] for s in segments)[:2000]
        return detect(text) if text.strip() else None
    except Exception:
        return None


# ------------------------------------------------------------------ subtitle-guided repair

def _silent_stretches(segments, duration):
    """Stretches longer than MIN_SILENCE_SEC where Parakeet heard nothing, head and tail of the file included."""
    edges = [(0.0, segments[0]["start"])] if segments else [(0.0, duration)]
    edges += [(segments[i]["end"], segments[i + 1]["start"]) for i in range(len(segments) - 1)]
    if segments:
        edges.append((segments[-1]["end"], duration))
    return [(a, b) for a, b in edges if b - a > MIN_SILENCE_SEC]


def _missing_speech_zones(segments, unanchored, duration):
    """Time windows to re-transcribe: the silent stretches that hold a cue which did not anchor.

    A cue that stays unanchored although Parakeet has words around it is a text mismatch, and re-transcribing would
    not help. A cue whose expected position falls inside a stretch where Parakeet heard nothing is most likely speech
    Parakeet dropped (shouts, loud scenes). The expected position is only an estimate, so cues match a stretch they are
    near (CUE_REACH_SEC), and each zone is limited to ZONE_REACH_SEC around the cue so a long musical silence is not
    transcribed in full.
    """
    silences = _silent_stretches(segments, duration)
    zones = []
    for cue in unanchored:
        for a, b in silences:
            if a - CUE_REACH_SEC <= cue["end"] and cue["start"] <= b + CUE_REACH_SEC:
                zones.append((max(a, cue["start"] - ZONE_REACH_SEC, 0.0), min(b, cue["end"] + ZONE_REACH_SEC, duration)))
    zones = [z for z in zones if z[1] > z[0]]
    zones.sort()
    merged = []
    for z in zones:
        if merged and z[0] - merged[-1][1] < ZONE_MERGE_SEC:
            merged[-1] = (merged[-1][0], max(merged[-1][1], z[1]))
        else:
            merged.append(z)
    return merged


def repair_missing_speech(segments, subs, video_path, duration, language, model_name, device, compute_type, cpu_threads, console):
    """Re-transcribes, with faster-whisper loaded once, only the windows where the subtitle has cues Parakeet did not hear.

    `subs` is the subtitle that will be aligned (the translated copy when the languages differ). Shows a progress bar
    (this is the slow part of a sync: a few seconds per zone). Returns (segments, zone_count, recovered_word_count);
    the input is returned unchanged when nothing needs repair.
    """
    from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
    from . import alignment
    from .alignment import GlobalAligner

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(),
                  TaskProgressColumn(), TimeElapsedColumn(), console=console, transient=True) as progress:
        task = progress.add_task("[cyan]Checking which cues Parakeet missed...", total=None)

        # A first alignment shows which cues did not anchor. The aligner prints its own steps: silence it for this pass.
        work = copy.deepcopy(subs)
        aligner = GlobalAligner(work, segments)
        was_quiet, alignment.console.quiet = alignment.console.quiet, True
        try:
            synced, _ = aligner.run()
        finally:
            alignment.console.quiet = was_quiet
        if synced is None:
            return segments, 0, 0
        unanchored = []
        for i, event in enumerate(work):
            if i in aligner.anchored_idx or event.is_comment:
                continue
            words = len(clean_text(event.text).split())
            if words:
                unanchored.append({"start": event.start / 1000.0, "end": event.end / 1000.0, "words": words})
        zones = _missing_speech_zones(segments, unanchored, duration)
        if not zones:
            return segments, 0, 0

        import whisperx
        from ..core.transcribe import transcribe as T

        progress.update(task, description=f"[cyan]Re-transcribing {len(zones)} zone(s) Parakeet missed...", total=len(zones), completed=0)
        fw = T.WhisperModel(model_name, device=device, compute_type=compute_type, cpu_threads=cpu_threads)
        try:
            found = []
            for start, end in zones:
                repaired = T.repair_zone_best(model_name, device, compute_type, video_path, [{"start": start, "end": end, "text": ""}],
                                              T.REPAIR_PADDING_PASS_1, language, cpu_threads, whisper_model=fw)
                found += [seg for seg in repaired if (seg.get("text") or "").strip()]
                progress.advance(task)
        finally:
            del fw
            gc.collect()
            if device == "cuda":
                import torch
                torch.cuda.empty_cache()
        if not found:
            return segments, len(zones), 0

        progress.update(task, description="[cyan]Timing the recovered speech...", total=None)
        plain = [{"start": seg["start"], "end": seg["end"], "text": seg["text"]} for seg in found]
        try:
            align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
            audio = whisperx.load_audio(str(video_path))
            with alignment.quiet_library_logs():
                plain = whisperx.align(plain, align_model, metadata, audio, device, return_char_alignments=False)["segments"]
            T.trim_stretched_leading_words(plain)
            del align_model, audio
            gc.collect()
            if device == "cuda":
                import torch
                torch.cuda.empty_cache()
        except Exception:
            pass   # no alignment model for this language: the aligner spreads the words evenly across each segment

    recovered = [{"start": seg["start"], "end": seg["end"], "text": seg["text"], "words": seg.get("words", [])} for seg in plain]
    merged = sorted(segments + recovered, key=lambda seg: seg["start"])
    return merged, len(zones), sum(len(seg["words"]) for seg in recovered)
