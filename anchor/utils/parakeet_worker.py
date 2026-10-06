"""Runs Parakeet inside the isolated environment created by anchor.utils.parakeet.

This file is executed with that environment's Python (`python -I parakeet_worker.py <wav> <out.json> <gpu|cpu>`),
never imported by Anchor, so it may only use what the environment installs: onnx-asr, soundfile, numpy.

Long audio is cut into CHUNK_SEC windows with OVERLAP_SEC of overlap (the model has a length limit); each token is kept
from the window that owns its timestamp, so nothing is transcribed twice.
"""
import json
import math
import sys
import time

import onnx_asr
import soundfile as sf

MODEL = "nemo-parakeet-tdt-0.6b-v3"
CHUNK_SEC = 240.0
OVERLAP_SEC = 8.0


def main(wav_path, out_path, mode):
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if mode == "gpu" else ["CPUExecutionProvider"]
    started = time.time()
    model = onnx_asr.load_model(MODEL, providers=providers).with_timestamps()
    load_sec = time.time() - started

    audio, rate = sf.read(wav_path, dtype="float32")
    if rate != 16000:
        raise RuntimeError(f"expected 16 kHz audio, got {rate} Hz")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    duration = len(audio) / rate
    total_chunks = 1 if duration <= CHUNK_SEC else math.ceil((duration - OVERLAP_SEC) / (CHUNK_SEC - OVERLAP_SEC))
    print(f"CHUNKS 0 {total_chunks}", flush=True)

    tokens, stamps = [], []
    start, chunks = 0.0, 0
    started = time.time()
    while start < duration:
        end = min(start + CHUNK_SEC, duration)
        result = model.recognize(audio[int(start * rate):int(end * rate)], sample_rate=rate)
        keep_from = start + (OVERLAP_SEC / 2 if start > 0 else 0.0)
        keep_to = end - (OVERLAP_SEC / 2 if end < duration else 0.0) + (1e-6 if end >= duration else 0.0)
        for token, stamp in zip(result.tokens, result.timestamps):
            when = start + float(stamp)
            if keep_from <= when < keep_to:
                tokens.append(token)
                stamps.append(when)
        chunks += 1
        print(f"CHUNKS {chunks} {total_chunks}", flush=True)
        if end >= duration:
            break
        start = end - OVERLAP_SEC

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"tokens": tokens, "timestamps": stamps, "duration": duration, "chunks": chunks,
                   "load_sec": load_sec, "infer_sec": time.time() - started}, f)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
