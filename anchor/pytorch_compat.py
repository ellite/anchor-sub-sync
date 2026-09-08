import inspect

_applied = False


def apply_patches():
    """
    Applies runtime patches required before using torch / whisperx / pyannote.

    Idempotent and lazy: importing this module is cheap; `torch` and `torchaudio`
    are only imported when this function is actually called. Heavy commands call
    this before importing any whisper/pyannote code.
    """
    global _applied
    if _applied:
        return
    _applied = True

    _patch_torch_load()
    _apply_torchaudio_shim()


def _patch_torch_load():
    """Force weights_only=False on torch.load (safe for both new and old PyTorch)."""
    import torch

    _original_torch_load = torch.load

    # 'weights_only' was added in torch 1.13. Passing it on 1.12 would crash.
    arg_spec = inspect.getfullargspec(_original_torch_load)
    supports_weights_only = (
        "weights_only" in arg_spec.args or "weights_only" in arg_spec.kwonlyargs
    )

    if supports_weights_only:
        def patched_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return _original_torch_load(*args, **kwargs)

        torch.load = patched_load


def _apply_torchaudio_shim():
    """torchaudio 2.11+ removed AudioMetaData and torchaudio.info; pyannote 3.x needs both."""
    import torchaudio

    if not hasattr(torchaudio, "AudioMetaData"):
        from dataclasses import dataclass

        @dataclass
        class _AudioMetaData:
            sample_rate: int
            num_frames: int
            num_channels: int
            bits_per_sample: int
            encoding: str

        torchaudio.AudioMetaData = _AudioMetaData

    if not hasattr(torchaudio, "info"):
        import re as _re

        import soundfile as _sf

        def _torchaudio_info(path, format=None, backend=None):
            info = _sf.info(path)
            # subtype is like "PCM_16", "PCM_24", "FLOAT" - extract the numeric part
            bps_match = _re.search(r"(\d+)", info.subtype)
            bits_per_sample = int(bps_match.group(1)) if bps_match else 0
            return torchaudio.AudioMetaData(
                sample_rate=info.samplerate,
                num_frames=info.frames,
                num_channels=info.channels,
                bits_per_sample=bits_per_sample,
                encoding=info.subtype,
            )

        torchaudio.info = _torchaudio_info

    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]
