import logging
import warnings
import sys

# Ensure logging handlers write to the real stderr FD so temporary redirections
# (PTY replacement) don't cause handler writes to a closed file descriptor.
logging.basicConfig(stream=sys.__stderr__)

__version__ = "1.16.1"

# ================= TORCHAUDIO COMPAT SHIM =================
# torchaudio 2.11+ removed AudioMetaData and torchaudio.info; pyannote 3.x needs both.
import torchaudio as _torchaudio
if not hasattr(_torchaudio, "AudioMetaData"):
    from dataclasses import dataclass

    @dataclass
    class _AudioMetaData:
        sample_rate: int
        num_frames: int
        num_channels: int
        bits_per_sample: int
        encoding: str

    _torchaudio.AudioMetaData = _AudioMetaData

if not hasattr(_torchaudio, "info"):
    import soundfile as _sf

    def _torchaudio_info(path, format=None, backend=None):
        info = _sf.info(path)
        # subtype is like "PCM_16", "PCM_24", "FLOAT" — extract the numeric part
        import re as _re
        bps_match = _re.search(r'(\d+)', info.subtype)
        bits_per_sample = int(bps_match.group(1)) if bps_match else 0
        return _torchaudio.AudioMetaData(
            sample_rate=info.samplerate,
            num_frames=info.frames,
            num_channels=info.channels,
            bits_per_sample=bits_per_sample,
            encoding=info.subtype,
        )

    _torchaudio.info = _torchaudio_info

if not hasattr(_torchaudio, "list_audio_backends"):
    _torchaudio.list_audio_backends = lambda: ["soundfile"]
# ===========================================================

# ================= NOISE SUPPRESSION =================
warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")
warnings.filterwarnings("ignore", message=".*This deprecation is part.*")
warnings.filterwarnings("ignore", message=".*was trained.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio|speechbrain")
logging.getLogger("speechbrain").setLevel(logging.ERROR)
logging.getLogger("pyannote").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)
# =====================================================