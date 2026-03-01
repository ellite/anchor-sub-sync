import logging
import warnings
import sys

# Ensure logging handlers write to the real stderr FD so temporary redirections
# (PTY replacement) don't cause handler writes to a closed file descriptor.
logging.basicConfig(stream=sys.__stderr__)

__version__ = "1.10.0"

# ================= NOISE SUPPRESSION =================
warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")
warnings.filterwarnings("ignore", message=".*This deprecation is part.*")
warnings.filterwarnings("ignore", message=".*was trained.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio|speechbrain")
logging.getLogger("speechbrain").setLevel(logging.ERROR)
logging.getLogger("pyannote").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)
# =====================================================