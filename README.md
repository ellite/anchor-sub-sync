<div align="center">
  <picture>
    <img src="https://github.com/ellite/anchor-sub-sync/raw/main/resources/anchorlogo.png" alt="Anchor logo" width="160"
  </picture>

  <p>Anchor Sub Sync</p>

  [![Stars](https://img.shields.io/github/stars/ellite/anchor-sub-sync?style=flat-square)](https://github.com/ellite/anchor-sub-sync)
  [![Downloads](https://img.shields.io/pepy/dt/anchor-sub-sync?style=flat-square&color=blue)](https://pepy.tech/project/anchor-sub-sync)
  [![PyPi](https://img.shields.io/pypi/v/anchor-sub-sync.svg)](https://pypi.python.org/pypi/anchor-sub-sync/)
  [![GitHub contributors](https://img.shields.io/github/contributors/ellite/anchor-sub-sync?style=flat-square)](https://github.com/ellite/anchor-sub-sync/graphs/contributors)
  [![GitHub Sponsors](https://img.shields.io/github/sponsors/ellite?style=flat-square)](https://github.com/sponsors/ellite)
</div>

# ⚓ Anchor Sub Sync

**Anchor** is a GPU-accelerated tool that automatically synchronizes subtitle files (.srt, .ass) to video files using audio alignment. It uses OpenAI's Whisper (via WhisperX) to listen to the video track, applies some alignment techniques and perfectly align every subtitle line.


⚡ Core Capabilities

- 🔊 **Audio Sync**: Auto-align subtitles to video using Whisper (no reference text needed).
- 📑 **Reference Sync**: Automatic Sync using a perfectly timed reference subtitle
- 📍 **Point Sync**: Fix linear drift by matching distinct lines against a reference subtitle.
- 🌐 **Translation**: Context-aware translation using NLLB, with dual-speaker preservation and auto-formatting.
- 📝 **Transcriptions**: Generate subtitles directly from the video.
- 📦 **Container Tasks**: Extract, Embed, or Strip subtitles from media
- 🔥 **Burn-in**: Permanently burn subtitles into video
- 🧽 **Clean & Fix**:  Repair and clean subtitle files
- 🔄 **Convert**: Convert between subtitle formats
- 📥 **Download**: Automatically find and download matching subtitles
- 🔌 **API Mode**: Expose Anchor's translation engine as a local API server, compatible with Subtitle Edit's Auto-Translate feature.

## 🚀 Requirements

* **OS:** Linux | Windows | MacOS
* **GPU:** GPU Recommended, CPU only is possible (but slower)
* **Python:** 3.10 or 3.11
* **ffmpeg / ffprobe:** required for audio probing, metadata language detection, and audio extraction

## 📦 Installation

### 1. Install Anchor
Install the tool directly from the repository.

```bash
pip install anchor-sub-sync
```

**⚠️ Important:** Because this tool relies on hardware acceleration, standard `pip install` often pulls the wrong drivers (CPU versions). If it does not work out of the box, please follow these steps in order.

### 2. Install PyTorch (choose per hardware)

Before installing PyTorch, run a few quick checks to detect your hardware:

```bash
# NVIDIA / CUDA
nvidia-smi || true
python3 -c "import torch; print('cuda', torch.cuda.is_available(), getattr(torch.cuda, 'get_device_name', lambda i: '')(0))"

# Apple Silicon (MPS)
python3 -c "import torch; print('mps', torch.backends.mps.is_available())"
```

Then install the appropriate build:

- NVIDIA GPU with CUDA drivers installed:

```bash
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121
```

- Apple Silicon (MPS) - follow PyTorch MPS instructions (example):

```bash
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/mps
```

- AMD / ROCm: follow the official PyTorch ROCm install instructions for your distribution (see https://pytorch.org).

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.1
```

> ⚠️ **AMD GPU acceleration requires a ROCm-compatible CTranslate2 build.** The standard `pip install ctranslate2` only supports NVIDIA CUDA and CPU. Anchor will automatically detect this at startup and fall back to CPU if the standard build is installed. To enable full AMD GPU acceleration, you must build CTranslate2 with ROCm support from source — see the [AMD CTranslate2 ROCm guide](https://rocm.blogs.amd.com/artificial-intelligence/ctranslate2/README.html). Without it, Anchor still works correctly on CPU.

- CPU-only systems:

```bash
pip install torch torchaudio torchvision
```

Notes:
- Only install the CUDA wheel if you have an NVIDIA GPU and matching drivers; installing a GPU wheel on a CPU-only system can produce subtle runtime errors.
- If you encounter the `torchvision::nms` error, force-reinstall matching PyTorch/TorchVision wheels (see Troubleshooting below).

### 3. Finalize Dependencies (Critical)
Some libraries may downgrade during installation. Run this command to ensure the GPU translation engine is up to date and compatible with your drivers:

```bash
pip install --upgrade ctranslate2
```

> ⚠️ **AMD ROCm users:** Skip this step if you have built CTranslate2 from source with ROCm support. Running `pip install --upgrade ctranslate2` will replace your custom build with the standard CUDA-only wheel, reverting GPU acceleration back to CPU.

## 🔌 API Mode (Subtitle Edit Integration)

Anchor can run as a local translation API server, compatible with **Subtitle Edit's Auto-Translate** feature. This lets you use Anchor's NLLB translation engine directly from within Subtitle Edit.

Start the API server with:

```bash
anchor --api
```

The server will idle with no VRAM usage and automatically load the translation model on the first request, unloading it again after the configured idle timeout.

### ⚙️ Configuration

The API server is configured via `~/.anchor/config.json`:

```json
"api_server": {
    "host": "127.0.0.1",
    "port": 6060,
    "idle_timeout_seconds": 60
}
```

| Setting | Description |
| :--- | :--- |
| `host` | `127.0.0.1` for local access only. Change to `0.0.0.0` to allow access from other machines on the network. |
| `port` | The port the API listens on. |
| `idle_timeout_seconds` | How long (in seconds) the model stays in VRAM after the last request before being unloaded. |

### 🖥️ Subtitle Edit Setup

1. In Subtitle Edit, open **Auto-translate** (Video → Auto-translate).
2. In the engine dropdown, select **thammegowda-nllb-serve**.
3. Set the URL to `http://<host>:<port>/translate` (e.g. `http://192.168.2.195:6060/translate` if accessing from another machine).
4. Select your source and target languages and click **Translate**.

> ⚠️ If running Anchor on a different machine than Subtitle Edit, make sure to set `host` to `0.0.0.0` in the config.

### 🔄 Running in the Background

**Linux — using `nohup`:**
```bash
nohup anchor --api > ~/.anchor/api.log 2>&1 &
echo $! > ~/.anchor/api.pid
```

To stop it:
```bash
kill $(cat ~/.anchor/api.pid)
```

**Linux — as a systemd service** (runs on boot):

First, find the full path to the anchor binary:
```bash
which anchor
```

Create `/etc/systemd/system/anchor-api.service`:
```ini
[Unit]
Description=Anchor Subtitle API
After=network.target

[Service]
User=YOUR_USERNAME
ExecStart=/home/YOUR_USERNAME/.local/bin/anchor --api
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Replace `YOUR_USERNAME` and the `ExecStart` path with the output from `which anchor`. Then enable it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now anchor-api
```

**Windows — using PowerShell:**
```powershell
Start-Process anchor --api -WindowStyle Hidden
```

## How It Works - Under the Hood

A short summary: Anchor uses WhisperX plus a multi-stage elastic alignment pipeline (phoneme alignment, global fuzzy alignment, rolling-window drift correction, and timeline interpolation) to produce millisecond-accurate subtitle sync.

- **Transcription:** WhisperX for high-quality transcripts + forced alignment.
- **Phoneme Alignment:** Maps audio to phonemes for word-level timing.
- **Global Alignment:** Fuzzy matching to find anchors between script and audio.
- **Drift Correction:** Rolling-window filter to correct long-term drift.
- **Cleanup:** Interpolation + overlap-fixer ("Zipper") for clean subtitles.

<details>
<summary>Full explanation (expand)</summary>

Many syncing tools rely on simple waveform matching or standard speech-to-text, which often results in "drift" (where subtitles slowly get out of sync) or fails when the actor changes a line.

The engine combines WhisperX with a multi-stage Elastic Alignment Algorithm. Here is what is happening during those processing steps and why it matters.

1. Why WhisperX instead of Faster-Whisper?
Standard faster-whisper is incredible at transcription, but it often "hallucinates" timestamps. It guesses roughly when a sentence started, often missing by 0.5 to 1.0 seconds.

WhisperX adds a crucial post-processing step called Forced Alignment.

The Difference: Instead of just listening for words, it maps the audio to specific Phonemes (the distinct units of sound, like the "t" in "cat").

The Result: The tool doesn't just know what was said; it identifies word-level timing precision down to the millisecond. This provides the bedrock for perfect sync.

2. The Alignment Logic (Step-by-Step)
Once transcription is complete, the custom syncing pipeline merges the provided script with the actual audio.

📏 Aligning Phonemes: The system maps the raw audio sounds to the text characters. This ensures that even if the actor speaks quickly or mumbles, the tool catches the exact moment the sound occurs.

📐 Global Alignment (e.g., 6153 vs 6063 words): Scripts rarely match the final audio perfectly. Actors add-lib, scenes get cut, or words are repeated.

Standard tools fail here because they look for a 1:1 match.

This tool performs a "fuzzy match" global alignment, mathematically calculating the best fit between the two datasets even when the word counts differ.

🔍 Rolling Window Drift Filter: Over a long video, audio timing can "drift" (often due to frame rate conversions like 23.976fps vs 24fps). This step analyzes the timeline in moving segments to detect and correct gradual desynchronization before it becomes noticeable to the viewer.

⚓️ Valid Anchors (e.g., 618 Anchors): The system identifies "Anchors"-points of absolute certainty where the audio and text match perfectly with high confidence.

Rejection: It automatically discards "Outliers" (matches that seem statistically unlikely or erroneous), ensuring the timeline is pinned down only by high-quality data.

🔨 Reconstructing Timeline (Interpolation): Using the Anchors as fixed distinct points, the system mathematically "stretches" or "compresses" the text between them. This ensures that the dialogue between the perfect matches flows naturally and stays in sync.

🧹 Running The Zipper (Overlap Cleanup): Subtitle overlaps are messy and hard to read. "The Zipper" is a final polish pass that detects when two subtitle events collide. It dynamically adjusts the start/end times to ensure one line finishes exactly as the next one begins, resolving conflicts automatically.

Why use this over other tools?
Most tools treat subtitles as a static block of text. This system treats them as a dynamic, elastic timeline. By using Phoneme-level anchoring combined with Drift Correction, the tool can sync messy, imperfect scripts to audio with a precision that manual matching simply cannot achieve.

</details>

## 🌍 Automatic Cross-Language Sync

Anchor 1.1+ can now synchronize subtitles even when they are in a different language than the audio (for example, English audio with Portuguese subtitles). This is done by creating a temporary "ghost" translation of your subtitles into the audio language, syncing that translation to the audio, and then applying the improved timestamps back to your original file.

<details>
<summary>How it works (expand)</summary>

1. **Detection:** Anchor detects a mismatch between the audio language and the subtitle file language (e.g., Audio: EN, Subtitle: PT).

2. **Translation:** Anchor uses a fast neural translation model (NLLB via CTranslate2) to create a temporary translated subtitle file in the audio language.

3. **Sync:** The temporary "ghost" translation is synchronized against the audio track using the normal alignment pipeline.

4. **Restoration:** The accurate timestamps are transferred back to your original subtitle file, preserving the original text while fixing timing.

The result is your original subtitle text, timed accurately to the foreign-language audio.

</details>

### 🧠 Translation Models

Anchor uses NLLB-200 (No Language Left Behind) via the `ctranslate2` engine for high speed and low memory usage. You can override the automatic choice with the `-t` / `--translation-model` flag.

Note: The first time you use a translation model it will be downloaded automatically (approx. 1GB - 3.5GB depending on model). Models are cached locally for reuse.

Supported Translation Models
----------------------------

The following pre-built CTranslate2 models are supported and can be passed directly to `-t`:

- `JustFrederik/nllb-200-distilled-600M-ct2-int8`
- `OpenNMT/nllb-200-distilled-1.3B-ct2-int8`
- `OpenNMT/nllb-200-3.3B-ct2-int8`

For convenience, Anchor also supports shorthand names that map to reasonable defaults:

| Shorthand | Model |
| --------- | ----- |
| `small` | `JustFrederik/nllb-200-distilled-600M-ct2-int8` |
| `medium` | `OpenNMT/nllb-200-distilled-1.3B-ct2-int8` |
| `large` | `OpenNMT/nllb-200-3.3B-ct2-int8` |

Example usages:

```bash
anchor -t small
anchor --translation-model OpenNMT/nllb-200-1.3B-ct2-int8
```

And ensure you have enough disk space for model downloads.

## ⚡ Performance Test with model large-v3

- GPU (NVIDIA RTX 2000E): synced a 44-minute episode in ~82 seconds.
- CPU (Intel i5-12600H): synced the same 44-minute episode in ~16 minutes.
- Notes: Results vary by hardware, drivers, and model precision. Haven't tested other devices extensively; it should work on AMD and Intel GPUs as well.

### ⚡ Performance Benchmarks

*Tests performed on a 44-minute video file (English). With music and gun fire scenes.* **Hardware:** NVIDIA RTX 2000E Ada Generation (16 GB)

| Model | Translation | Time | Speed | Anchors Found | User Score | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **Large-v3** | 3.3B | 98s | ~27x | 514 | 10 / 10 | Translation of 669 lines added 16s. |
| **Large-v3** | N/A | 82s | ~32x | 600 | 10 / 10 | The gold standard for accuracy, but slower. |
| **Medium.en** | N/A | 62s | ~42x | 598 | 10 / 10 | Best Balance. Perfect sync, identical to Large-v3 but 25% faster. |
| **Small.en** | N/A | 44s | ~60x | 603 | 9.9 / 10 | Fastest usable. Twice as fast as Large-v3. Minor sync drift (<0.1s). |
| **Base.en** | N/A | 35s | ~75x | 585 | 9.7 / 10 | Very fast, but prone to drift on non-speech sounds (e.g., gunshots). |

### 📉 CPU Performance Benchmarks

*Tests performed on a 44-minute video file (English).* **Hardware:** 12th Gen Intel Core i5-12600H

| Model | Translation | Time | Speed | Anchors Found | User Score | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Large-v3** | N/A | ~16 min | ~2.7x | 599 | 10 / 10 | *Baseline.* Too slow for daily use on CPU. |
| **Medium.en** |	1.3B | 12.5 min | ~3.5x | 518 | 10 / 10 | Translation of 669 lines added ~60s. |
| **Medium.en** | N/A | 11.5 min | ~3.8x | 595 | 10 / 10 | Accurate, but the extra 6-minute wait didn't add much value. |
| **Small.en** | N/A | 5.7 min | ~7.6x | 601 | 10 / 10 | 2x faster than Medium with *better* anchor detection. |
| **Base.en** | N/A | 4.1 min | ~11x | 579 | 9.7 / 10 | Decent, but not significantly better than Tiny to justify the extra time. |
| **Tiny.en** | N/A | 3.7 min | ~12x | 572 | 9.5 / 10 | Incredibly fast. Good overall sync, but slightly less precise start times (ms delay). |

## 📥 Subtitle Download

Automatically or manually download subtitles for a specific file or all files in the directory.

Available Providers:
- SubDL (If credentials set on the config file)
- OpenSubtitles (If credentials set on the config file)
- Addic7ed 
- Podnapisi (Disabled by default. It slows down every subtitles search significantly without providing extra value. Can be enabled on the config file)

### 🎯 Smart Scoring System

When searching for subtitles, Anchor doesn't just blindly download the first result. It uses a weighted scoring engine to analyze your local video filename against the API results, bubbling the perfect match to the top of the list and burying incompatible files.

Here is a simplified breakdown of how the scoring works:

| Match Criteria | Score Impact | Description |
| :--- | :---: | :--- |
| **Hash Match** | `+ High` | The Holy Grail. The video file hash matches the subtitle hash exactly, guaranteeing perfect sync. |
| **Source** | `+ Medium` | Matches the source type (e.g., `WEB`, `BluRay`). |
| **Release Group** | `+ Medium` | Matches the specific ripper/group. |
| **Cut** | `+ Medium` | Matches the specific cut (e.g., `Theatrical`, `Extended`). |
| **Network** | `+ Medium` | Matches the specific network. |
| **SDH/Forced Preference** | `+/-` | Awards or deducts points based on your `prefer_sdh` and `prefer_forced` configuration setting. |
| **Wrong Episode** | `-100` | **Instant Rejection.** If the season or episode numbers do not match perfectly, it is heavily penalized to prevent downloading the wrong file. |

## 🛠️ Troubleshooting

### "Aborted" (Core Dump) on Launch
If the tool crashes instantly without an error message, it is a library path issue common on Linux. Run this command before starting `anchor`:

```bash
export LD_LIBRARY_PATH=$(python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))')
```
*(Tip: Add the line above to your `~/.bashrc` file to make it permanent.)*

### "Operator torchvision::nms does not exist"
This means you have a mismatch between PyTorch and TorchVision (usually one is CPU and one is GPU). Fix it by forcing a reinstall:

```bash
pip install --force-reinstall torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```

### "Numpy is not compatible"
If you see errors related to `numpy` (e.g., `module 'numpy' has no attribute...`), downgrade it to a stable version:

```bash
pip install "numpy<2.0" "pandas<2.0"
```

### _curses missing on windows ###
```bash
pip install windows-curses
```

### AMD GPU: Transcription/Translation Running on CPU Instead of GPU

Anchor detected your AMD GPU but is running on CPU. This is expected behaviour if the standard `pip install ctranslate2` wheel is installed, as it has no ROCm support.

To verify which mode Anchor selected, check the startup output:
- ✅ `ROCm-compatible CTranslate2 detected. GPU acceleration enabled.` — GPU is active.
- ⚠️ `CTranslate2 has no ROCm support in this install. Falling back to CPU.` — CPU fallback is active.

To enable GPU acceleration on AMD GPU, build CTranslate2 with ROCm from source:
```bash
# See full instructions at:
# https://rocm.blogs.amd.com/artificial-intelligence/ctranslate2/README.html
```

Additionally, on some distros (e.g. Fedora), `torchaudio` may fail to find ROCm libraries. Fix it by exporting:
```bash
export LD_LIBRARY_PATH=$(python3 -c 'import torch; import os; print(os.path.dirname(torch.__file__))')/lib:$LD_LIBRARY_PATH
```
*(Add to `~/.bashrc` to make it permanent.)*

## 🎬 Usage

Navigate to the folder containing your subtitles and video files, then run:

```bash
anchor
```

Command line options
--------------------

You can override the automatic hardware detection or control specific settings using flags:

| Option | Alias | Description |
| ------ | ----- | ----------- |
| --audio-model | -a | Force a specific Whisper model (e.g., tiny, medium, large-v3-turbo). |
| --batch-size | -b | Manually set the batch size (e.g., 8, 16). Useful for optimizing VRAM usage. |
| --translation-model | -t | Force a specific translation model (overrides automatic selection). |
| --subtitle | -s | Runs unattended sync on a single subtitle file (provide path to .srt, .ass, etc.) |
| --reference | -r | For unattended sync, provide reference subtitle file path for reference sync |
| --video | -v | For unattended sync, provide path to the video file if the script fails to auto-match |
| --overwrite | -o | Will overwrite the synced subtitle instead of saving it as file.synced.srt |
| --help | -h  | Show the help message and exit. |
| --language | -l | For unattended mode, provide the target language code (e.g. 'en', 'pt', 'fr') for translation or download |
| --download | -d | For unattended mode, automatically download subtitles. Provide -v with the video file path, or anchor downloads subtitles for all videos in the directory. |

Examples
--------

Force a specific model:

```bash
anchor --audio-model large-v3-turbo
```

Force batch size (to prevent crashes or increase speed):

```bash
anchor --batch-size 8
```

Combine flags:

```bash
anchor -a medium -b 16
```

Run unattended sync:

```bash
anchor -s A.3.Minutes.Example.Video.en.srt -v A.3.Minutes.Example.Video.mkv 
```

Run unattended reference sync:

```bash
anchor -s A.3.Minutes.Example.Video.en.srt -r A.3.Minutes.Example.Video.pt.srt 
```
Run unattended translation:

```bash
anchor -s A.3.Minutes.Example.Video.en.srt -l pt 
```

Run unattended download for all files:

```bash
anchor -d
```

Run unattended download for a specific file:

```bash
anchor -d -v A.3.Minutes.Example.Video.mkv
```

Run unattended download for all files for specific languages:

```bash
anchor -d -l en,fr
```

## ⚙️ Development

To modify the code locally:

```bash
git clone [https://github.com/ellite/anchor-sub-sync.git](https://github.com/ellite/anchor-sub-sync.git)
cd anchor-sub-sync
pip install -e .
```
## 🖼️ Screenshots 

<p align="center">
  <img src="./resources/screenshot.png" alt="Anchor screenshot" width="800" />
</p>

---

## Contributors

<a href="https://github.com/ellite/scrob/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ellite/scrob" />
</a>

---

## ❌ Uninstallation

If you installed Anchor via `pip` (GitHub), uninstall with:

```bash
pip uninstall anchor-sub-sync
```

If you used an editable install during development (`pip install -e .`), also remove the cloned repository directory and any virtual environment you created:

```bash
# from the project directory
deactivate  # if in a venv
rm -rf venv/  # or the name of your virtualenv
rm -rf anchor-sub-sync/  # delete the cloned repo if desired
```

To remove model caches or large downloaded files, delete the relevant cache directories (varies by model/backends), for example:

```bash
rm -rf ~/.cache/whisper ~/.cache/whisperx
```

## ⚓ Links
- [GitHub](https://github.com/ellite/anchor-sub-sync)
- [PyPI](https://pypi.org/project/anchor-sub-sync/)