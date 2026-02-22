import json
from pathlib import Path
import subprocess


def get_video_info(video_path: Path):
    """Uses ffprobe to extract the video codec and bitrate."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,bit_rate:format=bit_rate",
        "-of", "json", str(video_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        stream = data.get("streams", [{}])[0]
        fmt = data.get("format", {})

        codec = stream.get("codec_name", "h264")
        
        # Try to get stream bitrate first, fallback to overall format bitrate
        bitrate = stream.get("bit_rate") or fmt.get("bit_rate")
        return codec, bitrate
    except Exception:
        return "h264", None