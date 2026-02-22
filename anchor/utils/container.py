import json
import subprocess

def get_subtitle_streams(file_path):
    """
    Scans a media file using ffprobe and returns a list of subtitle streams.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-of", "json",
        "-show_entries", "stream=index,codec_name:stream_tags=language,title:stream_disposition=hearing_impaired,forced",
        "-select_streams", "s",
        file_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        return data.get("streams", [])
    except subprocess.CalledProcessError:
        return None
    except FileNotFoundError:
        # Catch if ffprobe is missing from the system path
        return False