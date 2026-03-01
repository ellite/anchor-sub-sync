import struct
from pathlib import Path

def hash_file(file_path: Path | str) -> str:
    """
    Generates the official OpenSubtitles hash for a video file.
    Reads the first 64kb and last 64kb of the file to create a 64-bit checksum.
    """
    file_path = Path(file_path)
    
    try:
        file_size = file_path.stat().st_size
    except FileNotFoundError:
        return ""

    chunk_size = 65536
    if file_size < chunk_size * 2:
        return ""

    file_hash = file_size

    with open(file_path, 'rb') as f:
        # Read the first 64kb and unpack it instantly into 8192 unsigned 64-bit integers
        buffer_head = f.read(chunk_size)
        file_hash += sum(struct.unpack('<8192Q', buffer_head))

        # Seek to the last 64kb and repeat
        f.seek(file_size - chunk_size)
        buffer_tail = f.read(chunk_size)
        file_hash += sum(struct.unpack('<8192Q', buffer_tail))

    # Apply the 64-bit mask once at the end to simulate C-style integer overflow
    file_hash &= 0xFFFFFFFFFFFFFFFF
    
    # Return as a 16-character zero-padded hex string
    return f"{file_hash:016x}"