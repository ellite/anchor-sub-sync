import subprocess
import tempfile
import re
import time
from pathlib import Path
import pysubs2
from PIL import Image
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

try:
    import easyocr
except ImportError:
    easyocr = None

def is_image_blank(img_path: Path):
    """Instantly checks if an image is completely transparent."""
    try:
        with Image.open(img_path) as img:
            return img.convert("RGBA").getbbox() is None
    except Exception:
        return True

def are_images_identical(img1_path: Path, img2_path: Path) -> bool:
    """Checks if two images are pixel-perfect identical to catch .sup duplicate frames."""
    try:
        with Image.open(img1_path) as im1, Image.open(img2_path) as im2:
            # Fast fail if they aren't even the same resolution
            if im1.size != im2.size:
                return False
            # Compare the raw pixel bytes
            return im1.tobytes() == im2.tobytes()
    except Exception:
        return False        
    
def clean_ocr_text(text: str) -> str:
    """Fixes common OCR punctuation hallucinations."""
    
    # Fix broken ellipses
    ellipses_fixes = {
        "___": "...",
        "__.": "...",
        "_..": "...",
        ".._": "...",
        ".__": "...",
        "._.": "...",
        "_._": "...",
        "_.": "...",
        "._": "...",
    }
    for bad, good in ellipses_fixes.items():
        text = text.replace(bad, good)
        
    # Fix trailing underscore at the end of a line or file
    # This regex looks for an underscore right before a line break (\N) or the end of the string
    text = re.sub(r"_+(?=\\N|$)", ".", text)
    
    # Fix trailing underscore at the end of a word mid-sentence
    # This looks for an underscore attached to a letter/number, right before a space
    text = re.sub(r"(?<=[a-zA-Z0-9])_+(?=\s)", ".", text)

    # Fix trailing colon at the end of a line or file
    # Looks for a colon (and any accidental trailing spaces) right before a line break or string end
    text = re.sub(r":\s*(?=\\N|$)", ".", text)
    
    return text.strip()    

def add_solid_background(img_path: Path):
    """
    Replaces transparent pixels with a solid black background.
    This stops EasyOCR from hallucinating edge-characters like 'l' and 'I'.
    """
    try:
        with Image.open(img_path) as img:
            # Check if the image has a transparency layer (Alpha)
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                img = img.convert("RGBA")
                
                # Create a solid black canvas the exact same size as our subtitle
                bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
                
                # Paste the subtitle onto the black canvas, using its own transparency as the glue
                bg.paste(img, (0, 0), img)
                
                # Convert to standard RGB (stripping away the transparency channel forever)
                final_img = bg.convert("RGB")
                
                # Overwrite the image on the hard drive
                final_img.save(img_path)
    except Exception as e:
        pass
    
def extract_subtitle_images(file_path: Path, temp_dir: str, console):
    """
    Uses FFmpeg to extract subtitle frames to PNGs and logs the exact PTS of every frame.
    Uses the timestamp of blank 'clear' frames to calculate perfect durations.
    """
    input_file = file_path
    if input_file.suffix.lower() == ".sub":
        input_file = input_file.with_suffix(".idx")
        if not input_file.exists():
            console.print(f"[bold red]❌ Critical Error: Missing {input_file.name}![/bold red]")
            return []

    # Render all PNGs and use `showinfo` to log the exact PTS of every generated frame
    img_pattern = str(Path(temp_dir) / "frame_%04d.png")
    
    ffmpeg_cmd = [
        "ffmpeg", "-y", 
        "-i", str(input_file),
        "-vsync", "0", 
        "-filter_complex", "[0:s:0]format=rgba,showinfo[out]",
        "-map", "[out]",
        "-c:v", "png",
        img_pattern
    ]
    
    console.print(f"   [dim]🎞️ Rendering frames and analyzing timestamps for {input_file.name}...[/dim]")
    
    # Changed spinner to "dots" for UI consistency
    with console.status(" [cyan]Rendering binary frames to PNG (This may take a minute)...[/cyan]", spinner="dots"):
        process = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        
    if process.returncode != 0:
        console.print(f"[bold red]❌ FFmpeg Error:[/bold red]\n{process.stderr}")
        return []

    # Parse the showinfo log to get the exact PTS for every single frame
    pts_times = []
    for line in process.stderr.splitlines():
        if "showinfo" in line and "pts_time:" in line:
            match = re.search(r"pts_time:\s*([0-9.]+)", line)
            if match:
                pts_times.append(float(match.group(1)))

    exported_images = sorted(list(Path(temp_dir).glob("*.png")))
    
    # Safety Check
    if len(pts_times) != len(exported_images):
        console.print(f"   [yellow]⚠️ Internal map error: {len(pts_times)} times but {len(exported_images)} images.[/yellow]")
        min_len = min(len(pts_times), len(exported_images))
        pts_times = pts_times[:min_len]
        exported_images = exported_images[:min_len]

    # Match the timestamps, calculate true durations, and filter duplicates!
    extracted_data = []
    current_event = None
    blank_count = 0
    duplicate_count = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
        transient=True # Disappears when finished
    ) as progress:
        
        analyze_task = progress.add_task(f"[cyan]Filtering {len(exported_images)} extracted frames...", total=len(exported_images))
        
        for img_path, pts in zip(exported_images, pts_times):
            is_blank = is_image_blank(img_path)
            
            if is_blank:
                blank_count += 1
                if current_event:
                    current_event["end"] = pts
                    extracted_data.append(current_event)
                    current_event = None
                    
                img_path.unlink()
            else:
                if current_event and are_images_identical(current_event["img_path"], img_path):
                    duplicate_count += 1
                    img_path.unlink()
                else:
                    if current_event:
                        current_event["end"] = pts - 0.1
                        extracted_data.append(current_event)
                        
                    current_event = {
                        "start": pts,
                        "end": None,
                        "img_path": img_path
                    }
                    
            progress.advance(analyze_task)

    if current_event:
        current_event["end"] = current_event["start"] + 2.0
        extracted_data.append(current_event)

    console.print(f"   [dim]✅ Processed {len(exported_images)} total frames ({blank_count} clear frames, {duplicate_count} identical duplicates removed).[/dim]")

    return extracted_data

def run_ocr_engine(file_path: Path, target_ext: str, console, device: str):
    """Handles extracting images from binary subtitle formats and running OCR."""
    if easyocr is None:
        console.print("\n[bold red]❌ EasyOCR is not installed![/bold red]")
        return

    # --- START FILE TIMER ---
    file_start_time = time.perf_counter()

    console.print(f"   [cyan]⚙️ Initializing OCR Pipeline on {device.upper()}...[/cyan]")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # --- Image Extraction ---
        try:
            extracted_frames = extract_subtitle_images(file_path, temp_dir, console)
            console.print(f"   [bold green]✓[/bold green] Found {len(extracted_frames)} valid text frames.")
        except Exception as e:
            console.print(f"   [bold red]❌ Failed to extract images:[/bold red] {e}")
            return

        if not extracted_frames:
            console.print("   [yellow]⚠️ No text frames were extracted. Is this file empty?[/yellow]")
            return

        # --- OCR Brain ---
        console.print("\n   [dim]👁️ Loading AI Vision model (EasyOCR)...[/dim]")
        
        use_gpu = device in ["cuda", "xpu"]
        with console.status("   [dim]🧠 Loading vision models into memory...[/dim]", spinner="dots"):
            reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)

        subs = pysubs2.SSAFile()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
            transient=True
        ) as progress:
            
            task = progress.add_task(f" [cyan]🔎 Scanning {len(extracted_frames)} frames...", total=len(extracted_frames))
            
            for frame in extracted_frames:
                # Slide a solid black background behind the text!
                add_solid_background(frame["img_path"])
                
                # Read raw text boxes
                results = reader.readtext(str(frame["img_path"]))
                
                # Cluster boxes into horizontal lines
                lines_dict = []
                
                for bbox, text, conf in results:
                    y_center = (bbox[0][1] + bbox[2][1]) / 2
                    x_center = (bbox[0][0] + bbox[1][0]) / 2
                    height = bbox[2][1] - bbox[0][1]
                    
                    placed = False
                    for line in lines_dict:
                        if abs(line['y_center'] - y_center) < (height * 0.5):
                            line['words'].append({'text': text.strip(), 'x': x_center})
                            line['y_center'] = (line['y_center'] * len(line['words']) + y_center) / (len(line['words']) + 1)
                            placed = True
                            break
                            
                    if not placed:
                        lines_dict.append({
                            'y_center': y_center,
                            'words': [{'text': text.strip(), 'x': x_center}]
                        })
                
                # Sort lines top-to-bottom
                lines_dict.sort(key=lambda l: l['y_center'])
                
                final_lines = []
                for line in lines_dict:
                    # Sort words within the line left-to-right
                    line['words'].sort(key=lambda w: w['x'])
                    final_lines.append(" ".join([w['text'] for w in line['words']]))
                    
                final_text = "\\N".join(final_lines)
                
                # Run our cleanup filter to fix punctuation!
                final_text = clean_ocr_text(final_text)
                
                if final_text.strip():
                    start_ms = int(frame["start"] * 1000)
                    end_ms = int(frame["end"] * 1000)
                    
                    event = pysubs2.SSAEvent(start=start_ms, end=end_ms, text=final_text)
                    subs.append(event)
                    
                progress.advance(task)

        # --- Save Output ---
        output_path = file_path.with_suffix(target_ext)
        subs.save(str(output_path))
        
        # --- STOP FILE TIMER AND FORMAT ---
        file_elapsed = time.perf_counter() - file_start_time
        mins, secs = divmod(file_elapsed, 60)
        time_str = f"{int(mins)}m {secs:.1f}s" if mins > 0 else f"{file_elapsed:.1f}s"
        
        console.print(f"   [bold green]🔄 OCR Complete in {time_str}! Saved as:[/bold green] {output_path.name}")