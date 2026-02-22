import subprocess
from pathlib import Path
from ...utils.files import select_files_interactive, get_files, _run_curses_picker
from ...utils.languages import get_subtitle_language
from ...utils.container import get_subtitle_streams
from ...utils.mappings import normalize_language_code, ISO_639_MAPPING, get_iso_639_2_code

def run_container_tasks(args, container_mode, console):
    if container_mode == "extract":
        run_extract(args, console)
    elif container_mode == "embed":
        run_embed(args, console)
    elif container_mode == "strip":
        run_strip(args, console)
    else:
        console.print(f"[bold red]❌ Unknown container task: {container_mode}[/bold red]")
        return
    

def run_extract(args, console):
    console.print("[bold cyan]🧲 Running Extract Task[/bold cyan]\n")

    # Select Media Files using picker
    cwd = Path.cwd()
    media_files = list(cwd.glob("*.mkv")) + list(cwd.glob("*.mp4"))
    
    if not media_files:
        console.print(" [yellow]No .mkv or .mp4 files found in the current directory.[/yellow]")
        return

    selected_media = select_files_interactive(
        media_files, 
        header_lines=["Select Media Files to Extract From (Space to select, Enter to confirm)"], 
        multi_select=True
    )

    if not selected_media:
        return

    total_files = len(selected_media)
    console.print(f"[bold green]🚀 Starting Extraction ({total_files} file{'s' if total_files > 1 else ''})...[/bold green]")

    # Process Each Selected Media File
    for task_idx, media_path in enumerate(selected_media, 1):
        console.print(f"\n[black on white] Task {task_idx}/{total_files} [/black on white] [bold cyan]{media_path.name}[/bold cyan]")
        console.print(" 🔍 Probing media file...")
        
        streams = get_subtitle_streams(str(media_path))
        
        if not streams:
            console.print(" 🚫 [dim]No subtitle streams found in this file. Skipping.[/dim]")
            continue

        # Build UI options for the stream picker
        options = []
        for s in streams:
            idx = s.get("index")
            codec = s.get("codec_name", "unknown")
            lang = s.get("tags", {}).get("language", "und")
            disp = s.get("disposition", {})
            
            flags = []
            if disp.get("forced") == 1: flags.append("Forced")
            if disp.get("hearing_impaired") == 1: flags.append("SDH")
            flag_str = f" [{','.join(flags)}]" if flags else ""
            
            options.append(f"Track {idx}: {lang.upper()} ({codec}){flag_str}")

        # Pick the streams for this specific file
        selected_indices = _run_curses_picker(
            options, 
            title=f"Select Streams to Extract: {media_path.name}", 
            multi_select=True
        )

        if not selected_indices:
            console.print(" ⏭️  [dim]No streams selected. Skipping.[/dim]")
            continue

        # Extract the Selected Streams
        for i in selected_indices:
            stream = streams[i]
            idx = stream.get("index")
            codec = stream.get("codec_name", "unknown")
            disp = stream.get("disposition", {})
            
            lang = stream.get("tags", {}).get("language", "und")
            is_text = codec in ["subrip", "ass", "webvtt", "mov_text", "text"]
            
            if is_text:
                ext = ".srt"
                extract_codec = "srt"
            else:
                extract_codec = "copy"
                if codec == "hdmv_pgs_subtitle": ext = ".sup"
                elif codec in ["dvd_subtitle", "dvdsub"]: ext = ".sub"
                else: ext = f".{codec}"

            base_name = media_path.stem
            
            modifier = ""
            if disp.get("hearing_impaired") == 1:
                modifier = ".hi"
            elif disp.get("forced") == 1:
                modifier = ".forced"
            elif not is_text:
                modifier = f".track_{idx}"

            temp_file = Path(f"temp_extract_{idx}{ext}")
            cmd = [
                "ffmpeg", "-y", "-v", "error", 
                "-i", str(media_path), 
                "-map", f"0:{idx}", 
                "-c:s", extract_codec, 
                str(temp_file)
            ]
            
            console.print(f" 🔧  Extracting Track {idx}...")
            subprocess.run(cmd, check=True)

            # AI Language Detection
            if is_text and lang.lower() == "und":
                console.print(f" 🤖 [dim]Analyzing language for Track {idx}...[/dim]")
                detected_lang = get_subtitle_language(temp_file)
                if detected_lang and detected_lang != "unknown":
                    lang = detected_lang

            lang = normalize_language_code(lang)

            # Final Naming & Conflict Resolution
            final_name = f"{base_name}.{lang}{modifier}{ext}"
            final_path = cwd / final_name
            
            counter = 1
            while final_path.exists():
                final_name = f"{base_name}.{lang}{modifier}.{counter}{ext}"
                final_path = cwd / final_name
                counter += 1

            temp_file.rename(final_path)
            
            console.print(f" 💾 Saved to: [u]{final_name}[/u]")

    console.print("\n[bold green]✨ Extraction Complete![/bold green]")

def run_embed(args, console):
    console.print("[bold cyan]🧩 Running Embed Task[/bold cyan]\n")

    cwd = Path.cwd()

    # Select the TARGET Media File (Single Select)
    media_files = list(cwd.glob("*.mkv")) + list(cwd.glob("*.mp4"))
    if not media_files:
        console.print(" [yellow]No .mkv or .mp4 files found in the current directory.[/yellow]")
        return

    selected_media = select_files_interactive(
        media_files, 
        header_lines=["Select the TARGET Media File (Space to select, Enter to confirm)"], 
        multi_select=False
    )
    if not selected_media: return
    target_media = selected_media[0]

    # Select Subtitles to Embed (Multi Select)
    sub_files = list(cwd.glob("*.srt")) + list(cwd.glob("*.vtt")) + list(cwd.glob("*.ass"))
    if not sub_files:
        console.print(" [yellow]No subtitle files (.srt, .vtt, .ass) found to embed.[/yellow]")
        return

    selected_subs = select_files_interactive(
        sub_files, 
        header_lines=[f"Select Subtitles to Embed into: {target_media.name}"], 
        multi_select=True
    )
    if not selected_subs: return

    total_subs = len(selected_subs)
    console.print(f"\n[bold green]🚀 Starting Embedding ({total_subs} subtitle{'s' if total_subs > 1 else ''})...[/bold green]")

    # Analyze Existing Streams
    console.print(f"\n[black on white] Task 1/1 [/black on white] [bold cyan]{target_media.name}[/bold cyan]")
    console.print(" 🔍 Probing existing streams...")
    
    existing_streams = get_subtitle_streams(str(target_media))
    existing_sub_count = len(existing_streams) if existing_streams else 0
    is_mp4 = target_media.suffix.lower() == ".mp4"

    # Build the FFmpeg Command
    temp_output = target_media.with_name(f".temp_embed_{target_media.name}")
    
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(target_media)]
    for sub in selected_subs:
        cmd.extend(["-i", str(sub)])

    # Map all original streams
    cmd.extend(["-map", "0"])
    cmd.extend(["-c", "copy"])

    # Map the new subtitle streams and process metadata
    valid_iso_codes = set(ISO_639_MAPPING.keys()) | set(ISO_639_MAPPING.values())
    
    console.print(" 📋 Preparing metadata and language mappings...")
    for i, sub in enumerate(selected_subs):
        file_index = i + 1 # FFmpeg input index (0 is the video, 1 is the first sub)
        new_stream_idx = existing_sub_count + i # FFmpeg output stream index
        
        cmd.extend(["-map", f"{file_index}:0"]) # Safely map the first stream of the sub file
        
        # Determine Language
        lang = "und"
        for part in sub.stem.split('.'):
            if part.lower() in valid_iso_codes:
                lang = normalize_language_code(part.lower())
                break
        
        if lang == "und":
            console.print(f" 🤖 [dim]Analyzing language for {sub.name}...[/dim]")
            lang = normalize_language_code(get_subtitle_language(sub))

        container_lang = get_iso_639_2_code(lang)

        # Set Metadata
        cmd.extend([f"-metadata:s:s:{new_stream_idx}", f"language={container_lang}"])

        # Force Codec for MP4
        if is_mp4:
            cmd.extend([f"-c:s:{new_stream_idx}", "mov_text"])

    cmd.append(str(temp_output))

    # Execute and Overwrite
    console.print(" 🎬 Muxing tracks losslessly (this takes a few seconds)...")
    try:
        subprocess.run(cmd, check=True)
        # Safely overwrite the original file
        if temp_output.exists():
            temp_output.replace(target_media)
        
        console.print(f" 💾 Saved to: [u]{target_media.name}[/u] [dim](Overwritten)[/dim]")
        console.print("\n[bold green]✨ Embed Complete![/bold green]")
        
    except subprocess.CalledProcessError:
        console.print(f" ❌ [bold red]FFmpeg Error during embed.[/bold red]")
        if temp_output.exists():
            temp_output.unlink() # Cleanup failed temp file

def run_strip(args, console):
    console.print("[bold cyan]🧹 Running Strip Task[/bold cyan]\n")

    cwd = Path.cwd()

    # Select Media Files (Multi-Select allowed for batch cleaning)
    media_files = list(cwd.glob("*.mkv")) + list(cwd.glob("*.mp4"))
    if not media_files:
        console.print(" [yellow]No .mkv or .mp4 files found in the current directory.[/yellow]")
        return

    selected_media = select_files_interactive(
        media_files, 
        header_lines=["Select Media Files to Strip Subtitles From (Space to select, Enter to confirm)"], 
        multi_select=True
    )
    if not selected_media: return

    total_files = len(selected_media)
    console.print(f"\n[bold green]🚀 Starting Strip Task ({total_files} file{'s' if total_files > 1 else ''})...[/bold green]")

    # Process Each File
    for task_idx, media_path in enumerate(selected_media, 1):
        console.print(f"\n[black on white] Task {task_idx}/{total_files} [/black on white] [bold cyan]{media_path.name}[/bold cyan]")
        console.print(" 🔍 Probing existing streams...")
        
        streams = get_subtitle_streams(str(media_path))
        
        if not streams:
            console.print(" 🚫 [dim]No subtitle streams found in this file. Skipping.[/dim]")
            continue

        # Build UI options
        options = []
        for s in streams:
            idx = s.get("index")
            codec = s.get("codec_name", "unknown")
            lang = s.get("tags", {}).get("language", "und").upper()
            title = s.get("tags", {}).get("title", "")
            title_str = f" - {title}" if title else ""
            
            options.append(f"Track {idx}: {lang} ({codec}){title_str}")

        # Pick the streams TO DELETE
        selected_indices = _run_curses_picker(
            options, 
            title=f"Select Tracks to STRIP (Delete) from: {media_path.name}", 
            multi_select=True
        )

        if not selected_indices:
            console.print(" ⏭️  [dim]No tracks selected for deletion. Skipping file.[/dim]")
            continue

        # Build the Negative Map FFmpeg Command
        temp_output = media_path.with_name(f".temp_strip_{media_path.name}")
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(media_path)]
        
        # Start by mapping EVERYTHING from the original file
        cmd.extend(["-map", "0"])
        
        # Now apply NEGATIVE mapping for the tracks we want to strip
        for i in selected_indices:
            stream_idx = streams[i].get("index")
            cmd.extend(["-map", f"-0:{stream_idx}"])
            console.print(f" ✂️ Marking Track {stream_idx} for removal...")

        # Global copy to do it losslessly
        cmd.extend(["-c", "copy", str(temp_output)])

        # Execute and Overwrite
        console.print(" 🎬 Scrubbing tracks losslessly...")
        try:
            subprocess.run(cmd, check=True)
            if temp_output.exists():
                temp_output.replace(media_path)
            
            console.print(f" 💾 Saved to: [u]{media_path.name}[/u] [dim](Overwritten)[/dim]")
        except subprocess.CalledProcessError:
            console.print(f" ❌ [bold red]FFmpeg Error during strip.[/bold red]")
            if temp_output.exists():
                temp_output.unlink()

    console.print("\n[bold green]✨ Strip Complete![/bold green]")