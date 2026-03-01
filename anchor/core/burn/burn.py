import subprocess
import threading
import queue
from pathlib import Path

from ...utils.files import get_files, find_best_video_match, select_video_fallback, select_files_interactive
from ...utils.video import get_video_info


def _build_output_name(video_path: Path, suffix=".burn"):
    base = video_path.stem
    out = video_path.with_name(f"{base}{suffix}{video_path.suffix}")
    counter = 1
    while out.exists():
        out = video_path.with_name(f"{base}{suffix}_{counter}{video_path.suffix}")
        counter += 1
    return out


def run_burn(args, device, console):
    """Batch burn-in subtitles onto video files."""

    # Pick subtitle files
    subs = get_files({".srt", ".ass", ".vtt", ".sub"})
    if not subs:
        console.print("[bold red]❌ No subtitle files found in this folder.[/bold red]")
        return

    selected = select_files_interactive(subs, header_lines=["Select subtitle file(s) to burn into video:"], multi_select=True)
    if not selected:
        console.print("[yellow]No files selected. Aborting burn.[/yellow]")
        return

    # Build work items (sub, video)
    work_items = []
    for sub in selected:
        vid = find_best_video_match(sub)
        if not vid:
            console.print(f"[dim]No auto-match for [cyan]{sub.name}[/cyan]. Asking for video...[/dim]")
            vid = select_video_fallback(sub.name)

        if vid:
            work_items.append((sub, vid))
        else:
            console.print(f" [yellow]Skipping {sub.name} — no video selected.[/yellow]")

    if not work_items:
        console.print("[bold red]❌ No valid subtitle/video pairs to process.[/bold red]")
        return

    console.print(f"\n[bold green]🚀 Starting Burn-in ({len(work_items)} items)...[/bold green]")

    q = queue.Queue()
    for item in work_items:
        q.put(item)

    def _worker():
        while not q.empty():
            sub, vid = q.get()
            console.print(f"\n🔥 Burning: [cyan]{sub.name}[/cyan] into [yellow]{vid.name}[/yellow]")
            out_file = _build_output_name(vid)

            # --- METADATA EXTRACTION ---
            console.print(" 🔍 Analyzing original video codec and quality...")
            orig_codec, orig_bitrate = get_video_info(vid)
            
            # --- PATH ESCAPING FOR FFMPEG FILTER ---
            # FFmpeg's subtitles filter is very strict. Replace \ with / and escape colons/quotes.
            sub_escaped = str(sub.absolute()).replace('\\', '/').replace(':', '\\:').replace("'", "'\\''")
            subtitles_filter = f"subtitles='{sub_escaped}'"
            vf = subtitles_filter

            # --- CODEC & DEVICE MAPPING ---
            vcodec = "libx264"
            encoder_args = []

            if device == "cuda":
                if orig_codec == "hevc": vcodec = "hevc_nvenc"
                elif orig_codec == "av1": vcodec = "av1_nvenc"
                else: vcodec = "h264_nvenc"
                encoder_args = ["-preset", "p4"]

            elif device == "xpu":
                if orig_codec == "hevc": vcodec = "hevc_qsv"
                elif orig_codec == "av1": vcodec = "av1_qsv"
                else: vcodec = "h264_qsv"
                vf = f"{subtitles_filter},format=nv12,hwupload"
                encoder_args = ["-preset", "medium"]

            else: # CPU Fallback
                if orig_codec == "hevc": vcodec = "libx265"
                elif orig_codec == "vp9": vcodec = "libvpx-vp9"
                else: vcodec = "libx264"
                encoder_args = ["-preset", "medium"]

            # --- QUALITY / BITRATE TARGETING ---
            encoder_args = ["-c:v", vcodec] + encoder_args
            
            if orig_bitrate:
                console.print(f" 🎯 Matching original format: [bold]{orig_codec.upper()}[/bold] at [bold]{int(orig_bitrate)//1000} kbps[/bold]")
                # Use standard ABR targeting to match file size
                encoder_args.extend([
                    "-b:v", str(orig_bitrate), 
                    "-maxrate", str(int(orig_bitrate) * 1.5), 
                    "-bufsize", str(int(orig_bitrate) * 2)
                ])
            else:
                console.print(f" 🎯 Matching original format: [bold]{orig_codec.upper()}[/bold] [dim](Using High-Quality Fallback)[/dim]")
                # Fallback if bitrate is missing: High Quality Constant Rate
                if "nvenc" in vcodec: encoder_args.extend(["-cq", "22", "-rc", "vbr"])
                elif "qsv" in vcodec: encoder_args.extend(["-global_quality", "22"])
                else: encoder_args.extend(["-crf", "22"])

            cmd = [
                "ffmpeg", "-hide_banner", "-y", "-v", "error",
                "-i", str(vid),
                "-vf", vf,
                *encoder_args,
                "-c:a", "copy",
                str(out_file)
            ]

            try:
                # Using rich status spinner because encoding takes time!
                with console.status("🎬 Burning subtitles into video stream...", spinner="dots"):
                    subprocess.run(cmd, check=True)
                console.print(f" [bold green]✅ Burn complete:[/bold green] {out_file.name}")
            except subprocess.CalledProcessError as e:
                console.print(f" [bold red]❌ ffmpeg failed for {vid.name}:[/bold red] Ensure the codec is supported by your hardware.")
            except Exception as e:
                console.print(f" [bold red]❌ Unexpected error:[/bold red] {e}")

            q.task_done()

    # Single-threaded worker preserves simplicity and avoids ffmpeg contention
    t = threading.Thread(target=_worker)
    t.start()
    t.join()

    console.print(f"\n[bold green]✨ Burn-in batch complete.[/bold green]")