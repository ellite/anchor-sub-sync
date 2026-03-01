from pathlib import Path
from rich.console import Console
import re
import time
from ...utils.files import select_files_interactive, select_languages_interactive
from ...utils.selections import get_subtitle_mode
from .providers.opensubtitles import search_opensubtitles, download_opensubtitles, get_os_token
from .providers.podnapisi import search_podnapisi, download_podnapisi
from .providers.subdl import search_subdl, download_subdl
from .providers.addic7ed import search_addic7ed, download_addic7ed
from .scoring import calculate_score
from ...utils.hashing import hash_file
from ...utils.parsers import parse_video_filename

class SubtitleItem:
    def __init__(self, sub_dict):
        self.sub_dict = sub_dict
        score = sub_dict.get('score', 0)
        provider = sub_dict.get('provider', 'API')
        filename = sub_dict.get('filename', 'Unknown')
        lang = (sub_dict.get('language') or '??').upper() 
        
        # 1. Determine the score color dynamically
        if score > 0:
            score_color = "green"
        elif score < 0:
            score_color = "red"
        else:
            score_color = "white"
            
        # 2. Wrap the components in Rich markup tags!
        self.name = f"[{score_color}][★ {score:4d}][/{score_color}] [yellow][{lang:2s}][/yellow] [magenta][{provider}][/magenta] {filename}"

        
def run_download(args, config: dict, console: Console):
    console.print("\n[bold cyan]📥 Subtitle Downloader[/bold cyan]")

    # --- Check Credentials ---
    creds = config.get("credentials", {})
    os_api_key = creds.get("opensubtitles_api_key", "").strip()
    os_username = creds.get("opensubtitles_username", "").strip()
    os_password = creds.get("opensubtitles_password", "").strip()
    subdl_api_key = creds.get("subdl_api_key", "").strip()

    use_opensubtitles = False
    use_subdl = False
    use_podnapisi = False
    use_addic7ed = True
    if not os_api_key or not os_username or not os_password:
        console.print("[red]⚠️ OpenSubtitles credentials missing in ~/.anchor/config.json[/red]\n")
        console.print("[dim]   Opensubtitles will be unavailable.[/dim]")
    else:
        use_opensubtitles = True
        console.print("[bold green]✓[/bold green][dim] OpenSubtitles API configured.[/dim]")

    if not subdl_api_key:
        console.print("[yellow]⚠️ SubDL API key missing in ~/.anchor/config.json[/yellow]")
        console.print("[dim]   SubDL will be unavailable.[/dim]\n")
    else:
        use_subdl = True
        console.print("[bold green]✓[/bold green][dim] SubDL API configured.[/dim]")

    if not use_opensubtitles and not use_subdl and not use_podnapisi:
        console.print("[red]❌ No subtitle providers are properly configured. Please update your config in ~/.anchor/config.json and try again.[/red]")
        return
    
    # Gather all video files
    video_extensions = {".mkv", ".mp4", ".avi", ".mov", ".webm", ".ts"}
    available_videos = []
    
    # Check what the user passed into -v
    target_arg = getattr(args, 'video', None) 
    target_path = Path(target_arg) if target_arg else Path.cwd()

    if target_path.is_file():
        # If they pointed directly at ONE video, only process that video!
        if target_path.suffix.lower() in video_extensions:
            available_videos = [target_path]
    elif target_path.is_dir():
        # If they pointed at a folder (or passed nothing), scan the whole directory
        available_videos = [f for f in target_path.iterdir() if f.is_file() and f.suffix.lower() in video_extensions]

    if not available_videos:
        console.print(f"[yellow]⚠️ No video files found for target: {target_path.name}[/yellow]")
        return

    is_unattended = bool(getattr(args, 'download', None))

    if is_unattended:
        # UNATTENDED: Grab everything and force auto-match
        selected_files = available_videos
        download_mode = "auto"
        console.print(f"\n[bold green]⚡ Unattended Mode Activated:[/bold green] Processing {len(selected_files)} file(s) automatically.")
    else:
        # INTERACTIVE: Launch the TUI Picker
        header = [
            "Select the video files you want to fetch subtitles for.",
            "Use SPACE to select, ENTER to confirm. (Multi-select enabled)"
        ]
        
        selected_files = select_files_interactive(available_videos, header_lines=header, multi_select=True)

        if not selected_files:
            console.print("[yellow]⚠️ No video files selected. Returning to menu.[/yellow]")
            return

        console.print(f"\n[bold green]✓[/bold green] Selected {len(selected_files)} video file(s).")

        # Determine Interactive Download Mode
        download_mode = get_subtitle_mode(console)

    if not download_mode:
        console.print("[yellow]⚠️ Operation cancelled.[/yellow]")
        return

    console.print(f"\n[bold cyan]🚀 Starting {download_mode.upper()} download process...[/bold cyan]")

    # --- Prepare API & Languages ---
    # Get all configured languages, default to just English if missing
    target_langs_list = config.get("subtitle_preferences", {}).get("subtitle_languages", ["en"])

    # If unattended and user passed `-l/--language`, prefer that instead
    if is_unattended:
        lang_arg = getattr(args, 'language', None)
        if lang_arg:
            target_langs_list = [l.strip() for l in str(lang_arg).split(',') if l.strip()]

    # INTERACTIVE LANGUAGE FILTER (skip when unattended)
    if not is_unattended and len(target_langs_list) > 1:
        header = ["Select the languages you want to pull for this run:"]
        
        selected_langs = select_languages_interactive(target_langs_list, header_lines=header)
        
        if not selected_langs:
            console.print("   [yellow]⚠️ No languages selected. Aborting...[/yellow]")
            return
            
        target_langs_list = selected_langs

    target_langs_str = ",".join(target_langs_list)
    
    # Grab the SDH / Forced preference (defaults to False if not set)
    prefer_sdh = config.get("subtitle_preferences", {}).get("prefer_sdh", False)
    prefer_forced = config.get("subtitle_preferences", {}).get("prefer_forced", False)
    
    os_token = ""
    with console.status("   [dim]Authenticating with OpenSubtitles...[/dim]", spinner="dots"):
        os_token = get_os_token(os_api_key, os_username, os_password)
        
    if not os_token:
        console.print("   [red]❌ Authentication failed. Please check your username and password.[/red]")
        return

    # --- The Main Download Loop ---
    for file in selected_files:
        console.print(f"\n[cyan]🔍 Processing:[/cyan] {file.name}")
        
        file_hash = hash_file(file)
        parsed_data = parse_video_filename(file)
        
        results = []
        with console.status(f"   [dim]Searching subtitles ({target_langs_str})...[/dim]", spinner="dots"):
            for lang in target_langs_list:
                if use_opensubtitles:
                    results += search_opensubtitles(parsed_data, file_hash, lang, os_api_key)
                if use_subdl:
                    results += search_subdl(parsed_data, lang, subdl_api_key)
                if use_podnapisi:
                    results += search_podnapisi(parsed_data, file_hash, lang)
                if use_addic7ed:
                    results += search_addic7ed(parsed_data, file_hash, lang)    

            
        if not results:
            console.print("   [yellow]⚠️ No subtitles found for this release.[/yellow]")
            continue
            
        # --- SCORING ENGINE ---
        for sub in results:
            sub["score"] = calculate_score(parsed_data, sub, target_langs_list, prefer_sdh, prefer_forced)

        results.sort(key=lambda x: x["score"], reverse=True)
        # ------------------------------

        best_sub = None
        
        # Mode Fork
        subs_to_download = []
        
        if download_mode == "manual":
            sub_items = [SubtitleItem(r) for r in results]
            header = [
                f"Select subtitle(s) for: {file.name}",
                "Use SPACE to select, ENTER to confirm. (Multi-select enabled)"
            ]
            
            selected_items = select_files_interactive(sub_items, header_lines=header, multi_select=True)
            
            if not selected_items:
                console.print("   [dim]Skipped by user.[/dim]")
                continue
                
            for item in selected_items:
                subs_to_download.append(item.sub_dict)
            
        elif download_mode == "auto":
            # Find the absolute best subtitle for EACH configured language
            for lang in target_langs_list:
                lang_code = lang.lower()
                lang_results = [r for r in results if (r.get("language") or "").lower() == lang_code and r.get("score", -100) > 0]
                
                if lang_results:
                    best_sub = lang_results[0]
                    subs_to_download.append(best_sub)
                    console.print(f"   [dim]Auto-selected ({lang_code.upper()} - Score {best_sub['score']} - From {best_sub['provider']}): {best_sub['filename']}[/dim]")
                else:
                    console.print(f"   [dim]No suitable positive-scoring subtitles found for language: {lang_code.upper()}[/dim]")
            
        # Download & Save All Queued Subs
        for best_sub in subs_to_download:
            sub_lang = (best_sub.get('language') or 'en').lower()
            
            combined_text = (best_sub.get("filename", "") + " " + " ".join(best_sub.get("releases", []))).lower()
            is_sdh = bool(re.search(r'\b(hi|sdh|cc)\b', combined_text))
            is_forced = bool(re.search(r'\b(forced|foreign)\b', combined_text))
            
            # Build the base custom suffix without the extension
            modifier = ""
            if is_sdh:
                modifier += ".hi"
            if is_forced:
                modifier += ".forced"
                
            base_suffix = f".{sub_lang}{modifier}"
            final_suffix = f"{base_suffix}.srt"
            
            # --- COLLISION DETECTION ---
            # If the file exists on disk, keep bumping the number until we find a free name
            counter = 1
            final_path = file.with_suffix(final_suffix)
            
            while final_path.exists():
                final_suffix = f"{base_suffix}.{counter}.srt"
                final_path = file.with_suffix(final_suffix)
                counter += 1
            
            if best_sub["provider"] == "Podnapisi":
                with console.status(f"   [dim]Downloading {sub_lang.upper()} subtitle from Podnapisi...[/dim]", spinner="dots"):
                    success = download_podnapisi(best_sub["id"], file, custom_suffix=final_suffix)
            elif best_sub["provider"] == "OpenSubs":
                with console.status(f"   [dim]Downloading {sub_lang.upper()} subtitle from OpenSubtitles...[/dim]", spinner="dots"):
                    success = download_opensubtitles(best_sub["id"], file, os_api_key, os_token, custom_suffix=final_suffix)
            elif best_sub["provider"] == "SubDL":
                with console.status(f"   [dim]Downloading {sub_lang.upper()} subtitle from SubDL...[/dim]", spinner="dots"):
                    success = download_subdl(best_sub["id"], file, custom_suffix=final_suffix, episode=parsed_data.get("episode"))
            elif best_sub["provider"] == "Addic7ed":
                with console.status(f"   [dim]Downloading {sub_lang.upper()} subtitle from Addic7ed...[/dim]", spinner="dots"):
                    success = download_addic7ed(best_sub["id"], file, custom_suffix=final_suffix)
                
            if success:
                console.print(f"   [bold green]💾 Saved subtitle to:[/bold green] {final_path.name}")
            else:
                console.print(f"   [bold red]❌ Failed to download {sub_lang.upper()} subtitle.[/bold red]")

            # API Pacing (Be polite to the servers!)
            # Only pause if batch processing, and don't pause after the very last file
            if len(selected_files) > 1 and file != selected_files[-1]:
                time.sleep(1)    

    console.print("\n[bold green]🎉 All downloads complete![/bold green]\n")