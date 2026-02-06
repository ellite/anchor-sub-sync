import gc
import sys
import time
import os
import torch
import whisperx
import pysubs2
import difflib
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from .utils import clean_text, open_subtitle, make_ui_console, CaptureProgress

console = Console()

# ================= CONSTANTS =================
SCENE_GAP_SEC = 5.0          
MIN_DURATION_MS = 600        
GAP_MS = 50                  
OUTLIER_THRESHOLD_SEC = 1.5  
# =============================================

def smooth_offsets_by_block(anchors):
    if not anchors: return []
    console.print("[dim]   ⚖️ Applying Block Averaging (Smoothing)...[/dim]")
    
    scenes = []
    current_scene = [anchors[0]]
    
    for i in range(1, len(anchors)):
        prev = anchors[i-1]
        curr = anchors[i]
        
        if (curr['orig_start'] - prev['orig_start']) > SCENE_GAP_SEC:
            scenes.append(current_scene)
            current_scene = []
        current_scene.append(curr)
    scenes.append(current_scene)
    
    smoothed = []
    for scene in scenes:
        drifts = [(a['raw_match_time'] - a['orig_start']) for a in scene]
        avg_drift = np.median(drifts)
        for a in scene:
            a['final_start'] = a['orig_start'] + avg_drift
            smoothed.append(a)
    return smoothed

def enforce_strict_spacing(subs):
    console.print("[dim]   🧹 Running Zipper (Overlap Cleanup)...[/dim]")
    subs.sort()
    fix_count = 0
    
    for i in range(1, len(subs)):
        prev = subs[i-1]
        curr = subs[i]
        required_start = prev.end + GAP_MS
        
        if required_start > curr.start:
            new_prev_end = curr.start - GAP_MS
            prev_duration = new_prev_end - prev.start
            
            if prev_duration < MIN_DURATION_MS:
                prev.end = prev.start + MIN_DURATION_MS
                curr.start = prev.end + GAP_MS
                curr_dur = curr.end - curr.start
                curr.end = curr.start + curr_dur
            else:
                prev.end = new_prev_end
            fix_count += 1
            
    console.print(f"[dim]      ➡️ 🔧 Resolved {fix_count} overlaps.[/dim]")
    return subs

class GlobalAligner:
    def __init__(self, original_subs, whisper_data):
        self.subs = original_subs
        self.whisper = whisper_data
        
    def _tokenize_subs(self):
        sub_words = []
        for idx, sub in enumerate(self.subs):
            text = clean_text(sub.text) 
            words = text.split()
            for i, w in enumerate(words):
                if w.strip():
                    sub_words.append({"word": w.strip(), "sub_idx": idx})
        return sub_words

    def _tokenize_whisper(self):
        whisper_words = []
        for seg in self.whisper:
            if 'words' in seg and seg['words']:
                for w in seg['words']:
                    if 'start' in w:
                        whisper_words.append({"word": clean_text(w['word']), "start": w['start']})
            else:
                text = clean_text(seg['text'])
                words = text.split()
                if not words: continue
                duration = seg['end'] - seg['start']
                wd = duration / len(words)
                for i, w in enumerate(words):
                    whisper_words.append({"word": w.strip(), "start": seg['start'] + i*wd})
        return whisper_words

    def run(self):
        console.print("[dim]   🧩 Tokenizing data...[/dim]")
        sub_tokens = self._tokenize_subs()
        wh_tokens = self._tokenize_whisper()
        
        sub_strs = [x['word'] for x in sub_tokens]
        wh_strs = [x['word'] for x in wh_tokens]

        console.print(f"[dim]   📐 Global Alignment ({len(sub_strs)} vs {len(wh_strs)} words)...[/dim]")
        matcher = difflib.SequenceMatcher(None, sub_strs, wh_strs, autojunk=False)
        matches = matcher.get_matching_blocks()
        
        sub_matches = {i: [] for i in range(len(self.subs))}
        for match in matches:
            for i in range(match.size):
                sub_token = sub_tokens[match.a + i]
                wh_time = wh_tokens[match.b + i]['start']
                sub_matches[sub_token['sub_idx']].append(wh_time)

        candidates = []
        for idx in range(len(self.subs)):
            if sub_matches[idx]:
                sub = self.subs[idx]
                match_start = sub_matches[idx][0]
                drift = match_start - (sub.start / 1000.0)
                candidates.append({
                    'idx': idx, 'orig_start': sub.start/1000.0, 
                    'raw_match_time': match_start, 'drift': drift
                })

        if not candidates: return None, 0

        console.print("[dim]   🔍 Applying Rolling Window Drift Filter...[/dim]")
        raw_anchors = []
        rejected_count = 0
        window_size = 10 
        
        for i, cand in enumerate(candidates):
            start_i = max(0, i - window_size)
            end_i = min(len(candidates), i + window_size + 1)
            neighbors = [n['drift'] for n in candidates[start_i:end_i]]
            
            if abs(cand['drift'] - np.median(neighbors)) > OUTLIER_THRESHOLD_SEC:
                rejected_count += 1
            else:
                raw_anchors.append(cand)

        console.print(f"[dim]   ⚓️ Valid Anchors: {len(raw_anchors)} (Rejected {rejected_count} outliers)[/dim]")
        
        anchors = smooth_offsets_by_block(raw_anchors)
        
        console.print("[dim]   🔨 Reconstructing Timeline (Interpolation)...[/dim]")
        new_subs = pysubs2.SSAFile()
        new_subs.info = self.subs.info
        
        xp = [a['orig_start'] for a in anchors]
        fp = [a['final_start'] for a in anchors]
        
        for i, sub in enumerate(self.subs):
            orig = sub.start / 1000.0
            dur = (sub.end - sub.start) / 1000.0
            
            if len(anchors) > 0:
                if i < anchors[0]['idx']:
                    shift = anchors[0]['final_start'] - anchors[0]['orig_start']
                    new_start = orig + shift
                elif i > anchors[-1]['idx']:
                    shift = anchors[-1]['final_start'] - anchors[-1]['orig_start']
                    new_start = orig + shift
                else:
                    new_start = np.interp(orig, xp, fp)
            else:
                new_start = orig
            
            sub.start = int(max(0, new_start) * 1000)
            sub.end = int(max(0, new_start + dur) * 1000)
            new_subs.append(sub)

        new_subs = enforce_strict_spacing(new_subs)
        return new_subs, rejected_count

def load_whisper_model(device, compute_type, language, model_size="large-v3"):
    # Safe console capture
    real_stdout_fd = os.dup(1)
    safe_console = Console(file=os.fdopen(real_stdout_fd, "w"))

    model = None
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=safe_console,
            transient=True 
        ) as progress:
            lang_str = language.upper() if language else "AUTO"
            task_id = progress.add_task(f"[cyan]Loading model ({lang_str})...", total=None)
            
            with CaptureProgress(progress, task_id):
                model = whisperx.load_model(model_size, device, compute_type=compute_type, language=language)
            
            safe_console.print("[dim]🤖 Model loaded.[/dim]")
    finally:
        try: os.close(real_stdout_fd)
        except: pass

    return model

def run_anchor_sync(video_path, sub_path, device, compute_type, batch_size, model, language=None):
    safe_console = Console(force_terminal=True)
    try:
        audio = whisperx.load_audio(str(video_path))
    except Exception as e:
        safe_console.print(f"[bold red]❌ Failed to load audio: {e}[/bold red]")
        return None, 0, 0

    result = None 
    current_batch_size = batch_size
    is_windows = (os.name != 'posix')
    
    while current_batch_size >= 1:
        try:
            sys.stdout.flush()
            sys.stderr.flush()

            ui_console = make_ui_console()

            # --- DYNAMIC UI ---
            columns = [SpinnerColumn(), TextColumn("[progress.description]{task.description}")]
            if not is_windows:
                columns.append(BarColumn())
                columns.append(TextColumn("[progress.percentage]{task.percentage:>3.0f}%"))
            columns.append(TimeElapsedColumn()) 

            with Progress(
                *columns,
                console=ui_console,
                transient=True,
                refresh_per_second=10 
            ) as progress:
                
                ui_console.print(f"[dim]🎤 Transcribing audio (Batch Size: {current_batch_size}, Compute: {compute_type})...[/dim]")
                task_id = progress.add_task(f"[cyan]Transcribing...", total=100 if not is_windows else None)
                progress.refresh()
                
                with CaptureProgress(progress, task_id, ui_console=ui_console):
                    result = model.transcribe(
                        audio, 
                        batch_size=current_batch_size, 
                        language=language,
                        print_progress=not is_windows,  # Linux/Mac=True, Windows=False
                        combined_progress=False
                    )
            break

        except Exception as e:
            error_msg = str(e).lower()
            is_oom = any(x in error_msg for x in ["cuda", "out of memory", "alloc", "cudnn"])
            
            if current_batch_size == 1 or not is_oom:
                safe_console.print(f"[bold red]❌ Fatal Error: {e}[/bold red]")
                return None, 0, 0

            os.write(1, f"\033[93m⚠️ Batch size {current_batch_size} failed. Retrying with {current_batch_size // 2}...\033[0m\n".encode())
            if "cuda" in str(device):
                try:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                except: pass
            
            gc.collect()
            current_batch_size //= 2
            time.sleep(1) 
            
    if not result:
        return None, 0, 0

    del audio
    gc.collect()
    if device == "cuda": torch.cuda.empty_cache()
    
    console.print(f"[dim]📝 Transcription complete.[/dim]")

    # Align Phonemes
    # Use a spinner context to show activity during the CPU-heavy task
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True
    ) as progress:
        progress.add_task("[cyan] Aligning phonemes...", total=None)
        
        try:
            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
            audio_for_align = whisperx.load_audio(str(video_path))
            
            # Pass print_progress=False to disable the ugly tqdm bar and rely on clean Rich spinner instead.
            aligned_result = whisperx.align(
                result["segments"], 
                model_a, 
                metadata, 
                audio_for_align, 
                device, 
                return_char_alignments=False,
            )
            segments = aligned_result["segments"]
            
            # Cleanup
            del model_a; del audio_for_align; gc.collect(); 
            if device == "cuda": torch.cuda.empty_cache()
            
        except Exception as e:
            # If it fails, we catch it inside the spinner, print error, and continue
            console.print(f"[yellow]⚠️ Phoneme alignment failed ({e}). Using raw timestamps.[/yellow]")
            segments = result["segments"]

    console.print("[dim]📏 Phoneme alignment complete.[/dim]")

    # Prepare Data
    whisper_data = []
    for seg in segments:
        whisper_data.append({
            'start': seg['start'], 'end': seg['end'],
            'text': seg['text'], 'words': seg.get('words', [])
        })

    # Global Aligner
    console.print("[dim]🧮 Calculating sync offsets...[/dim]")
    original_subs = open_subtitle(sub_path)
    aligner = GlobalAligner(original_subs, whisper_data)
    synced_subs, rejected = aligner.run()
    
    if synced_subs is None:
        raise Exception("Zero matches found.")

    output_path = sub_path.with_name(f"{sub_path.stem}.synced{sub_path.suffix}")
    synced_subs.save(str(output_path))
    
    return output_path, len(original_subs), rejected
    