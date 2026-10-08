"""Proves the tests catch breakage: breaks one piece of logic at a time in a throwaway copy of the repo and runs the suite.

    python tests/mutation_check.py            # every mutation must be CAUGHT (the suite fails)

A SURVIVED line means no test protects that choice: tighten a test (compare against the intended value, not the constant it
tests) or add one. When you add a key piece of logic, add a mutation here too. Takes about a minute and a half.
"""
import shutil, subprocess, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; PY = sys.argv[1] if len(sys.argv) > 1 else sys.executable
A = 'anchor/utils/alignment.py'; V = 'anchor/utils/syncverdict.py'; F = 'anchor/utils/files.py'; P = 'anchor/utils/pairing.py'
M = [
 ("strong anchors skip the neighbour filter", A, "AUDIO_STRONG_ANCHORS = True", "AUDIO_STRONG_ANCHORS = False"),
 ("weak tolerance 1.0 -> 1.5", A, "AUDIO_WEAK_DRIFT_TOL_SEC = 1.0", "AUDIO_WEAK_DRIFT_TOL_SEC = 1.5"),
 ("fuzzy snap cutoff 0.85 -> 0.8", A, "SNAP_MIN_SIMILARITY = 0.85", "SNAP_MIN_SIMILARITY = 0.8"),
 ("fuzzy snap switched off", A, "            self._snap_to_script(sub_tokens, wh_tokens)", "            pass"),
 ("snap also touches short words", A, "SNAP_MIN_LETTERS = 5", "SNAP_MIN_LETTERS = 3"),
 ("leading-word back-off pace", A, "LEADING_WORD_SEC = 0.3", "LEADING_WORD_SEC = 0.0"),
 ("leading-word max 3 -> 9", A, "LEADING_WORD_MAX = 3", "LEADING_WORD_MAX = 9"),
 ("no clamp to the strong drift curve", A, "if curve_start < c['raw_match_time']:", "if False:"),
 ("parakeet strong anchors smoothed like whisper", A, "        if a.get('strong'):\n            a['final_start'] = a['raw_match_time']", "        if False:\n            a['final_start'] = a['raw_match_time']"),
 ("no per-block trend", A, "BLOCK_TREND_MIN_ANCHORS = 5", "BLOCK_TREND_MIN_ANCHORS = 500"),
 ("no edge trend", A, "EDGE_FIT_ANCHORS = 10", "EDGE_FIT_ANCHORS = 0"),
 ("edge extra uncapped", A, "EDGE_MAX_EXTRA_SEC = 6.0", "EDGE_MAX_EXTRA_SEC = 600.0"),
 ("scene gap 5 -> 500 s", A, "SCENE_GAP_SEC = 5.0", "SCENE_GAP_SEC = 500.0"),
 ("zipper gap 50 -> 0", A, "GAP_MS = 50 ", "GAP_MS = 0 "),
 ("zipper minimum duration off", A, "MIN_DURATION_MS = 600 ", "MIN_DURATION_MS = 0 "),
 ("reference mode uses audio spacing rules", A, "new_subs = enforce_strict_spacing(new_subs, gap_ms=0, min_duration_ms=0)", "new_subs = enforce_strict_spacing(new_subs)"),
 ("reference start truncated not rounded", A, "to_ms = (lambda t: int(round(max(0, t) * 1000))) if self.reference", "to_ms = (lambda t: int(max(0, t) * 1000)) if self.reference"),
 ("reference strong needs 99 words", A, "STRONG_MATCH_WORDS = 4 ", "STRONG_MATCH_WORDS = 99 "),
 ("reference weak tolerance 5 -> 1", A, "WEAK_DRIFT_TOL_SEC = 5.0", "WEAK_DRIFT_TOL_SEC = 1.0"),
 ("stray leading word not skipped", A, "if end == i and end + 1 < len(estimates) and cluster_end(end + 1) > end + 1:", "if False:"),
 ("reference end never taken from reference", A, "        if cue_words - 1 - pos == tok['n'] - 1 - tok['pos']:\n            return tok['ev_end']", "        if False:\n            return tok['ev_end']"),
 ("timing filter slow rule off", A, "SLOW_SEGMENT_MAX_WPS = 0.6", "SLOW_SEGMENT_MAX_WPS = 0.0"),
 ("overlaps and tight gaps counted together", A, "            if prev.end > curr.start:\n                overlaps += 1", "            if True:\n                overlaps += 1"),
 ("fps ambiguity ignored", V, "FPS_AMBIGUOUS = 0.0015", "FPS_AMBIGUOUS = 0.0"),
 ("video frame rate not used", V, "if a == b or (video_fps and abs(b - video_fps) > 0.02):", "if a == b:"),
 ("step jump 3 s -> 1 s", V, "STEP_JUMP_SEC = 3.0", "STEP_JUMP_SEC = 0.5"),
 ("in-sync limit 0.6 -> 1.0", V, "SYNC_MEDIAN_SEC = 0.6", "SYNC_MEDIAN_SEC = 1.0"),
 ("coverage limit off", V, "MIN_COVERAGE = 0.25", "MIN_COVERAGE = 0.0"),
 ("drift limit huge", V, "DRIFT_TOTAL_SEC = 1.0", "DRIFT_TOTAL_SEC = 1000.0"),
 ("steps measured on raw drift", V, "    left = y - fitted", "    left = y.copy()"),
 ("-o not needed to overwrite", F, "return Path(path) if args is not None and getattr(args, \"overwrite\", False) else unique_path(path)", "return Path(path)"),
 ("backup replaces older backup", F, "path.rename(unique_path(path.with_suffix(path.suffix + \".bak\")))", "path.rename(path.with_suffix(path.suffix + \".bak\"))"),
 ("counter not ignored in video match", F, "        return before\n    return name", "        return name\n    return name"),
 ("bare number stripped from titles", F, "    if previous in {\"synced\", \"sync\", \"hi\", \"ai\", \"forced\"} or previous in codes or", "    if True or"),
 ("unique_path overwrites", F, "    if not os.path.lexists(path):\n        return path\n    counter = 1", "    return path\n    counter = 1"),
 ("frame rate factor inverted", 'anchor/core/framerate/framerate.py', "factor = from_fps / to_fps", "factor = to_fps / from_fps"),
 (".sub not refused", 'anchor/core/framerate/framerate.py', 'if path.suffix.lower() == ".sub":', "if False:"),
 ("frame rate output may replace", 'anchor/core/framerate/framerate.py', 'output = unique_path(path.with_name(f"{path.stem}.{to_label}fps{path.suffix}"))', 'output = path.with_name(f"{path.stem}.{to_label}fps{path.suffix}")'),
 ("unpicking does not renumber", P, "            self.picks[pane].remove(index)", "            pass"),
 ("select-all ignores the filter", P, "        shown = self.visible(pane)\n        if shown and all", "        shown = list(range(len(self.names[pane])))\n        if shown and all"),
 ("counts need not match", P, "        if len(left) != len(right):", "        if False:"),
 ("-s -r routed to point sync", 'anchor/cli.py', '        if args.reference:\n            return "reference"', '        if args.reference:\n            return "point"'),
 ("framerate task not light", 'anchor/cli.py', 'LIGHT_TASKS = {"download", "container", "burn", "clean_fix", "convert", "framerate"}', 'LIGHT_TASKS = {"download", "container", "burn", "clean_fix", "convert"}'),
 ("--fix allowed without --check", 'anchor/cli.py', "    if args.fix and not args.check:", "    if False:"),
 ("parakeet silence limit 4 -> 1 s", 'anchor/utils/parakeet.py', "MIN_SILENCE_SEC = 4.0", "MIN_SILENCE_SEC = 1.0"),
 ("repair zone reach 12 -> 1 s", 'anchor/utils/parakeet.py', "ZONE_REACH_SEC = 12.0", "ZONE_REACH_SEC = 1.0"),
 ("segment pause 0.8 -> 10 s", 'anchor/utils/parakeet.py', "SEGMENT_GAP_SEC = 0.8", "SEGMENT_GAP_SEC = 10.0"),
 ("stretched word limit 1.5 -> 100 s", 'anchor/core/transcribe/transcribe.py', "STRETCHED_WORD_SEC = 1.5", "STRETCHED_WORD_SEC = 100.0"),
]

A2='anchor/utils/alignment.py'; V2='anchor/utils/syncverdict.py'; PK='anchor/utils/parakeet.py'; TR='anchor/core/transcribe/transcribe.py'
M += [
 ("zipper gap 50 -> 30 ms", A2, "GAP_MS = 50 ", "GAP_MS = 30 "),
 ("zipper minimum 600 -> 500 ms", A2, "MIN_DURATION_MS = 600 ", "MIN_DURATION_MS = 500 "),
 ("edge extra cap 6 -> 5 s", A2, "EDGE_MAX_EXTRA_SEC = 6.0", "EDGE_MAX_EXTRA_SEC = 5.0"),
 ("edge slope cap 0.1 -> 0.2", A2, "EDGE_MAX_SLOPE = 0.1 ", "EDGE_MAX_SLOPE = 0.2 "),
 ("edge needs 2 anchors not 4", A2, "EDGE_MIN_ANCHORS = 4 ", "EDGE_MIN_ANCHORS = 2 "),
 ("edge pair span 20 -> 1 s", A2, "EDGE_MIN_SPAN_SEC = 20.0", "EDGE_MIN_SPAN_SEC = 1.0"),
 ("back-off pace 0.3 -> 0.25", A2, "LEADING_WORD_SEC = 0.3 ", "LEADING_WORD_SEC = 0.25 "),
 ("strong ratio 0.6 -> 0.9", A2, "STRONG_MATCH_RATIO = 0.6 ", "STRONG_MATCH_RATIO = 0.9 "),
 ("neighbour filter 1.5 -> 3 s", A2, "OUTLIER_THRESHOLD_SEC = 1.5", "OUTLIER_THRESHOLD_SEC = 3.0"),
 ("scene gap 5 -> 6 s", A2, "SCENE_GAP_SEC = 5.0", "SCENE_GAP_SEC = 6.0"),
 ("slow segment 5.5 -> 6.5 s", A2, "SLOW_SEGMENT_SEC = 5.5", "SLOW_SEGMENT_SEC = 6.5"),
 ("fast segment 8 -> 12 wps", A2, "FAST_SEGMENT_MAX_WPS = 8.0", "FAST_SEGMENT_MAX_WPS = 12.0"),
 ("block trend slope cap", A2, "BLOCK_TREND_MAX_SLOPE = 0.1 ", "BLOCK_TREND_MAX_SLOPE = 0.0001 "),
 ("block trend span 20 -> 1 s", A2, "BLOCK_TREND_MIN_SPAN_SEC = 20.0", "BLOCK_TREND_MIN_SPAN_SEC = 1.0"),
 ("verdict spread 1.0 -> 2.0", V2, "SYNC_SPREAD_SEC = 1.0", "SYNC_SPREAD_SEC = 2.0"),
 ("verdict residual 1.5 -> 3", V2, "STEP_RESIDUAL_SEC = 1.5", "STEP_RESIDUAL_SEC = 3.0"),
 ("verdict needs 3 anchors", V2, "MIN_ANCHORS = 8", "MIN_ANCHORS = 3"),
 ("fps tolerance 0.0004 -> 0", V2, "FPS_SLOPE_TOL = 0.0004", "FPS_SLOPE_TOL = 0.0"),
 ("parakeet silence 4 -> 3.5 s", PK, "MIN_SILENCE_SEC = 4.0", "MIN_SILENCE_SEC = 3.5"),
 ("parakeet pause 0.8 -> 0.6 s", PK, "SEGMENT_GAP_SEC = 0.8", "SEGMENT_GAP_SEC = 0.6"),
 ("parakeet zone merge 4 -> 1 s", PK, "ZONE_MERGE_SEC = 4.0", "ZONE_MERGE_SEC = 1.0"),
 ("parakeet cue reach 3 -> 0 s", PK, "CUE_REACH_SEC = 3.0", "CUE_REACH_SEC = 0.0"),
 ("stretched word 1.5 -> 1.2 s", TR, "STRETCHED_WORD_SEC = 1.5", "STRETCHED_WORD_SEC = 1.2"),
 ("stretched back-off 0.3 -> 0.2", TR, "STRETCHED_WORD_BACKOFF_SEC = 0.3", "STRETCHED_WORD_BACKOFF_SEC = 0.2"),
 ("summary: nothing counts as needing a fix", V2, 'NEEDS_FIX = ("offset", "drifting", "steps")', 'NEEDS_FIX = ()'),
 ("summary: good files listed first", V2, '"in_sync": 5, "failed": 6', '"in_sync": -1, "failed": 6'),
 ("summary: file names not escaped", V2, 'escape(entry["name"])', 'entry["name"]'),
 ("summary: commands not quoted", V2, "shlex.quote(entry['name'])", "entry['name']"),
 ("summary: fixed files still get a command", V2, 'and not e.get("action")]', ']'),
 ("--check without -s opens the menu", 'anchor/cli.py', '    if args.check:\n        return "check"', '    if args.check and args.subtitle:\n        return "check"'),
 ("--check on the command line asks questions", 'anchor/core/audiosync/audiosync.py', 'if args.subtitle or getattr(args, "report_only", False):', 'if args.subtitle:'),
 ("a batch from the menu asks per file", 'anchor/core/audiosync/audiosync.py', 'return "batch" if queue_len > 1 else "ask"', 'return "ask"'),
 ("--fix never acts", 'anchor/core/audiosync/audiosync.py', 'return "auto" if getattr(args, "fix", False) else "report"', 'return "report"'),
 ("new writer bypasses the helper", 'anchor/core/new_task.py', None, "def go(subs, path):\n    subs.save(str(path))\n"),
]

def run(mut=None):
    d = Path(tempfile.mkdtemp())
    for item in ("anchor", "tests", "pyproject.toml"):
        src = ROOT / item
        shutil.copytree(src, d / item, ignore=shutil.ignore_patterns("__pycache__")) if src.is_dir() else shutil.copy(src, d / item)
    if mut:
        name, f, old, new = mut
        text = (d / f).read_text() if (d / f).exists() else ''
        if old is None:
            (d / f).write_text(new)
        else:
            assert old in text, f"mutation text not found: {name}"
            (d / f).write_text(text.replace(old, new, 1))
    r = subprocess.run([PY, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"], cwd=d, capture_output=True, text=True)
    shutil.rmtree(d, ignore_errors=True)
    return r
def main():
    base = run(); print("unmutated:", base.stdout.strip().splitlines()[-1])
    survived = []
    for m in M:
        r = run(m); last = r.stdout.strip().splitlines()[-1]
        ok = r.returncode != 0
        first_fail = next((l for l in r.stdout.splitlines() if l.startswith("FAILED")), "")
        print(("caught   " if ok else "SURVIVED ") + m[0].ljust(48), (first_fail.split("::")[-1][:70] if ok else ""))
        if not ok: survived.append(m[0])
    print(f"\n{len(M)-len(survived)} of {len(M)} mutations caught; survivors: {survived}")
    return 1 if survived else 0


if __name__ == '__main__':
    sys.exit(main())
