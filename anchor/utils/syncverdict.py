"""Decides whether a subtitle is in sync with its audio from the drift of its anchored cues.

Input is what GlobalAligner already measures: for every cue whose words were found in the speech, the *drift* (where the
speech is minus where the subtitle puts the cue) at the cue's original start. Matching is by text, so a subtitle that is
30 s out still matches and its drift is simply 30. The verdict reads the shape of drift over time:

    in sync          small drift everywhere
    constant offset  one shift for the whole file
    drifting         drift grows (or shrinks) steadily: a frame rate mismatch, or a different release speed
    steps            drift changes in jumps: another cut, removed or added scenes, a partly synced file
    not comparable   too few cues matched: wrong language, another episode, or a corrupt subtitle

No audio, torch or file access here, so it can be tested on its own.
"""
import numpy as np

# Thresholds (seconds), calibrated on the test fixtures (see AGENTS.md, "Sync check").
SYNC_MEDIAN_SEC = 0.6      # |median drift| up to this counts as aligned (people time cues 0.2-1 s off the speech)
SYNC_SPREAD_SEC = 1.0      # 90th percentile of |drift - median| up to this counts as steady
DRIFT_TOTAL_SEC = 1.0      # the fitted line changes by more than this over the file: it drifts
STEP_RESIDUAL_SEC = 1.5    # 90th percentile of |drift - fitted line| above this: not a straight line, so steps
STEP_JUMP_SEC = 3.0        # a jump of at least this between neighbouring runs of cues (block medians wander by about 1 s)
FPS_AMBIGUOUS = 0.0015     # pairs whose drift differs by less than this cannot be told apart by measuring
FPS_SLOPE_TOL = 0.0004     # drift per second: matching noise alone gives about 0.0003 over a full episode
MIN_ANCHORS = 8
MIN_COVERAGE = 0.25        # fewer anchored cues than this share of the file: not comparable
SECTIONS = 8
TS_MAX_POINTS = 400        # Theil-Sen is quadratic: thin the points beyond this

# Frame rate pairs a straight drift is compared with (from, to): drift per second of video is from / to - 1 ... see _match_fps
_RATES = [("23.976", 24000 / 1001), ("24", 24.0), ("25", 25.0), ("29.97", 30000 / 1001), ("30", 30.0),
          ("50", 50.0), ("59.94", 60000 / 1001), ("60", 60.0)]


def _theil_sen(x, y):
    n = len(x)
    if n > TS_MAX_POINTS:
        keep = np.linspace(0, n - 1, TS_MAX_POINTS).astype(int)
        x, y = x[keep], y[keep]
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    mask = dx > 1.0                      # pairs at least a second apart: closer ones only add noise
    if not mask.any():
        return 0.0, float(np.median(y))
    slope = float(np.median(dy[mask] / dx[mask]))
    return slope, float(np.median(y - slope * x))


def _match_fps(slope, video_fps=None):
    """The (from, to) frame rate pair that explains this drift slope, but only when exactly one pair does.

    A subtitle made for `from` fps shown on `to` fps video drifts by (from / to - 1) seconds per second (positive: the
    subtitle runs early, the speech comes later). A pair fits within 10% of that slope, or within FPS_SLOPE_TOL (what matters
    for 23.976 against 24, whose whole drift is a few seconds). The video's own frame rate, when known, must equal `to`.
    Pairs whose drift differs by less than FPS_AMBIGUOUS (0.1%: 23.976 -> 25 against 24 -> 25) cannot be told apart by
    measuring (the slope is only good to about 0.2%), so with more than one such pair nothing is suggested: guessing wrong
    leaves seconds of drift on a full episode. Returns ((from, to) or None, ["a to b", ...] when it could not choose).
    """
    found = []
    for fa, a in _RATES:
        for fb, b in _RATES:
            if a == b or (video_fps and abs(b - video_fps) > 0.02):
                continue
            expected = a / b - 1.0
            error = abs(slope - expected)
            if error <= max(0.1 * abs(expected), FPS_SLOPE_TOL):
                found.append((error, expected, fa, fb))
    if not found:
        return None, []
    found.sort()
    close = [f for f in found if abs(f[1] - found[0][1]) < FPS_AMBIGUOUS]
    if len(close) == 1:
        return (close[0][2], close[0][3]), []
    return None, [f"{f[2]} to {f[3]}" for f in close]


def assess(points, cue_count, duration=None, video_fps=None):
    """points: [(original_start_sec, drift_sec, strong)]. Returns a dict describing the verdict.

    Keys: verdict, summary, advice, anchors, cues, coverage, median, head, tail, slope, total_drift, spread, residual,
    sections [(start, end, median drift, anchors)], steps [(at, from_drift, to_drift)], fps (from, to) or None.
    """
    n = len(points)
    result = dict(verdict="not_comparable", summary="", advice="", anchors=n, cues=cue_count,
                  coverage=(n / cue_count) if cue_count else 0.0, median=None, head=None, tail=None, slope=0.0,
                  total_drift=0.0, spread=None, residual=None, sections=[], steps=[], fps=None, fps_alternatives=[])
    if n < MIN_ANCHORS or result["coverage"] < MIN_COVERAGE:
        result["summary"] = (f"Only {n} of {cue_count} cues were found in the speech. This subtitle probably belongs to another "
                             "video or language, or is badly damaged.")
        result["advice"] = "Check that the subtitle is for this video (episode, language, cut)."
        return result

    pts = np.array(sorted(points), dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    span_start, span_end = float(x[0]), float(x[-1])
    if duration:
        span_end = max(span_end, float(duration))
    median = float(np.median(y))
    spread = float(np.percentile(abs(y - median), 90))
    slope, intercept = _theil_sen(x, y)
    fitted = slope * x + intercept
    residual = float(np.percentile(abs(y - fitted), 90))
    total = slope * (float(x[-1]) - float(x[0]))
    k = max(5, n // 10)
    head, tail = float(np.median(y[:k])), float(np.median(y[-k:]))

    # Where does the drift jump? A step is a change of more than STEP_JUMP_SEC between neighbouring runs of anchors, measured on what
    # is left after removing the fitted line (a steady drift also changes between neighbouring runs, but along the line).
    steps = []
    run = max(3, n // 40)
    left = y - fitted
    for i in range(run, n - run + 1, max(1, run // 2)):
        before, after = float(np.median(left[i - run:i])), float(np.median(left[i:i + run]))
        if abs(after - before) > STEP_JUMP_SEC and (not steps or x[i] - steps[-1][0] > 30.0):
            steps.append((float(x[i]), float(np.median(y[i - run:i])), float(np.median(y[i:i + run]))))

    edges = np.linspace(span_start, span_end, SECTIONS + 1)
    sections = []
    for a, b in zip(edges[:-1], edges[1:]):
        inside = y[(x >= a) & (x < b if b < edges[-1] else x <= b)]
        sections.append((float(a), float(b), float(np.median(inside)) if len(inside) else None, int(len(inside))))

    result.update(median=median, head=head, tail=tail, slope=slope, total_drift=total, spread=spread, residual=residual,
                  sections=sections, steps=steps)

    if steps or residual > STEP_RESIDUAL_SEC:
        result["verdict"] = "steps"
        result["summary"] = ("The offset changes in jumps, not steadily" +
                             (f" (around {', '.join(_clock(s[0]) for s in steps[:4])})" if steps else "") +
                             ". This is typical of a different cut of the video, removed or added scenes, or a file that was "
                             "only partly synced.")
        result["advice"] = "Audio Sync follows such changes cue by cue; Reference Sync works if you have a subtitle for this cut."
    elif abs(total) > DRIFT_TOTAL_SEC:
        result["verdict"] = "drifting"
        result["fps"], result["fps_alternatives"] = _match_fps(slope, video_fps)
        start_note = "starts in sync" if abs(head) <= SYNC_MEDIAN_SEC else f"starts {_signed(head)} off"
        result["summary"] = (f"The subtitle {start_note} and drifts to {_signed(tail)} by the end "
                             f"({_signed(total)} over the video, {slope * 1000:+.2f} ms per second).")
        if result["fps"]:
            result["advice"] = (f"This matches a frame rate mismatch: use Change Frame Rate from {result['fps'][0]} to "
                                f"{result['fps'][1]} fps. Audio Sync also corrects it.")
        elif result["fps_alternatives"]:
            result["advice"] = (f"The drift looks like a frame rate mismatch, but it fits {' and '.join(result['fps_alternatives'])} "
                                "equally well (they differ by about 0.1%), so Anchor cannot tell which one. Run Audio Sync.")
        else:
            result["advice"] = "Run Audio Sync: it follows the drift. No standard frame rate pair explains this slope."
    elif abs(median) > SYNC_MEDIAN_SEC:
        result["verdict"] = "offset"
        result["summary"] = f"The whole subtitle is {_signed(median)} off (steady from start to end)."
        result["advice"] = f"Shift every cue by {_signed(median)}, or run Audio Sync."
    elif spread > SYNC_SPREAD_SEC:
        result["verdict"] = "loose"
        result["summary"] = (f"On average in sync ({_signed(median)}), but cue timing is loose: 90% of cues are within "
                             f"{spread:.1f} s of that.")
        result["advice"] = "Probably fine to watch; Audio Sync would tighten individual cues."
    else:
        result["verdict"] = "in_sync"
        result["summary"] = f"In sync: average {_signed(median)}, 90% of cues within {spread:.1f} s of that."
        result["advice"] = ""
    return result


def _signed(seconds):
    return f"{seconds:+.1f} s" if abs(seconds) >= 0.05 else "0.0 s"


def _clock(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


LABELS = {
    "in_sync": ("✅", "IN SYNC", "green"),
    "loose": ("🟡", "MOSTLY IN SYNC", "yellow"),
    "offset": ("❌", "CONSTANT OFFSET", "red"),
    "drifting": ("❌", "DRIFTING", "red"),
    "steps": ("❌", "OUT OF SYNC (steps)", "red"),
    "not_comparable": ("❓", "NOT COMPARABLE", "magenta"),
}


def print_report(console, name, result):
    """Prints the verdict and a per-section table of the drift."""
    icon, label, colour = LABELS[result["verdict"]]
    console.print(f"\n{icon} [bold {colour}]{label}[/bold {colour}]  [dim]{name}[/dim]")
    console.print(f"   {result['summary']}")
    if result["advice"]:
        console.print(f"   💡 {result['advice']}")
    console.print(f"   [dim]Cues found in the speech: {result['anchors']} of {result['cues']} ({result['coverage'] * 100:.0f}%)[/dim]")
    if result["sections"] and result["verdict"] != "not_comparable":
        console.print("   [dim]Offset over the video (positive = the subtitle is early, the speech comes later):[/dim]")
        for a, b, drift, count in result["sections"]:
            shown = "no cues" if drift is None else _signed(drift)
            noun = "cue" if count == 1 else "cues"
            sparse = "  (too few to trust)" if 0 < count < 3 else ""
            bar = "" if drift is None else "█" * min(30, int(round(abs(drift) * 3)))
            console.print(f"   [dim]{_clock(a)} - {_clock(b)}[/dim]  {shown:>8}  [{colour if abs(drift or 0) > SYNC_MEDIAN_SEC else 'green'}]{bar}[/]  [dim]{count} {noun}{sparse}[/dim]")
