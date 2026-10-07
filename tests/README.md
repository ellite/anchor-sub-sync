# Tests

These tests pin the choices that make Anchor's syncs accurate. They are deliberately **strict**: a failure means behaviour
changed, and you should stop and decide whether that was intended.

```bash
pip install -e ".[dev]"      # or just: pip install pytest numpy pysubs2 rich langdetect py-cpuinfo requests
python -m pytest             # about 2 seconds, no GPU, models, network or media files
python tests/mutation_check.py   # proves the tests bite: breaks the logic one piece at a time, every break must be caught
```

Only the small packages are needed (no torch, no whisperx); the one test that needs whisperx skips itself without it.

## When a test fails

1. Read its docstring: it names the choice it protects and why it exists (the real file or bug that taught us).
2. **Intentional change?** Update the test *and* the matching section of `AGENTS.md`, then re-run the real fixtures (the `A.*`
   clips: `.correct` is in sync, `.oos` is out of sync) before trusting the new behaviour. Unit tests with synthetic data cannot
   tell you a change is *better*, only that it is *different*.
3. **Not intentional?** You removed or weakened a piece of logic. Restore it.
4. Never loosen an assertion just to get green.

## What is covered

| File | Protects |
|---|---|
| `test_alignment_audio.py` | text matching, strong/weak anchors and their tolerances, fuzzy respelling, leading-word back-off, the clamp to the drift curve, Whisper vs Parakeet smoothing, edge trend, the zipper, the Whisper timing filter |
| `test_alignment_reference.py` | exact reference timestamps (rounded, not truncated), end from the reference, strong matches skip the filter, the 5 s weak tolerance, stray-word skipping, no audio-mode spacing |
| `test_syncverdict.py` | every Sync Check verdict and its thresholds; a frame rate is suggested only when exactly one pair fits |
| `test_framerate.py` | the `from / to` factor, NTSC snapping, `.sub` refusal, output naming |
| `test_never_overwrite.py` | `unique_path`, `-o`, backups, numbered names still finding their video, and a tripwire for new file writers |
| `test_pairing.py` | the reference-sync picker: numbering, filters, select-all, validation |
| `test_cli_routing.py` | which task each command line runs, the menu mapping, light tasks never importing torch |
| `test_text_and_transcript_helpers.py` | `clean_text`, Parakeet token/segment/repair-zone logic, stretched-word trimming |

## Conventions

- Test data is synthetic and has a known answer (`helpers.py`): cues with distinct words, speech at a chosen drift.
- Pin the **intended value** (0.3 s, 50 ms, 600 ms, 6 s ...), not the constant that holds it, or changing the constant changes the test.
- Add a boundary test for every threshold (one side passes, the other fails).
- A new file writer must use `unique_path` / `resolve_output`; the tripwire in `test_never_overwrite.py` enforces it.
- `testscripts/` (provider API experiments with credentials) is never collected: `testpaths` is `tests`.
