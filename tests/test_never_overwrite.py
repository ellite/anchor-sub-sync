"""Anchor never overwrites a file it did not just create unless -o is set (utils/files.py). See AGENTS.md, 'Never overwrite'.

The last test is a tripwire: a new file writer that does not use the helper fails it.
"""
import argparse
import ast
from pathlib import Path

import pytest

from anchor.utils import files
from anchor.utils.files import backup_if_needed, find_best_video_match, resolve_output, unique_path

ROOT = Path(__file__).resolve().parent.parent


def test_a_free_name_is_used_as_is():
    assert unique_path(Path("/nonexistent/folder/Movie.en.synced.srt")) == Path("/nonexistent/folder/Movie.en.synced.srt")


def test_a_taken_name_gets_a_counter_before_the_extension(tmp_path):
    target = tmp_path / "Movie.en.synced.srt"
    target.write_text("first")
    assert unique_path(target).name == "Movie.en.synced.1.srt"
    (tmp_path / "Movie.en.synced.1.srt").write_text("second")
    assert unique_path(target).name == "Movie.en.synced.2.srt"


def test_a_dangling_symlink_counts_as_taken(tmp_path):
    link = tmp_path / "Movie.srt"
    link.symlink_to(tmp_path / "missing")
    assert unique_path(link).name == "Movie.1.srt"


def test_only_dash_o_allows_replacing(tmp_path):
    target = tmp_path / "Movie.srt"
    target.write_text("x")
    assert resolve_output(target, argparse.Namespace(overwrite=True)) == target
    assert resolve_output(target, argparse.Namespace(overwrite=False)).name == "Movie.1.srt"
    assert resolve_output(target).name == "Movie.1.srt"            # no args: the safe answer


def test_a_backup_never_replaces_an_older_backup(tmp_path):
    """-B with -o: each run keeps its own .bak (Movie.srt.bak, Movie.srt.1.bak ...)."""
    sub = tmp_path / "Movie.srt"
    args = argparse.Namespace(backup=True, overwrite=True)
    for text in ("one", "two", "three"):
        sub.write_text(text)
        backup_if_needed(sub, args)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["Movie.srt.1.bak", "Movie.srt.2.bak", "Movie.srt.bak"]
    assert (tmp_path / "Movie.srt.bak").read_text() == "one"


@pytest.mark.parametrize("subtitle", [
    "Movie.en.synced.1.srt", "Movie.en.1.srt", "Movie.pt.hi.2.srt", "Movie.25fps.2.srt", "Movie.23.976fps.1.srt",
])
def test_a_numbered_output_still_finds_its_video(tmp_path, monkeypatch, subtitle):
    """Without this a sync of Movie.en.synced.1.srt could not find Movie.mkv."""
    (tmp_path / "Movie.mkv").write_text("v")
    monkeypatch.chdir(tmp_path)
    assert find_best_video_match(tmp_path / subtitle).name == "Movie.mkv"


def test_a_number_that_belongs_to_the_title_is_not_stripped(tmp_path, monkeypatch):
    """Blade.Runner.2049 and Die.Hard.2 end in digits that are part of the name."""
    for video in ("Blade.Runner.2049.mkv", "Die.Hard.2.mkv", "Die.Hard.mkv"):
        (tmp_path / video).write_text("v")
    monkeypatch.chdir(tmp_path)
    assert find_best_video_match(tmp_path / "Blade.Runner.2049.en.srt").name == "Blade.Runner.2049.mkv"
    assert find_best_video_match(tmp_path / "Die.Hard.2.en.srt").name == "Die.Hard.2.mkv"


# Modules that write files the user would notice, and the helper each must use. Files that write only caches, config or
# downloads into a name the caller already made free are listed with the reason.
WRITERS_ALLOWED_WITHOUT_HELPER = {
    "anchor/utils/config.py": "writes ~/.anchor/config.json, not a user file",
    "anchor/utils/ui.py": "terminal plumbing",
    "anchor/utils/parakeet.py": "writes its install marker inside its own environment",
    "anchor/utils/parakeet_worker.py": "writes its result into a temp file it was given",
    "anchor/core/download/providers/podnapisi.py": "download.py picks a free name (counter) before calling a provider",
    "anchor/core/download/providers/subdl.py": "same",
    "anchor/core/download/providers/addic7ed.py": "same",
    "anchor/core/download/providers/opensubtitles.py": "same",
    "anchor/core/convert/image_subtitles.py": "its subtitle output uses resolve_output; the PNG frames go to a temp dir",
}
HELPERS = ("unique_path", "resolve_output")


def _writes_files(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name in ("save", "write_text", "write_bytes", "write_srt"):
                yield node
            if name == "open" and any(isinstance(a, ast.Constant) and isinstance(a.value, str) and ("w" in a.value or "a" in a.value)
                                      for a in list(node.args[1:2]) + [k.value for k in node.keywords if k.arg == "mode"]):
                yield node


def test_every_module_that_writes_a_user_file_goes_through_the_no_overwrite_helper():
    """Tripwire. A new `.save(...)` / `open(..., 'w')` in a module that never mentions unique_path or resolve_output is how
    an overwrite sneaks back in. Route the new output through the helper (and add a run-it-twice check), or, if it is not a
    user-visible file, add the module to WRITERS_ALLOWED_WITHOUT_HELPER with the reason."""
    offenders = []
    for path in sorted((ROOT / "anchor").rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        source = path.read_text()
        tree = ast.parse(source)
        if not any(True for _ in _writes_files(tree)):
            continue
        if rel in WRITERS_ALLOWED_WITHOUT_HELPER or any(h in source for h in HELPERS):
            continue
        offenders.append(rel)
    assert not offenders, f"these modules write files without unique_path / resolve_output: {offenders}"
