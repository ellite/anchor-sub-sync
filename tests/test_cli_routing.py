"""Which task a command line runs (cli._resolve_task) and what stays light. See AGENTS.md, 'Entry point and dispatch'."""
import subprocess
import sys

import pytest

from anchor import cli
from anchor.utils.args import parse_arguments


def task(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["anchor", *argv])
    return cli._resolve_task(parse_arguments())


@pytest.mark.parametrize("argv, expected", [
    (["-s", "a.srt", "-v", "a.mkv"], "audio"),
    (["-s", "a.srt"], "audio"),
    (["-s", "a.srt", "-r", "b.srt"], "reference"),        # NOT point sync: that one is interactive only
    (["-s", "a.srt", "-l", "pt"], "translate"),
    (["-v", "a.mkv"], "transcribe"),
    (["-d"], "download"),
    (["--api"], "api"),
    ([], None),                                           # no arguments: the interactive menu
    (["-s", "a.srt", "--check"], "check"),
    (["-s", "a.srt", "-v", "a.mkv", "--check", "--fix"], "check"),
    (["-s", "a.srt", "--from-fps", "25", "--to-fps", "24"], "framerate"),
])
def test_command_line_to_task(monkeypatch, argv, expected):
    assert task(monkeypatch, *argv) == expected


def test_a_frame_rate_change_wins_over_a_sync_even_with_a_video(monkeypatch):
    """-s + -v + fps retimes; it must not start a sync."""
    assert task(monkeypatch, "-s", "a.srt", "-v", "a.mkv", "--from-fps", "25", "--to-fps", "24") == "framerate"


def test_fix_without_check_is_an_error(monkeypatch):
    """--fix writes files by itself; it must never run without the check that decides what to do."""
    with pytest.raises(SystemExit) as exit_info:
        task(monkeypatch, "-s", "a.srt", "--fix")
    assert exit_info.value.code == 1


def test_light_tasks_never_need_torch():
    """AGENTS.md: a top-level torch import in a light path regresses startup time (about 2 s). Tasks that do not need the
    engine skip engine setup entirely."""
    assert {"framerate", "download", "container", "burn", "clean_fix", "convert"} <= cli.LIGHT_TASKS
    assert not {"audio", "check", "reference", "translate", "transcribe", "point", "api"} & cli.LIGHT_TASKS


def test_importing_the_cli_and_the_sync_logic_does_not_load_torch():
    code = ("import sys; import anchor.cli, anchor.utils.alignment, anchor.utils.syncverdict, anchor.utils.pairing, "
            "anchor.core.framerate.framerate; assert 'torch' not in sys.modules and 'whisperx' not in sys.modules")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_the_menu_entries_map_to_the_tasks_they_are_named_after():
    """The menu was renumbered when Sync Check was added; each id must still reach its own task."""
    import inspect
    from anchor.utils import selections
    source = inspect.getsource(selections.select_run_mode)
    for number, name in (("1", "audio"), ("2", "reference"), ("3", "point"), ("4", "framerate"), ("5", "check"),
                         ("6", "translate"), ("7", "transcribe"), ("12", "download")):
        assert f'"{number}": "{name}"' in source
