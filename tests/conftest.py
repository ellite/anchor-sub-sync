"""Shared fixtures. Nothing here needs a GPU, a model, the network or a media file."""
import pytest

from anchor.utils import alignment

REMINDER = (
    "These tests pin choices that make Anchor's syncs accurate (see AGENTS.md, 'Audio sync pipeline' and 'Reference sync').\n"
    "A failure means behaviour changed. Before editing the test, decide:\n"
    "  - Intentional? Update the test AND the matching AGENTS.md section, and re-run the real fixtures (A.* clips).\n"
    "  - Not intentional? You removed or weakened a piece of logic. Restore it.\n"
    "Never loosen an assertion just to get green."
)


@pytest.fixture(autouse=True)
def quiet_consoles():
    """The aligner prints progress lines; keep test output readable. Tests that read the output switch it back on."""
    previous = alignment.console.quiet
    alignment.console.quiet = True
    yield
    alignment.console.quiet = previous


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when == "call" and report.failed:
        report.sections.append(("Before you 'fix' this test", REMINDER))
