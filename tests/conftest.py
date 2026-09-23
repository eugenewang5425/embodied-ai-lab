"""Run each desktop integration test in its own Python/Tcl process.

Slow-test layering (issue #16): every `isolated_tk` test automatically
carries `slow` (a real Tk window is minutes), and modules may mark further
tests `@pytest.mark.slow` (full collect-replay records, CLI subprocess
end-to-end). Development uses `uv run python scripts/test.py quick`;
changed lessons get selected full tests before push, while shared-module
changes and milestones get the whole suite. Wall time varies with
concurrent simulation jobs, so the wrapper reports actual elapsed time.
"""

import os
import re
import subprocess
import sys

import pytest

# Subprocess timeout for an isolated Tk child; the slow-tier marker covers
# minutes-scale tests, and CI/push gates the full suite, so keep this generous.
SUBPROCESS_TIMEOUT_S = 600

# A single small training run (simulation collect-replay) costs minutes, so
# any test that requests one of these fixtures is minutes-scale by definition.
HEAVY_FIXTURE_NAMES = {"small_run", "light_stream", "light_record"}


def pytest_configure(config):
    config.addinivalue_line("markers", "isolated_tk: real Tk integration in a fresh process")
    config.addinivalue_line(
        "markers",
        "slow: minutes-scale test (real windows, full records); "
        'excluded from the development run by `-m "not slow"`',
    )
    config.addinivalue_line("markers", "record: experiment collection and record replay")
    config.addinivalue_line("markers", "gui: isolated desktop-window integration")
    config.addinivalue_line("markers", "training: lessons 28-42 learning and training replay")


def pytest_collection_modifyitems(config, items):
    # isolated_tk implies slow: one marker in the source, both semantics.
    # Heavy collect-replay fixtures imply slow for every consumer.
    for item in items:
        is_gui = item.get_closest_marker("isolated_tk") is not None
        if is_gui or set(item.fixturenames).intersection(HEAVY_FIXTURE_NAMES):
            item.add_marker(pytest.mark.slow)
        if item.get_closest_marker("slow") is None:
            continue
        lesson = re.match(r"test_session(\d+)", item.path.stem)
        if is_gui:
            item.add_marker(pytest.mark.gui)
        elif lesson is not None and 28 <= int(lesson.group(1)) <= 42:
            item.add_marker(pytest.mark.training)
        else:
            item.add_marker(pytest.mark.record)


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    if pyfuncitem.get_closest_marker("isolated_tk") is None:
        return None
    node = pyfuncitem.nodeid
    if os.environ.get("EMBODIED_TK_TEST_NODE") == node:
        return None  # Child executes the original test, including every assertion.
    env = {**os.environ, "EMBODIED_TK_TEST_NODE": node, "PYTHONUTF8": "1"}
    if "light_record" in pyfuncitem.funcargs:
        # The parent has already constructed this module-scoped recording.
        # Reuse its immutable files in the Tk child instead of simulating it
        # again, while keeping the real window test in a separate Tcl process.
        env["EMBODIED_TK_SHARED_RECORD"] = str(pyfuncitem.funcargs["light_record"][0])
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", node],
        cwd=pyfuncitem.config.rootpath,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=SUBPROCESS_TIMEOUT_S,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return True
