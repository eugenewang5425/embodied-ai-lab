"""Run each desktop integration test in its own Python/Tcl process.

Slow-test layering (issue #16): every `isolated_tk` test automatically
carries `slow` (a real Tk window is minutes), and modules may mark further
tests `@pytest.mark.slow` (full collect-replay records, CLI subprocess
end-to-end).  Development runs `uv run pytest -q -m "not slow"` (target
< 90 s); the full suite stays the pre-push gate.
"""

import os
import subprocess
import sys

import pytest

# Subprocess timeout for an isolated Tk child; the slow-tier marker covers
# minutes-scale tests, and CI/push gates the full suite, so keep this generous.
SUBPROCESS_TIMEOUT_S = 600

# A single small training run (simulation collect-replay) costs minutes, so
# any test that requests one of these fixtures is minutes-scale by definition.
HEAVY_FIXTURE_NAMES = {"small_run", "light_stream"}


def pytest_configure(config):
    config.addinivalue_line("markers", "isolated_tk: real Tk integration in a fresh process")
    config.addinivalue_line(
        "markers",
        "slow: minutes-scale test (real windows, full records); "
        'excluded from the development run by `-m "not slow"`',
    )


def pytest_collection_modifyitems(config, items):
    # isolated_tk implies slow: one marker in the source, both semantics.
    # Heavy collect-replay fixtures imply slow for every consumer.
    for item in items:
        if item.get_closest_marker("isolated_tk") is not None or set(
            item.fixturenames
        ).intersection(HEAVY_FIXTURE_NAMES):
            item.add_marker(pytest.mark.slow)


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    if pyfuncitem.get_closest_marker("isolated_tk") is None:
        return None
    node = pyfuncitem.nodeid
    if os.environ.get("EMBODIED_TK_TEST_NODE") == node:
        return None  # Child executes the original test, including every assertion.
    env = {**os.environ, "EMBODIED_TK_TEST_NODE": node, "PYTHONUTF8": "1"}
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
