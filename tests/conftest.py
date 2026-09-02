"""Run each desktop integration test in its own Python/Tcl process."""

import os
import subprocess
import sys

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "isolated_tk: real Tk integration in a fresh process")


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
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return True
