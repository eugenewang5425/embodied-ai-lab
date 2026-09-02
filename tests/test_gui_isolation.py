"""Verify the Tk process boundary propagates failures instead of hiding them."""

from types import SimpleNamespace

import conftest
import pytest


def item(tmp_path):
    return SimpleNamespace(
        nodeid="tests/test_example.py::test_window",
        config=SimpleNamespace(rootpath=tmp_path),
        get_closest_marker=lambda _: True,
    )


@pytest.mark.parametrize("returncode", [0, 1])
def test_child_failure_is_parent_failure(monkeypatch, tmp_path, returncode):
    monkeypatch.delenv("EMBODIED_TK_TEST_NODE", raising=False)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=returncode, stdout="child assertion", stderr="details")

    monkeypatch.setattr(conftest.subprocess, "run", run)
    if returncode:
        with pytest.raises(AssertionError, match="child assertiondetails"):
            conftest.pytest_pyfunc_call(item(tmp_path))
    else:
        assert conftest.pytest_pyfunc_call(item(tmp_path)) is True
    assert len(calls) == 1  # No automatic retry.
    assert calls[0][1]["timeout"] == 120
    assert calls[0][1]["env"]["EMBODIED_TK_TEST_NODE"] == item(tmp_path).nodeid


def test_child_runs_original_test_without_recursion(monkeypatch, tmp_path):
    test = item(tmp_path)
    monkeypatch.setenv("EMBODIED_TK_TEST_NODE", test.nodeid)
    assert conftest.pytest_pyfunc_call(test) is None
