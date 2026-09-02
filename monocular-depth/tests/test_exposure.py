"""ExposureLock behavior with a scripted camera double."""

from __future__ import annotations

import cv2
import pytest

from monocular_depth.exposure import ExposureLock, matches_targets, read_controls

GAIN = cv2.CAP_PROP_GAIN
WB = getattr(cv2, "CAP_PROP_WB_TEMPERATURE", 110)


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


class MockCamera:
    def __init__(self, values=None, reject=()):
        self.values = {
            GAIN: 1.0,
            WB: 5000.0,
            cv2.CAP_PROP_EXPOSURE: -5.0,
            cv2.CAP_PROP_AUTO_EXPOSURE: -1.0,
        }
        if values:
            self.values.update(values)
        self.reject = set(reject)

    def get(self, prop):
        return float(self.values.get(prop, 0.0))

    def set(self, prop, value):
        if prop in self.reject:
            return False
        self.values[prop] = float(value)
        return True


def test_read_controls_and_locked_flow():
    cam = MockCamera()
    reading = read_controls(cam)
    assert reading["gain"] == 1.0
    assert reading["wb_temperature"] == 5000.0
    clock = FakeClock()
    lock = ExposureLock(cam, gain=0.0, wb_temperature=5000.0, clock=clock)
    assert lock.request_lock()
    assert lock.state == "VERIFYING"
    clock.advance(lock.settle_seconds + 0.1)
    for _ in range(lock.samples + 2):
        snap = lock.observe()
    assert lock.state == "LOCKED"
    assert snap["targets"] == {"gain": 0.0, "wb_temperature": 5000.0}
    lock.verify_current()


def test_readback_mismatch_fails_honestly():
    cam = MockCamera(reject=(WB,))
    lock = ExposureLock(cam, gain=0.0, wb_temperature=5000.0)
    assert not lock.request_lock()
    assert lock.state == "FAILED"
    assert "readback" in lock.message or "rejected" in lock.message


def test_drift_after_lock_blocks_save():
    cam = MockCamera(values={GAIN: 0.0, WB: 5000.0})
    clock = FakeClock()
    lock = ExposureLock(cam, gain=0.0, wb_temperature=5000.0, clock=clock)
    lock.request_lock()
    clock.advance(lock.settle_seconds + 0.1)
    for _ in range(lock.samples + 2):
        lock.observe()
    assert lock.state == "LOCKED"
    cam.values[WB] = 5500.0
    lock.observe()
    assert lock.state == "FAILED"
    assert "drift" in lock.message
    with pytest.raises(ValueError, match="lock failed"):
        lock.verify_current()


def test_tolerances_and_empty_targets():
    assert matches_targets({"gain": 5.02}, {"gain": 5.0})
    assert not matches_targets({"gain": 5.20}, {"gain": 5.0})
    assert matches_targets({"wb_temperature": 5000.5}, {"wb_temperature": 5000.0})
    assert matches_targets({}, {})
    cam = MockCamera()
    lock = ExposureLock(cam)
    assert lock.request_lock()
    assert lock.targets == {}
    lock.observe()
    assert lock.state == "VERIFYING"
