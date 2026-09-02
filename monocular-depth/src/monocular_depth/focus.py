"""Driver-verified focus lock, not a guarantee of optical sharpness."""

from __future__ import annotations

import math
from time import perf_counter

import cv2

FOCUS_TOLERANCE = 0.5  # Driver units, not millimeters.


def read_focus(camera) -> dict:
    def finite_value(property_id):
        value = float(camera.get(property_id))
        return value if math.isfinite(value) else None

    raw = finite_value(cv2.CAP_PROP_AUTOFOCUS)
    backend = finite_value(cv2.CAP_PROP_BACKEND)
    # OpenCV DirectShow returns CameraControlFlags, not a Boolean:
    # Auto=1, Manual=2. A zero flag is not evidence that manual mode worked.
    if backend == cv2.CAP_DSHOW:
        autofocus = 0.0 if raw == 2 else 1.0 if raw in (1, 3) else None
    else:
        autofocus = raw if raw in (0, 1) else None
    return {
        "autofocus": autofocus,
        "autofocus_raw": raw,
        "backend_id": backend,
        "focus": finite_value(cv2.CAP_PROP_FOCUS),
    }


def matches_lock(reading: dict, target: float) -> bool:
    af, focus = reading.get("autofocus"), reading.get("focus")
    return (
        af is not None
        and focus is not None
        and math.isfinite(af)
        and math.isfinite(focus)
        and abs(af) < 0.01
        and abs(focus - target) <= FOCUS_TOLERANCE
    )


class FocusLock:
    def __init__(self, camera, *, clock=perf_counter, samples=8, settle_seconds=0.75):
        self.camera, self.clock = camera, clock
        self.samples, self.settle_seconds = samples, settle_seconds
        self.state = "UNLOCKED"
        self.target = None
        self.stable_samples = 0
        self.started = None
        self.message = "A autofocus | F lock when board is sharp"
        self.reading = read_focus(camera)

    def fail(self, message):
        self.state, self.message = "FAILED", message
        self.stable_samples = 0
        return False

    def enable_auto(self):
        self.target, self.started, self.stable_samples = None, None, 0
        success = self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        self.reading = read_focus(self.camera)
        if (
            not success
            or self.reading["autofocus"] is None
            or abs(self.reading["autofocus"] - 1) > 0.01
        ):
            return self.fail("Autofocus request failed; check driver")
        self.state = "AUTO"
        self.message = "Place board, wait until sharp, then press F"
        return True

    def request_lock(self, target=None):
        self.reading = read_focus(self.camera)
        target = self.reading["focus"] if target is None else float(target)
        if target is None or not math.isfinite(target) or target < 0:
            return self.fail("Focus position unavailable; cannot lock")
        if not self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 0):
            return self.fail("Driver rejected disabling autofocus")
        if not self.camera.set(cv2.CAP_PROP_FOCUS, target):
            return self.fail("Driver rejected manual focus position")
        self.reading = read_focus(self.camera)
        if not matches_lock(self.reading, target):
            return self.fail("Focus readback did not match request")
        self.target = target
        self.started = self.clock()
        self.stable_samples = 0
        self.state, self.message = "VERIFYING", "Waiting for stable driver readback"
        return True

    def observe(self, *, new_frame=True):
        self.reading = read_focus(self.camera)
        if self.state in ("VERIFYING", "LOCKED"):
            if not matches_lock(self.reading, self.target):
                self.fail("Focus drift detected; saving blocked")
            elif new_frame:
                self.stable_samples += 1
                if (
                    self.stable_samples >= self.samples
                    and self.clock() - self.started >= self.settle_seconds
                ):
                    self.state, self.message = (
                        "LOCKED",
                        "Driver lock verified; check image is sharp",
                    )
        return self.snapshot()

    def snapshot(self):
        return {
            "state": self.state,
            "target": self.target,
            **self.reading,
            "stable_samples": self.stable_samples,
            "required_samples": self.samples,
            "settle_seconds": self.settle_seconds,
            "note": "Driver readback only; visually confirm sharpness",
        }


def validate_frame_focus(before: dict, after: dict) -> float:
    target = before.get("target")
    if target is None or not math.isfinite(target) or target < 0:
        raise ValueError("No verified focus target")
    for state in (before, after):
        if (
            state.get("state") != "LOCKED"
            or state.get("target") != target
            or not matches_lock(state, target)
        ):
            raise ValueError("Frame was not captured under the verified focus lock")
    return float(target)
