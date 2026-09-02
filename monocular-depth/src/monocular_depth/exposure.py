"""Driver-verified gain / white-balance lock for repeatable capture.

Exposure itself is NOT controllable on this camera (CAP_PROP_EXPOSURE readback
is constant -5 and the image does not respond); auto exposure therefore stays
in charge of shutter time. Gain and white-balance temperature are both settable
and readable, so locking them removes two of the three auto-affected controls
and documents the state in every capture record.
"""

from __future__ import annotations

import math
import time
from time import perf_counter

import cv2

GAIN_TOLERANCE = 0.05
WB_TOLERANCE = 1.0

# OpenCV exposes white balance either as CAP_PROP_WB_TEMPERATURE (=45,
# Media Foundation) or through the legacy VideoProcAmp CAP_PROP_WHITE_BALANCE_BLUE_U
# (=17, DirectShow). This EMEET camera under DSHOW only answers 17, so the lock
# resolves the property the driver actually implements instead of hardcoding one.
WB_PROPERTY_CANDIDATES = tuple(
    dict.fromkeys(
        prop
        for prop in (
            getattr(cv2, "CAP_PROP_WB_TEMPERATURE", None),
            getattr(cv2, "CAP_PROP_WHITE_BALANCE_BLUE_U", None),
        )
        if prop is not None
    )
)
AUTO_WB_PROPERTY = getattr(cv2, "CAP_PROP_AUTO_WB", None)


def resolve_wb_property(camera) -> int | None:
    """First white-balance property the driver reports a finite positive value for."""
    for prop in WB_PROPERTY_CANDIDATES:
        try:
            value = float(camera.get(prop))
        except cv2.error:
            continue
        if math.isfinite(value) and value > 0:
            return prop
    return None


def read_controls(camera) -> dict:
    def finite_value(property_id):
        try:
            value = float(camera.get(property_id))
        except cv2.error:
            return None
        return value if math.isfinite(value) else None

    wb_property = resolve_wb_property(camera)
    reading = {
        "gain": finite_value(cv2.CAP_PROP_GAIN),
        "wb_temperature": finite_value(wb_property) if wb_property is not None else None,
        "wb_property": wb_property,
        "exposure": finite_value(cv2.CAP_PROP_EXPOSURE),
        "auto_exposure": finite_value(cv2.CAP_PROP_AUTO_EXPOSURE),
    }
    return reading


def matches_targets(reading: dict, targets: dict) -> bool:
    if not targets:
        return True
    for name, target in targets.items():
        value = reading.get(name)
        tolerance = WB_TOLERANCE if name == "wb_temperature" else GAIN_TOLERANCE
        if value is None or abs(value - target) > tolerance:
            return False
    return True


class ExposureLock:
    """Request, verify and monitor a fixed gain / white-balance setting."""

    def __init__(self, camera, *, gain=None, wb_temperature=None, clock=perf_counter,
                 samples=5, settle_seconds=0.5):
        self.camera, self.clock = camera, clock
        self.samples, self.settle_seconds = samples, settle_seconds
        self.targets = {}
        if gain is not None:
            self.targets["gain"] = float(gain)
        if wb_temperature is not None:
            self.targets["wb_temperature"] = float(wb_temperature)
        self.state = "UNLOCKED"
        self.stable_samples = 0
        self.started = None
        self.message = "not requested"
        self.wb_property = resolve_wb_property(camera)
        self.reading = read_controls(camera)

    def _fail(self, message):
        self.state, self.message = "FAILED", message
        self.stable_samples = 0
        return False

    def request_lock(self):
        self.reading = read_controls(self.camera)
        setters = {
            "gain": cv2.CAP_PROP_GAIN,
            "wb_temperature": self.wb_property,
        }
        if "wb_temperature" in self.targets and self.wb_property is None:
            return self._fail("Driver exposes no readable white-balance property")
        # Manual white balance only sticks with auto-WB disabled.
        if "wb_temperature" in self.targets and AUTO_WB_PROPERTY is not None:
            self.camera.set(AUTO_WB_PROPERTY, 0.0)
        # The UVC driver intermittently rejects a control write right after the
        # stream starts; retry briefly before reporting failure.
        for name, target in self.targets.items():
            accepted = False
            for _ in range(5):
                if self.camera.set(setters[name], target):
                    accepted = True
                    break
                self.reading = read_controls(self.camera)
                time.sleep(0.2)
            if not accepted:
                return self._fail(f"Driver rejected {name} setting after retries")
        self.reading = read_controls(self.camera)
        if not matches_targets(self.reading, self.targets):
            return self._fail(
                "Control readback did not match request: "
                + ", ".join(f"{k}={self.reading.get(k)}" for k in sorted(self.targets))
            )
        self.started = self.clock()
        self.stable_samples = 0
        self.state, self.message = "VERIFYING", "Waiting for stable driver readback"
        return True

    def observe(self):
        self.reading = read_controls(self.camera)
        if self.state in ("VERIFYING", "LOCKED"):
            if not matches_targets(self.reading, self.targets):
                self._fail("Control drift detected; saving blocked")
            else:
                self.stable_samples += 1
                if (
                    self.stable_samples >= self.samples
                    and self.clock() - self.started >= self.settle_seconds
                ):
                    self.state, self.message = "LOCKED", "Driver lock verified"
        return self.snapshot()

    def snapshot(self) -> dict:
        return {
            "state": self.state,
            "targets": self.targets,
            **self.reading,
            "stable_samples": self.stable_samples,
            "required_samples": self.samples,
            "settle_seconds": self.settle_seconds,
            "note": (
                "Driver readback only; exposure is driver-managed (auto exposure) "
                "on this camera"
            ),
        }

    def verify_current(self) -> None:
        self.observe()
        if self.state != "LOCKED":
            raise ValueError(f"Exposure control lock failed: {self.message}")
