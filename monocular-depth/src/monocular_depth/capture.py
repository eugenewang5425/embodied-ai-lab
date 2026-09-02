"""Restore calibrated focus and attach frame-local driver evidence to live RGB."""

from __future__ import annotations

import math
from time import perf_counter

import cv2

from .focus import FocusLock, validate_frame_focus
from .records import timestamp


class VerifiedCapture:
    def __init__(self, camera, calibration, camera_id, *, clock=perf_counter):
        self.camera, self.calibration, self.camera_id = camera, calibration, camera_id
        self.clock, self.focus = clock, None
        if calibration is None:
            return
        session = calibration.get("capture_session") or {}
        if session.get("camera_index") != camera_id:
            raise ValueError("Camera index does not match calibration")
        target = calibration.get("fixed_focus_driver_units")
        if not calibration.get("focus_lock_verified_for_accepted_frames") or target is None:
            raise ValueError("Calibrated live capture requires recorded fixed-focus provenance")
        if not math.isfinite(target) or target < 0:
            raise ValueError("Invalid calibrated focus target")
        fourcc = session.get("fourcc_reported")
        if fourcc is not None and camera.get(cv2.CAP_PROP_FOURCC) != fourcc:
            raise ValueError("Camera pixel format does not match calibration")
        self.focus = FocusLock(camera, clock=clock)
        if not self.focus.request_lock(target):
            raise ValueError(f"Cannot restore calibrated focus: {self.focus.message}")

    def _check_frame(self, ok, frame):
        if not ok or frame is None:
            raise RuntimeError("Camera stopped returning frames")
        if self.calibration and [frame.shape[1], frame.shape[0]] != self.calibration["image_size"]:
            raise ValueError("Camera resolution differs from calibration")

    def wait_for_lock(self, timeout=5.0):
        if self.focus is None:
            return None
        started = self.clock()
        while self.focus.state != "LOCKED":
            if self.clock() - started > timeout:
                raise RuntimeError("Timed out waiting for calibrated focus to settle")
            ok, frame = self.camera.read()
            self._check_frame(ok, frame)
            self.focus.observe()
            if self.focus.state == "FAILED":
                raise ValueError(f"Focus verification failed: {self.focus.message}")
        return self.focus.snapshot()

    def read(self):
        before = self.focus.observe(new_frame=False) if self.focus else None
        ok, frame = self.camera.read()
        received_at = timestamp()
        after = self.focus.observe() if self.focus else None
        self._check_frame(ok, frame)
        metadata = {
            "camera_index": self.camera_id,
            "frame_received_at_utc": received_at,
            "timestamp_note": "Host receive time, not hardware exposure timestamp",
            "camera_focus_verified": self.focus is not None,
        }
        if self.focus:
            metadata.update(
                {
                    "fixed_focus_driver_units": validate_frame_focus(before, after),
                    "focus_before_frame": before,
                    "focus_after_frame": after,
                }
            )
        return frame, metadata

    def verify_current(self, metadata):
        if self.focus is None:
            return None
        latest = self.focus.observe(new_frame=False)
        validate_frame_focus(metadata["focus_after_frame"], latest)
        return latest
