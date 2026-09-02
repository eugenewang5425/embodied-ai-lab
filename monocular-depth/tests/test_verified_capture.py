import cv2
import numpy as np
import pytest

from monocular_depth.capture import VerifiedCapture


class Camera:
    def __init__(self):
        self.now, self.released = 0.0, False
        self.values = {
            cv2.CAP_PROP_FOCUS: 300.0,
            cv2.CAP_PROP_AUTOFOCUS: 1.0,
            cv2.CAP_PROP_FOURCC: 1234.0,
            cv2.CAP_PROP_BACKEND: 700.0,
        }
        self.size = (64, 48)
        self.readable, self.ignore_focus = True, False

    def get(self, key):
        return self.values.get(key, 0.0)

    def set(self, key, value):
        if self.ignore_focus and key == cv2.CAP_PROP_FOCUS:
            return True
        self.values[key] = 2.0 if key == cv2.CAP_PROP_AUTOFOCUS and value == 0 else value
        return True

    def read(self):
        self.now += 0.1
        return self.readable, np.zeros((self.size[1], self.size[0], 3), np.uint8)

    def release(self):
        self.released = True


def calibration():
    return {
        "capture_session": {"camera_index": 0, "fourcc_reported": 1234},
        "fixed_focus_driver_units": 293.0,
        "focus_lock_verified_for_accepted_frames": True,
        "image_size": [64, 48],
    }


def test_restore_and_frame_metadata_then_detect_drift():
    camera = Camera()
    capture = VerifiedCapture(camera, calibration(), 0, clock=lambda: camera.now)
    with pytest.raises(ValueError):
        capture.read()
    state = capture.wait_for_lock()
    assert state["state"] == "LOCKED" and state["focus"] == 293
    assert state["stable_samples"] >= 8
    frame, metadata = capture.read()
    assert frame.shape == (48, 64, 3)
    assert metadata["camera_focus_verified"]
    assert metadata["fixed_focus_driver_units"] == 293
    capture.verify_current(metadata)
    camera.values[cv2.CAP_PROP_FOCUS] = 294
    with pytest.raises(ValueError):
        capture.verify_current(metadata)
    with pytest.raises(ValueError):
        capture.read()


@pytest.mark.parametrize("failure", ["camera", "focus_missing", "format", "ignored"])
def test_bad_camera_or_focus_provenance_refused(failure):
    camera, record = Camera(), calibration()
    if failure == "focus_missing":
        record.pop("fixed_focus_driver_units")
    elif failure == "format":
        record["capture_session"]["fourcc_reported"] = 999
    elif failure == "ignored":
        camera.ignore_focus = True
    with pytest.raises(ValueError):
        VerifiedCapture(camera, record, 1 if failure == "camera" else 0)


def test_timeout_resolution_and_uncalibrated_capture():
    camera = Camera()
    capture = VerifiedCapture(camera, calibration(), 0, clock=lambda: camera.now)
    with pytest.raises(RuntimeError, match="Timed out"):
        capture.wait_for_lock(timeout=0.1)
    camera.size = (32, 24)
    with pytest.raises(ValueError, match="resolution"):
        capture.wait_for_lock()
    capture = VerifiedCapture(camera, None, 0)
    assert capture.wait_for_lock() is None
    _, meta = capture.read()
    assert not meta["camera_focus_verified"]


@pytest.mark.parametrize("drift_during_inference", [False, True])
def test_webcam_save_last_verifies_before_release(tmp_path, monkeypatch, drift_during_inference):
    from types import SimpleNamespace

    from monocular_depth import webcam_demo

    camera = Camera()

    class Pipeline:
        def __init__(self, *_):
            self.calibration = calibration()

        def process(self, frame):
            if drift_during_inference:
                camera.values[cv2.CAP_PROP_FOCUS] = 300
            return frame, np.ones(frame.shape[:2], np.float32), {}

    saved = []

    def save(*args, **kwargs):
        assert not camera.released
        saved.append(kwargs["metadata"])
        return [tmp_path / "synthetic.json"]

    monkeypatch.setattr(webcam_demo, "load_model", lambda **_: (None, SimpleNamespace(type="cpu")))
    monkeypatch.setattr(webcam_demo, "DepthPipeline", Pipeline)
    monkeypatch.setattr(webcam_demo, "open_camera", lambda *_: camera)
    monkeypatch.setattr(
        webcam_demo,
        "VerifiedCapture",
        lambda *args: VerifiedCapture(*args, clock=lambda: camera.now),
    )
    monkeypatch.setattr(webcam_demo, "save_result", save)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "depth-webcam",
            "--headless",
            "--max-frames",
            "1",
            "--save-last",
            "--width",
            "64",
            "--height",
            "48",
            "--calibration",
            "synthetic.json",
        ],
    )
    if drift_during_inference:
        with pytest.raises(ValueError):
            webcam_demo.main()
        assert not saved
    else:
        webcam_demo.main()
        assert saved[0]["focus_before_save"]["state"] == "LOCKED"
        assert saved[0]["focus_after_inference"]["focus"] == 293
    assert camera.released
