import numpy as np
import pytest

from monocular_depth import webcam_demo
from monocular_depth.config import DEFAULT_CAMERA_HEIGHT, DEFAULT_CAMERA_WIDTH


class FakeCamera:
    def __init__(self, size=(3840, 2160), opened=True, readable=True):
        self.size = size
        self.opened = opened
        self.readable = readable
        self.released = False
        self.settings = []

    def isOpened(self):
        return self.opened

    def set(self, key, value):
        self.settings.append((key, value))
        return True

    def read(self):
        return self.readable, np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)

    def release(self):
        self.released = True


def test_verified_maximum_camera_defaults(monkeypatch):
    assert (DEFAULT_CAMERA_WIDTH, DEFAULT_CAMERA_HEIGHT) == (3840, 2160)
    camera = FakeCamera()
    monkeypatch.setattr(webcam_demo.cv2, "VideoCapture", lambda *_: camera)
    assert webcam_demo.open_camera(0, 3840, 2160) is camera
    settings = dict(camera.settings)
    assert settings[webcam_demo.cv2.CAP_PROP_FOURCC] == webcam_demo.cv2.VideoWriter_fourcc(*"MJPG")
    assert settings[webcam_demo.cv2.CAP_PROP_FPS] == 30
    assert not camera.released


@pytest.mark.parametrize("case", ["fallback", "closed", "no_frame"])
def test_camera_failure_releases_and_refuses_silent_fallback(monkeypatch, case):
    camera = FakeCamera(
        size=(1280, 720) if case == "fallback" else (3840, 2160),
        opened=case != "closed",
        readable=case != "no_frame",
    )
    monkeypatch.setattr(webcam_demo.cv2, "VideoCapture", lambda *_: camera)
    with pytest.raises(RuntimeError):
        webcam_demo.open_camera(0, 3840, 2160)
    assert camera.released
