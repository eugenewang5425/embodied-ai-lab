import cv2
import numpy as np
import pytest

from monocular_depth.calibration import collect_frames, save_calibration_frame
from monocular_depth.focus import FocusLock, validate_frame_focus
from monocular_depth.records import read_json


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


class FakeCamera:
    def __init__(self, *, rejected=None, ignored=None):
        self.values = {cv2.CAP_PROP_FOCUS: 301.0, cv2.CAP_PROP_AUTOFOCUS: 1.0}
        self.rejected, self.ignored = rejected, ignored
        self.released = False
        self.set_calls = []

    def get(self, key):
        return self.values.get(key, 30.0)

    def set(self, key, value):
        self.set_calls.append((key, value))
        if key == self.rejected:
            return False
        if key != self.ignored:
            self.values[key] = value
        return True

    def release(self):
        self.released = True


def lock(camera=None):
    camera = camera or FakeCamera()
    clock = Clock()
    controller = FocusLock(camera, clock=clock)
    assert controller.enable_auto()
    assert controller.request_lock()
    for i in range(8):
        clock.now = (i + 1) * 0.1
        controller.observe()
    assert controller.state == "LOCKED"
    return controller, camera


def test_lock_requires_frames_and_elapsed_time():
    clock = Clock()
    controller = FocusLock(FakeCamera(), clock=clock)
    assert controller.request_lock()
    for _ in range(20):
        controller.observe()
    assert controller.state == "VERIFYING"
    clock.now = 0.8
    assert controller.observe()["state"] == "LOCKED"
    assert validate_frame_focus(controller.snapshot(), controller.snapshot()) == 301.0


def test_non_frame_checks_do_not_complete_lock():
    clock = Clock()
    controller = FocusLock(FakeCamera(), clock=clock)
    controller.request_lock()
    clock.now = 3
    for _ in range(20):
        controller.observe(new_frame=False)
    assert controller.state == "VERIFYING"
    assert controller.stable_samples == 0


@pytest.mark.parametrize("property_id", [cv2.CAP_PROP_AUTOFOCUS, cv2.CAP_PROP_FOCUS])
def test_driver_rejection_is_not_lock(property_id):
    controller = FocusLock(FakeCamera(rejected=property_id))
    assert not controller.request_lock()
    assert controller.state == "FAILED"


def test_success_return_but_ignored_setting_is_not_lock():
    controller = FocusLock(FakeCamera(ignored=cv2.CAP_PROP_AUTOFOCUS))
    assert not controller.request_lock()
    assert controller.state == "FAILED"


def test_directshow_manual_flag_two_is_normalized():
    class DShowCamera(FakeCamera):
        def get(self, key):
            if key == cv2.CAP_PROP_BACKEND:
                return cv2.CAP_DSHOW
            value = super().get(key)
            return 2.0 if key == cv2.CAP_PROP_AUTOFOCUS and value == 0 else value

    controller, _ = lock(DShowCamera())
    assert controller.snapshot()["autofocus"] == 0
    assert controller.snapshot()["autofocus_raw"] == 2


def test_directshow_unknown_zero_flag_is_not_a_lock():
    camera = FakeCamera()
    camera.values[cv2.CAP_PROP_BACKEND] = cv2.CAP_DSHOW
    controller = FocusLock(camera)
    assert not controller.request_lock()


@pytest.mark.parametrize(
    "property_id,value", [(cv2.CAP_PROP_FOCUS, 307), (cv2.CAP_PROP_AUTOFOCUS, 1)]
)
def test_drift_blocks_saving(property_id, value):
    controller, camera = lock()
    before = controller.snapshot()
    camera.values[property_id] = value
    after = controller.observe()
    assert after["state"] == "FAILED"
    with pytest.raises(ValueError):
        validate_frame_focus(before, after)


def test_invalid_readback_and_unlocked_frame_rejected(tmp_path):
    camera = FakeCamera()
    camera.values[cv2.CAP_PROP_FOCUS] = float("nan")
    controller = FocusLock(camera)
    assert not controller.request_lock()
    with pytest.raises(ValueError):
        save_calibration_frame(
            tmp_path,
            0,
            np.zeros((10, 10, 3), np.uint8),
            controller.snapshot(),
            controller.snapshot(),
            "test",
            24,
        )
    assert not list(tmp_path.iterdir())


def test_each_saved_frame_has_focus_evidence_and_no_overwrite(tmp_path):
    controller, _ = lock()
    state = controller.snapshot()
    frame = np.zeros((10, 20, 3), np.uint8)
    save_calibration_frame(tmp_path, 0, frame, state, state, "synthetic-test-time", 24)
    record = read_json(tmp_path / "view_000.json")
    assert record["image_size"] == [20, 10]
    assert record["focus_before_frame"]["autofocus"] == 0
    assert record["fixed_focus_driver_units"] == 301
    with pytest.raises(FileExistsError):
        save_calibration_frame(tmp_path, 0, frame, state, state, "test", 24)


@pytest.mark.parametrize("fixed_focus", [None, 293.0])
def test_capture_key_flow_does_not_save_unlocked_or_refocus_mid_session(
    tmp_path, monkeypatch, fixed_focus
):
    from monocular_depth import calibration, webcam_demo
    from monocular_depth.calibration import DEFAULT_BOARD
    from monocular_depth.records import write_json

    clock = Clock()
    camera = FakeCamera()

    def read_frame():
        clock.now += 0.2
        return True, np.zeros((60, 80, 3), np.uint8)

    camera.read = read_frame
    monkeypatch.setattr(webcam_demo, "open_camera", lambda *_: camera)
    monkeypatch.setattr(
        calibration,
        "FocusLock",
        lambda cam: FocusLock(cam, clock=clock, samples=2, settle_seconds=0.1),
    )
    monkeypatch.setattr(calibration, "detect_board", lambda *_: (None, None, None, np.arange(24)))
    for name in ("namedWindow", "resizeWindow", "imshow", "destroyAllWindows"):
        monkeypatch.setattr(cv2, name, lambda *_: None)
    monkeypatch.setattr(cv2.aruco, "drawDetectedCornersCharuco", lambda *_: None)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *_: 1)
    keys = iter(
        [
            ord("s"),
            ord("f"),
            ord("a") if fixed_focus else 255,
            255,
            ord("s"),
            ord("a"),
            ord("f"),
            ord("q"),
        ]
    )
    monkeypatch.setattr(cv2, "waitKey", lambda *_: next(keys))
    board_path = tmp_path / "board.json"
    write_json(board_path, DEFAULT_BOARD)
    output = tmp_path / "new_session"
    collect_frames(board_path, output, 0, 80, 60, fixed_focus)
    assert len(list(output.glob("*.png"))) == 1
    assert read_json(output / "session.json")["focus_lock_required"]
    assert read_json(output / "view_000.json")["focus_after_frame"]["state"] == "LOCKED"
    assert camera.values[cv2.CAP_PROP_AUTOFOCUS] == 0
    assert camera.released
    assert read_json(output / "focus-lock.json")["state"] == "LOCKED"
    if fixed_focus is not None:
        assert camera.values[cv2.CAP_PROP_FOCUS] == fixed_focus
        assert camera.set_calls == [(cv2.CAP_PROP_AUTOFOCUS, 0), (cv2.CAP_PROP_FOCUS, 293.0)]
        assert read_json(output / "session.json")["fixed_focus_requested"] == fixed_focus
        assert read_json(output / "view_000.json")["fixed_focus_driver_units"] == fixed_focus


def test_explicit_focus_is_restored_not_inherited():
    camera, clock = FakeCamera(), Clock()
    controller = FocusLock(camera, clock=clock)
    assert controller.request_lock(293)
    assert camera.values[cv2.CAP_PROP_FOCUS] == 293
    assert controller.state == "VERIFYING"
    for i in range(8):
        clock.now = (i + 1) * 0.1
        controller.observe()
    assert controller.state == "LOCKED"
    assert controller.target == 293


@pytest.mark.parametrize("target", [float("nan"), float("inf"), -1])
def test_invalid_explicit_focus_rejected_before_camera_open(target, tmp_path):
    camera = FakeCamera()
    controller = FocusLock(camera)
    assert not controller.request_lock(target)
    assert camera.set_calls == []
    with pytest.raises(ValueError, match="finite and nonnegative"):
        collect_frames(tmp_path / "missing.json", tmp_path / "unused", 0, 80, 60, target)
    assert not (tmp_path / "unused").exists()


def test_ignored_explicit_focus_releases_camera_and_refuses_capture(tmp_path, monkeypatch):
    from monocular_depth import webcam_demo
    from monocular_depth.calibration import DEFAULT_BOARD
    from monocular_depth.records import write_json

    camera = FakeCamera(ignored=cv2.CAP_PROP_FOCUS)
    monkeypatch.setattr(webcam_demo, "open_camera", lambda *_: camera)
    board_path = tmp_path / "board.json"
    write_json(board_path, DEFAULT_BOARD)
    with pytest.raises(ValueError, match="Cannot restore fixed focus"):
        collect_frames(board_path, tmp_path / "refused", 0, 80, 60, 293)
    assert camera.released
    assert not list((tmp_path / "refused").glob("*.png"))
