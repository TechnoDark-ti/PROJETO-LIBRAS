import pytest
from core.camera import Camera

def test_camera_initial_state():
    cam = Camera()

    assert cam.cap is None
    assert cam.is_running is False


def test_camera_invalid_index():
    cam = Camera(camera_index=99)

    with pytest.raises(RuntimeError):
        cam.start()


    assert cam.is_running is False
    assert cam.cap is None


def test_camera_start_stop_without_crash(monkeypatch):
    class FakeCapture:
        def isOpened(self):
            return True

        def read(self):
            return True, "fake_frame"
        
        def release(self):
            pass
    
    def fake_videocapture(index):
        return FakeCapture()

    monkeypatch.setattr("cv2.VideoCapture", fake_videocapture)
    
    cam = Camera()
    cam.start()

    assert cam.is_running is True

    frame = cam.read()
    assert frame == "fake_frame"


    cam.stop()
    assert cam.is_running is False


    def test_camera_read_without_start():
        cam = Camera()
        frame = cam.read()
        
        assert frame is None