"""get_frame must offload the blocking OpenCV decode off the event loop.

``cv2.VideoCapture``/``cap.set``/``cap.read`` are synchronous and block
uvicorn's event loop for the whole seek+decode, starving concurrent
requests. The handler must factor that work into a sync helper and dispatch
it via ``run_in_executor`` so the loop stays free.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from fastapi.testclient import TestClient

import gomoveshift.web.app as web_app
from gomoveshift.config import Config
from gomoveshift.web.app import create_app


@pytest.fixture
def video_root(tmp_path):
    """A real, readable single-frame .mp4 inside an allowed input_folder."""
    import cv2

    root = tmp_path / "media"
    root.mkdir()
    video = root / "clip.mp4"
    writer = cv2.VideoWriter(
        str(video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30.0,
        (32, 32),
    )
    assert writer.isOpened(), "could not open VideoWriter for fixture"
    for _ in range(5):
        writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    writer.release()
    assert video.exists() and video.stat().st_size > 0
    return root, video


@pytest.fixture
def client(video_root):
    root, _ = video_root
    config = Config()
    config.paths.input_folder = str(root)
    return TestClient(create_app(config))


def test_read_frame_sync_helper_exists_and_decodes(video_root):
    """The blocking body lives in a plain sync helper returning bytes|None."""
    assert hasattr(web_app, "_read_frame_sync")
    _, video = video_root
    data = web_app._read_frame_sync(video, 0.0)
    assert isinstance(data, (bytes, bytearray))
    assert len(data) > 0
    # JPEG magic bytes — confirms the helper still imencodes to jpeg.
    assert bytes(data[:2]) == b"\xff\xd8"


def test_read_frame_sync_returns_none_when_unreadable(tmp_path):
    """A non-decodable path yields None (drives the handler's 404)."""
    bogus = tmp_path / "not-a-video.mp4"
    bogus.write_bytes(b"not a real video")
    assert web_app._read_frame_sync(bogus, 0.0) is None


def test_handler_dispatches_decode_through_run_in_executor(client, video_root, monkeypatch):
    """The async endpoint must hand the decode to run_in_executor, not run
    it inline on the event loop."""
    _, video = video_root
    seen = {}

    real_get_running_loop = asyncio.get_running_loop

    def patched_get_running_loop():
        loop = real_get_running_loop()
        real_run_in_executor = loop.run_in_executor

        def spy_run_in_executor(executor, func, *args):
            seen["func"] = func
            seen["args"] = args
            return real_run_in_executor(executor, func, *args)

        monkeypatch.setattr(loop, "run_in_executor", spy_run_in_executor, raising=False)
        return loop

    monkeypatch.setattr(web_app.asyncio, "get_running_loop", patched_get_running_loop)

    response = client.get(f"/api/frame/{video}?timestamp=0")
    assert response.status_code == 200, response.text
    assert seen.get("func") is web_app._read_frame_sync
    assert seen.get("args")[0] == video.resolve()
    assert seen.get("args")[1] == 0.0


def test_frame_endpoint_returns_jpeg_bytes(client, video_root):
    """Return shape unchanged: 200 image/jpeg with non-empty body."""
    _, video = video_root
    response = client.get(f"/api/frame/{video}?timestamp=0")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content[:2] == b"\xff\xd8"


def test_frame_endpoint_404_when_undecodable(client, video_root):
    """404 behavior unchanged when the frame can't be read."""
    root, _ = video_root
    bogus = root / "broken.mp4"
    bogus.write_bytes(b"not a real video")
    response = client.get(f"/api/frame/{bogus}?timestamp=0")
    assert response.status_code == 404, response.text
