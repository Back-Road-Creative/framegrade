"""The gate asks whether you played each video, so the page can play it.

`review_gate.py` phrases Gate C/D as "Played each LF and reviewed all metadata
for <date>?" and the console answered it with one 320px still frame per video
(`/drives/gates/frame`). A reviewer cannot answer that question from a
still, and the approval it records is what stops a delivery reaching the
queue. These pin a playback route that reuses the gate's own containment --
an index into `question["artifacts"]`, never a request-supplied path.
"""

import json

import pytest
from fastapi.testclient import TestClient

from gomoveshift.console import drives, jobs
from gomoveshift.console.app import create_app

DATE_KEY = "2026-08-01"


def _write_tiny_video(path) -> None:
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (32, 32))
    assert writer.isOpened(), "could not open VideoWriter for fixture"
    for _ in range(5):
        writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    writer.release()
    assert path.exists() and path.stat().st_size > 0


def _gate(tmp_path, monkeypatch, *, artifacts=None, suffix=".mp4"):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    output_dir = tmp_path / "out"
    video = output_dir / f"lf_01{suffix}"
    _write_tiny_video(video)
    metadata = output_dir / "day_lf01_youtube.json"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(json.dumps({"title": "Sample Drive"}), encoding="utf-8")
    (output_dir / "_gates").mkdir(parents=True, exist_ok=True)
    if artifacts is None:
        artifacts = [str(video), str(metadata)]
    (output_dir / "_gates" / f"gate_cd_{DATE_KEY}.question.json").write_text(
        json.dumps(
            {
                "question": f"Played each LF and reviewed all metadata for {DATE_KEY}?",
                "choices": ["approve", "reject"],
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )
    store = jobs.JobStore(tmp_path / "jobs.sqlite3")
    store.create("drive", ["gms", "drive", "d", "--output-dir", str(output_dir)], log_path="")
    return output_dir, video


def _question(output_dir):
    return drives.read_gate_question(output_dir, DATE_KEY)


def test_the_gate_serves_the_video_it_is_asking_about(tmp_path, monkeypatch):
    output_dir, video = _gate(tmp_path, monkeypatch)

    resolved = drives.resolve_gate_video_media(output_dir, _question(output_dir), 0)

    assert resolved is not None
    path, content_type = resolved
    assert path == video.resolve()
    assert content_type == "video/mp4"


def test_an_index_past_the_end_is_not_a_video(tmp_path, monkeypatch):
    output_dir, _ = _gate(tmp_path, monkeypatch)

    assert drives.resolve_gate_video_media(output_dir, _question(output_dir), 5) is None
    assert drives.resolve_gate_video_media(output_dir, _question(output_dir), -1) is None


def test_an_artifact_outside_the_output_dir_is_never_served(tmp_path, monkeypatch):
    """The containment that already guards the frame route, on the route that
    serves whole files rather than one re-encoded frame."""
    outside = tmp_path / "elsewhere" / "private.mp4"
    _write_tiny_video(outside)
    output_dir, video = _gate(tmp_path, monkeypatch, artifacts=None)
    question = dict(_question(output_dir))
    question["artifacts"] = [str(outside)]

    assert drives.resolve_gate_video_media(output_dir, question, 0) is None


def test_a_non_video_artifact_is_never_served(tmp_path, monkeypatch):
    output_dir, video = _gate(tmp_path, monkeypatch)
    question = dict(_question(output_dir))
    question["artifacts"] = [str(output_dir / "day_lf01_youtube.json")]

    assert drives.resolve_gate_video_media(output_dir, question, 0) is None


@pytest.mark.parametrize(
    "suffix,expected",
    [(".mp4", "video/mp4"), (".mov", "video/quicktime"), (".mkv", "video/x-matroska")],
)
def test_every_supported_suffix_has_a_content_type(suffix, expected):
    """Derived from one table, never a second hand-kept list: a suffix that
    passes the allowlist with no content type would be a KeyError -- a 500 on
    a legitimate video."""
    assert drives.GATE_VIDEO_CONTENT_TYPES[suffix] == expected
    assert set(drives.GATE_VIDEO_CONTENT_TYPES) == set(drives.SUPPORTED_VIDEO_SUFFIXES)


def test_the_route_streams_the_bytes(tmp_path, monkeypatch):
    output_dir, video = _gate(tmp_path, monkeypatch)

    response = TestClient(create_app()).get(
        "/drives/gates/video",
        params={"output_dir": str(output_dir), "date_key": DATE_KEY, "index": 0},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("video/mp4")
    assert response.content == video.read_bytes()


def test_the_route_supports_seeking(tmp_path, monkeypatch):
    """A reviewer who cannot scrub has not been given a player. A 200 for a
    ranged request means the browser must download the whole file to move."""
    output_dir, video = _gate(tmp_path, monkeypatch)

    response = TestClient(create_app()).get(
        "/drives/gates/video",
        params={"output_dir": str(output_dir), "date_key": DATE_KEY, "index": 0},
        headers={"Range": "bytes=0-99"},
    )

    assert response.status_code == 206
    assert response.content == video.read_bytes()[:100]


def test_an_unknown_output_dir_is_a_404_not_a_read(tmp_path, monkeypatch):
    output_dir, _ = _gate(tmp_path, monkeypatch)

    response = TestClient(create_app()).get(
        "/drives/gates/video",
        params={"output_dir": str(tmp_path / "nope"), "date_key": DATE_KEY, "index": 0},
    )

    assert response.status_code == 404


def test_a_bad_date_key_is_a_404(tmp_path, monkeypatch):
    output_dir, _ = _gate(tmp_path, monkeypatch)

    response = TestClient(create_app()).get(
        "/drives/gates/video",
        params={"output_dir": str(output_dir), "date_key": "../../etc", "index": 0},
    )

    assert response.status_code == 404


def test_the_gate_page_offers_a_player_for_each_video(tmp_path, monkeypatch):
    output_dir, _ = _gate(tmp_path, monkeypatch)

    text = (
        TestClient(create_app())
        .get("/drives/gates", params={"output_dir": str(output_dir), "date_key": DATE_KEY})
        .text
    )

    assert "<video" in text
    assert "index=0" in text


def test_the_still_frame_stays_as_the_poster(tmp_path, monkeypatch):
    """The accept-case: the existing frame route is not removed, it becomes
    the thing shown before playback starts."""
    output_dir, _ = _gate(tmp_path, monkeypatch)

    text = (
        TestClient(create_app())
        .get("/drives/gates", params={"output_dir": str(output_dir), "date_key": DATE_KEY})
        .text
    )

    assert "/drives/gates/frame" in text
    assert "poster=" in text


def test_an_unreadable_video_still_renders_an_honest_row(tmp_path, monkeypatch):
    """The other accept-case: a gate whose artifact is missing keeps the
    "could not be read" state rather than a broken player."""
    output_dir, video = _gate(tmp_path, monkeypatch)
    video.unlink()

    text = (
        TestClient(create_app())
        .get("/drives/gates", params={"output_dir": str(output_dir), "date_key": DATE_KEY})
        .text
    )

    assert "could not be read" in text
    assert "<video" not in text
