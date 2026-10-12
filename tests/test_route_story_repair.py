"""``route-story --repair``: re-render only a prior run's QA-faulted frames.

A repair reads the fault report and the kept frames a QA-refused render left
behind (see ``film._kept_frames_root``), recaptures ONLY the faulted poses,
and refuses loudly rather than guessing whenever it cannot be sure it would
overwrite the right frame -- never a silent full re-render.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gomoveshift.pipeline.route_story import film as film_module
from gomoveshift.pipeline.route_story import frame_qa


def _write_report(path: Path, **overrides) -> dict:
    report = {
        "frames_faulted": 1,
        "frames": [1],
        "problems": {"fake refusal": 1},
        "render_identity": {"output": "film.mp4", "fps": 2, "width": 32, "height": 18, "legs": 1},
        "poses": {"1": {"longitude": 1.0, "latitude": 2.0, "zoom": 10.0}},
    }
    report.update(overrides)
    path.write_text(json.dumps(report), encoding="utf-8")
    return report


def _make_kept_root(output: Path, monkeypatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("GMS_INTERMEDIATE_DIR", str(tmp_path / "staging"))
    kept_root = film_module._kept_frames_root(output)
    kept_root.mkdir(parents=True)
    frames_dir = kept_root / "main-1-frames"
    frames_dir.mkdir()
    for index, color in enumerate([(1, 1, 1), (2, 2, 2), (3, 3, 3)]):
        film_module._save_frame(Image.new("RGB", (32, 18), color), frames_dir, index)
    recap_frames = kept_root / "recap-frames"
    recap_frames.mkdir()
    film_module._save_frame(Image.new("RGB", (32, 18), (4, 4, 4)), recap_frames, 0)
    return kept_root


def test_report_qa_faults_writes_render_identity_and_poses(tmp_path):
    output = tmp_path / "film.mp4"
    result = frame_qa.FrameQaResult(
        frame=1,
        void_fraction=1.0,
        unloaded_fraction=0.0,
        black_fraction=0.0,
        problems=("frame 1: fake refusal",),
    )
    ledger = frame_qa.FrameQaLedger()
    ledger.record(result, {"longitude": 1.0, "latitude": 2.0, "zoom": 10.0})

    film_module._report_qa_faults(ledger, output, fps=2, width=32, height=18, legs=1)

    report = json.loads((tmp_path / "film.qa-faults.json").read_text())
    assert report["render_identity"] == {
        "output": "film.mp4",
        "fps": 2,
        "width": 32,
        "height": 18,
        "legs": 1,
    }
    assert report["poses"] == {"1": {"longitude": 1.0, "latitude": 2.0, "zoom": 10.0}}
    assert report["frames"] == [1]


def test_repair_recaptures_only_the_faulted_frame(tmp_path, monkeypatch):
    output = tmp_path / "film.mp4"
    kept_root = _make_kept_root(output, monkeypatch, tmp_path)
    frames_dir = kept_root / "main-1-frames"
    before = {path.name: path.read_bytes() for path in frames_dir.glob("*.jpg")}

    report_path = tmp_path / "film.qa-faults.json"
    _write_report(report_path)

    recaptured_poses = []

    def recapture(pose, frame_index):
        recaptured_poses.append(pose)
        return Image.new("RGB", (32, 18), (9, 9, 9))

    film_module.repair_route_story_film(
        report_path,
        output=output,
        fps=2,
        width=32,
        height=18,
        recapture=recapture,
    )

    assert output.is_file()
    assert recaptured_poses == [{"longitude": 1.0, "latitude": 2.0, "zoom": 10.0}]
    after = {path.name: path.read_bytes() for path in frames_dir.glob("*.jpg")}
    assert after["000000.jpg"] == before["000000.jpg"]
    assert after["000002.jpg"] == before["000002.jpg"]
    assert after["000001.jpg"] != before["000001.jpg"]
    repaired = Image.open(frames_dir / "000001.jpg").convert("RGB")
    assert repaired.getpixel((0, 0)) == (9, 9, 9)


def test_repair_refuses_on_identity_mismatch(tmp_path, monkeypatch):
    output = tmp_path / "film.mp4"
    _make_kept_root(output, monkeypatch, tmp_path)
    report_path = tmp_path / "film.qa-faults.json"
    _write_report(report_path)

    with pytest.raises(ValueError, match="identity"):
        film_module.repair_route_story_film(
            report_path,
            output=output,
            fps=30,  # report says fps=2
            width=32,
            height=18,
            recapture=lambda pose, frame_index: Image.new("RGB", (32, 18)),
        )


def test_repair_refuses_on_missing_kept_dir(tmp_path):
    output = tmp_path / "film.mp4"
    report_path = tmp_path / "film.qa-faults.json"
    _write_report(report_path)

    with pytest.raises(ValueError, match="kept frames"):
        film_module.repair_route_story_film(
            report_path,
            output=output,
            fps=2,
            width=32,
            height=18,
            recapture=lambda pose, frame_index: Image.new("RGB", (32, 18)),
        )


def test_repair_refuses_on_a_multi_leg_report(tmp_path, monkeypatch):
    output = tmp_path / "film.mp4"
    _make_kept_root(output, monkeypatch, tmp_path)
    report_path = tmp_path / "film.qa-faults.json"
    _write_report(
        report_path,
        render_identity={
            "output": "film.mp4",
            "fps": 2,
            "width": 32,
            "height": 18,
            "legs": 2,
        },
    )

    with pytest.raises(ValueError, match="leg"):
        film_module.repair_route_story_film(
            report_path,
            output=output,
            fps=2,
            width=32,
            height=18,
            recapture=lambda pose, frame_index: Image.new("RGB", (32, 18)),
        )


def test_repair_refuses_on_a_faulted_frame_with_no_pose(tmp_path, monkeypatch):
    output = tmp_path / "film.mp4"
    _make_kept_root(output, monkeypatch, tmp_path)
    report_path = tmp_path / "film.qa-faults.json"
    _write_report(report_path, frames=[1, 2], poses={"1": {"longitude": 1.0, "latitude": 2.0}})

    with pytest.raises(ValueError, match="pose"):
        film_module.repair_route_story_film(
            report_path,
            output=output,
            fps=2,
            width=32,
            height=18,
            recapture=lambda pose, frame_index: Image.new("RGB", (32, 18)),
        )


class _FakeSceneCapture:
    """Stands in for ``scene.capture.SceneCapture``: one fixed raw map array."""

    def __init__(self, array: np.ndarray) -> None:
        self.array = array
        self.bypass_flags: list[bool] = []

    def render(self, pose, *, bypass_cache=False):
        self.bypass_flags.append(bypass_cache)
        return self.array.copy(), []


def test_repair_refuses_a_recaptured_frame_that_still_fails_frame_qa():
    # A repair that recaptures a pose which is STILL bad must refuse exactly as
    # a normal render would -- never paste the bad tile back into the film.
    still_black = np.zeros((18, 32, 3), dtype=np.uint8)
    with pytest.raises(frame_qa.FrameQaError):
        film_module._repair_scene_frame(
            _FakeSceneCapture(still_black),
            {"longitude": 1.0, "latitude": 2.0, "zoom": 10.0},
            frame_index=1,
            width=32,
            height=24,
        )


def test_repair_keeps_a_recaptured_frame_that_passes_frame_qa():
    rng = np.random.default_rng(7)
    healthy = rng.integers(40, 220, size=(18, 32, 3), dtype=np.uint8)
    capture = _FakeSceneCapture(healthy)
    image = film_module._repair_scene_frame(
        capture,
        {"longitude": 1.0, "latitude": 2.0, "zoom": 10.0},
        frame_index=1,
        width=32,
        height=24,
    )
    assert image.size == (32, 24)
    assert capture.bypass_flags == [True]
