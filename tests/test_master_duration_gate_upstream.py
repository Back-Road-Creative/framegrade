"""The day-master duration-integrity gate must run at PROMOTION time.

``assert_master_duration_integrity`` compares a freshly-encoded day master's
own ffprobe duration against ``day_batch.total_duration``. It used to run only
from ``build_master_record`` — the LAST step of ``run_day_pipeline``, i.e. after
the per-LF OST burns, every short and every superlong part had already been
NVENC-encoded FROM that master. A truncated master therefore cost a whole day of
derived encodes before anything complained (a master is ~60 GB / ~6 h).

These tests pin the gate to ``encode_day_master``, immediately after the atomic
``os.replace`` promotion, so the bad master is impossible to derive anything
from. Tolerance/exception-type coverage stays in
``test_day_master_reuse_no_raw.py``.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gomoveshift.pipeline.day_grouper import DayBatch
from _state_helpers import make_state


DATE = "2026-02-20"
EXPECTED_S = 3600.0
NUM_COLUMNS = 9
COL_TIMESTAMP = 0
COL_COMPOSITE = 1
COL_MOTION_ENERGY = 2


@pytest.fixture(autouse=True)
def _blur_off(monkeypatch):
    """Duration-gate MECHANICS, orthogonal to the privacy blur that is default-ON
    in ``encode_day_master`` (covered by ``test_day_master_blur.py``)."""
    monkeypatch.setattr("gomoveshift.pipeline.blur_gate.blur_required", lambda cfg: False)


@dataclass
class _FakeEpisode:
    index: int
    concat_path: Path
    total_duration: float
    date: str
    creation_time: object = None


def _make_day_batch(tmp_path: Path, ep_count: int = 2) -> DayBatch:
    episodes = []
    for i in range(ep_count):
        p = tmp_path / f"ep_{i:03d}.mp4"
        p.touch()
        episodes.append(
            _FakeEpisode(
                index=i + 1,
                concat_path=p,
                total_duration=EXPECTED_S / ep_count,
                date=DATE,
            )
        )
    return DayBatch(date=DATE, episodes=episodes)


def _make_scores(duration_sec: int, score: float = 0.6, motion: float = 0.05) -> np.ndarray:
    arr = np.zeros((duration_sec, NUM_COLUMNS), dtype=np.float64)
    arr[:, COL_TIMESTAMP] = np.arange(duration_sec, dtype=np.float64)
    arr[:, COL_COMPOSITE] = score
    arr[:, COL_MOTION_ENERGY] = motion
    return arr


def _stub_encode_io(monkeypatch, probed_duration: float):
    """Stand in for the heavy encode: ffmpeg concat "succeeds", SinglePass writes
    the ``.tmp`` master, and every ffprobe reports ``probed_duration``."""

    def fake_process(self, video_path, output_path, location, **kwargs):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"fake master")
        return MagicMock(success=True, output_path=output_path)

    monkeypatch.setattr(
        "gomoveshift.pipeline.single_pass.SinglePassProcessor.process", fake_process
    )
    monkeypatch.setattr(
        "gomoveshift.pipeline.day_master_encoder.subprocess.run",
        lambda cmd, *a, **kw: MagicMock(returncode=0, stdout=b"", stderr=b""),
    )
    monkeypatch.setattr(
        "gomoveshift.pipeline.validation.ffprobe_client.FFprobeClient.probe_duration",
        lambda self, path: probed_duration,
    )


def _write(output_path, payload: bytes):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_bytes(payload)
    return Path(output_path)


@contextmanager
def _run_day(tmp_path, monkeypatch, probed_duration: float):
    """Run ``run_day_pipeline`` with the REAL ``encode_day_master`` and every
    derived-surface encoder mocked, so the mocks record whether any derived
    encode was attempted."""
    from gomoveshift.pipeline import day_pipeline

    _stub_encode_io(monkeypatch, probed_duration)
    lf_plans = [
        SimpleNamespace(
            index=1,
            segments=[
                SimpleNamespace(
                    source_file_index=0,
                    start_time=100.0,
                    end_time=200.0,
                    duration=100.0,
                    gps_center_lat=0.0,
                    gps_center_lon=0.0,
                )
            ],
            montage_peaks=[],
        )
    ]
    with (
        patch(
            "gomoveshift.pipeline.day_pipeline.apply_ost",
            side_effect=lambda input_path, output_path, location, layout, **kw: _write(
                output_path, b"ost"
            ),
        ) as mock_ost,
        patch(
            "gomoveshift.pipeline.day_pipeline._render_short_ffmpeg",
            side_effect=lambda source_video, output_path, start_time, duration, location, **kw: (
                bool(_write(output_path, b"short"))
            ),
        ) as mock_short,
        patch(
            "gomoveshift.pipeline.day_pipeline.extract_and_concat_segments",
            side_effect=lambda master_path, segments, file_boundaries, output_path, work_dir, **kw: (
                _write(output_path, b"lf")
            ),
        ) as mock_cut,
        patch(
            "gomoveshift.pipeline.day_pipeline.prepend_montage",
            side_effect=lambda raw_lf_path, peaks, output_path, work_dir, **kw: _write(
                output_path, b"montage"
            ),
        ),
    ):
        run = lambda: day_pipeline.run_day_pipeline(  # noqa: E731
            day_batch=_make_day_batch(tmp_path),
            day_scores=_make_scores(int(EXPECTED_S)),
            output_dir=tmp_path / "out",
            location="Test Location, SC",
            config=None,
            lf_plans=lf_plans,
            state=make_state(ost_audit_confirmed=True),
        )
        yield run, mock_ost, mock_short, mock_cut


def test_truncated_master_raises_from_encode_day_master(tmp_path, monkeypatch):
    """The gate fires inside ``encode_day_master`` itself — the caller never even
    receives a path to a short master."""
    from gomoveshift.config import Config
    from gomoveshift.pipeline.day_master_encoder import (
        BatchDurationIntegrityError,
        encode_day_master,
    )

    _stub_encode_io(monkeypatch, probed_duration=EXPECTED_S / 2)
    with pytest.raises(BatchDurationIntegrityError, match="freshly-encoded master"):
        encode_day_master(
            day_batch=_make_day_batch(tmp_path),
            output_dir=tmp_path / "out",
            config=Config(),
        )


def test_healthy_master_is_returned_by_encode_day_master(tmp_path, monkeypatch):
    """A master whose duration matches the batch promotes and returns normally."""
    from gomoveshift.config import Config
    from gomoveshift.pipeline.day_master_encoder import encode_day_master

    _stub_encode_io(monkeypatch, probed_duration=EXPECTED_S)
    result = encode_day_master(
        day_batch=_make_day_batch(tmp_path),
        output_dir=tmp_path / "out",
        config=Config(),
    )
    assert result == tmp_path / "out" / f"day_{DATE}_master.mp4"
    assert result.exists()


def test_truncated_master_blocks_before_any_derived_encode(tmp_path, monkeypatch):
    """Core ordering guarantee: no LF cut, no OST burn, no short is produced from
    a master that failed the duration gate."""
    from gomoveshift.pipeline.day_master_encoder import BatchDurationIntegrityError

    with _run_day(tmp_path, monkeypatch, probed_duration=EXPECTED_S / 2) as (
        run,
        mock_ost,
        mock_short,
        mock_cut,
    ):
        with pytest.raises(BatchDurationIntegrityError, match="freshly-encoded master"):
            run()
    assert mock_cut.call_count == 0, "an LF was cut from a master that failed the gate"
    assert mock_ost.call_count == 0, "an OST burn ran on a master that failed the gate"
    assert mock_short.call_count == 0, "a short was rendered from a failed master"


def test_healthy_master_completes_the_day_pipeline(tmp_path, monkeypatch):
    """The gate is not a new failure mode for a good day: LFs and shorts still ship."""
    with _run_day(tmp_path, monkeypatch, probed_duration=EXPECTED_S) as (
        run,
        mock_ost,
        mock_short,
        _mock_cut,
    ):
        artifacts = run()

    assert artifacts is not None
    assert artifacts.date == DATE
    assert len(artifacts.regular_lf_paths) >= 1
    assert mock_ost.call_count == len(artifacts.regular_lf_paths)
    assert mock_short.call_count == len(artifacts.short_paths)
