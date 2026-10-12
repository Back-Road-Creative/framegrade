"""full_drive synthetic LFSegments must carry a real GPS centroid derived
from the day's GPX (``day_gps_points``) instead of always ``(0.0, 0.0)`` —
the (0.0, 0.0) sentinel makes ``lf_listing`` treat a GPS-equipped day as
"no GPS", forcing a needless operator ``--per-lf-location`` override.
"""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from gomoveshift.pipeline.day_grouper import DayBatch
from _state_helpers import make_state


NUM_COLUMNS = 9
COL_TIMESTAMP = 0
COL_COMPOSITE = 1
COL_MOTION_ENERGY = 2


def _make_scores(duration_sec: int, score: float = 0.7, motion: float = 0.02) -> np.ndarray:
    arr = np.zeros((duration_sec, NUM_COLUMNS), dtype=np.float64)
    arr[:, COL_TIMESTAMP] = np.arange(duration_sec, dtype=np.float64)
    arr[:, COL_COMPOSITE] = score
    arr[:, COL_MOTION_ENERGY] = motion
    return arr


@dataclass
class _FakeEpisode:
    index: int
    concat_path: Path
    total_duration: float
    date: str
    creation_time: object = None


def _make_day_batch(date: str, ep_count: int = 1, per_ep_duration: float = 2400.0) -> DayBatch:
    episodes = [
        _FakeEpisode(
            index=i + 1,
            concat_path=Path(f"/fake/ep_{i:03d}.mp4"),
            total_duration=per_ep_duration,
            date=date,
        )
        for i in range(ep_count)
    ]
    return DayBatch(date=date, episodes=episodes)


def _make_config(full_drive_max_sec: int = 7200):
    return SimpleNamespace(
        drive=SimpleNamespace(
            brand_logo_path=None,
            montage_max_duration_sec=15.0,
            ost_title_crossfade_sec=0.5,
            lf_selection_mode="full_drive",
            full_drive_max_duration_sec=full_drive_max_sec,
            shorts_target_count=None,
            shorts_min_gap_sec=60.0,
        )
    )


def _patch_targets(tmp_path):
    master_path = tmp_path / "out" / "day_2026-05-01" / "day_2026-05-01_master.mp4"
    master_path.parent.mkdir(parents=True, exist_ok=True)
    master_path.write_bytes(b"fake master")

    def _fake_ost(input_path, output_path, location, layout, **kwargs):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"fake ost")
        return output_path

    def _fake_render(source_video, output_path, start_time, duration, location, **kwargs):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"fake short")
        return True

    def _fake_assemble(master_path, segments, file_boundaries, output_path, work_dir, **kw):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"concat")
        return Path(output_path)

    def _fake_montage(raw_lf_path, peaks, output_path, work_dir, **kw):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"with_montage")
        return Path(output_path)

    return master_path, _fake_ost, _fake_render, _fake_assemble, _fake_montage


class TestFullDriveSegmentGpsCentroid:
    def test_full_drive_segments_get_gps_centroid_from_day_gpx(self, tmp_path):
        """A full_drive day with GPX covering the whole span must produce
        synthetic LFSegment(s) whose gps_center_lat/lon are the centroid of
        the in-window fixes — not the (0.0, 0.0) sentinel."""
        from gomoveshift.pipeline import day_pipeline

        day_batch = _make_day_batch("2026-05-01", ep_count=1, per_ep_duration=2400.0)
        day_scores = _make_scores(2400, score=0.7)
        output_base = tmp_path / "out"
        config = _make_config(full_drive_max_sec=7200)

        master_path, fake_ost, fake_render, fake_assemble, fake_montage = _patch_targets(tmp_path)

        # GPS points spanning the whole 2400s range, centered on a known
        # non-zero lat/lon.
        day_gps_points = [
            (0.0, 51.0, -9.0),
            (1200.0, 51.5, -9.5),
            (2399.0, 52.0, -10.0),
        ]

        captured = {}

        def _capture_assemble(master_path, segments, file_boundaries, output_path, work_dir, **kw):
            captured["segments"] = list(segments)
            return fake_assemble(
                master_path, segments, file_boundaries, output_path, work_dir, **kw
            )

        with (
            patch(
                "gomoveshift.pipeline.day_pipeline.encode_day_master",
                return_value=master_path,
            ),
            patch(
                "gomoveshift.pipeline.day_pipeline.apply_ost",
                side_effect=fake_ost,
            ),
            patch(
                "gomoveshift.pipeline.day_pipeline._render_short_ffmpeg",
                side_effect=fake_render,
            ),
            patch(
                "gomoveshift.pipeline.day_pipeline.extract_and_concat_segments",
                side_effect=_capture_assemble,
            ),
            patch(
                "gomoveshift.pipeline.day_pipeline.prepend_montage",
                side_effect=fake_montage,
            ),
        ):
            artifacts = day_pipeline.run_day_pipeline(
                day_batch=day_batch,
                day_scores=day_scores,
                output_dir=output_base,
                location="Test Drive, Test County",
                config=config,
                lf_plans=None,
                gps_points_by_date={"2026-05-01": day_gps_points},
                state=make_state(ost_audit_confirmed=True),
            )

        assert artifacts is not None
        assert "segments" in captured
        segs = captured["segments"]
        assert len(segs) >= 1
        for seg in segs:
            assert (seg.gps_center_lat, seg.gps_center_lon) != (0.0, 0.0)
            # Sane bounds — within the supplied fixes' range.
            assert 51.0 <= seg.gps_center_lat <= 52.0
            assert -10.0 <= seg.gps_center_lon <= -9.0

    def test_full_drive_segment_stays_zero_when_window_has_no_points(self, tmp_path):
        """A segment window with zero GPS points inside it keeps the
        (0.0, 0.0) sentinel — no drive-wide fallback."""
        from gomoveshift.pipeline import day_pipeline

        day_batch = _make_day_batch("2026-05-01", ep_count=1, per_ep_duration=2400.0)
        day_scores = _make_scores(2400, score=0.7)
        output_base = tmp_path / "out"
        config = _make_config(full_drive_max_sec=7200)

        master_path, fake_ost, fake_render, fake_assemble, fake_montage = _patch_targets(tmp_path)

        # GPX only covers the first half of the day; the rest of the
        # (single, contiguous) full_drive range has no fixes.
        day_gps_points = [
            (0.0, 51.0, -9.0),
            (10.0, 51.0, -9.0),
        ]

        captured = {}

        def _capture_assemble(master_path, segments, file_boundaries, output_path, work_dir, **kw):
            captured["segments"] = list(segments)
            return fake_assemble(
                master_path, segments, file_boundaries, output_path, work_dir, **kw
            )

        with (
            patch(
                "gomoveshift.pipeline.day_pipeline.encode_day_master",
                return_value=master_path,
            ),
            patch(
                "gomoveshift.pipeline.day_pipeline.apply_ost",
                side_effect=fake_ost,
            ),
            patch(
                "gomoveshift.pipeline.day_pipeline._render_short_ffmpeg",
                side_effect=fake_render,
            ),
            patch(
                "gomoveshift.pipeline.day_pipeline.extract_and_concat_segments",
                side_effect=_capture_assemble,
            ),
            patch(
                "gomoveshift.pipeline.day_pipeline.prepend_montage",
                side_effect=fake_montage,
            ),
        ):
            day_pipeline.run_day_pipeline(
                day_batch=day_batch,
                day_scores=day_scores,
                output_dir=output_base,
                location="Test Drive, Test County",
                config=config,
                lf_plans=None,
                gps_points_by_date=None,
                state=make_state(ost_audit_confirmed=True),
            )

        assert "segments" in captured
        for seg in captured["segments"]:
            assert (seg.gps_center_lat, seg.gps_center_lon) == (0.0, 0.0)
