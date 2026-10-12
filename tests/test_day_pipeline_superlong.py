"""sl_NN.mp4 production must never inflate lf_plans/num_lfs."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from gomoveshift.pipeline.day_grouper import DayBatch
from _state_helpers import make_state


def _make_scores(duration_sec: int) -> np.ndarray:
    arr = np.zeros((duration_sec, 9), dtype=np.float64)
    arr[:, 0] = np.arange(duration_sec, dtype=np.float64)  # timestamp
    arr[:, 1], arr[:, 2] = 0.7, 0.02  # composite score, motion energy
    return arr


def _make_config(superlong_enabled: bool, part_max_sec: int = 1200) -> SimpleNamespace:
    return SimpleNamespace(
        drive=SimpleNamespace(
            brand_logo_path=None,
            montage_max_duration_sec=15.0,
            ost_title_crossfade_sec=0.5,
            lf_selection_mode="best_moments",
            shorts_target_count=None,
            shorts_min_gap_sec=60.0,
            superlong_enabled=superlong_enabled,
            superlong_part_max_duration_sec=part_max_sec,
            superlong_split_grace=0.08,
        )
    )


def _run(tmp_path: Path, date: str, config, state):
    from gomoveshift.pipeline import day_pipeline

    ep = SimpleNamespace(
        index=1, concat_path=Path("/fake/ep_000.mp4"), total_duration=3600.0, date=date
    )
    day_batch = DayBatch(date=date, episodes=[ep])
    day_scores = _make_scores(3600)
    master_path = tmp_path / "out" / f"day_{date}" / f"day_{date}_master.mp4"
    master_path.parent.mkdir(parents=True, exist_ok=True)
    master_path.write_bytes(b"fake master")

    def _fake_ost(input_path, output_path, location, layout, **kwargs):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"fake ost")
        return output_path

    def _fake_assemble(master_path, segments, file_boundaries, output_path, work_dir, **kw):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"concat")
        return Path(output_path)

    def _fake_montage(raw_lf_path, peaks, output_path, work_dir, **kw):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"with_montage")
        return Path(output_path)

    def _fake_render(source_video, output_path, start_time, duration, location, **kwargs):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"fake short")
        return True

    seg = SimpleNamespace(source_file_index=0, start_time=0.0, end_time=3600.0)
    plan = SimpleNamespace(index=1, segments=[seg], montage_peaks=[])

    with (
        patch("gomoveshift.pipeline.day_pipeline.encode_day_master", return_value=master_path),
        patch("gomoveshift.pipeline.day_pipeline.apply_ost", side_effect=_fake_ost) as mock_ost,
        patch(
            "gomoveshift.pipeline.day_pipeline.extract_and_concat_segments",
            side_effect=_fake_assemble,
        ) as mock_extract,
        patch("gomoveshift.pipeline.day_pipeline.prepend_montage", side_effect=_fake_montage),
        patch("gomoveshift.pipeline.day_pipeline._render_short_ffmpeg", side_effect=_fake_render),
    ):
        artifacts = day_pipeline.run_day_pipeline(
            day_batch=day_batch,
            day_scores=day_scores,
            output_dir=tmp_path / "out",
            location="Test Drive, Test County",
            config=config,
            lf_plans=[plan],
            state=state,
        )
    return artifacts, mock_extract, mock_ost


class TestSuperlongPartProduction:
    def test_enabled_produces_parts_without_inflating_lf_count_and_resume_skips(self, tmp_path):
        date = "2026-06-01"
        # 1h day capped at 20min/part -> 3 parts, independent of the 1 LF plan.
        config = _make_config(superlong_enabled=True, part_max_sec=1200)
        state = make_state(ost_audit_confirmed=True)

        artifacts, mock_extract, mock_ost = _run(tmp_path, date, config, state)

        assert artifacts is not None
        # num_lfs must stay 1 — superlong parts never enter lf_plans/num_lfs.
        assert len(artifacts.regular_lf_paths) == 1
        sl_dir = tmp_path / "out" / f"day_{date}" / "superlong"
        for name in ("sl_01.mp4", "sl_02.mp4", "sl_03.mp4"):
            final = sl_dir / name
            assert final.is_file() and final.stat().st_size > 0, f"{name} missing/empty"
        record = state.get("superlong_parts")[date]
        assert record["part_count"] == 3
        assert len(record["records"]) == 3
        assert (mock_extract.call_count, mock_ost.call_count) == (4, 4)  # 1 LF + 3 parts

        # Every part record carries real metadata (title/chapters/description)
        # alongside path/index/duration — GPS-less fixture here (episodes carry
        # no gpx_path) so chapters are structurally skipped (has_gps=False).
        for i, rec in enumerate(record["records"], start=1):
            assert rec["title"] == f"Driving Test Drive, Test County — Part {i} of 3"
            assert rec["has_gps"] is False
            assert rec["chapters"] == []
            assert rec["location"] == "Test Drive, Test County"
            assert "Test Drive, Test County" in rec["description"]

        # Resume: LF (fingerprint) and superlong parts (final exists) are
        # both already fresh -> nothing re-encodes.
        _artifacts2, mock_extract2, mock_ost2 = _run(tmp_path, date, config, state)
        assert (mock_extract2.call_count, mock_ost2.call_count) == (0, 0)


class TestSuperlongOutputSpaceGate:
    def test_insufficient_space_raises_before_any_part_encode(self, tmp_path):
        from gomoveshift.pipeline import day_pipeline

        master_path = tmp_path / "day_master.mp4"
        master_path.write_bytes(b"fake master")
        day_dir = tmp_path / "day_2026-06-01"
        day_dir.mkdir()
        stop_mask = np.zeros(3600, dtype=bool)
        ep = SimpleNamespace(concat_path=Path("/fake/ep_000.mp4"))
        config = _make_config(superlong_enabled=True, part_max_sec=1200)

        with (
            patch(
                "gomoveshift.pipeline.day_pipeline.check_superlong_output_space",
                side_effect=RuntimeError(
                    "Insufficient output space for superlong parts: need 40.0 GB"
                ),
            ),
            patch("gomoveshift.pipeline.day_pipeline.extract_and_concat_segments") as mock_extract,
            patch("gomoveshift.pipeline.day_pipeline.apply_ost") as mock_ost,
        ):
            with pytest.raises(RuntimeError, match="output space"):
                day_pipeline._produce_superlong_parts(
                    date="2026-06-01",
                    stop_mask=stop_mask,
                    episodes=[ep],
                    file_boundaries=[(0.0, 3600.0)],
                    master_path=master_path,
                    day_dir=day_dir,
                    config=config,
                    location="Test Drive, Test County",
                    ost_crossfade_sec=0.5,
                )
        mock_extract.assert_not_called()
        mock_ost.assert_not_called()


def test_superlong_lower_thirds_threaded_into_ost_and_persisted_for_review(tmp_path):
    """Chapter-derived town events threaded into apply_ost and persisted
    onto the part record for review (state["superlong_parts"]). The
    part's OWN ``record["chapters"]`` (via ``chapter_town_events``) is the
    only source — there is no override kwarg."""
    from gomoveshift.pipeline import day_pipeline

    master_path = tmp_path / "day_master.mp4"
    master_path.write_bytes(b"fake master")
    day_dir = tmp_path / "day_2026-06-01"
    day_dir.mkdir()
    ep = SimpleNamespace(concat_path=Path("/fake/ep_000.mp4"))
    config = _make_config(superlong_enabled=True, part_max_sec=1200)
    chapters = [
        {"timestamp": "0:00", "title": "Start - Test Drive"},
        {"timestamp": "5:00", "title": "Entering Beaufort, SC"},
        {"timestamp": "15:00", "title": "Entering Yemassee, SC"},
        {"timestamp": "19:45", "title": "Finish - Test County"},
    ]
    transitions = [("Beaufort, SC", 300.0), ("Yemassee, SC", 900.0)]

    def _fake_ost(input_path, output_path, location, layout, **kwargs):
        Path(output_path).write_bytes(b"fake ost")
        return output_path

    def _fake_assemble(master_path, segments, file_boundaries, output_path, work_dir, **kw):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"concat")
        return Path(output_path)

    with (
        patch("gomoveshift.pipeline.day_pipeline.apply_ost", side_effect=_fake_ost) as mock_ost,
        patch(
            "gomoveshift.pipeline.day_pipeline.extract_and_concat_segments",
            side_effect=_fake_assemble,
        ),
        patch(
            "gomoveshift.pipeline.day_pipeline._build_superlong_part_metadata",
            return_value={
                "title": "Driving Test Drive, Test County — Part 1 of 1",
                "description": "desc",
                "chapters": chapters,
                "video_duration": 1200.0,
                "has_gps": True,
                "location": "Test Drive, Test County",
            },
        ),
    ):
        result = day_pipeline._produce_superlong_parts(
            date="2026-06-01",
            stop_mask=np.zeros(1200, dtype=bool),
            episodes=[ep],
            file_boundaries=[(0.0, 1200.0)],
            master_path=master_path,
            day_dir=day_dir,
            config=config,
            location="Test Drive, Test County",
            ost_crossfade_sec=0.5,
            # G-sl gate: drives the burn directly, so it stands in for a reviewer.
            state=make_state(ost_audit_confirmed_this_run=True),
        )

    assert mock_ost.call_args[1]["lower_thirds"] == transitions
    assert result["records"][0]["lower_thirds"] == [
        {"label": "Beaufort, SC", "time_sec": 300.0},
        {"label": "Yemassee, SC", "time_sec": 900.0},
    ]


def test_produce_superlong_parts_wires_chapter_derived_lower_thirds_in_production(tmp_path):
    """The ONLY production call site (day_pipeline.py run_day_pipeline) never
    passes ``place_transitions_by_part`` — regression pin for the bug where
    the mechanism was wired but stayed inert because nothing called it
    with real events. Without the kwarg, the part's OWN already-resolved
    ``record["chapters"]`` (via ``chapter_town_events``) must still produce
    non-empty ``lower_thirds``."""
    from gomoveshift.pipeline import day_pipeline

    master_path = tmp_path / "day_master.mp4"
    master_path.write_bytes(b"fake master")
    day_dir = tmp_path / "day_2026-06-01"
    day_dir.mkdir()
    ep = SimpleNamespace(concat_path=Path("/fake/ep_000.mp4"))
    config = _make_config(superlong_enabled=True, part_max_sec=1200)
    chapters = [
        {"timestamp": "0:00", "title": "Start - Test Drive"},
        {"timestamp": "5:00", "title": "Entering Riverside, SC"},
        {"timestamp": "15:00", "title": "En route — Somewhere"},
        {"timestamp": "19:45", "title": "Finish - Test County"},
    ]

    def _fake_ost(input_path, output_path, location, layout, **kwargs):
        Path(output_path).write_bytes(b"fake ost")
        return output_path

    def _fake_assemble(master_path, segments, file_boundaries, output_path, work_dir, **kw):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"concat")
        return Path(output_path)

    with (
        patch("gomoveshift.pipeline.day_pipeline.apply_ost", side_effect=_fake_ost) as mock_ost,
        patch(
            "gomoveshift.pipeline.day_pipeline.extract_and_concat_segments",
            side_effect=_fake_assemble,
        ),
        patch(
            "gomoveshift.pipeline.day_pipeline._build_superlong_part_metadata",
            return_value={
                "title": "Driving Test Drive — Part 1 of 1",
                "description": "desc",
                "chapters": chapters,
                "video_duration": 1200.0,
                "has_gps": True,
                "location": "Test Drive, Test County",
            },
        ),
    ):
        day_pipeline._produce_superlong_parts(
            date="2026-06-01",
            stop_mask=np.zeros(1200, dtype=bool),
            episodes=[ep],
            file_boundaries=[(0.0, 1200.0)],
            master_path=master_path,
            day_dir=day_dir,
            config=config,
            location="Test Drive, Test County",
            ost_crossfade_sec=0.5,
            # No place_transitions_by_part kwarg -- the ONLY shape the real
            # caller (run_day_pipeline) uses.
            state=make_state(ost_audit_confirmed_this_run=True),
        )

    lower_thirds = mock_ost.call_args[1]["lower_thirds"]
    assert lower_thirds, "production call site never threads chapter-derived lower-thirds"
    assert lower_thirds == [("Riverside, SC", 300.0)]


def test_concat_failure_hard_fails_instead_of_skipping_the_part(tmp_path):
    """A part whose concat fails must RAISE, not warn and continue.

    If a superlong concat hits keycut's 1800s stream-copy
    timeout and ``extract_and_concat_segments`` returns None, the loop must not just log
    ``superlong part 1 concat failed`` and move on; the day would then report
    its counts and walk into Phase D with no part on disk, the same
    silent-superlong-loss class as the resume skip. A part that cannot be built
    is a failed day, not a warning.
    """
    from gomoveshift.pipeline import day_pipeline

    master_path = tmp_path / "day_master.mp4"
    master_path.write_bytes(b"fake master")
    day_dir = tmp_path / "day_2026-06-01"
    day_dir.mkdir()
    ep = SimpleNamespace(concat_path=Path("/fake/ep_000.mp4"))
    config = _make_config(superlong_enabled=True, part_max_sec=1200)

    with (
        patch("gomoveshift.pipeline.day_pipeline.apply_ost") as mock_ost,
        patch(
            "gomoveshift.pipeline.day_pipeline.extract_and_concat_segments",
            return_value=None,  # the timeout's observable result
        ),
        patch(
            "gomoveshift.pipeline.day_pipeline._build_superlong_part_metadata",
            return_value={
                "title": "Driving Test Drive, Test County",
                "description": "desc",
                "chapters": [],
                "video_duration": 1200.0,
                "has_gps": True,
                "location": "Test Drive, Test County",
            },
        ),
        pytest.raises(RuntimeError, match="(?i)superlong part 1.*concat"),
    ):
        day_pipeline._produce_superlong_parts(
            date="2026-06-01",
            stop_mask=np.zeros(1200, dtype=bool),
            episodes=[ep],
            file_boundaries=[(0.0, 1200.0)],
            master_path=master_path,
            day_dir=day_dir,
            config=config,
            location="Test Drive, Test County",
            ost_crossfade_sec=0.5,
            state=make_state(ost_audit_confirmed_this_run=True),
        )

    mock_ost.assert_not_called()  # never burn from a failed concat
