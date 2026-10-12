"""Tests for ``state["per_lf_recording_date"]`` population on every Phase BC
exit path + standardized "📅 Filmed: <date>" description line.

Bug context: ``state["per_lf_recording_date"]`` was populated only inside the
FRESH-encode partition path (``_build_per_lf_recording_date_map(lf_plans_by_date)``).
The finals-only resume fast-path skips ``_partition_lf_plans_by_date`` and left
the field empty, so long-form descriptions lost their Filmed line. The disk layout
(``day_<DATE>/regular_lfs/lf_NN.mp4``) is the source of truth.

This test file pins:

  1. Path-derivation helper extracts the date from each LF's day_<DATE>
     parent directory.
  2. The fresh-encode path (original behaviour) keeps producing the map
     it always did.
  3. The finals-only fast-path now ALSO produces the map.
  4. The dates match the day directory each LF lives under.
  5. The Phase BC invariant raises when an LF goes unmapped.
  6. The longform metadata description emits "📅 Filmed: <date>".
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from _state_helpers import make_state


# ── Helpers (mirror tests/test_phase_bc_fast_path_master_integrity.py) ─


def _make_state(tmp_path, dates=("2026-02-20", "2026-02-21"), per_day_total=1500.0):
    output_base = tmp_path / "out"
    stages_dir = output_base / "_stages"
    stages_dir.mkdir(parents=True)

    episodes = []
    episode_files = []
    episode_dates_list = []
    file_durations = []
    file_boundaries = []
    cursor = 0.0
    for i, d in enumerate(dates):
        ep_path = f"/fake/ep_{i:03d}.mp4"
        episode_files.append(ep_path)
        episode_dates_list.append(d)
        file_durations.append(per_day_total)
        file_boundaries.append((cursor, cursor + per_day_total))
        cursor += per_day_total
        episodes.append(
            SimpleNamespace(
                index=i,
                concat_path=Path(ep_path),
                date=d,
                total_duration=per_day_total,
                gpx_path=None,
                creation_time=None,
            )
        )
    ingest_plan = SimpleNamespace(episodes=episodes)

    state = make_state(
        output_base=output_base,
        ingest_plan=ingest_plan,
        episode_files=episode_files,
        episode_dates=episode_dates_list,
        episode_locations=[""] * len(dates),
        file_durations=file_durations,
        region="usa",
        location="Test",
        drive_scores=None,
        file_boundaries=file_boundaries,
    )
    return state, output_base, stages_dir


def _write_finals(day_dir: Path, lf_indices=(1,)) -> list[Path]:
    """Create the lf_NN.mp4 + short_NN.mp4 layout the resume fast-paths read."""
    lf_dir = day_dir / "regular_lfs"
    sh_dir = day_dir / "shorts"
    lf_dir.mkdir(parents=True, exist_ok=True)
    sh_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for idx in lf_indices:
        p = lf_dir / f"lf_{idx:02d}.mp4"
        p.write_bytes(b"final-lf-bytes")
        paths.append(p)
        (sh_dir / f"short_{idx:02d}.mp4").write_bytes(b"final-short-bytes")
    return paths


def _run_phase(state, run_all_days_impl):
    from gomoveshift.config import Config
    from gomoveshift.pipeline.drive_phases.phase_bc_day_first import PhaseBCDayFirst

    geocoder = MagicMock()
    geocoder.reverse_geocode.return_value = None

    class _FakeFFprobeClient:
        def __init__(self, _subprocess_module, *, timeout=30):
            pass

        def probe_duration(self, _path):
            return 1500.0

    cfg = Config()
    phase = PhaseBCDayFirst(cfg)
    with (
        patch(
            "gomoveshift.pipeline.gps_enrichment._build_shared_geocoder",
            return_value=geocoder,
        ),
        patch(
            "gomoveshift.pipeline.drive_phases.phase_bc_day_first.run_all_days",
            side_effect=run_all_days_impl,
        ),
        patch(
            "gomoveshift.pipeline.validation.ffprobe_client.FFprobeClient",
            _FakeFFprobeClient,
        ),
    ):
        phase.run(state, probe_duration_fn=lambda p: 100.0)


# ── Tests ────────────────────────────────────────────────────────────


class TestPathDerivationHelper:
    def test_extracts_date_from_day_dir(self, tmp_path):
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
            _derive_per_lf_recording_date_from_paths,
        )

        paths = [
            tmp_path / "stages" / "day_2026-02-20" / "regular_lfs" / "lf_01.mp4",
            tmp_path / "stages" / "day_2026-02-21" / "regular_lfs" / "lf_02.mp4",
            tmp_path / "stages" / "day_2026-02-21" / "regular_lfs" / "lf_03.mp4",
        ]
        skeleton_lfs = [
            SimpleNamespace(index=1),
            SimpleNamespace(index=2),
            SimpleNamespace(index=3),
        ]
        result = _derive_per_lf_recording_date_from_paths(paths, skeleton_lfs)
        assert result == {1: "2026-02-20", 2: "2026-02-21", 3: "2026-02-21"}

    def test_unparseable_path_skipped(self, tmp_path):
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
            _derive_per_lf_recording_date_from_paths,
        )

        paths = [tmp_path / "regular_lfs" / "lf_01.mp4"]
        skeleton_lfs = [SimpleNamespace(index=1)]
        result = _derive_per_lf_recording_date_from_paths(paths, skeleton_lfs)
        assert result == {}

    def test_empty_inputs(self):
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
            _derive_per_lf_recording_date_from_paths,
        )

        assert _derive_per_lf_recording_date_from_paths([], []) == {}


class TestFreshEncodePathStillPopulates:
    """Existing behaviour: lf_plans_by_date drives the map."""

    def test_full_encode_populates_per_lf_recording_date(self, tmp_path):
        state, _, stages_dir = _make_state(tmp_path, dates=("2026-02-20",))
        # No per_day_completed → the day goes through run_all_days.

        def _fake(**kwargs):
            day_dir = stages_dir / "day_2026-02-20"
            lf_paths = _write_finals(day_dir, lf_indices=(1,))
            return [
                SimpleNamespace(
                    date="2026-02-20",
                    master_path=stages_dir / "day_2026-02-20_master.mp4",
                    regular_lf_paths=lf_paths,
                    short_paths=[],
                )
            ]

        _run_phase(state, _fake)
        assert state["per_lf_recording_date"], (
            "Fresh-encode path must populate per_lf_recording_date (regression check)."
        )
        # All values are "YYYY-MM-DD" strings.
        for v in state["per_lf_recording_date"].values():
            assert isinstance(v, str) and len(v) == 10


class TestFinalsOnlyFastPathPopulates:
    """The fix: resume with finals on disk + master dropped MUST populate
    per_lf_recording_date — derived from the day_<DATE> directory."""

    def test_finals_only_fast_path_populates_from_disk(self, tmp_path):
        state, _, stages_dir = _make_state(tmp_path, dates=("2026-02-20", "2026-02-21"))

        # Pre-stage finals + per_day_completed entries with master_path that
        # no longer exists on disk → forces fast-path #2 (finals-only).
        for d, lf_idxs in (("2026-02-20", (1,)), ("2026-02-21", (2,))):
            _write_finals(stages_dir / f"day_{d}", lf_indices=lf_idxs)

        state["per_day_completed"] = {
            "2026-02-20": {"master_path": str(stages_dir / "missing-1.mp4"), "master_sha": ""},
            "2026-02-21": {"master_path": str(stages_dir / "missing-2.mp4"), "master_sha": ""},
        }

        # run_all_days must NOT be called for these days (fast-path absorbs them).
        def _fake(**kwargs):
            assert not (kwargs.get("day_batches") or []), (
                "Finals-only fast-path should absorb both days; run_all_days should not see them."
            )
            return []

        _run_phase(state, _fake)

        per_lf = state["per_lf_recording_date"]
        # Indices match the lf_NN.mp4 numbering we wrote on disk.
        assert per_lf, "Finals-only resume left per_lf_recording_date empty (the bug)."
        # Every LF index should be mapped to a real date string.
        for idx, date_str in per_lf.items():
            assert isinstance(idx, int)
            assert date_str in {"2026-02-20", "2026-02-21"}


class TestDateMatchesDayDirectory:
    def test_dates_match_day_dir_names(self, tmp_path):
        state, _, stages_dir = _make_state(tmp_path, dates=("2026-02-20", "2026-02-21"))

        # One LF + one short per day keeps shorts↔LF attribution
        # unambiguous (a single LF per day takes the source-window guard's
        # unambiguous branch), so the finals-only resume fast-path never
        # trips the >1-LF ambiguous-short hard-fail added by the
        # gate-bypass hardening. That hard-fail (shorts + >1 LF + no
        # short_ranges) is pinned separately by
        # tests/test_phase_bc_resume_short_ranges.py::
        # test_resume_without_sidecar_hard_fails_instead_of_equal_split — the
        # finals-only path can't reach the deterministic source-window
        # branch here without a full DrivePlan + day_batch, which the sibling
        # test_finals_only_fast_path_populates_from_disk also avoids.
        #
        # The lf_NN filenames are sequential in day-processing order (days
        # process ascending by date), so each maps 1:1 to its synthesized
        # skeleton-LF index. The point of THIS test is that each LF's
        # recording date is read from the day_<DATE> directory its file
        # lives under — never a sibling day's.
        _write_finals(stages_dir / "day_2026-02-20", lf_indices=(1,))
        _write_finals(stages_dir / "day_2026-02-21", lf_indices=(2,))

        state["per_day_completed"] = {
            "2026-02-20": {"master_path": str(stages_dir / "x.mp4"), "master_sha": ""},
            "2026-02-21": {"master_path": str(stages_dir / "y.mp4"), "master_sha": ""},
        }

        def _fake(**kwargs):
            return []

        _run_phase(state, _fake)
        per_lf = state["per_lf_recording_date"]

        # Every rendered LF's recording date must equal the date encoded in
        # the day_<DATE> directory its file lives under — never a sibling
        # day's. Covers BOTH days so a swapped or duplicated date is caught.
        for day_date in ("2026-02-20", "2026-02-21"):
            lf_dir = stages_dir / f"day_{day_date}" / "regular_lfs"
            materialised = list(lf_dir.glob("lf_*.mp4"))
            assert materialised, f"no LF finals materialised under day_{day_date}"
            for p in materialised:
                idx = int(p.stem.split("_")[1])
                assert per_lf.get(idx) == day_date, (
                    f"LF {idx} lives under day_{day_date} — its recording "
                    f"date must be {day_date}, got {per_lf.get(idx)!r}."
                )


class TestInvariantRaisesWhenUnmapped:
    def test_invariant_fires_when_lf_cannot_be_mapped(self, tmp_path, monkeypatch):
        state, _, stages_dir = _make_state(tmp_path, dates=("2026-02-20",))

        # Force an LF whose path doesn't match the day_<DATE> regex by
        # monkeypatching the path-derivation helper to return {} — and
        # ensure the partition path also yields nothing for that LF.
        from gomoveshift.pipeline.drive_phases import phase_bc_day_first as mod

        monkeypatch.setattr(
            mod,
            "_derive_per_lf_recording_date_from_paths",
            lambda _paths, _lfs: {},
        )
        monkeypatch.setattr(
            mod,
            "_build_per_lf_recording_date_map",
            lambda _lf_plans_by_date: {},
        )

        def _fake(**kwargs):
            day_dir = stages_dir / "day_2026-02-20"
            lf_paths = _write_finals(day_dir, lf_indices=(1,))
            return [
                SimpleNamespace(
                    date="2026-02-20",
                    master_path=stages_dir / "day_2026-02-20_master.mp4",
                    regular_lf_paths=lf_paths,
                    short_paths=[],
                )
            ]

        with pytest.raises(RuntimeError, match="per_lf_recording_date"):
            _run_phase(state, _fake)


class TestFilmedLineEmitted:
    """``LongFormMetadataGenerator`` must emit the standardized
    "📅 Filmed: <date>" line — not the legacy "📹 Recorded:" form."""

    def test_description_includes_filmed_line(self, tmp_path):
        from gomoveshift.pipeline.longform_metadata import (
            LongFormMetadataGenerator,
        )

        # ``LongFormMetadataGenerator``'s first positional arg is
        # ``seo_storage_path`` (a Path), not a Config — passing ``Config()``
        # was always wrong; it only "worked" because a broad except in
        # ``_init_ledger`` swallowed the resulting TypeError.
        gen = LongFormMetadataGenerator(tmp_path)
        meta = gen.generate(
            location="Beaufort County, SC",
            publish_day="Wednesday",
            video_duration=600.0,
            project_name="synthetic_lf01",
            recording_date="2026-02-20",
        )
        desc = meta["description"]
        # Long-form routes through the shared "Mon YYYY" formatter.
        assert "\U0001f4c5 Filmed: Feb 2026" in desc, (
            f"Description must include the canonical 📅 Filmed: <date> line; got: {desc[:400]!r}"
        )
        assert "\U0001f4f9 Recorded:" not in desc, (
            "Legacy 📹 Recorded: line must not be emitted alongside the new format."
        )

    def test_no_filmed_line_when_recording_date_none(self, tmp_path):
        from gomoveshift.pipeline.longform_metadata import (
            LongFormMetadataGenerator,
        )

        gen = LongFormMetadataGenerator(tmp_path)
        meta = gen.generate(
            location="Beaufort County, SC",
            publish_day="Wednesday",
            video_duration=600.0,
            project_name="synthetic_lf01",
        )
        desc = meta["description"]
        assert "Filmed:" not in desc
        assert "Recorded:" not in desc
