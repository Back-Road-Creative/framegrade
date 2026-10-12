"""Phase D resume invalidates when required outputs are missing.

Defense in depth on top of the fingerprint check. After the
fingerprint check passes (Phase D would be skipped), verify required
Phase D outputs actually exist on disk. If any are missing, drop
``phase_d`` from ``state["phases_completed"]`` and re-run with a
WARNING log naming the missing files. Closes the failure mode where
state says ``phase_d done`` but the output files were deleted,
corrupted, or never written due to a partial failure.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from gomoveshift.config import Config
from gomoveshift.pipeline._phase_d_outputs import (
    expected_phase_d_outputs,
    find_missing_phase_d_outputs,
)
from gomoveshift.pipeline.drive_state_model import PipelineState
from gomoveshift.pipeline.orchestrator_fingerprint import (
    invalidate_phase_d_if_outputs_missing,
)


ORCH_LOGGER = "gomoveshift.pipeline.drive_orchestrator"


def _make_state_with_outputs(
    *,
    output_base: Path,
    project_name: str = "demo",
    lf_indices: list[int] | None = None,
    shorts: bool = True,
) -> PipelineState:
    """Construct state pointing at ``output_base`` with the given LFs.

    ``regular_lf_paths`` is populated with synthesised ``lf_NN.mp4``
    paths whose stems encode the LF index (the only canonical way to
    enumerate Phase D outputs without a live ``DrivePlan``).
    """
    if lf_indices is None:
        lf_indices = [1]
    output_base.mkdir(parents=True, exist_ok=True)
    source = output_base.parent / project_name
    source.mkdir(parents=True, exist_ok=True)
    state = PipelineState(
        source_dir=source,
        output_base=output_base,
        publish_day="wed",
        location="Ireland",
    )
    state["regular_lf_paths"] = [
        str(output_base / f"day_2025-09-14/regular_lfs/lf_{idx:02d}.mp4") for idx in lf_indices
    ]
    if shorts:
        # Non-empty mapping → Phase D would have written the shorts CSV.
        state["shorts_per_lf"] = {idx: [f"short_{idx:02d}.mp4"] for idx in lf_indices}
    else:
        state["shorts_per_lf"] = {}
    return state


def _write_all_outputs(
    output_base: Path,
    project_name: str,
    lf_indices: list[int],
    *,
    include_shorts_csv: bool = True,
    include_thumbs: bool = True,
    include_youtube_jsons: bool = True,
) -> None:
    """Materialise the canonical Phase D outputs on disk."""
    if include_youtube_jsons:
        for idx in lf_indices:
            (output_base / f"{project_name}_lf{idx:02d}_youtube.json").write_text("{}")
    if include_shorts_csv:
        (output_base / f"{project_name}_shorts_metricool.csv").write_text("a,b,c\n")
    if include_thumbs:
        thumbs_dir = output_base / "thumbnails"
        thumbs_dir.mkdir(parents=True, exist_ok=True)
        for idx in lf_indices:
            (thumbs_dir / f"lf_{idx:02d}.jpg").write_bytes(b"\xff\xd8")


# ────────────────────────────────────────────────────────────────────────
# Pure helper coverage
# ────────────────────────────────────────────────────────────────────────


class TestExpectedPhaseDOutputsHelpers:
    """Pin the canonical expected-outputs derivation."""

    def test_expected_outputs_lists_youtube_json_thumb_and_csv(self, tmp_path):
        out = tmp_path / "demo"
        state = _make_state_with_outputs(output_base=out, project_name="demo", lf_indices=[1, 2])
        expected = expected_phase_d_outputs(state, out, "demo")
        names = {p.name for p in expected}
        assert "demo_lf01_youtube.json" in names
        assert "demo_lf02_youtube.json" in names
        assert "demo_shorts_metricool.csv" in names
        assert "lf_01.jpg" in names
        assert "lf_02.jpg" in names

    def test_no_shorts_omits_metricool_csv(self, tmp_path):
        out = tmp_path / "demo"
        state = _make_state_with_outputs(
            output_base=out, project_name="demo", lf_indices=[1], shorts=False
        )
        expected = expected_phase_d_outputs(state, out, "demo")
        names = {p.name for p in expected}
        assert "demo_shorts_metricool.csv" not in names
        assert "demo_lf01_youtube.json" in names
        assert "lf_01.jpg" in names

    def test_default_routes_expect_no_sidecar(self, tmp_path):
        """All four ``publishing_route_*`` at the built-in ``"off"`` default
        route nothing, so a default ``Config()`` expects exactly what
        ``without_config`` does -- no publish-plan sidecar. A platform is
        opted in by the operator's config (next test)."""
        out = tmp_path / "demo"
        state = _make_state_with_outputs(output_base=out, project_name="demo", lf_indices=[1, 2])
        config = Config()
        expected = expected_phase_d_outputs(state, out, "demo", config=config)
        without_config = expected_phase_d_outputs(state, out, "demo")
        assert expected == without_config
        assert out / "demo_publish_plan.json" not in expected

    @pytest.mark.parametrize(
        "route_field",
        [
            "publishing_route_youtube",
            "publishing_route_instagram",
            "publishing_route_facebook",
            "publishing_route_tiktok",
        ],
    )
    def test_any_first_party_route_expects_publish_plan_sidecar(self, tmp_path, route_field):
        out = tmp_path / "demo"
        state = _make_state_with_outputs(output_base=out, project_name="demo", lf_indices=[1])
        config = Config()
        setattr(config.drive, route_field, "first_party")
        expected = expected_phase_d_outputs(state, out, "demo", config=config)
        names = {p.name for p in expected}
        assert "demo_publish_plan.json" in names
        assert out / "demo_publish_plan.json" in expected
        # CSV still written unconditionally regardless of routes.
        assert "demo_shorts_metricool.csv" in names


# ────────────────────────────────────────────────────────────────────────
# Orchestrator method behaviour
# ────────────────────────────────────────────────────────────────────────


class TestInvalidatePhaseDIfOutputsMissing:
    def _resumed(self) -> tuple[list[str], list[str]]:
        rs = ["phase_a", "phase_bc_day_first", "phase_d"]
        st = ["phase_a", "phase_bc_day_first", "phase_d"]
        return rs, st

    def test_missing_youtube_metadata_triggers_rerun(self, tmp_path, caplog):
        out = tmp_path / "demo"
        state = _make_state_with_outputs(output_base=out, project_name="demo")
        # Materialise everything EXCEPT the LF metadata JSON.
        _write_all_outputs(
            out,
            "demo",
            [1],
            include_shorts_csv=True,
            include_thumbs=True,
            include_youtube_jsons=False,
        )
        state["phases_completed"] = ["phase_a", "phase_bc_day_first", "phase_d"]
        resumed, stages = self._resumed()
        with caplog.at_level(logging.WARNING, logger=ORCH_LOGGER):
            result = invalidate_phase_d_if_outputs_missing(state, resumed, stages)
        assert "phase_d" not in result
        assert "phase_d" not in stages
        assert "phase_d" not in state["phases_completed"]
        joined = "\n".join(r.getMessage() for r in caplog.records)
        assert "demo_lf01_youtube.json" in joined, joined
        assert any(r.levelno == logging.WARNING for r in caplog.records), (
            "missing-output invalidation must log at WARNING"
        )

    def test_missing_metricool_csv_triggers_rerun(self, tmp_path, caplog):
        out = tmp_path / "demo"
        state = _make_state_with_outputs(output_base=out, project_name="demo")
        _write_all_outputs(
            out,
            "demo",
            [1],
            include_shorts_csv=False,
            include_thumbs=True,
            include_youtube_jsons=True,
        )
        state["phases_completed"] = ["phase_a", "phase_bc_day_first", "phase_d"]
        resumed, stages = self._resumed()
        with caplog.at_level(logging.WARNING, logger=ORCH_LOGGER):
            result = invalidate_phase_d_if_outputs_missing(state, resumed, stages)
        assert "phase_d" not in result
        joined = "\n".join(r.getMessage() for r in caplog.records)
        assert "demo_shorts_metricool.csv" in joined, joined

    def test_missing_thumbnail_triggers_rerun(self, tmp_path, caplog):
        out = tmp_path / "demo"
        state = _make_state_with_outputs(output_base=out, project_name="demo")
        _write_all_outputs(
            out,
            "demo",
            [1],
            include_shorts_csv=True,
            include_thumbs=False,
            include_youtube_jsons=True,
        )
        state["phases_completed"] = ["phase_a", "phase_bc_day_first", "phase_d"]
        resumed, stages = self._resumed()
        with caplog.at_level(logging.WARNING, logger=ORCH_LOGGER):
            result = invalidate_phase_d_if_outputs_missing(state, resumed, stages)
        assert "phase_d" not in result
        joined = "\n".join(r.getMessage() for r in caplog.records)
        assert "lf_01.jpg" in joined, joined

    def test_all_outputs_present_skips_phase_d(self, tmp_path):
        out = tmp_path / "demo"
        state = _make_state_with_outputs(output_base=out, project_name="demo", lf_indices=[1, 2])
        _write_all_outputs(out, "demo", [1, 2])
        state["phases_completed"] = ["phase_a", "phase_bc_day_first", "phase_d"]
        resumed, stages = self._resumed()
        result = invalidate_phase_d_if_outputs_missing(state, resumed, stages)
        assert "phase_d" in result
        assert "phase_d" in stages
        assert "phase_d" in state["phases_completed"]

    def test_missing_output_log_lists_all_missing_files(self, tmp_path, caplog):
        out = tmp_path / "demo"
        state = _make_state_with_outputs(output_base=out, project_name="demo", lf_indices=[1, 2])
        # LF 2 metadata + LF 1 thumbnail both missing.
        _write_all_outputs(
            out,
            "demo",
            [1, 2],
            include_shorts_csv=True,
            include_thumbs=True,
            include_youtube_jsons=True,
        )
        (out / "demo_lf02_youtube.json").unlink()
        (out / "thumbnails" / "lf_01.jpg").unlink()
        state["phases_completed"] = ["phase_a", "phase_bc_day_first", "phase_d"]
        resumed, stages = self._resumed()
        with caplog.at_level(logging.WARNING, logger=ORCH_LOGGER):
            result = invalidate_phase_d_if_outputs_missing(state, resumed, stages)
        assert "phase_d" not in result
        joined = "\n".join(r.getMessage() for r in caplog.records)
        # Both missing files must appear in the log.
        assert "demo_lf02_youtube.json" in joined, joined
        assert "lf_01.jpg" in joined, joined

    def test_no_shorts_planned_does_not_check_metricool_csv(self, tmp_path, caplog):
        out = tmp_path / "demo"
        state = _make_state_with_outputs(
            output_base=out, project_name="demo", lf_indices=[1], shorts=False
        )
        # Only youtube.json + thumbnail exist; CSV deliberately absent.
        _write_all_outputs(
            out,
            "demo",
            [1],
            include_shorts_csv=False,
            include_thumbs=True,
            include_youtube_jsons=True,
        )
        state["phases_completed"] = ["phase_a", "phase_bc_day_first", "phase_d"]
        resumed, stages = self._resumed()
        result = invalidate_phase_d_if_outputs_missing(state, resumed, stages)
        # Missing CSV must NOT trigger invalidation when no shorts planned.
        assert "phase_d" in result
        assert "phase_d" in state["phases_completed"]

    def test_check_uses_state_lf_indices_not_filesystem_glob(self, tmp_path):
        """Helper enumerates expected outputs from state, not by globbing
        the output dir. A corrupted dir with no LF artefacts on disk MUST
        still surface the state-recorded LFs as missing — otherwise a
        wiped output dir silently bypasses the check entirely.
        """
        out = tmp_path / "demo"
        state = _make_state_with_outputs(output_base=out, project_name="demo", lf_indices=[1, 2, 3])
        # Nothing on disk at all (output dir exists but empty).
        out.mkdir(parents=True, exist_ok=True)
        missing = find_missing_phase_d_outputs(state, out, "demo")
        names = {p.name for p in missing}
        # All 3 LFs' youtube.json + 3 thumbnails + the shorts CSV: 7 missing.
        assert "demo_lf01_youtube.json" in names
        assert "demo_lf02_youtube.json" in names
        assert "demo_lf03_youtube.json" in names
        assert "lf_01.jpg" in names
        assert "lf_02.jpg" in names
        assert "lf_03.jpg" in names
        assert "demo_shorts_metricool.csv" in names
        assert len(missing) == 7
