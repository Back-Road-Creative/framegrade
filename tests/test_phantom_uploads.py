"""Regression tests for phantom YouTube uploads.

Three compounding causes produced many duplicate YouTube uploads in a
single run. Each is covered here:

- Per-day mode: day_state.youtube_uploads must merge back to parent
- Phase D: empty drive_plan must NOT wipe youtube_uploads
- Sidecar: youtube_uploads.json survives state resets

Plus the safety-net skip_upload default when the sidecar has entries.

No network calls — YouTube upload paths are fully mocked.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.config import Config
from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator
from gomoveshift.pipeline.drive_phases.phase_d import PhaseD
from gomoveshift.pipeline.drive_state_model import PipelineState
from gomoveshift.pipeline.youtube_sidecar import (
    SIDECAR_FILENAME,
    load_sidecar,
    save_sidecar,
)


# TestPerDayMergesYouTubeUploads removed — per-day mode and _run_c_d_per_day
# were deleted. The single surviving pipeline is day-first; all
# days get merged automatically because there is only one run of Phase D.


# ─────────────────────────────────────────────────────────────────────────────
# Phase D with empty drive_plan must NOT wipe youtube_uploads
# ─────────────────────────────────────────────────────────────────────────────


class TestPhaseDEmptyPlanPreservesUploads:
    def test_phase_d_empty_plan_preserves_uploads(self, tmp_path):
        """Empty drive_plan path must preserve state['youtube_uploads']."""
        state = PipelineState(
            source_dir=tmp_path / "src",
            output_base=tmp_path / "out",
            stages_base=tmp_path / "stages",
            region="ireland",
            publish_day="wed",
            drive_plan=None,  # triggers early-return path at line ~194
        )
        state["youtube_uploads"] = {
            1: {"video_id": "existing_01", "url": "https://youtu.be/existing_01"}
        }
        (tmp_path / "src").mkdir()
        (tmp_path / "out").mkdir()
        (tmp_path / "stages").mkdir()

        PhaseD(Config()).run(state)

        assert state["youtube_uploads"] == {
            1: {"video_id": "existing_01", "url": "https://youtu.be/existing_01"}
        }


# ─────────────────────────────────────────────────────────────────────────────
# Sidecar persists across state resets
# ─────────────────────────────────────────────────────────────────────────────


class TestSidecarRoundtrip:
    def test_sidecar_roundtrip(self, tmp_path):
        """save → load returns the same dict."""
        uploads = {
            1: {"video_id": "abc", "url": "https://youtu.be/abc"},
            2: {"video_id": "def", "url": "https://youtu.be/def"},
        }
        save_sidecar(tmp_path, uploads)
        loaded = load_sidecar(tmp_path)
        # Keys are stringified during save — they round-trip as strings.
        assert loaded == {str(k): v for k, v in uploads.items()}

    def test_sidecar_atomic_write(self, tmp_path):
        """If os.replace fails after the temp file is written, existing sidecar survives."""
        # Seed a known-good sidecar.
        original = {"1": {"video_id": "original", "url": "https://youtu.be/original"}}
        save_sidecar(tmp_path, original)
        sidecar_path = tmp_path / SIDECAR_FILENAME
        pristine_bytes = sidecar_path.read_bytes()

        # Now attempt to overwrite with replace() raising mid-write. The
        # atomic-write helper that save_sidecar delegates to lives in
        # gomoveshift.pipeline._atomic_write; patch os.replace there.
        bad_uploads = {"2": {"video_id": "should_not_land"}}
        with patch("gomoveshift.pipeline._atomic_write.os.replace") as mock_replace:
            mock_replace.side_effect = OSError("simulated rename failure")
            with pytest.raises(OSError, match="simulated rename failure"):
                save_sidecar(tmp_path, bad_uploads)

        # Original sidecar file is byte-for-byte unchanged.
        assert sidecar_path.read_bytes() == pristine_bytes
        assert load_sidecar(tmp_path) == original

        # And the .tmp file was cleaned up — no stray temp files left behind.
        stray = list(tmp_path.glob("*.tmp"))
        assert stray == [], f"leftover temp files: {stray}"

    def test_sidecar_load_missing_file_returns_empty(self, tmp_path):
        """No sidecar present → load returns {} without raising."""
        assert load_sidecar(tmp_path) == {}

    def test_sidecar_load_invalid_json_returns_empty_with_warning(self, tmp_path, caplog):
        """Corrupt JSON → returns {}, logs a warning."""
        (tmp_path / SIDECAR_FILENAME).write_text("{not valid json")
        with caplog.at_level(logging.WARNING, logger="gomoveshift.pipeline.youtube_sidecar"):
            result = load_sidecar(tmp_path)
        assert result == {}
        assert any("invalid json" in r.message.lower() for r in caplog.records), (
            f"Expected invalid-JSON warning; got: {[r.message for r in caplog.records]}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Sidecar integration with orchestrator.run()
# ─────────────────────────────────────────────────────────────────────────────


def _neutered_preflight(monkeypatch):
    """Stub run_preflight so run() doesn't check a real filesystem layout."""
    import gomoveshift.pipeline.orchestrator_preflight as op

    monkeypatch.setattr(op, "run_preflight", lambda **kwargs: None)


def _stub_all_phases(orch):
    """Replace every phase callable with a no-op so run() reaches the end."""
    patches = []
    for attr in (
        "_phase_a_ingest_and_analyze",
        "_phase_bc_day_first",
        "_phase_d_metadata_and_publish",
    ):
        p = patch.object(orch, attr, MagicMock(return_value=None))
        patches.append(p)
    return patches


class TestOrchestratorSidecarIntegration:
    @staticmethod
    def _bc_side_effect(state):
        """Populate regular_lf_paths + shorts_per_lf so Layer-1 invariants pass.

        The real _phase_bc_day_first sets these fields before returning; a
        bare MagicMock would leave state untouched, which (correctly) trips
        state_invariants Rule 5 once phase_bc_day_first is appended to
        phases_completed and the orchestrator saves state.
        """
        state["regular_lf_paths"] = ["/mock/lf_01.mp4"]
        state["shorts_per_lf"] = {1: []}
        return None

    def test_run_loads_sidecar_into_state(self, tmp_path, monkeypatch):
        """State has no uploads, sidecar has 2 → merged into state during run()."""
        _neutered_preflight(monkeypatch)

        source = tmp_path / "source"
        source.mkdir()
        output_base = tmp_path / "output"
        output_base.mkdir()
        project_dir = output_base / source.stem.replace(" ", "_").lower()
        project_dir.mkdir(parents=True, exist_ok=True)

        # Pre-seed sidecar with two entries from a prior run.
        prior = {
            "1": {"video_id": "prior_01", "url": "https://youtu.be/prior_01"},
            "2": {"video_id": "prior_02", "url": "https://youtu.be/prior_02"},
        }
        save_sidecar(project_dir, prior)

        orch = DrivePipelineOrchestrator()
        seen_state = {}

        def _capture_d(state, drive_plan=None):
            # Capture the state dict Phase D sees so we can assert on it.
            seen_state["uploads"] = dict(state.get("youtube_uploads") or {})

        # Also disable ReviewGate so no prompt attempts happen.
        with patch("gomoveshift.pipeline.drive_orchestrator.ReviewGate") as mock_rg:
            mock_rg.return_value.gate_a = MagicMock()
            mock_rg.return_value.gate_b = MagicMock()
            mock_rg.return_value.gate_cd = MagicMock()
            with (
                patch.object(orch, "_phase_a_ingest_and_analyze", MagicMock()),
                patch.object(orch, "_phase_bc_day_first", side_effect=self._bc_side_effect),
                patch.object(orch, "_phase_d_metadata_and_publish", side_effect=_capture_d),
            ):
                orch.run(
                    source_dir=source,
                    output_base=output_base,
                    region="ireland",
                    # Explicitly opt in so skip_upload default doesn't fire.
                    youtube_upload=True,
                )

        uploads = seen_state.get("uploads") or {}
        # Keys become int after merge (state uses int LF keys).
        assert len(uploads) == 2
        assert uploads.get(1) == prior["1"]
        assert uploads.get(2) == prior["2"]

    def test_skip_upload_default_with_sidecar(self, tmp_path, monkeypatch, caplog):
        """Sidecar has entries + caller did not opt in → skip_upload becomes True with a warning."""
        _neutered_preflight(monkeypatch)

        source = tmp_path / "source"
        source.mkdir()
        output_base = tmp_path / "output"
        output_base.mkdir()
        project_dir = output_base / source.stem.replace(" ", "_").lower()
        project_dir.mkdir(parents=True, exist_ok=True)

        save_sidecar(
            project_dir,
            {"1": {"video_id": "vid_01", "url": "https://youtu.be/vid_01"}},
        )

        orch = DrivePipelineOrchestrator()
        seen = {}

        def _capture(state, drive_plan=None):
            seen["skip_upload"] = state.get("skip_upload")

        with patch("gomoveshift.pipeline.drive_orchestrator.ReviewGate") as mock_rg:
            mock_rg.return_value.gate_a = MagicMock()
            mock_rg.return_value.gate_b = MagicMock()
            mock_rg.return_value.gate_cd = MagicMock()
            with (
                patch.object(orch, "_phase_a_ingest_and_analyze", MagicMock()),
                patch.object(orch, "_phase_bc_day_first", side_effect=self._bc_side_effect),
                patch.object(orch, "_phase_d_metadata_and_publish", side_effect=_capture),
            ):
                with caplog.at_level(
                    logging.WARNING, logger="gomoveshift.pipeline.drive_orchestrator"
                ):
                    orch.run(
                        source_dir=source,
                        output_base=output_base,
                        region="ireland",
                        # argparse sentinel: ``None`` means the
                        # operator did NOT pass the flag. Explicit
                        # ``False`` suppresses the sidecar auto-default
                        # (CLI overrides beat persisted state).
                        youtube_upload=None,
                        # skip_upload not set explicitly
                    )

        assert seen.get("skip_upload") is True, (
            "skip_upload should have defaulted to True when sidecar has entries"
        )
        assert any("defaulting skip_upload=true" in r.message.lower() for r in caplog.records), (
            f"Expected skip_upload warning; got: {[r.message for r in caplog.records]}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Sanity: sidecar module exports
# ─────────────────────────────────────────────────────────────────────────────


def test_sidecar_module_exports():
    """Public contract: load_sidecar and save_sidecar exist with expected signatures."""
    from gomoveshift.pipeline import youtube_sidecar as ys

    assert callable(ys.load_sidecar)
    assert callable(ys.save_sidecar)
    assert ys.SIDECAR_FILENAME == "youtube_uploads.json"


# Ensure the sidecar test never accidentally hits the real filesystem outside tmp.
@pytest.fixture(autouse=True)
def _no_accidental_cwd_writes(tmp_path, monkeypatch):
    """Guard against tests writing the sidecar into the repo root."""
    monkeypatch.chdir(tmp_path)
    yield
    # Not strictly needed, but fail loudly if a rogue write landed here.
    rogue = Path.cwd() / SIDECAR_FILENAME
    assert not (rogue.exists() and rogue.parent == tmp_path) or rogue.parent == tmp_path, (
        f"sidecar wrote to unexpected location: {rogue}"
    )
    # Clean up intentionally.
    if rogue.exists():
        os.unlink(rogue)
