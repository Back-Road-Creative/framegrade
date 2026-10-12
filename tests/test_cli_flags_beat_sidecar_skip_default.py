"""CLI upload flags beat the sidecar auto-skip default.

When ``youtube_uploads.json`` sidecar has entries, the orchestrator auto-sets
``state["skip_upload"]=True`` as a phantom-upload safety net. This safety net
must NOT fire when the operator has explicitly passed any of:
  * ``--youtube-upload``
  * ``--skip-youtube-upload``
  * ``--skip-drive-upload``
  * ``--skip-upload``

Per-invocation CLI
flags must beat persisted/derived state on resume.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator
from gomoveshift.pipeline.youtube_sidecar import save_sidecar


def _neutered_preflight(monkeypatch):
    import gomoveshift.pipeline.orchestrator_preflight as op

    monkeypatch.setattr(op, "run_preflight", lambda **kwargs: None)


def _bc_side_effect(state):
    state["regular_lf_paths"] = ["/mock/lf_01.mp4"]
    state["shorts_per_lf"] = {1: []}
    return None


def _run_with_sidecar(tmp_path, monkeypatch, sidecar_entries, **run_kwargs):
    """Drive orchestrator.run with a pre-seeded sidecar; capture state Phase D sees."""
    _neutered_preflight(monkeypatch)
    source = tmp_path / "source"
    source.mkdir()
    output_base = tmp_path / "output"
    output_base.mkdir()
    project_dir = output_base / source.stem.replace(" ", "_").lower()
    project_dir.mkdir(parents=True, exist_ok=True)
    if sidecar_entries:
        save_sidecar(project_dir, sidecar_entries)

    orch = DrivePipelineOrchestrator()
    seen = {}

    def _capture(state, drive_plan=None):
        seen["skip_upload"] = state.get("skip_upload")
        seen["skip_drive_upload"] = state.get("skip_drive_upload")
        seen["skip_youtube_upload"] = state.get("skip_youtube_upload")
        seen["youtube_upload"] = state.get("youtube_upload")

    with patch("gomoveshift.pipeline.drive_orchestrator.ReviewGate") as mock_rg:
        mock_rg.return_value.gate_a = MagicMock()
        mock_rg.return_value.gate_b = MagicMock()
        mock_rg.return_value.gate_cd = MagicMock()
        with (
            patch.object(orch, "_phase_a_ingest_and_analyze", MagicMock()),
            patch.object(orch, "_phase_bc_day_first", side_effect=_bc_side_effect),
            patch.object(orch, "_phase_d_metadata_and_publish", side_effect=_capture),
        ):
            orch.run(
                source_dir=source,
                output_base=output_base,
                region="ireland",
                **run_kwargs,
            )
    return seen


class TestCliFlagsBeatSidecarSkipDefault:
    def test_sidecar_present_no_cli_flags_auto_skip_fires(self, tmp_path, monkeypatch, caplog):
        """Baseline: sidecar entries + zero upload CLI flags → auto-skip preserved."""
        with caplog.at_level(logging.WARNING, logger="gomoveshift.pipeline.drive_orchestrator"):
            seen = _run_with_sidecar(
                tmp_path,
                monkeypatch,
                {"1": {"video_id": "v1", "url": "https://youtu.be/v1"}},
                # NO upload-related flags supplied — all four default.
            )
        assert seen["skip_upload"] is True
        # The safety net keeps the granular flags coherent with the
        # coarse alias so any reader of raw state (not just phase_d's
        # ``or skip_upload`` re-derivation) sees the skip.
        assert seen["skip_drive_upload"] is True
        assert seen["skip_youtube_upload"] is True
        assert any("defaulting skip_upload=true" in r.message.lower() for r in caplog.records), (
            f"Expected auto-skip warning; got: {[r.message for r in caplog.records]}"
        )

    def test_sidecar_present_skip_youtube_upload_explicit_no_auto_skip(
        self, tmp_path, monkeypatch, caplog
    ):
        """Operator passes --skip-youtube-upload only → auto-default MUST NOT fire.

        Expected resulting state:
          skip_youtube_upload=True (operator intent)
          skip_drive_upload=False  (Drive shorts upload still runs)
          skip_upload NOT auto-set to True
        """
        with caplog.at_level(logging.WARNING, logger="gomoveshift.pipeline.drive_orchestrator"):
            seen = _run_with_sidecar(
                tmp_path,
                monkeypatch,
                {
                    "1": {"video_id": "v1", "url": "https://youtu.be/v1"},
                    "2": {"video_id": "v2", "url": "https://youtu.be/v2"},
                },
                skip_youtube_upload=True,  # explicit CLI
            )
        assert seen["skip_youtube_upload"] is True
        assert seen["skip_drive_upload"] is False
        assert seen["skip_upload"] is not True, (
            "Auto-default fired despite explicit --skip-youtube-upload "
            f"(state['skip_upload']={seen['skip_upload']!r})"
        )
        assert not any(
            "defaulting skip_upload=true" in r.message.lower() for r in caplog.records
        ), "Auto-skip warning emitted despite explicit CLI flag"

    def test_sidecar_present_youtube_upload_explicit_no_auto_skip(
        self, tmp_path, monkeypatch, caplog
    ):
        """Operator passes --youtube-upload → uploads Drive + YouTube; no auto-skip."""
        with caplog.at_level(logging.WARNING, logger="gomoveshift.pipeline.drive_orchestrator"):
            seen = _run_with_sidecar(
                tmp_path,
                monkeypatch,
                {"1": {"video_id": "v1", "url": "https://youtu.be/v1"}},
                youtube_upload=True,  # explicit CLI opt-in
            )
        assert seen["youtube_upload"] is True
        assert seen["skip_upload"] is not True
        assert seen["skip_drive_upload"] is False
        assert seen["skip_youtube_upload"] is False
        assert not any("defaulting skip_upload=true" in r.message.lower() for r in caplog.records)

    def test_sidecar_present_skip_drive_upload_explicit_no_auto_skip(
        self, tmp_path, monkeypatch, caplog
    ):
        """Operator passes --skip-drive-upload only → auto-default MUST NOT fire."""
        with caplog.at_level(logging.WARNING, logger="gomoveshift.pipeline.drive_orchestrator"):
            seen = _run_with_sidecar(
                tmp_path,
                monkeypatch,
                {"1": {"video_id": "v1", "url": "https://youtu.be/v1"}},
                skip_drive_upload=True,
            )
        assert seen["skip_drive_upload"] is True
        assert seen["skip_youtube_upload"] is False
        assert seen["skip_upload"] is not True
        assert not any("defaulting skip_upload=true" in r.message.lower() for r in caplog.records)

    def test_sidecar_present_skip_upload_explicit_preserves_intent(
        self, tmp_path, monkeypatch, caplog
    ):
        """--skip-upload is the back-compat alias; explicit True must still set state."""
        with caplog.at_level(logging.WARNING, logger="gomoveshift.pipeline.drive_orchestrator"):
            seen = _run_with_sidecar(
                tmp_path,
                monkeypatch,
                {"1": {"video_id": "v1", "url": "https://youtu.be/v1"}},
                skip_upload=True,
            )
        # The PipelineState seed is skip_upload=True (operator intent), and
        # auto-default does NOT need to fire because the CLI was explicit.
        assert seen["skip_upload"] is True
        assert seen["skip_drive_upload"] is True
        assert seen["skip_youtube_upload"] is True
        assert not any(
            "defaulting skip_upload=true" in r.message.lower() for r in caplog.records
        ), "Auto-skip warning emitted even though operator was explicit"

    def test_sidecar_empty_skip_youtube_upload_existing_behavior(self, tmp_path, monkeypatch):
        """No sidecar + --skip-youtube-upload → skip_youtube_upload set, no auto-default."""
        seen = _run_with_sidecar(
            tmp_path,
            monkeypatch,
            None,  # no sidecar
            skip_youtube_upload=True,
        )
        assert seen["skip_youtube_upload"] is True
        assert seen["skip_drive_upload"] is False
        assert seen["skip_upload"] is not True
