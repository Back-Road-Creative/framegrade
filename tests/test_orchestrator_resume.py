"""Regression tests for CLI-per-invocation-flags-vs-persisted-state on resume.

Pins CLI-over-state handling for per-invocation flags on resume.

Concrete bite these tests guard against:
  A prior audit run persisted ``skip_upload=True`` into
  ``pipeline_state.json``. The next invocation passed ``--youtube-upload``
  (which implies ``skip_upload=False``). On resume, ``state.update(saved)``
  silently overwrote the CLI's ``skip_upload=False`` with the persisted
  ``True``, and both ``D-upload-shorts`` and ``D-upload-lfs`` in
  ``phase_d`` skipped because ``state.get("skip_upload")`` returned
  ``True``. No warning fired; no uploads happened; the operator only
  discovered it by inspecting YouTube Studio after the run.

Priority rule codified by these tests:
  * CLI > state for **per-invocation intent** flags: ``skip_upload``,
    ``youtube_upload``, ``publish_day``, ``start_date``, ``shorts_per_day``
    (orchestration knobs the user sets fresh each run).
  * state > CLI for **persistent config**: ingest artifacts, phase
    completion, ``shorts_target`` / ``shorts_gap`` (resume-authoritative —
    see ``test_silent_degradation_upload.py``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest


ORCH_LOGGER = "gomoveshift.pipeline.drive_orchestrator"


def _write_saved_state(state_file: Path, saved: dict) -> None:
    """Write a JSON-backed saved state for ``_try_resume`` to load."""
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(saved))


# ─────────────────────────────────────────────────── skip_upload CLI > state


class TestSkipUploadCliBeatsState:
    """``skip_upload=False`` (implied by ``--youtube-upload``) must win
    over a persisted ``skip_upload=True`` from a prior audit run."""

    def test_cli_skip_upload_false_beats_state_true(self, tmp_path):
        """Concrete bite: audit left ``skip_upload=True`` persisted.
        Live run with ``--youtube-upload`` must propagate ``skip_upload=False``
        through resume so ``phase_d`` actually uploads.
        """
        from gomoveshift.config import Config
        from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator
        from gomoveshift.pipeline.drive_state_model import PipelineState

        orch = DrivePipelineOrchestrator(config=Config())

        state_file = tmp_path / "pipeline_state.json"
        # Audit-run leftover: skip_upload persisted as True.
        _write_saved_state(
            state_file,
            {
                "phases_completed": ["phase_b_preview"],
                "skip_upload": True,
                "youtube_upload": False,
            },
        )

        # Live invocation: CLI passed --youtube-upload (and no --skip-upload),
        # which yields skip_upload=False, youtube_upload=True on state.
        state = PipelineState(
            source_dir=Path("/tmp/src"),
            output_base=tmp_path,
            skip_upload=False,
        )
        state["youtube_upload"] = True

        stages_completed: list[str] = []
        orch._try_resume(state, state_file, tmp_path, stages_completed, fresh=False)

        assert state.get("skip_upload") is False, (
            "CLI skip_upload=False must win over persisted skip_upload=True. "
            "If state.get('skip_upload')==True here, phase_d's "
            "skip_upload guard silently falls through and no upload happens."
        )
        # phase_d reads via state.get(...) — assert both access paths agree.
        assert state["skip_upload"] is False
        # Also confirm youtube_upload was preserved from CLI (existing behavior).
        assert state.get("youtube_upload") is True


class TestYoutubeUploadCliBeatsState:
    """``youtube_upload=True`` (CLI ``--youtube-upload``) must win over a
    persisted ``youtube_upload=False`` from a prior audit run."""

    def test_cli_youtube_upload_true_beats_state_false(self, tmp_path):
        """Mirror of ``test_cli_skip_upload_false_beats_state_true`` for
        the youtube_upload flag — defense in depth on both sides of the
        same conditional."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator
        from gomoveshift.pipeline.drive_state_model import PipelineState

        orch = DrivePipelineOrchestrator(config=Config())

        state_file = tmp_path / "pipeline_state.json"
        _write_saved_state(
            state_file,
            {
                "phases_completed": ["phase_b_preview"],
                "skip_upload": False,
                "youtube_upload": False,
            },
        )

        state = PipelineState(
            source_dir=Path("/tmp/src"),
            output_base=tmp_path,
            skip_upload=False,
        )
        state["youtube_upload"] = True  # CLI --youtube-upload

        stages_completed: list[str] = []
        orch._try_resume(state, state_file, tmp_path, stages_completed, fresh=False)

        assert state.get("youtube_upload") is True, (
            "CLI youtube_upload=True must win over persisted False."
        )
        assert state["youtube_upload"] is True


class TestCliFlagUnsetPreservesStateValue:
    """When the CLI does NOT pass a flag, the persisted state value must
    survive resume (backward-compat — do not accidentally clobber state
    with a CLI default).

    For bool fields with a ``False`` default (``skip_upload``,
    ``youtube_upload``) there is no distinguishable "unset" — every CLI
    invocation produces a definitive True/False, and CLI always wins.
    These tests cover the fields with ``None`` defaults (``start_date``,
    ``shorts_per_day``) where the filter ``val is not None`` correctly
    lets state survive.
    """

    def test_cli_start_date_unset_preserves_state_value(self, tmp_path):
        """``start_date`` default None: state wins when CLI omits the flag."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator
        from gomoveshift.pipeline.drive_state_model import PipelineState

        orch = DrivePipelineOrchestrator(config=Config())

        state_file = tmp_path / "pipeline_state.json"
        _write_saved_state(
            state_file,
            {
                "phases_completed": ["phase_b_preview"],
                "start_date": "2026-04-29",
            },
        )

        # CLI did NOT pass --start-date — state field stays at None.
        state = PipelineState(source_dir=Path("/tmp/src"), output_base=tmp_path)
        assert state.get("start_date") is None

        stages_completed: list[str] = []
        orch._try_resume(state, state_file, tmp_path, stages_completed, fresh=False)

        assert state.get("start_date") == "2026-04-29", (
            "When CLI omits --start-date, persisted state value must survive. "
            "Regression would indicate the CLI-override filter is clobbering "
            "state with CLI default None."
        )

    def test_cli_shorts_per_day_unset_preserves_state_value(self, tmp_path):
        """``shorts_per_day`` default None: state wins when CLI omits the flag."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator
        from gomoveshift.pipeline.drive_state_model import PipelineState

        orch = DrivePipelineOrchestrator(config=Config())

        state_file = tmp_path / "pipeline_state.json"
        _write_saved_state(
            state_file,
            {
                "phases_completed": ["phase_b_preview"],
                "shorts_per_day": 14,
            },
        )

        state = PipelineState(source_dir=Path("/tmp/src"), output_base=tmp_path)
        assert state.get("shorts_per_day") is None

        stages_completed: list[str] = []
        orch._try_resume(state, state_file, tmp_path, stages_completed, fresh=False)

        assert state.get("shorts_per_day") == 14


class TestSkipUploadCliOverrideReadableViaDictProtocol:
    """Pin that the fix restoration propagates through EVERY read path
    ``phase_d`` uses — ``state.get("skip_upload")`` and
    ``state["skip_upload"]`` (dict-protocol) must both see the CLI value.

    This also verifies that ``object.__setattr__`` on a dataclass field
    correctly propagates to ``PipelineState.__getitem__`` / ``.get()``
    — the concern that motivated these tests.
    """

    def test_cli_override_propagates_to_state_get_and_getitem(self, tmp_path):
        from gomoveshift.config import Config
        from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator
        from gomoveshift.pipeline.drive_state_model import PipelineState

        orch = DrivePipelineOrchestrator(config=Config())

        state_file = tmp_path / "pipeline_state.json"
        _write_saved_state(
            state_file,
            {
                "phases_completed": ["phase_b_preview"],
                "skip_upload": True,
                "youtube_upload": False,
            },
        )

        state = PipelineState(
            source_dir=Path("/tmp/src"),
            output_base=tmp_path,
            skip_upload=False,
        )
        state["youtube_upload"] = True

        stages_completed: list[str] = []
        orch._try_resume(state, state_file, tmp_path, stages_completed, fresh=False)

        # All three access paths must agree after restoration.
        assert state.skip_upload is False
        assert state.get("skip_upload") is False
        assert state["skip_upload"] is False

        assert state.youtube_upload is True
        assert state.get("youtube_upload") is True
        assert state["youtube_upload"] is True


# ─────────────────────────────────────────────── combined scenario (bite)


class TestConcreteBiteScenario:
    """Reproduces the resume failure in one test:
    audit run persisted ``skip_upload=True, youtube_upload=False``;
    live run with ``--youtube-upload`` must upload."""

    def test_audit_then_live_upload_scenario(self, tmp_path, caplog):
        from gomoveshift.config import Config
        from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator
        from gomoveshift.pipeline.drive_state_model import PipelineState

        orch = DrivePipelineOrchestrator(config=Config())

        state_file = tmp_path / "pipeline_state.json"
        # Exactly what the audit run wrote.
        _write_saved_state(
            state_file,
            {
                "phases_completed": [
                    "phase_a",
                    "phase_b_preview",
                    "phase_bc_day_first",
                ],
                "skip_upload": True,
                "youtube_upload": False,
                "run_start": "2026-04-20T10:00:00+00:00",
            },
        )

        # Next-day live invocation: --youtube-upload (no --skip-upload).
        state = PipelineState(
            source_dir=Path("/tmp/src"),
            output_base=tmp_path,
            skip_upload=False,
        )
        state["youtube_upload"] = True

        stages_completed: list[str] = []
        with caplog.at_level(logging.INFO, logger=ORCH_LOGGER):
            orch._try_resume(state, state_file, tmp_path, stages_completed, fresh=False)

        # Mirror the exact reads of the skip_upload guard in phase_d.
        skip_upload = state.get("skip_upload", False)
        youtube_upload_enabled = state.get("youtube_upload", False)

        # This is the skip_upload guard in phase_d for the long-form upload.
        assert not skip_upload and youtube_upload_enabled, (
            "Live upload gate must open: "
            f"skip_upload={skip_upload} youtube_upload={youtube_upload_enabled}. "
            "Bite-regression: a stale audit state with skip_upload=True "
            "would keep the gate shut and silently skip both "
            "the shorts upload and the long-form upload."
        )


# ─────────────────────────────────────────────────────────────────────────
# CLI override handling bugs.
#
# Two latent bugs in the former ``schedule_overrides`` merge loop of drive_orchestrator.py
# (now ``flag_registry.merge_on_resume``):
#
#   (a) ``object.__setattr__(state, key, val)`` silently vanishes for keys
#       not declared as ``PipelineState`` dataclass fields. When the fix
#       applies, ``allow_seo_degraded`` and ``force_sidecar_reset`` must
#       survive a ``to_dict()`` round-trip so the end-of-Phase-D quality
#       gate and the Phase D sidecar gate see the CLI-requested value on
#       resume.
#   (b) Boolean CLI flags captured as overrides unconditionally: the
#       current filter ``val = state.get(key); if val is not None``
#       captures ``False`` as an explicit override, which then overwrites
#       a saved ``True`` from a prior run. Mirror image of the earlier
#       bug (skip_upload=True from an audit run overwriting live). The fix is
#       the argparse-sentinel pattern: the CLI parser's ``default=None``
#       tells downstream code "not supplied". Only explicitly-supplied
#       values populate ``schedule_overrides``.
# ─────────────────────────────────────────────────────────────────────────


class TestDeclaredOverrideFieldSerializationContracts:
    """Bug (a): ``allow_seo_degraded`` and ``force_sidecar_reset`` must be
    declared PipelineState fields (``object.__setattr__`` alone silently
    vanishes at serialization). Their serialization CONTRACTS differ,
    though: ``allow_seo_degraded`` is a sticky override that must survive a
    ``to_dict()``/``from_dict()`` round trip. ``force_sidecar_reset`` is a
    one-shot operator recovery flag that must NOT — a saved ``True``
    re-wipes both upload sidecars and re-uploads every published LF on
    every later resume (see ``_UNSERIALIZED_FIELDS`` in
    ``drive_state_model.py``)."""

    def test_allow_seo_degraded_is_declared_field(self):
        import dataclasses

        from gomoveshift.pipeline.drive_state_model import PipelineState

        names = {f.name for f in dataclasses.fields(PipelineState)}
        assert "allow_seo_degraded" in names, (
            "allow_seo_degraded must be a declared PipelineState field. "
            "Without the declaration, object.__setattr__ lets reads work via "
            "hasattr() but to_dict() never serializes the value — the override "
            "silently vanishes on resume (declared-field requirement)."
        )

    def test_force_sidecar_reset_is_declared_field(self):
        import dataclasses

        from gomoveshift.pipeline.drive_state_model import PipelineState

        names = {f.name for f in dataclasses.fields(PipelineState)}
        assert "force_sidecar_reset" in names, (
            "force_sidecar_reset must be a declared PipelineState field. "
            "Same rationale as allow_seo_degraded (declared-field requirement)."
        )

    def test_allow_seo_degraded_survives_to_dict(self):
        """Set allow_seo_degraded=True on a state, serialize, and confirm
        it appears in the to_dict() output. Prior to the fix, the key
        was set via object.__setattr__ but dataclasses.fields() did not
        know about it, so to_dict() dropped it on the floor."""
        from gomoveshift.pipeline.drive_state_model import PipelineState

        state = PipelineState(
            source_dir=Path("/tmp/src"),
            output_base=Path("/tmp/out"),
        )
        state["allow_seo_degraded"] = True
        d = state.to_dict()
        assert d.get("allow_seo_degraded") is True, (
            "allow_seo_degraded did not survive to_dict. This is the concrete "
            "bite of the declared-field requirement: the override set via object.__setattr__ "
            "works for hasattr()/get() reads but is silently dropped at "
            "serialization because dataclasses.fields() doesn't include it."
        )

    def test_force_sidecar_reset_does_not_survive_to_dict(self):
        """A one-shot operator recovery flag must not be written into the
        checkpoint at all — persisting it re-wipes both upload sidecars
        (youtube_uploads.json / superlong_uploads.json) and re-uploads
        every already-published LF as a new public video on every later
        resume that loads this checkpoint."""
        from gomoveshift.pipeline.drive_state_model import PipelineState

        state = PipelineState(
            source_dir=Path("/tmp/src"),
            output_base=Path("/tmp/out"),
        )
        state["force_sidecar_reset"] = True
        d = state.to_dict()
        assert d.get("force_sidecar_reset") is None

    def test_allow_seo_degraded_survives_force_sidecar_reset_does_not(self):
        """Full serialize -> deserialize round trip: allow_seo_degraded (a
        sticky override) must be preserved; force_sidecar_reset (a one-shot
        recovery flag) must come back unset."""
        from gomoveshift.pipeline.drive_state_model import PipelineState

        state = PipelineState(
            source_dir=Path("/tmp/src"),
            output_base=Path("/tmp/out"),
        )
        state["allow_seo_degraded"] = True
        state["force_sidecar_reset"] = True
        d = state.to_dict()
        restored = PipelineState.from_dict(d)
        assert restored.get("allow_seo_degraded") is True
        assert not restored.get("force_sidecar_reset")


class TestBooleanCliSentinelPattern:
    """Bug (b): boolean CLI flags must only populate ``schedule_overrides``
    when the CLI EXPLICITLY supplied them. Otherwise a default ``False``
    from a CLI invocation that did not pass the flag overwrites a saved
    ``True``.

    The fix uses an argparse-sentinel: the parser defaults bool flags to
    ``None`` on the CLI side; the orchestrator's ``run()`` translates None
    (or a fresh True/False from argparse) into a tri-state the resume path
    respects — ``None`` means "not supplied, state wins".

    These tests go through ``DrivePipelineOrchestrator._try_resume``
    because that's where the override is captured.
    """

    def test_cli_allow_seo_degraded_unset_preserves_state_true(self, tmp_path):
        """Operator ran once with --allow-seo-degraded (state has True).
        Resumes the next day without the flag — state MUST stay True.
        Before the fix, CLI default False overwrote state True."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator
        from gomoveshift.pipeline.drive_state_model import PipelineState

        orch = DrivePipelineOrchestrator(config=Config())

        state_file = tmp_path / "pipeline_state.json"
        _write_saved_state(
            state_file,
            {
                "phases_completed": ["phase_a"],
                "allow_seo_degraded": True,
            },
        )

        state = PipelineState(
            source_dir=Path("/tmp/src"),
            output_base=tmp_path,
        )
        # Simulate CLI default: operator did NOT pass --allow-seo-degraded.
        # Sentinel pattern: the flag is None on state (not False) when
        # unsupplied, so the override filter skips it and state wins.
        state["allow_seo_degraded"] = None

        stages_completed: list[str] = []
        orch._try_resume(state, state_file, tmp_path, stages_completed, fresh=False)

        assert state.get("allow_seo_degraded") is True, (
            "CLI default for --allow-seo-degraded must NOT "
            "overwrite a saved True. The sentinel pattern lets state win "
            "when the CLI did not supply the flag."
        )

    def test_cli_allow_seo_degraded_explicit_false_overrides_state_true(self, tmp_path):
        """When the operator explicitly passes an explicit False (direct
        Python call), CLI must win over state True."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator
        from gomoveshift.pipeline.drive_state_model import PipelineState

        orch = DrivePipelineOrchestrator(config=Config())

        state_file = tmp_path / "pipeline_state.json"
        _write_saved_state(
            state_file,
            {
                "phases_completed": ["phase_a"],
                "allow_seo_degraded": True,
            },
        )

        state = PipelineState(
            source_dir=Path("/tmp/src"),
            output_base=tmp_path,
        )
        # Explicit False via the sentinel — operator said "no".
        state["allow_seo_degraded"] = False

        stages_completed: list[str] = []
        orch._try_resume(state, state_file, tmp_path, stages_completed, fresh=False)

        assert state.get("allow_seo_degraded") is False, (
            "Explicit CLI False must win over persisted True (intent flag)."
        )

    def test_cli_allow_seo_degraded_explicit_true_overrides_state_none(self, tmp_path):
        """Happy path: operator opts in to allow_seo_degraded on resume.
        CLI True wins over state default (None or absent)."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator
        from gomoveshift.pipeline.drive_state_model import PipelineState

        orch = DrivePipelineOrchestrator(config=Config())

        state_file = tmp_path / "pipeline_state.json"
        _write_saved_state(
            state_file,
            {
                "phases_completed": ["phase_a"],
            },
        )

        state = PipelineState(
            source_dir=Path("/tmp/src"),
            output_base=tmp_path,
        )
        state["allow_seo_degraded"] = True

        stages_completed: list[str] = []
        orch._try_resume(state, state_file, tmp_path, stages_completed, fresh=False)

        assert state.get("allow_seo_degraded") is True


class TestForceSidecarResetDoesNotSurviveResume:
    """A poisoned on-disk checkpoint (``force_sidecar_reset: true``, e.g.
    from before this field was excluded from serialization) must not
    re-wipe both upload sidecars on an ordinary resume (CLI flag absent).
    """

    def test_try_resume_alone_reimports_a_poisoned_saved_true(self, tmp_path):
        """Documents WHY the post-merge reset is required: ``_try_resume``'s
        ``merge_on_resume`` call, by itself, re-imports a legacy poisoned
        ``force_sidecar_reset=True`` from the checkpoint even though this
        invocation's CLI did not pass ``--force-sidecar-reset``."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator
        from gomoveshift.pipeline.drive_state_model import PipelineState

        orch = DrivePipelineOrchestrator(config=Config())
        state_file = tmp_path / "pipeline_state.json"
        _write_saved_state(
            state_file,
            {
                "phases_completed": ["phase_a"],
                "force_sidecar_reset": True,
            },
        )

        state = PipelineState(source_dir=Path("/tmp/src"), output_base=tmp_path)
        stages_completed: list[str] = []
        orch._try_resume(state, state_file, tmp_path, stages_completed, fresh=False)

        assert state.get("force_sidecar_reset") is True, (
            "merge_on_resume is expected to re-import a legacy poisoned "
            "checkpoint value — this is exactly the gap "
            "reset_force_sidecar_reset_after_resume must close afterward."
        )

    def test_reset_after_resume_clears_a_poisoned_saved_true(self, tmp_path):
        """The gate input: after ``_try_resume`` re-imports a poisoned
        checkpoint True, ``reset_force_sidecar_reset_after_resume`` (called
        immediately afterward in ``run()``, with the CLI flag absent) must
        clear it before Phase D's sidecar-reset gate ever reads it."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.drive_orchestrator import (
            DrivePipelineOrchestrator,
            reset_force_sidecar_reset_after_resume,
        )
        from gomoveshift.pipeline.drive_state_model import PipelineState

        orch = DrivePipelineOrchestrator(config=Config())
        state_file = tmp_path / "pipeline_state.json"
        _write_saved_state(
            state_file,
            {
                "phases_completed": ["phase_a"],
                "force_sidecar_reset": True,
            },
        )

        state = PipelineState(source_dir=Path("/tmp/src"), output_base=tmp_path)
        stages_completed: list[str] = []
        orch._try_resume(state, state_file, tmp_path, stages_completed, fresh=False)
        reset_force_sidecar_reset_after_resume(state, cli_value=None)

        assert not state.get("force_sidecar_reset"), (
            "A poisoned on-disk force_sidecar_reset=True must not survive "
            "a resume where the CLI flag was not passed — the sidecar "
            "reset gate reads this field and, if True, wipes "
            "youtube_uploads.json/superlong_uploads.json and re-uploads "
            "every published LF as a new public video."
        )

    def test_reset_after_resume_honors_an_explicit_cli_flag(self):
        """--force-sidecar-reset passed on THIS invocation must still win,
        even after the resume merge."""
        from gomoveshift.pipeline.drive_orchestrator import (
            reset_force_sidecar_reset_after_resume,
        )
        from gomoveshift.pipeline.drive_state_model import PipelineState

        state = PipelineState(source_dir=Path("/tmp/src"), output_base=Path("/tmp/out"))
        state["force_sidecar_reset"] = None

        reset_force_sidecar_reset_after_resume(state, cli_value=True)

        assert state.get("force_sidecar_reset") is True
