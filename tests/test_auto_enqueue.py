"""Shorts are automatic: a successful publish run spools its own plan.

Regression: no short reached any platform because nothing
called ``enqueue-plan`` after a run (pipeline/auto_enqueue.py)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from gomoveshift.pipeline.auto_enqueue import auto_enqueue_shorts, should_auto_enqueue

_HANDLER = "gomoveshift.cli.handlers.enqueue_plan.cmd_enqueue_plan"


def _state(**over):
    base = dict(
        phases_completed=["phase_a", "phase_bc_day_first", "phase_d"],
        youtube_upload=True,
        skip_upload=False,
        skip_youtube_upload=False,
    )
    base.update(over)
    return SimpleNamespace(**base)


@pytest.mark.parametrize(
    "over",
    [
        {"phases_completed": ["phase_a", "phase_bc_day_first"]},
        {"youtube_upload": False},
        {"skip_upload": True},
        {"skip_youtube_upload": True},
    ],
)
def test_a_run_that_did_not_publish_enqueues_nothing(over, tmp_path):
    errors: list = []
    with mock.patch(_HANDLER) as handler:
        assert auto_enqueue_shorts(_state(**over), tmp_path / "s.json", object(), errors) is None
    handler.assert_not_called()
    assert errors == []
    assert should_auto_enqueue(_state(**over)) is False


def test_a_publish_run_enqueues_with_apply(tmp_path):
    errors: list = []
    config = object()
    state_file = tmp_path / "pipeline_state.json"
    with mock.patch(_HANDLER, return_value=0) as handler:
        rc = auto_enqueue_shorts(_state(), state_file, config, errors)
    assert rc == 0
    assert errors == []
    args, cfg = handler.call_args.args
    assert cfg is config
    assert args.apply is True
    assert args.shadow is False
    assert args.state_path == str(state_file)
    assert args.output_dir is None  # resolved from state.output_base by the handler


def test_a_refused_enqueue_is_a_run_error(tmp_path):
    errors: list = []
    with mock.patch(_HANDLER, return_value=1):
        assert auto_enqueue_shorts(_state(), tmp_path / "s.json", object(), errors) == 1
    assert len(errors) == 1 and "refused" in errors[0]


def test_a_crashing_enqueue_is_a_run_error(tmp_path):
    errors: list = []
    with mock.patch(_HANDLER, side_effect=RuntimeError("db locked")):
        assert auto_enqueue_shorts(_state(), tmp_path / "s.json", object(), errors) == 1
    assert "db locked" in errors[0]


def test_orchestrator_calls_auto_enqueue_before_the_dod_validator():
    src = Path(__file__).resolve().parents[1] / "gomoveshift/pipeline/drive_orchestrator.py"
    text = src.read_text(encoding="utf-8")
    call = text.index("auto_enqueue_shorts(state, state_file, self.config, errors)")
    dod = text.index("self._run_dod_validator(\n                state=state,")
    assert call < dod
