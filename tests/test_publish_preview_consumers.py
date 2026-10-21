"""GMS-17a: the consumers of a publish plan refuse a publish preview on disk.

``load_publish_plan`` refuses a ``kind: publish_preview`` file
(``tests/test_phase_d_publish_preview.py``). These tests pin the two
commands that read plans off disk -- ``enqueue-plan`` and ``dispatch-queue``
-- against the preview Phase D really writes, including a preview copied
over the plan's file name.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from gomoveshift.cli.handlers.dispatch_queue import SidecarRecordLoader
from gomoveshift.cli.handlers.enqueue_plan import cmd_enqueue_plan
from gomoveshift.publishing.plan_io import PublishPlanFormatError
from gomoveshift.publishing.queue import enqueue, open_queue_db

sys.path.insert(0, str(Path(__file__).parent))

from test_phase_d_publish_preview import PREVIEW_NAME, _write  # noqa: E402

PROJECT = PREVIEW_NAME.removesuffix("_publish_preview.json")
PLAN_NAME = f"{PROJECT}_publish_plan.json"


def _preview(out: Path, *, as_plan: bool = False) -> Path:
    _write(out, publish_pass=False)
    path = out / PREVIEW_NAME
    if as_plan:
        path = path.rename(out / PLAN_NAME)
    return path


def _enqueue(out: Path, tmp_path: Path, *, apply: bool) -> int:
    state = SimpleNamespace(
        regular_lf_paths=[], gate_cd_confirmed=None, gate_cd_confirmation_hash=None
    )
    state_path = out / "pipeline_state.json"
    state_path.write_text("{}", encoding="utf-8")
    args = SimpleNamespace(
        state_path=str(state_path), apply=apply, output_dir=str(out), shadow=False
    )
    config = SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path / "seo")))
    with mock.patch("gomoveshift.cli.handlers.enqueue_plan._load_state", return_value=state):
        return cmd_enqueue_plan(args, config)


def _queued(tmp_path: Path) -> int:
    conn = open_queue_db(tmp_path / "seo")
    try:
        return conn.execute("SELECT COUNT(*) AS n FROM deliveries").fetchone()["n"]
    finally:
        conn.close()


@pytest.fixture
def out(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    return d


class TestEnqueuePlan:
    def test_a_preview_is_never_picked_up_as_a_sidecar(self, out, tmp_path, capsys):
        _preview(out)
        assert _enqueue(out, tmp_path, apply=True) == 0
        assert "No *_publish_plan.json sidecar" in capsys.readouterr().out
        assert _queued(tmp_path) == 0

    def test_apply_refuses_a_preview_under_the_plan_name(self, out, tmp_path, capsys):
        _preview(out, as_plan=True)
        assert _enqueue(out, tmp_path, apply=True) == 1
        assert "publish_preview" in capsys.readouterr().out
        assert _queued(tmp_path) == 0

    def test_dry_run_refuses_a_preview_under_the_plan_name(self, out, tmp_path, capsys):
        """A dry run is the operator's check; it must not report a preview as
        a plan that is ready to spool."""
        _preview(out, as_plan=True)
        assert _enqueue(out, tmp_path, apply=False) == 1
        printed = capsys.readouterr().out
        assert "publish_preview" in printed
        assert "DRY RUN" not in printed
        assert _queued(tmp_path) == 0


class TestDispatchQueue:
    def test_loader_refuses_a_preview_under_the_plan_name(self, out):
        _preview(out, as_plan=True)
        with pytest.raises(PublishPlanFormatError, match="publish_preview"):
            SidecarRecordLoader(out)(PROJECT, 1)

    def test_loader_refuses_a_row_that_points_at_a_preview(self, out, tmp_path):
        path = _preview(out)
        conn = open_queue_db(tmp_path / "seo")
        try:
            enqueue(conn, PROJECT, 1, "youtube", "HASH1", sidecar_path=str(path))
            with pytest.raises(PublishPlanFormatError, match="publish_preview"):
                SidecarRecordLoader(tmp_path / "unrelated", conn=conn)(PROJECT, 1)
        finally:
            conn.close()
