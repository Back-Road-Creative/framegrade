"""The standalone ``gomoveshift upload`` command must gate the directory.

It previously globbed ALL ``short_*.mp4`` on disk and shipped them to Drive +
regenerated the Metricool CSV with ZERO loudness/content/count/geometry gates.
A pre-fix or stale short left on disk would ship through it unreviewed. The gate
now refuses an ungated / stale / low-quality directory upstream of any upload.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _args(project_dir):
    return SimpleNamespace(
        project_dir=project_dir,
        credentials=None,
        token=None,
        init_oauth=False,
        skip_csv=True,
    )


def _run(project_dir):
    """Invoke cmd_upload with the uploader stubbed; return (rc, uploader_inst)."""
    import gomoveshift.pipeline.uploader as _uploader_mod

    inst = MagicMock()
    with patch.object(_uploader_mod, "PipelineUploader", MagicMock(return_value=inst)):
        from gomoveshift.cli.handlers.upload import cmd_upload

        rc = cmd_upload(_args(project_dir), {})
    return rc, inst


def _make_project(tmp_path, n_shorts=1, n_posts=1, with_manifest=True):
    project = tmp_path / "proj"
    shorts = project / "shorts"
    shorts.mkdir(parents=True)
    for i in range(1, n_shorts + 1):
        (shorts / f"short_{i:02d}.mp4").write_bytes(b"x")
    if with_manifest:
        (shorts / "social_posts.json").write_text(
            json.dumps([{"short_index": i, "caption": "c"} for i in range(1, n_posts + 1)])
        )
    return project


def test_refuses_ungated_dir_without_manifest(tmp_path):
    """No social_posts.json → the dir was not produced by a pipeline run."""
    project = _make_project(tmp_path, n_shorts=1, with_manifest=False)
    rc, uploader = _run(project)
    assert rc == 1
    uploader.upload_shorts.assert_not_called()


def test_refuses_stale_short_count_mismatch(tmp_path):
    """A stale short left on disk (2 files, manifest plans 1) must not ship."""
    project = _make_project(tmp_path, n_shorts=2, n_posts=1)
    rc, uploader = _run(project)
    assert rc == 1
    uploader.upload_shorts.assert_not_called()


def test_refuses_when_shorts_quality_gate_fails(tmp_path):
    """A count-matched dir whose shorts fail the quality gate must not ship."""
    project = _make_project(tmp_path, n_shorts=1, n_posts=1)
    from gomoveshift.pipeline.pipeline_validator import PipelineValidator

    failing = MagicMock(passed=False, summary=lambda: "bad geometry")
    with patch.object(PipelineValidator, "validate_shorts", lambda self, d: failing):
        rc, uploader = _run(project)
    assert rc == 1
    uploader.upload_shorts.assert_not_called()


def test_gated_dir_passes_to_upload(tmp_path):
    """A manifest-backed, count-matched, quality-passing dir reaches upload."""
    project = _make_project(tmp_path, n_shorts=1, n_posts=1)
    from gomoveshift.pipeline import uploader as _uploader_mod
    from gomoveshift.pipeline.pipeline_validator import PipelineValidator

    passing = MagicMock(passed=True, summary=lambda: "ok")
    manifest = MagicMock()
    manifest.files = [MagicMock()]
    inst = MagicMock()
    inst.upload_shorts.return_value = manifest
    with (
        patch.object(PipelineValidator, "validate_shorts", lambda self, d: passing),
        patch.object(_uploader_mod, "PipelineUploader", MagicMock(return_value=inst)),
        patch.object(_uploader_mod, "DEFAULT_CREDENTIALS_ENC", MagicMock(exists=lambda: True)),
        patch.object(_uploader_mod, "DEFAULT_TOKEN_ENC", MagicMock(exists=lambda: True)),
    ):
        from gomoveshift.cli.handlers.upload import cmd_upload

        rc = cmd_upload(_args(project), {})
    assert rc == 0
    inst.upload_shorts.assert_called_once()
