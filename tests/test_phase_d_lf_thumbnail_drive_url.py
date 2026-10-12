"""Phase D: LF canonical thumbnails are uploaded to Drive.

Originally closed a bug where the per-LF
Metricool CSV emitted ``https://youtu.be/<id>`` for ``Picture Url 1``.

Per-LF Metricool CSVs were retired entirely
since they were never imported by Metricool. The CSV-side assertions in
this module were removed with that retirement; what survives is the
Drive thumbnail upload itself, kept around for any future consumer
(manual Metricool LF import, BC operator preview, etc.).

Tests in this module reuse the helpers from ``test_phase_d_lf_outputs``
(``_install_fake_youtube_uploader``, ``_build_state``, ``_phase_d_patches``,
``_setup_mocks``) so the fixture surface remains a single source of truth.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from gomoveshift.config import get_config
from gomoveshift.pipeline.drive_phases.phase_d import PhaseD

from test_phase_d_lf_outputs import (  # type: ignore[import-not-found]
    _build_state,
    _install_fake_youtube_uploader,
    _phase_d_patches,
    _setup_mocks,
)


# A predictable Drive thumbnail URL the mocked uploader returns. Matches the
# shape produced by ``gomoveshift.pipeline.uploader.drive_download_link``
# (``/file/d/<id>/view?usp=sharing``).
FAKE_THUMB_URL_TEMPLATE = "https://drive.google.com/file/d/THUMB_LF{idx:02d}/view?usp=sharing"


def _fake_thumbnail_factory(tmp_root):
    """Same shape as ``test_phase_d_lf_outputs.TestPhaseDThumbnails._fake_generator_factory``."""

    def _fake(video_path, title, subtitle, output_dir, duration):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        variant = output_dir / "frame1_bold.jpg"
        variant.write_bytes(b"fake-thumb-bytes" * 16)
        return variant

    return _fake


def _run_phase_d_with_upload(state, num_lfs):
    """Run Phase D with the standard mocks AND a working `_upload_file`
    side-effect on the ``PipelineUploader`` mock so § 5c (thumbnail Drive
    upload) returns a deterministic URL per LF. Mirrors the pattern from
    ``test_phase_d_lf_outputs._run_phase_d`` but threads through the
    thumbnail upload return value."""
    patches = _phase_d_patches()
    with (
        patches[0] as m0,
        patches[1] as m1,
        patches[2] as m2,
        patches[3] as m3,
        patches[4] as m4,
        patches[5] as m5,
        patches[6] as m6,
    ):
        _setup_mocks((m0, m1, m2, m3, m4, m5, m6), num_lfs)
        uploader = m4.return_value

        # Side-effect: return a deterministic Drive URL keyed off the
        # uploaded file's name (``lf_NN.jpg`` → ``THUMB_LFNN``).
        def _upload_file(drive, file_path, folder_id, make_shareable=True, mimetype=None):
            stem = Path(file_path).stem  # ``lf_01``
            idx = int(stem.split("_")[1])
            return FAKE_THUMB_URL_TEMPLATE.format(idx=idx)

        uploader._upload_file.side_effect = _upload_file
        # `_resolve_drive_subfolder_id` indirection: the helper falls back
        # to the override on the uploader mock when present; return a
        # fake folder id so the call chain doesn't try to walk a real
        # Drive client.
        uploader._resolve_drive_subfolder_id = lambda subfolder: f"folder_{subfolder}"

        PhaseD(get_config()).run(state)
        return uploader


class TestLFThumbnailDriveUpload:
    def test_phase_d_uploads_lf_thumbnails_to_drive(self, tmp_path, monkeypatch):
        """Phase D § 5c uploads each canonical thumbnail JPG to Drive."""
        _install_fake_youtube_uploader(monkeypatch)
        state = _build_state(tmp_path, num_lfs=3, youtube_upload=True, skip_upload=False)
        fake = _fake_thumbnail_factory(tmp_path)
        with patch(
            "gomoveshift.pipeline.drive_phases.thumbnail_generator.generate_lf_thumbnail",
            side_effect=fake,
        ):
            uploader = _run_phase_d_with_upload(state, num_lfs=3)

        # `_upload_file` was called once per LF for the canonical thumbnail.
        thumb_upload_calls = [
            c
            for c in uploader._upload_file.call_args_list
            if Path(c.args[1] if len(c.args) > 1 else c.kwargs["file_path"]).name.startswith("lf_")
        ]
        assert len(thumb_upload_calls) == 3, (
            f"expected 3 thumbnail uploads, got {len(thumb_upload_calls)}"
        )

        # State carries the resulting URLs keyed by LF index.
        urls = state.get("lf_thumbnail_drive_urls") or {}
        assert set(urls.keys()) == {1, 2, 3}, urls
        for idx in (1, 2, 3):
            assert urls[idx] == FAKE_THUMB_URL_TEMPLATE.format(idx=idx)

    def test_missing_thumbnail_hard_fails(self, tmp_path, monkeypatch):
        """A generator that produces no JPG aborts the run, it does not warn.

        This used to assert only that Phase D logged a WARN and carried on —
        precisely how a long-form could publish with no thumbnail.
        """
        _install_fake_youtube_uploader(monkeypatch)
        state = _build_state(tmp_path, num_lfs=1, youtube_upload=True, skip_upload=False)

        def _no_write(video_path, title, subtitle, output_dir, duration):
            return None

        with patch(
            "gomoveshift.pipeline.drive_phases.thumbnail_generator.generate_lf_thumbnail",
            side_effect=_no_write,
        ):
            with pytest.raises(RuntimeError, match="thumbnail"):
                _run_phase_d_with_upload(state, num_lfs=1)

        urls = state.get("lf_thumbnail_drive_urls") or {}
        assert 1 not in urls or not urls.get(1)
