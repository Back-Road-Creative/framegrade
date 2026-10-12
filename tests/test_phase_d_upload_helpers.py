"""Unit tests for the Phase D upload helpers.

The helpers are pure module-level functions extracted from
``PhaseD.run``'s per-LF Drive + YouTube loops. Behavior is unchanged
from the inline form; these tests pin the per-helper contract so future
refactors can move the bodies without re-deriving semantics.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gomoveshift.pipeline.drive_phases import external_systems_coordinator as esc
from gomoveshift.pipeline.drive_phases import phase_d
from gomoveshift.pipeline.drive_phases.phase_d import (
    _resolve_shorts_paths,
    _upload_all_lfs_shorts_to_drive,
    _upload_lf_shorts_to_drive,
    _upload_lf_to_youtube,
)
from gomoveshift.pipeline.exceptions import YouTubeQuotaError

from _state_helpers import make_state


def _lf(idx):
    return SimpleNamespace(index=idx)


class TestResolveShortsPaths:
    def test_none_map_returns_empty(self):
        assert _resolve_shorts_paths(_lf(1), None) == []

    def test_empty_map_returns_empty(self):
        assert _resolve_shorts_paths(_lf(1), {}) == []

    def test_missing_lf_index_returns_empty(self):
        assert _resolve_shorts_paths(_lf(7), {1: [Path("/x/a.mp4")]}) == []

    def test_none_value_for_lf_index_returns_empty(self):
        assert _resolve_shorts_paths(_lf(1), {1: None}) == []

    def test_coerces_path_objects_to_strings(self):
        paths = [Path("/x/short_01.mp4"), Path("/x/short_02.mp4")]
        out = _resolve_shorts_paths(_lf(1), {1: paths})
        assert out == ["/x/short_01.mp4", "/x/short_02.mp4"]
        assert all(isinstance(p, str) for p in out)

    def test_passes_through_strings(self):
        out = _resolve_shorts_paths(_lf(2), {2: ["/x/a.mp4", "/x/b.mp4"]})
        assert out == ["/x/a.mp4", "/x/b.mp4"]


class TestUploadLfShortsToDriveDayfirst:
    def test_day_first_synthesises_manifest_from_zip(self, monkeypatch):
        uploader = MagicMock()
        monkeypatch.setattr(
            esc,
            "_resolve_drive_subfolder_id",
            lambda u, name: f"folder-{name}",
        )
        uploader.upload_shorts.return_value = [
            "https://drive/short_01",
            "https://drive/short_02",
        ]
        manifest = _upload_lf_shorts_to_drive(
            uploader=uploader,
            lf=_lf(3),
            project_name="proj_lf03",
            shorts_paths=["/x/short_01.mp4", "/x/short_02.mp4"],
            output_base=Path("/out"),
            use_dayfirst=True,
        )
        uploader.upload_shorts.assert_called_once_with(
            shorts_dir=["/x/short_01.mp4", "/x/short_02.mp4"],
            folder_id="folder-proj_lf03",
        )
        assert manifest.project_name == "proj_lf03"
        assert manifest.folder_id == "folder-proj_lf03"
        assert manifest.completed is True
        assert manifest.files == {
            "short_01.mp4": "https://drive/short_01",
            "short_02.mp4": "https://drive/short_02",
        }


class TestUploadLfShortsToDriveLegacy:
    def test_legacy_delegates_to_directory_form(self):
        uploader = MagicMock()
        sentinel = object()
        uploader.upload_shorts.return_value = sentinel
        result = _upload_lf_shorts_to_drive(
            uploader=uploader,
            lf=_lf(5),
            project_name="proj_lf05",
            shorts_paths=[],
            output_base=Path("/out"),
            use_dayfirst=False,
        )
        uploader.upload_shorts.assert_called_once_with(
            shorts_dir="/out/shorts/lf_05",
            project_name="proj_lf05",
            drive_subfolder="proj_lf05",
        )
        assert result is sentinel


class TestUploadLfToYoutube:
    def _mk_yt(self, return_value=None, side_effect=None):
        yt = MagicMock()
        if side_effect is not None:
            yt.upload_if_needed.side_effect = side_effect
        else:
            yt.upload_if_needed.return_value = return_value
        return yt

    def test_success_returns_record_and_forwards_args(self):
        record = {"video_id": "abc"}
        yt = self._mk_yt(return_value=record)
        out = _upload_lf_to_youtube(
            yt=yt,
            state={"k": 1},
            lf=_lf(2),
            lf_path=Path("/v/lf_02.mp4"),
            title="T",
            description="D",
            tags=["t1"],
            publish_at=None,
            metadata={"title": "T"},
            config=MagicMock(),
        )
        assert out is record
        kwargs = yt.upload_if_needed.call_args.kwargs
        assert kwargs["lf_idx"] == 2
        assert kwargs["video_path"] == Path("/v/lf_02.mp4")
        assert kwargs["category_id"] == "2"
        assert kwargs["privacy"] == "private"
        assert kwargs["metadata"] == {"title": "T"}

    def test_refresh_error_is_wrapped_as_runtime_error(self):
        # _RefreshError is bound at import time and may be ``type(None)``
        # when google.auth isn't installed; force the real class for this
        # test so the wrapper logic exercises.
        class _RefreshError(Exception):
            pass

        _Refresh = _RefreshError

        from gomoveshift.pipeline.drive_phases import external_systems_coordinator as pd

        original = pd._RefreshError
        pd._RefreshError = _Refresh
        try:
            yt = self._mk_yt(side_effect=_Refresh("token expired"))
            with pytest.raises(RuntimeError) as exc:
                _upload_lf_to_youtube(
                    yt=yt,
                    state=make_state(),
                    lf=_lf(4),
                    lf_path=Path("/v/lf.mp4"),
                    title="t",
                    description="",
                    tags=[],
                    publish_at=None,
                    metadata={},
                    config=MagicMock(),
                )
            assert "LF 4" in str(exc.value)
            assert "youtube-auth" in str(exc.value)
        finally:
            pd._RefreshError = original

    def test_quota_error_propagates_unchanged(self):
        quota = YouTubeQuotaError("daily quota exceeded")
        yt = self._mk_yt(side_effect=quota)
        with pytest.raises(YouTubeQuotaError) as exc:
            _upload_lf_to_youtube(
                yt=yt,
                state=make_state(),
                lf=_lf(1),
                lf_path=Path("/v/lf.mp4"),
                title="t",
                description="",
                tags=[],
                publish_at=None,
                metadata={},
                config=MagicMock(),
            )
        assert exc.value is quota

    def test_http_error_returns_none(self, caplog):
        class _HttpError(Exception):
            pass

        _Http = _HttpError

        from gomoveshift.pipeline.drive_phases import external_systems_coordinator as pd

        original = pd._HttpError
        pd._HttpError = _Http
        try:
            yt = self._mk_yt(side_effect=_Http("503"))
            with caplog.at_level("WARNING"):
                out = _upload_lf_to_youtube(
                    yt=yt,
                    state=make_state(),
                    lf=_lf(6),
                    lf_path=Path("/v/lf.mp4"),
                    title="t",
                    description="",
                    tags=[],
                    publish_at=None,
                    metadata={},
                    config=MagicMock(),
                )
            assert out is None
            assert any("LF 6" in r.getMessage() for r in caplog.records)
        finally:
            pd._HttpError = original

    def test_generic_exception_returns_none(self, caplog):
        yt = self._mk_yt(side_effect=ValueError("oops"))
        with caplog.at_level("ERROR"):
            out = _upload_lf_to_youtube(
                yt=yt,
                state=make_state(),
                lf=_lf(9),
                lf_path=Path("/v/lf.mp4"),
                title="t",
                description="",
                tags=[],
                publish_at=None,
                metadata={},
                config=MagicMock(),
            )
        assert out is None
        assert any("LF 9" in r.getMessage() for r in caplog.records)


class TestUploadAllLfsShortsToDriveRefreshError:
    """An expired Drive credential must fail fast for the whole batch, not
    warn per-LF and complete with a partial manifest."""

    def test_refresh_error_aborts_loop_with_runtime_error(self, monkeypatch):
        class _RefreshError(Exception):
            pass

        monkeypatch.setattr(phase_d, "_RefreshError", _RefreshError)

        # Make the per-LF call raise RefreshError on the first invocation.
        def _fail(*args, **kwargs):
            raise _RefreshError("token expired")

        monkeypatch.setattr(phase_d, "_upload_lf_shorts_to_drive", _fail)

        drive_plan = SimpleNamespace(regular_lfs=[_lf(1), _lf(2), _lf(3)])
        state = make_state(
            shorts_per_lf={1: ["/x/short.mp4"], 2: ["/x/short.mp4"], 3: ["/x/short.mp4"]}
        )

        with pytest.raises(RuntimeError) as exc:
            _upload_all_lfs_shorts_to_drive(
                drive_plan=drive_plan,
                state=state,
                project_base="proj",
                output_base=Path("/tmp/out"),
                uploader_factory=lambda: MagicMock(),
                skip_drive_upload=False,
            )
        assert "LF 1" in str(exc.value)
        assert "refresh the Drive token" in str(exc.value)
        assert "GMS_DRIVE_TOKEN_ENC" in str(exc.value)
