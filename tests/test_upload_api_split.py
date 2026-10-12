"""Tests for upload_shorts API split and _upload_file signature correctness.

Covers:
- upload_shorts_by_paths passes Drive service to _upload_file
- upload_shorts_from_dir passes Drive service to _upload_file
- upload_shorts dispatches correctly to each sub-method
- Signature mismatch regression: _upload_file always receives drive as first arg
"""

import ssl
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from gomoveshift.pipeline.uploader import (
    DriveUploadError,
    PipelineUploader,
    UploadManifest,
)


def _make_uploader(tmp_path):
    """Return a PipelineUploader in legacy mode so no SOPS decrypt is needed."""
    creds = tmp_path / "credentials.json"
    token = tmp_path / "token.json"
    creds.write_text('{"installed":{}}')
    token.write_text("{}")
    return PipelineUploader(credentials_path=creds, token_path=token)


class TestUploadShortsByPaths:
    """Tests for the list-based upload_shorts_by_paths method."""

    def test_passes_drive_service_to_upload_file(self, tmp_path):
        """_upload_file must receive a Drive service object as first positional arg."""
        uploader = _make_uploader(tmp_path)
        video = tmp_path / "short.mp4"
        video.write_bytes(b"fake video")

        mock_drive = MagicMock()
        upload_calls = []

        original_upload = uploader._upload_file

        def spy_upload_file(*args, **kwargs):
            upload_calls.append((args, kwargs))
            # Return a fake link
            return "https://drive.google.com/file/d/fake123/view?usp=sharing"

        with (
            patch.object(uploader, "_build_drive_service", return_value=mock_drive),
            patch.object(uploader, "_upload_file", side_effect=spy_upload_file),
        ):
            result = uploader.upload_shorts_by_paths([str(video)], folder_id="folder_abc")

        assert len(result) == 1
        # First positional arg must be the drive service object
        first_call_args = upload_calls[0][0]
        assert first_call_args[0] is mock_drive, (
            "_upload_file must receive drive service as first arg"
        )

    def test_returns_list_of_dicts(self, tmp_path):
        """upload_shorts_by_paths must return a list of results."""
        uploader = _make_uploader(tmp_path)
        v1 = tmp_path / "a.mp4"
        v2 = tmp_path / "b.mp4"
        v1.write_bytes(b"x")
        v2.write_bytes(b"x")

        mock_drive = MagicMock()

        def mock_upload(drive, file_path, folder_id, make_shareable=True):
            return f"https://drive.google.com/file/d/{Path(file_path).stem}/view"

        with (
            patch.object(uploader, "_build_drive_service", return_value=mock_drive),
            patch.object(uploader, "_upload_file", side_effect=mock_upload),
        ):
            result = uploader.upload_shorts_by_paths([str(v1), str(v2)], folder_id="folder1")

        assert len(result) == 2

    def test_ssl_retry_passes_drive_service(self, tmp_path):
        """Even on SSL retry, _upload_file must still get the drive service."""
        uploader = _make_uploader(tmp_path)
        video = tmp_path / "short.mp4"
        video.write_bytes(b"data")

        mock_drive = MagicMock()
        call_count = {"n": 0}

        def mock_upload(drive, file_path, folder_id, make_shareable=True):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ssl.SSLError("DECRYPTION_FAILED_OR_BAD_RECORD_MAC")
            assert drive is mock_drive, "Drive service must be passed on retry too"
            return "https://drive.google.com/file/d/ok/view"

        with (
            patch.object(uploader, "_build_drive_service", return_value=mock_drive),
            patch.object(uploader, "_upload_file", side_effect=mock_upload),
            patch("gomoveshift.pipeline.uploader.time") as mock_time,
        ):
            mock_time.sleep = MagicMock()
            result = uploader.upload_shorts_by_paths([str(video)], folder_id="folder1")

        assert len(result) == 1
        assert call_count["n"] == 2

    def test_empty_list_returns_empty(self, tmp_path):
        """Passing an empty list should return an empty list."""
        uploader = _make_uploader(tmp_path)
        mock_drive = MagicMock()
        with patch.object(uploader, "_build_drive_service", return_value=mock_drive):
            result = uploader.upload_shorts_by_paths([], folder_id="folder1")
        assert result == []


class TestUploadShortsFromDir:
    """Tests for the directory-based upload_shorts_from_dir method."""

    def test_returns_upload_manifest(self, tmp_path):
        """upload_shorts_from_dir must return an UploadManifest."""
        uploader = _make_uploader(tmp_path)
        shorts_dir = tmp_path / "shorts"
        shorts_dir.mkdir()
        (shorts_dir / "short_01.mp4").write_bytes(b"video data")

        mock_drive = MagicMock()
        mock_drive.files.return_value.list.return_value.execute.return_value = {
            "files": [{"id": "folder1", "name": "test"}]
        }

        def mock_upload(drive, file_path, folder_id, make_shareable=True):
            return "https://drive.google.com/file/d/abc/view"

        with (
            patch.object(uploader, "_get_client", return_value=mock_drive),
            patch.object(uploader, "_build_drive_service", return_value=mock_drive),
            patch.object(uploader, "_upload_file", side_effect=mock_upload),
            patch.object(uploader, "_refresh_credentials_if_needed"),
            patch.object(uploader, "_get_or_create_folder_path", return_value="folder1"),
        ):
            result = uploader.upload_shorts_from_dir(
                shorts_dir=str(shorts_dir),
                project_name="test_project",
            )

        assert isinstance(result, UploadManifest)
        assert "short_01.mp4" in result.files

    def test_missing_dir_raises(self, tmp_path):
        """upload_shorts_from_dir must raise FileNotFoundError for missing dirs."""
        uploader = _make_uploader(tmp_path)
        with pytest.raises(FileNotFoundError):
            uploader.upload_shorts_from_dir(
                shorts_dir=str(tmp_path / "nonexistent"),
                project_name="test",
            )


class TestUploadShortsDispatcher:
    """The upload_shorts method should dispatch to the correct sub-method."""

    def test_list_input_dispatches_to_by_paths(self, tmp_path):
        """Passing a list dispatches to upload_shorts_by_paths."""
        uploader = _make_uploader(tmp_path)
        video = tmp_path / "short.mp4"
        video.write_bytes(b"x")

        expected = [{"id": "test"}]
        with patch.object(
            uploader, "upload_shorts_by_paths", return_value=expected
        ) as mock_by_paths:
            result = uploader.upload_shorts([str(video)], folder_id="folder1")

        assert result == expected
        mock_by_paths.assert_called_once()

    def test_folder_id_kwarg_dispatches_to_by_paths(self, tmp_path):
        """Passing folder_id kwarg with string dispatches to upload_shorts_by_paths."""
        uploader = _make_uploader(tmp_path)
        video = tmp_path / "short.mp4"
        video.write_bytes(b"x")

        expected = [{"id": "test"}]
        with patch.object(
            uploader, "upload_shorts_by_paths", return_value=expected
        ) as mock_by_paths:
            result = uploader.upload_shorts(str(video), folder_id="folder1")

        assert result == expected
        mock_by_paths.assert_called_once()

    def test_dir_string_dispatches_to_from_dir(self, tmp_path):
        """Passing a directory string with no folder_id dispatches to from_dir."""
        uploader = _make_uploader(tmp_path)

        expected = UploadManifest(project_name="test")
        with patch.object(
            uploader, "upload_shorts_from_dir", return_value=expected
        ) as mock_from_dir:
            result = uploader.upload_shorts(str(tmp_path), project_name="test")

        assert result is expected
        mock_from_dir.assert_called_once()
