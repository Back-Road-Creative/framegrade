"""Tests for Drive link format handling in uploader.

Validates that verify_uploads correctly extracts file_id from the
file/d/ID/view URL format used by drive_download_link().
"""

import re
from unittest.mock import MagicMock, patch

from gomoveshift.pipeline.uploader import (
    PipelineUploader,
    UploadManifest,
    drive_download_link,
)


class TestDriveLinkRegex:
    """Test that verify_uploads regex matches file/d/ID/view format."""

    def test_extracts_file_id_from_view_url(self):
        """Regex should extract file_id from file/d/ID/view format."""
        link = "https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQrStUvWxYz/view?usp=sharing"
        match = re.search(r"/d/([a-zA-Z0-9_-]+)/", link)
        assert match is not None
        assert match.group(1) == "1AbCdEfGhIjKlMnOpQrStUvWxYz"  # pragma: allowlist secret

    def test_drive_download_link_produces_matchable_url(self):
        """drive_download_link() output should be parseable by verify regex."""
        file_id = "1AbCdEfGhIjKlMnOpQrStUvWxYz"
        link = drive_download_link(file_id)
        match = re.search(r"/d/([a-zA-Z0-9_-]+)/", link)
        assert match is not None
        assert match.group(1) == file_id

    def test_old_query_param_format_does_not_match(self):
        """Old ?id= format should NOT match the new regex."""
        link = "https://drive.google.com/uc?id=1AbCdEfGhIjKlMnOpQrStUvWxYz&export=download"
        match = re.search(r"/d/([a-zA-Z0-9_-]+)/", link)
        assert match is None

    def test_file_id_with_hyphens_and_underscores(self):
        """File IDs with hyphens and underscores should be captured."""
        link = "https://drive.google.com/file/d/1A-b_C-d_2E/view?usp=sharing"
        match = re.search(r"/d/([a-zA-Z0-9_-]+)/", link)
        assert match is not None
        assert match.group(1) == "1A-b_C-d_2E"

    @patch.object(PipelineUploader, "_get_client")
    def test_verify_uploads_uses_correct_regex(self, mock_get_client):
        """verify_uploads should successfully extract IDs from file/d/ URLs."""
        mock_drive = MagicMock()
        mock_get_client.return_value = mock_drive
        mock_drive.files.return_value.get.return_value.execute.return_value = {
            "id": "abc123",
            "name": "short_01.mp4",
        }

        uploader = PipelineUploader.__new__(PipelineUploader)
        uploader._client = mock_drive

        manifest = UploadManifest(
            project_name="test",
            files={
                "short_01.mp4": "https://drive.google.com/file/d/abc123/view?usp=sharing",
            },
        )

        result = uploader.verify_uploads(manifest)
        assert "short_01.mp4" in result.verified
        assert len(result.failed) == 0
