"""Tests for save_sidecar SHA-256 return value."""

import hashlib
import json

from gomoveshift.pipeline.youtube_sidecar import save_sidecar


class TestSaveSidecarReturnsSha:
    """save_sidecar must return the SHA-256 hex digest of the written file."""

    def test_returns_64_char_hex_string(self, tmp_path):
        sha = save_sidecar(tmp_path, {"1": {"video_id": "abc123"}})
        assert isinstance(sha, str)
        assert len(sha) == 64
        # Must be valid hex
        int(sha, 16)

    def test_sha_matches_file_on_disk(self, tmp_path):
        uploads = {"1": {"video_id": "abc123"}, "2": {"video_id": "def456"}}
        sha = save_sidecar(tmp_path, uploads)

        # Independently compute SHA of the file that was written
        path = tmp_path / "youtube_uploads.json"
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                digest.update(chunk)
        expected = digest.hexdigest()

        assert sha == expected

    def test_empty_uploads_returns_sha(self, tmp_path):
        sha = save_sidecar(tmp_path, {})
        assert len(sha) == 64

        # Verify file contents match
        path = tmp_path / "youtube_uploads.json"
        assert json.loads(path.read_text()) == {}

    def test_none_uploads_treated_as_empty(self, tmp_path):
        sha = save_sidecar(tmp_path, None)
        assert len(sha) == 64

        path = tmp_path / "youtube_uploads.json"
        assert json.loads(path.read_text()) == {}
