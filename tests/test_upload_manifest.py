"""Tests for UploadManifest — resume-capable upload tracking."""

from gomoveshift.pipeline.uploader import UploadManifest


class TestUploadManifestSerialization:
    """UploadManifest round-trip to/from JSON."""

    def test_round_trip_structured(self, tmp_path):
        """Structured manifest saves and loads correctly."""
        m = UploadManifest(
            project_name="test_drive_lf01",
            folder_id="abc123",
            files={"short_01.mp4": "https://drive.google.com/file/d/x/view"},
            completed=True,
        )
        path = tmp_path / "manifest.json"
        m.save(path)

        loaded = UploadManifest.load(path)
        assert loaded.project_name == "test_drive_lf01"
        assert loaded.folder_id == "abc123"
        assert loaded.files["short_01.mp4"] == "https://drive.google.com/file/d/x/view"
        assert loaded.completed is True
        assert loaded.updated_at != ""

    def test_from_dict_flat_format(self):
        """Legacy flat format {filename: link} should be parsed correctly."""
        data = {
            "short_01.mp4": "https://drive.google.com/file/d/a/view",
            "short_02.mp4": "https://drive.google.com/file/d/b/view",
        }
        m = UploadManifest.from_dict(data)
        assert len(m.files) == 2
        assert m.completed is True  # Flat format assumed complete

    def test_empty_manifest(self):
        """Empty manifest is valid."""
        m = UploadManifest(project_name="empty")
        assert m.files == {}
        assert m.completed is False


class TestUploadManifestResume:
    """Resume capability: tracking pending vs completed files."""

    def test_get_pending_files(self):
        """Pending files = all - already uploaded."""
        m = UploadManifest(
            project_name="test",
            files={"short_01.mp4": "link_a", "short_02.mp4": "link_b"},
        )
        all_files = ["short_01.mp4", "short_02.mp4", "short_03.mp4"]
        pending = m.get_pending_files(all_files)
        assert pending == ["short_03.mp4"]

    def test_get_link_for_existing_file(self):
        """Can retrieve link for a specific file."""
        m = UploadManifest(
            project_name="test",
            files={"short_01.mp4": "link_a"},
        )
        assert m.get_link_for_file("short_01.mp4") == "link_a"
        assert m.get_link_for_file("nonexistent.mp4") is None

    def test_save_after_each_upload(self, tmp_path):
        """Manifest should persist incrementally (save after each upload)."""
        path = tmp_path / "manifest.json"
        m = UploadManifest(project_name="test")

        # Upload first file
        m.files["short_01.mp4"] = "link_a"
        m.save(path)

        # Simulate crash + reload
        loaded = UploadManifest.load(path)
        assert "short_01.mp4" in loaded.files
        assert "short_02.mp4" not in loaded.files

        # Upload second file
        loaded.files["short_02.mp4"] = "link_b"
        loaded.save(path)

        # Final reload
        final = UploadManifest.load(path)
        assert len(final.files) == 2
