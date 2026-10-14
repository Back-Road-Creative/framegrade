"""Characterization tests for the photo-post cleanup.

Two independent properties are pinned here:

1. ``pinterest_description`` is fully retired. The Metricool golden schema
   exposes only ``Pinterest`` / ``Pinterest Board`` — no description column
   ever routed the value anywhere, so the field must not exist on either
   dataclass nor appear in any serialized output.
2. The photo orchestrator resolves its env-var-backed directories through
   ``gomoveshift.paths`` rather than calling ``os.environ`` directly, with
   byte-identical defaults.
"""

from dataclasses import fields
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift import paths
from gomoveshift.pipeline.photos.caption_generator import CaptionBatch, PhotoCaption
from gomoveshift.pipeline.photos.csv_generator import PhotoPost
from gomoveshift.pipeline.photos.orchestrator import PhotoOrchestrator
from gomoveshift.pipeline.uploader import VerificationResult
from gomoveshift.publishing.photo_run_hooks import photo_run_hooks


class TestPinterestDescriptionRetired:
    def test_photo_caption_has_no_pinterest_field(self):
        assert "pinterest_description" not in {f.name for f in fields(PhotoCaption)}

    def test_photo_post_has_no_pinterest_field(self):
        assert "pinterest_description" not in {f.name for f in fields(PhotoPost)}

    def test_photo_post_to_dict_omits_pinterest(self):
        post = PhotoPost(
            photo_index=0,
            caption="A quiet road through Connemara.",
            tiktok_caption="Connemara coast road",
            hashtags="#ireland",
            first_comment="More photos: https://gomoveshift.com/",
            photo_filename="example-2025-01_photo-001.jpg",
        )
        assert "pinterest_description" not in post.to_dict()

    def test_photo_caption_rejects_pinterest_kwarg(self):
        """The field is gone, not merely defaulted to an empty string."""
        with pytest.raises(TypeError):
            PhotoCaption(
                photo_index=0,
                main_caption="A quiet road.",
                tiktok_caption="Quiet road",
                pinterest_description="Sunset over Tuscany.",
                hashtags=["ireland"],
            )


class TestPhotoPathsRouting:
    def test_defaults_resolve_under_the_data_home(self, monkeypatch, tmp_path):
        """Both defaults ride the per-user data home, and both env vars override."""
        monkeypatch.delenv("GMS_PHOTO_OUTPUT_DIR", raising=False)
        monkeypatch.delenv("GMS_EXPORT_DIR", raising=False)
        monkeypatch.setenv("GMS_DATA_HOME", str(tmp_path / "data"))
        assert paths.gms_photo_output_dir() == tmp_path / "data" / "photos"
        assert paths.gms_export_dir() == tmp_path / "data" / "output"
        monkeypatch.setenv("GMS_PHOTO_OUTPUT_DIR", str(tmp_path / "p"))
        monkeypatch.setenv("GMS_EXPORT_DIR", str(tmp_path / "e"))
        assert paths.gms_photo_output_dir() == tmp_path / "p"
        assert paths.gms_export_dir() == tmp_path / "e"

    def test_output_dir_defaults_to_accessor_when_env_unset(self, monkeypatch, tmp_path):
        monkeypatch.delenv("GMS_PHOTO_OUTPUT_DIR", raising=False)
        assert PhotoOrchestrator(site_root=tmp_path).output_dir == paths.gms_photo_output_dir()

    def test_output_dir_honors_env_var(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GMS_PHOTO_OUTPUT_DIR", str(tmp_path / "photo_out"))
        orchestrator = PhotoOrchestrator(site_root=tmp_path)
        assert orchestrator.output_dir == tmp_path / "photo_out"
        assert orchestrator.output_dir == paths.gms_photo_output_dir()

    def test_output_dir_routes_through_paths_accessor(self, monkeypatch, tmp_path):
        """Patching the accessor must move the orchestrator's default."""
        sentinel = tmp_path / "sentinel_out"
        monkeypatch.setattr(paths, "gms_photo_output_dir", lambda: sentinel)
        assert PhotoOrchestrator(site_root=tmp_path).output_dir == sentinel

    def test_explicit_output_dir_still_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GMS_PHOTO_OUTPUT_DIR", str(tmp_path / "ignored"))
        assert PhotoOrchestrator(
            site_root=tmp_path, output_dir=tmp_path / "explicit"
        ).output_dir == (tmp_path / "explicit")

    def test_export_dir_routes_through_paths_accessor(self, monkeypatch, tmp_path):
        """The Windows CSV export target comes from paths.gms_export_dir()."""
        export_dir = tmp_path / "windows_export"
        export_dir.mkdir()
        monkeypatch.delenv("GMS_EXPORT_DIR", raising=False)
        monkeypatch.setattr(paths, "gms_export_dir", lambda: export_dir)

        batch = CaptionBatch(
            week_start="2026-02-23",
            voice_style="voice_first_person",
            captions=[
                PhotoCaption(
                    photo_index=0,
                    main_caption="Test caption",
                    tiktok_caption="Test",
                    hashtags=["test"],
                    journey_slug="test-journey",
                    filename="photo-001.jpg",
                    location="Test",
                ),
            ],
            reviewed=True,
        )
        yaml_path = tmp_path / "captions.yaml"
        batch.save(yaml_path)

        # The count-match gate requires the caption's source file on disk.
        site_root = tmp_path / "site"
        photo_dir = site_root / "static" / "images" / "journeys" / "test-journey"
        photo_dir.mkdir(parents=True)
        (photo_dir / "photo-001.jpg").write_bytes(b"fake photo")

        with patch(
            "gomoveshift.pipeline.photos.orchestrator.PipelineUploader"
        ) as mock_uploader_cls:
            from gomoveshift.pipeline.uploader import UploadManifest

            manifest = UploadManifest(project_name="test")
            manifest.files = {"test-journey_photo-001.jpg": "https://drive.google.com/uc?id=abc123"}
            mock_uploader = MagicMock()
            mock_uploader.upload_photos.return_value = manifest
            mock_uploader.verify_uploads.return_value = VerificationResult(
                verified=["test-journey_photo-001.jpg"], failed=[]
            )
            mock_uploader_cls.return_value = mock_uploader

            with patch("gomoveshift.publishing.photo_run_hooks.record_confirmed_upload") as _book:
                orchestrator = PhotoOrchestrator(
                    site_root=site_root, output_dir=tmp_path, **photo_run_hooks()
                )
                resized = tmp_path / "photo_resized" / "2026-02-23" / "test-journey_photo-001.jpg"
                resized.parent.mkdir(parents=True, exist_ok=True)
                resized.write_bytes(b"fake resized")
                orchestrator.resizer.resize_batch = MagicMock(return_value=[resized])
                orchestrator.scheduler.ledger = MagicMock()

                from gomoveshift.pipeline.photos.scheduler import PhotoSlot

                orchestrator.scheduler.schedule_batch = MagicMock(
                    return_value=[
                        PhotoSlot(date="2026-02-23", time="10:00:00", day_of_week="Monday")
                    ]
                )
                orchestrator._export_to_windows = MagicMock()

                orchestrator.publish_batch(str(yaml_path), force=True)

        orchestrator._export_to_windows.assert_called_once()
        assert orchestrator._export_to_windows.call_args[0][1] == export_dir
