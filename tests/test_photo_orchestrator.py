"""Tests for photo posting orchestrator."""

import builtins
import json
import logging
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline.photos.caption_generator import CaptionBatch, PhotoCaption
from gomoveshift.pipeline.photos.orchestrator import PhotoOrchestrator
from gomoveshift.publishing.photo_run_hooks import (
    next_photo_voice_style,
    photo_run_hooks,
    record_photo_theme_usage,
)
from gomoveshift.pipeline.uploader import VerificationResult


class TestPhotoOrchestratorStatus:
    @pytest.fixture
    def site_root(self, tmp_path):
        """Create minimal Hugo site for testing."""
        journeys_dir = tmp_path / "content" / "journeys"
        journeys_dir.mkdir(parents=True)

        journey = journeys_dir / "test-journey.md"
        journey.write_text(
            textwrap.dedent("""\
            ---
            title: Test Journey
            photo_count: 3
            days:
            - day_number: 1
              date: '2025-01-01'
              summary: Test day
              locations: [TestPlace]
              statistics:
                duration_hours: 2.0
              photos:
              - location: TestPlace
                photos:
                - photo-001.jpg
                - photo-002.jpg
                - photo-003.jpg
            ---
            Content
        """)
        )

        img_dir = tmp_path / "static" / "images" / "journeys" / "test-journey"
        img_dir.mkdir(parents=True)
        for i in range(1, 4):
            (img_dir / f"photo-00{i}.jpg").write_bytes(b"fake")

        return tmp_path

    def test_status_returns_photo_counts(self, site_root, tmp_path):
        orchestrator = PhotoOrchestrator(
            site_root=site_root,
            output_dir=tmp_path / "output",
        )
        # Override selector state path
        orchestrator.selector.state_path = tmp_path / "state.json"

        status = orchestrator.status()
        assert status["total_photos"] == 3
        assert status["posted"] == 0
        assert status["remaining"] == 3
        assert status["journeys"] == 1


class TestPhotoOrchestratorPublish:
    def _make_batch(self, tmp_path, reviewed=False):
        """Helper to create a CaptionBatch and save to YAML."""
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
            reviewed=reviewed,
        )
        yaml_path = tmp_path / "captions.yaml"
        batch.save(yaml_path)
        return yaml_path

    def test_publish_rejects_unreviewed(self, tmp_path):
        yaml_path = self._make_batch(tmp_path, reviewed=False)
        orchestrator = PhotoOrchestrator(site_root=tmp_path / "site", output_dir=tmp_path)

        with pytest.raises(ValueError, match="not yet reviewed"):
            orchestrator.publish_batch(str(yaml_path))

    def _make_site(self, tmp_path):
        """Site checkout holding the batch's photo (the count-match gate
        refuses to publish captions whose source files are missing)."""
        site_root = tmp_path / "site"
        photo_dir = site_root / "static" / "images" / "journeys" / "test-journey"
        photo_dir.mkdir(parents=True)
        (photo_dir / "photo-001.jpg").write_bytes(b"fake photo")
        return site_root

    def test_publish_allows_force(self, tmp_path):
        yaml_path = self._make_batch(tmp_path, reviewed=False)
        site_root = self._make_site(tmp_path)

        # Mock the uploader and its methods
        with patch(
            "gomoveshift.pipeline.photos.orchestrator.PipelineUploader"
        ) as mock_uploader_cls:
            mock_uploader = MagicMock()
            from gomoveshift.pipeline.uploader import UploadManifest

            mock_manifest = UploadManifest(project_name="test")
            mock_manifest.files = {
                "test-journey_photo-001.jpg": "https://drive.google.com/uc?export=download&id=a123"
            }
            mock_uploader.upload_photos.return_value = mock_manifest
            mock_uploader.verify_uploads.return_value = VerificationResult(
                verified=["test-journey_photo-001.jpg"],
                failed=[],
            )
            mock_uploader_cls.return_value = mock_uploader

            with patch("gomoveshift.publishing.photo_run_hooks.record_confirmed_upload") as _book:
                orchestrator = PhotoOrchestrator(
                    site_root=site_root, output_dir=tmp_path, **photo_run_hooks()
                )

                # Mock the resizer to return fake paths (slug-prefixed names)
                resized_path = (
                    tmp_path / "photo_resized" / "2026-02-23" / "test-journey_photo-001.jpg"
                )
                resized_path.parent.mkdir(parents=True, exist_ok=True)
                resized_path.write_bytes(b"fake resized")
                orchestrator.resizer.resize_batch = MagicMock(return_value=[resized_path])

                orchestrator.scheduler.ledger = MagicMock()
                from gomoveshift.pipeline.photos.scheduler import PhotoSlot

                orchestrator.scheduler.schedule_batch = MagicMock(
                    return_value=[
                        PhotoSlot(date="2026-02-23", time="10:00:00", day_of_week="Monday"),
                    ]
                )

                # This should not raise because force=True
                csv_path = orchestrator.publish_batch(str(yaml_path), force=True)
                assert csv_path.exists()

                # Verify resizer was called
                orchestrator.resizer.resize_batch.assert_called_once()

                # Verify uploads were verified
                mock_uploader.verify_uploads.assert_called_once_with(mock_manifest)

    def test_publish_reuploads_failed_verifications(self, tmp_path):
        yaml_path = self._make_batch(tmp_path, reviewed=True)
        site_root = self._make_site(tmp_path)

        with patch(
            "gomoveshift.pipeline.photos.orchestrator.PipelineUploader"
        ) as mock_uploader_cls:
            mock_uploader = MagicMock()
            from gomoveshift.pipeline.uploader import UploadManifest

            mock_manifest = UploadManifest(project_name="test")
            mock_manifest.files = {
                "test-journey_photo-001.jpg": "https://drive.google.com/uc?export=download&id=a123"
            }

            # First upload succeeds, verification finds failure, re-upload fixes
            # it: the uploader relinks whatever it is handed and the re-verify
            # comes back clean. A failure that persists is now a hard fail, so
            # the mock has to model the repair rather than fail forever.
            def _upload(paths, **kwargs):
                for path in paths:
                    mock_manifest.files[Path(path).name] = (
                        "https://drive.google.com/uc?export=download&id=abc123"
                    )
                return mock_manifest

            mock_uploader.upload_photos.side_effect = _upload
            mock_uploader.verify_uploads.side_effect = [
                VerificationResult(verified=[], failed=["test-journey_photo-001.jpg"]),
                VerificationResult(verified=["test-journey_photo-001.jpg"], failed=[]),
            ]
            mock_uploader_cls.return_value = mock_uploader

            with patch("gomoveshift.publishing.photo_run_hooks.record_confirmed_upload") as _book:
                orchestrator = PhotoOrchestrator(
                    site_root=site_root, output_dir=tmp_path, **photo_run_hooks()
                )

                resized_path = (
                    tmp_path / "photo_resized" / "2026-02-23" / "test-journey_photo-001.jpg"
                )
                resized_path.parent.mkdir(parents=True, exist_ok=True)
                resized_path.write_bytes(b"fake resized")
                orchestrator.resizer.resize_batch = MagicMock(return_value=[resized_path])

                orchestrator.scheduler.ledger = MagicMock()
                from gomoveshift.pipeline.photos.scheduler import PhotoSlot

                orchestrator.scheduler.schedule_batch = MagicMock(
                    return_value=[
                        PhotoSlot(date="2026-02-23", time="10:00:00", day_of_week="Monday"),
                    ]
                )

                orchestrator.publish_batch(str(yaml_path))

                # upload_photos called twice: initial + re-upload
                assert mock_uploader.upload_photos.call_count == 2


class TestPhotoOrchestratorEndDateFlow:
    """TDD tests for end_date flowing through the orchestrator."""

    def test_publish_batch_passes_end_date(self, tmp_path):
        """Verify end_date flows through publish_batch to scheduler."""
        batch = CaptionBatch(
            week_start="2026-02-23",
            voice_style="voice_first_person",
            captions=[
                PhotoCaption(
                    photo_index=0,
                    main_caption="Test",
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

        # Create the photo at the expected site path so it's not skipped
        site_root = tmp_path / "site"
        photo_dir = site_root / "static" / "images" / "journeys" / "test-journey"
        photo_dir.mkdir(parents=True, exist_ok=True)
        (photo_dir / "photo-001.jpg").write_bytes(b"fake photo")

        with patch(
            "gomoveshift.pipeline.photos.orchestrator.PipelineUploader"
        ) as mock_uploader_cls:
            mock_uploader = MagicMock()
            from gomoveshift.pipeline.uploader import UploadManifest

            mock_manifest = UploadManifest(project_name="test")
            # Resizer prefixes journey slug: "test-journey_photo-001.jpg"
            mock_manifest.files = {
                "test-journey_photo-001.jpg": "https://example.com/photo.jpg",
            }
            mock_uploader.upload_photos.return_value = mock_manifest
            mock_uploader.verify_uploads.return_value = VerificationResult(
                verified=["test-journey_photo-001.jpg"],
                failed=[],
            )
            mock_uploader_cls.return_value = mock_uploader

            with patch("gomoveshift.publishing.photo_run_hooks.record_confirmed_upload") as _book:
                # Photos resolve from the selector's configured root, not a default.
                orchestrator = PhotoOrchestrator(
                    site_root=site_root, output_dir=tmp_path, **photo_run_hooks()
                )
                resized_path = (
                    tmp_path / "photo_resized" / "2026-02-23" / "test-journey_photo-001.jpg"
                )
                resized_path.parent.mkdir(parents=True, exist_ok=True)
                resized_path.write_bytes(b"fake")
                orchestrator.resizer.resize_batch = MagicMock(return_value=[resized_path])

                from gomoveshift.pipeline.photos.scheduler import PhotoSlot

                orchestrator.scheduler.schedule_batch = MagicMock(
                    return_value=[
                        PhotoSlot(date="2026-02-23", time="10:00:00", day_of_week="Monday"),
                    ]
                )

                orchestrator.publish_batch(
                    str(yaml_path),
                    end_date="2026-03-01",
                )

                # Verify scheduler was called with end_date
                orchestrator.scheduler.schedule_batch.assert_called_once_with(
                    count=1,
                    start_date="2026-02-23",
                    end_date="2026-03-01",
                )

    def test_auto_batch_passes_end_date_to_scheduler(self, tmp_path):
        """Verify end_date flows from auto_batch() to scheduler."""
        with patch.object(PhotoOrchestrator, "generate_batch") as mock_gen:
            mock_batch = CaptionBatch(
                week_start="2026-02-23",
                voice_style="voice_first_person",
                captions=[
                    PhotoCaption(
                        photo_index=0,
                        main_caption="Test",
                        tiktok_caption="Test",
                        hashtags=["test"],
                        journey_slug="test-journey",
                        filename="photo-001.jpg",
                        location="Test",
                    ),
                ],
                reviewed=False,
            )
            mock_gen.return_value = mock_batch

            with patch.object(PhotoOrchestrator, "publish_batch") as mock_pub:
                mock_pub.return_value = tmp_path / "output.csv"
                (tmp_path / "output.csv").write_text("fake")

                orchestrator = PhotoOrchestrator(site_root=tmp_path / "site", output_dir=tmp_path)

                orchestrator.auto_batch("2026-02-23", "2026-03-01")

                # Verify publish_batch was called with end_date
                mock_pub.assert_called_once()
                call_kwargs = mock_pub.call_args
                assert (
                    call_kwargs.kwargs.get("end_date") == "2026-03-01"
                    or (len(call_kwargs.args) > 1 and False)
                    or call_kwargs[1].get("end_date") == "2026-03-01"
                )


class TestPhotoOrchestratorCleanup:
    def test_cleanup_dry_run_default(self, tmp_path):
        orchestrator = PhotoOrchestrator(
            site_root=tmp_path / "site", output_dir=tmp_path / "output"
        )
        plan = orchestrator.cleanup()
        # Empty plan since no artifacts exist
        assert len(plan.files) == 0
        assert len(plan.skipped) == 0

    def test_cleanup_delegates_to_handler(self, tmp_path):
        orchestrator = PhotoOrchestrator(
            site_root=tmp_path / "site", output_dir=tmp_path / "output"
        )
        orchestrator.cleanup_handler = MagicMock()
        orchestrator.cleanup_handler.discover_batches.return_value = ["2026-02-01"]
        mock_plan = MagicMock()
        mock_plan.files = []
        orchestrator.cleanup_handler.plan_cleanup.return_value = mock_plan

        orchestrator.cleanup(min_age_days=7, confirm=False)

        orchestrator.cleanup_handler.discover_batches.assert_called_once()
        orchestrator.cleanup_handler.plan_cleanup.assert_called_once_with(
            ["2026-02-01"],
            min_age_days=7,
        )
        # Should NOT execute when confirm=False
        orchestrator.cleanup_handler.execute_plan.assert_not_called()

    def test_cleanup_executes_with_confirm(self, tmp_path):
        orchestrator = PhotoOrchestrator(
            site_root=tmp_path / "site", output_dir=tmp_path / "output"
        )
        orchestrator.cleanup_handler = MagicMock()
        orchestrator.cleanup_handler.discover_batches.return_value = ["2026-02-01"]
        mock_plan = MagicMock()
        mock_plan.files = [{"path": "/tmp/test", "size": 100, "reason": "test"}]
        orchestrator.cleanup_handler.plan_cleanup.return_value = mock_plan
        orchestrator.cleanup_handler.execute_plan.return_value = 1

        orchestrator.cleanup(confirm=True)

        orchestrator.cleanup_handler.execute_plan.assert_called_once_with(mock_plan)


class TestGenerateHoldRelease:
    """Selection persists photo holds BEFORE captioning; every caption-phase
    exit must either ship the batch or return this run's holds to the pool —
    a leaked hold makes the photo permanently unselectable."""

    @pytest.fixture
    def orch(self, tmp_path):
        journeys_dir = tmp_path / "content" / "journeys"
        journeys_dir.mkdir(parents=True)
        (journeys_dir / "test-journey.md").write_text(
            textwrap.dedent("""\
            ---
            title: Test Journey
            photo_count: 3
            days:
            - day_number: 1
              date: '2025-01-01'
              summary: Test day
              locations: [TestPlace]
              photos:
              - location: TestPlace
                photos:
                - photo-001.jpg
                - photo-002.jpg
                - photo-003.jpg
            ---
        """)
        )
        img_dir = tmp_path / "static" / "images" / "journeys" / "test-journey"
        img_dir.mkdir(parents=True)
        for i in range(1, 4):
            (img_dir / f"photo-00{i}.jpg").write_bytes(b"fake")
        return PhotoOrchestrator(site_root=tmp_path, output_dir=tmp_path / "output")

    def _state(self, orch):
        from gomoveshift.pipeline.photos.selector import PhotoSelectionState

        return PhotoSelectionState.load(orch.selector.state_path)

    def _voice(self):
        from gomoveshift.pipeline.photos.voice_styles import VoiceStyle

        return VoiceStyle.FIRST_PERSON

    def test_preexisting_yaml_raises_before_any_hold(self, orch):
        """The no-clobber guard must fire BEFORE selection persists holds."""
        orch.caption_gen.caption_dir.mkdir(parents=True, exist_ok=True)
        (orch.caption_gen.caption_dir / "2026-03-01_captions.yaml").write_text(
            "reviewed: true\ncaptions: []\n"
        )
        orch.caption_gen._call_llm = MagicMock()

        with pytest.raises(FileExistsError):
            orch.generate_batch(week_start="2026-03-01", count=3, voice_style=self._voice())

        state = self._state(orch)
        assert state.pending_photos == []
        assert state.posted_photos == []
        orch.caption_gen._call_llm.assert_not_called()

    def test_caption_failure_releases_this_runs_holds(self, orch):
        """A dead LLM call must release the just-taken holds and propagate."""
        with patch.object(orch.caption_gen, "generate_batch", side_effect=RuntimeError("LLM dead")):
            with pytest.raises(RuntimeError, match="LLM dead"):
                orch.generate_batch(week_start="2026-03-01", count=3, voice_style=self._voice())

        state = self._state(orch)
        assert state.pending_photos == []
        assert state.posted_photos == []

    def test_dropped_selection_hold_released_rest_kept(self, orch):
        """A selection the caption phase drops (empty metadata) never reaches
        the batch, so publish's release can't see it — free it at generate."""

        def fake_captions(selections, voice_style, week_start=""):
            kept = selections[:-1]  # caption phase drops the last selection
            return CaptionBatch(
                week_start=week_start,
                voice_style=voice_style.value,
                captions=[
                    PhotoCaption(
                        photo_index=i,
                        main_caption="c",
                        tiktok_caption="c",
                        hashtags=[],
                        journey_slug=s.journey_slug,
                        filename=s.filename,
                    )
                    for i, s in enumerate(kept)
                ],
            )

        with patch.object(orch.caption_gen, "generate_batch", side_effect=fake_captions):
            batch = orch.generate_batch(week_start="2026-03-01", count=3, voice_style=self._voice())

        kept = {f"{c.journey_slug}/{c.filename}" for c in batch.captions}
        assert len(kept) == 2
        state = self._state(orch)
        assert set(state.pending_photos) == kept
        assert set(state.posted_photos) == kept


class TestReleaseBatch:
    """``photos release <week>`` — operator recovery for leaked holds."""

    def _seed(self, tmp_path, published):
        orch = PhotoOrchestrator(site_root=tmp_path / "site", output_dir=tmp_path / "output")
        keys = ["test-journey/photo-001.jpg", "test-journey/photo-002.jpg"]
        from gomoveshift.pipeline.photos.selector import PhotoSelectionState

        PhotoSelectionState(posted_photos=list(keys), pending_photos=list(keys)).save(
            orch.selector.state_path
        )
        batch = CaptionBatch(
            week_start="2026-03-01",
            voice_style="voice_first_person",
            captions=[
                PhotoCaption(
                    photo_index=i,
                    main_caption="c",
                    tiktok_caption="c",
                    hashtags=[],
                    journey_slug="test-journey",
                    filename=f"photo-00{i + 1}.jpg",
                )
                for i in range(2)
            ],
            published=published,
        )
        batch.save(orch.caption_gen.caption_dir / "2026-03-01_captions.yaml")
        return orch, keys

    def _state(self, orch):
        from gomoveshift.pipeline.photos.selector import PhotoSelectionState

        return PhotoSelectionState.load(orch.selector.state_path)

    def test_release_frees_unpublished_batch(self, tmp_path):
        orch, keys = self._seed(tmp_path, published=False)
        released = orch.release_batch("2026-03-01")
        assert released == sorted(keys)
        state = self._state(orch)
        assert state.pending_photos == []
        assert state.posted_photos == []

    def test_release_refuses_published_batch(self, tmp_path):
        orch, keys = self._seed(tmp_path, published=True)
        with pytest.raises(ValueError, match="published"):
            orch.release_batch("2026-03-01")
        state = self._state(orch)
        assert set(state.pending_photos) == set(keys)  # untouched

    def test_release_missing_yaml_raises(self, tmp_path):
        orch = PhotoOrchestrator(site_root=tmp_path / "site", output_dir=tmp_path / "output")
        with pytest.raises(FileNotFoundError):
            orch.release_batch("2026-01-01")

    def test_release_subcommand_parses(self):
        from gomoveshift.cli.parser import _build_parser

        args = _build_parser().parse_args(["photos", "release", "2026-03-01"])
        assert args.photos_command == "release"
        assert args.week == "2026-03-01"


class TestThemeRotatorHardening:
    """The SEO-manager ThemeRotator is optional and lives in a sibling service;
    a RUNTIME failure from it must never crash caption generation, and a failed
    usage-record must be visible (WARNING), not swallowed at debug."""

    def test_runtime_rotator_error_defaults_without_crashing(self, caplog):
        """A non-ImportError failure (e.g. corrupt rotation state) must fall back
        to FIRST_PERSON with a WARNING, not propagate into generate_batch."""
        from gomoveshift.pipeline.photos.voice_styles import VoiceStyle

        with patch(
            "seo_manager.analytics.theme_rotator.ThemeRotator",
            side_effect=RuntimeError("rotation state corrupt"),
        ):
            with caplog.at_level(logging.WARNING):
                style = next_photo_voice_style()

        assert style == VoiceStyle.FIRST_PERSON
        assert any(r.levelno >= logging.WARNING for r in caplog.records)

    def test_import_error_still_defaults(self):
        """Regression: a missing seo_manager still defaults to FIRST_PERSON."""
        from gomoveshift.pipeline.photos.voice_styles import VoiceStyle

        real_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name == "seo_manager.analytics.theme_rotator":
                raise ImportError("no seo_manager")
            return real_import(name, *a, **k)

        with patch("builtins.__import__", side_effect=fake_import):
            style = next_photo_voice_style()

        assert style == VoiceStyle.FIRST_PERSON

    def test_record_theme_usage_failure_logs_warning(self, caplog):
        """A failed usage-record must surface at WARNING (was swallowed at debug)
        and must never raise into the publish flow."""
        with patch(
            "seo_manager.analytics.theme_rotator.ThemeRotator",
            side_effect=RuntimeError("disk full"),
        ):
            with caplog.at_level(logging.DEBUG):
                record_photo_theme_usage("voice_first_person", "2025-01-06")  # must not raise

        assert any(
            r.levelno >= logging.WARNING and "theme" in r.message.lower() for r in caplog.records
        ), "usage-record failure must be logged at WARNING, not swallowed at debug"


class TestPhotoOrchestratorLedgerBooking:
    """Booking must survive a crash between the published-save and the
    ledger write, and the unbooked-marker sidecar must never be corrupted."""

    def _make_batch(self, tmp_path, *, published=False, published_slots=None):
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
            published=published,
            published_slots=published_slots or [],
        )
        yaml_path = tmp_path / "captions.yaml"
        batch.save(yaml_path)
        return yaml_path

    def _make_site(self, tmp_path):
        site_root = tmp_path / "site"
        photo_dir = site_root / "static" / "images" / "journeys" / "test-journey"
        photo_dir.mkdir(parents=True)
        (photo_dir / "photo-001.jpg").write_bytes(b"fake photo")
        return site_root

    def test_republish_after_crash_before_booking_still_books_slots(self, tmp_path, monkeypatch):
        """Simulate a crash between the ``batch.save`` (published=True,
        published_slots persisted) and the ledger booking: on the NEXT
        invocation, republish=True must not skip booking forever. Checked
        against the real ledger file, not a mock."""
        ledger_path = tmp_path / "ledger.json"
        monkeypatch.setattr(
            "gomoveshift.pipeline.content_ledger._default_ledger_path", lambda: ledger_path
        )

        slot_dict = {"date": "2026-02-23", "time": "10:00:00", "day_of_week": "Monday"}
        yaml_path = self._make_batch(tmp_path, published=True, published_slots=[slot_dict])
        site_root = self._make_site(tmp_path)

        with patch(
            "gomoveshift.pipeline.photos.orchestrator.PipelineUploader"
        ) as mock_uploader_cls:
            mock_uploader = MagicMock()
            from gomoveshift.pipeline.uploader import UploadManifest

            mock_manifest = UploadManifest(project_name="test")
            mock_manifest.files = {
                "test-journey_photo-001.jpg": "https://drive.google.com/uc?export=download&id=a123"
            }
            mock_uploader.upload_photos.return_value = mock_manifest
            mock_uploader.verify_uploads.return_value = VerificationResult(
                verified=["test-journey_photo-001.jpg"], failed=[]
            )
            mock_uploader_cls.return_value = mock_uploader

            orchestrator = PhotoOrchestrator(
                site_root=site_root, output_dir=tmp_path, **photo_run_hooks()
            )

            resized_path = tmp_path / "photo_resized" / "2026-02-23" / "test-journey_photo-001.jpg"
            resized_path.parent.mkdir(parents=True, exist_ok=True)
            resized_path.write_bytes(b"fake resized")
            orchestrator.resizer.resize_batch = MagicMock(return_value=[resized_path])

            orchestrator.publish_batch(str(yaml_path))

        from gomoveshift.pipeline.content_ledger import ContentLedger

        ledger = ContentLedger.load(ledger_path)
        assert any(
            e.date == "2026-02-23" and e.time == "10:00:00" and e.content_type == "photo"
            for e in ledger.entries
        ), "republish must still book the persisted slots into the ledger"

    def test_book_slots_is_idempotent(self, tmp_path, monkeypatch):
        """Booking the same slots twice must yield ONE ledger entry per slot,
        not a duplicate — this is what makes it safe to book unconditionally
        on every publish call, including a republish."""
        ledger_path = tmp_path / "ledger.json"
        monkeypatch.setattr(
            "gomoveshift.pipeline.content_ledger._default_ledger_path", lambda: ledger_path
        )

        from gomoveshift.pipeline.content_ledger import ContentLedger
        from gomoveshift.pipeline.photos.caption_generator import PhotoCaption
        from gomoveshift.pipeline.photos.scheduler import PhotoSlot

        orchestrator = PhotoOrchestrator(output_dir=tmp_path, **photo_run_hooks())
        slots = [PhotoSlot(date="2026-02-23", time="10:00:00", day_of_week="Monday")]
        caption = PhotoCaption(
            photo_index=0,
            main_caption="Test caption",
            tiktok_caption="Test",
            hashtags=["test"],
            journey_slug="test-journey",
            filename="photo-001.jpg",
            location="Test",
        )

        orchestrator._book_slots(slots, [caption], "csv.csv")
        orchestrator._book_slots(slots, [caption], "csv.csv")

        ledger = ContentLedger.load(ledger_path)
        matches = [e for e in ledger.entries if e.date == "2026-02-23" and e.time == "10:00:00"]
        assert len(matches) == 1

    def test_mark_ledger_unbooked_marker_write_is_atomic(self, tmp_path, monkeypatch):
        """A crash mid-write must never leave the marker truncated/corrupt —
        a partial marker reads back as valid-but-empty, silently erasing the
        double-book warning."""
        from gomoveshift.pipeline.photos.preflight import LEDGER_UNBOOKED_MARKER
        from gomoveshift.pipeline.photos.scheduler import PhotoSlot

        orchestrator = PhotoOrchestrator(site_root=tmp_path / "site", output_dir=tmp_path)
        marker = tmp_path / LEDGER_UNBOOKED_MARKER
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"slots": [{"date": "2026-01-01", "time": "10:00:00"}]}))

        def _boom(*_a, **_kw):
            raise OSError("disk full mid-write")

        monkeypatch.setattr("gomoveshift.pipeline._atomic_write.os.replace", _boom)

        slot = PhotoSlot(date="2026-01-02", time="11:00:00", day_of_week="Friday")
        orchestrator._mark_ledger_unbooked([slot], "csv.csv")

        data = json.loads(marker.read_text())
        assert data["slots"] == [{"date": "2026-01-01", "time": "10:00:00"}]

    def test_mark_ledger_unbooked_accumulates_across_failures(self, tmp_path):
        """Two successive booking failures must leave BOTH sets of slots in
        the marker — no failed slot lost."""
        from gomoveshift.pipeline.photos.preflight import LEDGER_UNBOOKED_MARKER
        from gomoveshift.pipeline.photos.scheduler import PhotoSlot

        orchestrator = PhotoOrchestrator(site_root=tmp_path / "site", output_dir=tmp_path)
        slot1 = PhotoSlot(date="2026-01-01", time="10:00:00", day_of_week="Thursday")
        slot2 = PhotoSlot(date="2026-01-02", time="11:00:00", day_of_week="Friday")

        orchestrator._mark_ledger_unbooked([slot1], "csv1.csv")
        orchestrator._mark_ledger_unbooked([slot2], "csv2.csv")

        marker = tmp_path / LEDGER_UNBOOKED_MARKER
        data = json.loads(marker.read_text())
        dates = {(s["date"], s["time"]) for s in data["slots"]}
        assert dates == {("2026-01-01", "10:00:00"), ("2026-01-02", "11:00:00")}


class TestSiteRootNotConfigured:
    """``photos.site_root`` has no default: an orchestrator built without one must
    still construct, run cleanup, and name the missing key where the site is needed."""

    @pytest.fixture
    def unconfigured(self, monkeypatch):
        import gomoveshift.config as config_module
        from gomoveshift.config import Config, PhotosConfig

        monkeypatch.setattr(
            config_module, "get_config", lambda *a, **k: Config(photos=PhotosConfig(site_root=""))
        )

    def test_construction_and_cleanup_need_no_site(self, unconfigured, tmp_path):
        orchestrator = PhotoOrchestrator(output_dir=tmp_path)
        plan = orchestrator.cleanup(min_age_days=14, confirm=False)
        assert plan.files == []

    def test_selector_access_names_the_missing_key(self, unconfigured, tmp_path):
        from gomoveshift.config import ConfigError

        orchestrator = PhotoOrchestrator(output_dir=tmp_path)
        with pytest.raises(ConfigError, match="photos.site_root"):
            orchestrator.selector

    def test_release_batch_needs_no_site(self, unconfigured, tmp_path):
        """Releasing holds only rewrites the state file, so it works with the site unset."""
        from gomoveshift.pipeline.photos.selector import PhotoSelectionState

        keys = ["journey-a/photo-001.jpg", "journey-a/photo-002.jpg"]
        state_path = tmp_path / ".photo_selection_state.json"
        PhotoSelectionState(posted_photos=list(keys), pending_photos=list(keys)).save(state_path)
        orchestrator = PhotoOrchestrator(output_dir=tmp_path)
        caption_dir = tmp_path / "photo_captions"
        CaptionBatch(
            week_start="2026-01-05",
            voice_style="first_person",
            captions=[
                PhotoCaption(
                    photo_index=i,
                    main_caption="x",
                    tiktok_caption="x",
                    hashtags=[],
                    journey_slug="journey-a",
                    filename=f"photo-00{i}.jpg",
                )
                for i in (1, 2)
            ],
        ).save(caption_dir / "2026-01-05_captions.yaml")

        released = orchestrator.release_batch("2026-01-05")

        assert sorted(released) == keys
        state = PhotoSelectionState.load(state_path)
        assert state.pending_photos == []
        assert state.posted_photos == []
        assert orchestrator._selector is None  # the site was never needed

    def test_failed_publish_reraises_and_releases_holds(self, unconfigured, tmp_path):
        """With the site unset, publish fails at the first selector touch with the
        original ConfigError, and the batch's holds still go back to the pool."""
        from gomoveshift.config import ConfigError
        from gomoveshift.pipeline.photos.selector import PhotoSelectionState

        keys = ["journey-a/photo-001.jpg", "journey-a/photo-002.jpg"]
        state_path = tmp_path / ".photo_selection_state.json"
        PhotoSelectionState(posted_photos=list(keys), pending_photos=list(keys)).save(state_path)
        caption_yaml = tmp_path / "photo_captions" / "2026-01-05_captions.yaml"
        CaptionBatch(
            week_start="2026-01-05",
            voice_style="first_person",
            captions=[
                PhotoCaption(
                    photo_index=i,
                    main_caption="x",
                    tiktok_caption="x",
                    hashtags=[],
                    journey_slug="journey-a",
                    filename=f"photo-00{i}.jpg",
                )
                for i in (1, 2)
            ],
            reviewed=True,
        ).save(caption_yaml)
        orchestrator = PhotoOrchestrator(output_dir=tmp_path)

        with pytest.raises(ConfigError, match="photos.site_root"):
            orchestrator.publish_batch(str(caption_yaml))

        state = PhotoSelectionState.load(state_path)
        assert state.pending_photos == []
        assert state.posted_photos == []
        assert orchestrator._selector is None

    def test_state_path_needs_no_site(self, unconfigured, tmp_path):
        orchestrator = PhotoOrchestrator(output_dir=tmp_path)
        assert orchestrator.state_path == tmp_path / ".photo_selection_state.json"
        assert orchestrator._selector is None  # the site was never needed

    def test_explicit_site_root_still_overrides(self, unconfigured, tmp_path):
        orchestrator = PhotoOrchestrator(site_root=tmp_path / "site", output_dir=tmp_path)
        assert orchestrator.selector.site_root == tmp_path / "site"
        assert orchestrator.selector.state_path == tmp_path / ".photo_selection_state.json"
