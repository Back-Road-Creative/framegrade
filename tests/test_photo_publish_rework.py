"""Photo publish ordering and idempotency.

Every step that can fail or drop a photo must run BEFORE the irreversible
Drive upload, and re-running a published batch must replay it, not shift it.
"""

import csv
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline.photos.caption_generator import CaptionBatch, PhotoCaption
from gomoveshift.pipeline.photos.orchestrator import PhotoOrchestrator, PhotoPublishShortfallError
from gomoveshift.pipeline.photos.scheduler import PhotoSlot
from gomoveshift.pipeline.uploader import UploadManifest, VerificationResult

WEEK = "2026-02-23"
SLUG = "test-journey"


def _site(tmp_path, n=2):
    """Minimal Hugo checkout with one journey of ``n`` photos."""
    root = tmp_path / "site"
    (root / "content" / "journeys").mkdir(parents=True)
    images = root / "static" / "images" / "journeys" / SLUG
    images.mkdir(parents=True)
    names = [f"photo-{i:03d}.jpg" for i in range(1, n + 1)]
    for name in names:
        (images / name).write_bytes(b"x")
    (root / "content" / "journeys" / f"{SLUG}.md").write_text(
        "---\ntitle: Test Journey\ndays:\n- day_number: 1\n  date: '2025-01-01'\n"
        "  summary: Test day\n  locations: [TestPlace]\n  photos:\n"
        f"  - location: TestPlace\n    photos: [{', '.join(names)}]\n---\n"
    )
    return root


def _captions(tmp_path, n=2):
    batch = CaptionBatch(
        week_start=WEEK,
        voice_style="voice_first_person",
        captions=[
            PhotoCaption(i, f"Caption {i}", "T", ["t"], SLUG, f"photo-{i + 1:03d}.jpg", "TestPlace")
            for i in range(n)
        ],
        reviewed=True,
    )
    batch.save(tmp_path / "captions.yaml")
    return str(tmp_path / "captions.yaml")


def _slots(*dates):
    return [PhotoSlot(date=d, time="10:00:00", day_of_week="Monday") for d in dates]


def _dates(csv_path):
    with open(csv_path, encoding="utf-8") as f:
        return [row["Date"] for row in csv.DictReader(f)]


def _uploaded(env):
    return [Path(p).name for p in env.uploader.upload_photos.call_args.args[0]]


def _state(env):
    """Persisted selection state as the next run would read it."""
    from gomoveshift.pipeline.photos.selector import PhotoSelectionState

    return PhotoSelectionState.load(env.orch.selector.state_path)


def _keys(*indices):
    return sorted(f"{SLUG}/photo-{i:03d}.jpg" for i in indices)


def _selected(selections):
    return sorted(f"{s.journey_slug}/{s.filename}" for s in selections)


@pytest.fixture
def env(tmp_path, request):
    site = _site(tmp_path, getattr(request, "param", 2))
    with (
        patch("gomoveshift.pipeline.photos.orchestrator.PipelineUploader") as up_cls,
        patch("gomoveshift.pipeline.photos.orchestrator.record_confirmed_upload") as led_cls,
    ):

        def _upload(paths, **kwargs):
            manifest = UploadManifest(project_name="test")
            manifest.files = {Path(p).name: f"https://example.com/{Path(p).name}" for p in paths}
            return manifest

        uploader = MagicMock()
        uploader.upload_photos.side_effect = _upload
        uploader.verify_uploads.return_value = VerificationResult(verified=[], failed=[])
        up_cls.return_value = uploader
        orch = PhotoOrchestrator(site_root=site, output_dir=tmp_path)
        (tmp_path / "resized").mkdir()

        def _resize(paths, week, journey_slugs=None):
            out = [
                tmp_path / "resized" / f"{slug}_{Path(p).name}"
                for p, slug in zip(paths, journey_slugs or [SLUG] * len(paths))
            ]
            for dest in out:
                dest.write_bytes(b"x")
            return out

        orch.resizer.resize_batch = MagicMock(side_effect=_resize)
        yield SimpleNamespace(orch=orch, uploader=uploader, book=led_cls, site=site, tmp=tmp_path)


def test_republish_reuses_slots_and_books_no_second_week(env):
    """A re-run must not shift dates or book a second week of slots.

    Booking now runs on every ``publish_batch`` call, not just the first —
    ``ContentLedger.add`` keys entries on ``(date, time, content_type)``, so
    re-booking the same two slots on a republish is how a crash between the
    durable ``published=True`` save and the ledger write self-heals; the old
    ``if not republish`` guard made that crash window permanently
    unrecoverable. What this test actually protects — no second week ever
    gets booked — is asserted directly against every booking call's slots,
    rather than via a call-count proxy that no longer holds.
    """
    yaml_path = _captions(env.tmp)
    env.orch.scheduler.schedule_batch = MagicMock(
        side_effect=[_slots("2026-02-23", "2026-02-24"), _slots("2026-03-02", "2026-03-03")]
    )

    first = _dates(env.orch.publish_batch(yaml_path))
    second = _dates(env.orch.publish_batch(yaml_path))

    assert first == ["2026-02-23", "2026-02-24"]
    assert second == first
    assert env.orch.scheduler.schedule_batch.call_count == 1

    # Every booking call must carry exactly the originally-scheduled two
    # slots, and the union across ALL calls must still be just those two —
    # literally "books no second week", strictly stronger than a call count.
    all_dates = set()
    for call in env.book.call_args_list:
        entries = call.args[0]
        assert {e.date for e in entries} == {"2026-02-23", "2026-02-24"}
        all_dates.update(e.date for e in entries)
    assert all_dates == {"2026-02-23", "2026-02-24"}


def test_slot_shortfall_truncates_before_upload(env):
    """A caption dropped for want of a slot must never reach Drive."""
    env.orch.scheduler.schedule_batch = MagicMock(return_value=_slots("2026-02-23"))

    env.orch.publish_batch(_captions(env.tmp))

    assert _uploaded(env) == [f"{SLUG}_photo-001.jpg"]


def test_photos_resolve_from_selector_site_root(env):
    """Publish reads the tree the selector selected from."""
    env.orch.scheduler.schedule_batch = MagicMock(return_value=_slots("2026-02-23", "2026-02-24"))

    env.orch.publish_batch(_captions(env.tmp))

    assert _uploaded(env) == [f"{SLUG}_photo-001.jpg", f"{SLUG}_photo-002.jpg"]


class TestConsumeOnPublishSuccess:
    """A photo is burned only once it has actually shipped."""

    def test_selection_holds_photos_pending(self, env):
        """Selection places a hold; it does not confirm consumption."""
        env.orch.selector.select_batch(count=2)

        assert _keys(1, 2) == sorted(_state(env).pending_photos)
        assert env.orch.status()["pending"] == 2

    @pytest.mark.parametrize("env", [4], indirect=True)
    def test_mid_publish_failure_leaves_photos_reselectable(self, env):
        """A raise between the upload and the posted-mark must not burn photos."""
        env.orch.scheduler.schedule_batch = MagicMock(
            return_value=_slots("2026-02-23", "2026-02-24")
        )
        env.orch.selector.select_batch(count=2)
        env.orch.csv_gen.generate = MagicMock(side_effect=RuntimeError("CSV blew up"))

        with pytest.raises(RuntimeError, match="CSV blew up"):
            env.orch.publish_batch(_captions(env.tmp))

        # Not merely absent from posted — actually offered again by the next run.
        assert _keys(1, 2) == _selected(env.orch.selector.select_batch(count=2))

    @pytest.mark.parametrize("env", [4], indirect=True)
    def test_slot_truncated_photo_is_not_consumed(self, env):
        """The photo dropped for want of a slot stays available."""
        env.orch.scheduler.schedule_batch = MagicMock(return_value=_slots("2026-02-23"))
        env.orch.selector.select_batch(count=2)

        env.orch.publish_batch(_captions(env.tmp))

        assert _keys(1) == sorted(_state(env).posted_photos)
        assert _keys(2) == _selected(env.orch.selector.select_batch(count=1))

    def test_missing_source_file_is_not_consumed(self, env):
        """A file the checkout does not have must not be silently burned."""
        (env.site / "static" / "images" / "journeys" / SLUG / "photo-002.jpg").unlink()

        env.orch.selector.select_batch(count=2, allow_shortfall=True)

        assert f"{SLUG}/photo-002.jpg" not in _state(env).posted_photos

    @pytest.mark.parametrize("env", [4], indirect=True)
    def test_unreviewed_batch_keeps_its_holds(self, env):
        """Failing the review gate is an outstanding batch, not a failed publish."""
        env.orch.selector.select_batch(count=2)
        batch = CaptionBatch.load(Path(_captions(env.tmp)))
        batch.reviewed = False
        batch.save(env.tmp / "captions.yaml")

        with pytest.raises(ValueError, match="not yet reviewed"):
            env.orch.publish_batch(str(env.tmp / "captions.yaml"))

        assert _keys(1, 2) == sorted(_state(env).pending_photos)
        assert _keys(3, 4) == _selected(env.orch.selector.select_batch(count=2))

    @pytest.mark.parametrize("env", [4], indirect=True)
    def test_release_pending_recovers_a_crashed_run(self, env):
        """Holds stranded by a hard crash are recoverable, never permanent."""
        env.orch.selector.select_batch(count=2)

        assert _keys(1, 2) == env.orch.selector.release_pending()

        assert _state(env).pending_photos == []
        assert _keys(1, 2) == _selected(env.orch.selector.select_batch(count=2))

    @pytest.mark.parametrize("env", [4], indirect=True)
    def test_successful_publish_consumes_and_does_not_reoffer(self, env):
        """The other half: a shipped photo must never come back around."""
        env.orch.scheduler.schedule_batch = MagicMock(
            return_value=_slots("2026-02-23", "2026-02-24")
        )
        env.orch.selector.select_batch(count=2)

        env.orch.publish_batch(_captions(env.tmp))

        assert _keys(1, 2) == sorted(_state(env).posted_photos)
        assert _state(env).pending_photos == []
        assert _keys(3, 4) == _selected(env.orch.selector.select_batch(count=2))


class TestPublishCountMatch:
    """The photo count-match gate: fewer publishable photos than the batch
    captioned (or zero) hard-fails BEFORE any irreversible step — resize,
    upload, CSV, ledger booking, published marker, export."""

    def test_missing_file_hard_fails_before_any_side_effect(self, env):
        (env.site / "static" / "images" / "journeys" / SLUG / "photo-002.jpg").unlink()
        env.orch.scheduler.schedule_batch = MagicMock(
            return_value=_slots("2026-02-23", "2026-02-24")
        )
        yaml_path = _captions(env.tmp)

        with pytest.raises(PhotoPublishShortfallError, match="1 of 2"):
            env.orch.publish_batch(yaml_path)

        env.orch.resizer.resize_batch.assert_not_called()
        env.uploader.upload_photos.assert_not_called()
        env.book.assert_not_called()
        assert not list((env.tmp / "photo_metricool").glob("*.csv"))
        assert not CaptionBatch.load(Path(yaml_path)).published

    def test_zero_publishable_photos_hard_fails(self, env):
        images = env.site / "static" / "images" / "journeys" / SLUG
        for photo in images.glob("*.jpg"):
            photo.unlink()
        env.orch.scheduler.schedule_batch = MagicMock(return_value=[])

        with pytest.raises(PhotoPublishShortfallError):
            env.orch.publish_batch(_captions(env.tmp))

        env.uploader.upload_photos.assert_not_called()
        env.book.assert_not_called()
        assert not list((env.tmp / "photo_metricool").glob("*.csv"))

    def test_zero_slots_hard_fails_before_upload(self, env):
        """Truncated-to-nothing must never 'publish' a header-only CSV."""
        env.orch.scheduler.schedule_batch = MagicMock(return_value=[])

        with pytest.raises(PhotoPublishShortfallError):
            env.orch.publish_batch(_captions(env.tmp))

        env.uploader.upload_photos.assert_not_called()
        env.book.assert_not_called()
        assert not list((env.tmp / "photo_metricool").glob("*.csv"))


class TestBookingMarkerOrdering:
    """No crash point may leave a ledger booking without its published marker:
    the marker + slots persist BEFORE the booking, so a crash mid-booking never
    strands ghost bookings that a re-publish would reschedule around."""

    def test_crash_at_booking_leaves_marker_already_persisted(self, env):
        yaml_path = _captions(env.tmp)
        env.orch.scheduler.schedule_batch = MagicMock(
            return_value=_slots("2026-02-23", "2026-02-24")
        )
        env.book.side_effect = RuntimeError("crash mid-booking")

        with pytest.raises(RuntimeError, match="crash mid-booking"):
            env.orch.publish_batch(yaml_path)

        saved = CaptionBatch.load(Path(yaml_path))
        assert saved.published, "published marker must hit disk before the booking crash point"
        assert [s["date"] for s in saved.published_slots] == ["2026-02-23", "2026-02-24"]

    def test_republish_after_booking_crash_replays_same_week(self, env):
        yaml_path = _captions(env.tmp)
        env.orch.scheduler.schedule_batch = MagicMock(
            side_effect=[_slots("2026-02-23", "2026-02-24"), _slots("2026-03-02", "2026-03-03")]
        )
        env.book.side_effect = RuntimeError("crash mid-booking")
        with pytest.raises(RuntimeError, match="crash mid-booking"):
            env.orch.publish_batch(yaml_path)

        env.book.side_effect = None
        second = _dates(env.orch.publish_batch(yaml_path))

        # Same week replayed, not a shifted duplicate scheduled around ghosts.
        assert second == ["2026-02-23", "2026-02-24"]
        assert env.orch.scheduler.schedule_batch.call_count == 1
