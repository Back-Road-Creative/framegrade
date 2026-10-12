"""Photo posting orchestrator — two-phase generate/publish workflow.

Phase 1 (generate):
  Select photos → get voice style from ThemeRotator → LLM captions → cache YAML

Phase 2 (publish):
  Upload to Drive → schedule → Metricool CSV → update ledger + record theme usage
"""

import contextlib
import json
import logging
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from ... import paths as _paths
from .. import fs_safety
from .._atomic_write import atomic_write_json
from ..content_ledger import ContentEntry, record_confirmed_upload
from ..orchestrator_lock import _acquire_run_lock, _release_run_lock
from ..uploader import PipelineUploader
from .caption_generator import CaptionBatch, CaptionGenerator
from .cleanup import PhotoCleanup, PhotoCleanupPlan
from .csv_generator import PhotoCSVGenerator, PhotoPost
from .cull import CullFloors
from .preflight import LEDGER_UNBOOKED_MARKER, preflight_photo_run
from .resizer import PhotoResizer
from .scheduler import PhotoScheduler, PhotoSlot
from .selector import IMAGES_BASE, PhotoSelector, release_pending_holds
from .voice_styles import VoiceStyle

logger = logging.getLogger(__name__)

# Run-lock sidecar for photo runs, under the photo output dir. Deliberately a
# DIFFERENT name from the drive lock so a running drive never blocks a photo
# batch — they contend only on the ledger, which carries its own lock.
PHOTO_RUN_LOCK_NAME = "photo_run.lock"


class CaptionsNotReviewedError(ValueError):
    """Batch has not passed the human review gate. ValueError for compatibility."""


class PhotoUploadVerificationError(RuntimeError):
    """Drive links could not be verified even after a re-upload."""


class PhotoPublishShortfallError(RuntimeError):
    """Fewer publishable photos than the batch captioned, or zero — the photo
    mirror of the shorts-shortfall contract: never quietly ship a thin week."""


def describe_scheduler_export(csv_path) -> str:
    """One line for both photo CLI routes, so neither can invent a receipt.

    The publish route printed "Published!" while the identical auto route
    printed only the CSV path -- the same terminal state described two ways,
    one of them false. Single-sourcing it here is what keeps a print statement
    from being the place publication gets asserted.
    """
    return (
        f"Exported for scheduling — Metricool CSV: {csv_path}\n"
        "No platform receipt yet: Metricool posts on its own schedule, and "
        "nothing here confirms a post went live."
    )


class PhotoOrchestrator:
    """Orchestrates the two-phase photo posting workflow."""

    def __init__(
        self,
        site_root: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        social_aspect: Optional[str] = None,
        cull_floors: Optional[CullFloors] = None,
    ):
        self.output_dir = output_dir or _paths.gms_photo_output_dir()
        # The selector reads the Hugo site checkout (``photos.site_root``, which has no
        # default), so it is built on first use: cleanup needs no site, and an
        # unconfigured site surfaces as a ConfigError naming the key at the step that
        # needs it rather than when the orchestrator is constructed.
        self._selector_args = {
            "site_root": site_root,
            "state_path": self.output_dir / ".photo_selection_state.json",
            "cull_floors": cull_floors,
        }
        self._selector: Optional[PhotoSelector] = None
        self.caption_gen = CaptionGenerator(
            caption_dir=self.output_dir / "photo_captions",
        )
        self.scheduler = PhotoScheduler()
        self.csv_gen = PhotoCSVGenerator()
        # None = take photos.social_aspect ("" = the source's own aspect).
        self.resizer = PhotoResizer(
            output_base=self.output_dir / "photo_resized", social_aspect=social_aspect
        )
        self.cleanup_handler = PhotoCleanup(
            caption_dir=self.output_dir / "photo_captions",
            metricool_dir=self.output_dir / "photo_metricool",
            resized_dir=self.output_dir / "photo_resized",
        )

    @property
    def selector(self) -> PhotoSelector:
        """The photo selector, built on first access.

        Raises ``ConfigError`` naming ``photos.site_root`` when no site checkout is
        configured and none was passed to the constructor.
        """
        if self._selector is None:
            self._selector = PhotoSelector(**self._selector_args)
        return self._selector

    @selector.setter
    def selector(self, value: PhotoSelector) -> None:
        self._selector = value

    @property
    def state_path(self) -> Path:
        """The selection-state file, known without building the selector (no site needed)."""
        return self._selector_args["state_path"]

    def _release_pending(self, photo_keys) -> list[str]:
        """Release photo holds without needing the site checkout.

        Releasing holds only rewrites the selection-state file, so it goes through
        the state path directly unless a selector has already been built (or
        injected), in which case that selector does it.
        """
        if self._selector is not None:
            return self._selector.release_pending(photo_keys)
        return release_pending_holds(self._selector_args["state_path"], photo_keys)

    @contextlib.contextmanager
    def _run_lock(self):
        """Hold the photo-run lock for the photo output dir.

        Two concurrent ``gomoveshift photos`` runs would otherwise interleave
        over the same selection state and caption/CSV dirs. Reuses the drive
        orchestrator's lock machinery (PID-probed stale recovery included) on
        its own sidecar, so a live drive run never blocks a photo batch.
        Reentrant per instance: ``auto_batch`` chains generate + publish, and a
        nested acquire on a second fd would self-deadlock into a bogus
        ConcurrentRunError.
        """
        if getattr(self, "_run_lock_depth", 0):
            self._run_lock_depth += 1
            try:
                yield
            finally:
                self._run_lock_depth -= 1
            return
        fd, lock_path = _acquire_run_lock(
            self.output_dir, self.output_dir, lock_name=PHOTO_RUN_LOCK_NAME
        )
        self._run_lock_depth = 1
        try:
            yield
        finally:
            self._run_lock_depth = 0
            _release_run_lock(fd, lock_path)

    def generate_batch(
        self,
        week_start: str = "",
        count: int = 7,
        voice_style: Optional[VoiceStyle] = None,
        filter_faces: bool = False,
        allow_shortfall: bool = False,
        overwrite: bool = False,
    ) -> CaptionBatch:
        """Phase 1 under the photo-run lock (see :meth:`_run_lock`)."""
        with self._run_lock():
            return self._generate_batch(
                week_start, count, voice_style, filter_faces, allow_shortfall, overwrite
            )

    def _generate_batch(
        self,
        week_start: str = "",
        count: int = 7,
        voice_style: Optional[VoiceStyle] = None,
        filter_faces: bool = False,
        allow_shortfall: bool = False,
        overwrite: bool = False,
    ) -> CaptionBatch:
        """Phase 1: Select photos, generate captions, cache YAML.

        Args:
            week_start: Week label (YYYY-MM-DD), defaults to today
            count: Number of photos to select
            voice_style: Override voice style (default: from ThemeRotator)
            filter_faces: Skip photos with detected human faces
            allow_shortfall: Accept fewer than ``count`` photos instead of
                hard-failing (operator escape for an exhausted pool)
            overwrite: Move an existing captions YAML aside so the week can be
                regenerated (operator escape for the no-clobber guard)

        Returns:
            CaptionBatch with reviewed=False
        """
        if not week_start:
            week_start = datetime.now().strftime("%Y-%m-%d")

        if overwrite:
            self._archive_existing_captions(week_start)

        # No-clobber guard BEFORE selection: the caption phase would refuse a
        # pre-existing review YAML anyway, but only after selection had
        # persisted its holds — leaking the whole batch as permanently
        # unselectable. Checking here means a doomed run takes no holds and
        # burns no LLM spend.
        self.caption_gen.ensure_week_unwritten(week_start)

        # Get voice style from ThemeRotator if not specified
        if voice_style is None:
            voice_style = self._get_voice_style_from_rotator()

        logger.info(
            f"Generating photo batch: week={week_start}, count={count}, voice={voice_style.value}"
        )

        # Select photos
        selections = self.selector.select_batch(
            count, filter_faces=filter_faces, allow_shortfall=allow_shortfall
        )
        if self.selector.cull_skips:
            print(
                f"Cull: skipped {len(self.selector.cull_skips)} photo(s): "
                + "; ".join(
                    f"{k.photo_key} ({k.reason}; sharpness {k.score.sharpness:.1f}, "
                    f"white {k.score.exposure_white_fraction:.2f}, "
                    f"black {k.score.exposure_black_fraction:.2f})"
                    for k in self.selector.cull_skips
                )
            )
        logger.info(
            f"Selected {len(selections)} photos from {len(set(s.journey_slug for s in selections))} journeys"
        )
        selected_keys = [f"{s.journey_slug}/{s.filename}" for s in selections]

        # Generate captions — releasing THIS run's just-taken holds on any
        # failure (LLM call, parse, count mismatch), so a dead caption phase
        # can't strand the photos as permanently unselectable.
        try:
            batch = self.caption_gen.generate_batch(selections, voice_style, week_start)
        except BaseException:
            released = self._release_pending(selected_keys)
            logger.warning(
                f"Caption generation failed — returned {len(released)} photo(s) to the pool"
            )
            raise

        # A selection the caption phase dropped (empty metadata) is absent from
        # the batch, so publish's release/commit will never see its hold —
        # free it now rather than leaking it forever.
        captioned = {f"{c.journey_slug}/{c.filename}" for c in batch.captions}
        self._release_pending([k for k in selected_keys if k not in captioned])

        logger.info(
            f"Generated {len(batch.captions)} captions. "
            f"Review at: {self.caption_gen.caption_dir / f'{week_start}_captions.yaml'}"
        )

        return batch

    def _archive_existing_captions(self, week_start: str) -> None:
        """Rename an existing week's captions YAML out of the way (``--overwrite``).

        The caption generator refuses to clobber a review file; the old file is
        renamed, never deleted, so hand-reviewed captions stay recoverable.
        """
        cache_path = self.caption_gen.caption_dir / f"{week_start}_captions.yaml"
        if not cache_path.exists():
            return
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = cache_path.parent / f"{cache_path.name}.bak-{stamp}"
        cache_path.rename(backup)
        logger.warning(f"--overwrite: existing captions moved aside: {cache_path} -> {backup}")

    def publish_batch(
        self,
        caption_yaml: str,
        force: bool = False,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Path:
        """Phase 2 under the photo-run lock, with the batch's photo holds
        released if it fails.

        A batch that dies part-way through must not consume its photos: the
        holds go back to the pool so the next run can select them again.
        Failing the review gate is not a failed publish — the batch is still
        outstanding, so it keeps its holds. The lock wraps the hold-release
        handler too: releasing holds is itself a state write.
        """
        with self._run_lock():
            # Read the keys up front: re-reading the YAML inside the handler
            # would mask the real failure if the YAML itself is what went wrong.
            keys = self._batch_keys(caption_yaml)
            try:
                return self._publish_batch(caption_yaml, force, start_date, end_date)
            except CaptionsNotReviewedError:
                raise
            except BaseException:
                released = self._release_pending(keys)
                logger.warning(f"Publish failed — returned {len(released)} photo(s) to the pool")
                raise

    @staticmethod
    def _batch_keys(caption_yaml: str) -> list[str]:
        """Selection keys for every caption in a batch."""
        batch = CaptionBatch.load(Path(caption_yaml))
        return [f"{c.journey_slug}/{c.filename}" for c in batch.captions]

    def release_batch(self, week_start: str) -> list[str]:
        """Operator recovery: return an unpublished batch's photo holds to the
        pool (``photos release <week>``), keyed by its captions YAML.

        Refuses a published batch — its photos are spent, not held, and
        releasing them would let a later batch re-post already-shipped stills.
        The lock wraps the release: it is a selection-state write.
        """
        with self._run_lock():
            caption_yaml = self.caption_gen.caption_dir / f"{week_start}_captions.yaml"
            if not caption_yaml.exists():
                raise FileNotFoundError(f"No captions YAML for week {week_start}: {caption_yaml}")
            if CaptionBatch.load(caption_yaml).published:
                raise ValueError(
                    f"Batch {week_start} is already published — its photos are spent, "
                    "not held. Refusing to release them back into the pool."
                )
            released = self._release_pending(self._batch_keys(str(caption_yaml)))
            logger.info(f"Released {len(released)} held photo(s) from batch {week_start}")
            return released

    def _publish_batch(
        self,
        caption_yaml: str,
        force: bool = False,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Path:
        """Phase 2: resolve, schedule, resize, upload, CSV, ledger.

        Ends at a Metricool import CSV. Metricool is an external scheduler:
        nothing below calls a platform API and nothing returns a post ID, so
        the batch is EXPORTED FOR SCHEDULING, not published. ``batch.published``
        is the persisted name for "this batch is spent" (holds consumed, slots
        booked) and is kept for on-disk compatibility -- it is not a claim that
        anything went live. Contrast ``publishing/dispatch.py``, which counts
        ``result.published`` only after a registered handler really posts.

        Order matters: every step that can drop a photo runs BEFORE the
        irreversible Drive upload.

        Args:
            caption_yaml: Path to the reviewed captions YAML
            force: Skip reviewed check
            start_date: First post date (YYYY-MM-DD). Defaults to the batch's
                week start, which the scheduler clamps forward when past.
            end_date: End date (YYYY-MM-DD) for scheduling boundary

        Returns:
            Path to generated Metricool CSV

        Raises:
            ValueError: If captions not reviewed and not forced
            PhotoPublishShortfallError: Fewer publishable photos than the
                batch captioned, or zero — before anything irreversible
        """
        batch = CaptionBatch.load(Path(caption_yaml))

        if not batch.reviewed and not force:
            raise CaptionsNotReviewedError(
                f"Captions not yet reviewed! Edit {caption_yaml} and set reviewed: true, "
                f"or use --force to skip review."
            )

        logger.info(
            f"Publishing photo batch: week={batch.week_start}, "
            f"{len(batch.captions)} photos, voice={batch.voice_style}"
        )

        # Upstream resource gate: fail loud on exhausted disk/RAM BEFORE the
        # first irreversible step (resize copies, Drive upload), and surface any
        # leftover unbooked-ledger marker, rather than dying with photos shipped.
        preflight_photo_run(self.output_dir)

        # Resolve against the tree the selector selected from — the module
        # default would publish a different checkout than the one reviewed.
        images_root = self.selector.site_root / IMAGES_BASE
        photo_paths = []
        valid_captions = []
        for caption in batch.captions:
            photo_path = images_root / caption.journey_slug / caption.filename
            if photo_path.exists():
                photo_paths.append(str(photo_path))
                valid_captions.append(caption)
            else:
                logger.warning(f"Photo not found, skipping from batch: {photo_path}")

        # Count-match gate (the photo mirror of the shorts-shortfall contract):
        # a caption whose source file is gone hard-fails HERE, upstream of
        # scheduling, resize, upload, CSV, ledger and the published marker —
        # never a quietly thinner week. Zero publishable photos would otherwise
        # "publish" a header-only CSV and mark the batch published.
        if not valid_captions or len(valid_captions) < len(batch.captions):
            raise PhotoPublishShortfallError(
                f"Only {len(valid_captions)} of {len(batch.captions)} captioned photo(s) "
                f"resolve to files under {images_root} — refusing to publish a short "
                f"batch. Restore the missing file(s) or regenerate the batch."
            )

        # Schedule BEFORE the upload. A republish reuses the slots the first
        # publish booked: rescheduling would see those days self-occupied in the
        # ledger and shift the whole week forward, overwriting the CSV with
        # duplicated posts and leaving the original bookings as ghosts.
        republish = batch.published and bool(batch.published_slots)
        if republish:
            slots = [PhotoSlot(**slot) for slot in batch.published_slots]
            logger.info(f"Already published — reusing {len(slots)} booked slot(s), not rebooking")
        else:
            slots = self.scheduler.schedule_batch(
                count=len(valid_captions),
                start_date=start_date or batch.week_start,
                end_date=end_date,
            )

        # Truncate to the available slots BEFORE anything irreversible, so a
        # dropped photo has not already been uploaded and consumed.
        if len(slots) < len(valid_captions):
            dropped = len(valid_captions) - len(slots)
            logger.warning(
                f"Only {len(slots)} slots available but {len(valid_captions)} captions — "
                f"dropping {dropped} captions to match slot count"
            )
            valid_captions = valid_captions[: len(slots)]
            photo_paths = photo_paths[: len(slots)]

        # A batch truncated to nothing must never "publish" (header-only CSV,
        # published marker, Windows export) — no schedulable slots is a hard
        # fail, same contract as the count-match gate above.
        if not valid_captions:
            raise PhotoPublishShortfallError(
                f"No publish slots available for week {batch.week_start} — "
                f"refusing to publish an empty batch."
            )

        # Photos that resolved away or lost their slot never ship — release
        # their holds now rather than letting them ride to the posted mark.
        shipping = [f"{c.journey_slug}/{c.filename}" for c in valid_captions]
        dropped_keys = [
            f"{c.journey_slug}/{c.filename}"
            for c in batch.captions
            if f"{c.journey_slug}/{c.filename}" not in set(shipping)
        ]
        self._release_pending(dropped_keys)

        # Build journey slug list for resizer (prevents Drive link collisions)
        journey_slugs = [caption.journey_slug for caption in valid_captions]

        # Resize photos before upload
        logger.info("Resizing photos for upload...")
        resized_paths = self.resizer.resize_batch(
            photo_paths, batch.week_start, journey_slugs=journey_slugs
        )
        upload_paths = [str(p) for p in resized_paths]

        # Upload resized photos to Drive. Pin the manifest path so the verify
        # repair provably reads and rewrites the same file this upload wrote.
        uploader = PipelineUploader(drive_folder="GoMoveShift/Photos/Daily")
        manifest_path = (
            Path(upload_paths[0]).parent / "photo_gdrive_manifest.json" if upload_paths else None
        )
        manifest = uploader.upload_photos(
            upload_paths, manifest_path=str(manifest_path) if manifest_path else None
        )

        # Verify uploads, re-upload failures, and hard-fail if still unverified
        manifest = self._verify_and_repair_uploads(uploader, upload_paths, manifest, manifest_path)

        # Build PhotoPost objects from valid captions only
        photo_posts = []
        for i, caption in enumerate(valid_captions):
            hashtag_str = " ".join(f"#{tag}" for tag in caption.hashtags)
            # Build first comment: journey page link + hashtags
            journey_url = f"https://gomoveshift.com/journeys/{caption.journey_slug}/"
            first_comment = f"More photos from this journey: {journey_url}\n\n{hashtag_str}"
            photo_posts.append(
                PhotoPost(
                    photo_index=i,
                    caption=caption.main_caption,
                    tiktok_caption=caption.tiktok_caption,
                    hashtags=hashtag_str,
                    first_comment=first_comment,
                    photo_filename=f"{caption.journey_slug}_{caption.filename}",
                    first_comment_hook=caption.first_comment_hook,
                )
            )

        # Generate Metricool CSV
        csv_dir = self.output_dir / "photo_metricool"
        csv_path = self.csv_gen.generate(
            photo_posts=photo_posts,
            slots=slots,
            manifest=manifest,
            output_dir=str(csv_dir),
            week_label=batch.week_start,
        )

        # Persist the published marker + booked slots BEFORE booking the
        # ledger, so no crash point leaves a booking without its marker: a
        # crash (or a failed YAML save) between booking and this save used to
        # leave live bookings with ``published`` unset, and the re-publish then
        # rescheduled a shifted duplicate week around its own ghost bookings.
        batch.published = True
        batch.published_slots = [asdict(slot) for slot in slots]
        batch.save(Path(caption_yaml))

        # Book the ledger on EVERY publish call, including a republish. A
        # crash between the save above and the booking below used to leave
        # ``published``/``published_slots`` persisted with the slots never
        # booked, and the "if not republish" guard then skipped booking
        # forever on every later invocation. ``_book_slots`` is idempotent
        # (``ContentLedger.add`` keys on date/time/content_type and refreshes
        # a same-source re-add rather than duplicating it), so re-booking on
        # a genuine republish just re-confirms the same entries. Theme usage
        # stays gated on ``not republish`` — ``ThemeRotator.record_usage``
        # increments ``theme_counts`` (drives ``get_next_theme``'s min-count
        # pick), so calling it on every republish would skew the voice-style
        # rotation. RESIDUAL WINDOW: a crash between ``batch.save`` above and
        # ``_record_theme_usage`` below still leaves ``republish=True``
        # forever, so that one batch's usage is never recorded — accepted as
        # minor (rotation-distribution drift toward the under-counted voice,
        # no lost/duplicated published content) rather than made speculatively
        # idempotent.
        self._book_slots(slots, valid_captions, csv_path)
        if not republish:
            self._record_theme_usage(batch)

        # Consumption commits HERE — the photos are on Drive, in the CSV and in
        # the ledger. Any earlier exit releases the holds instead, so a photo is
        # never spent on a batch that did not ship it.
        self.selector.commit_posted(shipping)

        # Export the CSV to the configured export dir (GMS_EXPORT_DIR) for the
        # scheduler upload. Under WSL (the Windows PowerShell executable is found) the
        # copy goes through PowerShell Copy-Item because WSL 9p shutil.copy2 is
        # unreliable; elsewhere it is a plain copy.
        export_dir = _paths.gms_export_dir()
        if export_dir.exists():
            self._export_to_windows(csv_path, export_dir)

        logger.info(f"Exported for scheduling — Metricool CSV at: {csv_path}")
        return csv_path

    def _book_slots(self, slots, valid_captions, csv_path) -> None:
        """Book this batch's photo slots into the shared content ledger."""
        entries = []
        for i, slot in enumerate(slots):
            title = valid_captions[i].main_caption[:40] if i < len(valid_captions) else ""
            entries.append(
                ContentEntry(
                    date=slot.date,
                    time=slot.time,
                    content_type="photo",
                    title=title,
                    platform="all",
                    status="scheduled",
                    source="photo_pipeline",
                    source_file=str(csv_path),
                )
            )
        # Locked writer holds the ledger lock across the whole load→add→save so
        # a concurrent booking can't lose-update this one. Best-effort: never
        # RAISES post-upload (that would release_pending shipped photos),
        # but a SILENT miss risks a later double-book, so mark the failure.
        if not record_confirmed_upload(entries):
            self._mark_ledger_unbooked(slots, csv_path)

    def _mark_ledger_unbooked(self, slots, csv_path) -> None:
        """Persist a durable marker that confirmed uploads were NOT booked, so
        the next run's preflight flags the double-book risk. Accumulates across
        runs (no failed slot lost) and is itself best-effort (write failure only
        logs, never raises over a shipped batch)."""
        marker = self.output_dir / LEDGER_UNBOOKED_MARKER
        existing: list[dict] = []
        if marker.exists():
            with contextlib.suppress(OSError, ValueError):
                existing = json.loads(marker.read_text()).get("slots", [])
        payload = {
            "reason": "ledger-booking-failed",
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
            "csv_path": str(csv_path),
            "slots": existing + [{"date": s.date, "time": s.time} for s in slots],
        }
        with contextlib.suppress(OSError):
            atomic_write_json(marker, payload)
        logger.error(
            "LEDGER-UNBOOKED: %d confirmed photo slot(s) could not be booked; "
            "marker written to %s — the next run's preflight will flag the "
            "double-book risk.",
            len(slots),
            marker,
        )

    @staticmethod
    def _verify_and_repair_uploads(uploader, upload_paths, manifest, manifest_path):
        """Verify Drive links, re-upload the failures, hard-fail if still dead.

        ``upload_photos`` only uploads files that are absent from the manifest
        (its resume filter), so re-uploading without purging the failed entries
        finds nothing pending, re-marks the batch complete, and ships the dead
        link to the CSV — which only rejects EMPTY links. Purge first so the
        failures are genuinely pending, then re-verify: a batch that still
        cannot be verified raises instead of reporting success. The caller
        releases the batch's photo holds on the way out, so a hard fail leaves
        the photos re-selectable rather than spent.
        """
        result = uploader.verify_uploads(manifest)
        if not result.failed:
            return manifest

        failed = set(result.failed)
        failed_paths = [p for p in upload_paths if Path(p).name in failed]
        if not failed_paths:
            raise PhotoUploadVerificationError(
                f"Drive verification failed for {sorted(failed)} but no local file matches, "
                "so they cannot be re-uploaded. Refusing to ship dead Drive links."
            )

        # Purge BEFORE re-uploading: upload_photos skips anything already in the
        # manifest, so leaving the dead entries makes the re-upload a no-op.
        logger.warning(f"Re-uploading {len(failed_paths)} failed file(s)...")
        for name in failed:
            manifest.files.pop(name, None)
        manifest.completed = False
        manifest.save(Path(manifest_path))

        manifest = uploader.upload_photos(failed_paths, manifest_path=str(manifest_path))

        recheck = uploader.verify_uploads(manifest)
        if recheck.failed:
            raise PhotoUploadVerificationError(
                f"{len(recheck.failed)} photo(s) still fail Drive verification after "
                f"re-upload: {sorted(recheck.failed)}. Refusing to ship dead Drive links."
            )
        return manifest

    def auto_batch(
        self,
        start_date: str,
        end_date: str,
        output_csv_dir: Optional[str] = None,
        voice_style: Optional[VoiceStyle] = None,
        filter_faces: bool = False,
    ) -> Path:
        """One-shot: select -> caption -> upload -> schedule -> CSV -> ledger.

        Args:
            start_date: First post date (YYYY-MM-DD)
            end_date: Last post date (YYYY-MM-DD), inclusive
            output_csv_dir: Where to copy the final CSV (default: no copy)
            voice_style: Override voice style (default: ThemeRotator)
            filter_faces: Skip photos with detected human faces

        Returns:
            Path to the generated Metricool CSV
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        if end < start:
            raise ValueError(f"end_date ({end_date}) is before start_date ({start_date})")

        count = (end - start).days + 1
        logger.info(f"Auto-batch: {start_date} to {end_date} = {count} posts")

        # Phase 1: generate
        batch = self.generate_batch(
            week_start=start_date,
            count=count,
            voice_style=voice_style,
            filter_faces=filter_faces,
        )

        # auto_batch bypasses the manual `reviewed: true` gate (it self-approves
        # and publishes with force=True). Enforce the ContentValidator blocklist
        # over every caption field here — an upstream gate before the irreversible
        # Metricool CSV / Windows export — so no hype/misleading text can ship
        # without a human review. Raises ContentValidationError on any violation.
        self._enforce_content_safety(batch)

        # Auto-approve and re-save
        batch.reviewed = True
        caption_path = self.caption_gen.caption_dir / f"{start_date}_captions.yaml"
        batch.save(caption_path)

        # Phase 2: publish with end_date boundary
        csv_path = self.publish_batch(
            caption_yaml=str(caption_path),
            force=True,
            end_date=end_date,
        )

        # Copy CSV to output directory if requested
        if output_csv_dir:
            out_dir = Path(output_csv_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            dest_name = f"gms_photos_{start_date}_to_{end_date}_metricool.csv"
            dest = out_dir / dest_name
            fs_safety.safe_copy(csv_path, dest)
            logger.info(f"Copied CSV to {dest}")
            csv_path = dest

        return csv_path

    @staticmethod
    def _enforce_content_safety(batch: CaptionBatch) -> None:
        """Enforce the ContentValidator blocklist over all caption fields.

        Called from auto_batch, which self-approves the LLM caption batch
        instead of routing it through the manual `reviewed: true` gate. This
        guarantees no blocklisted/hype text reaches the Metricool CSV or the
        Windows export. Hashtags are already sanitized at generation time.

        Raises:
            ContentValidationError: if any field fails validation.
        """
        from ..content_validator import ContentValidator

        validator = ContentValidator()
        validator.enforce(
            captions=[c.main_caption for c in batch.captions]
            + [c.tiktok_caption for c in batch.captions],
            hooks=[c.first_comment_hook for c in batch.captions],
        )

    def status(self) -> dict:
        """Get current photo posting status."""
        from .selector import PhotoSelectionState

        state = PhotoSelectionState.load(path=self.selector.state_path)
        total = self.selector.get_total_photos()
        pending = set(state.pending_photos)
        # Held-but-unshipped photos are not spent — don't count them as posted.
        posted = len(set(state.posted_photos) - pending)

        return {
            "total_photos": total,
            "posted": posted,
            "pending": len(pending),
            "remaining": total - posted,
            "journeys": len(self.selector.journeys),
            "days_of_content": (total - posted),
            "rotation_index": state.journey_rotation_index,
        }

    def _get_voice_style_from_rotator(self) -> VoiceStyle:
        """Get next voice style from SEO manager's ThemeRotator."""
        try:
            from seo_manager.analytics.theme_rotator import ContentTheme, ThemeRotator

            rotator = ThemeRotator()
            theme = rotator.get_next_theme(category="photo")

            # Map ContentTheme back to VoiceStyle
            mapping = {
                ContentTheme.VOICE_FIRST_PERSON: VoiceStyle.FIRST_PERSON,
                ContentTheme.VOICE_MINIMAL_ZEN: VoiceStyle.MINIMAL_ZEN,
                ContentTheme.VOICE_ENGAGING: VoiceStyle.ENGAGING_CONVERSATIONAL,
            }
            # VOICE_INTERACTIVE may not exist in older seo_manager versions
            if hasattr(ContentTheme, "VOICE_INTERACTIVE"):
                mapping[ContentTheme.VOICE_INTERACTIVE] = VoiceStyle.INTERACTIVE
            return mapping.get(theme, VoiceStyle.FIRST_PERSON)

        except ImportError:
            logger.warning("seo_manager not available, defaulting to FIRST_PERSON")
            return VoiceStyle.FIRST_PERSON
        except Exception as e:
            # A RUNTIME ThemeRotator failure (corrupt/locked rotation state, a
            # concurrent writer, etc.) must never crash caption generation —
            # the rotator is an optional A/B-tracking nicety. Fall back to the
            # default voice and surface the failure.
            logger.warning(
                f"ThemeRotator failed ({type(e).__name__}: {e}); defaulting to FIRST_PERSON"
            )
            return VoiceStyle.FIRST_PERSON

    def cleanup(
        self,
        min_age_days: int = 14,
        confirm: bool = False,
    ) -> PhotoCleanupPlan:
        """Clean up photo pipeline artifacts (dry-run by default).

        Runs under the photo-run lock (see :meth:`_run_lock`): a cleanup racing
        a live generate/publish could delete artifacts — resized copies, the
        Drive manifest — that the run is mid-way through producing or consuming.

        Args:
            min_age_days: Minimum age in days before cleanup (default: 14)
            confirm: Actually delete files (default: dry-run only)

        Returns:
            PhotoCleanupPlan showing what was/would be deleted
        """
        with self._run_lock():
            batches = self.cleanup_handler.discover_batches()
            plan = self.cleanup_handler.plan_cleanup(batches, min_age_days=min_age_days)

            if confirm and plan.files:
                deleted = self.cleanup_handler.execute_plan(plan)
                logger.info(f"Deleted {deleted} file(s)")

        return plan

    def _record_theme_usage(self, batch: CaptionBatch):
        """Record voice style usage in ThemeRotator for A/B tracking."""
        try:
            from seo_manager.analytics.theme_rotator import ContentTheme, ThemeRotator

            rotator = ThemeRotator()
            voice_map = {
                "voice_first_person": ContentTheme.VOICE_FIRST_PERSON,
                "voice_minimal_zen": ContentTheme.VOICE_MINIMAL_ZEN,
                "voice_engaging": ContentTheme.VOICE_ENGAGING,
            }
            if hasattr(ContentTheme, "VOICE_INTERACTIVE"):
                voice_map["voice_interactive"] = ContentTheme.VOICE_INTERACTIVE
            theme = voice_map.get(batch.voice_style)
            if theme:
                content_id = f"photo_batch_{batch.week_start}"
                rotator.record_usage(
                    content_id=content_id,
                    video_id=batch.week_start,
                    theme=theme,
                    category="photo",
                )
        except Exception as e:
            # Non-fatal (A/B tracking only) but must be visible: a silent debug
            # swallow hid this failing for every batch. Warn, don't raise.
            logger.warning(f"Theme usage recording failed ({type(e).__name__}): {e}")

    @staticmethod
    def _export_to_windows(csv_path: Path, export_dir: Path):
        """Copy the CSV into the export dir: through PowerShell under WSL (reliable across 9p), a plain copy elsewhere."""
        # Standard WSL location of the Windows PowerShell executable; when absent
        # (not WSL) the plain copy fallback below is used.
        powershell = Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
        if not powershell.exists():
            # Fallback to shutil if not on WSL
            try:
                dest = export_dir / csv_path.name
                fs_safety.safe_copy(csv_path, dest)
                logger.info(f"Exported CSV to {dest}")
            except OSError as e:
                logger.warning(f"Could not export CSV to {dest}: {e}")
            return

        try:
            win_src = (
                subprocess.check_output(
                    ["wslpath", "-w", str(csv_path)],
                    text=True,
                )
                .strip()
                .replace("'", "''")
            )
            win_dst = (
                subprocess.check_output(
                    ["wslpath", "-w", str(export_dir / csv_path.name)],
                    text=True,
                )
                .strip()
                .replace("'", "''")
            )
            subprocess.run(
                [str(powershell), "-Command", f"Copy-Item '{win_src}' '{win_dst}' -Force"],
                check=True,
                capture_output=True,
                timeout=30,
            )
            logger.info(f"Exported CSV to {export_dir / csv_path.name}")
        except Exception as e:
            logger.warning(f"PowerShell export failed: {e}")
