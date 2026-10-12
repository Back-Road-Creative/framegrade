"""External-systems coordinator: Drive + YouTube upload helpers extracted from phase_d."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from ...config import Config
from ..exceptions import YouTubeQuotaError

# Logger keeps phase_d's name so caplog assertions still flow.
logger = logging.getLogger("gomoveshift.pipeline.drive_phases.phase_d")

# Lazy-loaded exception types from google API libraries.
# Resolved at usage time so the module can still be imported when the
# google packages are absent (the upload path raises RuntimeError instead).
try:
    from google.auth.exceptions import RefreshError as _RefreshError
except ImportError:
    _RefreshError = type(None)  # never matched

try:
    from googleapiclient.errors import HttpError as _HttpError
except ImportError:
    _HttpError = type(None)  # never matched


def _resolve_drive_subfolder_id(uploader, subfolder: str) -> str:
    """Return the Drive folder ID for ``<uploader.drive_folder>/<subfolder>``.

    Wraps ``uploader._get_or_create_folder_path`` so phase_d can use the
    list-form ``upload_shorts(shorts_dir=[paths], folder_id=…)`` invocation
    introduced for the day-first shorts upload.

    Test seam: if the uploader exposes a public ``_resolve_drive_subfolder_id``
    method (mocks do), delegate to it; otherwise drive into the real
    PipelineUploader internals. This avoids requiring a live Drive client in
    unit tests while keeping a single resolution path in production.
    """
    override = getattr(uploader, "_resolve_drive_subfolder_id", None)
    # Only prefer the override when it's actually a callable distinct from
    # this helper itself (avoids infinite recursion if the attr is bound).
    if callable(override) and override is not _resolve_drive_subfolder_id:
        return override(subfolder)
    drive = uploader._get_client()
    folder_path = f"{uploader.drive_folder}/{subfolder}"
    return uploader._get_or_create_folder_path(drive, folder_path)


def _resolve_youtube_uploader_class():
    """Late-import the ``YouTubeUploader`` class."""
    from .youtube_uploader import YouTubeUploader  # type: ignore[attr-defined]

    return YouTubeUploader


def _upload_lf_to_youtube(
    *,
    yt,
    state,
    lf,
    lf_path: Path,
    title: str,
    description: str,
    tags: list,
    publish_at,
    metadata: dict,
    config: Config,
    persist=None,
):
    """Upload one LF to YouTube; return the upload record or None.

    Exception policy mirrors the original inline loop body:
      * ``_RefreshError`` → re-raised as ``RuntimeError`` with an
        operator-readable "re-run youtube-auth" message.
      * ``YouTubeQuotaError`` → propagated unchanged (the caller must
        persist state and emit the batch-paused banner).
      * ``_HttpError`` → logged WARNING, return ``None`` (skip this LF).
      * Any other ``Exception`` → logged ERROR, return ``None``.
    """
    # Surface the per-LF filmed-on date in the YouTube
    # ``recordingDetails.recordingDate`` API field. Source is the
    # already-canonical ``metadata["recording_date"]`` (populated at
    # phase_d_metadata._build_metadata_per_lf from
    # ``state["per_lf_recording_date"]``, itself BC-invariant-guarded).
    # When absent (legacy records) we pass None — the uploader skips
    # the field, matching prior behavior.
    recording_date = metadata.get("recording_date") if isinstance(metadata, dict) else None
    try:
        return yt.upload_if_needed(
            state=state,
            lf_idx=lf.index,
            video_path=lf_path,
            title=title,
            description=description,
            tags=tags,
            category_id="2",
            publish_at=publish_at,
            privacy="private",
            # Route through
            # ``upload_video_with_playlist`` so the resolved
            # playlist gets a ``playlistItems.insert``. Total
            # by construction via ``youtube_playlists.default``.
            metadata=metadata,
            config=config,
            recording_date=recording_date,
            persist=persist,
        )
    except _RefreshError:
        raise RuntimeError(
            f"YouTube auth failed for LF {lf.index} "
            "— re-run `gomoveshift youtube-auth` to refresh credentials"
        ) from None
    except YouTubeQuotaError:
        raise
    except _HttpError as e:
        logger.warning("  LF %d: YouTube upload HTTP error: %s", lf.index, e)
        return None
    except Exception as e:
        logger.error("  LF %d: YouTube upload unexpected error: %s", lf.index, e)
        return None


def _upload_superlong_parts_to_youtube(
    *,
    yt,
    state,
    output_base: Path,
    config: Config,
    save_state,
    project_base: str | None = None,
) -> dict:
    """Upload every ``state.superlong_parts`` record, AFTER the LF loop
    above so a part failure never takes down an already-finished LF
    batch. Reuses ``upload_if_needed`` against a throwaway
    ``SimpleNamespace(youtube_uploads=sidecar_uploads)`` so records land
    in a STANDALONE ``superlong_uploads.json``, never
    ``state.youtube_uploads`` (would leak into the LF sidecar's
    plan/index gate). Slots are booked via ``next_free_superlong_slot``.
    Exception policy mirrors
    ``_upload_lf_to_youtube``: quota propagates, else skip.
    """
    from .. import youtube_sidecar
    from ..content_ledger import EST, ContentLedger, next_free_superlong_slot
    from ..longform_metadata import _location_tags

    parts = youtube_sidecar.iter_superlong_parts(state)
    if not parts:
        return {}
    youtube_sidecar.assert_superlong_sidecar_integrity(output_base, state)
    sidecar_uploads = youtube_sidecar.load_superlong_sidecar(output_base)
    ledger = ContentLedger.load(Path(config.seo.storage_path) / ".content_ledger.json")
    region = getattr(state, "region", None)
    for part_key, record in parts:
        existing = sidecar_uploads.get(part_key)
        # A record with video_id but no playlist_id is an orphan left by a
        # playlistItems.insert failure (e.g. a 409) after videos.insert
        # already succeeded — skip re-upload only once BOTH ids are
        # present; otherwise fall through so upload_if_needed's
        # orphan-recovery branch (youtube_uploader.py) can re-run just the
        # playlist assignment.
        if existing and existing.get("video_id") and existing.get("playlist_id"):
            continue
        # Captured before the fall-through below: an orphan (video_id
        # present, playlist_id missing) reaches upload_if_needed's
        # playlist-only recovery, never a real upload — its returned
        # publish_at is the ORIGINAL slot, not one this iteration could
        # have overrun. _recheck_publish_at_after_upload's overrun logic
        # only applies to an upload THIS run performed; running it on a
        # recovered orphan would flip an already-published (or
        # correctly-scheduled) video back to private.
        recovered_orphan = bool(existing and existing.get("video_id"))
        ledger.reload()
        # project_base/record["title"] are this part's real per-video identity
        # (record.get("title") already carries the "— Part N of M" suffix
        # _book_lf_upload_in_ledger books alongside project_base below) —
        # threading them here is what makes rules 3/4 of
        # resolve_superlong_publish_day actually govern this booking instead
        # of staying inert.
        try:
            publish_at = next_free_superlong_slot(
                ledger,
                config,
                region,
                after=datetime.now(EST),
                project_name=project_base,
                title=record.get("title") or "",
            )
        except RuntimeError as e:
            logger.error("  superlong part %s: no free publish slot: %s", part_key, e)
            continue
        # Precedent: _upload_lf_to_youtube sources tags/recording_date from
        # already-canonical metadata rather than composing a second tag
        # vocabulary (docstring above). ``recording_date`` is the part's
        # own day — the ``"YYYY-MM-DD"`` prefix of ``part_key`` (see
        # ``youtube_sidecar.iter_superlong_parts``) — and tags reuse the
        # SAME whole-phrase location/season builder longform_metadata uses
        # for regular LFs.
        location = record.get("location")
        recording_date = part_key.split(":", 1)[0]
        tags = _location_tags(location, recording_date) if location else []
        # ``format: "superlong"`` is an explicit override read by
        # ``playlist_resolver.resolve_playlist`` (parts carry no
        # ``trim_ratio``, so the highlights/full trim-band derivation
        # never fires) — routes parts to their OWN playlist instead of
        # falling through to the region-only "Drives" rules.
        metadata = {
            "location": location,
            "region": region,
            "recording_date": recording_date,
            "format": "superlong",
        }
        try:
            record_out = yt.upload_if_needed(
                state=SimpleNamespace(youtube_uploads=sidecar_uploads),
                lf_idx=part_key,
                video_path=Path(record["path"]),
                title=record.get("title", ""),
                description=record.get("description", ""),
                tags=tags,
                category_id="2",
                publish_at=publish_at,
                privacy="private",
                metadata=metadata,
                config=config,
                recording_date=recording_date,
                # videos.insert above is IRREVERSIBLE; flush the STANDALONE
                # superlong sidecar (this path's throwaway SimpleNamespace
                # state makes save_state meaningless) the instant it returns,
                # before the fallible playlist assignment — same boundary the
                # regular-LF call site guards with its own persist=.
                persist=lambda: youtube_sidecar.save_superlong_sidecar(
                    output_base, sidecar_uploads, state=state
                ),
            )
        except _RefreshError:
            raise RuntimeError(f"YouTube auth failed for superlong part {part_key}") from None
        except YouTubeQuotaError:
            youtube_sidecar.save_superlong_sidecar(output_base, sidecar_uploads, state=state)
            if save_state is not None:
                save_state(state)
            raise
        except _HttpError as e:
            logger.warning("  superlong part %s: HTTP error: %s", part_key, e)
            record_out = None
        except Exception as e:
            logger.error("  superlong part %s: upload error: %s", part_key, e)
            record_out = None
        if record_out is not None:
            if not recovered_orphan:
                record_out = _recheck_publish_at_after_upload(
                    yt=yt,
                    record=record_out,
                    ledger=ledger,
                    config=config,
                    region=region,
                    project_name=project_base,
                )
            sidecar_uploads[part_key] = record_out  # explicit write-through
        youtube_sidecar.save_superlong_sidecar(output_base, sidecar_uploads, state=state)
        if record_out is None:
            continue
        _book_lf_upload_in_ledger(
            record=record_out, state=state, config=config, project_base=project_base
        )
        if save_state is not None:
            save_state(state)
    return sidecar_uploads


def _recheck_publish_at_after_upload(
    *, yt, record: dict, ledger, config: Config, region, project_name: str | None = None
) -> dict:
    """Re-check ``record["publish_at"]`` AFTER the (possibly multi-hour,
    tens-of-GB) upload that produced ``record`` has completed. The slot was
    resolved BEFORE that upload began (``next_free_superlong_slot`` above),
    so a slow upload can overrun it — the video would go private with a
    ``publishAt`` already in the past. If overrun, re-resolve a fresh slot
    and push it to YouTube via ``update_privacy`` (the upload already sent
    the stale value) before the ledger ever sees it.

    ``project_name``/``record["title"]`` are threaded through so a re-resolve
    honours the same rules 3/4 precedence as the original booking above,
    rather than silently dropping back to the flat default.
    """
    from ..content_ledger import EST, next_free_superlong_slot

    publish_at = record.get("publish_at")
    video_id = record.get("video_id")
    if not (publish_at and video_id):
        return record
    now = datetime.now(EST)
    if datetime.fromisoformat(publish_at).astimezone(EST) > now:
        return record
    ledger.reload()
    new_slot = next_free_superlong_slot(
        ledger,
        config,
        region,
        after=now,
        project_name=project_name,
        title=record.get("title") or "",
    )
    yt.update_privacy(video_id, privacy="private", publish_at=new_slot)
    logger.warning(
        "  superlong part video_id=%s: publish_at %s was overrun by the upload — re-resolved to %s",
        video_id,
        publish_at,
        new_slot.isoformat(),
    )
    record["publish_at"] = new_slot.isoformat()
    return record


def _book_lf_upload_in_ledger(
    *,
    record: dict,
    state,
    config: Config,
    project_base: str | None = None,
    lf_index: int | None = None,
) -> None:
    """Book one CONFIRMED LF upload: called only after the upload
    returned a record; books the REAL ``publishAt`` slot (not the generic
    grid) + video_id/region/title. Best-effort + video_id-deduped inside
    ``record_confirmed_upload`` — never fails the live upload, exactly-once on resume.

    ``project_base`` is the caller-derived project slug (``phase_d``'s
    ``source_dir.stem``-derived value) — ``PipelineState`` carries no
    ``project_base``/``project_name`` field, so there is nothing to infer
    it from here. A missing value books with an empty ``project_name``
    (provenance-only — never blocks the booking) and logs a warning
    rather than failing the already-irreversible upload.

    ``lf_index`` (a regular LF's index) also mirrors the booking into
    ``schedule.sqlite3`` beside the ledger when that store exists, so
    seo_manager's nightly audit sees it; superlong parts pass none and stay
    ledger-only (the GMS superlong resolver reads the ledger).
    """
    from ..content_ledger import EST, ContentEntry, record_confirmed_upload

    vid, publish_at = record.get("video_id"), record.get("publish_at")
    if not (isinstance(vid, str) and vid and isinstance(publish_at, str) and publish_at):
        return  # nothing schedulable to book (unscheduled/mocked record)
    if not project_base:
        logger.warning(
            "  ledger booking for video_id=%s: no project_base supplied — "
            "booking with an empty project_name (provenance untraceable)",
            vid,
        )
    dt = datetime.fromisoformat(publish_at).astimezone(EST)
    entry = ContentEntry(
        date=dt.strftime("%Y-%m-%d"),
        time=dt.strftime("%H:%M:%S"),
        content_type="longform",
        title=record.get("title") or "",
        project_name=project_base or "",
        source="pipeline",
        video_id=vid,
        region=getattr(state, "region", None),
    )
    ledger_booked = record_confirmed_upload(
        [entry], Path(config.seo.storage_path) / ".content_ledger.json"
    )
    if not ledger_booked:
        # A live upload is never silently unbooked: record_confirmed_upload
        # already logged LEDGER-BOOKING-FAILED with exc_info; name the
        # video_id again here so an operator scanning for this specific
        # slot's fate doesn't have to correlate two log lines.
        logger.error(
            "  LEDGER-UNBOOKED: confirmed LF upload video_id=%s was NOT booked "
            "into the ledger — the next run's preflight will not see this "
            "slot as taken; risk of a later double-book.",
            vid,
        )
    # The store write does not depend on the ledger write: a confirmed upload missing from
    # the ledger must still reach schedule.sqlite3 (best effort, logs on failure).
    if lf_index is not None:
        from ..seo_slots import book_in_store

        book_in_store(
            Path(config.seo.storage_path) / ".content_ledger.json",
            config,
            project=project_base or "",
            lf_index=lf_index,
            region=getattr(state, "region", None),
            slot_at=dt,
            video_id=vid,
        )


def _upload_all_lfs_shorts_to_drive(
    *,
    drive_plan,
    state,
    project_base: str,
    output_base: Path,
    uploader_factory,
    skip_drive_upload: bool,
    save_state=None,
) -> tuple[dict, object]:
    """Drive the per-LF shorts upload loop. Returns ``(upload_manifests, uploader)``.

    Honours skip-when-complete: a prior manifest skips re-upload
    only if its file count equals the expected day-first shorts count;
    partial manifests trigger re-upload. The day-first path uses the
    list-form ``upload_shorts(shorts_dir=[paths], folder_id=…)``. When
    ``state["shorts_per_lf"]`` is empty while regular LFs exist, the call
    refuses with ``RuntimeError`` rather than glob-ship ungated shorts from
    the legacy directory. ``_RefreshError`` re-raises as an operator-actionable
    ``RuntimeError``. Any other per-LF failure propagates
    immediately (WARN-and-continue would ship a Metricool CSV with empty
    ``Picture Url 1`` cells).
    """
    # Late-bind phase_d so test patches on phase_d._upload_lf_shorts_to_drive /
    # _resolve_shorts_paths / _RefreshError intercept the inner refs below.
    from . import phase_d as _pd

    # Operate on state's OWN upload_manifests dict and persist per-LF
    # (below) so a mid-loop raise — a Drive failure here, or the recoverable
    # YouTubeQuotaError exit later in Phase D that saves state — leaves the
    # already-uploaded manifests on state. Otherwise they were only assigned at
    # end-of-phase, so every short re-uploaded to Drive on resume (the
    # skip-when-complete check never fired).
    upload_manifests: dict = state.upload_manifests
    if upload_manifests is None:
        upload_manifests = {}
        state.upload_manifests = upload_manifests
    if skip_drive_upload:
        logger.info("  D-upload-shorts: SKIPPED")
        return upload_manifests, None

    logger.info("  D-upload-shorts: Uploading shorts per LF...")

    shorts_paths_by_lf = state.shorts_per_lf
    use_dayfirst = bool(shorts_paths_by_lf)
    if not use_dayfirst and drive_plan.regular_lfs:
        # The legacy fallback used to glob ``output_base/shorts/lf_NN/``
        # and ship whatever sat on disk with ZERO LUFS/geometry/OST gating when
        # ``shorts_per_lf`` was empty/absent — the silent glob-ship hole. Kill
        # it: refuse rather than ship ungated shorts. The upstream
        # ``_assert_shorts_quality_gate`` normally raises first; this is the
        # defence-in-depth backstop at the actual upload site. Refuse BEFORE
        # building an uploader so a doomed run makes no Drive client.
        raise RuntimeError(
            "D-upload-shorts: shorts_per_lf is empty/absent while regular LFs "
            "exist — refusing the legacy glob-ship of output_base/shorts/lf_NN/ "
            "(ungated shorts). Rebuild shorts_per_lf from the day-first cache."
        )

    uploader = uploader_factory()

    for lf in drive_plan.regular_lfs:
        project_name = f"{project_base}_lf{lf.index:02d}"
        # Skip re-upload only if the prior manifest is COMPLETE
        # (file count matches the expected day-first shorts count).
        prior_manifest = upload_manifests.get(lf.index)
        if prior_manifest is not None and use_dayfirst:
            expected_shorts = list((shorts_paths_by_lf or {}).get(lf.index, []) or [])
            manifest_files = prior_manifest.files or {}
            if expected_shorts and len(manifest_files) == len(expected_shorts):
                logger.info(
                    "  LF %d: prior upload_manifest complete (%d/%d files) — skipping re-upload",
                    lf.index,
                    len(manifest_files),
                    len(expected_shorts),
                )
                continue
            elif manifest_files:
                logger.info(
                    "  LF %d: prior upload_manifest partial (%d/%d files) — re-uploading",
                    lf.index,
                    len(manifest_files),
                    len(expected_shorts),
                )
        try:
            if use_dayfirst:
                path_strs = _pd._resolve_shorts_paths(lf, shorts_paths_by_lf)
                if not path_strs:
                    logger.warning(
                        "  LF %d: no day-first shorts paths in state "
                        "— skipping Drive upload for this LF",
                        lf.index,
                    )
                    continue
            else:
                path_strs = []
            manifest = _pd._upload_lf_shorts_to_drive(
                uploader=uploader,
                lf=lf,
                project_name=project_name,
                shorts_paths=path_strs,
                output_base=output_base,
                use_dayfirst=use_dayfirst,
            )
            upload_manifests[lf.index] = manifest
            # Persist per-LF so a later mid-loop raise doesn't discard it.
            if save_state is not None:
                save_state(state)
        except _pd._RefreshError:
            # An expired credential will fail every subsequent LF identically;
            # fail fast with the operator-actionable message rather than logging
            # one warning per LF and completing with a partial manifest.
            raise RuntimeError(
                f"Drive auth failed during shorts upload (LF {lf.index}) "
                "— refresh the Drive token (GMS_DRIVE_TOKEN_ENC / GMS_DRIVE_CREDENTIALS_ENC)"
            ) from None
        except Exception as e:
            # The previous broad ``except Exception`` swallowed every
            # other failure mode and let the loop continue, leaving
            # ``upload_manifests[lf.index]`` unpopulated and shipping a
            # Metricool CSV with empty ``Picture Url 1`` cells. Propagate
            # immediately on first failure (robust-by-construction; a
            # gather-all-failures variant was rejected). Log with LF index first so
            # the operator sees which LF failed without having to parse the
            # traceback.
            logger.error("Drive upload failed for LF %d: %s", lf.index, e)
            raise
    return upload_manifests, uploader


def _upload_lf_thumbnails_to_drive(
    *,
    drive_plan,
    state,
    output_base: Path,
    project_base: str,
    uploader,
    uploader_factory,
    skip_drive_upload: bool,
):
    """Upload each LF's canonical thumbnail JPG to Drive (publicly-shareable).

    Mirrors the legacy Ireland behaviour: the per-LF Metricool CSV's
    ``Picture Url 1`` MUST be a publicly-fetchable URL, and youtu.be
    won't work for scheduled-PRIVATE LFs (Metricool's media fetcher
    can't see the video pre-publish). Resolves the per-LF Drive
    subfolder and uploads the canonical ``thumbnails/lf_NN.jpg``.
    Persists URLs in ``state["lf_thumbnail_drive_urls"]`` (idempotent —
    a prior URL skips re-upload). Returns the (possibly-newly-built)
    uploader for re-use by downstream stages.
    """
    lf_thumbnail_drive_urls: dict[int, str] = state.lf_thumbnail_drive_urls or {}
    if skip_drive_upload or not drive_plan.regular_lfs:
        if skip_drive_upload:
            logger.info("  D-upload-thumbs: SKIPPED")
        state.lf_thumbnail_drive_urls = lf_thumbnail_drive_urls
        return uploader

    logger.info("  D-upload-thumbs: Uploading LF thumbnails to Drive...")
    if uploader is None:
        uploader = uploader_factory()
    thumbs_root = output_base / "thumbnails"
    for lf in drive_plan.regular_lfs:
        if lf.index in lf_thumbnail_drive_urls and lf_thumbnail_drive_urls[lf.index]:
            continue
        canonical_thumb = thumbs_root / f"lf_{lf.index:02d}.jpg"
        if not canonical_thumb.exists():
            logger.warning(
                "  LF %d: canonical thumbnail missing at %s — Drive upload skipped",
                lf.index,
                canonical_thumb,
            )
            continue
        project_name = f"{project_base}_lf{lf.index:02d}"
        try:
            folder_id = _resolve_drive_subfolder_id(uploader, project_name)
            drive = uploader._get_client()
            thumb_url = uploader._upload_file(
                drive,
                canonical_thumb,
                folder_id,
                make_shareable=True,
                mimetype="image/jpeg",
            )
            lf_thumbnail_drive_urls[lf.index] = thumb_url
        except Exception as e:
            logger.warning("  LF %d: thumbnail Drive upload failed: %s", lf.index, e)
    state.lf_thumbnail_drive_urls = lf_thumbnail_drive_urls
    return uploader


def _resolve_shorts_paths(lf, shorts_paths_by_lf) -> list[str]:
    """Return per-LF day-first shorts paths coerced to strings.

    Empty list when ``shorts_paths_by_lf`` is None or has no entry for
    ``lf.index``. The caller treats an empty result as "skip this LF's
    Drive upload" (promoted to a hard fail elsewhere).
    """
    if not shorts_paths_by_lf:
        return []
    paths = list(shorts_paths_by_lf.get(lf.index, []) or [])
    return [str(p) for p in paths]


def _upload_lf_shorts_to_drive(
    *,
    uploader,
    lf,
    project_name: str,
    shorts_paths: list[str],
    output_base: Path,
    use_dayfirst: bool,
):
    """Upload one LF's shorts to Drive and return an ``UploadManifest``.

    Day-first path resolves the per-LF subfolder ID and uploads exact
    paths (avoids the over-upload that the legacy directory-form would
    cause when day shorts dirs are shared across LFs in multi-LF days).
    Legacy path globs ``<output_base>/shorts/lf_NN/``.

    Raises whatever the underlying uploader raises. The caller
    does not swallow per-LF failures — non-``_RefreshError`` errors
    propagate immediately so the run aborts rather than shipping a
    Metricool CSV with empty ``Picture Url 1`` cells.
    """
    if use_dayfirst:
        folder_id = _resolve_drive_subfolder_id(uploader, project_name)
        links = uploader.upload_shorts(
            shorts_dir=shorts_paths,
            folder_id=folder_id,
        )
        from ..uploader import UploadManifest as _ManifestCls

        return _ManifestCls(
            project_name=project_name,
            folder_id=folder_id,
            files={Path(p).name: link for p, link in zip(shorts_paths, links)},
            completed=True,
        )
    legacy_dir = output_base / "shorts" / f"lf_{lf.index:02d}"
    return uploader.upload_shorts(
        shorts_dir=str(legacy_dir),
        project_name=project_name,
        drive_subfolder=project_name,
    )


def _pending_youtube_insert_count(
    *, drive_plan, youtube_uploads: dict, state, output_base: Path
) -> int:
    """Count uploads this run would still need to perform.

    Regular LFs already recorded with a ``video_id`` in ``youtube_uploads``
    don't count — resume must not re-refuse a batch it has already made
    progress on. Superlong parts are counted the same way against their
    own sidecar (read-only; never fails) so a mid-superlong resume isn't
    penalised for parts already shipped.
    """
    pending_lfs = sum(
        1
        for lf in drive_plan.regular_lfs
        if not (youtube_uploads.get(lf.index) or {}).get("video_id")
    )
    from .. import youtube_sidecar

    parts = youtube_sidecar.iter_superlong_parts(state)
    if not parts:
        return pending_lfs
    sidecar_uploads = youtube_sidecar.load_superlong_sidecar(output_base)
    pending_parts = sum(
        1 for part_key, _ in parts if not (sidecar_uploads.get(part_key) or {}).get("video_id")
    )
    return pending_lfs + pending_parts


def _refuse_if_over_youtube_quota_budget(*, pending_inserts: int, config: Config) -> None:
    """Refuse up front when the run's planned quota cost exceeds budget.

    Each pending upload spends one ``videos.insert`` plus one
    ``playlistItems.insert`` (every upload is routed through
    ``upload_video_with_playlist`` — see its "total by construction"
    docstring). Costs and the daily budget are read from
    ``config.drive``; when they aren't real numbers (a test
    double such as ``MagicMock()``) the check is skipped rather than
    raising on a non-numeric comparison.
    """
    drive_cfg = getattr(config, "drive", None)
    budget = getattr(drive_cfg, "youtube_quota_daily_budget", None)
    video_cost = getattr(drive_cfg, "youtube_video_insert_quota_cost", None)
    playlist_cost = getattr(drive_cfg, "youtube_playlist_insert_quota_cost", None)
    if not all(isinstance(v, (int, float)) for v in (budget, video_cost, playlist_cost)):
        return
    planned_cost = pending_inserts * (video_cost + playlist_cost)
    if planned_cost <= budget:
        return
    detail = (
        f"planned batch needs {pending_inserts} videos.insert + "
        f"playlistItems.insert pair(s) (~{planned_cost} quota units) > "
        f"the configured daily budget of {budget}. Reduce this run's "
        "LF/superlong-part count, split it across two days, or raise "
        "drive.youtube_quota_daily_budget if your project has a higher "
        "approved quota."
    )
    if not getattr(drive_cfg, "youtube_quota_enforce", False):
        # Warn-only default: the run proceeds and uploads what fits, then the
        # reactive 403 path pauses it and a resume finishes the remainder.
        # Refusing outright would strip that recovery from a batch that is
        # merely large, and the budget is Google's number rather than ours.
        logger.warning("YouTube quota projection: %s Proceeding (warn-only).", detail)
        return
    raise YouTubeQuotaError(
        reason="quotaExceeded",
        message=(
            f"{detail} Refusing before the first upload rather than aborting "
            "mid-batch (drive.youtube_quota_enforce is on)."
        ),
    )


def _upload_lfs_to_youtube(
    *,
    drive_plan,
    regular_lf_paths,
    metadata_per_lf: dict,
    drive_schedule,
    youtube_uploads: dict,
    state,
    output_base: Path,
    config: Config,
    save_state,
    skip_youtube_upload: bool,
    youtube_upload_enabled: bool,
    project_base: str | None = None,
):
    """Drive the per-LF YouTube upload loop. Returns ``(yt_instance_or_None, youtube_uploads)``.

    On ``YouTubeQuotaError``: persists ``youtube_uploads`` via
    ``save_state`` (orchestrator-supplied callback), logs the operator
    batch-paused banner, re-raises so the orchestrator records a
    graceful exit. Other per-LF failures are logged and skipped
    (handled inside ``_upload_lf_to_youtube``).
    """
    if skip_youtube_upload or not youtube_upload_enabled:
        if not youtube_upload_enabled:
            logger.info("  D-upload-lfs: SKIPPED (pass --youtube-upload to enable)")
        else:
            logger.info("  D-upload-lfs: SKIPPED")
        return None, youtube_uploads

    # Every publish_at comes from the schedule Phase D built over seo_manager's slot
    # decisions. No lf_slots means Phase D did not run (or was skipped): stop rather
    # than invent a cadence here.
    if not (drive_schedule and drive_schedule.lf_slots):
        raise RuntimeError(
            "D-upload-lfs: drive_schedule has no lf_slots — refusing to upload with "
            "a made-up publish date. Phase D builds lf_slots from seo_manager before "
            "any upload; re-run Phase D."
        )
    schedule_dates = [
        datetime.fromisoformat(s.datetime_est).astimezone(timezone.utc)
        for s in drive_schedule.lf_slots
    ]

    # Operate on state's OWN youtube_uploads dict. ``upload_if_needed``
    # writes each LF record directly into ``state.youtube_uploads``; a
    # divergent local working dict lets the per-LF ``state.youtube_uploads =
    # youtube_uploads`` reassignment below clobber a record whose LF then
    # failed its playlist insert (wrapper returned None) — resume re-runs
    # ``videos.insert`` and duplicates the scheduled video. Aliasing the local
    # to state's dict makes that clobber impossible by construction.
    if state.youtube_uploads is None:
        state.youtube_uploads = youtube_uploads or {}
    youtube_uploads = state.youtube_uploads

    # Refuse before the FIRST upload of a batch this run cannot
    # complete within the configured daily quota, rather than discovering
    # the same ceiling as a mid-batch HTTP 403 (see ``YouTubeQuotaError``
    # for the resume story — this raise is caught the same way).
    pending_inserts = _pending_youtube_insert_count(
        drive_plan=drive_plan,
        youtube_uploads=youtube_uploads,
        state=state,
        output_base=output_base,
    )
    _refuse_if_over_youtube_quota_budget(pending_inserts=pending_inserts, config=config)

    logger.info("  D-upload-lfs: Uploading LFs to YouTube...")
    try:
        YouTubeUploader = _resolve_youtube_uploader_class()  # noqa: N806
    except Exception as e:
        raise RuntimeError(
            "YouTubeUploader not available — install google-api-python-client "
            f"or check imports: {e}"
        ) from e

    yt = YouTubeUploader()
    operator_checklist_path = output_base / "operator_checklist.txt"
    yt.set_operator_checklist_path(operator_checklist_path)
    logger.info(
        "  D-upload-lfs: operator end-screen checklist → %s "
        "(open each link in Studio Editor, doc-only sidecar)",
        operator_checklist_path,
    )

    for i, (lf, lf_path) in enumerate(zip(drive_plan.regular_lfs, regular_lf_paths)):
        meta = metadata_per_lf.get(lf.index)
        if not meta or not meta.get("title"):
            raise ValueError(
                f"LF {lf.index}: no gated metadata title in metadata_per_lf "
                f"(present indices: {sorted(metadata_per_lf)}). Refusing to "
                "upload a 'Drive LF N' placeholder title / empty description "
                "that no gate reviewed to a scheduled public YouTube slot. "
                "Ensure phase_d metadata generation populated this LF before "
                "the upload phase runs."
            )
        title = meta["title"]
        description = meta.get("description", "")
        tags = meta.get("tags", [])
        publish_at = schedule_dates[i] if i < len(schedule_dates) else None
        try:
            record = _upload_lf_to_youtube(
                yt=yt,
                state=state,
                lf=lf,
                lf_path=Path(lf_path),
                title=title,
                description=description,
                tags=tags,
                publish_at=publish_at,
                metadata=meta,
                config=config,
                # Flush the checkpoint the instant ``videos.insert``
                # returns, before the fallible playlist assignment — a
                # live video's id must never exist only in memory.
                persist=lambda: save_state(state),
            )
        except YouTubeQuotaError as quota_exc:
            state.youtube_uploads = youtube_uploads
            save_state(state)
            uploaded_count = len(youtube_uploads)
            total_count = len(drive_plan.regular_lfs)
            logger.error(
                "\n"
                + "=" * 72
                + "\n"
                + "  YOUTUBE QUOTA HIT — batch paused for resume\n"
                + "=" * 72
                + "\n"
                + "  Reason: %s\n"
                + "  Uploaded so far: %d / %d LFs\n"
                + "  Recovery: wait for the YouTube Data API quota window to\n"
                + "            reset (midnight Pacific Time for daily quota, ~60s for\n"
                + "            per-user rate limit) then re-run the same\n"
                + "            command; already-uploaded LFs are skipped via\n"
                + "            ``state.youtube_uploads``.\n"
                + "=" * 72,
                quota_exc.reason,
                uploaded_count,
                total_count,
            )
            raise
        if record is None:
            # The wrapper swallows HTTP/unexpected errors, but the upload may
            # already have succeeded and written its record into
            # ``state.youtube_uploads`` before the failing step. Flush before
            # skipping: a ``continue`` that also skips ``save_state`` is what
            # left an irreversible upload unpersisted for the rest of the run.
            save_state(state)
            # An orphaned upload: videos.insert succeeded (record landed in
            # state.youtube_uploads via upload_if_needed's write-through)
            # even though the wrapper returned None because the SUBSEQUENT
            # playlist insert failed. The video is live and holds a publish
            # slot the ledger never records — reconcile_ledger_with_youtube
            # does not self-heal a missing entry, only an existing one. Book
            # it from the persisted record, never from the None return.
            orphan = youtube_uploads.get(lf.index)
            if orphan and orphan.get("video_id"):
                _book_lf_upload_in_ledger(
                    record=orphan,
                    state=state,
                    config=config,
                    project_base=project_base,
                    lf_index=lf.index,
                )
            continue
        youtube_uploads[lf.index] = record
        state.youtube_uploads = youtube_uploads
        save_state(state)
        # Transactional booking: ledger entry only AFTER the upload is
        # confirmed (record in hand). Best-effort + video_id-deduped inside.
        _book_lf_upload_in_ledger(
            record=record,
            state=state,
            config=config,
            project_base=project_base,
            lf_index=lf.index,
        )

    # Parts are excluded from lf_plans/drive_plan — reachable only here.
    _upload_superlong_parts_to_youtube(
        yt=yt,
        state=state,
        output_base=output_base,
        config=config,
        save_state=save_state,
        project_base=project_base,
    )

    return yt, youtube_uploads
