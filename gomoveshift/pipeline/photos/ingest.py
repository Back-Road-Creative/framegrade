"""Camera-card photo ingest: discover -> develop -> register in journey frontmatter.

Photo input was journey frontmatter only, so nothing could reach the pipeline
from a card. Ingest closes that at the source end and *registers* what it
develops, so the selector picks the photos up unchanged. Frontmatter
round-trips through ``journal_publisher`` (ruamel, comment preserving) and is
written atomically — a torn page blocks the site deploy — and only once the
whole card has developed, so a corrupt file can't half-register a day.
"""

from __future__ import annotations

import hashlib
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from .._atomic_write import atomic_write_text
from ..day_master_encoder import PRIVACY_VERIFIED
from ..journal_publisher import JourneyLockBusyError, journey_lock, load_journey, serialize_journey
from .cull import CullFloors, CullScore, compute_cull_score, cull_decision
from .develop import (
    PREVIEW_FALLBACK_DECODER,
    RAW_EXTENSIONS,
    STYLE_STRENGTH_DEFAULT,
    SUPPORTED_EXTENSIONS,
    compute_batch_corrections,
    decode,
    develop,
    export_jpeg,
)
from .exif_gps import (
    OPERATOR,
    PhotoLocation,
    derive_location_from_photos,
    extract_exif_capture_date,
    extract_exif_capture_metadata,
)
from .face_filter import FaceBlurError, attestable_privacy_classes, blur_faces_for_publish
from .preflight import preflight_photo_ingest
from .selector import IMAGES_BASE

logger = logging.getLogger(__name__)


class PhotoIngestError(RuntimeError):
    """Ingest refused to continue. Always names the offending file or day."""


#: Frontmatter key: gallery filename -> decoder tag; enables ``redevelop``.
PROVENANCE_KEY = "photo_provenance"

#: Frontmatter key: gallery filename -> recorded blur outcome for that export.
#: Written in the same registration step as ``photo_hashes``, so it lands or
#: doesn't land atomically with the rest of that photo's record. A filename
#: absent from this map was ingested before this field existed (or never
#: ingested through this path at all) — that is a FOURTH state, "not
#: recorded", and must never be confused with any of the three below.
BLUR_KEY = "photo_blur"
#: Blurring ran and found at least one privacy region.
BLUR_REGIONS_FOUND = "blurred_regions_found"
#: Blurring ran and found none.
BLUR_NO_REGIONS = "blurred_no_regions"
#: The ``--skip-blur`` escape shipped this export unblurred.
BLUR_SKIPPED = "skipped"

#: Frontmatter key: gallery filename -> required privacy classes this run
#: could NOT attest (``face_filter.attestable_privacy_classes``), beside
#: ``BLUR_KEY``, written in the same registration step. Storing only the
#: NOT-attested classes keeps the common case ("all four attested") a single
#: short ``[]`` line per photo instead of four positive entries. As with
#: ``BLUR_KEY``, a filename absent from this map is a FOURTH state — "not
#: recorded" (ingested before this field existed, or never through this
#: path) — and must never be read as "all fine" the way ``[]`` is.
PRIVACY_UNATTESTED_KEY = "photo_privacy_unattested"


@dataclass
class IngestReport:
    """What one card run developed, and what it deliberately left alone."""

    ingested: list[str] = field(default_factory=list)
    #: (source filename, reason) — reported, never silently re-developed.
    skipped: list[tuple[str, str]] = field(default_factory=list)
    #: The ONE location resolved for this card, as registered in frontmatter.
    #: Consume this; re-deriving a label from raw coordinates at a second site
    #: is the documented cause of far-road label bleed.
    location: str = ""
    #: Ingested photos whose export carried at least one blurred privacy region.
    blurred: int = 0
    #: True when ``skip_blur`` shipped this card's exports unblurred (operator escape).
    blur_skipped: bool = False
    #: ``DecodedPhoto.decoder`` tag -> develops that used it this run.
    decoder_counts: dict[str, int] = field(default_factory=dict)
    #: Gallery names whose preview-fallback export was replaced (``redevelop``).
    redeveloped: list[str] = field(default_factory=list)
    #: True when this report is a plan only (nothing decoded or written).
    dry_run: bool = False
    #: Newly developed frames that fail the cull floor (soft/clipped). They
    #: still land unless EVERY new frame fails, which refuses the card.
    cull_failed: int = 0


#: Cull floor applied to this run's newly developed frames (tests may patch).
CULL_FLOORS = CullFloors()


def _cull_refusal(card_dir, failed: list[tuple[str, str, CullScore]], floors: CullFloors) -> str:
    lines = [
        f"card {card_dir}: all {len(failed)} developed frame(s) fail the cull floor "
        "- nothing was ingested"
    ]
    for name, reason, score in failed:
        lines.append(
            f"  {name}: {reason} (sharpness {score.sharpness:.1f}, "
            f"white clip {score.exposure_white_fraction:.2f}, "
            f"black clip {score.exposure_black_fraction:.2f})"
        )
    lines.append(
        f"  floors: sharpness >= {floors.min_sharpness:g}, white clip <= "
        f"{floors.max_white_fraction:g}, black clip <= {floors.max_black_fraction:g}"
    )
    return "\n".join(lines)


def discover(card_dir) -> list[Path]:
    """Every supported photo on a card, recursively, in stable name order.
    Cards nest under ``DCIM/<n>``; sidecars, clips and card metadata are
    ignored rather than failed on."""
    card_dir = Path(card_dir)
    if not card_dir.is_dir():
        raise PhotoIngestError(f"{card_dir}: not a readable directory")
    found = [
        p for p in card_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(found, key=lambda p: str(p).lower())


def gallery_filename(source: Path, day_date: str | None = None) -> str:
    """Deterministic gallery name for a card file — the idempotency key. A
    re-inserted card resolves to the same name and is skipped, rather than
    developing a second copy under a fresh serial. ``day_date`` (the journey's
    ISO date token) yields the day-disambiguated form used when a recycled
    stem — file-counter reset, a second body, a later day — carries different
    content than the name already registered."""
    stem = source.stem.lower()
    return f"{day_date}-{stem}.jpg" if day_date else f"{stem}.jpg"


def _source_hash(source: Path) -> str:
    """SHA-256 of the source bytes — the content identity a skip is verified
    against. Streamed full-file; never derived from the developed output."""
    digest = hashlib.sha256()
    with open(source, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_checked_name(
    source: Path, day_date: str, registered: dict[str, str | None]
) -> tuple[str | None, str | None]:
    """The idempotency decision for one source: ``(gallery name to develop
    under, skip reason)`` — exactly one is non-None.

    A skip is only honest when it is the SAME pixels. Both name forms are
    skip candidates (legacy bare stem, day-disambiguated), and a registered
    candidate skips only when its recorded hash matches the card bytes.
    Different content behind the bare name develops under the day form; when
    that is taken by different content too, no honest name is left and the
    card hard-fails naming both. A legacy entry with no recorded hash keeps
    the skip — never a blind re-develop — but is loudly marked unverified.
    """
    primary = gallery_filename(source)
    prefixed = gallery_filename(source, day_date)
    candidate = primary if primary in registered else prefixed if prefixed in registered else None
    if candidate is None:
        return primary, None
    recorded = registered[candidate]
    if recorded is None:
        logger.warning(
            "%s: registered as %s with no recorded content hash — skip is unverified",
            source.name,
            candidate,
        )
        return None, f"already ingested as {candidate} (unverified)"
    digest = _source_hash(source)
    if digest == recorded:
        return None, f"already ingested as {candidate}"
    if candidate == primary:
        if prefixed not in registered:
            return prefixed, None
        if registered[prefixed] == digest:
            return None, f"already ingested as {prefixed}"
    raise PhotoIngestError(
        f"{source}: content differs from registered {candidate}, and the "
        f"day-disambiguated {prefixed} cannot take it — rename the source file"
    )


def _dedupe_raw_pairs(sources: list[Path]) -> tuple[list[Path], list[tuple[str, str]]]:
    """Collapse each same-directory RAW + developed-sibling pair onto the RAW.

    Pair shooting (DSC_0001.NEF + DSC_0001.JPG on one card) is a standard
    camera layout, not a collision: the JPEG is the camera's own develop of
    the same frame. Preferring the RAW here — upstream of any decode or
    export — keeps the gallery-name idempotency intact instead of failing
    the card after its first file already exported. Same-stem files of one
    kind (two JPEGs, two RAWs) are genuinely different content and keep the
    downstream collision hard-fail.
    """
    groups: dict[tuple[Path, str], list[Path]] = {}
    for source in sources:
        groups.setdefault((source.parent, source.stem.lower()), []).append(source)
    kept: list[Path] = []
    skipped: list[tuple[str, str]] = []
    for group in groups.values():
        raws = [p for p in group if p.suffix.lower() in RAW_EXTENSIONS]
        if len(raws) == 1 and len(group) > 1:
            kept.append(raws[0])
            skipped.extend(
                (p.name, f"RAW+JPEG pair: developed from {raws[0].name}")
                for p in group
                if p is not raws[0]
            )
        else:
            kept.extend(group)
    return sorted(kept, key=lambda p: str(p).lower()), skipped


def _partition_by_day(
    sources: list[Path], day_date: str
) -> tuple[list[Path], list[tuple[str, str]]]:
    """Keep only sources whose EXIF capture date matches ``day_date``.

    Cards are rarely wiped daily, so one card usually carries several days.
    Registering them all under one ``--day`` ships wrong travel dates and lets
    an out-of-day fix drag the card's ONE location vote across a county line —
    the far-road-bleed class. Runs upstream of any decode (cheap EXIF read
    only); an out-of-day file is skipped with its actual date, never silently
    registered. A file with no readable capture date ingests under ``--day``
    with a warning, so legacy/EXIF-less cards stay usable.
    """
    in_day: list[Path] = []
    skipped: list[tuple[str, str]] = []
    for source in sources:
        captured = extract_exif_capture_date(source)
        if captured is None:
            logger.warning(
                "%s: no readable EXIF capture date — ingesting under --day %s",
                source.name,
                day_date,
            )
            in_day.append(source)
        elif captured == day_date:
            in_day.append(source)
        else:
            skipped.append((source.name, f"captured {captured}, not {day_date}"))
    return in_day, skipped


def _registered_filenames(doc: dict) -> set[str]:
    """Every photo filename already registered anywhere in the journey."""
    days = doc.get("days", []) or []
    groups = [g for day in days for g in (day.get("photos", []) or [])]
    return {str(name) for g in groups for name in (g.get("photos", []) or [])}


def _registered_hashes(doc: dict) -> dict[str, str | None]:
    """Registered filename -> its recorded source hash, journey-wide (None
    for a legacy entry registered before hashes were persisted)."""
    hashes: dict[str, str | None] = {}
    for day in doc.get("days", []) or []:
        for group in day.get("photos", []) or []:
            recorded = group.get("photo_hashes") or {}
            for name in group.get("photos", []) or []:
                key = str(name)
                hashes[key] = str(recorded[key]) if key in recorded else hashes.get(key)
    return hashes


def _find_day(doc: dict, day_date: str) -> dict:
    for day in doc.get("days", []) or []:
        if str(day.get("date")) == day_date:
            return day
    known = sorted(str(d.get("date")) for d in doc.get("days", []) or [])
    raise PhotoIngestError(
        f"day {day_date} is not in this journey (days: {known}); author the stub first"
    )


def _location_group(day: dict, location: str) -> dict:
    groups = day.setdefault("photos", [])
    for group in groups:
        if str(group.get("location")) == location:
            return group
    groups.append({"location": location, "photos": []})
    return groups[-1]


def _resolve_location(operator_label, sources: list[Path]) -> PhotoLocation:
    """The ONE location for this card, resolved once and then persisted.

    Cascade mirrors the drive pipeline's: an operator label is authoritative
    and is registered verbatim without a geocode; otherwise the card's own
    EXIF fixes elect it, voted over the dominant county. There is no third
    rung — an unresolved card fails rather than defaulting, because a
    fabricated label is exactly how a non-US day ships under a US one.
    """
    label = str(operator_label or "").strip()
    if label:
        return PhotoLocation(location=label, reason=OPERATOR)
    return derive_location_from_photos(sources)


@contextmanager
def _journey_lock(journey_path: Path):
    """The shared per-journey write lock (:func:`journal_publisher.journey_lock`
    documents the mutual exclusion every journey writer now takes part in), in
    this path's error type. Ingest refuses rather than warn-and-skips: nothing
    irreversible has happened when it is taken — no decode, no gallery export —
    so a busy lock is a ``PhotoIngestError`` to re-run, never a clobber."""
    try:
        with journey_lock(journey_path):
            yield
    except JourneyLockBusyError as exc:
        raise PhotoIngestError(str(exc)) from exc


def _develop_styled(source, params, cluster_index, style, strength):
    """Bake ``style`` over the cluster's base correction from a SINGLE decode.

    The chosen style adapts to this frame's ``SceneProfile`` (chroma headroom,
    warmth, dynamic range) exactly as ``photos develop`` does, then grades
    through the shared develop chain. Decoding once and handing the decoded
    frame to both ``compute_scene_profile`` and ``develop`` avoids a second
    demosaic — ``develop`` accepts an already-decoded photo. An unknown style
    or non-positive strength raises ``ValueError`` (from ``apply_style``); the
    develop loop wraps it as ``PhotoIngestError`` naming the file.
    """
    from .scene import compute_scene_profile
    from .styles import apply_style

    decoded = decode(source)
    scene = compute_scene_profile(decoded.rgb, extract_exif_capture_metadata(decoded.path))
    styled = apply_style(style, params, scene, strength)
    return develop(decoded, styled, cluster_index)


def ingest(
    card_dir,
    journey_path,
    *,
    day_date: str,
    location: str | None = None,
    site_root=None,
    develop_fn=None,
    skip_blur: bool = False,
    redevelop: bool = False,
    dry_run: bool = False,
    style: str = "",
    style_strength: float = STYLE_STRENGTH_DEFAULT,
) -> IngestReport:
    """Develop every new photo on ``card_dir`` into ``journey_path``'s gallery.

    ``location`` is optional: without it the card's EXIF GPS elects the label.

    The card is **cluster-graded**: once the skip/dedupe/date decisions have
    picked the to-develop files, one bounded measure pass clusters them by
    lighting and every file in a cluster is graded with the cluster's shared
    correction, so a single scene can't drift shot to shot. Each file is still
    decoded, graded and exported one at a time — the measure pass returns
    parameters, never pixels, and a batch of graded full-resolution frames is
    never held in memory. A skipped file is never measured; a dry run skips
    the measure pass entirely (measuring is a decode); a redevelop replacement
    clusters with the rest of the run (alone, it forms a singleton cluster —
    its own per-photo grade). An injected ``develop_fn`` (the test seam) is
    called as ``develop_fn(source)``, bypasses the measure pass, and owns its
    own grading.

    ``style`` (empty by default) bakes one creative look over each cluster's
    base correction — the same scene-adaptive transforms ``photos develop``
    renders, from a single decode per frame. With no style the cluster base
    grades verbatim, byte-identical to before this option existed; ``style`` is
    ignored on the injected-``develop_fn`` test seam, which owns its grading.

    Faces, plates and vehicle occupants are blurred on each developed frame
    BEFORE its gallery JPEG is written — the gallery dir is rendered by the
    public site, so a frame whose privacy regions cannot be proven blurred
    raises before any write. ``skip_blur=True`` is the operator escape and is
    recorded on the report.

    Only files whose EXIF capture date matches ``day_date`` ingest: a card is
    rarely wiped daily, and an out-of-day file would register under the wrong
    travel date and pull the location vote off-county. Out-of-day files are
    skipped with their actual date; a file with no readable date ingests under
    ``day_date`` with a warning.

    Idempotent and content-aware: a file whose gallery name is already
    registered is skipped before any decode only when its recorded source
    hash matches the card bytes (a legacy entry with no hash keeps the skip
    with an unverified warning), so a re-run leaves the page untouched while
    a *different* photo behind a recycled stem develops under the
    day-disambiguated name instead of being silently dropped. Under
    ``redevelop``, a *recorded* preview-fallback develop runs again and its
    JPEG is replaced in place (never re-registered or re-voted);
    other/unrecorded provenance still skips. ``dry_run`` walks the same
    decision paths, returning the plan with nothing decoded or written.
    A RAW+JPEG sibling pair develops the RAW and reports the JPEG as skipped.
    Raises ``PhotoIngestError`` — naming the file or day, never a silent skip — on a
    disk/RAM preflight shortfall, an unknown day, same-kind colliding card
    names, a name collision no day prefix can split, a source that won't
    decode, a gallery export the filesystem refuses, or a card whose location
    can't be resolved.

    The whole run holds the per-journey file lock (see ``_journey_lock``), and
    on ANY failure the gallery JPEGs this run already exported are removed
    before re-raising — frontmatter registers only once at the end, so leaving
    them would strand unregistered residue in the public gallery dir that a
    re-run's idempotency check never reclaims. Pre-existing files are never
    touched.
    """
    journey_path = Path(journey_path)
    exported: list[Path] = []
    with _journey_lock(journey_path):
        try:
            return _ingest_locked(
                card_dir,
                journey_path,
                exported,
                day_date=day_date,
                location=location,
                site_root=site_root,
                develop_fn=develop_fn,
                skip_blur=skip_blur,
                redevelop=redevelop,
                dry_run=dry_run,
                style=style,
                style_strength=style_strength,
            )
        except BaseException:
            for path in exported:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    logger.warning("could not remove unregistered gallery export %s", path)
            raise


def _ingest_locked(
    card_dir,
    journey_path: Path,
    exported: list[Path],
    *,
    day_date: str,
    location: str | None,
    site_root,
    develop_fn,
    skip_blur: bool,
    redevelop: bool = False,
    dry_run: bool = False,
    style: str = "",
    style_strength: float = STYLE_STRENGTH_DEFAULT,
) -> IngestReport:
    """Body of :func:`ingest`, under the journey lock. Appends every gallery
    write to ``exported`` (cleared once frontmatter registers them) so the
    caller can remove this run's unregistered residue on any failure."""
    site_root = Path(site_root) if site_root else journey_path.parents[2]
    gallery_dir = site_root / IMAGES_BASE / journey_path.stem

    # Heaviest photo path — full-resolution decode + float64 grade transient
    # per frame — so gate disk/RAM up front, before the first decode or gallery
    # write, in the error type the CLI handler already surfaces. A dry run
    # decodes and writes nothing, so the gate is skipped.
    if not dry_run:
        try:
            preflight_photo_ingest(gallery_dir, blur_enabled=not skip_blur)
        except RuntimeError as exc:
            raise PhotoIngestError(f"preflight: {exc}") from exc

    doc, body = load_journey(journey_path)
    day = _find_day(doc, day_date)
    registered = _registered_hashes(doc)

    report = IngestReport(blur_skipped=skip_blur, dry_run=dry_run)
    sources, paired = _dedupe_raw_pairs(discover(card_dir))
    report.skipped.extend(paired)
    # Date partition after pair dedupe (a paired JPEG keeps its pair reason,
    # never a second date one) and before any decode. Everything downstream —
    # develop, registration, the ONE location vote over ``claimed`` — sees only
    # this day's files by construction.
    sources, out_of_day = _partition_by_day(sources, day_date)
    report.skipped.extend(out_of_day)
    recorded = doc.get(PROVENANCE_KEY) or {}
    new_provenance: dict[str, str] = {}
    claimed: dict[str, Path] = {}
    ingested_hashes: dict[str, str] = {}
    blur_states: dict[str, str] = {}
    privacy_unattested_states: dict[str, list[str]] = {}
    plan: list[tuple[Path, str, bool]] = []
    for source in sources:
        name, skip_reason = _content_checked_name(source, day_date, registered)
        replacing = False
        if name is None:
            # A verified (or legacy-unverified) re-insert would skip — unless
            # ``redevelop`` targets a recorded preview-fallback develop, which
            # re-runs in place under its registered name.
            candidate = gallery_filename(source)
            prior = str(recorded.get(candidate) or "")
            if redevelop and prior == PREVIEW_FALLBACK_DECODER:
                name = candidate
                replacing = True
            else:
                if dry_run:
                    skip_reason += f" (developed via {prior or 'unrecorded'})"
                report.skipped.append((source.name, skip_reason))
                continue
        if not replacing and name in claimed:
            # Two card files developing to one name would silently drop a photo.
            raise PhotoIngestError(f"{source} collides with {claimed[name]} (both -> {name})")
        if dry_run:  # the same decision path as a real run, up to the decode
            (report.redeveloped if replacing else report.ingested).append(name)
            if not replacing:
                claimed[name] = source
            continue
        if not replacing:
            claimed[name] = source
        plan.append((source, name, replacing))

    # Cluster-grade the survivors: one bounded measure pass elects a shared
    # correction per lighting cluster, so same-scene stills can't drift shot
    # to shot. The decisions above ran first — a skipped file is never
    # measured — and a dry run never reaches here (its plan stays empty), so
    # measuring never breaks the decode-nothing contract. An injected
    # ``develop_fn`` bypasses the measure pass and owns its own grading.
    corrections = None
    if plan and develop_fn is None:
        try:
            corrections = compute_batch_corrections([source for source, _, _ in plan])
        except Exception as exc:
            raise PhotoIngestError(f"lighting measure pass failed: {exc}") from exc

    # What this run can actually attest is a fact about config/detectors, not
    # any one photo — computed once and applied to every newly-ingested name
    # below, never per-photo, never gating (report only).
    privacy_classes = attestable_privacy_classes(blur=not skip_blur) if plan else {}
    unattested_this_run = sorted(
        cls for cls, info in privacy_classes.items() if info["state"] != PRIVACY_VERIFIED
    )

    cull_floors = CULL_FLOORS
    cull_results: list[tuple[str, str, CullScore, bool]] = []
    for index, (source, name, replacing) in enumerate(plan):
        try:
            if corrections is not None:
                # Grade one file at a time with its cluster's shared params —
                # never an all-at-once batch of graded 20 MP arrays. A chosen
                # style bakes over those params from a single decode; with no
                # style the cluster base grades verbatim (byte-identical).
                params, cluster_index = corrections[index]
                if style:
                    developed = _develop_styled(
                        source, params, cluster_index, style, style_strength
                    )
                else:
                    developed = develop(source, params, cluster_index)
            else:
                developed = develop_fn(source)
        except Exception as exc:  # decode/grade is a file-format boundary
            raise PhotoIngestError(f"{source}: could not be developed: {exc}") from exc
        # Blur BEFORE the write: the gallery dir is the public website surface,
        # so an export that can't be proven blurred must never land in it. The
        # grade has already quantized to uint8, so blurring the export-bound
        # array leaves the 16-bit develop untouched upstream.
        try:
            frame, regions = blur_faces_for_publish(developed.rgb, source.name, blur=not skip_blur)
        except FaceBlurError as exc:
            raise PhotoIngestError(f"{source}: {exc}") from exc
        # The per-photo fact this run just proved (or deliberately skipped
        # proving) — recorded so it outlives this function call. skip_blur
        # is checked directly, not ``regions``: a skipped photo never ran
        # detection at all, so 0 regions there means "not asked", never
        # "asked and found none".
        if skip_blur:
            blur_state = BLUR_SKIPPED
        elif regions:
            blur_state = BLUR_REGIONS_FOUND
        else:
            blur_state = BLUR_NO_REGIONS
        # An OSError here (missing/readonly gallery dir, disk pulled) must not
        # escape as a raw traceback — the CLI catches PhotoIngestError only.
        try:
            out = export_jpeg(frame, gallery_dir / name)
        except OSError as exc:
            raise PhotoIngestError(f"{source}: gallery export failed: {exc}") from exc
        if not replacing:
            score = compute_cull_score(developed.rgb)
            verdict = cull_decision(score, floors=cull_floors)
            cull_results.append((source.name, verdict.reason, score, verdict.keep))
            # Residue cleanup may unlink this run's exports on failure; a
            # REPLACED file is already-registered content, so it stays.
            exported.append(out)
        if decoder := str(getattr(developed, "decoder", "") or ""):
            new_provenance[name] = decoder
            report.decoder_counts[decoder] = report.decoder_counts.get(decoder, 0) + 1
        if replacing:
            report.redeveloped.append(name)
        else:
            ingested_hashes[name] = _source_hash(source)
            blur_states[name] = blur_state
            privacy_unattested_states[name] = unattested_this_run
            report.ingested.append(name)
        if regions:
            report.blurred += 1

    if cull_results:
        cull_bad = [(n, r, sc) for n, r, sc, keep in cull_results if not keep]
        report.cull_failed = len(cull_bad)
        # Before location resolution and any frontmatter mutation; the caller's
        # residue cleanup removes this run's exports on the raise.
        if len(cull_bad) == len(cull_results):
            raise PhotoIngestError(_cull_refusal(card_dir, cull_bad, cull_floors))

    if not report.ingested and not report.redeveloped:
        logger.info("Ingest: nothing new on %s (%d already in)", card_dir, len(report.skipped))
        return report

    if report.ingested:
        # Resolved after the whole card develops and before any mutation, so an
        # unresolvable location leaves the journey byte-identical — computed once,
        # persisted, never re-derived. A redevelop-only run has no vote to re-run.
        resolved = _resolve_location(location, list(claimed.values()))
        if not resolved.resolved:
            raise PhotoIngestError(
                f"day {day_date}: no location for this card ({resolved.reason}); "
                "pass an explicit location — one is never defaulted or invented"
            )
        report.location = resolved.location

    if dry_run:
        n_new, n_redo = len(report.ingested), len(report.redeveloped)
        logger.info("Ingest dry run: %d new, %d re-develop(s); nothing written", n_new, n_redo)
        return report

    if report.ingested:
        group = _location_group(day, resolved.location)
        group.setdefault("photos", []).extend(report.ingested)
        # Content identity travels with the entry: the hash a future skip is
        # verified against, recorded going forward (legacy entries have none).
        group.setdefault("photo_hashes", {}).update(ingested_hashes)
        # Same registration step as photo_hashes: lands or doesn't land
        # atomically with the rest of this photo's record.
        group.setdefault(BLUR_KEY, {}).update(blur_states)
        # Same registration step and grouping as BLUR_KEY, beside it.
        group.setdefault(PRIVACY_UNATTESTED_KEY, {}).update(privacy_unattested_states)
    # Mirrors the video_count contract, and only when the key already exists.
    if "photo_count" in doc:
        doc["photo_count"] = len(_registered_filenames(doc))
    if new_provenance:
        doc.setdefault(PROVENANCE_KEY, {}).update(new_provenance)
    atomic_write_text(journey_path, serialize_journey(doc, body))
    # Registered: these exports are now gallery content, no longer residue
    # candidates for the caller's failure cleanup.
    exported.clear()

    logger.info("Ingest: %d new photo(s) into %s", len(report.ingested), journey_path)
    return report
