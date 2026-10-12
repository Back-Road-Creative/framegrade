"""Photo resizer — scale photos to a configured width for social media upload.

Source stills have no fixed size: `photos ingest` develops full-resolution
camera RAW/HEIF/JPEG/TIFF onto the gallery, and hand-placed journey photos
vary too. Anything wider than the configured target width (1080px by
default) is scaled down before Google Drive upload; an already-clean,
undersized sRGB JPEG is copied through unresized.
"""

import io
import json
import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

GPS_IFD_TAG = 0x8825
# 4:2:2 keeps chroma edges (signage, foliage) cleaner than PIL's implicit
# 4:2:0 at a small size cost for a 1080px social upload; full 4:4:4 is
# reserved for the gallery masters these copies are cut from.
SOCIAL_JPEG_SUBSAMPLING = "4:2:2"
# GPS can ride the XMP APP1 packet or a JPEG comment even when the EXIF IFD is clean.
_GPS_MARKERS = (b"gpslatitude", b"gpslongitude", b"exif:gps")
# Optional social target aspects for the upload copy: label -> width/height.
# ONE output per photo, never a variant per platform: the publish path below the
# resizer (upload manifest -> Metricool CSV -> scheduler -> ContentLedger) is one
# resized file per selected photo.
SOCIAL_ASPECTS = {"square": 1.0, "4:5": 0.8, "9:16": 0.5625}


def _gps_in_xmp_or_comment(img) -> bool:
    """True when the image's XMP packet or JPEG comment carries GPS markers."""
    for key in ("xmp", "comment"):
        blob = img.info.get(key)
        if isinstance(blob, str):
            blob = blob.encode("utf-8", "surrogateescape")
        if blob and any(marker in blob.lower() for marker in _GPS_MARKERS):
            return True
    return False


def _photos_config():
    """Live photo config, resolved per call rather than at import time."""
    from ...config import get_config

    return get_config().photos


def _prepare(img):
    """Return (RGB image, ICC bytes to save or None), upright and colour-managed.

    Applies EXIF orientation, converts any source mode/colour space to sRGB RGB so
    the JPEG save cannot fail, and drops all source EXIF (including GPS).

    The returned profile is sRGB for a converted source and for a source that had
    no profile (matching the developed gallery exports); it is None only when a
    source profile was present but unusable, so unconverted pixels are never
    labelled sRGB.
    """
    from PIL import ImageCms, ImageOps

    from .develop import srgb_icc_profile

    img = ImageOps.exif_transpose(img) or img
    icc = img.info.get("icc_profile")

    if img.mode in ("P", "PA"):
        img = img.convert("RGBA" if "transparency" in img.info else "RGB")

    out_icc = None if icc else srgb_icc_profile()
    if icc:
        try:
            srgb = ImageCms.createProfile("sRGB")
            img = ImageCms.profileToProfile(
                img, ImageCms.ImageCmsProfile(io.BytesIO(icc)), srgb, outputMode="RGB"
            )
            out_icc = ImageCms.ImageCmsProfile(srgb).tobytes()
        except Exception as e:
            logger.warning(f"  Unusable ICC profile ({e}); colours assumed sRGB, saving untagged")

    if img.mode != "RGB":
        img = img.convert("RGB")
    # PIL transforms copy .info, and JPEG save re-attaches info["comment"] by
    # default — drop XMP/comment here so GPS can't ride them through a re-encode.
    for key in ("xmp", "comment"):
        img.info.pop(key, None)
    return img, out_icc


def aspect_ratio(label: str) -> Optional[float]:
    """Width/height for a social aspect label; None when unset ("" = off)."""
    if not label:
        return None
    if label not in SOCIAL_ASPECTS:
        raise ValueError(
            f"unknown social aspect {label!r} — must be one of: "
            f"{', '.join(SOCIAL_ASPECTS)} (or '' for the source's own aspect)"
        )
    return SOCIAL_ASPECTS[label]


def _centre_crop(img, ratio: float):
    """Largest centred rectangle of `ratio` (width/height) that fits inside `img`.

    Applied AFTER ``_prepare`` has run ``exif_transpose``, so the trim follows the
    upright frame — cropping the stored orientation would trim a rotated portrait
    along the wrong axis.
    """
    w, h = img.size
    if w > h * ratio:
        new_w, new_h = min(w, max(1, round(h * ratio))), h
    else:
        new_w, new_h = w, min(h, max(1, round(w / ratio)))
    if (new_w, new_h) == (w, h):
        return img
    left, top = (w - new_w) // 2, (h - new_h) // 2
    return img.crop((left, top, left + new_w, top + new_h))


def _is_clean_jpeg(img, photo_path: Path) -> bool:
    """True when the source is already a plain sRGB JPEG with no metadata to strip."""
    return (
        img.format == "JPEG"
        and img.mode == "RGB"
        and not img.info.get("icc_profile")
        and not dict(img.getexif())
        and not img.info.get("xmp")
        and not img.info.get("comment")
        and photo_path.suffix.lower() in (".jpg", ".jpeg")
    )


def _assert_no_gps(out_path: Path) -> None:
    """Hard-fail before upload if location metadata survived into the output."""
    from PIL import Image

    with Image.open(out_path) as check:
        if GPS_IFD_TAG in check.getexif():
            raise ValueError(f"GPS EXIF survived resize of {out_path.name}")
        if _gps_in_xmp_or_comment(check):
            raise ValueError(f"GPS XMP/comment survived resize of {out_path.name}")


#: Filename of the privacy attestation written beside each resized batch.
#: The still analogue of ``day_master_encoder.PRIVACY_ATTESTATION_NAME``: the
#: batch dir is where the published copies are written, so it is where the
#: record of what was proven about them belongs. Per WEEK, not per photo --
#: what a run can attest is a fact about config and detectors, never about one
#: frame, exactly as ``photos/ingest.py`` already treats it.
PHOTO_PRIVACY_ATTESTATION_NAME = "photos_privacy_attestation.json"


def _write_privacy_attestation(out_dir: Path, week_start: str) -> None:
    """Record what this run could attest for each required privacy class.

    Written in the vocabulary ``publishing/dispatch.py`` already reads --
    ``{"schema", "required_classes", "classes"}``, each class carrying
    ``state`` and ``reason`` -- so ``privacy_hold_reason`` can consume it with
    no translation. An attestation in a private dialect is a record nobody can
    act on.

    **Writes; never gates.** Whether a still export is HELD on an unattested
    class is the publisher's decision, and nothing here makes it. Fail-closed already
    lives at the reader: ``privacy_hold_reason(None)`` is a hold, so a missing
    or unreadable file is already the safe answer and this function has no
    reason to raise. The caller keeps it out of the batch's failure path for
    that reason -- a record that can abort an export is a record that decides
    an outcome.
    """
    # Imported here, not at module scope: ``face_filter`` pulls the detector
    # stack and (through it) the encoder, and this module is imported by paths
    # that never resize a thing.
    from ..day_master_encoder import PRIVACY_ATTESTATION_SCHEMA, REQUIRED_PRIVACY_CLASSES
    from .face_filter import attestable_privacy_classes

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / PHOTO_PRIVACY_ATTESTATION_NAME).write_text(
        json.dumps(
            {
                "schema": PRIVACY_ATTESTATION_SCHEMA,
                "week_start": week_start,
                "required_classes": list(REQUIRED_PRIVACY_CLASSES),
                "classes": attestable_privacy_classes(),
            },
            indent=2,
        )
    )


#: Per-batch record of which source photo each exported JPEG was made from.
#: A separate file from the privacy attestation beside it: that one is a
#: fact about the RUN (what config and detectors could attest), this one is a
#: fact about each FILE, and merging them would make a per-photo failure look
#: like a run-wide one.
PHOTO_PROVENANCE_NAME = "photos_provenance.json"

#: Recorded on a provenance entry whose source ICC profile was unusable: the file
#: was saved untagged and its colours were never converted.
ASSUMED_SRGB_NOTE = (
    "assumed sRGB: the source's ICC profile was unusable, so its colours were "
    "not converted and the file is saved untagged"
)

#: Schema of that record, versioned like the video sidecars so a reader can
#: tell a v1 document from whatever grows later.
PHOTO_PROVENANCE_SCHEMA = "gms.photo_provenance/v1"


def _source_fingerprint(path) -> Optional[str]:
    """Cheap ``size:mtime_ns`` identity of a source photo, or ``None`` when it
    cannot be stat'd.

    Deliberately the same shape as the day master's fingerprint, and for the
    same reason: enough to tell "a different file is now in play", never a
    tamper-proof hash. A gallery week is thousands of photos and hashing them
    all would cost more than the resize it is recording.
    """
    try:
        st = Path(path).stat()
    except OSError:
        return None
    return f"{st.st_size}:{st.st_mtime_ns}"


def _write_provenance(out_dir: Path, week_start: str, entries: list) -> None:
    """Write the batch's source-identity record.

    The published copy is not the source: it is centre-cropped, blurred,
    re-encoded and renamed. Given a Drive link and a complaint, nothing on disk
    led back to the original -- the same gap the video side closed for
    long-forms, shorts and thumbnails.

    **Records; never gates.** Absent evidence is null-with-reason, never an
    omitted entry, because an entry that quietly disappears reads like a photo
    that was never exported.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / PHOTO_PROVENANCE_NAME).write_text(
        json.dumps(
            {
                "schema": PHOTO_PROVENANCE_SCHEMA,
                "week_start": week_start,
                "photos": entries,
            },
            indent=2,
        )
    )


def _flush_provenance(out_dir: Path, week_start: str, entries: list) -> None:
    """Write the record, or log why it could not be written.

    Never raises. By the time this runs the JPEGs are on disk, and a record
    that can abort an export is a record deciding an outcome -- the same rule
    the privacy attestation beside it follows, and for the same reason.
    """
    try:
        _write_provenance(out_dir, week_start, entries)
    except Exception as exc:  # noqa: BLE001 -- a record must not undo an export
        logger.warning(
            "  source provenance not written for %s: %s — the exports in this "
            "batch cannot name the photos they were made from",
            week_start,
            exc,
        )


class PhotoResizer:
    """Resize photos to a target width for social media posting."""

    def __init__(
        self,
        target_width: Optional[int] = None,
        quality: Optional[int] = None,
        output_base: Optional[Path] = None,
        social_aspect: Optional[str] = None,
    ):
        # None means "take the configured value" — resolved here rather than
        # as a def-time default so config/env overrides are not frozen at import.
        photos = _photos_config()
        self.target_width = target_width if target_width is not None else photos.target_width
        self.quality = quality if quality is not None else photos.jpeg_quality
        self.output_base = output_base if output_base is not None else photos.resolve_resized_base()
        self.social_aspect = (
            social_aspect if social_aspect is not None else getattr(photos, "social_aspect", "")
        )
        aspect_ratio(self.social_aspect)  # fail closed on an unknown label, before any write

    def resize_batch(
        self,
        photo_paths: List[str],
        week_start: str,
        journey_slugs: Optional[List[str]] = None,
    ) -> List[Path]:
        """Resize a batch of photos to target width, maintaining aspect ratio.

        With ``social_aspect`` set, each photo is first centre-cropped to that
        target ratio (one output per photo, never a variant per platform).

        Args:
            photo_paths: Paths to source photos
            week_start: Week label for output subdirectory
            journey_slugs: Optional journey slug per photo — prefixed to output
                filename to prevent Drive link collisions between journeys
                sharing the same bare filename.

        Returns:
            List of paths to resized copies
        """
        if journey_slugs and len(journey_slugs) != len(photo_paths):
            raise ValueError(
                f"journey_slugs length ({len(journey_slugs)}) != "
                f"photo_paths length ({len(photo_paths)})"
            )

        import numpy as np
        from PIL import Image

        # Fail closed: an unimportable blur must abort the batch, never publish past it.
        from .face_filter import blur_faces_for_publish

        # None (the default) = ship the source's own aspect, byte-identical to
        # the aspect-preserving path below; resolved before the first mkdir.
        crop_ratio = aspect_ratio(self.social_aspect)

        out_dir = self.output_base / week_start
        out_dir.mkdir(parents=True, exist_ok=True)

        # Before the first decode, so a batch that dies on photo one still
        # leaves a record of what was proven about the bytes it did write --
        # the day master's "written on EVERY path" rule. Rewritten each run:
        # config can change between them, and a stale `verified` outliving the
        # run that earned it is the cached-evidence bug. A
        # write failure is logged and dropped: see _write_privacy_attestation.
        try:
            _write_privacy_attestation(out_dir, week_start)
        except Exception as exc:  # noqa: BLE001 -- the reader is what fails closed
            logger.warning(
                "  privacy attestation not written for %s: %s — a publish gate "
                "reading this batch will treat it as unattested",
                week_start,
                exc,
            )

        resized = []
        # One entry per exported file, naming the source it was made
        # from. Accumulated as we go rather than rebuilt at the end: the
        # loop is the only place that knows which of the two write paths
        # each photo took.
        provenance: List[dict] = []
        for i, photo_path_str in enumerate(photo_paths):
            photo_path = Path(photo_path_str)
            # Output bytes are always JPEG — force the extension so the Drive upload
            # can't advertise a source-derived MIME type it doesn't match.
            out_name = f"{photo_path.stem}.jpg"
            if journey_slugs and i < len(journey_slugs):
                out_name = f"{journey_slugs[i]}_{out_name}"
            out_path = out_dir / out_name
            tmp_path = out_path.with_name(f".{out_path.name}.tmp")

            try:
                with Image.open(photo_path) as img:
                    orig_w, orig_h = img.size
                    prepared, icc = _prepare(img)

                    # Compose for the platform ourselves rather than letting its
                    # auto-crop decide. Rides here, downstream of the orientation
                    # and colour handling and upstream of the blur/scale/gates.
                    if crop_ratio is not None:
                        prepared = _centre_crop(prepared, crop_ratio)
                        logger.info(
                            f"  {photo_path.name}: {self.social_aspect} centre-crop → "
                            f"{prepared.size[0]}x{prepared.size[1]}"
                        )

                    # Faces are blurred HERE, on the copy about to be published —
                    # the source photo in the site checkout is never rewritten.
                    frame, blurred = blur_faces_for_publish(np.array(prepared), photo_path.name)
                    if blurred:
                        prepared = Image.fromarray(frame)
                    elif (
                        crop_ratio is None
                        and orig_w <= self.target_width
                        and _is_clean_jpeg(img, photo_path)
                    ):
                        # Nothing to resize, strip, convert or blur — copy, don't re-encode
                        shutil.copyfile(photo_path, tmp_path)
                        _assert_no_gps(tmp_path)
                        os.replace(tmp_path, out_path)
                        logger.info(
                            f"  {photo_path.name}: {orig_w}x{orig_h} (copied, no re-encode)"
                        )
                        resized.append(out_path)
                        fingerprint = _source_fingerprint(photo_path)
                        provenance.append(
                            {
                                "exported": out_path.name,
                                "source": str(photo_path),
                                "source_fingerprint": fingerprint,
                                "null_reasons": (
                                    {}
                                    if fingerprint
                                    else {
                                        "source_fingerprint": (
                                            "the source could not be stat'd when the record "
                                            "was written -- moved or deleted after its export"
                                        )
                                    }
                                ),
                            }
                        )
                        continue

                    src_w, src_h = prepared.size

                    if src_w > self.target_width:
                        new_h = int(src_h * (self.target_width / src_w))
                        prepared = prepared.resize(
                            (self.target_width, new_h),
                            Image.LANCZOS,
                        )
                        logger.info(
                            f"  {photo_path.name}: {src_w}x{src_h} → {self.target_width}x{new_h}"
                        )
                    else:
                        logger.info(
                            f"  {photo_path.name}: {src_w}x{src_h} (no resize, quality optimized)"
                        )

                    # _prepare tags converted and profile-less sources sRGB
                    # (matching the developed gallery exports) and returns None
                    # when the source profile was unusable: those pixels were
                    # never converted, so they ship untagged, not mislabelled.
                    prepared.save(
                        tmp_path,
                        "JPEG",
                        quality=self.quality,
                        icc_profile=icc,
                        subsampling=SOCIAL_JPEG_SUBSAMPLING,
                    )

                # Atomic: only a fully written, GPS-clean, blurred file becomes the artifact.
                _assert_no_gps(tmp_path)
                os.replace(tmp_path, out_path)
                resized.append(out_path)
                fingerprint = _source_fingerprint(photo_path)
                entry = {
                    "exported": out_path.name,
                    "source": str(photo_path),
                    "source_fingerprint": fingerprint,
                    "null_reasons": (
                        {}
                        if fingerprint
                        else {
                            "source_fingerprint": (
                                "the source could not be stat'd when the record "
                                "was written -- moved or deleted after its export"
                            )
                        }
                    ),
                }
                if icc is None:
                    entry["color_assumption"] = ASSUMED_SRGB_NOTE
                provenance.append(entry)

            except Exception as e:
                tmp_path.unlink(missing_ok=True)
                logger.error(f"  Failed to resize {photo_path.name}: {e}")
                # The bytes that already reached disk are facts this
                # failure does not unmake -- record them before propagating.
                _flush_provenance(out_dir, week_start, provenance)
                raise

        logger.info(f"Resized {len(resized)} photos to {out_dir}")
        _flush_provenance(out_dir, week_start, provenance)
        return resized
