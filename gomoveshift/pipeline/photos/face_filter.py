"""Still-photo privacy — faces, plates, vehicle occupants and whole persons,
blurred before publish.

The blur is applied to the copy that actually ships — the resized upload, and the
gallery export ``photos ingest`` writes — so the source photos on the card or in
the site checkout are never rewritten. Regions mirror the video blur
contract: faces ∪ aspect/area-gated plates ∪ the cabin band of plate-corroborated
vehicles ∪ whole-person boxes (a person photographed from behind, side-on, or with
an occluded face has no face box but is still caught here), built from the same
download-cached models and config gates. Fail CLOSED: a photo whose privacy
regions cannot be detected or blurred is not published. Operator opt-out is
``--skip-blur``.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from redactcam import (
    build_image_detectors,
    build_person_detector,
    build_vehicle_detector,
    cabin_region,
    detect_image,
    plate_inside,
    render_mask_frame,
)
from redactcam.detect import filter_vehicles

from ...analysis import (
    resolve_face_model,
    resolve_person_model,
    resolve_plate_model,
    resolve_vehicle_model,
)
from ...config import get_config
from ..day_master_encoder import (
    PRIVACY_INCONCLUSIVE,
    REQUIRED_PRIVACY_CLASSES,
    # Private on purpose and imported anyway: it is the ONE definition of
    # "a model that did not resolve is not a clean detection", and a stills
    # copy of that judgement is a copy that drifts while both keep returning
    # plausible states.
    _detector_class,
)

logger = logging.getLogger(__name__)

FACE_CONF = 0.4  # stills are cleaner than moving 4K, so the video threshold carries over
BLUR_FEATHER_PX = 12  # mask-edge feather so the blur blends instead of hard-cutting
BLUR_KERNEL_FRAC = 0.5  # kernel as a fraction of the box — scaled so a small face isn't just soft

_detectors = None  # ONNX construction dominates one still — build once, reuse per batch
_drive_cfg = None  # config.drive read once — the video blur contract's models and gates
_blur_enabled = True  # blur is ON by default; only the CLI opt-out turns it off


class FaceBlurError(RuntimeError):
    """Faces could not be detected or blurred — the photo must not be published."""


def set_blur_enabled(enabled: bool) -> None:
    """Apply the per-run operator opt-out (``--skip-blur``). Set once from the CLI,
    before any selection or resize, so no artifact is published under a stale default."""
    global _blur_enabled  # noqa: PLW0603
    _blur_enabled = enabled


def blur_enabled() -> bool:
    """True when detected faces will be blurred on published copies."""
    return _blur_enabled


def _resolve_blur(blur: Optional[bool]) -> bool:
    """``None`` means "take the per-run setting" — the sentinel is None rather than
    False so an explicit caller opt-out can't be re-stomped by the default."""
    return _blur_enabled if blur is None else blur


def _drive():
    """Cached ``config.drive`` — stills reuse the VIDEO blur contract's model
    resolution and gates verbatim, so the two surfaces can never drift."""
    global _drive_cfg  # noqa: PLW0603
    if _drive_cfg is None:
        _drive_cfg = get_config().drive
    return _drive_cfg


def _get_detectors():
    """Lazily build and cache the shared detector sessions.

    Same models and plate gates as the video blur: face/plate ONNX resolve through
    the download-once model cache and the plate aspect band carries over (real
    plates run wide; squarish road signs must stay readable). The third slot is
    the COCO vehicle detector for the occupant cabin blur (None when disabled)."""
    global _detectors  # noqa: PLW0603
    if _detectors is None:
        drive = _drive()
        vehicle = None
        if drive.blur_vehicle_cabin_frac > 0:
            # Session params mirror build_image_detectors (infer size / NMS iou / threads).
            vehicle = build_vehicle_detector(resolve_vehicle_model(drive), 640, 0.45, 0)
        person = None
        if getattr(drive, "blur_person_enabled", False):
            # Same session params as vehicle — whole-frame, no tiling (redactcam's own
            # contract for both COCO detectors).
            person = build_person_detector(resolve_person_model(drive), 640, 0.45, 0)
        _detectors = build_image_detectors(
            plate_model_path=resolve_plate_model(drive),
            face_model_path=resolve_face_model(drive),
            plate_min_aspect=drive.blur_plate_min_aspect,
            plate_max_aspect=drive.blur_plate_max_aspect,
        ) + (vehicle, person)
    return _detectors


def _cabin_boxes(rgb: np.ndarray, vehicle_detector, plates: List[tuple]) -> List[tuple]:
    """Occupant regions: the top ``blur_vehicle_cabin_frac`` of each PLATE-CORROBORATED
    vehicle box. Occupants can't be face-detected through a windshield, and a COCO
    "vehicle" with no plate inside its box is a billboard/sign misfire — only a plate
    proves a real car (the video cabin contract). Stills have no ego hood, so only the
    min-height gate applies (no bottom-strip cut)."""
    drive = _drive()
    h, w = rgb.shape[:2]
    vehicles = filter_vehicles(
        vehicle_detector.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), drive.blur_vehicle_conf),
        h,
        drive.blur_vehicle_min_height_frac,
        0.0,
    )
    boxes: List[tuple] = []
    for vehicle in vehicles:
        if not plate_inside(vehicle, plates):
            continue
        x0, y0, x1, y1 = cabin_region(vehicle, drive.blur_vehicle_cabin_frac, w, h)
        if x1 > x0 and y1 > y0:
            boxes.append((x0, y0, x1 - x0, y1 - y0))
    return boxes


def _person_boxes(rgb: np.ndarray, person_detector) -> List[tuple]:
    """Whole-person regions for the whole-person blur — catches a person photographed
    from behind, side-on, at distance, or with an occluded face, none of which produce
    a face box. Only the min-height gate applies (stills have no ego hood, so no
    bottom-strip cut, unlike the video cabin filter)."""
    drive = _drive()
    h = rgb.shape[0]
    boxes = person_detector.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), drive.blur_person_conf)
    return filter_vehicles(boxes, h, drive.blur_person_min_height_frac, 0.0)


def _detect_regions(rgb: np.ndarray, label: str) -> List[tuple]:
    """Blur boxes in an RGB frame: faces ∪ gated plates ∪ plate-corroborated vehicle
    cabins ∪ whole-person boxes — the video privacy contract applied to one still. A
    detector failure RAISES — it must never read as "no regions, ship it"."""
    try:
        detectors = _get_detectors()
        regions = detect_image(rgb, detectors=detectors, conf=FACE_CONF)
        boxes = list(regions.boxes)
        vehicle_detector = detectors[2] if len(detectors) > 2 else None
        if vehicle_detector is not None and regions.plates:
            boxes += _cabin_boxes(rgb, vehicle_detector, regions.plates)
        person_detector = detectors[3] if len(detectors) > 3 else None
        if person_detector is not None:
            boxes += _person_boxes(rgb, person_detector)
        return boxes
    except Exception as e:
        raise FaceBlurError(f"Privacy detection failed for {label}: {e}") from e


def attestable_privacy_classes(blur: Optional[bool] = None) -> dict:
    """What this run can actually attest for each of the four required
    privacy classes (``REQUIRED_PRIVACY_CLASSES``, imported from — never
    redeclared beside — the video attestation vocabulary in
    ``day_master_encoder``, so the two surfaces cannot drift apart).

    Returns ``{class: {"state": ..., "reason": ...}}``. A config flag that
    switches a class off does not make it "not required" — it is
    ``inconclusive`` with a reason naming the cause, mirroring the video
    contract's own rule. ``blur=False`` (``--skip-blur``) means nothing was
    asked at all, so every class is ``inconclusive`` — never a ``verified``
    for a class nothing ran for.

    **Reports; never gates.** It cannot change whether an ingest succeeds:
    a detector that cannot be built is recorded as ``inconclusive`` here and
    still raises, as it always did, from ``_detect_regions`` on the photo
    itself. Moving that failure earlier — or letting it escape unwrapped
    past ``PhotoIngestError`` — would make an attestation decide an outcome.
    """
    if not _resolve_blur(blur):
        reason = (
            "blur skipped by operator request (--skip-blur); nothing was asked, so nothing is known"
        )
        return {
            cls: {"state": PRIVACY_INCONCLUSIVE, "reason": reason}
            for cls in REQUIRED_PRIVACY_CLASSES
        }

    drive = _drive()
    # The same resolution the detectors are built from, so "verified" means
    # the model this run would actually use resolved -- not merely that blur
    # was switched on. Unconditional "verified" for faces/plates would be
    # this module reporting silence as a measurement, which is the exact
    # defect this function exists to end.
    result = {
        "faces": _detector_class("face", resolve_face_model(drive)),
        "plates": _detector_class("plate", resolve_plate_model(drive)),
    }

    if drive.blur_vehicle_cabin_frac <= 0:
        result["vehicle_occupants"] = {
            "state": PRIVACY_INCONCLUSIVE,
            "reason": (
                "occupant cabin blur is off (blur_vehicle_cabin_frac<=0); the "
                "class is required regardless, so this run cannot attest it"
            ),
        }
    else:
        result["vehicle_occupants"] = _detector_class("vehicle", resolve_vehicle_model(drive))

    if not getattr(drive, "blur_person_enabled", False):
        result["people"] = {
            "state": PRIVACY_INCONCLUSIVE,
            "reason": (
                "person blur is off (blur_person_enabled=false); the class is "
                "required regardless, so this run cannot attest it"
            ),
        }
    else:
        result["people"] = _detector_class("person", resolve_person_model(drive))

    return {cls: result[cls] for cls in REQUIRED_PRIVACY_CLASSES}


def blur_faces_for_publish(
    rgb: np.ndarray, label: str, blur: Optional[bool] = None
) -> Tuple[np.ndarray, int]:
    """Return (frame, privacy regions blurred) for the copy about to be published.

    Composites through the same feathered ``blur_mask`` raster the encode uses —
    the still analogue of ``maskedmerge``. Operates on an in-memory frame only:
    the caller writes the published artifact, the source photo is never touched.
    """
    if not _resolve_blur(blur):
        return rgb, 0

    boxes = _detect_regions(rgb, label)
    if not boxes:
        return rgb, 0

    h, w = rgb.shape[:2]
    k = max(3, int(max(max(b[2], b[3]) for b in boxes) * BLUR_KERNEL_FRAC) | 1)  # odd kernel
    alpha = (render_mask_frame(w, h, boxes, BLUR_FEATHER_PX).astype(np.float32) / 255.0)[:, :, None]
    out = (cv2.GaussianBlur(rgb, (k, k), 0) * alpha + rgb * (1.0 - alpha)).astype(np.uint8)
    logger.info("  %s: blurred %d privacy region(s)", label, len(boxes))
    return out, len(boxes)


def should_withhold_photo(photo_path: str, blur: Optional[bool] = None) -> bool:
    """Selection gate: True means WITHHOLD this photo from the batch.

    With blur on (the default) a privacy region is no reason to withhold — it is
    blurred on the published copy at resize time — so only an unreadable photo is
    withheld. Under ``--skip-blur`` the legacy detect-and-skip applies instead, now
    over ALL regions (a face, plate or corroborated cabin that can't be blurred must
    not ship). A missing file needs no gate (False); a detector failure raises.
    Nothing here writes to disk.
    """
    path = Path(photo_path)
    if not path.exists():
        logger.warning(f"Photo not found for face check: {photo_path}")
        return False

    img = cv2.imread(str(path))
    if img is None:  # unreadable, so unblurrable and unpublishable
        logger.warning(f"Could not read image, withholding: {photo_path}")
        return True

    if _resolve_blur(blur):
        return False
    return bool(_detect_regions(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), str(path)))
