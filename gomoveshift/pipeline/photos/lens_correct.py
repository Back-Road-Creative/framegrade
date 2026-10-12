"""Lens-profile correction: a lensfun DB lookup keyed on the photo's
own EXIF camera/lens identity, applied as one distortion + vignetting +
transverse-CA correction before crop/straighten (``develop.develop_variants``)
so geometry measures the corrected frame.

The ``lensfunpy`` binding is an OPTIONAL extra (``pip install '.[lens]'``) —
absent it, or absent a matching profile for the photo's EXIF, this stage is a
clear, NAMED no-op: it never crashes the run and never guesses a profile.
``LensCorrectionResult.reason`` carries that decision back to the caller,
which threads it into ``DevelopedPhoto``/the grade sidecar the same way
``denoise_applied`` is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import lensfunpy

    LENSFUN_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by tests via monkeypatch
    lensfunpy = None  # type: ignore[assignment]
    LENSFUN_AVAILABLE = False

# Standard EXIF tag IDs (PIL's ``ExifTags.Base``): Make/Model live in the top
# IFD, LensModel and FocalLength in the EXIF sub-IFD (0x8769).
_EXIF_MAKE_TAG = 271
_EXIF_MODEL_TAG = 272
_EXIF_SUBIFD_TAG = 0x8769
_EXIF_LENS_MODEL_TAG = 42036
_EXIF_FOCAL_LENGTH_TAG = 37386


@dataclass(frozen=True)
class LensExif:
    """The EXIF identity a lensfun lookup keys on. Any missing tag is its
    field's default (empty string / 0.0), never a guess: an absent
    ``camera_make``/``camera_model`` is exactly the "no profile matched"
    case a caller must treat as a no-op, not a crash."""

    camera_make: str = ""
    camera_model: str = ""
    lens_model: str = ""
    focal_length_mm: float = 0.0


@dataclass(frozen=True)
class LensCorrectionResult:
    """``rgb`` is the input frame unchanged when ``applied`` is False —
    never a guess, never a partial correction. ``reason`` is the one place
    this stage's decision is recorded, whatever it decided."""

    rgb: np.ndarray
    applied: bool
    reason: str


def read_lens_exif(path) -> LensExif:
    """Camera make/model, lens model and focal length off ``path``'s EXIF.
    Any read failure (missing file, no EXIF, unreadable image) returns the
    all-empty default rather than raising — a photo with no lens EXIF is the
    same "no profile matched" case as one whose lens isn't in the DB."""
    from PIL import Image

    try:
        with Image.open(path) as img:
            exif = img.getexif()
    except (OSError, ValueError):
        return LensExif()

    make = str(exif.get(_EXIF_MAKE_TAG, "") or "").strip()
    model = str(exif.get(_EXIF_MODEL_TAG, "") or "").strip()

    try:
        sub_ifd = exif.get_ifd(_EXIF_SUBIFD_TAG)
    except (AttributeError, KeyError, ValueError):
        sub_ifd = {}

    lens_model = str(sub_ifd.get(_EXIF_LENS_MODEL_TAG, "") or "").strip()
    focal_length_mm = 0.0
    focal = sub_ifd.get(_EXIF_FOCAL_LENGTH_TAG)
    if focal is not None:
        try:
            focal_length_mm = float(focal)
        except (TypeError, ValueError):
            focal_length_mm = 0.0

    return LensExif(
        camera_make=make,
        camera_model=model,
        lens_model=lens_model,
        focal_length_mm=focal_length_mm,
    )


def _find_profile(exif: LensExif):
    """The lensfun DB lookup, isolated in its own function so a test can
    mock it without the real package installed. Returns ``(camera, lens)``
    on a match, ``None`` when either side of the lookup comes up empty —
    never the wrong camera's profile, never a guess."""
    db = lensfunpy.Database()
    cameras = db.find_cameras(exif.camera_make, exif.camera_model)
    if not cameras:
        return None
    camera = cameras[0]
    lenses = db.find_lenses(camera, exif.lens_model) if exif.lens_model else []
    if not lenses:
        return None
    return camera, lenses[0]


def _remap(rgb: np.ndarray, camera, lens, focal_length_mm: float) -> np.ndarray:
    """Distortion + vignetting + transverse-CA correction for the matched
    ``camera``/``lens`` pair, isolated in its own function so a test can mock
    it without the real package installed."""
    import cv2

    height, width = rgb.shape[:2]
    modifier = lensfunpy.Modifier(lens, camera.crop_factor, width, height)
    modifier.initialize(focal_length_mm or lens.min_focal, lens.min_aperture or 1.0, 1000.0)

    vignetted = rgb.copy()
    modifier.apply_color_modification(vignetted)

    tca_coords = modifier.apply_subpixel_distortion()
    if tca_coords is not None:
        corrected = np.empty_like(vignetted)
        for channel in range(3):
            corrected[..., channel] = cv2.remap(
                vignetted[..., channel],
                tca_coords[..., channel, 0],
                tca_coords[..., channel, 1],
                cv2.INTER_LANCZOS4,
            )
        return corrected

    undist_coords = modifier.apply_geometry_distortion()
    return cv2.remap(vignetted, undist_coords[..., 0], undist_coords[..., 1], cv2.INTER_LANCZOS4)


def correct(rgb: np.ndarray, path) -> LensCorrectionResult:
    """Correct ``rgb`` (a decoded photo frame) against ``path``'s
    EXIF-matched lensfun profile.

    A clear, named no-op — never a guess, never a crash — when the ``lens``
    extra is absent, the photo carries no camera EXIF, or no lensfun profile
    matches that EXIF identity."""
    if not LENSFUN_AVAILABLE:
        return LensCorrectionResult(rgb=rgb, applied=False, reason="lensfun extra not installed")

    exif = read_lens_exif(path)
    if not exif.camera_make or not exif.camera_model:
        return LensCorrectionResult(
            rgb=rgb, applied=False, reason="no camera EXIF to match a lens profile"
        )

    match = _find_profile(exif)
    if match is None:
        return LensCorrectionResult(
            rgb=rgb,
            applied=False,
            reason=(
                f"no lensfun profile for {exif.camera_make} {exif.camera_model} / "
                f"{exif.lens_model or 'unknown lens'}"
            ),
        )

    camera, lens = match
    corrected = _remap(rgb, camera, lens, exif.focal_length_mm)
    return LensCorrectionResult(
        rgb=corrected,
        applied=True,
        reason=f"corrected: {exif.camera_make} {exif.camera_model} / {lens.model}",
    )
