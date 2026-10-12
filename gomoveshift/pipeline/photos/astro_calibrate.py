"""Linear Bayer decode + dark calibration for astro stacking.

Astro stacking needs SCENE-LINEAR light, not display-referred sRGB: darks
subtract cleanly only in linear space, and the later stretch (astro_develop.py)
needs full sensor headroom, not develop.py's tone-mapped output. This module
is develop.py's ``_RAW_POSTPROCESS`` with the gamma curve
swapped for identity, plus the pure per-pixel math a caller applies to the raw
mosaic BEFORE demosaic: dark subtraction, then a hot-pixel cosmetic pass.
Hot pixels are fixed in SENSOR coordinates, so an unaligned ground stack's
median keeps them where an aligned sky stack's sigma-clip would reject them.

Kept thin: rawpy is only imported inside ``decode_linear``, lazily, the way
develop.py's decoders do it; every other function is
pure numpy, so tests never need a real RAW file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# develop._RAW_POSTPROCESS (display sRGB) with gamma replaced by (1, 1):
# scene-linear light, still 16-bit / no_auto_bright / camera WB.
_LINEAR_POSTPROCESS = {
    "output_bps": 16,
    "no_auto_bright": True,
    "gamma": (1, 1),
    "use_camera_wb": True,
}

# Hot-pixel threshold: this many robust-sigma above the master dark's own
# median. Provisional — to be tuned against a real night card + darks, the
# same tune-against-the-real-card discipline bracket.py's constants document.
HOT_PIXEL_SIGMA_K = 6.0

# Scales median absolute deviation to a normal-consistent sigma estimate
# (1 / Phi^-1(0.75)). MAD's breakdown point is 50% of pixels vs. raw std's 0%,
# so the hot pixels this function is LOOKING for can't corrupt the estimate
# of what "normal" is in the first place.
_MAD_TO_SIGMA = 1.4826

# Same-CFA-color neighbors on a standard Bayer mosaic: the pattern repeats
# every 2x2, so a neighbor keeps the photosite's channel iff both offsets are
# even. These four are the nearest such neighbors on the sensor's own grid.
_NEIGHBOR_OFFSETS = ((-2, 0), (2, 0), (0, -2), (0, 2))

# No-darks hot-pixel correction: PROVISIONAL, pending a reference night card. Robust-sigma threshold on a photosite's
# LOCAL residual (value minus its same-CFA neighbors' median, per frame).
# Suppresses smooth illumination GRADIENTS (checked to ~50 ADU/px) -- NOT
# curvature at the 2px neighbor scale: stars, blurred lights, ridgelines
# and edges all flag persistently too.
NO_DARK_LOCAL_SIGMA_K = 6.0

# Share of frames a site must clear the local-residual test in before it's
# trusted as persistent. NOT a star-drift bound -- a PSF still wins the SAME
# site most frames at the documented ~1-2px/frame floor; see the docstring.
NO_DARK_PERSISTENCE_SHARE = 0.8

# Below this many frames there's not enough independent evidence to trust
# persistence -- warn and correct nothing (a duplicate-cluster run).
NO_DARK_MIN_FRAMES = 5


class AstroCalibrationError(ValueError):
    """Calibration input can't be used (mixed dark shapes, no darks, ...)."""


class AstroDarkIdentityMismatchError(AstroCalibrationError):
    """A dark's capture identity (camera/shape/CFA/ISO/exposure) doesn't match
    the lights it would calibrate -- a mismatched master crushes signal."""


# PROVISIONAL: dark current roughly doubles every ~6-7 C; a gap this large
# warrants a warning, not a refusal -- cross-session darks are normal here.
DARK_TEMPERATURE_WARN_DELTA_C = 5.0

# Refused on mismatch; temperature is separate (warn, not refuse -- see above).
_STRICT_IDENTITY_FIELDS = ("camera", "raw_shape", "cfa_pattern", "iso", "exposure_s")


class AstroHighEfficiencyError(RuntimeError):
    """An HE/HE* NEF exposes no Bayer mosaic — no calibration is possible.

    ``read_raw_mosaic`` (darks) retries through DNG conversion itself, right
    here, mirroring ``astro_orchestrator.decode_with_he_fallback``'s
    light-frame path. ``decode_linear`` (lights) still relies on its caller
    for that retry: ``decode_with_he_fallback`` catches this exception and
    retries via DNG conversion before giving up. Neither path ever falls
    back to an embedded preview, which carries no mosaic to calibrate.
    """


@dataclass(frozen=True)
class DecodedAstroFrame:
    """A scene-linear decode plus a small record of how it was made."""

    rgb: np.ndarray
    path: Path
    decoder: str
    bit_depth: int
    flat_field_clipped_pixels: int = 0


@dataclass(frozen=True)
class CaptureIdentity:
    """One RAW file's capture identity, compared field by field in
    ``validate_dark_identity``; ``None`` means "unreadable", skipped not mismatched.
    """

    path: Path
    camera: str | None
    raw_shape: tuple[int, ...]
    cfa_pattern: tuple[int, ...] | None
    iso: float | None
    exposure_s: float | None
    temperature_c: float | None = None


_IFD0_BYTE_ORDER = {b"II\x2a\x00": "<", b"MM\x00\x2a": ">"}
_IFD0_MAKE, _IFD0_MODEL = 0x010F, 0x0110
_EXIF_SUB_IFD, _AMBIENT_TEMPERATURE = 0x8769, 0x9400


def _camera_label_and_temperature(path: Path) -> tuple[str | None, float | None]:
    """(camera, temperature_c) off the RAW container's TIFF IFD0 and Exif
    sub-IFD, one open -- rawpy exposes neither. Camera is IFD0 Make/Model;
    temperature is EXIF AmbientTemperature (0x9400) in the sub-IFD
    ``exif_gps.py`` walks for ISO/exposure. ``None`` if unreadable/unwritten.
    """
    import struct

    from PIL import TiffImagePlugin

    try:
        with open(path, "rb") as fh:
            header = fh.read(8)
            byte_order = _IFD0_BYTE_ORDER.get(header[:4])
            if byte_order is None:
                return None, None
            fh.seek(struct.unpack(byte_order + "I", header[4:8])[0])
            ifd0 = TiffImagePlugin.ImageFileDirectory_v2(header)
            ifd0.load(fh)
            make, model = ifd0.get(_IFD0_MAKE), ifd0.get(_IFD0_MODEL)
            temperature_c, exif_at = None, ifd0.get(_EXIF_SUB_IFD)
            if exif_at:
                fh.seek(int(exif_at[0] if isinstance(exif_at, tuple) else exif_at))
                sub = TiffImagePlugin.ImageFileDirectory_v2(header)
                sub.load(fh)
                temp = sub.get(_AMBIENT_TEMPERATURE)
                temperature_c = float(temp) if temp is not None else None
    except Exception as exc:  # unreadable/corrupt file -- degrade, never raise
        logger.debug("Capture identity read failed for %s: %s", path, exc)
        return None, None
    label = " ".join(str(v).strip() for v in (make, model) if v and str(v).strip())
    return label or None, temperature_c


def read_capture_identity(path) -> CaptureIdentity:
    """This RAW file's ``CaptureIdentity`` (lazy rawpy import, metadata-only
    ``sizes``/``raw_pattern``/``other`` -- cheap even on a dark needing DNG).
    Camera/temperature come off a second TIFF read; an unwritten tag reads ``None``.

    Raises:
        AstroCalibrationError: the file could not be opened at all.
    """
    import rawpy

    path = Path(path)
    try:
        with rawpy.imread(str(path)) as raw:
            pattern = raw.raw_pattern
            cfa = tuple(int(v) for v in pattern.flatten()) if pattern is not None else None
            other = raw.other
            iso = float(other.iso_speed) if other.iso_speed else None
            exposure_s = float(other.shutter_speed) if other.shutter_speed else None
            raw_shape = (int(raw.sizes.raw_height), int(raw.sizes.raw_width))
    except rawpy.LibRawError as exc:
        raise AstroCalibrationError(f"{path}: could not open RAW file ({exc})") from exc

    camera, temperature_c = _camera_label_and_temperature(path)
    return CaptureIdentity(
        path=path,
        camera=camera,
        raw_shape=raw_shape,
        cfa_pattern=cfa,
        iso=iso,
        exposure_s=exposure_s,
        temperature_c=temperature_c,
    )


def validate_dark_identity(light: CaptureIdentity, dark: CaptureIdentity) -> None:
    """Refuse ``dark`` if camera/raw_shape/cfa_pattern/iso/exposure_s disagrees
    with ``light`` (a field either side couldn't read is skipped, not treated
    as a mismatch). Temperature never refuses: missing on either side skips,
    a large-but-present difference only warns (see ``DARK_TEMPERATURE_WARN_DELTA_C``).
    """
    for field in _STRICT_IDENTITY_FIELDS:
        light_val, dark_val = getattr(light, field), getattr(dark, field)
        if light_val is None or dark_val is None:
            continue
        if light_val != dark_val:
            raise AstroDarkIdentityMismatchError(
                f"{dark.path}: dark {field} does not match light {light.path} -- "
                f"light={light_val!r} dark={dark_val!r}"
            )
    if light.temperature_c is not None and dark.temperature_c is not None:
        delta = abs(light.temperature_c - dark.temperature_c)
        if delta > DARK_TEMPERATURE_WARN_DELTA_C:
            logger.warning(
                "%s: dark sensor temperature %.1fC is %.1fC from light %s's %.1fC "
                "(over the %.1fC provisional threshold) -- prefer a same-session dark",
                dark.path,
                dark.temperature_c,
                delta,
                light.path,
                light.temperature_c,
                DARK_TEMPERATURE_WARN_DELTA_C,
            )


def build_master_dark(
    dark_mosaics, *, light_identity=None, dark_identities=None, master_bias=None
) -> np.ndarray:
    """Per-pixel median of >=1 raw Bayer mosaics; uint16 in, float32 out.

    Median (not mean) rejects a stray cosmic-ray hit in one dark without
    dragging the calibration frame off the sensor's true dark level.

    ``light_identity``/``dark_identities`` default ``None`` (today's behavior,
    unchanged) -- an opt-in check: supply both (one ``CaptureIdentity``
    per dark, same order) to validate every dark before the median is built.
    Supplying only one of the pair raises rather than silently half-skipping.

    ``master_bias`` (opt-in) is subtracted from every dark mosaic BEFORE
    the median, per standard flat/bias calibration practice; omitted, behavior
    is unchanged (today's bias-blind master dark, still carrying the black
    pedestal that ``subtract_dark``/``decode_linear`` expect).
    """
    if (light_identity is None) != (dark_identities is None):
        raise AstroCalibrationError(
            "build_master_dark: light_identity and dark_identities must be supplied "
            "together (opt in to dark-identity validation) or not at all"
        )
    mosaics = [np.asarray(m) for m in dark_mosaics]
    if not mosaics:
        raise AstroCalibrationError("build_master_dark: no dark frames supplied")
    if dark_identities is not None:
        if len(dark_identities) != len(mosaics):
            raise AstroCalibrationError(
                f"build_master_dark: {len(dark_identities)} dark_identities for "
                f"{len(mosaics)} dark_mosaics -- need exactly one per dark, same order"
            )
        for dark_identity in dark_identities:
            validate_dark_identity(light_identity, dark_identity)
    shape = mosaics[0].shape
    for m in mosaics:
        if m.shape != shape:
            raise AstroCalibrationError(
                f"build_master_dark: mixed mosaic shapes ({shape} vs {m.shape})"
            )
    if master_bias is not None:
        bias_arr = np.asarray(master_bias, dtype=np.float32)
        if bias_arr.shape != shape:
            raise AstroCalibrationError(
                f"build_master_dark: master_bias {bias_arr.shape} != dark {shape}"
            )
        stack = np.stack([m.astype(np.float32) - bias_arr for m in mosaics], axis=0)
    else:
        stack = np.stack([m.astype(np.float32) for m in mosaics], axis=0)
    return np.median(stack, axis=0).astype(np.float32)


def build_master_bias(bias_mosaics) -> np.ndarray:
    """Per-pixel median of >=1 zero-exposure (cap-on, shortest shutter) raw Bayer
    mosaics; uint16 in, float32 out -- the bias half of the flat/bias pair,
    same median-not-mean rationale as ``build_master_dark``.
    """
    mosaics = [np.asarray(m) for m in bias_mosaics]
    if not mosaics:
        raise AstroCalibrationError("build_master_bias: no bias frames supplied")
    shape = mosaics[0].shape
    for m in mosaics:
        if m.shape != shape:
            raise AstroCalibrationError(
                f"build_master_bias: mixed mosaic shapes ({shape} vs {m.shape})"
            )
    stack = np.stack([m.astype(np.float32) for m in mosaics], axis=0)
    return np.median(stack, axis=0).astype(np.float32)


def build_master_flat(flat_mosaics, *, master_bias=None) -> np.ndarray:
    """Master flat: per-pixel median of (flat - bias) frames, normalised
    to its own median so the result centres on 1.0 -- dividing a light by it
    corrects vignetting/PRNU without changing the light's overall level.
    ``master_bias`` omitted skips the bias subtraction (median of the raw
    flats is normalised directly).
    """
    mosaics = [np.asarray(m) for m in flat_mosaics]
    if not mosaics:
        raise AstroCalibrationError("build_master_flat: no flat frames supplied")
    shape = mosaics[0].shape
    for m in mosaics:
        if m.shape != shape:
            raise AstroCalibrationError(
                f"build_master_flat: mixed mosaic shapes ({shape} vs {m.shape})"
            )
    if master_bias is not None:
        bias_arr = np.asarray(master_bias, dtype=np.float32)
        if bias_arr.shape != shape:
            raise AstroCalibrationError(
                f"build_master_flat: master_bias {bias_arr.shape} != flat {shape}"
            )
        stack = np.stack([m.astype(np.float32) - bias_arr for m in mosaics], axis=0)
    else:
        stack = np.stack([m.astype(np.float32) for m in mosaics], axis=0)
    combined = np.median(stack, axis=0).astype(np.float32)
    median = float(np.median(combined))
    if median <= 0:
        raise AstroCalibrationError(
            "build_master_flat: master flat median is <= 0 -- cannot normalize"
        )
    return (combined / median).astype(np.float32)


# Divide-by-small guard: a normalised flat pixel below this floor is
# clipped up to it before a light is divided by the flat -- a near-dead or
# extreme-vignette photosite would otherwise blow the light up rather than
# correct it. Provisional, like HOT_PIXEL_SIGMA_K -- pending a real flat card.
FLAT_FIELD_FLOOR = 0.1


@dataclass(frozen=True)
class FlatFieldResult:
    """``apply_flat_field``'s output plus the divide-by-small guard's receipt."""

    mosaic: np.ndarray
    clipped_pixel_count: int


def apply_flat_field(
    mosaic: np.ndarray, master_flat: np.ndarray, *, floor: float = FLAT_FIELD_FLOOR
):
    """Divide ``mosaic`` (already dark/bias-subtracted) by ``master_flat``
    (normalised to ~1.0 by ``build_master_flat``), correcting vignetting/PRNU.

    Normalised flat pixels below ``floor`` are clipped up to it before the
    division -- guards a near-dead or extreme-vignette photosite from being
    amplified rather than corrected -- and the clipped count is reported on
    the returned ``FlatFieldResult`` so a caller can log/record it.
    """
    if mosaic.shape != master_flat.shape:
        raise AstroCalibrationError(
            f"apply_flat_field: mosaic {mosaic.shape} != master_flat {master_flat.shape}"
        )
    flat = np.asarray(master_flat, dtype=np.float32)
    below_floor = flat < floor
    clipped_count = int(below_floor.sum())
    safe_flat = np.where(below_floor, np.float32(floor), flat)
    result = np.asarray(mosaic, dtype=np.float32) / safe_flat
    if clipped_count:
        logger.warning(
            "apply_flat_field: %d pixel(s) below the flat floor %.3f were clipped "
            "up to it before dividing",
            clipped_count,
            floor,
        )
    return FlatFieldResult(mosaic=result.astype(np.float32), clipped_pixel_count=clipped_count)


def read_raw_mosaic(path, *, he_cache: set[Path] | None = None) -> np.ndarray:
    """Read one RAW file's Bayer mosaic as a standalone uint16 array -- the
    path-reading side for ``build_master_dark``'s in-memory arrays. Copied,
    since ``raw.raw_image`` is a view rawpy frees on context exit. Lazy
    ``rawpy`` import, same convention as ``decode_linear`` below.

    An HE/HE* dark (no Bayer mosaic) retries through DNG conversion before
    giving up -- an HE body shoots HE darks too, and they hit the identical
    LibRaw wall lights do. ``he_cache``, when passed, is a per-invocation set
    of paths already proven HE this run (mutated in place): a later dark in
    the same set skips straight to the DNG probe instead of paying the
    failed direct-decode attempt again. Never process-global -- the caller
    owns the set's lifetime.

    Raises:
        AstroCalibrationError: the file could not be opened at all (missing,
            unsupported, corrupt) -- an open failure, not an HE mosaic gap.
        AstroHighEfficiencyError: neither the file nor its DNG conversion
            exposes a Bayer mosaic.
    """
    import rawpy

    path = Path(path)
    if he_cache is not None and path in he_cache:
        return _dng_mosaic(path, None)

    try:
        raw_ctx = rawpy.imread(str(path))
    except rawpy.LibRawError as exc:
        raise AstroCalibrationError(f"{path}: could not open RAW file ({exc})") from exc

    try:
        with raw_ctx as raw:
            return raw.raw_image.copy()
    except rawpy.LibRawError as exc:
        if he_cache is not None:
            he_cache.add(path)
        return _dng_mosaic(path, exc)


def _dng_mosaic(path: Path, probe_exc: Exception | None) -> np.ndarray:
    """Shared DNG-fallback tail for ``read_raw_mosaic``'s fresh-probe and
    cache-hit routes: convert then re-read the mosaic, or raise
    ``AstroHighEfficiencyError`` naming the file and what the operator
    should do about it."""
    import rawpy

    from .develop import _convert_he_to_dng  # lazy: no develop<->astro import cycle

    dng_path = _convert_he_to_dng(path)
    if dng_path is None:
        raise AstroHighEfficiencyError(
            f"{path}: raw stream undecodable"
            + (f" ({probe_exc})" if probe_exc is not None else "")
            + " and no DNG converter recovered it (configure "
            "photos.he_dng_converter_exe) — this dark cannot be calibrated; "
            "shoot darks on a non-HE body or drop this file from --darks"
        ) from probe_exc
    try:
        with rawpy.imread(str(dng_path)) as raw:
            return raw.raw_image.copy()
    except rawpy.LibRawError as dng_exc:
        raise AstroHighEfficiencyError(
            f"{path}: converted DNG {dng_path} also exposes no Bayer mosaic "
            f"({dng_exc}) — this dark cannot be calibrated; shoot darks on a "
            "non-HE body or drop this file from --darks"
        ) from dng_exc


def hot_pixel_map(master_dark: np.ndarray) -> np.ndarray:
    """Bool map of photosites stuck/hot in the master dark: > median + K*sigma."""
    median = np.median(master_dark)
    sigma = _MAD_TO_SIGMA * np.median(np.abs(master_dark - median))
    return master_dark > median + HOT_PIXEL_SIGMA_K * sigma


def _local_cfa_outliers(mosaic: np.ndarray) -> np.ndarray:
    """One frame's local-residual outlier map, sensor coordinates, same-CFA
    neighbors only (``_NEIGHBOR_OFFSETS``) -- never demosaiced RGB, never an
    aligned frame (a moving star would line up and misread as fixed). A
    same-CFA neighbor within 2px of the sensor edge is dropped, never
    reflected (reflecting a gradient would fold it into a same-sign residual
    every frame -- indistinguishable from a real defect).
    """
    mosaic = np.asarray(mosaic, dtype=np.float32)
    height, width = mosaic.shape
    padded = np.pad(mosaic, 2, mode="constant", constant_values=np.nan)
    neighbor_stack = np.stack(
        [padded[2 + dr : 2 + dr + height, 2 + dc : 2 + dc + width] for dr, dc in _NEIGHBOR_OFFSETS],
        axis=0,
    )
    residual = mosaic - np.nanmedian(neighbor_stack, axis=0)
    center = np.median(residual)
    sigma = _MAD_TO_SIGMA * np.median(np.abs(residual - center))
    if sigma <= 0:
        # MAD's 50% breakdown point: a clipped sky/flat region can collapse
        # sigma to 0 -- warn and flag nothing rather than guess.
        logger.warning(
            "no_dark_hot_pixel_map: local-residual scale collapsed to zero "
            "for one frame (sky/foreground too flat to estimate sigma) -- "
            "flagging nothing for that frame"
        )
        return np.zeros(mosaic.shape, dtype=bool)
    return residual > center + NO_DARK_LOCAL_SIGMA_K * sigma


def no_dark_hot_pixel_map(mosaics) -> np.ndarray:
    """Persistent-defect CANDIDATE map for lights with NO darks --
    detection only, never correction; independent of the darks path
    (``hot_pixel_map``/``build_master_dark``), which is unaffected by it.

    Each of ``mosaics`` (raw, unaligned, per-frame mosaics -- never a
    temporal median, which would preserve stars/foreground as "hot") is
    scored by ``_local_cfa_outliers``; reported only when it clears that
    test in >= ``NO_DARK_PERSISTENCE_SHARE`` of frames. Below
    ``NO_DARK_MIN_FRAMES`` frames this warns and returns an all-clear map.

    An INTERIOR site of a contiguous same-CFA defect reads its own local
    residual near zero (its same-CFA neighbors are hot too) -- only a
    cluster's BOUNDARY sites clear the test, so the returned count is a
    LOWER BOUND on affected photosites, never a census. Persistence alone
    also doesn't separate a star from a defect at the ~1-2px/frame drift
    floor (15 stars -> 150+ false positives, see ``test_stars_at_...``).
    Every site needs human review -- shoot darks and use
    ``hot_pixel_map``/``build_master_dark`` for a verified correction.
    """
    mosaics = [np.asarray(m) for m in mosaics]
    if not mosaics:
        raise AstroCalibrationError("no_dark_hot_pixel_map: no light frames supplied")
    shape = mosaics[0].shape
    for m in mosaics:
        if m.shape != shape:
            raise AstroCalibrationError(
                f"no_dark_hot_pixel_map: mixed mosaic shapes ({shape} vs {m.shape})"
            )
    if len(mosaics) < NO_DARK_MIN_FRAMES:
        logger.warning(
            "no_dark_hot_pixel_map: only %d frame(s), below the %d-frame "
            "confidence floor for no-darks hot-pixel detection -- flagging nothing",
            len(mosaics),
            NO_DARK_MIN_FRAMES,
        )
        return np.zeros(shape, dtype=bool)

    flag_counts = np.zeros(shape, dtype=np.int32)
    for m in mosaics:
        flag_counts += _local_cfa_outliers(m)
    persistent = flag_counts >= NO_DARK_PERSISTENCE_SHARE * len(mosaics)

    count = int(persistent.sum())
    if count:
        logger.warning(
            "no_dark_hot_pixel_map: %d candidate persistent hot photosite(s) "
            "found without darks -- a LOWER BOUND (only a cluster's boundary "
            "sites clear the local-residual test, not its interior); NOT "
            "filtered to defects (e.g. a real star field) and NOT verified "
            "safe to auto-correct -- shoot darks, use hot_pixel_map/build_master_dark",
            count,
        )
    return persistent


def cosmetic_correct(mosaic: np.ndarray, hotmap: np.ndarray) -> np.ndarray:
    """Replace flagged photosites with the median of their same-color neighbors.

    Pure — returns a new array, never mutates ``mosaic``. An edge/corner hot
    pixel falls back to whichever of the four neighbors land in bounds; one
    with none in bounds is left alone.
    """
    if mosaic.shape != hotmap.shape:
        raise AstroCalibrationError(
            f"cosmetic_correct: mosaic {mosaic.shape} != hotmap {hotmap.shape}"
        )
    out = mosaic.copy()
    height, width = mosaic.shape
    for r, c in zip(*np.nonzero(hotmap)):
        r, c = int(r), int(c)
        neighbors = [
            mosaic[r + dr, c + dc]
            for dr, dc in _NEIGHBOR_OFFSETS
            if 0 <= r + dr < height and 0 <= c + dc < width
        ]
        if neighbors:
            out[r, c] = np.median(neighbors)
    return out


def subtract_dark(mosaic: np.ndarray, master_dark: np.ndarray, *, black_level: float) -> np.ndarray:
    """Dark-subtracted mosaic, float32, with the sensor's black PEDESTAL LEFT IN.

    Removes only the master dark's EXCESS over ``black_level`` -- its thermal and read
    component -- never the pedestal itself. The light and the master dark both carry
    the same +``black_level`` offset, so a bare ``mosaic - master_dark`` cancels it
    here, and then ``postprocess`` (which subtracts the camera black level again, and
    has no way to know it is already gone) removes it a SECOND time. Measured on real
    ISO-6400/15s data -- one frame against a 4-frame master dark, pedestal 1008 -- that
    double subtraction drove the decoded frame from mean 0.027970 / 94.21% non-zero to
    mean 0.000022 / 0.72% non-zero: a 1270x signal loss, i.e. supplying darks produced
    a far worse render than supplying none.

    ``black_level`` is keyword-only and REQUIRED on purpose: a default would let a
    caller silently reintroduce exactly that bug. Keeping the pedestal also keeps the
    clip at zero from firing on ordinary noise -- on the same frame it zeroed 8.35% of
    pixels before, because every pixel whose dark exceeded its light was clamped.
    """
    if mosaic.shape != master_dark.shape:
        raise AstroCalibrationError(
            f"subtract_dark: mosaic {mosaic.shape} != master_dark {master_dark.shape}"
        )
    dark_excess = np.asarray(master_dark, dtype=np.float32) - float(black_level)
    delta = np.asarray(mosaic, dtype=np.float32) - dark_excess
    return np.clip(delta, 0.0, None).astype(np.float32)


def decode_linear(path, master_dark=None, hotmap=None, master_flat=None) -> DecodedAstroFrame:
    """Scene-linear rawpy decode with calibration applied to the Bayer mosaic
    BEFORE demosaic — dark subtraction, then the hot-pixel cosmetic pass, then
    flat-field division, in that order (cosmetic correction
    reads the dark-subtracted level; flat-fielding runs last so it corrects
    the already-cleaned signal, not a hot pixel). Every stage is optional and
    independently skippable.

    ``master_flat`` is a ``build_master_flat`` result (normalised to ~1.0);
    the divide-by-small floor's clipped-pixel count is returned on
    ``DecodedAstroFrame.flat_field_clipped_pixels`` (0 when ``master_flat`` is
    ``None``) as this module's calibration receipt.

    Raises:
        AstroCalibrationError: the file could not be opened at all (missing,
            unsupported, corrupt) -- an open failure, not an HE mosaic gap.
        AstroHighEfficiencyError: an HE/HE* NEF exposes no Bayer mosaic.
    """
    import rawpy

    path = Path(path)
    try:
        raw_ctx = rawpy.imread(str(path))
    except rawpy.LibRawError as exc:
        raise AstroCalibrationError(f"{path}: could not open RAW file ({exc})") from exc

    flat_clipped = 0
    with raw_ctx as raw:
        try:
            mosaic = raw.raw_image
            calibrated = mosaic
            if master_dark is not None:
                # Read the pedestal ONLY on the path that needs it, so a decode without
                # darks is byte-identical to before and depends on no extra rawpy
                # attribute. subtract_dark leaves this pedestal in, because postprocess
                # below subtracts it -- taking it out here too removes it twice. Levels
                # are equal across channels on every sensor seen so far (1008 on all
                # four of the sensors measured); the mean stays correct if one ever differs.
                calibrated = subtract_dark(
                    mosaic, master_dark, black_level=float(np.mean(raw.black_level_per_channel))
                )
            if hotmap is not None:
                calibrated = cosmetic_correct(calibrated, hotmap)
            if master_flat is not None:
                flat_result = apply_flat_field(calibrated, master_flat)
                calibrated = flat_result.mosaic
                flat_clipped = flat_result.clipped_pixel_count
            if calibrated is not mosaic:
                raw.raw_image[:] = np.clip(calibrated, 0, 65535).astype(np.uint16)
            rgb16 = raw.postprocess(**_LINEAR_POSTPROCESS)
        except rawpy.LibRawError as exc:
            raise AstroHighEfficiencyError(
                f"{path}: raw stream undecodable ({exc}) — likely a High "
                "Efficiency (HE/HE*) NEF; the orchestrator's HE fallback "
                "retries this file through DNG conversion automatically. If "
                "that already ran and still failed, check "
                "photos.he_dng_converter_exe or why the converter didn't resolve"
            ) from exc

    return DecodedAstroFrame(
        rgb=rgb16.astype(np.float32) / 65535.0,
        path=path,
        decoder="rawpy-linear",
        bit_depth=16,
        flat_field_clipped_pixels=flat_clipped,
    )
