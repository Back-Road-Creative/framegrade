"""Photo decode front-end and develop stage.

Decode turns RAW / HEIC / JPEG-PNG-TIFF into a gradeable sRGB array — 16-bit
for true-raw decodes (the grade quantizes to uint8 only after its float
chain), uint8 elsewhere. Develop grades that array by reusing the video grade
engine in its PHOTO mode — ``analyze_frame`` -> ``compute_correction(mode=GRADE_MODE_PHOTO)`` ->
the ``_apply_*`` chain, in the same order the LUT baker uses. Photo mode
drops the limited-range lift/pull floors (stills ship full-range sRGB, not
the 16-235 YouTube squeeze) and preserves warm golden-hour casts instead of
neutralizing them.

The photo-only spatial stages ride on top of that chain: denoise runs on the
source (before grading amplifies grain), sharpen runs last on the graded
output, which is the order a stills workflow uses. Both are measured-first and
ramped -- never unconditional, and never an on/off cliff.

The LOOK surface rides on top of all of it and also ships from here: ``.cube``
LUT variants, the named recipes ``looks`` declares (a colour chain plus its own
spatial amounts), masked ``layers`` that grade ONE region through a ``masks``
name, printed borders -- and, ahead of every one of them, ``fuse_bracket``,
which merges a detected exposure bracket into the single frame the grade sees.
"""

from __future__ import annotations

import functools
import io
import json
import logging
import math
import os
from dataclasses import asdict, astuple, dataclass, field, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from pycube_lut import Cube

    from .scene import SceneProfile

from ..color_analysis import (
    SKY_ROI_UPPER_FRACTION,
    SPAN_SIMILARITY_THRESHOLD,
    SegmentStats,
    aggregate_segment_stats,
    analyze_frame,
    cluster_files_by_lighting,
)
from ..color_correction import (
    GRADE_ALGO_VERSION,
    GRADE_MODE_PHOTO,
    CorrectionParams,
    _apply_channel_curves,
    _apply_colorbalance,
    _apply_curves,
    _apply_exposure,
    _apply_gamma,
    _apply_hsl,
    _apply_luma_curve,
    _apply_monomix,
    _apply_reference_match,
    _apply_saturation,
    _apply_splittone,
    _apply_vibrance,
    compute_cluster_correction,
    compute_correction,
    lab_mean_std,
)

logger = logging.getLogger(__name__)

RAW_EXTENSIONS = frozenset({".dng", ".cr2", ".nef", ".arw", ".cr3"})
HEIF_EXTENSIONS = frozenset({".heic", ".heif"})
PIL_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff"})
SUPPORTED_EXTENSIONS = RAW_EXTENSIONS | HEIF_EXTENSIONS | PIL_EXTENSIONS


class PhotoDecodeError(RuntimeError):
    """A photo could not be decoded. Always names the offending file."""


class UnusableColorProfileError(PhotoDecodeError, ValueError):
    """A source's embedded ICC profile could not be converted to sRGB.

    Decoding on would leave source-space numbers that every export then tags
    sRGB, so the photo is refused: no master or gallery file is written.
    """

    def __init__(self, path, reason):
        self.path = Path(path)
        self.reason = str(reason)
        super().__init__(
            f"{self.path}: unusable ICC colour profile ({self.reason}); no master written "
            "-- re-save the photo with a valid (or no) embedded profile and develop it again"
        )

    def __reduce__(self):
        # Rebuild from (path, reason): the default replays only the message, which
        # does not fit __init__, so a pooled worker's error could not be unpickled.
        return (type(self), (self.path, self.reason))


@dataclass(frozen=True)
class DecodedPhoto:
    """An upright sRGB frame plus a small record of how it was decoded."""

    rgb: np.ndarray
    path: Path
    decoder: str
    width: int
    height: int
    bit_depth: int
    is_raw: bool
    #: One record per frame of a fused bracket, in input order (see ``fuse_bracket``); empty for
    #: a single decode and for a bracket fused with ``align=False``.
    bracket_shifts: tuple["BracketShift", ...] = ()


@dataclass(frozen=True)
class BracketShift:
    """What the pre-fusion alignment did to one bracket frame.

    ``dx``/``dy`` is the translation (pixels, +x right, +y down) that brings the frame onto the
    reference; ``applied`` is False for the reference itself, a zero shift, and a shift the
    guard refused (``BRACKET_ALIGN_MAX_FRACTION``), which leaves the frame exactly as decoded."""

    path: Path
    dx: int
    dy: int
    applied: bool


_Decoded = tuple[np.ndarray, int, int, int, str, bool]


def _srgb_from_icc(img, path: Path):
    """Colour-manage an ICC-tagged image to real sRGB numbers before the array grab.

    Without this, a Display P3 iPhone HEIC or wide-gamut TIFF is graded and
    exported as if its numbers were already sRGB — desaturated and hue-shifted.
    A source with no profile is assumed sRGB (returned unchanged); one whose
    profile cannot be converted raises ``UnusableColorProfileError`` rather than
    letting never-converted pixels be exported labelled sRGB.
    """
    from PIL import ImageCms

    icc = img.info.get("icc_profile")
    if not icc:
        return img
    try:
        return ImageCms.profileToProfile(
            img,
            ImageCms.ImageCmsProfile(io.BytesIO(icc)),
            ImageCms.createProfile("sRGB"),
            outputMode="RGB",
        )
    except Exception as exc:
        raise UnusableColorProfileError(path, exc) from exc


def _decode_pil(path: Path) -> _Decoded:
    """Decode anything Pillow can open, upright and flattened to sRGB RGB.

    ``exif_transpose`` only clears the orientation tag — it preserves the GPS
    IFD, so transposing is never a privacy strip. Width/height come from the
    upright array, so a portrait frame reports its true dimensions.
    """
    from PIL import Image, ImageOps

    with Image.open(path) as img:
        img.load()
        upright = ImageOps.exif_transpose(img) or img
        if upright.mode.startswith("I;16"):
            # ``convert("RGB")`` clips 16-bit values into uint8, so lift the
            # 16-bit gray plane via numpy instead; it rides the grade chain at
            # full depth exactly like a true-raw decode. ImageCms has no I;16
            # path, and a gray source carries no wide-gamut primaries anyway.
            gray = np.asarray(upright).astype(np.uint16)
            rgb = np.repeat(gray[:, :, np.newaxis], 3, axis=2)
            bit_depth = 16
        else:
            rgb = np.asarray(_srgb_from_icc(upright, path).convert("RGB"))
            bit_depth = 8
    height, width = rgb.shape[:2]

    return rgb, bit_depth, width, height, "pil", False


def _decode_heif(path: Path) -> _Decoded:
    """Register the HEIF opener, then reuse the Pillow path."""
    import pillow_heif

    pillow_heif.register_heif_opener()
    return _decode_pil(path)[:4] + ("pillow-heif", False)


# The fallback preview must carry (nearly) the whole frame: full-size embedded
# previews are ~the sensor area, index thumbnails are orders of magnitude under.
PREVIEW_MIN_AREA_RATIO = 0.5

#: ``DecodedPhoto.decoder`` tag of the embedded-preview fallback — the one
#: provenance a later true-raw develop should supersede.
PREVIEW_FALLBACK_DECODER = "rawpy-preview"

# LibRaw flip → np.rot90 quarter-turns (CCW) to bring the stored frame upright.
_ROT90_BY_FLIP = {3: 2, 5: 1, 6: 3}

_HE_NEF_HINT = (
    "the raw stream could not be unpacked — the signature of a Nikon High Efficiency "
    "(HE/HE*) NEF, whose TicoRAW compression LibRaw cannot decode (it misreports the "
    "intact file as data-corrupted). Shoot Lossless Compressed or pre-convert to DNG "
    "for a true raw develop"
)


# Adobe DNG Converter wall-clock bound per file: a full-res HE NEF converts in
# well under a minute, so minutes means a hung converter, not a slow one.
HE_DNG_CONVERT_TIMEOUT = 300.0

# Adobe DNG Converter flags: -c lossless-compressed, -p1 medium preview,
# -cr11.2 Camera Raw 11.2 compatibility. No wildcards — one invocation per file.
_DNG_CONVERTER_FLAGS = ("-c", "-p1", "-cr11.2")

# Standard Adobe DNG Converter install locations (the vendor's default Windows
# install: through the WSL ``/mnt/c`` mount, or natively on Windows), probed in order when
# ``photos.he_dng_converter_exe`` is unset — an explicit config value always
# wins and skips the probe. First existing file wins; no hit keeps the hook
# disabled (embedded-preview fallback), exactly as before the probe existed.
HE_DNG_CONVERTER_DEFAULT_PATHS = (
    "/mnt/c/Program Files/Adobe/Adobe DNG Converter/Adobe DNG Converter.exe",
    # Native Windows install: a plain Windows path, passed to the converter as is
    # (the ``/mnt/`` WSL-to-Windows argv translation applies only to the entry above).
    "C:\\Program Files\\Adobe\\Adobe DNG Converter\\Adobe DNG Converter.exe",
)


def _windows_path(p: Path | str) -> str:
    """Windows form of a WSL drive mount: ``/mnt/<d>/rest`` -> ``D:\\rest``.

    Pure string translation, no ``wslpath`` subprocess. A path not under a
    single-letter ``/mnt/<drive>/`` mount returns unchanged — it has no
    Windows form a Windows exe could resolve.
    """
    parts = str(p).split("/")
    if len(parts) >= 3 and parts[:2] == ["", "mnt"] and len(parts[2]) == 1 and parts[2].isalpha():
        return parts[2].upper() + ":\\" + "\\".join(parts[3:])
    return str(p)


def _fresh_cached_dng(dng_path: Path, source: Path) -> bool:
    """Reusable: exists, non-empty, at least as new as its source. An unreadable path on
    EITHER side is just "not reusable", never a traceback."""
    try:
        dng_stat = dng_path.stat()
        source_mtime_ns = source.stat().st_mtime_ns
    except OSError:
        return False
    return dng_stat.st_size > 0 and dng_stat.st_mtime_ns >= source_mtime_ns


def resolve_he_dng_converter() -> str:
    """The one rule for locating the DNG converter: ``photos.he_dng_converter_exe`` when
    set (authoritative — it skips the probe), else the first existing entry in
    ``HE_DNG_CONVERTER_DEFAULT_PATHS``; ``""`` when nothing resolves. Shared with the
    ``photos astro`` preflight, so gate and per-frame conversion cannot disagree."""
    from ...config import get_config

    photos = get_config().photos
    return photos.he_dng_converter_exe or next(
        (p for p in HE_DNG_CONVERTER_DEFAULT_PATHS if Path(p).is_file()), ""
    )


def _he_dng_cache_path(path: Path) -> Path:
    """``path``'s converted ``.dng``: in ``photos.he_dng_cache_dir``, else beside it."""
    from ...config import get_config

    cache_dir = get_config().photos.he_dng_cache_dir
    parent = Path(cache_dir).expanduser() if cache_dir else path.parent
    return parent / (path.stem + ".dng")


def he_dng_inputs_needing_conversion(paths) -> tuple[Path, ...]:
    """Which of ``paths`` the DNG converter would be asked about, in input order — a
    candidate list, not a verdict on whether they decode. Cheap by construction: a suffix
    test plus one ``stat`` each, never a decode, since ``_convert_he_to_dng`` returns
    ``None`` for a non-``.nef`` and a fresh cached ``.dng`` is reused without it."""
    return tuple(
        p
        for p in map(Path, paths)
        if p.suffix.lower() == ".nef" and not _fresh_cached_dng(_he_dng_cache_path(p), p)
    )


# RAW postprocess policy, shared by the direct branch and the HE->DNG branch:
# 16-bit output carries the sensor's tonal headroom into the float grade (an
# 8-bit decode quantizes BEFORE grading — banding in skies and shadows);
# ``no_auto_bright`` drops LibRaw's content-dependent 1%-clip gain so the
# measured stats meter the lighting, not a per-frame gain; the explicit sRGB
# transfer replaces rawpy's default (2.222, 4.5), which is BT.709 and
# contradicted the array's sRGB claim.
_RAW_POSTPROCESS = {
    "output_bps": 16,
    "no_auto_bright": True,
    "gamma": (2.4, 12.92),
    "use_camera_wb": True,
}


def _sensor_bit_depth(raw) -> int:
    """Honest sensor depth: bits to reach saturation (``white_level``), not the
    16-bit container ``raw_image`` is stored in (whose itemsize always says 16).
    Falls back to the container depth when LibRaw reports no usable level."""
    white = int(getattr(raw, "white_level", 0) or 0)
    return white.bit_length() if white > 0 else int(raw.raw_image.dtype.itemsize * 8)


def _convert_he_to_dng(path: Path) -> Path | None:
    """Convert an HE/HE* NEF to a ready-to-read ``.dng``, or ``None`` on failure.

    Enabled by ``photos.he_dng_converter_exe`` (authoritative when set) or,
    when that is unset, by probing ``HE_DNG_CONVERTER_DEFAULT_PATHS`` for a
    standard install — first existing file wins. The converter reports errors
    *silently* (exit 0, no output file), so success is judged by the ``.dng``
    existing non-empty, never by the return code. A fresh cached ``.dng`` (at
    least as new as its source) is reused without re-running the converter.
    Every failure warns and returns ``None`` — the caller then falls back to
    the embedded preview, exactly as if the hook were disabled. Performs no
    postprocess of its own: the returned path is handed to whatever decode
    the caller needs, display-referred or scene-linear.
    """
    import subprocess

    if path.suffix.lower() != ".nef":
        return None
    exe = resolve_he_dng_converter()
    if not exe:
        return None

    dng_path = _he_dng_cache_path(path)
    cache_dir = dng_path.parent

    if _fresh_cached_dng(dng_path, path):
        return dng_path

    if exe.startswith("/mnt/"):
        # A /mnt-mounted exe is a Windows binary run via WSL interop: it
        # resolves argv paths with Windows semantics, so hand it Windows
        # forms (WSL forms make it exit "Invalid destination directory").
        # Every filesystem check stays on the WSL-side Path objects.
        invisible = [str(p) for p in (cache_dir, path) if _windows_path(p) == str(p)]
        if invisible:
            logger.warning(
                "%s: the Windows DNG converter cannot see Linux-filesystem path(s) "
                "%s — falling back to the embedded preview",
                path,
                ", ".join(invisible),
            )
            return None
        dest_arg, source_arg = _windows_path(cache_dir), _windows_path(path)
    else:
        dest_arg, source_arg = str(cache_dir), str(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dng_path.unlink(missing_ok=True)
    argv = [exe, *_DNG_CONVERTER_FLAGS, "-d", dest_arg, source_arg]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=HE_DNG_CONVERT_TIMEOUT)
    except (OSError, subprocess.SubprocessError) as run_exc:
        logger.warning(
            "%s: DNG converter failed to run (%s) — falling back to the embedded preview",
            path,
            run_exc,
        )
        return None
    if not dng_path.is_file() or dng_path.stat().st_size == 0:
        logger.warning(
            "%s: DNG converter produced no usable %s (exit %s, output=%r) — "
            "falling back to the embedded preview",
            path,
            dng_path,
            proc.returncode,
            proc.stdout + proc.stderr,
        )
        return None
    return dng_path


def _decode_he_via_dng(path: Path, exc: Exception) -> _Decoded | None:
    """True-raw path for an HE/HE* NEF: convert to DNG via ``_convert_he_to_dng``,
    then decode with develop's display-referred sRGB postprocess.

    The DNG carries the full Bayer mosaic and camera WB, so it rides the
    normal rawpy branch as true raw. A conversion or decode failure warns and
    returns ``None`` — the caller then falls back to the embedded preview,
    exactly as if the hook were disabled.
    """
    import rawpy

    dng_path = _convert_he_to_dng(path)
    if dng_path is None:
        return None

    try:
        with rawpy.imread(str(dng_path)) as raw:
            bit_depth = _sensor_bit_depth(raw)
            rgb = raw.postprocess(**_RAW_POSTPROCESS)
    except Exception as dng_exc:
        logger.warning(
            "%s: converted DNG did not decode (%s) — falling back to the embedded preview",
            path,
            dng_exc,
        )
        return None

    height, width = rgb.shape[:2]
    logger.info(
        "%s: raw stream undecodable (%s) — developed from converted DNG (true raw)",
        path,
        exc,
    )
    return rgb, bit_depth, width, height, "rawpy-dng", True


def _decode_raw_preview(path: Path, exc: Exception) -> _Decoded:
    """Full-resolution embedded-JPEG fallback for an unpackable raw stream.

    High Efficiency NEF bodies embed a full-sensor-size JPEG preview, so the
    develop chain can still run at full resolution — 8-bit, no raw headroom,
    surfaced honestly as ``decoder="rawpy-preview"`` / ``is_raw=False`` plus a
    loud warning. Anything without an adequate preview fails closed with the
    remediation hint instead of LibRaw's misleading text.
    """
    import rawpy
    from PIL import Image

    # A failed unpack poisons the LibRaw handle for thumbnail calls, so the
    # preview must come from a fresh open of the same file.
    try:
        with rawpy.imread(str(path)) as raw:
            thumb = raw.extract_thumb()
            sizes = raw.sizes
    except Exception:
        thumb = None
        sizes = None
    if thumb is None or sizes is None or thumb.format != rawpy.ThumbFormat.JPEG:
        raise PhotoDecodeError(
            f"{path}: {_HE_NEF_HINT} (no embedded JPEG preview to fall back to: {exc})"
        ) from exc

    with Image.open(io.BytesIO(thumb.data)) as img:
        img.load()
        width, height = img.size
        rgb = np.asarray(img.convert("RGB"))

    sensor_area = int(sizes.width) * int(sizes.height)
    if sensor_area and width * height < PREVIEW_MIN_AREA_RATIO * sensor_area:
        raise PhotoDecodeError(
            f"{path}: {_HE_NEF_HINT} (embedded preview is only {width}x{height})"
        ) from exc

    # Embedded previews carry no EXIF orientation; the camera's flip lives in
    # LibRaw's sizes, so apply it here or portrait frames ship sideways.
    turns = _ROT90_BY_FLIP.get(int(getattr(sizes, "flip", 0)), 0)
    if turns:
        rgb = np.rot90(rgb, turns)

    logger.warning(
        "%s: raw stream undecodable (%s) — likely a High Efficiency (HE/HE*) NEF; falling "
        "back to the full-resolution embedded JPEG preview (8-bit, raw headroom lost). "
        "Shoot Lossless Compressed, install Adobe DNG Converter (>=17.1) at its default "
        "path, or set photos.he_dng_converter_exe for a true raw develop.",
        path,
        exc,
    )
    return rgb, 8, width, height, "rawpy-preview", False


def _decode_raw(path: Path) -> _Decoded:
    """Demosaic a camera RAW to 16-bit sRGB (see ``_RAW_POSTPROCESS``).

    ``user_flip`` is deliberately left at its default so LibRaw applies the
    camera's orientation itself; overriding it would ship sideways frames.
    An unpackable stream (High Efficiency NEF) is converted to DNG for a true
    raw develop when the converter hook is configured, else it falls back to
    the embedded full-resolution preview rather than failing the whole card.
    """
    import rawpy

    with rawpy.imread(str(path)) as raw:
        try:
            bit_depth = _sensor_bit_depth(raw)
            rgb = raw.postprocess(**_RAW_POSTPROCESS)
        except rawpy.LibRawError as exc:
            converted = _decode_he_via_dng(path, exc)
            return converted if converted is not None else _decode_raw_preview(path, exc)

    height, width = rgb.shape[:2]
    return rgb, bit_depth, width, height, "rawpy", True


_DECODERS = (
    (RAW_EXTENSIONS, _decode_raw),
    (HEIF_EXTENSIONS, _decode_heif),
    (PIL_EXTENSIONS, _decode_pil),
)


def decode(path) -> DecodedPhoto:
    """Decode a photo to an upright HxWx3 sRGB array plus its metadata.

    An 8-bit array is exactly what ``color_analysis.analyze_frame`` consumes;
    a 16-bit true-raw decode is metered by develop through a uint8 view and
    graded at full depth, quantizing to uint8 only at the end of the grade.

    Raises:
        PhotoDecodeError: unreadable, corrupt, or unsupported input. Failure is
            always closed — never a silent skip, never a half-decoded array.
    """
    path = Path(path)
    if not path.is_file():
        raise PhotoDecodeError(f"{path}: not a readable file")

    suffix = path.suffix.lower()
    for extensions, decoder_fn in _DECODERS:
        if suffix in extensions:
            break
    else:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise PhotoDecodeError(
            f"{path}: unsupported photo format '{suffix}' (supported: {supported})"
        )

    try:
        rgb, bit_depth, width, height, decoder, is_raw = decoder_fn(path)
    except PhotoDecodeError:
        raise
    except Exception as exc:
        raise PhotoDecodeError(f"{path}: could not decode this file: {exc}") from exc

    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise PhotoDecodeError(f"{path}: {decoder} returned a non-RGB array, refusing to continue")
    if rgb.dtype not in (np.uint8, np.uint16):
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    logger.debug(
        "Decoded %s via %s (%dx%d, %d-bit source)", path, decoder, width, height, bit_depth
    )
    return DecodedPhoto(
        rgb=rgb,
        path=path,
        decoder=decoder,
        width=width,
        height=height,
        bit_depth=bit_depth,
        is_raw=is_raw,
    )


#: Fused exports take the bracket's first stem plus this, so they can never clobber a
#: single-frame develop of that same file — ``<lut>_over_<style>``'s rule, one level up.
HDR_STEM_SUFFIX = "_hdr"


#: A frame whose measured shift exceeds this fraction of the SHORTER side is left unaligned and
#: recorded. A handheld bracket drifts by a few tenths of a percent of the frame; 3% (120 px on a
#: 24 MP frame) is already far more than a hand does between shots, so a larger "shift" is far
#: likelier a mismatch (a moved scene, a mis-grouped frame) than camera motion, and warping a
#: frame that far would smear the replicated border across the picture.
BRACKET_ALIGN_MAX_FRACTION = 0.03


def _align_bracket(decoded):
    """Translate every frame onto the middle exposure; returns (frames, shifts).

    The median-threshold bitmap method compares each frame's above/below-median mask, which an
    exposure change leaves in place, so it needs no brightness matching. The shift is measured on
    an 8-bit grey view and applied to the frame at its ORIGINAL precision (a 16-bit decode stays
    16-bit), with the border replicated so the shape never changes. Integer shifts only; rotation
    and moving subjects (deghosting) are not handled."""
    ref = len(decoded) // 2
    grey = [
        cv2.cvtColor(
            (d.rgb >> 8).astype(np.uint8) if d.rgb.dtype == np.uint16 else d.rgb,
            cv2.COLOR_RGB2GRAY,
        )
        for d in decoded
    ]
    bound = BRACKET_ALIGN_MAX_FRACTION * min(grey[ref].shape)
    # MTB halves the image max_bits times and searches +-1 px per level; OpenCV's default of 6
    # underflows on a frame whose coarsest level would be < ~8 px, so cap it to the frame.
    mtb = cv2.createAlignMTB(
        max_bits=int(np.clip(math.log2(max(min(grey[ref].shape), 16) / 8), 1, 6))
    )
    frames, shifts = [], []
    for i, d in enumerate(decoded):
        dx, dy = (0, 0) if i == ref else (int(v) for v in mtb.calculateShift(grey[ref], grey[i]))
        apply = i != ref and (dx, dy) != (0, 0) and max(abs(dx), abs(dy)) <= bound
        rgb = d.rgb
        if apply:
            h, w = rgb.shape[:2]
            rgb = cv2.warpAffine(
                rgb,
                np.float32([[1, 0, dx], [0, 1, dy]]),
                (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_REPLICATE,
            )
        frames.append(rgb)
        shifts.append(BracketShift(d.path, dx, dy, apply))
        if i != ref and (dx, dy) != (0, 0):
            logger.info(
                "Bracket align %s: shift (%d, %d) px %s", d.path.name, dx, dy,
                "applied" if apply else f"skipped (over the {bound:.0f} px bound)",
            )  # fmt: skip
    return frames, tuple(shifts)


def fuse_bracket(paths, align: bool = True) -> DecodedPhoto:
    """Fuse one exposure bracket into the single frame the normal chain develops.

    Mertens exposure fusion, not tone-mapped HDR: a per-pixel weighted blend, so the result
    is display-referred and every stage after it behaves as on one decode. Two properties
    are bought here rather than gated later. **Reproducible**: OpenCV's parallel stripes
    move the float rounding (two runs over a 20 MP bracket differ on 17% of
    pixels), so the merge is pinned to one thread — 7.7 s against 4.1 s on three 20 MP
    frames. **Never clips what it was given**: the pyramid reconstruction overshoots [0, 1]
    on a bright region (1.48 on the test scene), and clamping would flatten the very highlights
    the darkest frame was shot to hold, so an out-of-range merge is mapped IN affinely — in
    range that map is the identity. It keeps the first path, so the profile still reads real
    EXIF, and tags ``<decoder>+mertens<N>``.

    ``align`` (default on) first translates each frame onto the middle exposure
    (``_align_bracket``) — equal size does not mean registered, and a handheld bracket ghosts
    otherwise. A tripod bracket measures zero everywhere and is bit-identical to ``align=False``,
    which is exactly the pre-alignment path. The per-frame result is on ``bracket_shifts``."""
    decoded = [decode(p) for p in paths]
    if len(decoded) < 2:
        raise ValueError(f"fuse_bracket needs a bracket, got {len(decoded)} frame(s)")
    if len({d.rgb.shape for d in decoded}) > 1:
        # A preview fallback beside a true raw: blending those pixelwise would mean resampling
        # one rendering onto another, so it is a refusal naming both, never a resize.
        shapes = ", ".join(f"{d.path.name} {d.width}x{d.height} via {d.decoder}" for d in decoded)
        raise ValueError(
            f"{decoded[0].path}: bracket frames decoded to different sizes ({shapes}) — "
            "fix the decode (the DNG hook) rather than resampling one rendering onto another"
        )
    frames, shifts = _align_bracket(decoded) if align else ([d.rgb for d in decoded], ())
    # Mertens weighs contrast/saturation/well-exposedness on DISPLAY-referred frames and reads
    # OpenCV's BGR luma. A bracket with any 16-bit decode goes in as float32 on OpenCV's 0-255
    # scale (it divides every input by 255, uint8 or float; 65535 / 257 = 255 exactly), so the
    # blend keeps the full precision the decode gave it and is re-quantized at that depth below;
    # an all-8-bit bracket takes the 8-bit path unchanged.
    wide = any(d.rgb.dtype == np.uint16 for d in decoded)
    ldr = [
        f.astype(np.float32) / (257.0 if f.dtype == np.uint16 else 1.0) if wide else f
        for f in frames
    ]
    pool = cv2.getNumThreads()
    try:
        cv2.setNumThreads(1)
        fused = cv2.createMergeMertens().process([np.ascontiguousarray(f[..., ::-1]) for f in ldr])
    finally:
        cv2.setNumThreads(pool)
    low, high = min(0.0, float(fused.min())), max(1.0, float(fused.max()))
    peak = 65535.0 if wide else 255.0
    scaled = np.clip((fused[:, :, ::-1] - low) / (high - low), 0.0, 1.0) * peak
    first = decoded[0]
    logger.debug("Fused %d frames into %s via %s", len(decoded), first.path, first.decoder)
    return replace(
        first,
        rgb=np.rint(scaled).astype(np.uint16 if wide else np.uint8),
        decoder=f"{first.decoder}+mertens{len(decoded)}",
        bit_depth=max(d.bit_depth for d in decoded),
        bracket_shifts=shifts,
    )


# ── Develop Stage ─────────────────────────────────────────────────────


# Denoise: ``params.denoise_strength`` (0-6) authorises a budget, this frame's
# measured noise decides how much of it is spent. Floor/full bracket the ramp in
# luma levels; below the floor a frame is clean enough to pass through untouched.
DENOISE_NOISE_FLOOR = 1.5
DENOISE_NOISE_FULL = 8.0
DENOISE_H_PER_STRENGTH = 1.6
DENOISE_MIN_H = 0.5

# Sharpen: driven by measured detail (log10 variance-of-Laplacian). Zero at both
# ends -- a crisp frame must not be over-sharpened, and a featureless one (sky,
# fog, blank wall) has no edges to recover, so sharpening it only lifts grain.
SHARPEN_MAX_AMOUNT = 0.8
SHARPEN_DETAIL_FLOOR = 0.5
SHARPEN_DETAIL_CRISP = 3.0
SHARPEN_RADIUS = 1.2
# Hard ceiling on the amount AFTER the operator scale, so an amplified soft
# frame can only reach twice the calibrated maximum instead of haloing.
SHARPEN_MAX_SCALED_AMOUNT = SHARPEN_MAX_AMOUNT * 2.0

# Resolution bounds for the spatial stages; at or under each bound the behavior
# is an exact no-op, so small frames keep their calibrated byte-identical path.
# NL-means wall clock is linear in pixels with a huge constant (7x7 patches
# compared across a 21x21 window per pixel -- minutes at 20MP), so over-bound
# frames are denoised on a downscale and re-applied as an upsampled residual.
# Variance-of-Laplacian is resolution- and noise-sensitive (sensor-scale
# high-ISO grain reads as crisp detail), so the sharpen ramp meters a bounded
# view. The unsharp radius was calibrated near the reference width; wider
# frames scale it proportionally, or 1.2px collapses to ~0.2px once the site
# downscales a 5568-wide frame. Floats so `_spatial_tunables` hashes them.
DENOISE_MAX_DIM = 2048.0
SHARPEN_DETAIL_MAX_DIM = 1536.0
SHARPEN_REFERENCE_WIDTH = 1920.0

_NOISE_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)


@dataclass(frozen=True)
class DevelopedPhoto:
    """A graded uint8 sRGB frame plus the correction that produced it.

    The ``*_applied`` flags and their amounts are the review surface: a caller
    can report exactly which spatial stages fired on a given frame.
    """

    rgb: np.ndarray
    path: Path
    params: CorrectionParams
    cluster_index: int
    denoise_applied: bool = False
    denoise_h: float = 0.0
    sharpen_applied: bool = False
    sharpen_amount: float = 0.0
    #: Whether ``lens_correct.correct`` matched an EXIF profile and remapped
    #: the frame; ``lens_correct_reason`` is the review surface for why not,
    #: when it didn't (extra absent, no camera EXIF, no matching profile).
    lens_correct_applied: bool = False
    lens_correct_reason: str = ""
    #: ``DecodedPhoto.decoder`` tag of the frame this develop consumed.
    decoder: str = ""


def _luma(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)


def _peak(rgb: np.ndarray) -> float:
    """Full-scale code for ``rgb``'s dtype — 255 at 8 bits, 65535 at 16. The ONE
    place the look stages learn their depth: each normalizes by this rather than
    a literal 255 and rounds back via ``_requantize``, so a 16-bit frame rides the
    chain at 16 bits and an 8-bit one is byte-identical (the peak IS 255.0)."""
    return 65535.0 if rgb.dtype == np.uint16 else 255.0


def _requantize(out: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Round a float stage result (already in ``rgb``'s code range) to that dtype."""
    return np.clip(np.rint(out), 0.0, _peak(rgb)).astype(rgb.dtype)


def _noise_sigma(rgb: np.ndarray) -> float:
    """Estimate this frame's noise, in luma levels, without counting texture as noise.

    Immerkaer's high-pass kernel cancels flat areas *and* linear ramps, so only
    high-frequency content survives. Taking the **median** of the response
    rather than its mean is what separates grain from detail: noise is dense so
    it sets the median, while edges are sparse and only fatten the tail.

    Variance-of-Laplacian is deliberately not used here -- it scores a clean,
    finely-detailed frame *higher* than a genuinely noisy one, so driving
    denoise from it would smear exactly the frames worth keeping sharp.
    """
    response = cv2.filter2D(_luma(rgb), cv2.CV_64F, _NOISE_KERNEL, borderType=cv2.BORDER_REPLICATE)
    return float(np.median(np.abs(response[1:-1, 1:-1])) / (0.6745 * 6.0))


def _detail_score(rgb: np.ndarray) -> float:
    """Variance of the Laplacian: the classic focus measure. Soft frames score low."""
    return float(cv2.Laplacian(_luma(rgb), cv2.CV_64F).var())


def _bounded_view(rgb: np.ndarray, max_dim: float) -> np.ndarray:
    """INTER_AREA downscale until the long edge fits ``max_dim``; identity at or
    under the bound. Averaging (not striding) is deliberate: it suppresses
    per-pixel grain the way the site's final downscale does, so the estimators
    read what a viewer sees rather than sensor-scale noise."""
    long_edge = max(rgb.shape[0], rgb.shape[1])
    if long_edge <= max_dim:
        return rgb
    scale = max_dim / long_edge
    size = (max(1, round(rgb.shape[1] * scale)), max(1, round(rgb.shape[0] * scale)))
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)


def _denoise(rgb: np.ndarray, params: CorrectionParams) -> tuple[np.ndarray, float]:
    """Spend the already-measured denoise budget in proportion to measured noise.

    ``params.denoise_strength`` is consumed as-is -- the lighting analysis that
    produced it is not repeated here. It sets the ceiling; this frame's noise
    sets how much of that ceiling is used, so an authorised-but-clean frame is
    returned untouched instead of being softened on a flag alone.

    NL-means runtime is unbounded in frame area, so over-bound frames measure
    and denoise a ``_bounded_view`` and take the correction back as an
    upsampled residual -- the wall clock is capped by ``DENOISE_MAX_DIM``, not
    by the sensor. Measuring noise on that same view keeps the ramp consistent:
    ``h`` matches the noise actually present in the image NL-means sees.
    """
    strength = max(0.0, float(params.denoise_strength))
    if strength <= 0.0:
        return rgb, 0.0

    view = _analysis_rgb(rgb)
    work = _bounded_view(view, DENOISE_MAX_DIM)
    span = DENOISE_NOISE_FULL - DENOISE_NOISE_FLOOR
    ramp = min(1.0, max(0.0, (_noise_sigma(work) - DENOISE_NOISE_FLOOR) / span))
    h = round(strength * DENOISE_H_PER_STRENGTH * ramp, 3)
    if h < DENOISE_MIN_H:
        return rgb, 0.0

    denoised = cv2.fastNlMeansDenoisingColored(np.ascontiguousarray(work), None, h, h, 7, 21)
    if work is rgb:
        return denoised, h
    # OpenCV's NL-means is 8-bit-only and bounded to ``work``, so the
    # correction it computed there is applied to the original frame as a
    # residual (upsampled when ``work`` was downscaled): pixels it left alone
    # keep their gradations instead of being requantized.
    residual = denoised.astype(np.int32) - work.astype(np.int32)
    if work is not view:
        residual = np.rint(
            cv2.resize(
                residual.astype(np.float32),
                (view.shape[1], view.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        ).astype(np.int32)
    if view is rgb:
        return np.clip(rgb.astype(np.int32) + residual, 0, 255).astype(np.uint8), h
    lifted = np.clip(rgb.astype(np.int32) + residual * 257, 0, 65535)
    return lifted.astype(np.uint16), h


def _sharpen_amount(detail: float) -> float:
    """Ramp the unsharp amount down as measured detail rises; zero at both ends."""
    log_detail = math.log10(max(detail, 1e-6))
    if log_detail <= SHARPEN_DETAIL_FLOOR or log_detail >= SHARPEN_DETAIL_CRISP:
        return 0.0

    span = SHARPEN_DETAIL_CRISP - SHARPEN_DETAIL_FLOOR
    t = (log_detail - SHARPEN_DETAIL_FLOOR) / span
    return round(SHARPEN_MAX_AMOUNT * (1.0 - t), 3)


def _sharpen(rgb: np.ndarray, scale: float = 1.0) -> tuple[np.ndarray, float]:
    """Unsharp-mask a soft frame; pass a crisp or featureless one straight through.

    Detail is metered on a resolution-normalized view so the ramp reads the
    same at any sensor size, and the radius scales with width above the
    reference so a full-res frame keeps a perceptually equivalent sharpen.

    ``scale`` is the operator lever: a MULTIPLIER on the measured amount, never
    a replacement for it, so the detail ramp still decides which frames get
    sharpened at all. 1.0 (the default) short-circuits to the measured amount
    and is bit-identical to the automatic-only chain; 0.0 is off; above 1.0 the
    result is clamped to ``SHARPEN_MAX_SCALED_AMOUNT`` so it cannot halo.
    """
    # Metered on the 8-bit companion (a no-op on an 8-bit frame): the ramp is
    # calibrated in 8-bit levels and variance-of-Laplacian goes as the SQUARE of
    # the code range, so a raw 16-bit read scores ~4.8 decades high.
    view = _bounded_view(_analysis_rgb(rgb), SHARPEN_DETAIL_MAX_DIM)
    amount = _sharpen_amount(_detail_score(view))
    if scale != 1.0:
        amount = round(min(amount * scale, SHARPEN_MAX_SCALED_AMOUNT), 3)
    if amount <= 0.0:
        return rgb, 0.0

    radius = SHARPEN_RADIUS * max(1.0, rgb.shape[1] / SHARPEN_REFERENCE_WIDTH)
    source = rgb.astype(np.float32)
    blurred = cv2.GaussianBlur(source, (0, 0), radius)
    out = source * (1.0 + amount) - blurred * amount
    return np.clip(out, 0.0, _peak(rgb)).astype(rgb.dtype), amount


# Vignette falloff shape. The corner darkening ramps up as the normalized radius
# grows; a higher exponent holds more of the frame bright and tightens the
# darkening into the extreme corners. Float so ``_tunable_constants`` hashes it.
VIGNETTE_FEATHER = 2.0


def _apply_gain_mask(rgb: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """Multiply an (H,W,3) uint8 frame by a per-pixel gain, clip, return uint8.

    The reusable spatial seam every position-aware stage rides on — vignette
    here; clarity, grain, halation and the sky mask later. A stage measures the
    (H,W,3) frame and hands back an (H,W) or (H,W,1) float ``gain``; this applies
    it. An all-ones gain is a byte-identical no-op *by construction* (the float
    multiply is exact for gain 1.0 and every uint8 code), so a stage that decides
    "no change" costs nothing. ``gain`` broadcasts over the channel axis.
    """
    gain = np.asarray(gain, dtype=np.float32)
    if gain.ndim == 2:
        gain = gain[:, :, np.newaxis]
    return _requantize(rgb.astype(np.float32) * gain, rgb)


def _radial_vignette_mask(height: int, width: int, amount: float) -> np.ndarray:
    """A smooth radial darkening gain: 1.0 at the center, down by ``amount`` at
    the corners. Coordinates are NORMALIZED (the corners sit at radius 1.0
    regardless of frame size), so the falloff looks identical at any resolution —
    the same resolution-independence ``SHARPEN_REFERENCE_WIDTH`` gives sharpen.
    ``amount<=0`` yields an all-ones (no-op) mask."""
    ny = np.linspace(-1.0, 1.0, height, dtype=np.float32) if height > 1 else np.zeros(1, np.float32)
    nx = np.linspace(-1.0, 1.0, width, dtype=np.float32) if width > 1 else np.zeros(1, np.float32)
    gy, gx = np.meshgrid(ny, nx, indexing="ij")
    radius = np.sqrt((gx * gx + gy * gy) * 0.5)  # 0 at center, 1.0 at the corners
    falloff = np.clip(radius, 0.0, 1.0) ** VIGNETTE_FEATHER
    return (1.0 - amount * falloff).astype(np.float32)


def _vignette(rgb: np.ndarray, amount: float) -> np.ndarray:
    """Apply a radial vignette to a graded frame; ``amount<=0`` is a no-op that
    returns the input unchanged (byte-identical), so the default develop path is
    exactly what it was before the spatial seam existed."""
    if amount <= 0.0:
        return rgb
    mask = _radial_vignette_mask(rgb.shape[0], rgb.shape[1], amount)
    return _apply_gain_mask(rgb, mask)


# Sky mask -- the sky RENDERED, where ``analyze_frame`` only ever metered it to
# EXCLUDE it. Same reasoning, so mask and meter agree on what the sky is: bright
# pixels in the upper band (``SKY_ROI_UPPER_FRACTION``, imported so the two can never
# drift), landscape frames only. Bright is measured against THIS frame's own
# distribution (``SKY_LUMA_PERCENTILES``, a ramp ``SKY_LUMA_SPAN_FRACTION`` of the way
# from median to highlight end): an absolute threshold is how a stage ships inert here,
# since a real develop peaks near 0.66 luma and never clips. ``SKY_MIN_CONTRAST`` is the
# median-to-highlight separation a frame needs before it is judged to hold sky at all --
# a flat frame's mask goes to zero by construction rather than becoming a mask of its
# own noise. Softness is not cosmetic: a hard-edged sky is the classic tell of a bad
# landscape edit, so the band ramps (``SKY_BAND_FEATHER``, a fraction of the band) and
# the whole mask is blurred at ``SKY_MASK_BLUR_FRACTION`` of the long edge, metered on a
# ``SKY_MASK_MAX_DIM``-bounded view and bilinearly upsampled -- resolution-independent
# and smooth by construction. ``SKY_DARKEN_MAX`` is the darkening amount 1.0 buys
# (deliberately under 1.0: a fully black sky reads synthetic). Floats and a tuple, so
# ``_tunable_constants`` hashes a retune.
SKY_LUMA_PERCENTILES = (50.0, 99.0)
SKY_LUMA_SPAN_FRACTION = 0.5
SKY_MIN_CONTRAST = 0.10
SKY_BAND_FEATHER = 0.5
SKY_MASK_BLUR_FRACTION = 0.02
SKY_MASK_MAX_DIM = 512.0
SKY_DARKEN_MAX = 0.40
# ``SKY_SATURATE_MAX`` is the extra chroma amount 1.0 buys inside the mask, so the
# fully-masked sky is graded at 1.0 + this. Deliberately far under the grade chain's
# own 2.0 chroma cap: a doubled sky reads as a poster, not a polariser.
SKY_SATURATE_MAX = 0.60


def _sky_mask(rgb: np.ndarray) -> np.ndarray:
    """Soft (H,W) float32 mask of this frame's sky, 1 = sky, 0 = everything else.

    A PORTRAIT frame returns all zeros: the upper-band heuristic assumes landscape
    horizon framing and a portrait still's bright top is usually the subject (a face
    against sky), so ``analyze_frame`` skips sky exclusion there and this must skip
    sky rendering for the same reason -- silently darkening a subject's head is the
    failure mode. Everything else is a product of three soft terms: the feathered
    upper band, brightness relative to the frame's OWN median-to-highlight span, and
    a confidence weight that vanishes on a frame too flat to hold a sky.
    """
    height, width = rgb.shape[:2]
    if width < height:
        return np.zeros((height, width), np.float32)
    view = _bounded_view(rgb, SKY_MASK_MAX_DIM)
    luma = (_luma(view) / _peak(rgb)).astype(np.float32)
    low, high = np.percentile(luma, SKY_LUMA_PERCENTILES)
    bright = np.clip((luma - low) / max(SKY_LUMA_SPAN_FRACTION * (high - low), 1e-3), 0.0, 1.0)
    weight = min((high - low) / SKY_MIN_CONTRAST, 1.0)  # a flat frame holds no sky
    confidence = weight * weight * (3.0 - 2.0 * weight)  # smoothstep: no cliff at the knee
    rows = np.arange(view.shape[0], dtype=np.float32) / max(view.shape[0] - 1, 1)
    band = np.clip(
        (SKY_ROI_UPPER_FRACTION - rows) / (SKY_ROI_UPPER_FRACTION * SKY_BAND_FEATHER), 0.0, 1.0
    )
    band = band * band * (3.0 - 2.0 * band)  # smoothstep: no kink where the band ends
    mask = cv2.GaussianBlur(
        bright * confidence * band[:, np.newaxis],
        (0, 0),
        SKY_MASK_BLUR_FRACTION * max(view.shape[0], view.shape[1]),
    )
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    return mask.astype(np.float32)


def _sky_darken(rgb: np.ndarray, amount: float) -> np.ndarray:
    """Darken the sky of a graded frame; ``amount<=0`` is a byte-identical no-op
    (early return), so the default develop path is exactly what it was before the
    sky mask existed. The mask becomes a per-pixel GAIN and rides the shared
    ``_apply_gain_mask`` seam -- the frame is never re-flattened on a parallel path.
    Runs with the other gain-mask stages and BEFORE ``_vignette``: darkening the sky
    is a SUBJECT-region edit, while the vignette is a lens effect that belongs over
    everything the picture contains. ``_sky_saturate`` is the second lever on this same
    mask; the graduated ND is NOT -- a filter rides geometry, not a region
    (``_graduated_nd``)."""
    if amount <= 0.0:
        return rgb
    return _apply_gain_mask(rgb, 1.0 - (amount * SKY_DARKEN_MAX) * _sky_mask(rgb))


def _sky_saturate(rgb: np.ndarray, amount: float) -> np.ndarray:
    """Boost the CHROMA of the sky alone -- the polariser move ``_sky_darken`` cannot
    make; ``amount<=0`` is a byte-identical no-op (early return), so the default
    develop path is exactly what it was before this stage existed.

    Shares ``_sky_mask`` with ``_sky_darken`` and NOT the ``_apply_gain_mask`` seam:
    that seam applies a per-pixel SCALAR gain, and scaling all three channels of a
    pixel by one factor moves chroma and luma together, leaving ``(max-min)/max``
    exactly where it was -- a gain provably cannot saturate. Saturation is a lerp
    AWAY from the pixel's own luma, so this takes the masked-lerp shape ``_dehaze``
    and ``_halation`` use instead. Written as an ADDED chroma term rather than
    ``luma + mult * (rgb - luma)``: where the mask is 0 the term is exactly 0.0, so
    everything outside the sky keeps its bytes BY CONSTRUCTION -- the same guarantee
    the gain seam gets from an all-ones gain, on a different shape.

    Runs immediately after ``_sky_darken`` and, with it, BEFORE ``_vignette``: it is
    the same subject-region edit, and the lerp is around each pixel's own luma, so it
    grades the chroma of the sky the darkener has already placed instead of a tone
    that is about to move."""
    if amount <= 0.0:
        return rgb
    src = rgb.astype(np.float32)
    chroma = src - _luma(rgb)[:, :, np.newaxis]
    return _requantize(src + (amount * SKY_SATURATE_MAX) * _sky_mask(rgb)[:, :, None] * chroma, rgb)


# Graduated ND -- the smoked-glass filter in a holder over the lens, dark at one edge
# and clear at the other, held over a bright sky so the foreground keeps its exposure.
# It METERS NOTHING, which is exactly what separates it from ``_sky_darken``: the ramp is
# ``masks._linear``, pure geometry, so one amount is one filter on every frame instead of
# a heuristic that can go inert. ``GRAD_ND_ANGLE`` is LEVEL in that mask's convention (90
# ramps toward the BOTTOM, so a sky-holding grad is its mirror) and the operator's own
# angle is a TILT off it, bounded at +/-90 at the flag and the look door -- which is what
# makes a ground-darkening grad, the whole failure mode, unexpressible rather than merely
# discouraged. ``GRAD_ND_MAX`` is what amount 1.0 buys: one stop at the smoked edge,
# deliberately under 1.0 for the same reason ``SKY_DARKEN_MAX`` is. The span stays the
# shared ``masks.LINEAR_SPAN``: a hard-edge grad's only new capability is a visible seam
# across the picture. Floats, so ``_tunable_constants`` hashes a retune.
GRAD_ND_ANGLE = -90.0
GRAD_ND_MAX = 0.5


def _graduated_nd(rgb: np.ndarray, amount: float, angle: float = 0.0) -> np.ndarray:
    """Lay a graduated ND over a graded frame; ``amount<=0`` is a byte-identical no-op
    (early return) at EVERY angle, so an angle alone can never move a pixel. ``angle``
    is the filter's rotation in its holder, degrees off level: 0 holds the sky, +/-90
    stands it upright to hold one SIDE, and the door refuses anything past that, so the
    ramp cannot be turned on the ground. The ramp is ``masks._linear``'s own -- imported
    INSIDE the function because ``masks`` imports back from here -- turned into a
    per-pixel GAIN on the shared ``_apply_gain_mask`` seam. Runs AFTER ``_sky_darken``
    and immediately BEFORE ``_vignette``: a sky burn is a darkroom edit of a region,
    while this is glass in FRONT of the lens, so it belongs with the optics, ahead of the
    lens's own corner falloff -- and, metering nothing itself, it cannot perturb the sky
    mask measured before it."""
    if amount <= 0.0:
        return rgb
    from . import masks

    ramp = masks._linear(rgb, angle=GRAD_ND_ANGLE + angle, span=masks.LINEAR_SPAN)
    return _apply_gain_mask(rgb, 1.0 - (amount * GRAD_ND_MAX) * ramp)


# Dehaze -- genuine atmospheric-veil SUBTRACTION via the dark-channel prior, NOT the
# bounded S-curve contrast increment ``color_correction`` calls dehaze (that one adds
# punch; it cannot lift a veil). Haze composites as ``I = J*t + A*(1-t)``; this inverts
# it. ``DEHAZE_OMEGA_MAX`` is how much of the veil amount 1.0 strips (under 1.0
# deliberately: a total strip reads synthetic); ``DEHAZE_HAZE_FLOOR_PERCENTILE`` is where
# this frame counts as haze-FREE; ``DEHAZE_MIN_TRANSMISSION`` caps the amplification where
# the prior is least reliable; ``DEHAZE_SHADOW_RETENTION`` is the fraction of its darkest
# channel a pixel always keeps. Dark channel and transmission are metered on a
# ``DEHAZE_MAX_DIM``-bounded view (a full-res min-filter at 20 MP is minutes of wall clock;
# transmission is low-frequency anyway) with the patch a FRACTION of that view's long edge,
# so the estimate is resolution-independent. Floats, so ``_tunable_constants`` hashes it.
DEHAZE_MAX_DIM = 512.0
DEHAZE_PATCH_FRACTION = 0.02
DEHAZE_AIRLIGHT_PERCENTILE = 99.0
DEHAZE_HAZE_FLOOR_PERCENTILE = 1.0
DEHAZE_OMEGA_MAX = 0.85
DEHAZE_MIN_TRANSMISSION = 0.35
DEHAZE_SHADOW_RETENTION = 0.05


def _dehaze_airlight(view: np.ndarray, patch: int) -> np.ndarray:
    """The (3,) veil colour: the MEAN of the frame's haziest pixels, in 0..1.

    Haze-opaque pixels are the ones whose dark channel is highest; averaging that set —
    rather than the single brightest pixel the published prior takes — stops one specular
    speck setting the veil colour. FRAME-RELATIVE by construction: on a frame peaking at
    0.66 luma the airlight lands near 0.66, where an absolute constant (or the prior's
    hard 1.0) would either do nothing or over-correct wildly."""
    dark = cv2.erode(view.min(axis=2), np.ones((patch, patch), np.uint8))
    haziest = view[dark >= float(np.percentile(dark, DEHAZE_AIRLIGHT_PERCENTILE))]
    return np.maximum(haziest.reshape(-1, 3).mean(axis=0), 1e-3).astype(np.float32)


def _dehaze_transmission(src: np.ndarray, amount: float) -> tuple[np.ndarray, np.ndarray]:
    """``(transmission, airlight)`` for a float32 0..1 frame — how much of each pixel
    reached the sensor unscattered, and the veil it was mixed with. ``t = 1 - omega *
    haze``: a patch whose darkest channel is bright even after normalizing by the veil can
    only BE veil, so its transmission is low. Metered on the bounded view, blurred (a
    blocky erosion is not a real transmission), upsampled.

    ``haze`` is that dark channel measured against THIS FRAME's haze-free floor
    (``DEHAZE_HAZE_FLOOR_PERCENTILE``), never a literal zero. The published prior assumes
    a clear scene's darkest patch is black; this chain never lands there (a real develop's
    1st percentile sits near 0.14 luma), so a raw dark channel bills every dark object's
    own colour as veil and the subtraction eats the shadows — on the reference
    misty-ridgeline NEF that crushed 29% of the frame under 16/255 at amount 1.0, where
    the floor-relative build crushes none of it.

    The last line restores by CONSTRUCTION what the upsample breaks: ``t >= 1 - I_c/A_c``
    is the exact condition for a recovered pixel to stay non-negative and the unblurred
    estimate meets it automatically, but blur and upsample let a hazy neighbour's low
    transmission bleed onto a dark pixel and drive it to black."""
    view = _bounded_view(src, DEHAZE_MAX_DIM)
    patch = max(3, round(DEHAZE_PATCH_FRACTION * max(view.shape[0], view.shape[1])))
    air = _dehaze_airlight(view, patch)
    dark = cv2.erode((view / air).min(axis=2), np.ones((patch, patch), np.uint8))
    clear = float(np.percentile(dark, DEHAZE_HAZE_FLOOR_PERCENTILE))
    haze = np.clip((dark - clear) / max(1.0 - clear, 1e-3), 0.0, 1.0)
    t = 1.0 - (amount * DEHAZE_OMEGA_MAX) * haze
    t = np.maximum(cv2.GaussianBlur(t, (0, 0), max(1.0, patch * 0.5)), DEHAZE_MIN_TRANSMISSION)
    if t.shape != src.shape[:2]:
        t = cv2.resize(t, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_LINEAR)
    ratio = np.clip((src / air).min(axis=2), 0.0, 1.0)
    return np.maximum(t, (1.0 - ratio) / (1.0 - DEHAZE_SHADOW_RETENTION * ratio)), air


def _dehaze(rgb: np.ndarray, amount: float) -> np.ndarray:
    """Subtract the estimated atmospheric veil from a graded frame; ``amount<=0`` is a
    byte-identical no-op (early return), so the default develop path is exactly what it
    was before dehaze existed. Runs FIRST of the spatial stages, before ``_clarity``:
    lifting the veil is a CORRECTION, so local contrast, sharpen and glow work on the
    corrected frame, not on the haze. ``amount`` scales ``omega``."""
    if amount <= 0.0:
        return rgb
    src = rgb.astype(np.float32) / _peak(rgb)
    t, air = _dehaze_transmission(src, amount)
    out = (src - air) / t[:, :, np.newaxis] + air
    return _requantize(out * _peak(rgb), rgb)


# Clarity (broad local contrast). Distinct from ``_sharpen``'s fine ~1.2px detail
# pass: this high-passes LUMA at a LARGE, width-scaled radius and adds the response
# back as a per-pixel *ratio* gain (color untouched), weighted to the midtones so
# near-black/near-white pixels barely move -- that protection is what keeps broad
# contrast from carving halos or blowing highlights. ``CLARITY_RADIUS`` is the px
# radius at ``SHARPEN_REFERENCE_WIDTH`` (scaled up on wider frames the same way
# sharpen's is); ``CLARITY_MIDTONE_KNEE`` shapes the tone window (higher = tighter
# to mid-gray); ``CLARITY_MAX_GAIN`` bounds the per-pixel gain so a dark pixel's
# small denominator can't blow up. Floats so ``_tunable_constants`` hashes a retune.
CLARITY_RADIUS = 48.0
CLARITY_MIDTONE_KNEE = 2.0
CLARITY_MAX_GAIN = 2.5


def _clarity_gain(rgb: np.ndarray, amount: float) -> np.ndarray:
    """Per-pixel luma gain that moves midtone local contrast; ``amount`` in [-1, 1].

    Luma is high-passed at a large, width-scaled radius (broad structure, not
    edges); the response feeds back as a *ratio* (one gain per pixel applied to
    every channel, so hue/saturation are preserved) weighted by a midtone window
    that tapers to ~0 at both ends of the tone range. Near-clipped pixels keep a
    gain of ~1.0 -- highlights/shadows are protected by construction -- while
    midtones take the full move. The formula is SYMMETRIC in ``amount``: a
    negative one subtracts the same high-pass response instead of adding it, so
    the frame's own broad structure is flattened rather than lifted, and the
    ``CLARITY_MAX_GAIN`` clip bounds it from both sides. Returns an (H,W) float32
    gain."""
    luma = _luma(rgb)  # (H,W) float64
    peak = _peak(rgb)
    radius = CLARITY_RADIUS * max(1.0, rgb.shape[1] / SHARPEN_REFERENCE_WIDTH)
    detail = luma - cv2.GaussianBlur(luma, (0, 0), radius)
    tone = luma / peak
    midtone = np.clip(1.0 - np.abs(2.0 * tone - 1.0) ** CLARITY_MIDTONE_KNEE, 0.0, 1.0)
    # Floor = ONE 8-bit level of this frame's range: same bound at either depth.
    gain = 1.0 + amount * midtone * detail / np.maximum(luma, peak / 255.0)
    return np.clip(gain, 1.0 / CLARITY_MAX_GAIN, CLARITY_MAX_GAIN).astype(np.float32)


def _clarity(rgb: np.ndarray, amount: float) -> np.ndarray:
    """Move midtone-weighted local contrast on a graded frame; ``amount == 0`` is a
    byte-identical no-op (early return), so the default develop path is exactly
    what it was before clarity existed. Runs BEFORE ``_sharpen`` -- broad contrast
    first, then the fine detail pass -- on the reusable ``_apply_gain_mask`` seam.
    The sign is the direction: POSITIVE lifts local contrast (the "pop" pass),
    NEGATIVE flattens it -- a texture-softening move (skin, portraits) the chain
    could not make at all while the gate read ``amount <= 0``. Only exactly zero
    short-circuits, so softening reaches the (already symmetric) gain formula."""
    if amount == 0.0:
        return rgb
    return _apply_gain_mask(rgb, _clarity_gain(rgb, amount))


# Orton glow -- the darkroom sandwich print: one sharp exposure and one soft,
# over-exposed one printed together, so the whole frame carries a diffused bloom
# while the detail underneath survives. One of two ways this chain can SOFTEN a
# frame, and the two are different looks: this is a screen-blended BLOOM laid over
# the whole frame, where a negative ``_clarity`` flattens the frame's own broad
# structure without adding any light. Deliberately NOT ``_halation``, which it otherwise
# mirrors: halation is a highlight-SEEDED, red-biased bleed of film physics (a blown
# window rimming warm), while this is a whole-frame, colour-CARRYING diffusion in
# which brighter tones simply contribute more of the soft copy. SCREEN-blended for
# the reason ``_halation`` documents -- screen cannot overshoot, so an already-bright
# highlight stays exactly where it is and only its surroundings lift.
# ``GLOW_SEED_PERCENTILES``/``GLOW_SEED_FRACTION`` place the tone the diffusion
# starts at FRAME-RELATIVELY: the ``_halation_seed`` lesson, since this chain's
# output tops out near 0.66 luma and never clips, so an absolute threshold ships
# GREEN AND INERT. The ramp then runs from that seed to the frame's OWN highlight
# end rather than a fixed distance above it (``GLOW_SPAN_FLOOR`` stops a flat frame
# dividing by ~0) -- where halation wants a bright frame to bleed harder than a dull
# one, a whole-frame diffusion has to engage the whole frame either way.
# ``GLOW_RADIUS`` is the px blur radius at ``SHARPEN_REFERENCE_WIDTH``, width-scaled
# exactly as halation's, clarity's and sharpen's are, so one flag value is one look
# at every export size; ``GLOW_STRENGTH`` is the screen opacity at amount
# 1.0. Floats and a tuple, so ``_tunable_constants`` hashes a retune.
GLOW_SEED_PERCENTILES = (25.0, 99.0)
GLOW_SEED_FRACTION = 0.25
GLOW_SPAN_FLOOR = 0.05
GLOW_RADIUS = 60.0
GLOW_STRENGTH = 0.75


def _glow_bloom(rgb: np.ndarray, amount: float) -> np.ndarray:
    """The (H,W,3) float32 soft copy this frame diffuses, in 0..1.

    The frame is weighted by its own tone (a soft ramp from the measured seed to
    its highlight end, so the brighter a pixel the more of it blooms), then blurred
    at the width-scaled radius. Blurring the WEIGHTED FRAME rather than a bare mask
    is what makes the bloom carry local colour -- a diffusion of the photograph,
    not a white haze laid over it. The seed is metered on the bounded view the
    sharpen ramp uses: a 20 MP percentile costs a full sort."""
    view = _luma(_bounded_view(rgb, SHARPEN_DETAIL_MAX_DIM)) / _peak(rgb)
    low, high = np.percentile(view, GLOW_SEED_PERCENTILES)
    seed = low + GLOW_SEED_FRACTION * (high - low)
    tone = (_luma(rgb) / _peak(rgb)).astype(np.float32)
    weight = np.clip((tone - seed) / max(high - seed, GLOW_SPAN_FLOOR), 0.0, 1.0)
    radius = GLOW_RADIUS * max(1.0, rgb.shape[1] / SHARPEN_REFERENCE_WIDTH)
    src = rgb.astype(np.float32) / _peak(rgb)
    bloom = cv2.GaussianBlur(src * weight[:, :, np.newaxis], (0, 0), radius)
    return np.clip((amount * GLOW_STRENGTH) * bloom, 0.0, 1.0)


def _glow(rgb: np.ndarray, amount: float = 0.0) -> np.ndarray:
    """Screen a wide, soft copy of a frame back over itself -- the Orton bloom;
    ``amount<=0`` is a byte-identical no-op (early return), so the default develop
    path is exactly what it was before glow existed. Runs after ``_dehaze`` and
    BEFORE ``_clarity``: glow and then re-resolve the detail reads as film, the
    reverse reads as haze. Dehaze stays first of the chain even so -- it SUBTRACTS
    the atmospheric veil, so running it after a deliberate glow would set the two
    against each other. DETERMINISTIC by construction: no rng, only the frame's own
    tone and geometry."""
    if amount <= 0.0:
        return rgb
    src = rgb.astype(np.float32) / _peak(rgb)
    out = 1.0 - (1.0 - src) * (1.0 - _glow_bloom(rgb, amount))
    return _requantize(out * _peak(rgb), rgb)


# Film grain -- the ADD counterpart to denoise's REMOVE. Denoise strips sensor
# grain from the SOURCE so grading can't amplify it; grain lays a controlled,
# filmic texture back onto the FINISHED frame so film-emulation LUTs don't render
# sterile-clean. A single MONOCHROME (luminance) noise field is added equally to
# R/G/B -- colour grain reads as sensor noise, luminance grain reads as film. The
# field is luma-WEIGHTED: strongest across the midtones, rolling to ~0 below
# ``GRAIN_SHADOW_KNEE`` and above ``GRAIN_HIGHLIGHT_KNEE`` (tone in 0..1), where real
# grain is least visible. ``GRAIN_SIGMA_SCALE`` is the noise sigma (8-bit levels)
# at amount 1.0 and full midtone weight; ``amount`` scales it. ``GRAIN_SEED`` is a
# FIXED module constant -- the field is drawn from a seeded ``default_rng``, never
# global ``np.random`` or entropy/time, so two develops of one frame at one amount
# are byte-identical (the develop cache/algo-version invariants require it). Floats
# and an int so ``_tunable_constants`` hashes a retune or a reseed.
GRAIN_SIGMA_SCALE = 12.0
GRAIN_SHADOW_KNEE = 0.15
GRAIN_HIGHLIGHT_KNEE = 0.85
GRAIN_SEED = 0x6D6F76  # fixed; determinism is a hard invariant
# Particle size is resolution-INDEPENDENT -- the promise ``_vignette`` documents and
# ``_halation`` keeps by scaling its radius with the frame width. A grain particle is
# a physical speck of silver, so a bigger scan of one negative resolves it across MORE
# pixels; drawn one independent sample per output pixel it covers exactly one at every
# size, which is the signature of the sensor noise this stage exists to replace. Worse,
# an uncorrelated field averages away as the frame grows: measured on a real NEF, one
# photo at 5600 px carried a SIXTH of the grain it carried at 900 px from the same
# ``--grain`` value (sigma 1.83 vs 11.98 at a common viewing size), so the flag named a
# different look for every export size. The field is therefore drawn on a fixed
# ``GRAIN_REFERENCE_DIM`` grid and resampled up to the frame. A frame already at or
# under the reference has nothing to scale and keeps the cheap per-pixel draw,
# byte-for-byte as before -- which is also every contact-sheet panel.
GRAIN_REFERENCE_DIM = 1200
# ``grain_size`` (``--grain-size``) is the OTHER half of a film stock: ``--grain`` says
# how MUCH silver, this says how COARSE each speck is -- a fine 100-ISO grain and a
# chunky pushed-Tri-X grain were one knob until it existed. It scales that reference
# grid INVERSELY, because a coarser particle means FEWER of them across one negative:
# the field is drawn on a SMALLER grid and stretched further. The ratio is ALL it does,
# so every property above survives -- the seed stays fixed, the grid stays derived from
# the frame's own dimensions (one value, one look at any export size), and the resample
# still renormalises the variance so ``GRAIN_SIGMA_SCALE`` keeps its meaning. 1.0 is the
# shipped grid, byte for byte, on both the per-pixel and the resampled path; the 0.25..4.0
# gate at the CLI door is reasoned about in ``config.LOOK_RANGES``. Not a constant here on
# purpose: the size is a per-CALL knob, so it must not join the hashed tunables.


def _grain_weight(rgb: np.ndarray) -> np.ndarray:
    """Per-pixel grain visibility in [0, 1]: ~1 across the midtones, smoothstepped
    to ~0 in deep shadows and near-clipped highlights where film grain barely
    reads. Returns an (H,W) float32."""
    tone = _luma(rgb) / _peak(rgb)
    rise = np.clip(tone / GRAIN_SHADOW_KNEE, 0.0, 1.0)
    fall = np.clip((1.0 - tone) / (1.0 - GRAIN_HIGHLIGHT_KNEE), 0.0, 1.0)
    window = rise * fall
    return (window * window * (3.0 - 2.0 * window)).astype(np.float32)  # smoothstep


def _grain_field(shape: tuple[int, int], size: float = 1.0) -> np.ndarray:
    """Unit-variance monochrome noise at ``shape``, particles sized to the export.

    At or under the reference dim this is the original per-pixel draw,
    unchanged bytes. Past it the field is drawn on a reference-sized grid and
    resampled up, so one particle covers ``long_edge / reference``
    pixels and the look holds at any export size. Cubic resampling both
    correlates the samples and drops their variance, so the result is
    renormalised: ``GRAIN_SIGMA_SCALE`` keeps meaning the same 8-bit sigma it
    means at the reference size, rather than a sigma that quietly decays with
    the frame. Deterministic — one fixed-seed rng, and a grid derived from the
    frame's own dimensions. ``size`` is the PARTICLE size: it divides the
    reference dim, so a coarser speck is drawn on a smaller grid and stretched
    further; 1.0 is ``GRAIN_REFERENCE_DIM`` itself, byte for byte."""
    height, width = shape
    rng = np.random.default_rng(GRAIN_SEED)
    long_edge = max(height, width)
    reference = GRAIN_REFERENCE_DIM / max(size, 1e-6)
    if long_edge <= reference:
        return rng.standard_normal(shape).astype(np.float32)
    scale = reference / long_edge
    grid = (max(2, round(height * scale)), max(2, round(width * scale)))
    grown = cv2.resize(
        rng.standard_normal(grid).astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    )
    return grown / max(float(grown.std()), 1e-6)


def _grain(rgb: np.ndarray, amount: float, size: float = 1.0) -> np.ndarray:
    """Add luma-weighted monochrome film grain to a finished frame; ``amount<=0`` is
    a byte-identical no-op (early return), so the default develop path is exactly
    what it was before grain existed. DETERMINISTIC by construction: the noise field
    is drawn from a fixed-seed ``default_rng``, so re-developing one frame at one
    amount reproduces the exact bytes. Unlike vignette/clarity the field is ADDED,
    not a gain, so it does not ride ``_apply_gain_mask``. ``amount`` is how MUCH
    grain, ``size`` how COARSE each particle is (1.0 = the shipped speck)."""
    if amount <= 0.0:
        return rgb
    # The sigma is in 8-BIT levels, so it is carried into the frame's own range —
    # the same look at either depth rather than 1/257th of it at 16 bits.
    sigma = amount * GRAIN_SIGMA_SCALE * (_peak(rgb) / 255.0)
    field = sigma * _grain_weight(rgb) * _grain_field(rgb.shape[:2], size)
    return _requantize(rgb.astype(np.float32) + field[:, :, np.newaxis], rgb)


# Halation -- the bright-highlight bleed of film: light that punched through the
# emulsion scattered off the base and re-exposed the neighbouring grains, so a
# bright area smears a glow into what surrounds it. That glow is
# RED-BIASED by physics rather than taste: the short wavelengths are absorbed in the
# upper layers and the anti-halation backing, so what scatters back is the long end
# -- a blown window rims warm, never white, and a neutral bloom reads as a video
# effect. The bleed starts at a seed luma the FRAME sets (``_halation_seed``: the
# SEED_* knobs place it, the THRESHOLD_* ones bound it) and reaches full
# ``HALATION_SPAN`` of luma above that seed -- a fixed distance, not a frame-relative
# one, so a genuinely bright frame still glows harder than a flat one.
# ``HALATION_RADIUS`` is the px blur radius at ``SHARPEN_REFERENCE_WIDTH``
# (width-scaled exactly as clarity's and sharpen's are, so the look is
# resolution-independent), ``HALATION_STRENGTH`` the screen opacity at amount 1.0,
# ``HALATION_RED_BIAS`` the DEFAULT (r,g,b) glow weights, which ``--halation-tint``
# replaces when an operator wants a bloom film never had (a neon cyan or magenta
# rim) -- floats and tuples, so ``_tunable_constants`` hashes a retune.
HALATION_SEED_PERCENTILES = (50.0, 99.5)
HALATION_SEED_FRACTION = 0.6
HALATION_THRESHOLD_FLOOR = 0.35
HALATION_THRESHOLD_CEILING = 0.85
HALATION_SPAN = 0.08
HALATION_RADIUS = 24.0
HALATION_STRENGTH = 0.55
HALATION_RED_BIAS = (1.0, 0.55, 0.3)


def _halation_seed(rgb: np.ndarray) -> float:
    """The luma (0..1) this frame's bleed starts at — measured, never a constant.

    An absolute threshold is how this stage ships INERT: this grade chain's output
    on a real photograph tops out near 0.67 luma and never clips, so a constant
    calibrated on a synthetic full-range frame sat above every pixel and bled
    nothing — while passing every blown-frame test. The seed is instead anchored
    ``HALATION_SEED_FRACTION`` of the way from the frame's median to its highlight
    end (``HALATION_SEED_PERCENTILES``): strictly BETWEEN the two, so pixels above it
    exist by construction unless the frame is flat. A bare high percentile is not
    enough — it lands INSIDE a large uniform highlight, where nothing exceeds it —
    and the high anchor is p99.5, not the max, so one hot pixel cannot set the seed.
    The FLOOR stops a dark frame blooming its noise (its whole range sits under it);
    the CEILING stops a blown one seeding at the clip point. Metered on the bounded
    view the sharpen ramp uses: a 20 MP percentile costs a full sort."""
    view = _luma(_bounded_view(rgb, SHARPEN_DETAIL_MAX_DIM)) / _peak(rgb)
    low, high = np.percentile(view, HALATION_SEED_PERCENTILES)
    seed = low + HALATION_SEED_FRACTION * (high - low)
    return float(min(max(seed, HALATION_THRESHOLD_FLOOR), HALATION_THRESHOLD_CEILING))


def _halation_glow(rgb: np.ndarray, amount: float, tint: tuple[float, ...] = ()) -> np.ndarray:
    """The (H,W,3) float32 glow this frame's highlights scatter, in 0..1.

    Only luma above ``_halation_seed`` scatters, ramped to full ``HALATION_SPAN``
    above it, so the bleed grows with how far into the highlights a pixel sits
    instead of switching on at an edge; that mask is Gaussian-blurred at the
    width-scaled radius and weighted per channel by ``tint`` -- an empty ``tint``
    (the default) IS ``HALATION_RED_BIAS``, so the film look is what an operator
    who never names a colour still gets, byte-for-byte."""
    luma = _luma(rgb).astype(np.float32) / _peak(rgb)
    excess = np.clip((luma - _halation_seed(rgb)) / HALATION_SPAN, 0.0, 1.0)
    radius = HALATION_RADIUS * max(1.0, rgb.shape[1] / SHARPEN_REFERENCE_WIDTH)
    bloom = cv2.GaussianBlur(excess, (0, 0), radius)[:, :, np.newaxis]
    weights = np.asarray(tint or HALATION_RED_BIAS, dtype=np.float32)
    return np.clip((amount * HALATION_STRENGTH) * bloom * weights, 0.0, 1.0)


def _halation(rgb: np.ndarray, amount: float, tint: tuple[float, ...] = ()) -> np.ndarray:
    """Bleed a soft glow -- red-biased film physics unless ``tint`` names another
    colour -- out of a finished frame's highlights; ``amount<=0`` is a
    byte-identical no-op (early return), so the default develop
    path is exactly what it was before halation existed. Runs AFTER ``_vignette``
    and BEFORE ``_grain`` because halation happens in the EMULSION: under the film
    grain, over the lens vignette (glass, in front of the film — corners it has
    already darkened have less left to bleed). SCREEN-blended rather than added:
    screen cannot overshoot, so an already-bright highlight stays put and only its
    surroundings brighten — a glow, not a global exposure lift."""
    if amount <= 0.0:
        return rgb
    src = rgb.astype(np.float32) / _peak(rgb)
    out = 1.0 - (1.0 - src) * (1.0 - _halation_glow(rgb, amount, tint))
    return _requantize(out * _peak(rgb), rgb)


# Light leak -- stray light that got past the camera back (a sprung door, a loose
# canister) and fogged the film before it was ever developed. It is not a highlight
# effect: the leak exposes the emulsion from ONE corner regardless of what was in the
# frame, so it is a fixed geometry rather than anything measured. Fixed is the whole
# design -- ``LIGHT_LEAK_CORNER`` is the (y, x) corner in normalized frame coordinates,
# ``LIGHT_LEAK_TINT`` the warm (r,g,b) weights of the fog, and there is no rng anywhere,
# so one amount renders one look every time (seeded placement variety would be a
# different feature). ``LIGHT_LEAK_STRENGTH`` is the screen opacity at that corner at
# amount 1.0, ``LIGHT_LEAK_REACH`` how far the ramp carries -- as a FRACTION OF THE
# FRAME DIAGONAL, never in pixels: a spatial effect sized in pixels ships a different
# look at every export size (the lesson ``GRAIN_REFERENCE_DIM`` records, measured).
# Floats and tuples, so ``_tunable_constants`` hashes a retune or a recolour.
LIGHT_LEAK_CORNER = (0.0, 1.0)  # top-right
LIGHT_LEAK_REACH = 0.55
LIGHT_LEAK_STRENGTH = 0.65
LIGHT_LEAK_TINT = (1.0, 0.62, 0.28)


def _light_leak_ramp(height: int, width: int) -> np.ndarray:
    """The (H,W) float32 fog profile in 0..1: 1.0 at ``LIGHT_LEAK_CORNER``,
    smoothstepped to exactly 0 at ``LIGHT_LEAK_REACH`` of the frame's DIAGONAL away
    and flat 0 beyond -- so the far side of the frame is untouched by construction,
    not by a threshold. Distance is metered in pixels over that diagonal, which keeps
    the ramp frame-relative (the same fraction of the frame at any export size) while
    staying honest about aspect: normalized coordinates would stretch it on a wide
    frame."""
    corner_y, corner_x = LIGHT_LEAK_CORNER
    diagonal = float(np.hypot(height, width))
    ys = (np.arange(height, dtype=np.float32) - corner_y * (height - 1)) / diagonal
    xs = (np.arange(width, dtype=np.float32) - corner_x * (width - 1)) / diagonal
    reach = np.hypot(ys[:, np.newaxis], xs[np.newaxis, :]) / LIGHT_LEAK_REACH
    ramp = np.clip(1.0 - reach, 0.0, 1.0)
    return (ramp * ramp * (3.0 - 2.0 * ramp)).astype(np.float32)  # smoothstep


def _light_leak(rgb: np.ndarray, amount: float = 0.0) -> np.ndarray:
    """Bleed a warm corner glow across a finished frame; ``amount<=0`` is a
    byte-identical no-op (early return), so the default develop path is exactly what
    it was before light leaks existed. Runs immediately AFTER ``_halation`` and
    BEFORE ``_chromatic_ab``: a leak is stray light exposing the EMULSION, so it
    belongs with halation's bleed rather than above the lens fringe, and under the
    grain like every other emulsion effect. SCREEN-blended rather than added, for
    the reason ``_halation`` gives: screen cannot overshoot, so an already-bright
    area stays put and only its surroundings lift -- a fog, not an exposure push.
    DETERMINISTIC by construction: one corner, one hue, one ramp shape, and no rng
    to seed."""
    if amount <= 0.0:
        return rgb
    src = rgb.astype(np.float32) / _peak(rgb)
    ramp = _light_leak_ramp(rgb.shape[0], rgb.shape[1])[:, :, np.newaxis]
    weights = np.asarray(LIGHT_LEAK_TINT, dtype=np.float32)
    fog = np.clip((amount * LIGHT_LEAK_STRENGTH) * ramp * weights, 0.0, 1.0)
    return _requantize((1.0 - (1.0 - src) * (1.0 - fog)) * _peak(rgb), rgb)


# Lateral chromatic aberration -- the LENS fringe, and the one look stage that is
# geometry rather than tone. Glass bends short wavelengths harder than long ones, so
# an off-axis point lands as three slightly different-sized images: red magnified a
# touch past green, blue a touch under it, green the reference that does not move.
# The displacement is ZERO on the optical axis and grows with distance from it --
# which is why real fringing only ever shows in the corners -- so the whole stage is
# one radial rescale per fringing channel about the frame centre.
# ``CHROMATIC_AB_CORNER_SHIFT`` is that corner displacement at amount 1.0, expressed
# as a FRACTION OF THE HALF-DIAGONAL rather than in pixels: a spatial effect sized in
# pixels ships a different look at every export size (the lesson ``GRAIN_REFERENCE_DIM``
# records, measured), and a 20 MP master downscaled to 2 K must carry the fringe the
# 2 K export carries. Float, so ``_tunable_constants`` hashes a retune.
CHROMATIC_AB_CORNER_SHIFT = 0.004


def _radial_scale_maps(height: int, width: int, magnification: float) -> tuple[np.ndarray, ...]:
    """The ``cv2.remap`` grid pair that resamples one plane at ``magnification``
    about the frame centre: above 1.0 throws content radially OUT, below 1.0 pulls
    it IN, linearly in radius as first-order lateral aberration is. The centre maps
    to ITSELF exactly, so the on-axis pixel cannot move whatever the amount."""
    centre_y, centre_x = (height - 1) / 2.0, (width - 1) / 2.0
    ys = (np.arange(height, dtype=np.float32) - centre_y) / magnification + centre_y
    xs = (np.arange(width, dtype=np.float32) - centre_x) / magnification + centre_x
    return np.repeat(xs[np.newaxis, :], height, axis=0), np.repeat(ys[:, np.newaxis], width, axis=1)


def _chromatic_ab(rgb: np.ndarray, amount: float) -> np.ndarray:
    """Fringe a finished frame the way a real lens does -- red thrown radially
    outward, blue radially inward, green fixed; ``amount<=0`` is a byte-identical
    no-op (early return), so the default develop path is exactly what it was before
    this stage existed. Runs AFTER ``_halation`` and BEFORE ``_grain`` for the same
    reason halation itself sits there: the fringe arrives through the GLASS, so it
    is over the emulsion's bleed, but the sensor still resolves it -- so it belongs
    under the noise floor ``_grain`` stands in for, not over it. DETERMINISTIC by
    construction: no RNG at all, only a geometry derived from the frame's own
    dimensions, and the resample is an interpolation of in-range codes, so the
    output cannot leave the frame's range or change its dtype."""
    if amount <= 0.0:
        return rgb
    shift = amount * CHROMATIC_AB_CORNER_SHIFT
    out = rgb.copy()
    for channel, magnification in ((0, 1.0 + shift), (2, 1.0 - shift)):
        map_x, map_y = _radial_scale_maps(rgb.shape[0], rgb.shape[1], magnification)
        out[:, :, channel] = cv2.remap(
            np.ascontiguousarray(rgb[:, :, channel]),
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            # The inward-pulled channel reads past the frame edge; replicating holds
            # the border colour there rather than fringing against invented black.
            borderMode=cv2.BORDER_REPLICATE,
        )
    return out


# Scanlines -- the CRT tube's line structure laid over the finished frame: a
# row-periodic darkening on a SOFT SINE profile. Soft is the whole design; a comb
# of hard on/off rows carries energy at the Nyquist limit, so any resize (the
# site's downscale, a contact sheet) folds it into moire. ``SCANLINE_DEPTH`` is
# the gain drop in the trough at amount 1.0 (the crest is always gain 1.0, so the
# frame darkens by at most half of it on average). ``SCANLINE_PERIOD`` is the row
# spacing IN PIXELS at ``SCANLINE_REFERENCE_DIM`` -- the same anchor
# ``GRAIN_REFERENCE_DIM`` gives the grain field, for the same reason: a line is a
# feature of the TUBE, not of the sensor, so one flag value must name one line
# DENSITY (here 300 cycles across the long edge) rather than one pixel spacing
# that quietly becomes a different look at every export size. ``SCANLINE_MIN_PERIOD``
# is the sampling floor: under three rows per cycle the sine cannot be resolved
# and aliases back into the hard comb it exists to avoid, so a frame far under the
# reference (a contact-sheet panel) holds the coarsest line structure it can carry.
# Floats, so ``_tunable_constants`` hashes a retune.
SCANLINE_DEPTH = 0.35
SCANLINE_PERIOD = 4.0
SCANLINE_REFERENCE_DIM = 1200
SCANLINE_MIN_PERIOD = 3.0


def _scanline(rgb: np.ndarray, amount: float) -> np.ndarray:
    """Lay a CRT line structure over a finished frame; ``amount<=0`` is a
    byte-identical no-op (early return), so the default develop path is exactly
    what it was before scanlines existed. Runs immediately BEFORE ``_grain`` (and
    after ``_chromatic_ab``, which lands above it because the lens fringe arrives
    through the GLASS): the tube is what the emulsion's
    glow is displayed ON, and the grain stays the last texture of all. Rides the
    ``_apply_gain_mask`` seam with an (H,1) gain -- ONE value per ROW, broadcast
    across every column, so the modulation cannot vary along x by construction.
    DETERMINISTIC by construction too: a pure function of the row index, with no
    rng to seed."""
    if amount <= 0.0:
        return rgb
    period = SCANLINE_PERIOD * max(rgb.shape[:2]) / SCANLINE_REFERENCE_DIM
    rows = np.arange(rgb.shape[0], dtype=np.float32).reshape(-1, 1)
    trough = 0.5 * (1.0 - np.cos(rows * (2.0 * np.pi / max(period, SCANLINE_MIN_PERIOD))))
    return _apply_gain_mask(rgb, 1.0 - (amount * SCANLINE_DEPTH) * trough)


# Holographic sheen -- the oil-slick read of a foil print: an iridescent tint whose HUE
# turns with TONE, so shadows, midtones and highlights each pick up a different cast and
# the frame shimmers instead of shifting as a whole. ``HOLO_TURNS`` is how much of the hue
# wheel that sweep covers from the frame's darkest metered tone to its brightest -- under
# one full turn, so no two tones can land on the same cast -- and ``HOLO_PHASE`` is where
# the shadows start on it (0.5 = cyan shadows, magenta midtones, warm highlights).
# ``HOLO_AMPLITUDE`` is the per-channel offset at amount 1.0 as a FRACTION OF FULL SCALE
# (~9 codes of 255): a SHEEN, an order of magnitude under ``duotone``/``mono_mix``, which
# already exist for wholesale recolouring. ``HOLO_CHANNEL_PHASES`` sit the three channels
# 120 degrees apart on that wheel, which is what makes the offsets sum to ZERO at every
# tone -- the tint is pure chroma by construction, so the stage cannot lift or crush
# exposure. The sweep runs over the frame's OWN metered span (``HOLO_TONE_PERCENTILES`` on
# a ``HOLO_MEASURE_MAX_DIM`` view -- a 20 MP percentile is a full sort, and sensor speckle
# must not set the span), and that calibration lives in THIS module's globals, where
# ``_tunable_constants`` hashes a retune as it does ``_halation_seed``'s. Under
# ``HOLO_MIN_SPAN`` a frame holds no range to sweep and falls back to absolute tone.
HOLO_AMPLITUDE = 0.035
HOLO_TURNS = 0.75
HOLO_PHASE = 0.5
HOLO_CHANNEL_PHASES = (0.0, 1.0 / 3.0, 2.0 / 3.0)
HOLO_TONE_PERCENTILES = (1.0, 99.0)
HOLO_MIN_SPAN = 0.05
HOLO_MEASURE_MAX_DIM = 512.0


def _holo_tone(rgb: np.ndarray) -> np.ndarray:
    """This frame's luma on its OWN measured span, in [0, 1] — 0 at the metered shadow end,
    1 at the highlight end. Frame-relative is not a refinement but the difference between
    working and inert: an absolute ramp over a chain that peaks near 0.66 luma sweeps two
    thirds of the wheel on one photograph and a third on the next."""
    peak = _peak(rgb)
    tone = _luma(rgb) / peak
    view = _luma(_bounded_view(rgb, HOLO_MEASURE_MAX_DIM)) / peak
    low, high = np.percentile(view, HOLO_TONE_PERCENTILES)
    if high - low < HOLO_MIN_SPAN:  # no range to sweep: absolute tone, not noise
        return np.clip(tone, 0.0, 1.0).astype(np.float32)
    return np.clip((tone - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _holo_sheen(rgb: np.ndarray, amount: float) -> np.ndarray:
    """Rotate an iridescent tint across a finished frame's tonal range; ``amount<=0``
    is a byte-identical no-op (early return), so the default develop path is exactly
    what it was before this stage existed. Runs AFTER ``_chromatic_ab`` and BEFORE
    ``_scanline``: the foil is a property of the PRINT SURFACE, so it lies over the
    fringe the glass delivered, and under both the tube that displays the print
    (``_scanline``) and the emulsion's own texture (``_grain``), last of all.
    FRAME-RELATIVE via ``_holo_tone``: an ABSOLUTE ramp is how a tone-keyed stage ships
    green and INERT, since this chain's output peaks near 0.66 luma and never clips (the
    lesson ``_halation_seed`` records at length). DETERMINISTIC by construction: no RNG at
    all, and a pixel's tint is decided by its own luma alone, so two pixels of equal tone
    match wherever they sit in the frame."""
    if amount <= 0.0:
        return rgb
    turn = (2.0 * np.pi) * (HOLO_PHASE + HOLO_TURNS * _holo_tone(rgb))[:, :, np.newaxis]
    phases = (2.0 * np.pi) * np.asarray(HOLO_CHANNEL_PHASES, dtype=np.float32)
    tint = (amount * HOLO_AMPLITUDE * _peak(rgb)) * np.cos(turn - phases)
    return _requantize(rgb.astype(np.float32) + tint, rgb)


# Glitch -- the digital-corruption read, and the one stage here that is a fault in the
# SIGNAL rather than in the glass or the emulsion: the frame is cut into a grid of
# horizontal BANDS, a subset is displaced sideways, and each torn band splits its channels
# by a fraction of its own displacement. ONE knob drives both halves, so the operator has a
# single dial rather than three: ``amount`` scales the FRACTION of bands that tear
# (``GLITCH_TORN_FRACTION`` at 1.0) and the displacement itself. ``GLITCH_SHIFT_RANGE`` is
# that displacement's min..max as a FRACTION OF FRAME WIDTH, never a pixel count -- the
# lesson ``GRAIN_REFERENCE_DIM`` records, measured -- so amount 1.0 tears at most a twentieth
# of the frame sideways at any export size, conservative on purpose because the picture must
# still read as the picture. ``GLITCH_BAND_HEIGHT`` is the band depth IN PIXELS at
# ``GLITCH_REFERENCE_DIM``, the anchor the grain field and the scanline spacing take and for
# the same reason: one value must name one band COUNT, not a pixel depth that quietly becomes
# a different look on every export. ``GLITCH_SEED`` is a FIXED module constant -- the pattern
# comes from a seeded ``default_rng``, never global ``np.random`` or entropy, so two develops
# of one frame at one amount are byte-identical (the cache/algo-version invariants require
# it). Floats, an int and a tuple, so ``_tunable_constants`` hashes a retune or a reseed.
GLITCH_SEED = 0x676C63  # fixed; determinism is a hard invariant
GLITCH_REFERENCE_DIM = 1200
GLITCH_BAND_HEIGHT = 24.0
GLITCH_TORN_FRACTION = 0.5
GLITCH_SHIFT_RANGE = (0.012, 0.05)
GLITCH_CHANNEL_SPLIT = 0.35


def _glitch_row_shift(shape: tuple[int, int], amount: float) -> np.ndarray:
    """Per-ROW horizontal displacement in whole pixels — one value per BAND, repeated across
    that band's rows, so a tear is a block of the picture and never per-row noise. The grid is
    reference-anchored, so its COUNT is a property of the frame's aspect alone and ONE seeded
    draw describes every size; the magnitude is a fraction of the frame's own width, floored
    away from zero so a torn band reads as torn."""
    height, width = shape
    band = max(1.0, GLITCH_BAND_HEIGHT * max(shape) / GLITCH_REFERENCE_DIM)
    count = max(1, round(height / band))
    rng = np.random.default_rng(GLITCH_SEED)
    draw = rng.uniform(-1.0, 1.0, count)
    low, high = GLITCH_SHIFT_RANGE
    torn = rng.random(count) < amount * GLITCH_TORN_FRACTION
    span = amount * width * (low + (high - low) * np.abs(draw))
    shift = np.where(torn, np.sign(draw) * span, 0.0)
    return np.rint(shift[np.arange(height) * count // height])


def _glitch(rgb: np.ndarray, amount: float = 0.0) -> np.ndarray:
    """Tear a finished frame into sideways-displaced scan bands; ``amount<=0`` is a
    byte-identical no-op (early return), so the default develop path is exactly what
    it was before glitch existed. Runs AFTER ``_scanline`` and BEFORE ``_grain``: the
    tear is a fault in the SIGNAL being displayed, so it sits over the tube's own line
    structure, and the grain stays the last texture of all. A band's three channels are
    gathered at ``GLITCH_CHANNEL_SPLIT``-separated offsets — the RGB fringe a corrupted band
    carries. The whole stage is a horizontal GATHER of existing codes along axis 1, so rows
    cannot move, no code is synthesized, and dtype and range hold by construction, not a gate."""
    if amount <= 0.0:
        return rgb
    shift = _glitch_row_shift(rgb.shape[:2], amount)
    columns = np.arange(rgb.shape[1], dtype=np.int32)
    out = rgb.copy()
    for channel, weight in enumerate((1.0 + GLITCH_CHANNEL_SPLIT, 1.0, 1.0 - GLITCH_CHANNEL_SPLIT)):
        offset = np.rint(weight * shift).astype(np.int32)[:, np.newaxis]
        # Clipping holds the border column where a band reads past the frame edge —
        # the same choice ``_chromatic_ab`` makes with BORDER_REPLICATE.
        taken = np.clip(columns - offset, 0, rgb.shape[1] - 1)
        out[:, :, channel] = np.take_along_axis(rgb[:, :, channel], taken, axis=1)
    return out


def _spatial_chain(rgb: np.ndarray, o) -> tuple[np.ndarray, float]:
    """The post-grade spatial stages of one variant in chain order — dehaze, glow,
    clarity, sharpen, sky, grad-ND, vignette, halation, light-leak, chromatic-ab,
    holo-sheen, scanline, glitch, grain — plus the sharpen amount they measured. ``o`` is
    the run's ``VariantPlan`` or one look's own ``DevelopOptions``; a stage is added HERE."""
    bloomed = _glow(_dehaze(rgb, o.dehaze), o.glow)
    sharpened, amount = _sharpen(_clarity(bloomed, o.clarity), o.sharpen)
    skied = _sky_saturate(_sky_darken(sharpened, o.sky_darken), o.sky_saturate)
    filtered = _graduated_nd(skied, o.graduated_nd, o.graduated_nd_angle)
    spatial = _halation(_vignette(filtered, o.vignette), o.halation, o.halation_tint)
    fogged = _light_leak(spatial, o.light_leak)
    fringed = _holo_sheen(_chromatic_ab(fogged, o.chromatic_ab), o.holo_sheen)
    grained = _grain(_glitch(_scanline(fringed, o.scanline), o.glitch), o.grain, o.grain_size)
    return grained, amount


def _analysis_rgb(rgb: np.ndarray) -> np.ndarray:
    """uint8 companion for metering: ``analyze_frame`` and the noise estimator
    are 8-bit-calibrated, so a 16-bit frame is downconverted for *measurement
    only* — the grade itself consumes the full-depth original."""
    if rgb.dtype == np.uint16:
        return (rgb >> 8).astype(np.uint8)
    return rgb


# Metering calibration scale (long-edge pixels). The grade engine's
# saturation/mid-tone thresholds were calibrated on the video sampler's fixed
# 160x90 analysis frames (``color_analysis.sample_frames``); metered per-pixel
# at 20 MP, saturation reads sensor noise as chroma and inflates relative to
# that calibration, so the thresholds misfire on stills. The meter therefore
# shrinks to the same long edge; the grade still consumes the full frame.
ANALYSIS_MAX_DIM = 160


def _bounded_resize(rgb: np.ndarray, max_dim: int) -> np.ndarray:
    """Area-average ``rgb`` down so its long edge fits ``max_dim``, aspect kept.

    INTER_AREA blends every source pixel into the output instead of discarding
    rows — the averaging is what cancels the per-pixel sensor noise a stride
    would carry through to the meter unchanged."""
    long_edge = max(rgb.shape[0], rgb.shape[1])
    if long_edge <= max_dim:
        return rgb
    scale = max_dim / long_edge
    size = (max(1, round(rgb.shape[1] * scale)), max(1, round(rgb.shape[0] * scale)))
    return cv2.resize(np.ascontiguousarray(rgb), size, interpolation=cv2.INTER_AREA)


def _frame_stats(rgb: np.ndarray) -> SegmentStats:
    """One frame's stats in the shape the cluster/correction helpers expect,
    metered at the video-calibrated analysis scale."""
    view = _bounded_resize(_analysis_rgb(rgb), ANALYSIS_MAX_DIM)
    return aggregate_segment_stats([analyze_frame(view)])


def _grade(rgb: np.ndarray, params: CorrectionParams, depth: int = 8) -> np.ndarray:
    """Run the shared numpy grade chain over a whole frame.

    Same filter order as ``params_to_lut3d_cube`` (curves -> colorbalance ->
    saturation -> gamma), so a still and a video frame of one scene land on the
    same look. The reference match is the one stage AFTER gamma: it lands the
    finished frame on a chosen hero frame's Lab statistics, so anything run after
    it would pull the frame back off the reference (OFF ⇒ bit-identical).
    Internals are float32 — float64 flattens a 20 MP frame to ~500 MB
    and every apply step copies it, for precision far below one 16-bit code. A
    16-bit source rides the chain at full depth and quantizes only here, at the
    very end. ``depth`` IS that quantization: 8 (the default) returns uint8 as
    before (``analyze_frame`` normalizes by 255 itself), 16 returns uint16 — the
    archival master's point, since this line is where the sensor headroom and the
    chain's own sub-8-bit precision used to be thrown away."""
    height, width, _ = rgb.shape
    peak = _peak(rgb)
    out_peak = 65535.0 if depth == 16 else 255.0
    flat = rgb.reshape(-1, 3).astype(np.float32) / peak

    out = _apply_exposure(flat, params.exposure_ev)
    out = _apply_curves(out, params)
    out = _apply_luma_curve(out, params)
    out = _apply_channel_curves(out, params)
    out = _apply_colorbalance(out, params)
    # The mono conversion runs FIRST of the colour-look stages, and specifically
    # ahead of the split-tone: it writes every pixel, so a tone laid down before
    # it was wiped and split-toned monochrome could not be expressed. Both mono
    # controls are empty on every shipped look and style, so this order is
    # invisible to them (``tests/test_photo_bw_mixer.py`` pins that).
    out = _apply_monomix(out, params)
    out = _apply_splittone(out, params)
    out = _apply_saturation(out, params.saturation_mult)
    out = _apply_vibrance(out, params)
    out = _apply_hsl(out, params)
    out = _apply_gamma(out, params.gamma)
    out = _apply_reference_match(out, params)

    quantized = np.rint(np.clip(out, 0.0, 1.0) * out_peak)
    return quantized.astype(np.uint16 if depth == 16 else np.uint8).reshape(height, width, 3)


def _straighten(rgb: np.ndarray, angle: float) -> np.ndarray:
    """Rotate ``rgb`` by ``angle`` degrees about its centre (positive =
    counter-clockwise), levelling a tilted horizon or straightening a canted
    frame — the same primitive serves either. The canvas size is UNCHANGED
    (corners exposed by the turn are filled by replicating the edge, never a
    hard black border), so shape stays stable for the crop that may follow.
    0.0 is an exact no-op — the array is copied, never rotated through
    ``cv2.warpAffine`` at all, so it is bit-identical to the source."""
    if angle == 0.0:
        return rgb.copy()
    h, w = rgb.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(rgb, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _crop(rgb: np.ndarray, rect: tuple[float, float, float, float]) -> np.ndarray:
    """``rgb`` trimmed to ``(left, top, right, bottom)`` fractions of its own
    frame, applied AFTER any straighten so the rect reads against the
    levelled frame. Empty ``rect`` (the default) is the full frame, an exact
    no-op — no copy, no bounds arithmetic at all."""
    if not rect:
        return rgb
    h, w = rgb.shape[:2]
    left, top, right, bottom = rect
    lo_x, hi_x = round(left * w), round(right * w)
    lo_y, hi_y = round(top * h), round(bottom * h)
    return rgb[max(0, lo_y) : min(h, hi_y), max(0, lo_x) : min(w, hi_x)]


def develop(
    source,
    params: CorrectionParams | None = None,
    cluster_index: int = 0,
    vignette: float = 0.0,
    dehaze: float = 0.0,
    sky_darken: float = 0.0,
    sky_saturate: float = 0.0,
    graduated_nd: float = 0.0,
    graduated_nd_angle: float = 0.0,
    glow: float = 0.0,
    clarity: float = 0.0,
    grain: float = 0.0,
    grain_size: float = 1.0,
    halation: float = 0.0,
    halation_tint: tuple[float, ...] = (),
    light_leak: float = 0.0,
    chromatic_ab: float = 0.0,
    holo_sheen: float = 0.0,
    scanline: float = 0.0,
    glitch: float = 0.0,
    sharpen: float = 1.0,
) -> DevelopedPhoto:
    """Decode (if needed) and grade one photo.

    ``source`` is a path or an already-decoded ``DecodedPhoto``; ``params=None``
    measures this frame alone, batch callers pass the cluster's correction.

    Denoise runs on the source so grading can't amplify grain first; sharpen
    runs on the graded result, which is the frame the viewer actually sees.
    Both are conditional on measurement, so a clean and already-sharp photo
    comes back byte-identical to the pure grade. ``sharpen`` scales that
    measured unsharp amount (1.0 = the measured ramp untouched). Dehaze subtracts
    the atmospheric veil FIRST, before clarity. ``glow`` screens the Orton bloom
    between the two, so clarity and sharpen resolve the detail back out of it.
    Sky-darken rides its soft mask
    under the vignette, and ``sky_saturate`` boosts the chroma of that same masked
    sky right after it; ``graduated_nd`` then lays a straight darkening ramp — a
    filter, not a region — between those two and the vignette, tilted off level by
    ``graduated_nd_angle``. Halation bleeds
    over the vignette, and grain (deterministic) is laid last of all, over both.
    ``grain_size`` sets how COARSE that grain's particles are (1.0 = the shipped speck).
    ``halation_tint`` colours that bleed (empty = the red film default).
    ``light_leak`` fogs one fixed corner warm, directly on that bleed.
    ``chromatic_ab`` fringes the corners between the two.
    ``holo_sheen`` rotates its iridescent tint over that fringe.
    ``scanline`` lays the CRT line structure between the two.
    ``glitch`` tears the displayed signal into displaced bands over that structure.
    """
    decoded = source if isinstance(source, DecodedPhoto) else decode(source)
    if params is None:
        params = compute_correction(_frame_stats(decoded.rgb), mode=GRADE_MODE_PHOTO)

    denoised, denoise_h = _denoise(decoded.rgb, params)
    corrected = _glow(_dehaze(_grade(denoised, params), dehaze), glow)
    sharpened, sharpen_amount = _sharpen(_clarity(corrected, clarity), sharpen)
    skied = _sky_saturate(_sky_darken(sharpened, sky_darken), sky_saturate)
    filtered = _graduated_nd(skied, graduated_nd, graduated_nd_angle)
    spatial = _halation(_vignette(filtered, vignette), halation, halation_tint)
    fringed = _holo_sheen(_chromatic_ab(_light_leak(spatial, light_leak), chromatic_ab), holo_sheen)
    graded = _grain(_glitch(_scanline(fringed, scanline), glitch), grain, grain_size)

    logger.debug(
        "Developed %s (cluster %d, denoise h=%.2f, sharpen amount=%.2f)",
        decoded.path,
        cluster_index,
        denoise_h,
        sharpen_amount,
    )
    return DevelopedPhoto(
        rgb=graded,
        path=decoded.path,
        params=params,
        cluster_index=cluster_index,
        denoise_applied=denoise_h > 0.0,
        denoise_h=denoise_h,
        sharpen_applied=sharpen_amount > 0.0,
        sharpen_amount=sharpen_amount,
        decoder=decoded.decoder,
    )


@dataclass(frozen=True)
class DevelopOptions:
    """Operator look/spatial knobs for one develop run. Every field defaults to
    the OFF/neutral value, so ``DevelopOptions()`` reproduces the un-flagged
    grade exactly.

    One record, not 17 positionals: a new creative knob is a FIELD added in its
    own group below — next to its relatives — instead of another positional
    appended at the one shared line of two signatures and every test double.
    Frozen, so the single shared default instance the two signatures bind
    cannot be mutated by a caller (a mutable default would carry one run's
    knobs into the next).
    """

    # ── spatial stages: rendered on the graded frame, after the grade chain ──
    vignette: float = 0.0
    dehaze: float = 0.0  # dark-channel-prior veil subtraction, FIRST of the chain
    sky_darken: float = 0.0  # soft sky mask as a gain, UNDER the vignette
    sky_saturate: float = 0.0  # the same mask as a CHROMA lerp, right after it
    graduated_nd: float = 0.0  # straight darkening ramp — glass, with the optics
    graduated_nd_angle: float = 0.0  # that filter's TILT off level, degrees, |angle| <= 90
    glow: float = 0.0  # screened Orton bloom, after the dehaze and BEFORE the clarity
    clarity: float = 0.0
    grain: float = 0.0
    grain_size: float = 1.0  # particle COARSENESS, not amount (1.0 = the shipped speck)
    halation: float = 0.0  # red-biased highlight bleed, under the grain
    #: (r,g,b) weights on that bleed; empty = ``HALATION_RED_BIAS``, the film default.
    halation_tint: tuple[float, ...] = ()
    light_leak: float = 0.0  # warm fixed-corner fog, on the bleed and under the fringe
    chromatic_ab: float = 0.0  # lateral lens fringe, over the bleed and under the grain
    holo_sheen: float = 0.0  # tone-keyed iridescent tint, over the fringe and under the tube
    scanline: float = 0.0  # CRT line structure, over the halation and under the grain
    glitch: float = 0.0  # torn scan bands, over the line structure and under the grain
    sharpen: float = 1.0  # MULTIPLIER on the measured unsharp amount, not an amount
    # ── geometry: applied to the DECODED frame, before measurement or any grade
    # stage runs — a run-level property of the photo, shared by every variant,
    # never a per-look creative knob (a look changing the frame's shape would
    # make two variants of one photo incomparable). ``develop_variants`` applies
    # lens correction, then straighten, then crop, in that order, via
    # ``lens_correct.correct``/``_straighten``/``_crop``, before the shared
    # ``_denoise`` call every variant grades from.
    #: Lens-profile correction: an EXIF-matched ``lensfun`` lookup that
    #: corrects distortion, vignetting and transverse CA. False (the default)
    #: is an exact no-op — ``lens_correct.correct`` is never even called, so
    #: the default develop path stays byte-identical to before this field
    #: existed. Runs BEFORE straighten/crop, so geometry measures the
    #: corrected frame.
    lens_correct: bool = False
    #: ``--straighten``: rotate by this many degrees (positive = counter-
    #: clockwise) to level a tilted horizon or straighten a canted frame — the
    #: same primitive serves either intent, the angle is just measured
    #: differently upstream. 0.0 (the default) is an exact no-op.
    straighten_angle: float = 0.0
    #: ``--crop``: ``(left, top, right, bottom)`` as fractions of the frame in
    #: [0, 1], applied AFTER straightening. Empty (the default) is the full
    #: frame, an exact no-op.
    crop_rect: tuple[float, float, float, float] = ()
    # ── colour chain: ``replace``d onto every styled variant, never ``base`` ──
    hue_shift: tuple[float, ...] = ()
    hue_sat: tuple[float, ...] = ()
    hue_lum: tuple[float, ...] = ()
    splittone_shadow: tuple[float, ...] = ()
    splittone_midtone: tuple[float, ...] = ()
    splittone_highlight: tuple[float, ...] = ()
    #: ``--curve-l``: the MASTER tone curve, one shape run on R/G/B alike (so no
    #: cast), UNDER the three cross-process channel curves. Empty = OFF.
    curve_l: tuple[float, ...] = ()
    curve_r: tuple[float, ...] = ()
    curve_g: tuple[float, ...] = ()
    curve_b: tuple[float, ...] = ()
    #: White balance, each named EXACTLY as its ``CorrectionParams`` field so
    #: ``_look_params`` carries it into the colour chain for free. The r/b pair is
    #: the amber-blue TEMPERATURE axis (warm = r up, b down) and ``_high`` is that
    #: axis in the highlights alone; all are ADDITIVE — see ``_wb_shifted``.
    white_balance_r: float = 0.0
    white_balance_b: float = 0.0
    white_balance_g: float = 0.0
    white_balance_r_high: float = 0.0
    white_balance_b_high: float = 0.0
    white_balance_g_high: float = 0.0
    #: ``--white-balance-kelvin``/``--white-balance-tint``: a friendlier pair over
    #: the raw r/b/g axes above. Kelvin (signed degrees; + warms) converts to the
    #: r/b temperature axis via ``KELVIN_TO_WB_SCALE``; tint (signed, same units as
    #: ``white_balance_g``) rides straight onto the green-magenta axis. Both are
    #: additive, on top of whatever the raw axes and the measured cast already
    #: hold — see ``_wb_shifted``. 0.0/0.0 (the default) is a no-op.
    white_balance_kelvin: float = 0.0
    white_balance_tint: float = 0.0
    #: ``--exposure-ev``: a stops gain, the FIRST stage of ``_grade`` (see
    #: ``CorrectionParams.exposure_ev``). 0.0 (the default) is a no-op.
    exposure_ev: float = 0.0
    #: ``--shadow-recovery``/``--highlight-recovery``: ADD onto the measured
    #: ``shadow_lift``/``highlight_pull`` inside ``_apply_curves`` (see
    #: ``CorrectionParams.shadow_recovery``/``highlight_recovery``). 0.0/0.0 (the
    #: default) is a no-op.
    shadow_recovery: float = 0.0
    highlight_recovery: float = 0.0
    vibrance: float = 0.0
    mono_mix: tuple[float, ...] = ()
    duotone: tuple[float, ...] = ()
    duotone_strength: float = 1.0  # 1.0 = the full ramp (shipped); 0.0 = plain mono
    duotone_mid: tuple[float, ...] = ()  # empty = a two-ender; else the tritone anchor
    #: ``--reference``: the hero frame's six Lab numbers (empty = OFF), measured
    #: once at the CLI door by ``reference_lab_from_file``, never a path here.
    reference_lab: tuple[float, ...] = ()
    #: Masked layers, each ``{mask: NAME, feather: F, invert: bool, <colour and
    #: tone knobs>}``, composited over the graded frame in list order (``_layered``).
    #: Empty (the default) = OFF and no layer code runs. Colour and tone ONLY — a
    #: spatial stage is a property of the whole frame and is refused at the door.
    layers: tuple[dict, ...] = ()
    #: ``--look-strength``: how much of a ``--look-variant`` panel's RENDERED frame
    #: survives, lerped toward the run's own base render (1.0 = the whole look and
    #: the byte-identical shipped path; 0.0 = the base render). Per-look by design,
    #: so a recipe can carry its own and one dial turns any look into a family.
    look_strength: float = 1.0
    # ── the LUT surface: levers the style lever must never scale ──
    lut_strength: float = 1.0
    lut_over: str = ""
    lut_auto: int = 0  # 0 = the manual path, byte-identical; N = keep the N best-scoring


@dataclass(frozen=True)
class VariantPlan:
    """Metering half of a variants develop — what the CLI dry run prints.

    ``lut_looks`` are film-emulation ``.cube`` variants rendered ON TOP of the
    natural (base) grade — or of ``lut_params``, the ``--lut-over`` styled grade,
    when the operator composes one — kept separate from ``styled`` because a LUT
    is a fixed 3D table, not a ``CorrectionParams`` the style lever can amplify.
    """

    decoded: DecodedPhoto
    scene: SceneProfile
    base: CorrectionParams
    styled: list[tuple[str, CorrectionParams]]
    lut_looks: list[tuple[str, "Cube"]] = field(default_factory=list)
    #: ``--look-variant``: ``(panel name, that look's OWN knob record)`` per named
    #: look, rendered over the base grade as a ``lut_looks`` entry is. Empty (the
    #: default) ⇒ no look panel renders and the run is byte-for-byte unchanged.
    look_variants: list[tuple[str, DevelopOptions]] = field(default_factory=list)
    #: Panel name → the STYLED correction that look's own ``styles`` names, the
    #: base its colour knobs ride (``lut_params`` for the LUT surface, per panel).
    #: A look naming no style is ABSENT and grades from ``base`` — the natural
    #: correction, exactly as every look did before this map existed.
    look_bases: dict[str, CorrectionParams] = field(default_factory=dict)
    #: LUT opacity applied to every ``lut_looks`` entry (1.0 = full-strength).
    lut_strength: float = 1.0
    #: ``--lut-over``: the STYLED correction every LUT look rides instead of
    #: ``base``. ``None`` (the default) keeps the base grade under the cube.
    lut_params: CorrectionParams | None = None
    #: ``--lut-auto``: ``(name, winning score)`` per AUTO pick — the dry run's WHY.
    lut_auto_picks: list[tuple[str, float]] = field(default_factory=list)
    #: Radial-vignette depth applied to every variant's graded frame (0 = off).
    vignette: float = 0.0
    #: Dark-channel-prior veil subtraction, first of the spatial chain (0 = off).
    dehaze: float = 0.0
    #: Soft-sky-mask darkening depth, applied under the vignette (0 = off).
    sky_darken: float = 0.0
    #: Chroma boost on that same masked sky, right after the darkener (0 = off).
    sky_saturate: float = 0.0
    #: Graduated-ND depth, laid with the optics under the vignette (0 = off).
    graduated_nd: float = 0.0
    #: That filter's rotation off level, degrees (0 = level; +/-90 = upright).
    graduated_nd_angle: float = 0.0
    #: Screened Orton bloom, after the dehaze and before the clarity (0 = off).
    glow: float = 0.0
    #: Midtone local-contrast (clarity) depth applied before sharpen (0 = off).
    clarity: float = 0.0
    #: Luminance film-grain amount laid last, after the vignette (0 = off).
    grain: float = 0.0
    #: How COARSE that grain's particles are (1.0 = the shipped speck size).
    grain_size: float = 1.0
    #: Red-biased highlight bleed, over the vignette and under the grain (0 = off).
    halation: float = 0.0
    #: (r,g,b) weights colouring that bleed; empty = the HALATION_RED_BIAS default.
    halation_tint: tuple[float, ...] = ()
    #: Warm fixed-corner light-leak fog, on the bleed and under the fringe (0 = off).
    light_leak: float = 0.0
    #: Lateral chromatic aberration, over the bleed and under the grain (0 = off).
    chromatic_ab: float = 0.0
    #: Iridescent tone-keyed tint, over the fringe and under the tube (0 = off).
    holo_sheen: float = 0.0
    #: CRT line-structure depth, over the halation and under the grain (0 = off).
    scanline: float = 0.0
    #: Torn-scan-band depth, over the line structure and under the grain (0 = off).
    glitch: float = 0.0
    #: Operator MULTIPLIER on the measured unsharp amount (1.0 = unchanged).
    sharpen: float = 1.0


# Panel-name prefix for a ``--look-variant`` look, so a look can never collide on
# disk with a style or a LUT of the same name (as ``_over_``/``_auto`` do there),
# and the ``.grade.json`` sidecar can read the look's name back off the panel.
LOOK_VARIANT_PREFIX = "look_"

# Knobs a look may name that ONE look-variant panel cannot own: the run owns the
# LUT surface, and ``reference`` names a FILE the CLI door already measured.
LOOK_VARIANT_RUN_KNOBS = frozenset({"lut_strength", "lut_over", "lut_auto", "reference_lab"})


#: The white-balance fields an operator value is ADDED to rather than swapped for.
#: ``compute_correction`` writes the measured amber-blue cast onto r/b (and their
#: highlight halves) on every photo base, so a ``replace`` would silently DISCARD
#: it: +0.03 of warmth means "the measured cast plus 0.03", never "0.03 instead".
#: Green rides the same rule for one rule rather than two — no producer sets it,
#: so + and = are the same number and every shipped green recipe is unmoved.
_WB_ADDITIVE = tuple(f.name for f in fields(CorrectionParams) if f.name.startswith("white_balance"))


#: Degrees Kelvin covering ``WB_OPERATOR_SPAN``'s r/b travel — ``+2000`` warms
#: the frame by the same amount ``white_balance_r=0.10`` does, so the two knobs
#: share one authority rather than a Kelvin dial that could outrun the raw axis.
KELVIN_TO_WB_SCALE = 20000.0


def _wb_shifted(params: CorrectionParams, o: DevelopOptions) -> CorrectionParams:
    """``params`` with the operator's white balance ADDED onto whatever the
    measurement and the style already put there. Deliberately unclamped: the
    style rails have already run by the time this lands, so ``WB_OPERATOR_SPAN``
    is chosen to hold the sum inside the renderer's own authority — rather than a
    late clamp, which would lie about what it did.

    ``white_balance_kelvin``/``white_balance_tint`` are the friendlier pair over
    those same raw axes (see the ``DevelopOptions`` fields): converted here and
    ADDED on top, so an operator may use either or both. 0.0/0.0 keeps this an
    exact no-op past the raw-axis sum above."""
    shifted = replace(params, **{f: getattr(params, f) + getattr(o, f) for f in _WB_ADDITIVE})
    if o.white_balance_kelvin or o.white_balance_tint:
        kelvin_rb = o.white_balance_kelvin / KELVIN_TO_WB_SCALE
        shifted = replace(
            shifted,
            white_balance_r=shifted.white_balance_r + kelvin_rb,
            white_balance_b=shifted.white_balance_b - kelvin_rb,
            white_balance_g=shifted.white_balance_g + o.white_balance_tint,
        )
    return shifted


def _look_params(base: CorrectionParams, o: DevelopOptions) -> CorrectionParams:
    """``base`` with ONE knob record's colour chain layered on — the same fields
    ``plan_variants`` lays onto a styled variant, over the natural grade instead.

    That chain IS the field names the two records SHARE (the spatial knobs render
    after the grade, so they share none): derived, not a hand-kept list that goes
    stale the next time a colour knob lands. White balance is the one ADDITIVE
    set — a look panel grades over the run's MEASURED base, so replacing there
    would throw its cast correction away."""
    shared = {f.name for f in fields(DevelopOptions)} & {f.name for f in fields(CorrectionParams)}
    laid = replace(base, **{name: getattr(o, name) for name in shared - set(_WB_ADDITIVE)})
    return _wb_shifted(laid, o)


def _layered(rgb: np.ndarray, layers=(), depth: int = 8) -> np.ndarray:
    """``rgb`` with each masked layer composited over it, in list order.

    A layer is ``{mask: NAME, feather: F, invert: bool, <that mask's own shape
    params>, <colour and tone knobs>}`` — or ``mask: [NAME, NAME]`` plus ``mask_op``,
    which combines the two RAW masks before feather and invert (``mask_for`` pins that
    order and says why), so a recipe can name a region narrower than any single mask.
    Its grade chain runs over the ALREADY-GRADED frame and the result is lerped through
    ``mask_for``'s soft mask — ``out = base + mask * (layered - base)`` — so a recipe can ride the
    frame's own regions (teal sky over warm ground, shadows lifted under a mask)
    instead of the whole rectangle. Each layer meters its
    mask on the RUNNING frame, so a later layer sees what the earlier ones left,
    the way a stack of local adjustments does. Colour and tone only, by
    construction: the layer knob set is gated at config load, and the spatial
    stages run whole-frame after this. Empty (the default) returns the input
    untouched WITHOUT reaching a mask — the shipped path, byte for byte.
    """
    if not layers:
        return rgb
    from ...config import LAYER_TONE_KNOBS
    from .masks import MASK_PARAMS, mask_for

    peak = _peak(rgb)
    for layer in layers:
        params = _look_params(CorrectionParams(), look_variant_options(layer, DevelopOptions()))
        # The TONE half, set on the record DIRECTLY. ``_look_params`` derives the
        # colour chain from the fields the two records SHARE, and no tone knob is
        # one — routed through that trip a ``gamma`` is dropped in silence and
        # renders nothing. ``replace`` refuses a name that is no CorrectionParams
        # field, so a knob that could not render cannot be added here unnoticed.
        params = replace(params, **{k: layer[k] for k in LAYER_TONE_KNOBS & layer.keys()})
        # A combined layer names a PAIR, so the accepted shape params are the union of
        # both operands' — the same set ``mask_for`` and ``check_layers`` accept on.
        named = layer["mask"] if isinstance(layer["mask"], list) else [layer["mask"]]
        accepted = {key for one in named for key in MASK_PARAMS[one]}
        shape = {k: v for k, v in layer.items() if k in accepted}
        mask = mask_for(
            layer["mask"],
            rgb,
            op=layer.get("mask_op"),
            feather=layer.get("feather", 1.0),
            invert=layer.get("invert", False),
            **shape,
        )[:, :, None]
        base = rgb.astype(np.float32)
        moved = _grade(rgb, params, depth).astype(np.float32)
        rgb = np.rint(base + mask * (moved - base)).clip(0.0, peak).astype(rgb.dtype)
    return rgb


def _dial_to_base(base: np.ndarray, look: np.ndarray, strength: float) -> np.ndarray:
    """One look panel's rendered frame lerped toward the run's own base render —
    ``out = base + strength * (look - base)``, the same blend ``--lut-strength``
    lays over a cube and ``_layered`` lays through a mask.

    Rendered pixels, not knobs: a look is a colour chain AND its spatial stages,
    and no per-knob scaling can halve a grain field or a vignette together. 0.0
    returns the base render exactly (the lerp is identity, and ``rint`` of a whole
    number is itself); 1.0 never reaches here — the caller keeps the shipped path.
    """
    under = base.astype(np.float32)
    moved = under + strength * (look.astype(np.float32) - under)
    return np.rint(moved).clip(0.0, _peak(look)).astype(look.dtype)


def look_variant_options(knobs: dict, options: DevelopOptions) -> DevelopOptions:
    """One named look as its OWN knob record: the run's resolved ``options`` with
    that look's colour/spatial knobs replaced on top.

    A look is keyed by develop FLAG name (``LOOK_ALIASES`` maps the seven that
    differ) and carries YAML lists where the record holds tuples; both are
    translated HERE, once. A key that is no ``DevelopOptions`` field
    (``styles``/``format``/``border``/the ``--lut`` specs) stays the run's — one
    panel cannot own a run-level choice. Grading a neutral base through the result
    is the upstream gate: a malformed split-tone or curve raises ``ValueError`` at
    the CLI door, not per photo, mid-batch."""
    from ...config import LOOK_ALIASES

    field_by_flag = {LOOK_ALIASES.get(f.name, f.name): f.name for f in fields(DevelopOptions)}
    moved = {
        name: tuple(v) if isinstance(v, list) else v
        for flag, v in knobs.items()
        if (name := field_by_flag.get(flag, "")) and name not in LOOK_VARIANT_RUN_KNOBS
    }
    resolved = replace(options, **moved)
    _look_params(CorrectionParams(), resolved)
    return resolved


# Auto-select renders this many variants — mirrors ``select_styles``' default.
VARIANT_COUNT = 3

# Style-lever default. Mirrors ``styles.DEFAULT_STRENGTH`` as a literal because
# styles -> scene -> develop is an import cycle (styles is imported lazily
# below); a test pins the two together.
STYLE_STRENGTH_DEFAULT = 1.0

# Params-space pre-filter distance between two auto-selected variants. Below
# it the candidate's params sit on top of an already-picked variant's —
# params-identical implies pixel-identical, so the planner drops it without
# paying a render. Passing this filter proves nothing about pixels: the
# rendered check below is the real keep test.
VARIANT_DISTINCTNESS_EPSILON = 0.05

# Rendered-distinctness metering view bound (long-edge px). Distinctness is
# judged on the same calibrated uint8 metering view the analyzers use, bounded
# to this size — milliseconds per candidate render at 256.
DISTINCT_RENDER_DIM = 256

# Minimum mean absolute rendered difference (8-bit levels) between two kept
# variants. The soft-vs-natural pair on a flat hazy frame, which fails the bar,
# measures ~1-2 levels; the weakest visibly-different pair measures ~6+.
MIN_RENDERED_DISTINCTNESS = 5.0

# Weight on the consistent colour-cast term of ``_rendered_distance``. Mean
# |diff| alone is a tone meter: golden-vs-natural on a real capture measures
# ~2.7 levels (Δluma 0.6) and was the ONLY style the floor ever dropped, while
# a blind judge picks it as best on 7 of 23 captures. At 2.0, golden's ~2.8-
# level luma-free cast clears the 5.0 floor with margin, while the term stays
# exactly zero for luma-only moves and for scattered zero-mean noise.
CAST_DISTINCTNESS_WEIGHT = 2.0


def _params_distance(a: CorrectionParams, b: CorrectionParams) -> float:
    """Distance between two styled corrections over the fields a style can move.

    Weighted L1: each |Δ| is normalized by that field's full styling authority
    (the rails ``styles.py`` clamps to), so a full-authority move on any field
    scores 1.0 and the fields are comparable; normalized fields weigh equally.
    A perceptual proxy for "these two variants will look different", not
    colorimetry. Lazy imports: ``styles`` imports ``scene``, which imports
    this module.
    """
    from ..color_correction import BLACK_POINT_MAX, BLACK_POINT_MIN, GAMMA_MAX
    from .styles import (
        GAMMA_MIN,
        HIGHLIGHT_PULL_MAX,
        SHADOW_LIFT_MAX,
        STYLE_CONTRAST_CAP,
        STYLE_SATURATION_CAP,
        STYLE_WB_CAP,
    )

    wb_span = 2.0 * STYLE_WB_CAP  # each WB field ranges ±STYLE_WB_CAP
    spans = {
        "saturation_mult": STYLE_SATURATION_CAP,  # mono's 0.0 floor -> the cap IS the span
        "contrast_curve": STYLE_CONTRAST_CAP,
        "white_balance_r": wb_span,
        "white_balance_b": wb_span,
        "white_balance_r_high": wb_span,
        "white_balance_b_high": wb_span,
        "shadow_lift": SHADOW_LIFT_MAX,
        "highlight_pull": HIGHLIGHT_PULL_MAX,
        "gamma": GAMMA_MAX - GAMMA_MIN,
        "black_point": BLACK_POINT_MAX - BLACK_POINT_MIN,  # full signed authority
    }
    return sum(abs(getattr(a, f) - getattr(b, f)) / span for f, span in spans.items())


def _rendered_distance(
    view: np.ndarray,
    a: CorrectionParams,
    b: CorrectionParams,
    _render_cache: dict[tuple, np.ndarray] | None = None,
) -> float:
    """Rendered pixel distance (8-bit levels) between two grades of ``view``.

    Two terms. Mean |diff| meters TONE, but judges a cast-type style (golden:
    opposite-signed channel shifts, near-zero luma move) far below a tone-type
    one of equal visibility — so ``CAST_DISTINCTNESS_WEIGHT`` times the
    magnitude of the consistent colour cast is added: the per-channel MEAN
    signed difference with its shared (luma) component subtracted. A
    directional cast survives the signed mean at full size; scattered noise
    cancels in it; a plain tone lift is removed with the shared component, so
    it is never counted twice. Params distance is only a proxy — a pastel move
    (gamma lift, fade, halved contrast, desat) can render invisibly on a frame
    that is already flat and hazy — so variant distinctness is measured where
    it matters: rendered pixels, on the cheap bounded metering view. ``_grade``
    is pure, so the result is deterministic. ``_render_cache`` (params-keyed)
    lets one selection walk render each candidate once, not once per
    comparison.
    """

    def rendered(params: CorrectionParams) -> np.ndarray:
        if _render_cache is None:
            return _grade(view, params)
        key = astuple(params)
        if key not in _render_cache:
            _render_cache[key] = _grade(view, params)
        return _render_cache[key]

    diff = rendered(a).astype(np.int16) - rendered(b).astype(np.int16)
    tone = float(np.mean(np.abs(diff)))
    channel_shift = diff.mean(axis=(0, 1))
    cast = channel_shift - channel_shift.mean()
    return tone + CAST_DISTINCTNESS_WEIGHT * float(np.linalg.norm(cast))


def _distinct_styled(
    scene,
    base: CorrectionParams,
    rgb: np.ndarray,
    strength: float = STYLE_STRENGTH_DEFAULT,
    subject=None,
) -> list[tuple[str, CorrectionParams]]:
    """Auto-select ``VARIANT_COUNT`` styles that actually RENDER distinct.

    ``select_styles`` scores the SceneProfile alone and cannot see that a
    picked style collapsed onto an already-picked variant, so this walks the
    FULL ranking in order (``natural`` always kept first) under a two-tier
    keep test: a candidate within ``VARIANT_DISTINCTNESS_EPSILON`` params
    distance of a kept variant is dropped WITHOUT rendering (params-identical
    implies pixel-identical), and a params-distant candidate is kept only if
    its ``_rendered_distance`` from EVERY kept variant reaches
    ``MIN_RENDERED_DISTINCTNESS`` on the bounded metering view. Each drop
    warns loudly naming its substitute (rendered drops include the measured
    distance). Exhausting the ranking short of the target falls back to the
    original top-ranked list — never fewer variants. ``subject``
    (optional) is handed straight to ``select_styles``, which is where the
    subject routing decision lives — see its docstring. Deterministic: pure
    functions of (scene, base, rgb, subject) throughout.
    """
    from .styles import STYLES, apply_style, select_styles

    view = _bounded_resize(_analysis_rgb(rgb), DISTINCT_RENDER_DIM)
    render_cache: dict[tuple, np.ndarray] = {}
    ranking = select_styles(scene, count=len(STYLES), subject=subject)
    kept: list[tuple[str, CorrectionParams]] = []
    # (dropped style, kept variant it collapsed onto, rendered distance | None)
    dropped: list[tuple[str, str, float | None]] = []
    for name in ranking:
        if len(kept) == VARIANT_COUNT:
            break
        params = apply_style(name, base, scene, strength)
        onto: str | None = None
        measured: float | None = None
        for kept_name, kept_params in kept:
            if _params_distance(params, kept_params) < VARIANT_DISTINCTNESS_EPSILON:
                onto = kept_name  # params-identical implies pixel-identical: skip the render
                break
            distance = _rendered_distance(view, params, kept_params, _render_cache=render_cache)
            if distance < MIN_RENDERED_DISTINCTNESS:
                onto, measured = kept_name, distance
                break
        if onto is not None:
            dropped.append((name, onto, measured))
            continue
        for drop_name, drop_onto, drop_measured in dropped:
            if drop_measured is None:
                logger.warning(
                    "Dropped style %r: railed params collapsed onto %r (distance < %s); "
                    "substituted next-ranked %r",
                    drop_name,
                    drop_onto,
                    VARIANT_DISTINCTNESS_EPSILON,
                    name,
                )
            else:
                logger.warning(
                    "Dropped style %r: rendered %.2f mean 8-bit levels from %r (minimum %s); "
                    "substituted next-ranked %r",
                    drop_name,
                    drop_measured,
                    drop_onto,
                    MIN_RENDERED_DISTINCTNESS,
                    name,
                )
        dropped.clear()
        kept.append((name, params))
    if len(kept) < VARIANT_COUNT:
        top = ranking[:VARIANT_COUNT]
        logger.warning(
            "Distinctness backfill exhausted at %d variant(s); falling back to the "
            "top-ranked %s despite near-identical params",
            len(kept),
            top,
        )
        return [(n, apply_style(n, base, scene, strength)) for n in top]
    return kept


def plan_variants(
    source,
    styles=None,
    strength: float = STYLE_STRENGTH_DEFAULT,
    luts=None,
    # ONE shared default instance, safe because ``DevelopOptions`` is frozen —
    # a mutable default here would leak one caller's knobs into the next run.
    options: DevelopOptions = DevelopOptions(),
    look_variants=None,
    look_styles=None,
    grade: "tuple[str, CorrectionParams] | None" = None,
    subject=None,
) -> VariantPlan:
    """Meter one photo for its style variants: decode once, profile the scene,
    compute the photo-mode base correction, and style it per variant. An
    explicit ``styles`` list is operator authority — rendered verbatim, never
    filtered (unknown name → ``ValueError``); ``None`` auto-selects via
    ``select_styles`` (``natural`` always anchors) under the distinctness
    guard above. ``strength`` is the style lever — 1.0 is the shipped bake,
    higher amplifies every style's delta over the measured base (rails scale
    with it, up to renderer authority). ``luts`` is a pre-resolved
    ``[(name, Cube), ...]`` list of film-emulation looks ADDED to the styled
    variants (the style lever never scales them). ``lut_strength`` is the LUT
    opacity in [0, 1] — the separate lever that blends every LUT look toward the
    natural grade (1.0 = full-strength LUT, 0.0 = no LUT). ``lut_over`` ("" = OFF,
    the byte-identical default) names a ``STYLES`` key to grade with BEFORE the
    cube, so a film look can ride e.g. moody's matte instead of the neutral base:
    its params are computed exactly as a styled variant's — ``apply_style(name,
    base, scene, strength)``, so the style lever DOES scale it — and carried on
    ``VariantPlan.lut_params``, with each composed look renamed
    ``<lut>_over_<style>`` so the two surfaces can never overwrite one another on
    disk. It composes a STYLE and nothing else: the operator colour fields below
    still never ride the LUT surface. ``lut_auto`` (0 = OFF, the byte-identical manual
    path) turns ``luts`` from the looks to render into the CANDIDATE SET
    ``pycube_lut.select_luts`` scores against the SceneProfile — which is why the pick happens
    here, not in the handler — keeping the best N, each renamed ``<lut>_auto``.

    ``look_variants`` is a pre-resolved ``[(name, DevelopOptions), ...]`` list of
    NAMED LOOKS, each rendered as its own panel and
    renamed ``look_<name>`` — the ``--look-variant`` surface, so two candidate
    looks compare on one contact sheet, which ``--look`` (a defaults layer, exactly
    one per run) cannot express. ``None``/empty renders no panel, byte-identically.
    ``look_styles`` is the matching ``{look name: its own base ``styles``}`` map —
    the key ``look_variant_options`` cannot carry, because a base style is no
    ``DevelopOptions`` field — and it is APPLIED here, exactly as a styled variant
    is (``apply_style(name, base, scene, strength)``, so the style lever scales it),
    onto ``VariantPlan.look_bases``. A look absent from the map keeps grading from
    the natural ``base``, byte-identically. Without it a panel rendered a look's
    colour knobs over the natural correction while ``--look NAME`` rendered the
    same knobs over the style that look names: two pictures under one name.

    Every operator knob named here is a FIELD of ``options`` (``DevelopOptions``),
    the one record they travel in; ``luts`` stays a parameter of its own because
    it is resolved data, not a knob. ``hue_shift`` / ``hue_sat`` / ``hue_lum`` are
    the operator per-hue HSL move —
    each an empty tuple (OFF) or six per-anchor-hue values. When any is non-empty
    it is layered onto EVERY styled variant's params (``natural`` included) AFTER
    ``apply_style``, so ``_grade``'s in-chain ``_apply_hsl`` renders it; the
    all-empty default leaves the params untouched, so the neutral path is
    bit-identical. It is deliberately NOT applied to ``base`` — the LUT looks ride
    the clean base grade and stay faithful film emulations, not operator-tinted.
    ``splittone_shadow`` / ``splittone_midtone`` / ``splittone_highlight`` (each
    empty=OFF or a signed length-3 (r,g,b) tint, one per tonal range) are the
    operator split-tone move, layered the same way
    onto every styled variant's params but never ``base``, so ``_grade``'s in-chain
    ``_apply_splittone`` renders it (empty ⇒ bit-identical no-op).
    ``curve_l`` is the operator MASTER tone curve and ``curve_r`` / ``curve_g`` /
    ``curve_b`` the cross-process per-channel ones (each empty = OFF); they layer
    onto every styled variant the same way and render via ``_grade``'s in-chain
    ``_apply_luma_curve`` (first, so the channel curves ride ON TOP of the master
    shape) and ``_apply_channel_curves``.
    ``vibrance`` (0.0 = OFF) is the operator low-sat-weighted chroma boost,
    ``replace``d onto every styled variant's params but never ``base`` and
    rendered by ``_grade``'s in-chain ``_apply_vibrance``. ``mono_mix`` (empty =
    OFF, else three r,g,b luma weights) and ``duotone`` (empty = OFF, else the two
    ends of a luma ramp, dialled back toward plain monochrome by
    ``duotone_strength``) are the operator B&W move, layered the same way onto
    every styled variant but never ``base``, rendered by ``_apply_monomix``.
    ``reference_lab`` (empty = OFF, else a hero frame's six Lab numbers from
    ``reference_lab_from_file``) is the operator reference-image colour match,
    layered the same way onto every styled variant but never ``base`` — a film
    emulation must stay the film's look, not the operator's hero frame — and
    rendered LAST in ``_grade`` by ``_apply_reference_match``. ``sharpen`` (1.0 =
    unchanged) SCALES the measured unsharp amount on every rendered surface.
    ``subject`` (optional) is a ``subject.SubjectReading``, forwarded verbatim
    to ``_distinct_styled``'s ``select_styles`` call for the AUTO-select path only
    (``None`` — the default — is byte-identical to before this parameter existed;
    an explicit ``styles`` list or ``--from-grade`` never consults it, same as
    ``select_styles`` itself is never consulted on those paths).
    Lazy imports: ``scene`` imports this module, and ``exif_gps`` imports ``scene``."""
    from .exif_gps import extract_exif_capture_metadata
    from .scene import compute_scene_profile
    from .styles import apply_style

    decoded = source if isinstance(source, DecodedPhoto) else decode(source)
    # Capture EXIF off the source file: ISO informs the noise reading, the hour
    # the look. Unreadable or EXIF-less sources yield an all-None record.
    scene = compute_scene_profile(decoded.rgb, extract_exif_capture_metadata(decoded.path))
    base = compute_correction(_frame_stats(decoded.rgb), mode=GRADE_MODE_PHOTO)
    if grade is not None:
        # ``--from-grade``: a RECORDED ``(variant, params)`` stands in for the
        # measured style, so the frame renders the approved numbers rather than
        # what this photo meters — which is what lets one photo's grade transfer
        # to its siblings. ``base`` is still measured: it is the denoise budget.
        styled = [grade]
    elif styles:
        styled = [(n, apply_style(n, base, scene, strength)) for n in styles]
    else:
        styled = _distinct_styled(scene, base, decoded.rgb, strength, subject=subject)
    o = options  # the knobs, read-only, from here down
    if o.hue_shift or o.hue_sat or o.hue_lum or o.curve_l or o.curve_r or o.curve_g or o.curve_b:
        styled = [
            (
                n,
                replace(
                    p,
                    hue_shift=o.hue_shift,
                    hue_sat=o.hue_sat,
                    hue_lum=o.hue_lum,
                    curve_l=o.curve_l,
                    curve_r=o.curve_r,
                    curve_g=o.curve_g,
                    curve_b=o.curve_b,
                ),
            )
            for n, p in styled
        ]
    if o.splittone_shadow or o.splittone_midtone or o.splittone_highlight:
        styled = [
            (
                n,
                replace(
                    p,
                    splittone_shadow=o.splittone_shadow,
                    splittone_midtone=o.splittone_midtone,
                    splittone_highlight=o.splittone_highlight,
                ),
            )
            for n, p in styled
        ]
    if any(getattr(o, f) for f in _WB_ADDITIVE) or o.white_balance_kelvin or o.white_balance_tint:
        styled = [(n, _wb_shifted(p, o)) for n, p in styled]
    if o.exposure_ev:
        styled = [(n, replace(p, exposure_ev=o.exposure_ev)) for n, p in styled]
    if o.shadow_recovery or o.highlight_recovery:
        styled = [
            (
                n,
                replace(
                    p, shadow_recovery=o.shadow_recovery, highlight_recovery=o.highlight_recovery
                ),
            )
            for n, p in styled
        ]
    if o.vibrance:
        styled = [(n, replace(p, vibrance=o.vibrance)) for n, p in styled]
    if o.mono_mix or o.duotone:
        styled = [
            (
                n,
                replace(
                    p,
                    mono_mix=o.mono_mix,
                    duotone=o.duotone,
                    duotone_strength=o.duotone_strength,
                    duotone_mid=o.duotone_mid,
                ),
            )
            for n, p in styled
        ]
    if o.reference_lab:
        styled = [(n, replace(p, reference_lab=o.reference_lab)) for n, p in styled]
    lut_params = None
    lut_looks = list(luts or [])
    lut_auto_picks: list[tuple[str, float]] = []
    if o.lut_auto:
        from pycube_lut import LUT_PROBE_DIM, select_luts  # needs the scene: it happens HERE

        # Scored on the frame's OWN graded pixels — see ``pycube_lut.characterize_cube``.
        view = _grade(_bounded_resize(_analysis_rgb(decoded.rgb), LUT_PROBE_DIM), base)
        picks = select_luts(lut_looks, scene, view, o.lut_auto)
        lut_looks = [(f"{n}_auto", cube) for n, cube, _ in picks]
        lut_auto_picks = [(f"{n}_auto", score) for n, _, score in picks]
    if o.lut_over:
        # A composed look is a DIFFERENT surface from the base-graded one, so it
        # gets a different name — two develops of one photo can't collide on disk.
        lut_params = apply_style(o.lut_over, base, scene, strength)
        lut_looks = [(f"{n}_over_{o.lut_over}", cube) for n, cube in lut_looks]
        lut_auto_picks = [(f"{n}_over_{o.lut_over}", s) for n, s in lut_auto_picks]
    return VariantPlan(
        decoded,
        scene,
        base,
        styled,
        lut_looks=lut_looks,
        look_variants=[(f"{LOOK_VARIANT_PREFIX}{n}", lo) for n, lo in (look_variants or [])],
        look_bases={
            f"{LOOK_VARIANT_PREFIX}{n}": apply_style(s, base, scene, strength)
            for n, s in (look_styles or {}).items()
            if s
        },
        lut_strength=o.lut_strength,
        lut_params=lut_params,
        lut_auto_picks=lut_auto_picks,
        vignette=o.vignette,
        dehaze=o.dehaze,
        sky_darken=o.sky_darken,
        sky_saturate=o.sky_saturate,
        graduated_nd=o.graduated_nd,
        graduated_nd_angle=o.graduated_nd_angle,
        glow=o.glow,
        clarity=o.clarity,
        grain=o.grain,
        grain_size=o.grain_size,
        halation=o.halation,
        halation_tint=o.halation_tint,
        light_leak=o.light_leak,
        chromatic_ab=o.chromatic_ab,
        holo_sheen=o.holo_sheen,
        scanline=o.scanline,
        glitch=o.glitch,
        sharpen=o.sharpen,
    )


def develop_variants(
    source,
    styles=None,
    strength: float = STYLE_STRENGTH_DEFAULT,
    luts=None,
    # Frozen record, so this ONE shared default instance is safe to bind here.
    options: DevelopOptions = DevelopOptions(),
    depth: int = 8,
    look_variants=None,
    look_styles=None,
    grade: "tuple[str, CorrectionParams] | None" = None,
    subject=None,
):
    """Render one photo's style variants from a single decode. Denoise also
    runs ONCE, on the source under the base correction's budget; every variant
    then grades the shared full-depth array through the same ``_grade`` chain —
    which copies, never mutates it — and sharpens its own graded output.
    ``luts`` looks render LAST: each applies its ``.cube`` over the natural
    (base) grade — or over ``lut_over``'s styled grade when set, the ONE extra
    grade that composition costs — at ``lut_strength`` opacity, then sharpens. The operator per-hue
    HSL move (``hue_shift`` / ``hue_sat`` / ``hue_lum``, empty = OFF) and the
    master and cross-process curves (``curve_l``, then ``curve_r`` / ``curve_g`` /
    ``curve_b``, empty = OFF) and the low-sat-weighted ``vibrance`` boost (0.0 = OFF) ride each
    styled variant's params but not the LUT-look base — see ``plan_variants``.
    All of those knobs arrive as ONE ``DevelopOptions`` record and are handed to
    ``plan_variants`` verbatim. ``look_variants`` looks render LAST of all: each
    named look grades through its OWN knob record — colour chain AND its
    own spatial amounts, so two differing only in grain still compare — over the
    base its own ``styles`` names (``look_styles`` → ``plan.look_bases``), or the
    natural base grade when it names none; one whose
    ``look_strength`` is under 1.0 is then lerped that far toward the run's base
    render (``_dial_to_base``), which is rendered lazily, once. Yields
    ``(name, DevelopedPhoto)`` in plan order — styled first, then LUT looks, then
    look variants — one at a time. ``depth`` is the RENDER depth,
    not a container: 8 (the default) is the unchanged uint8 chain; 16 quantizes
    the grade to uint16 and every stage after it runs in that range too.
    ``grade`` is the ``--from-grade`` replay, handed to ``plan_variants``.
    ``subject`` (optional) is handed to ``plan_variants`` verbatim — see its
    docstring."""
    plan = plan_variants(
        source, styles, strength, luts, options, look_variants, look_styles, grade, subject=subject
    )
    # Run-level geometry, applied ONCE before the shared denoise/measure — every
    # variant grades from the same lens-corrected, straightened and cropped
    # frame, never its own copy, so two variants of one photo stay directly
    # comparable in shape. ``options.lens_correct`` False (the default) never
    # even calls ``lens_correct.correct``, so the default path is unchanged.
    from . import lens_correct

    if options.lens_correct:
        lens_result = lens_correct.correct(plan.decoded.rgb, plan.decoded.path)
    else:
        lens_result = lens_correct.LensCorrectionResult(
            rgb=plan.decoded.rgb, applied=False, reason=""
        )
    frame = _crop(_straighten(lens_result.rgb, options.straighten_angle), options.crop_rect)
    denoised, denoise_h = _denoise(frame, plan.base)
    for name, params in plan.styled:
        layered = _layered(_grade(denoised, params, depth), options.layers, depth)
        graded, sharpen_amount = _spatial_chain(layered, plan)
        # fmt: off
        yield name, DevelopedPhoto(
            rgb=graded, path=plan.decoded.path, params=params, cluster_index=0,
            denoise_applied=denoise_h > 0.0, denoise_h=denoise_h,
            sharpen_applied=sharpen_amount > 0.0, sharpen_amount=sharpen_amount,
            lens_correct_applied=lens_result.applied, lens_correct_reason=lens_result.reason,
            decoder=plan.decoded.decoder,
        )
        # fmt: on
    if plan.lut_looks:
        from pycube_lut import apply_cube

        # The grade a film LUT rides on: the natural base, or --lut-over's style.
        under = plan.lut_params if plan.lut_params is not None else plan.base
        base_graded = _grade(denoised, under, depth)
        for name, cube in plan.lut_looks:
            veiled = apply_cube(base_graded, cube, plan.lut_strength)
            graded, sharpen_amount = _spatial_chain(veiled, plan)
            # fmt: off
            yield name, DevelopedPhoto(
                rgb=graded, path=plan.decoded.path, params=under, cluster_index=0,
                denoise_applied=denoise_h > 0.0, denoise_h=denoise_h,
                sharpen_applied=sharpen_amount > 0.0, sharpen_amount=sharpen_amount,
                lens_correct_applied=lens_result.applied, lens_correct_reason=lens_result.reason,
                decoder=plan.decoded.decoder,
            )
            # fmt: on
    # The run's own base render, for any look asking for less than the whole of
    # itself. Rendered LAZILY and at most once per photo: at full strength (the
    # default) nothing below reaches it, so the shipped path costs no extra pass.
    base_render: np.ndarray | None = None
    for name, look in plan.look_variants:
        # This look's OWN base: the style it names, else the natural correction.
        params = _look_params(plan.look_bases.get(name, plan.base), look)
        layered = _layered(_grade(denoised, params, depth), look.layers, depth)
        graded, sharpen_amount = _spatial_chain(layered, look)
        if look.look_strength < 1.0:
            if base_render is None:
                natural = _layered(_grade(denoised, plan.base, depth), options.layers, depth)
                base_render = _spatial_chain(natural, plan)[0]
            graded = _dial_to_base(base_render, graded, look.look_strength)
        # fmt: off
        yield name, DevelopedPhoto(
            rgb=graded, path=plan.decoded.path, params=params, cluster_index=0,
            denoise_applied=denoise_h > 0.0, denoise_h=denoise_h,
            sharpen_applied=sharpen_amount > 0.0, sharpen_amount=sharpen_amount,
            lens_correct_applied=lens_result.applied, lens_correct_reason=lens_result.reason,
            decoder=plan.decoded.decoder,
        )
        # fmt: on


# Measure-pass decode bound (long-edge pixels). The measure stats feed
# clustering only, so the measure pass decodes RAW at half size (LibRaw bins
# each 2x2 Bayer block instead of running the full demosaic — a fraction of
# the demosaic wall clock at 20MP) and area-averages any larger frame down
# under this bound. Averaging, not striding: ``_frame_stats`` averages again
# to the calibrated ANALYSIS_MAX_DIM, and a strided intermediate would
# re-inject the per-pixel noise that scale exists to cancel — skewing the
# bounded measure's stats away from a full decode's.
MEASURE_MAX_DIM = 1536


def _measure_rgb(path: Path) -> np.ndarray:
    """Bounded decode for the measure pass — never the full-resolution demosaic.

    A RAW whose cheap half-size decode fails (an HE NEF stream) falls back to
    the full ``decode`` path, which owns the DNG-convert/preview fallbacks and
    the fail-closed error naming the file.
    """
    rgb: np.ndarray | None = None
    if path.suffix.lower() in RAW_EXTENSIONS:
        import rawpy

        try:
            with rawpy.imread(str(path)) as raw:
                rgb = raw.postprocess(output_bps=8, use_camera_wb=True, half_size=True)
        except Exception:
            rgb = None
    if rgb is None:
        rgb = decode(path).rgb

    return _bounded_resize(rgb, MEASURE_MAX_DIM)


def reference_lab_from_file(path) -> tuple[float, ...]:
    """The six Lab numbers a ``--reference`` hero frame carries into the grade.

    Measured through the same bounded ``_measure_rgb`` decode the clustering
    measure pass uses — a mean and a standard deviation do not need the full
    demosaic — and taken ONCE per run at the CLI door, so an unreadable
    reference raises ``PhotoDecodeError`` naming the file before a single source
    photo is decoded, and no per-photo render ever re-reads it.
    """
    return lab_mean_std(_measure_rgb(Path(path)))


def compute_batch_corrections(
    paths, variance_threshold: float | None = None
) -> list[tuple[CorrectionParams, int]]:
    """One shared ``(params, cluster_index)`` per input path, in input order.

    ``variance_threshold=None`` resolves ``photos.cluster_variance_threshold``,
    falling back to ``SPAN_SIMILARITY_THRESHOLD`` — still one knob, now
    config-overridable; an explicit argument always wins.

    The measure pass pays only the bounded ``_measure_rgb`` decode and returns
    parameters, never pixels, so a caller can grade one file at a time instead
    of holding a batch of graded full-resolution frames in memory. Coverage is
    total by contract: an index the clustering fails to assign raises loudly,
    matching decode's fail-closed behavior — never a silent drop.
    """
    if variance_threshold is None:
        from ...config import get_config

        configured = getattr(get_config().photos, "cluster_variance_threshold", None)
        variance_threshold = configured if configured is not None else SPAN_SIMILARITY_THRESHOLD

    paths = [Path(p) for p in paths]
    file_stats = [(path, _frame_stats(_measure_rgb(path))) for path in paths]

    assigned: list[tuple[CorrectionParams, int] | None] = [None] * len(paths)
    for cluster_index, members in enumerate(
        cluster_files_by_lighting(file_stats, variance_threshold)
    ):
        params = compute_cluster_correction(
            [file_stats[i][1] for i in members], mode=GRADE_MODE_PHOTO
        )
        for i in members:
            assigned[i] = (params, cluster_index)

    if missing := [str(paths[i]) for i, entry in enumerate(assigned) if entry is None]:
        raise RuntimeError(
            f"lighting clustering returned no cluster for: {', '.join(missing)} — "
            "refusing to silently drop a photo from the batch"
        )
    return assigned


def develop_batch(paths, variance_threshold: float | None = None) -> list[DevelopedPhoto]:
    """Develop a set of photos with one shared correction per lighting cluster.

    Grading each still alone makes one scene drift shot to shot; clustering
    pins a single correction across every frame shot in the same light. Photos
    are decoded twice deliberately — holding a whole set of full-resolution
    RAW frames in memory is how a large batch OOMs — but only the grade pass
    pays the full demosaic; the measure pass inside
    ``compute_batch_corrections`` is bounded, since its stats feed clustering only.
    """
    paths = [Path(p) for p in paths]
    corrections = compute_batch_corrections(paths, variance_threshold)
    return [
        develop(path, params, cluster_index)
        for path, (params, cluster_index) in zip(paths, corrections, strict=True)
    ]


# Gallery masters are the archival source the social resizer re-encodes from:
# full-chroma 4:4:4 stops PIL's implicit 4:2:0 from halving chroma resolution
# before the second lossy encode compounds it.
GALLERY_JPEG_SUBSAMPLING = "4:4:4"


@functools.cache
def srgb_icc_profile() -> bytes:
    """Standard sRGB ICC profile bytes for tagging JPEG output explicitly.

    Deterministic by construction. littleCMS stamps the WALL CLOCK into the ICC
    header's ``dateTime`` field (bytes 24-36: year, month, day, hour, minute,
    second as big-endian uint16), so a freshly built profile differed second to
    second and no two runs of this pipeline produced a byte-equal JPEG for
    identical pixels — silently defeating any byte-level reproducibility check,
    the very property the reserved ``DEVELOP_ALGO_VERSION`` / cache-key contract
    assumes. ``dateTime`` is informational, so it is ZEROED, as reproducible-build
    toolchains do: a meaningless creation stamp is worth less than output you can
    diff. Zeroing is the load-bearing half — the pipeline runs as separate CLI
    invocations, so a per-process cache alone would still leave two processes
    disagreeing; the cache only saves the rebuild. Every other byte (primaries,
    whitepoint, transfer curve) is untouched, and the profile ID littleCMS emits
    is already zero, so no checksum goes stale.
    """
    from PIL import ImageCms

    profile = bytearray(ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())
    profile[24:36] = bytes(12)
    return bytes(profile)


def export_jpeg(rgb: np.ndarray, out_path, quality: int | None = None) -> Path:
    """Write a developed frame as an sRGB-tagged JPEG, atomically.

    EXIF (GPS included) is absent by construction rather than stripped after
    the fact — the image is built from a bare array, so there is no source
    metadata to carry. The default quality is the gallery-master knob
    (``photos.gallery_jpeg_quality``), not the social ``jpeg_quality``: the
    resizer re-encodes these masters, so exporting at the social quality pays
    the generation loss twice. Bytes land in a temp file and are renamed into
    place, so a failure mid-write never leaves a partial master behind.
    """
    from PIL import Image

    if quality is None:
        from ...config import get_config

        photos = get_config().photos
        # Minimal/older photo configs may lack the gallery knob; fall back to
        # the social quality rather than failing the export.
        quality = getattr(photos, "gallery_jpeg_quality", photos.jpeg_quality)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.name}.tmp")
    try:
        Image.fromarray(np.ascontiguousarray(rgb), mode="RGB").save(
            tmp_path,
            "JPEG",
            quality=quality,
            icc_profile=srgb_icc_profile(),
            subsampling=GALLERY_JPEG_SUBSAMPLING,
        )
        os.replace(tmp_path, out_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return out_path


# ``photos develop --format``; mirrored as a literal in PHOTO_EXPORT_FORMATS.
EXPORT_FORMATS = ("jpeg", "tiff16")


def export_tiff16(rgb: np.ndarray, out_path) -> Path:
    """Write a 16-bit developed frame as a 48-bit sRGB-tagged TIFF, atomically.

    Deflate-compressed and lossless, so it re-edits in Lightroom / Capture One
    without the generation loss a JPEG master pays. Same atomic temp-then-
    ``os.replace`` contract and the same deterministic ``srgb_icc_profile`` bytes
    ``export_jpeg`` uses: the chain's working space IS sRGB (a wide-gamut/ICC
    decode is converted before grading), so the tag names the space these numbers
    are actually in — a P3 tag would relabel them, not widen them. REFUSES a
    uint8 frame: an 8-bit render upsampled into a 16-bit container claims
    headroom the pixels do not carry, so depth is guaranteed at the door here
    rather than by a comment somewhere upstream.
    """
    import tifffile

    if rgb.dtype != np.uint16:
        raise ValueError(f"a 16-bit master needs a uint16 frame, got {rgb.dtype}")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.name}.tmp")
    try:
        tifffile.imwrite(
            tmp_path,
            np.ascontiguousarray(rgb),
            photometric="rgb",
            compression="deflate",
            iccprofile=srgb_icc_profile(),
            software=False,  # a version stamp would break byte-reproducibility
        )
        os.replace(tmp_path, out_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return out_path


# Border/frame styles for ``photos develop --border`` — a presentational matte
# applied at the CLI EXPORT step only, never fed into the grade or the contact
# sheet. ``none`` returns the frame UNCHANGED (an exact no-op), so the default
# export stays byte-identical to before borders existed. Thickness is a FRACTION
# of the long edge, so the frame looks the same at any resolution. ``filmstrip``
# widens the top/bottom bars and punches ``FILMSTRIP_HOLE_COUNT`` evenly spaced
# white sprocket holes into them. Floats/int so ``_tunable_constants`` hashes a
# retune, exactly as the contact-sheet knobs are.
BORDER_STYLES = ("none", "white", "black", "filmstrip")
BORDER_FRACTION = 0.04
FILMSTRIP_BAR_FRACTION = 0.08
FILMSTRIP_HOLE_COUNT = 8


def _add_border(rgb: np.ndarray, style: str) -> np.ndarray:
    """Pad a finished frame with a presentational matte/frame; return a NEW
    ``(H,W,3)`` uint8 array (a larger canvas with the frame pasted inside).

    Applied at EXPORT only — never the grade, never the contact-sheet panels.
    Pure and deterministic: a function of ``(rgb, style)`` alone. ``style="none"``
    returns the input UNCHANGED, so the default export is byte-identical to
    before borders existed. ``white`` / ``black`` lay an even matte of that colour
    on all four sides; ``filmstrip`` frames the shot in black with WIDER top/bottom
    bars carrying evenly spaced white sprocket holes (the sides keep the thin
    matte). Thickness is a fraction of the long edge (``BORDER_FRACTION``), so the
    frame is resolution-independent.
    """
    if style == "none":
        return rgb
    if style not in BORDER_STYLES:
        raise ValueError(f"unknown border style {style!r} (choose from {', '.join(BORDER_STYLES)})")
    height, width = rgb.shape[:2]
    white = int(_peak(rgb))  # full scale in the frame's OWN depth
    pad = max(1, round(BORDER_FRACTION * max(height, width)))
    if style in ("white", "black"):
        fill = white if style == "white" else 0
        out = np.full((height + 2 * pad, width + 2 * pad, 3), fill, dtype=rgb.dtype)
        out[pad : pad + height, pad : pad + width] = rgb
        return out

    # filmstrip: wide black top/bottom bars, thin black sides, white sprockets.
    bar = max(pad + 1, round(FILMSTRIP_BAR_FRACTION * max(height, width)))
    out = np.zeros((height + 2 * bar, width + 2 * pad, 3), dtype=rgb.dtype)
    out[bar : bar + height, pad : pad + width] = rgb
    hole_h = max(1, bar // 3)
    hole_w = max(1, width // (FILMSTRIP_HOLE_COUNT * 2))
    y_top = (bar - hole_h) // 2
    y_bot = bar + height + y_top
    for i in range(FILMSTRIP_HOLE_COUNT):
        cx = pad + round((i + 0.5) * width / FILMSTRIP_HOLE_COUNT)
        x0 = cx - hole_w // 2
        out[y_top : y_top + hole_h, x0 : x0 + hole_w] = white
        out[y_bot : y_bot + hole_h, x0 : x0 + hole_w] = white
    return out


# ── the per-frame grade record (``photos develop --grade-sidecar``) ──
# Opt-in metadata written BESIDE the export, never into it: the exact numbers
# that made this frame, so a photo the operator likes can be reproduced or
# transferred instead of only admired. Both constants are strings deliberately —
# ``_tunable_constants`` skips strings, so a metadata filename or schema bump can
# never invalidate the pixel fingerprint it has no bearing on.
GRADE_SIDECAR_SUFFIX = ".grade.json"
GRADE_SIDECAR_VERSION = "1"


def _from_sidecar(record, recorded: dict):
    """Rebuild a frozen record from the dict ``asdict`` wrote into a sidecar.

    JSON has no tuple, so every sequence field reads back as a list; the fields
    whose dataclass default is a tuple are coerced back, which makes the round
    trip EQUAL to the record that rendered the frame rather than merely similar.
    A key the dataclass does not carry raises ``TypeError`` and an out-of-range
    value ``ValueError`` — both are the door's refusal, never a mid-batch raise.
    """
    tupled = {f.name for f in fields(record) if isinstance(f.default, tuple)}
    return record(**{k: tuple(v) if k in tupled else v for k, v in recorded.items()})


def correction_params_from_sidecar(params: dict) -> CorrectionParams:
    """Rebuild the exact ``CorrectionParams`` a sidecar's ``params`` block records."""
    return _from_sidecar(CorrectionParams, params)


def develop_options_from_sidecar(options: dict) -> DevelopOptions:
    """Rebuild the exact ``DevelopOptions`` a sidecar's ``options`` block records.
    The spatial knobs come back with the colour ones, which is what makes
    ``--from-grade`` a replay of the frame the operator approved rather than a
    re-grade of its colour alone: a vignette or a grain is as much a part of that
    frame as its curves, and dropping one renders different pixels in silence."""
    return _from_sidecar(DevelopOptions, options)


def _lut_name_for(variant: str, luts, options: DevelopOptions) -> str:
    """The resolved ``.cube`` name a variant rendered, ``""`` for a styled one.

    Mirrors ``plan_variants``' LUT renames (``_auto`` under ``--lut-auto``, then
    ``_over_<style>`` under ``--lut-over``) as literals; a test pins the two.
    """
    suffix = ("_auto" if options.lut_auto else "") + (
        f"_over_{options.lut_over}" if options.lut_over else ""
    )
    return next((n for n, _ in (luts or ()) if variant == f"{n}{suffix}"), "")


def _style_kind(variant: str) -> str:
    """``"artistic"`` or ``"natural"`` -- whether ``variant`` is a deliberate
    restyling or the natural colour correction a viewer can trust as-shot.

    A base style panel is named exactly its ``STYLES`` key, so
    ``is_artistic_style`` classifies it directly. A LUT panel (``_auto`` /
    ``_over_<style>`` suffix) or a named ``LOOK_VARIANT_PREFIX`` panel is
    never natural correction by definition -- ``is_artistic_style`` only
    classifies base style names and would raise on either, so those read
    ``"artistic"`` without asking it.
    """
    from .styles import STYLES, is_artistic_style

    if variant in STYLES:
        return "artistic" if is_artistic_style(variant) else "natural"
    return "artistic"


def grade_sidecar_payload(
    developed, variant, style_strength, options, lut_name="", *, depth: int, lut_cube=None
) -> dict:
    """Everything needed to reproduce ONE developed frame, as a plain dict.

    ``params`` is the exact correction that graded THIS variant and is the
    authority on its colour chain (the operator colour knobs ride the styled
    variants only, never the LUT surface); ``options`` is the whole live
    ``DevelopOptions`` record of the run — spatial knobs included, read off the
    dataclass rather than a hand-kept list that goes stale the next time a knob
    is added — and ``measured`` the amounts the ramped stages actually chose.
    ``provenance`` READS the reserved ``DEVELOP_ALGO_VERSION`` /
    ``develop_cache_key`` pair to stamp which code and inputs produced the frame,
    which is why the render ``depth`` and this variant's ``lut_cube`` are threaded
    in: they change the pixels and so must change the stamped key. Recording that
    key is NOT wiring the cache: nothing here reads or writes a cache entry, and
    the reserved pair stays deliberately unwired. ``style_kind`` is
    ``"artistic"``/``"natural"`` per ``_style_kind`` -- what makes an
    artistic-restyling export identifiable in an artifact, not just at the
    call site that rendered it.
    """
    payload = {
        "sidecar_version": GRADE_SIDECAR_VERSION,
        "variant": variant,
        "style_kind": _style_kind(variant),
        "style_strength": float(style_strength),
        "source": str(developed.path),
        "params": asdict(developed.params),
        "options": asdict(options),
        "measured": {
            "denoise_applied": developed.denoise_applied,
            "denoise_h": developed.denoise_h,
            "sharpen_applied": developed.sharpen_applied,
            "sharpen_amount": developed.sharpen_amount,
            "lens_correct_applied": developed.lens_correct_applied,
            "lens_correct_reason": developed.lens_correct_reason,
        },
        "provenance": {
            "decoder": developed.decoder,
            "develop_algo_version": DEVELOP_ALGO_VERSION,
            "develop_cache_key": develop_cache_key(
                developed.path,
                developed.params,
                developed.decoder,
                options=options,
                depth=depth,
                lut=(lut_name, lut_cube) if lut_name and lut_cube is not None else None,
            ),
        },
    }
    if lut_name:
        payload["lut"] = {
            "name": lut_name,
            "lut_strength": options.lut_strength,
            "lut_over": options.lut_over,
        }
    # Which RECIPE drew this panel, read off the panel NAME rather than threaded
    # from the run — the two cannot drift. A sheet audits back to the look.
    if variant.startswith(LOOK_VARIANT_PREFIX):
        payload["look"] = {"name": variant.removeprefix(LOOK_VARIANT_PREFIX)}
    return payload


def write_grade_sidecar(
    developed,
    jpeg_path,
    variant: str,
    style_strength: float,
    options: DevelopOptions,
    luts=(),
    *,
    depth: int,
) -> Path | None:
    """Write ``<stem>.grade.json`` beside a developed export, atomically.

    Same temp-file-then-``os.replace`` pattern ``export_jpeg`` uses, so a crash
    mid-write can never leave a half-written record next to a good JPEG. Keys are
    sorted and floats keep Python's shortest round-trip repr, so two develops of
    one frame produce byte-identical JSON — the JPEG is already byte-reproducible
    and a wandering key order would undo that property for the pair. Nothing here
    can cost the photo: any failure is logged loudly and returns ``None``, never
    raises. The export is the product; the sidecar is metadata.
    """
    out_path = Path(jpeg_path).with_suffix(GRADE_SIDECAR_SUFFIX)
    tmp_path = out_path.with_name(f".{out_path.name}.tmp")
    try:
        lut_name = _lut_name_for(variant, luts, options)
        payload = grade_sidecar_payload(
            developed,
            variant,
            style_strength,
            options,
            lut_name,
            depth=depth,
            # The cube this variant rendered through, taken from the SAME resolved
            # ``luts`` the render used — never re-read from disk, where an edited
            # file would fingerprint a table these pixels never saw.
            lut_cube=dict(luts or ()).get(lut_name),
        )
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp_path, out_path)
    except Exception as exc:  # metadata must never lose an export
        logger.error("%s: grade sidecar not written (%s); the JPEG stands", out_path, exc)
        tmp_path.unlink(missing_ok=True)
        return None
    return out_path


# Contact-sheet panel bound (width px). The sheet exists for eyeballing style
# deltas side by side, so each panel is area-downscaled under this width
# rather than stitching N full-resolution frames into an unopenable strip.
CONTACT_PANEL_WIDTH = 1200

# Height (px) of the dark strip above the panels carrying the style labels.
CONTACT_LABEL_HEIGHT = 24


def compose_contact_sheet(variants, cols: int | None = None) -> np.ndarray:
    """Compose labeled style panels into a labeled RGB grid.

    ``variants`` is ``[(style_name, rgb_array), ...]`` in plan order — the
    in-memory frames ``develop_variants`` yields, never re-read from the
    exported JPEGs. Each panel is bounded to ``CONTACT_PANEL_WIDTH`` via PIL's
    area (BOX) filter, and its style name is drawn on the dark strip above it
    in PIL's built-in default font — no font-file dependency. ``cols`` wraps the
    panels into rows of that many; each row is sized to its tallest panel and
    carries its own label strip. ``cols=None`` (the default) keeps every panel
    in one unbroken row — bit-identical to the pre-wrap layout. Returns a bare
    array for ``export_jpeg``, which supplies the atomic write and sRGB tag.
    """
    from PIL import Image, ImageDraw

    if not variants:
        raise ValueError("contact sheet requires at least one rendered variant")

    panels = []
    for name, rgb in variants:
        img = Image.fromarray(np.ascontiguousarray(rgb), mode="RGB")
        if img.width > CONTACT_PANEL_WIDTH:
            height = max(1, round(img.height * CONTACT_PANEL_WIDTH / img.width))
            img = img.resize((CONTACT_PANEL_WIDTH, height), Image.Resampling.BOX)
        panels.append((name, img))

    per_row = len(panels) if cols is None else max(1, cols)
    rows = [panels[i : i + per_row] for i in range(0, len(panels), per_row)]
    row_heights = [max(img.height for _, img in row) for row in rows]
    sheet = Image.new(
        "RGB",
        (
            max(sum(img.width for _, img in row) for row in rows),
            sum(CONTACT_LABEL_HEIGHT + h for h in row_heights),
        ),
        color=(16, 16, 16),
    )
    draw = ImageDraw.Draw(sheet)
    y = 0
    for row, row_height in zip(rows, row_heights):
        x = 0
        for name, img in row:
            draw.text((x + 8, y + 6), name, fill=(255, 255, 255))
            sheet.paste(img, (x, y + CONTACT_LABEL_HEIGHT))
            x += img.width
        y += CONTACT_LABEL_HEIGHT + row_height
    return np.asarray(sheet)


def _stable_repr(value) -> str | None:
    """Deterministic, address-free serialization for a module constant.

    Returns ``None`` for anything with no such form (functions, the decoder
    dispatch table): their ``repr`` embeds a memory address, which would make
    the fingerprint differ between two processes running identical code.
    """
    if isinstance(value, np.ndarray):
        return f"ndarray:{value.dtype}:{value.shape}:{value.tobytes().hex()}"
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return repr(value)
    if isinstance(value, (tuple, list)):
        parts = [_stable_repr(item) for item in value]
        return None if None in parts else f"{type(value).__name__}[{','.join(parts)}]"
    if isinstance(value, (set, frozenset)):
        parts = [_stable_repr(item) for item in value]
        return None if None in parts else f"set[{','.join(sorted(parts))}]"
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda kv: repr(kv[0]))
        pairs = [(_stable_repr(k), _stable_repr(v)) for k, v in items]
        if any(k is None or v is None for k, v in pairs):
            return None
        return "dict[" + ",".join(f"{k}={v}" for k, v in pairs) + "]"
    return None


def _tunable_constants(namespace=None) -> list[tuple[str, str]]:
    """Every ALL_CAPS module constant that can steer output pixels, read live.

    ``namespace`` defaults to this module's globals; the fusion stage passes
    ``bracket``'s, whose thresholds steer pixels from outside this file.

    Hashing function *source* alone would not notice a retune, since these are
    module constants. They are collected by naming convention across every
    serializable type — float AND int, tuple, dict, ndarray — rather than
    listed by hand or filtered to floats, so a later-added knob, an int-valued
    retune, or an edit to the noise kernel is covered for free. A hand-kept or
    type-narrowed list is exactly how a retune ends up silently replaying stale
    cache entries. Top-level strings are excluded: they are labels and
    messages, and the derived ``DEVELOP_ALGO_VERSION`` itself must never feed
    back into its own hash.
    """
    return sorted(
        (name, serialized)
        for name, value in (globals() if namespace is None else namespace).items()
        if name.isupper()
        and not isinstance(value, str)
        and (serialized := _stable_repr(value)) is not None
    )


# The decode functions are hashed alongside the grade/spatial ones: decode
# output IS the input to any cached pixels, so a demosaic, fallback, or
# orientation change must invalidate exactly like a grade change. The
# ``_apply_*`` bodies are hashed HERE, not in GRADE_ALGO_VERSION, which
# deliberately covers only the correction *producers* — the video color cache
# stores params and rebuilds the translation fresh, but this cache stores
# pixels, so the appliers matter.
_HASHED_SOURCES = (
    decode,
    _decode_pil,
    _decode_heif,
    _decode_raw,
    _convert_he_to_dng,
    _decode_he_via_dng,
    _decode_raw_preview,
    _sensor_bit_depth,
    _apply_curves,
    _apply_luma_curve,
    _apply_channel_curves,
    _apply_colorbalance,
    _apply_splittone,
    _apply_saturation,
    _apply_vibrance,
    _apply_hsl,
    _apply_monomix,
    _apply_gamma,
    _apply_reference_match,
    _analysis_rgb,
    _peak,
    _requantize,
    _noise_sigma,
    _detail_score,
    _bounded_view,
    _denoise,
    _sharpen_amount,
    _sharpen,
    _apply_gain_mask,
    _radial_vignette_mask,
    _vignette,
    _sky_mask,
    _sky_darken,
    _sky_saturate,
    _graduated_nd,
    _dehaze_airlight,
    _dehaze_transmission,
    _dehaze,
    _clarity_gain,
    _clarity,
    _glow_bloom,
    _glow,
    _grain_weight,
    _grain_field,
    _grain,
    _halation_seed,
    _halation_glow,
    _halation,
    _light_leak_ramp,
    _light_leak,
    _scanline,
    _radial_scale_maps,
    _chromatic_ab,
    _holo_tone,
    _holo_sheen,
    _glitch_row_shift,
    _glitch,
    _grade,
    # The composers, not just the stages they call: the chain ORDER and the
    # masked-layer compositing move pixels as surely as any stage body does.
    _spatial_chain,
    _look_params,
    _wb_shifted,
    _layered,
    look_variant_options,
    fuse_bracket,
    develop,
    compute_batch_corrections,
    develop_batch,
)


def _develop_algo_inputs() -> str:
    """Everything whose change must invalidate a developed-photo cache entry.

    ``bracket``'s rule and thresholds are hashed alongside this module's own: they sit outside
    the sweep of these globals, yet with ``fuse_bracket`` wired they decide WHICH frames merge
    — so a key without them replays a develop cached from frames it no longer groups. ``masks``
    and ``looks`` sit outside it the same way and are wired TODAY: a masked look's pixels are
    the mask shape (``masks``' functions and its window/geometry constants) applied to a recipe
    (``BUILTIN_LOOKS``), and every developed photo stamps this fingerprint into its
    ``.grade.json``. ``masks`` is hashed as a WHOLE MODULE, not function by function, because
    the luma windows are lambdas inside ``MASKS`` that no namespace sweep can reach — and a
    mask added later is then covered with nothing to remember. Imported locally: this runs
    while ``develop`` is still importing, and ``masks`` imports back from it inside its own
    functions to dodge that cycle.

    The eight astro pixel-producing modules (``astro_calibrate``, ``astro_sequence``,
    ``astro_triage``, ``astro_register``, ``astro_stack``, ``astro_develop``, ``astro_meteors``,
    ``astro_orchestrator``)
    ride the normal develop chain but wire through none of GRADE_ALGO_VERSION,
    ``_HASHED_SOURCES``, or bracket's grouping, so they are hashed the same WHOLE-MODULE way as
    ``masks``: they are still being built out, and a hand-kept function tuple would silently stop
    covering new code the moment a stage is added and the tuple forgotten -- the same failure
    mode the ``masks`` reasoning above warns about. Their ALL_CAPS constants are ALSO swept live
    via ``_tunable_constants``, matching ``masks``'s own treatment: the whole-module source
    already reflects a real file edit, but a constant mutated at runtime (tests, or any future
    construction-time override) needs a live read to be caught, the same way ``masks.LUMA_EDGES``
    is. ``astro_sidecar`` is excluded -- it writes JSON metadata and never touches a pixel.
    ``astro_orchestrator`` IS hashed: it produces no pixel itself, but it picks the registration
    reference and decides which frames reach the stack, so a change there moves output pixels as
    surely as a stretch retune. It is also the one astro module that reaches back into ``develop``
    (the HE->DNG converter, the TIFF writer), so hashing it is where a cycle could close -- it
    stays open only because those reaches are local imports inside function bodies, the same
    discipline this function itself uses, pinned by a fresh-interpreter test."""
    import inspect

    from . import (
        astro_calibrate,
        astro_develop,
        astro_meteors,
        astro_orchestrator,
        astro_register,
        astro_sequence,
        astro_stack,
        astro_triage,
        bracket,
        looks,
        masks,
    )

    astro_modules = (
        astro_calibrate,
        astro_sequence,
        astro_triage,
        astro_register,
        astro_stack,
        astro_develop,
        astro_meteors,
        astro_orchestrator,
    )
    grouping = (bracket.detect_brackets, bracket._step, bracket._read_frame)
    try:
        local = "".join(inspect.getsource(fn) for fn in _HASHED_SOURCES + grouping)
        local += inspect.getsource(masks)
        local += "".join(inspect.getsource(module) for module in astro_modules)
    except (OSError, TypeError):
        local = "src-unavailable"
    tunables = (
        _tunable_constants()
        + _tunable_constants(vars(bracket))
        + _tunable_constants(vars(masks))
        + _tunable_constants(vars(looks))
    )
    for module in astro_modules:
        tunables += _tunable_constants(vars(module))
    return GRADE_ALGO_VERSION + local + repr(tunables)


def _develop_algo_version() -> str:
    """Fingerprint of the develop-producing code, mirroring GRADE_ALGO_VERSION.

    Hashed from source, never hand-bumped: a change to the shared grade math or
    to the photo-only stages self-invalidates the cache. A forgotten bump is how
    a stale cache replays a bug that was already fixed in code.
    """
    import hashlib

    return hashlib.sha256(_develop_algo_inputs().encode("utf-8")).hexdigest()[:12]


# No cache STORE is wired yet, deliberately (wiring one is a feature, not part
# of this integrity fix) — but the pair is not unread: ``grade_sidecar_payload``
# stamps both into every ``.grade.json`` as provenance. DEVELOP_ALGO_VERSION and
# develop_cache_key are the integrity contract that future cache must key on: a
# store keyed on them self-invalidates on any change to the source
# file, the grade params, the operator's DevelopOptions knobs, the render depth,
# the LUT cube a look rendered through, the decode/grade/spatial code, any
# ALL_CAPS tunable, the decoder identity (true raw vs converted DNG vs embedded
# preview), or the rawpy/LibRaw runtime. Wire a cache on this key or not at all.
DEVELOP_ALGO_VERSION = _develop_algo_version()


def _decoder_runtime() -> str:
    """rawpy/LibRaw identity: a decoder-library upgrade re-demosaics
    differently, so pixels cached under the old library must not replay."""
    try:
        import rawpy
    except Exception:
        return "rawpy-absent"
    version = getattr(rawpy, "__version__", "?")
    return f"rawpy={version},libraw={getattr(rawpy, 'libraw_version', '?')}"


def _lut_surface_identity(lut: "tuple[str, Cube] | None") -> str:
    """Which cube a LUT variant rendered through — name AND table contents.

    ``develop_variants`` renders every LUT look from the ONE shared under-grade
    and stamps each with those same ``params``, so the cube is the only thing
    that tells two film looks apart. The name alone is not identity either: two
    packs both ship a ``portra400``, and a cube gets re-exported in place under
    its old name — the table itself is what moved the pixels.
    """
    import hashlib

    if lut is None:
        return "lut=none"
    name, cube = lut
    table = np.ascontiguousarray(cube.table, dtype=np.float32).tobytes()
    return f"lut={name}:{cube.size}:{hashlib.sha256(table).hexdigest()[:16]}"


def develop_cache_key(
    path,
    params: CorrectionParams,
    decoder: str = "",
    *,
    options: DevelopOptions,
    depth: int,
    lut: "tuple[str, Cube] | None" = None,
) -> str:
    """Cache key for one developed photo: source + decoder + every render input.

    ``decoder`` is the ``DecodedPhoto.decoder`` tag of the frame being cached.
    Blank is a hard fail, not a default: the same NEF develops to different
    pixels via true raw, converted DNG, or embedded preview, and a key without
    decoder identity would replay one as another.

    ``params`` is only the COLOUR chain. Everything else that moves pixels is
    keyed here too and is REQUIRED, not defaulted, so a caller cannot silently
    key a vignetted render as the clean one: the whole ``options`` record (the
    spatial stages, the masked ``layers``, the LUT levers — read off the
    dataclass, so a knob added tomorrow is keyed the moment it exists), the
    render ``depth`` (8 and 16 quantize the same grade differently), and the
    ``lut`` surface's cube where one applies.
    """
    import hashlib

    if not decoder.strip():
        raise ValueError(
            f"{path}: develop_cache_key requires the DecodedPhoto.decoder tag — without "
            "it, preview-fallback and true-raw develops of one file share a cache entry"
        )
    path = Path(path)
    stat = path.stat()
    payload = "|".join(
        [
            str(path.resolve()),
            f"{stat.st_size}:{stat.st_mtime_ns}",
            decoder,
            _decoder_runtime(),
            DEVELOP_ALGO_VERSION,
            repr(astuple(params)),
            repr(astuple(options)),
            f"depth={depth}",
            _lut_surface_identity(lut),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
