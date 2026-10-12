"""Preflight resource checks for the photo posting pipeline.

The drive pipeline gates disk/RAM before a run; the photo path historically did
not, so a batch could die part-way with photos already resized/uploaded. This
runs the same disk/RAM checks (REUSING the drive preflight primitives) upstream
of the first irreversible step. Thresholds are lower than the drive path's
(50 GB / 2 GB): a photo batch writes a few hundred MB vs ~60 GB per-day masters,
so a 50 GB floor would only false-fail normal volumes. It also surfaces a
leftover unbooked-ledger marker so a prior silent booking miss can't turn into a
silent double-book on the next run.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ..preflight import check_blur_inference_provider, check_disk_space, check_ram
from . import face_filter

logger = logging.getLogger(__name__)

# Persisted beside the photo output dir when a confirmed upload could not be
# booked into the shared content ledger (see orchestrator._mark_ledger_unbooked).
LEDGER_UNBOOKED_MARKER = ".photo_ledger_unbooked.json"

# Photo batches are light relative to the drive path — a floor that protects
# without false-failing an otherwise-healthy volume.
_PHOTO_MIN_FREE_GB = 2
_PHOTO_MIN_RAM_GB = 1.0

# Ingest develops full-resolution stills. The grade chain (develop._grade)
# flattens a ~20.7 MP frame (5568x3712) to float64 — 20.7e6 px x 3 ch x 8 B
# ~= 0.46 GiB per copy — and holds up to ~4 copies concurrently (step input +
# output + two expression temporaries in the saturation/final-quantize steps),
# a ~2 GB transient per photo. 4 GB keeps ~2x headroom over that peak for the
# decode/denoise working sets and the interpreter; the publish-path 1 GB floor
# is resize/upload-sized and would wave a develop run into the OOM killer.
_INGEST_MIN_RAM_GB = 4.0
# Gallery output is full-resolution JPEGs — tens of MB per card, a few GB at
# the extreme — so the modest photo-path disk floor is reused as-is.
_INGEST_MIN_FREE_GB = _PHOTO_MIN_FREE_GB

# Astro stacking's peak float32 working set: astro_stack.stack_sky/stack_ground band
# directly off the caller's frame LIST (``np.stack([f[row0:row1] for f in frames])`` per
# BAND_ROWS-row band) instead of first materializing a whole ``(N, H, W, C)`` array, so no
# full sequence-equivalent copy is ever allocated. At the instant stack() reaches
# ``rgb = blend(sky, stack_ground(unaligned_frames), mask)`` (astro_stack.py), only the
# caller's own two lists are simultaneously live:
#   1. `aligned_frames` -- still bound, stack()'s own parameter
#   2. `unaligned_frames` -- definitionally live, the argument being passed to
#      stack_ground right here
# Everything else (a row band, the sigma-clip mask, the (H, W, C) output accumulator) is a
# small multiple of one band or one frame, not one whole sequence-equivalent, so it drops
# out of this budget. Per-sequence-equivalent bytes: frames x pixels x 3 channels x 4 bytes
# (float32); a 20-frame, 24 MP chunk is 20 x 24e6 x 3 x 4 ~= 5.4 GiB per sequence-equivalent,
# ~10.7 GiB for both before headroom. This term alone is NOT the whole model
# -- the headroom factor below covers only per-frame register/warp transients; the RAW
# decode buffer and the develop chain each get their OWN term (below), because a flat 1.25x headroom does not absorb them.
# ASSUMES frames arrive already float32 (see astro_stack._validated_frames' docstring) --
# any other dtype costs one extra sequence-equivalent at peak and this estimate under-counts.
_ASTRO_PEAK_SEQUENCE_EQUIVALENTS = 2
_ASTRO_RAM_HEADROOM_FACTOR = 1.25
_BYTES_PER_GIB = 1_073_741_824

# astro_develop.develop_stack() runs ONCE per stack() output -- frame-count-INDEPENDENT,
# unlike the sequence term above -- but each of its three steps (remove_gradient,
# background_wb, _stretch_detailed) converts the frame to float64 and holds several
# float64-sized copies at once (an ``np.asarray(..., dtype=np.float64)`` conversion, a
# working buffer the same size, and a downcast-back transient at return). Summed, not
# maxed, across the three steps as an upper bound -- nothing here proves the GC has freed
# one step's arrays before the next step starts -- that's roughly 8 float64-frame-
# equivalents. ESTIMATED, not measured directly (see the decode-overhead term below for
# how it was calibrated against the one real datapoint this model has).
_ASTRO_DEVELOP_GIB_PER_MP = 0.18

# The RAW decode working set for ONE frame (rawpy/LibRaw demosaic buffers inside
# astro_calibrate.decode_linear -- prepare_frames' concurrency beyond the first worker is
# charged separately, see the frame-equivalents term below) plus the sigma-clip loop's
# float64 promotion (astro_stack.
# _sigma_clipped_stack divides a float32 band by an int64 count, upcasting to float64 --
# band-height-bounded by BAND_ROWS, not the whole frame, but real) plus baseline
# interpreter/library overhead this model doesn't otherwise account for. NOT measured
# independently -- this is the residual needed to clear the one real peak we have
# (11.9 GB @ 5 frames x 20.9 MP, the measured peak that
# test_astro_ram_estimate_matches_measured_peak pins as a floor), so treat it as the
# least-trusted term here until astro_benchmark.py's child-process RSS measurement runs
# across more real frames and can decompose it properly.
_ASTRO_RAM_DECODE_OVERHEAD_GIB_PER_MP = 0.28

# Marginal cost of ONE extra concurrent decode, in decoded-frame-equivalents. The term
# above is a RESIDUAL -- it also absorbs the sigma-clip float64 promotion and baseline
# interpreter overhead, neither of which is per-worker -- so multiplying IT by the worker
# count would charge 5.85 GiB per extra thread at 20.9 MP and refuse runs that fit
# comfortably. A gate that refuses a fitting run is as broken as one that waves an OOM
# through, so the per-worker cost is measured and expressed in frame-equivalents instead,
# which scales with megapixels on its own rather than pinning a fitted GiB constant.
#
# Measured with a child-process peak-RSS sweep, each worker count in a FRESH child so no
# reading inherits the previous one's retained heap. Peak RSS grows roughly linearly with
# workers; the marginal cost per extra worker is about 1.5 decoded float32 RGB frames: the
# frame itself plus rawpy/LibRaw's demosaic buffers. 2 sits above that with room, and
# stays an upper bound as the allocator's reuse varies (more workers can read LOWER).
_ASTRO_CONCURRENT_DECODE_FRAME_EQUIVALENTS = 2

# astro_calibrate.build_master_dark: read_raw_mosaic returns each dark as a standalone
# uint16 Bayer mosaic (2 B/px), ALL dark_count of them held live at once by the caller's
# list comprehension (astro_orchestrator.prepare_frames). build_master_dark then builds a
# whole second list of float32 conversions (4 B/px each, fully materialized by the list
# comprehension before np.stack sees it) and np.stack copies those into one more float32
# array (4 B/px) -- 2 + 4 + 4 = 10 bytes/px/dark at peak. Dark-count-scaling, LIGHT-frame-
# count-independent.
_ASTRO_DARK_BYTES_PER_MP_PER_DARK = 10

# Meteor reinsertion is off by default and streams layers one at a time
# (astro_orchestrator._stream_meteor_layers: "at most two full-res layers ... are
# resident"). Bounded and, in the default posture (reinsertion off), zero -- this term
# only applies when a caller explicitly signals reinsertion is possible. Modeled as one
# more develop-chain-equivalent transient (a layer is developed via
# develop_stack_with_params, which re-runs remove_gradient's float64 chain), a
# conservative upper bound since only one layer is actively being developed at a time.
_ASTRO_METEOR_LAYER_GIB_PER_MP = _ASTRO_DEVELOP_GIB_PER_MP

# Stack output is one blended frame -- same modest disk footprint as the rest of the
# photo path, so the photo-path floor is reused as-is.
_ASTRO_MIN_FREE_GB = _PHOTO_MIN_FREE_GB

# A reusable chunk artifact is two float32 RGB planes (sky numerator + ground) and one
# uint16 contributor plane: 24 + 2 = 26 bytes/pixel. Publication is atomic, so the peak
# must include one full temporary artifact in addition to every completed chunk. A TIFF16
# is conservatively six bytes/pixel before compression; sidecar JSON is negligible at this
# scale. The 10% margin covers .npy headers, manifests, filesystem allocation, and estimates
# based on rounded megapixels rather than the exact sensor dimensions.
_ASTRO_LINEAR_BYTES_PER_PIXEL = 26
_ASTRO_TIFF16_BYTES_PER_PIXEL = 6
_ASTRO_RUN_DISK_SAFETY_FACTOR = 1.10


def estimate_astro_run_disk_gb(
    *, chunk_count: int, megapixels: float, keep_linear: bool, write_tiffs: bool
) -> float:
    """Peak output-disk requirement for the whole planned astro run.

    Unlike the legacy fixed 2 GiB floor, this includes all requested chunk artifacts,
    requested TIFFs, and one full temporary artifact for atomic publication.
    """
    pixels = megapixels * 1_000_000
    artifact_count = chunk_count + 1 if keep_linear else 0
    artifact_bytes = artifact_count * pixels * _ASTRO_LINEAR_BYTES_PER_PIXEL
    tiff_bytes = chunk_count * pixels * _ASTRO_TIFF16_BYTES_PER_PIXEL if write_tiffs else 0
    return (artifact_bytes + tiff_bytes) * _ASTRO_RUN_DISK_SAFETY_FACTOR / _BYTES_PER_GIB


def preflight_astro_run_disk(
    output_dir: Path,
    *,
    chunk_count: int,
    megapixels: float,
    keep_linear: bool,
    write_tiffs: bool,
) -> None:
    """Gate aggregate run output space once, before the first RAW decode."""
    estimated = estimate_astro_run_disk_gb(
        chunk_count=chunk_count,
        megapixels=megapixels,
        keep_linear=keep_linear,
        write_tiffs=write_tiffs,
    )
    check_disk_space(Path(output_dir), min_free_gb=max(_ASTRO_MIN_FREE_GB, estimated))


def preflight_photo_run(
    output_dir: Path,
    *,
    min_free_gb: int = _PHOTO_MIN_FREE_GB,
    min_ram_gb: float = _PHOTO_MIN_RAM_GB,
    blur_enabled: bool | None = None,
) -> None:
    """Gate disk/RAM (reusing the drive preflight primitives, same
    raise-on-shortfall semantics) and warn on a leftover unbooked-ledger marker,
    before the photo run's first irreversible step (resize, upload).

    ``blur_enabled=None`` (the default) resolves the per-run operator setting
    via ``face_filter.blur_enabled()`` — the same silent CPU-fallback the drive
    pipeline already refuses can hit the photo path's
    identical redactcam-backed blur, so a blur-enabled run refuses to start on a
    clobbered venv before any resize/upload work."""
    if blur_enabled is None:
        blur_enabled = face_filter.blur_enabled()
    if blur_enabled:
        check_blur_inference_provider()
    output_dir = Path(output_dir)
    check_disk_space(output_dir, min_free_gb=min_free_gb)
    check_ram(min_free_gb=min_ram_gb)
    warn_if_unbooked_marker(output_dir)


def preflight_photo_ingest(
    gallery_dir: Path,
    *,
    min_free_gb: int = _INGEST_MIN_FREE_GB,
    min_ram_gb: float = _INGEST_MIN_RAM_GB,
    blur_enabled: bool | None = None,
) -> None:
    """Gate disk/RAM before ingest's first decode — same primitives and
    raise-on-shortfall semantics as ``preflight_photo_run``, but with a
    develop-sized RAM floor: ingest is the heaviest photo path (full-resolution
    decode + float64 grade transient per frame), not a resize/upload batch.

    ``blur_enabled=None`` resolves via ``face_filter.blur_enabled()``, same as
    ``preflight_photo_run`` — see that docstring for why this check exists."""
    if blur_enabled is None:
        blur_enabled = face_filter.blur_enabled()
    if blur_enabled:
        check_blur_inference_provider()
    check_disk_space(Path(gallery_dir), min_free_gb=min_free_gb)
    check_ram(min_free_gb=min_ram_gb)


def estimate_astro_stack_ram_gb(
    frame_count: int,
    megapixels: float,
    *,
    dark_count: int = 0,
    meteor_reinsertion_enabled: bool = False,
    decode_workers: int = 1,
) -> float:
    """Estimated peak RAM (GiB) for one astro chunk end to end: prepare_frames'
    decode/dark calibration -> astro_stack.stack() -> astro_develop.develop_stack() ->
    (optional) meteor reinsertion. ``dark_count``/``meteor_reinsertion_enabled`` default
    to today's shape (no darks, meteors off) so an existing caller's estimate only moves
    because of the frame-count-independent terms below, never because of these new knobs.
    ``decode_workers`` is prepare_frames' concurrency: each worker beyond the first holds
    another frame and its demosaic buffers in flight, so the gate has to know the count
    BEFORE the first decode. See each constant above for WHERE in the pipeline its term
    comes from and how it
    scales -- the develop-chain/decode-overhead terms scale with megapixels but NOT frame
    count, which is why a small chunk's estimate doesn't collapse the way the old
    sequence-only model did."""
    pixels_per_frame = megapixels * 1_000_000
    bytes_per_sequence = frame_count * pixels_per_frame * 3 * 4
    sequence_gib = (
        bytes_per_sequence * _ASTRO_PEAK_SEQUENCE_EQUIVALENTS * _ASTRO_RAM_HEADROOM_FACTOR
    ) / _BYTES_PER_GIB
    develop_gib = megapixels * _ASTRO_DEVELOP_GIB_PER_MP
    decode_overhead_gib = megapixels * _ASTRO_RAM_DECODE_OVERHEAD_GIB_PER_MP
    dark_gib = (dark_count * pixels_per_frame * _ASTRO_DARK_BYTES_PER_MP_PER_DARK) / _BYTES_PER_GIB
    meteor_gib = megapixels * _ASTRO_METEOR_LAYER_GIB_PER_MP if meteor_reinsertion_enabled else 0.0
    # Serial decode (the default) holds one frame in flight and is already covered by the
    # residual above; only the EXTRA workers add anything, so decode_workers=1 leaves every
    # existing caller's estimate bit-identical.
    extra_workers = max(0, decode_workers - 1)
    concurrent_decode_gib = (
        extra_workers * _ASTRO_CONCURRENT_DECODE_FRAME_EQUIVALENTS * pixels_per_frame * 3 * 4
    ) / _BYTES_PER_GIB
    return (
        sequence_gib
        + develop_gib
        + decode_overhead_gib
        + dark_gib
        + meteor_gib
        + concurrent_decode_gib
    )


def preflight_astro_stack(
    output_dir: Path,
    *,
    frame_count: int,
    megapixels: float,
    dark_count: int = 0,
    meteor_reinsertion_enabled: bool = False,
    decode_workers: int = 1,
    min_free_gb: int = _ASTRO_MIN_FREE_GB,
    degrade_workers: bool = False,
) -> int:
    """Gate disk/RAM before astro stacking's first RAW decode — same primitives and
    raise-on-shortfall semantics as the other photo preflight gates, but with a RAM floor
    DERIVED from the run's own shape (``estimate_astro_stack_ram_gb``) instead of a fixed
    constant: a fixed floor would either false-fail a small chunk or wave a big one into
    the OOM killer. ``dark_count``/``meteor_reinsertion_enabled`` default to today's shape
    (no darks, meteors off) and ``decode_workers`` to serial decode -- a caller that knows its
    run uses darks/meteors, or decodes concurrently, should pass them so the gate reflects the
    actual run. Gating for serial decode and then decoding on N threads is how a run OOMs past
    a preflight that passed.

    Returns the ``decode_workers`` count it approved, which the caller must then decode with
    -- the gate and the decode agreeing is the whole point of the parameter.

    ``degrade_workers`` (mid-run callers only) tries each lower worker count before giving
    up. Concurrency is a SPEED choice and the frames are the product, so a chunk that no
    longer fits at 4 workers but still fits serially should be stacked slowly rather than
    thrown away: when free RAM falls below the parallel estimate (for example under
    concurrent CI), reduce workers rather than skip chunks that would fit
    serially. A chunk too big even for serial decode has nothing left to give up and still
    raises. It defaults OFF because the RUN-level gate must not quietly halve an
    operator's throughput for a whole night
    when the honest answer is a smaller --stack-len, which its refusal already suggests."""
    check_disk_space(Path(output_dir), min_free_gb=min_free_gb)
    attempts = range(decode_workers, 0, -1) if degrade_workers else (decode_workers,)
    refusal: RuntimeError | None = None
    for workers in attempts:
        try:
            check_ram(
                min_free_gb=estimate_astro_stack_ram_gb(
                    frame_count,
                    megapixels,
                    dark_count=dark_count,
                    meteor_reinsertion_enabled=meteor_reinsertion_enabled,
                    decode_workers=workers,
                )
            )
        except RuntimeError as exc:
            refusal = exc
            continue
        return workers
    raise refusal  # type: ignore[misc]  # attempts is never empty: decode_workers >= 1


def warn_if_unbooked_marker(output_dir: Path) -> None:
    """Log a loud WARNING if a prior run left an unbooked-ledger marker."""
    marker = Path(output_dir) / LEDGER_UNBOOKED_MARKER
    if not marker.exists():
        return
    try:
        slots = json.loads(marker.read_text()).get("slots", [])
    except (OSError, ValueError):
        slots = []
    logger.warning(
        "LEDGER-UNBOOKED: a prior photo run uploaded %d slot(s) that were never "
        "booked into the content ledger (marker: %s) — these publish slots may be "
        "double-booked. Review/hand-book them, then delete the marker.",
        len(slots),
        marker,
    )
