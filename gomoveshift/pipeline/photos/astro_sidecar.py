"""Provenance sidecar for an astro stacking run: frames used vs rejected
(with reasons), per-frame registration transforms and triage metrics,
streak classifications, and the develop/stack numbers -- written beside the
exported master exactly the way ``develop.write_grade_sidecar`` writes
``.grade.json``.

Deliberately standalone: must NOT import ``develop.py`` (``develop.py``
imports every astro module -- including this one -- for
cache-key hashing, so an import back from here would cycle) and must NOT
import ``astro_orchestrator`` (its caller; an import back would cycle). The algorithm-version string is therefore a caller-supplied
parameter, never read off ``develop.DEVELOP_ALGO_VERSION``.

The dataclasses this module reads are full of numpy arrays, numpy scalars,
and ``Path`` values -- none of which ``json.dumps`` accepts. Converting all
of those to plain Python types is the actual job here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .astro_calibrate import CaptureIdentity
from .astro_develop import AstroDevelopResult
from .astro_register import RegistrationResult
from .astro_stack import StackResult
from .astro_triage import FrameMetrics, StreakClassification, classify_shape

# Bound as a module attribute on purpose: the stack's own record of what its
# frames were recognised as is patched in tests at THIS name, and a
# ``from ... import`` here is also the one import edge between the astro engines
# and photos/subject.py. Previously there was none -- twenty ``astro_*.py``
# modules and not one of them asked what it was stacking.
from .subject import SOURCE_OPERATOR, SubjectCategory, recognise_photos

logger = logging.getLogger(__name__)

# Sidecar filename suffix and schema tag, the ``.grade.json`` pattern.
STACK_SIDECAR_SUFFIX = ".stack.json"
# v2: input/capture identity, converter/mask/crop, residuals, meteor decisions,
# warnings, output hash -- additive only, every v1 key and shape unchanged.
# v3: stack.coverage (min/median per-pixel contributor count, below-floor fraction)
# and stack.thin_coverage_fraction -- additive only, every v1/v2 key and shape unchanged.
# v4: per-frame metrics.shape_verdict (astro_triage.classify_shape) and metrics.shape,
# the major/minor/ellipticity split the isotropic fwhm_px averages away -- additive
# only, every v1/v2/v3 key and shape unchanged.
# v5: per-frame radial_inlier_residuals_px_in_sample (center/middle/outer RANSAC-inlier
# residuals, labeled in_sample -- fit-derived, not held-out geometry evidence) and
# match_yield_by_stage (``match_progressive``'s per-stage match counts) -- additive
# only, every v1-v4 key and shape unchanged.
# v6: the ``subject`` block -- what photos/subject.py recognised the stacked
# frames as, its per-category tally, whether each reading was the operator's or
# an inference, and every frame disagreeing with the dominant reading BY NAME.
# Additive only, every v1-v5 key and shape unchanged.
STACK_SIDECAR_VERSION = "6"

# Streamed read size for content hashing -- a multi-GB RAW is never held whole in memory.
_HASH_CHUNK_BYTES = 1 << 20


@dataclass(frozen=True)
class UsedFrame:
    """One frame the stack used: its triage metrics, plus the registration
    transform onto the reference (``None`` for the reference frame itself,
    which is never registered against anything)."""

    metrics: FrameMetrics
    registration: RegistrationResult | None


def _jsonable(value: Any) -> Any:
    """Recursively: numpy array -> list, numpy scalar -> float/int, Path -> str, tuple -> list."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, slice):
        return [_jsonable(value.start), _jsonable(value.stop)]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _hash_file_bytes(path) -> tuple[str | None, str | None]:
    """SHA-256 of ``path``'s bytes, streamed -- never loaded whole, never copied, RAW
    included. ``(digest, None)``, or ``(None, reason)`` for anything unreadable (missing,
    permission, directory, ``None``/wrong type). Never raises: one bad path can't abort
    a sidecar write."""
    try:
        fs_path = os.fspath(path)  # rejects int/dict/etc; open() would treat an int as a fd
        digest = hashlib.sha256()
        with open(fs_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(_HASH_CHUNK_BYTES), b""):
                digest.update(chunk)
        return digest.hexdigest(), None
    except (OSError, TypeError, ValueError) as exc:
        return None, str(exc)


def _input_frame_payload(path) -> dict:
    sha256, hash_error = _hash_file_bytes(path)
    try:
        # fsdecode, not Path(path): bytes hash above but Path() rejects them (null path).
        jsonable_path = _jsonable(Path(os.fsdecode(path))) if path is not None else None
    except TypeError:
        jsonable_path = None
    return {"path": jsonable_path, "sha256": sha256, "hash_error": hash_error}


def _capture_identity_payload(identity: CaptureIdentity | None) -> dict:
    """``CaptureIdentity`` as a plain dict, or an honest absence record."""
    if identity is None:
        return {"available": False, "reason": "no CaptureIdentity supplied by the caller"}
    return {
        "available": True,
        "path": _jsonable(identity.path),
        "camera": identity.camera,
        "raw_shape": _jsonable(identity.raw_shape),
        "cfa_pattern": _jsonable(identity.cfa_pattern),
        "iso": identity.iso,
        "exposure_s": identity.exposure_s,
        "temperature_c": identity.temperature_c,
    }


def _star_shape_payload(shape) -> dict | None:
    """``FrameMetrics.shape``'s major/minor/ellipticity split, or ``None`` when no star
    was measurable -- an honest absence, never a faked zero. ``fwhm_px`` alone cannot
    tell a fat round star from an elongated one, which is the whole diagnostic value."""
    if shape is None:
        return None
    return {
        "fwhm_major_px": float(shape.fwhm_major_px),
        "fwhm_minor_px": float(shape.fwhm_minor_px),
        "ellipticity": float(shape.ellipticity),
    }


def _frame_metrics_payload(metrics: FrameMetrics) -> dict:
    return {
        "path": _jsonable(metrics.path),
        "star_count": int(metrics.star_count),
        "background_median": float(metrics.background_median),
        "fwhm_px": float(metrics.fwhm_px),
        # None (not []) records that the streak scan was skipped -- see FrameMetrics.
        "streaks": None if metrics.streaks is None else _jsonable(metrics.streaks),
        "usable": bool(metrics.usable),
        # A report, never a gate: classify_shape's thresholds are PROVISIONAL and it
        # touches no usable/cloud/dew flag, so recording it cannot reject a frame.
        "shape": _star_shape_payload(metrics.shape),
        "shape_verdict": classify_shape(metrics),
    }


def _used_frame_payload(frame: UsedFrame) -> dict:
    payload = {"metrics": _frame_metrics_payload(frame.metrics)}
    registration = frame.registration
    if registration is not None:
        payload["transform"] = _jsonable(registration.transform)
        payload["n_matched"] = int(registration.n_matched)
        payload["inlier_fraction"] = float(registration.inlier_fraction)
        payload["residual_median_px"] = float(registration.residual_median_px)
        payload["residual_p95_px"] = float(registration.residual_p95_px)
        payload["match_yield"] = float(registration.match_yield)
        # In-sample: RANSAC-inlier residuals/stage yields under THIS fit's own transform,
        # not held-out geometry evidence (the lens solver's residuals are the held-out numbers).
        payload["radial_inlier_residuals_px_in_sample"] = (
            None
            if registration.radial_inlier_residuals_px is None
            else _jsonable(registration.radial_inlier_residuals_px)
        )
        payload["match_yield_by_stage"] = (
            None
            if registration.match_yield_by_stage is None
            else _jsonable(registration.match_yield_by_stage)
        )
    return payload


def coverage_payload(coverage) -> dict | None:
    """``StackResult.coverage`` (``astro_quality.CoverageStats``) as a plain dict, or
    ``None`` when the caller never supplied real per-frame ``sky_validity`` -- an
    honest "not measured", never a faked zero. Shared with ``astro_benchmark``, which
    threads the same ``StackResult.coverage`` into its own ``stack`` payload."""
    if coverage is None:
        return None
    return {
        "min_count": int(coverage.min_count),
        "median_count": float(coverage.median_count),
        "below_floor_fraction": float(coverage.below_floor_fraction),
    }


def _streak_payload(streak: StreakClassification) -> dict:
    return {
        "frame_index": int(streak.frame_index),
        "path": _jsonable(streak.path),
        "streak": _jsonable(streak.streak),
        "label": streak.label,
    }


def subject_payload(light_paths: Sequence) -> dict:
    """What subject recognition made of the frames this stack was handed.

    ``recognise_photos`` decodes no pixels -- it reads EXIF, bracket membership,
    session runs and each frame's operator-correction sidecar -- so a card
    resolves in milliseconds beside the file hashing this module already does.

    Four states, kept apart, because collapsing any of them into "fine" is the
    defect this block exists to end:

    * a dominant category with its tally -- what the frames were recognised as;
    * ``unknown`` counted as ``unknown`` and allowed to BE the dominant. The
      frames were read and nothing was proven, which is not agreement;
    * frames disagreeing with the dominant, listed **by path**. A bare count of
      disagreements is a number nobody can act on;
    * ``frames_read == 0`` with a ``reason`` -- no lights given, or recognition
      itself failed. An empty block must never read as "all four agreed".

    **Records; never gates.** A recognition failure costs the operator nothing:
    the stack already succeeded, and refusing to write its run record because
    the EXIF reader blew up would let a report decide an outcome.
    """
    # Coerce the way ``_input_frame_payload`` does -- fsdecode, not ``Path(p)``,
    # and each slot independently. This function's own suite pins that a junk
    # light-path slot (bytes, an int, a dict, ``None``) DEGRADES the record
    # rather than aborting it, and a list comprehension over ``Path(p)`` throws
    # the whole sidecar away over one bad entry.
    paths: list[Path] = []
    unreadable = 0
    for raw in light_paths or []:
        try:
            paths.append(Path(os.fsdecode(raw)))
        except TypeError:
            unreadable += 1
    empty = {
        "frames_read": 0,
        "dominant": None,
        "dominant_share": None,
        "categories": {},
        "sources": {},
        "disagreeing": [],
        "reason": "",
    }
    if not paths:
        reason = (
            f"none of the {unreadable} light-path slot(s) given was a readable path"
            if unreadable
            else "no light frames were given to record a subject for"
        )
        return {**empty, "reason": reason}

    try:
        readings = recognise_photos(paths)
    except Exception as exc:  # noqa: BLE001 -- a record must not cost a written run record
        logger.warning("astro sidecar: subject recognition failed; recording that: %s", exc)
        return {**empty, "reason": f"subject recognition failed: {exc}"}

    categories: dict[str, int] = {}
    sources: dict[str, int] = {}
    for path in paths:
        reading = readings.get(path)
        if reading is None:
            # A path the recogniser did not answer for is not an agreement.
            name = SubjectCategory.UNKNOWN.value
            categories[name] = categories.get(name, 0) + 1
            continue
        name = reading.category.value
        categories[name] = categories.get(name, 0) + 1
        sources[reading.source] = sources.get(reading.source, 0) + 1

    read = sum(categories.values())
    # Ties resolve on the category name so one card never records two different
    # dominants across runs; the tally beside it is what makes a tie visible.
    dominant = max(sorted(categories), key=lambda c: categories[c])
    disagreeing = [
        {"path": _jsonable(path), "category": readings[path].category.value}
        for path in paths
        if path in readings and readings[path].category.value != dominant
    ]
    operator_read = sources.get(SOURCE_OPERATOR, 0)
    notes = []
    if operator_read:
        notes.append(f"{operator_read} of {read} frame(s) carry an operator correction")
    if unreadable:
        # Never silently absent: a slot that was not a path was still handed in.
        notes.append(f"{unreadable} light-path slot(s) were not readable paths")
    return {
        "frames_read": read,
        "dominant": dominant,
        "dominant_share": categories[dominant] / read,
        "categories": categories,
        "sources": sources,
        "disagreeing": disagreeing,
        # Said plainly rather than left to be inferred from ``sources``.
        "reason": "; ".join(notes),
    }


def stack_sidecar_payload(
    *,
    used_frames,
    rejected_frames,
    develop: AstroDevelopResult,
    stack: StackResult,
    decoder: str,
    algo_version: str,
    streaks=(),
    meteor_mode: str = "off",
    meteor_candidate_count: int = 0,
    meteor_reinserted_count: int = 0,
    meteor_skip_reason: str | None = None,
    light_paths: Sequence = (),
    dark_paths: Sequence = (),
    capture_identity: CaptureIdentity | None = None,
    converter_version: str | None = None,
    horizon_frac: float | None = None,
    mask_mode: str | None = None,
    crop: Sequence[int] | None = None,
    meteor_reinsertion_decisions: Sequence = (),
    warnings: Sequence[str] = (),
) -> dict:
    """The run record for one astro stack, as a plain JSON-serialisable dict.
    ``develop.peak`` is recorded exactly as given, including its unmeasured
    ``0.0`` default, since ``develop_stack_with_params`` reads ``base.peak``
    verbatim and only skips normalisation when it is ``<= 0``. Every kwarg from
    ``light_paths`` on is optional with an honest "nothing given" default, never a
    guess. NOT pure: it reads and hashes the bytes at ``light_paths``/``dark_paths``,
    so equal arguments give equal dicts only while those files are unchanged."""
    return {
        "sidecar_version": STACK_SIDECAR_VERSION,
        "algo_version": algo_version,
        "decoder": decoder,
        "frames": {
            "used": [_used_frame_payload(f) for f in used_frames],
            "rejected": [
                {"path": _jsonable(path), "reason": reason} for path, reason in rejected_frames
            ],
        },
        "streaks": [_streak_payload(s) for s in streaks],
        "meteors": {
            "mode": meteor_mode,
            "candidate_count": int(meteor_candidate_count),
            "reinserted_count": int(meteor_reinserted_count),
            "skip_reason": meteor_skip_reason,
            "decisions": [_jsonable(d) for d in meteor_reinsertion_decisions],
        },
        "develop": {
            "black_point": float(develop.black_point),
            "m_values": [float(m) for m in develop.m_values],
            "background_before": float(develop.background_before),
            "background_after": float(develop.background_after),
            "background_gains": [float(g) for g in develop.background_gains],
            "peak": float(develop.peak),
        },
        "stack": {
            "n_frames": int(stack.n_frames),
            "mean_rejected_fraction": float(stack.mean_rejected_fraction),
            "coverage": coverage_payload(stack.coverage),
            "thin_coverage_fraction": (
                float(np.mean(stack.thin_coverage)) if stack.thin_coverage is not None else None
            ),
        },
        "inputs": {
            "light": [_input_frame_payload(p) for p in light_paths],
            "dark": [_input_frame_payload(p) for p in dark_paths],
        },
        "capture": _capture_identity_payload(capture_identity),
        "converter_version": converter_version,
        "mask": {"horizon_frac": horizon_frac, "mode": mask_mode},
        "crop": _jsonable(list(crop)) if crop is not None else None,
        "warnings": [_jsonable(w) for w in warnings],
        # Computed HERE, off the same light_paths this function already reads and
        # hashes, rather than taken as a kwarg: a caller that can forget to pass
        # it is a run record that can silently omit what it was stacking.
        "subject": subject_payload(light_paths),
    }


def write_stack_sidecar(payload: dict, image_path) -> Path | None:
    """Write ``<stem>.stack.json`` beside an astro export, atomically -- the
    ``write_grade_sidecar`` temp-file-then-``os.replace`` pattern.
    Any failure is logged and returns ``None``, never
    raises: the export is the product; the sidecar is metadata."""
    out_path = Path(image_path).with_suffix(STACK_SIDECAR_SUFFIX)
    tmp_path = out_path.with_name(f".{out_path.name}.tmp")
    output_hash, output_hash_error = _hash_file_bytes(image_path)
    record = {**payload, "output_hash": output_hash, "output_hash_error": output_hash_error}
    try:
        text = json.dumps(record, indent=2, sort_keys=True, default=str) + "\n"
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(tmp_path, out_path)
    except Exception as exc:  # metadata must never lose an export
        logger.error("%s: stack sidecar not written (%s); the export stands", out_path, exc)
        tmp_path.unlink(missing_ok=True)
        return None
    return out_path
