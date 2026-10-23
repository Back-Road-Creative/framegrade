"""Day-scoped master encoding for the day-first pipeline.

Given a `DayBatch` from `day_grouper`, produces a single day-level master
video file (`day_{date}_master.mp4`) that becomes the authoritative
source of truth for everything downstream: regular LFs are stream-copy
extractions from it, shorts are extractions from it or from the regular
LFs derived from it, and post-assembly OST text is burned on each
artifact by `ost_applier`.

Pipeline:
  1. Concatenate the day's raw per-episode files (the `concat_path` from
     each episode, which is itself a stream-copied chapter concat) via
     ffmpeg concat demuxer with `-c copy`. Output:
     `day_{date}_raw.mp4` — no re-encode, just joined bytes.
  2. Run ONE `SinglePassProcessor.process()` pass over the raw concat
     with color correction and audio loudness normalization. OST drawtext
     is explicitly disabled (`skip_ost=True`) — the title is burned after
     assembly by `ost_applier`. Output: `day_{date}_master.mp4`.
  3. Delete the intermediate raw concat file.

**Why this architecture**: color stats and audio loudness are computed
ONCE across the whole day's footage, guaranteeing visual and auditory
consistency across all subsequent regular-LF and short outputs. The
previous per-episode encoding approach produced potentially divergent
color/audio per episode.
"""

import hashlib
import json
import logging
import math
import os
import subprocess
from pathlib import Path
from typing import Optional

from ..config import Config
from . import blur_gate
from .cache import (
    TRANSIENT_COLOR_CACHE_SCHEMA_VERSION,
    file_fingerprint,
    load_cached_json,
    save_cache_json,
)
from .color_correction import GRADE_ALGO_VERSION
from .ffmpeg_runner import check_disk_space

logger = logging.getLogger(__name__)


# Tolerance for the per-episode ffprobe vs. batch.total_duration cross-check.
# 2.0s accommodates real-world GoPro HEVC container timing drift: Phase A
# records total_duration from MP4 atom metadata; on resume, fresh ffprobe may
# return sub-second-different results per episode due to fractional NTSC frame
# rounding differences (29.97/59.94 fps → ~16.7 ms per GOP boundary) that
# accumulate across multi-recording days. Observed: about 1.2 s of drift on a
# multi-recording day of over five hours with no source-file changes. The
# failure mode this guard exists for (truncated ingest files) produces under-reports of seconds to
# minutes — 2.0s is still far below that threshold.
DURATION_INTEGRITY_TOLERANCE_S = 2.0

# Day-master bytes per raw-source byte for the disk check, an upper bound.
# Recorded masters measure 0.63x to 0.89x their source; the old 1.5x rule
# assumed 0.5x, so a tight volume passed the check and could fill mid-encode.
MASTER_BYTES_PER_SOURCE_BYTE = 1.0


# Hand-maintained generation counter for master-encode rule changes that are
# NOT expressible as a constant — the shape of the encode itself (e.g. the
# attenuate-only audio + privacy blur rules). Bump it when the rule
# changes but every constant stays put.
MASTER_ENCODE_RULE_GENERATION = 2


# Constants the day-master encode consumes. Collected by naming convention so a
# new frozen knob is folded in without anyone remembering to list it; the shorts
# ladder is deliberately excluded because no shorts parameter reaches the master
# (invalidating a ~6 h / ~60 GB master over a shorts-only retune would be pure
# waste). Same convention-collection idea as ``DEVELOP_ALGO_VERSION``.
_ENCODE_CONSTANT_PREFIXES = ("LONGFORM_", "AUDIO_")


def _master_encode_algo_version() -> str:
    """Encode-rule identity for the day master: the generation counter joined to
    a fingerprint of the frozen encode constants.

    ``MASTER_ENCODE_ALGO_VERSION`` used to be a bare hand-bumped integer, so it
    tracked only the two rule changes someone remembered to bump it for. A
    change to NVENC CQ / preset / profile / pix_fmt / bitrate floor or maxrate
    ratio moved the pixels the master is made of and moved nothing else —
    ``verify_cached_master`` happily revalidated a pre-change master on a
    raw-episodes-purged resume, and every LF and short cut from it shipped the
    pre-change encode. Hashing the constants makes that bump automatic (robust
    by construction) instead of a thing to remember. Mirrors
    ``GRADE_ALGO_VERSION``, which does the same for the grade math.
    """
    from . import encoding_constants as ec

    names = sorted(n for n in vars(ec) if n.startswith(_ENCODE_CONSTANT_PREFIXES))
    payload = ";".join(f"{n}={getattr(ec, n)!r}" for n in names)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"{MASTER_ENCODE_RULE_GENERATION}-{digest}"


# Encode-rule version for the day master, computed once at import. Stamped into
# the ``per_day_completed[date]`` record by ``build_master_record`` and
# re-checked by ``verify_cached_master`` — a master missing the stamp, or
# carrying a different one, is refused (re-encode with ``--fresh``) rather than
# stream-copied. Complements — does not replace —
# ``blur_gate.require_master_blurred`` (which stays the blur-specific hard gate).
MASTER_ENCODE_ALGO_VERSION = _master_encode_algo_version()


class BatchDurationIntegrityError(RuntimeError):
    """Raised when the per-episode ffprobe sum disagrees with
    ``DayBatch.total_duration`` beyond ``DURATION_INTEGRITY_TOLERANCE_S``.

    Surfacing this as a hard error (rather than silently re-encoding) is
    deliberate: if the ingest-time per-episode durations are wrong, every
    downstream gate that trusts ``batch.total_duration`` is also compromised
    — re-encoding from the same truncated sources would just produce another
    bad master that passes the 99% gate.
    """


def assert_batch_duration_integrity(
    day_batch,
    concat_paths: list,
    *,
    probe_fn=None,
    tolerance_s: float = DURATION_INTEGRITY_TOLERANCE_S,
) -> float:
    """Independently ffprobe each episode source and assert the sum matches
    ``day_batch.total_duration`` within ``tolerance_s``.

    Run BEFORE any 99%-threshold compare against ``batch.total_duration`` so
    that a partial master cannot pass the gate just because the expected
    figure is itself under-reported (e.g. truncated ingest source files).

    Args:
        day_batch: a ``DayBatch`` (or any object exposing ``total_duration``
            and ``episodes``).
        concat_paths: ordered list of the per-episode source files that the
            master is being assembled from. Must be non-empty and match
            ``len(day_batch.episodes)`` — a missing concat path is itself
            evidence of partial state and is treated as an integrity error.
        probe_fn: optional callable ``(Path) -> float`` returning seconds.
            Defaults to ``FFprobeClient(subprocess).probe_duration``. Injected
            for testability.
        tolerance_s: absolute mismatch (in seconds) at or above which the
            disagreement is treated as a corruption signal. See module-level
            ``DURATION_INTEGRITY_TOLERANCE_S`` for the default + rationale.

    Returns:
        The sum of probed durations (seconds) on success — callers can use
        this as the authoritative expected duration in place of
        ``batch.total_duration`` if they prefer.

    Raises:
        BatchDurationIntegrityError on any of: missing concat paths,
        episodes without concat_path, individual episode probe returning
        zero / negative duration, or sum disagreement beyond tolerance.
    """
    expected = float(getattr(day_batch, "total_duration", 0.0) or 0.0)
    episodes = list(getattr(day_batch, "episodes", None) or [])
    date = getattr(day_batch, "date", "?")

    # All episodes must be present and have a concat_path. A None entry is
    # evidence of partial Phase A state — we will not silently average over
    # it.
    if not episodes:
        raise BatchDurationIntegrityError(
            f"Day {date}: no episodes on batch — cannot verify duration integrity"
        )
    missing = [
        getattr(ep, "index", i)
        for i, ep in enumerate(episodes)
        if getattr(ep, "concat_path", None) is None
    ]
    if missing:
        raise BatchDurationIntegrityError(
            f"Day {date}: {len(missing)} episode(s) missing concat_path "
            f"(indices {missing}) — cannot verify duration integrity"
        )
    if len(concat_paths) != len(episodes):
        raise BatchDurationIntegrityError(
            f"Day {date}: concat_paths count {len(concat_paths)} != episode "
            f"count {len(episodes)} — partial state, refusing to verify"
        )

    if probe_fn is None:
        from .validation.ffprobe_client import FFprobeClient

        client = FFprobeClient(subprocess)
        probe_fn = client.probe_duration

    probed_total = 0.0
    failures: list[str] = []
    for path in concat_paths:
        try:
            d = float(probe_fn(Path(path)))
        except Exception as e:
            failures.append(f"{Path(path).name}: probe raised {e!r}")
            continue
        if d <= 0.0:
            failures.append(f"{Path(path).name}: probe returned {d}")
            continue
        probed_total += d

    if failures:
        raise BatchDurationIntegrityError(
            f"Day {date}: {len(failures)} episode source(s) failed ffprobe — " + "; ".join(failures)
        )

    delta = abs(probed_total - expected)
    if delta > tolerance_s:
        raise BatchDurationIntegrityError(
            f"Day {date}: per-episode ffprobe sum disagrees with "
            f"batch.total_duration — probed={probed_total:.3f}s "
            f"expected={expected:.3f}s delta={delta:.3f}s "
            f"tolerance={tolerance_s}s. The batch's recorded duration is "
            f"likely under-reported (truncated ingest source?); a 99%-gate "
            f"resume on the master would silently accept a partial encode."
        )

    return probed_total


def _source_fingerprints(concat_paths: list) -> list[str]:
    """Compute fingerprints (size:mtime_ns) for each source episode file."""
    fps = []
    for p in concat_paths:
        try:
            fps.append(file_fingerprint(Path(p)))
        except OSError:
            fps.append("missing")
    return fps


def master_fingerprint(master_path: Path) -> str:
    """``"size:mtime_ns"`` fingerprint for a (potentially 60 GB) master.

    Uses ``file_fingerprint``, NOT sha256 — a full sha streams the whole 60 GB
    master from disk every resume just to detect a re-write, while
    ``size:mtime_ns`` flips on any re-write/truncation at zero I/O cost. Raises
    ``OSError`` when absent (callers treat that as "no fingerprint").
    """
    return file_fingerprint(Path(master_path))


def _day_source_bytes(day_batch) -> int:
    """Sum the day's raw source file sizes — the bytes preflight measures.

    Each episode's raw ``chapter_files()`` ARE the source-dir MP4s the
    preflight disk gate sizes; a slice-derived sub-episode (empty
    ``chapter_files``) is described by its own ``concat_path``. Captured at
    encode time so a resume can subtract a COMPLETED day from the peak-cache
    estimate (its master is cached, its raw episodes purgeable). Best-effort —
    an unstattable/missing file contributes 0 (a low estimate is conservative
    because preflight subtracts this from the required-space figure).
    """
    total = 0
    for ep in getattr(day_batch, "episodes", None) or []:
        files: list = []
        chapter_files = getattr(ep, "chapter_files", None)
        if callable(chapter_files):
            try:
                result = chapter_files()
                files = list(result) if result else []
            except TypeError:
                files = []
        if not files:
            concat_path = getattr(ep, "concat_path", None)
            if concat_path is not None:
                files = [concat_path]
        for f in files:
            try:
                total += Path(f).stat().st_size
            except (OSError, TypeError):
                continue
    return total


def assert_master_duration_integrity(
    date,
    expected_duration: float,
    master_duration: float,
    *,
    tolerance_s: float = DURATION_INTEGRITY_TOLERANCE_S,
) -> None:
    """Two-sided assert that a freshly-encoded master's own duration agrees
    with ``day_batch.total_duration`` within ``tolerance_s``.

    Called from ``encode_day_master`` the moment the master is promoted to its
    final path — upstream of every LF / short / superlong encode cut from it —
    and again from ``build_master_record``, which must not record a duration
    pair it never compared. Reuses ``DURATION_INTEGRITY_TOLERANCE_S`` rather
    than a second tolerance number.

    No-op when ``expected_duration < 1.0`` (missing/zero on the batch —
    cannot verify, mirrors the existing skip in ``encode_day_master``) or
    ``master_duration < 1.0`` (an encode-time ffprobe failure — already
    caught, without a hard crash here, by ``verify_cached_master``'s partial-
    metadata gate on the next resume).
    """
    if expected_duration < 1.0 or master_duration < 1.0:
        return
    delta = abs(master_duration - expected_duration)
    if delta > tolerance_s:
        raise BatchDurationIntegrityError(
            f"Day {date}: freshly-encoded master duration disagrees with "
            f"expected batch duration — master={master_duration:.3f}s "
            f"expected={expected_duration:.3f}s delta={delta:.3f}s "
            f"tolerance={tolerance_s}s. A master this far from the expected "
            f"day duration must not ship LFs/shorts derived from it."
        )


def _upscale_sidecar(master_path: Path) -> Path:
    """Where ``encode_day_master`` leaves the declared-upscale facts of a master."""
    return Path(master_path).with_suffix(".upscale.json")


def _record_upscale(master_path: Path, plan, profile) -> None:
    """Persist (or clear) the declared-upscale facts next to the promoted master.

    ``build_master_record`` folds them into ``per_day_completed`` as
    ``upscaled_from`` / ``native_resolution_claim``.
    """
    from .quality import claim_native_resolution_allowed

    sidecar = _upscale_sidecar(master_path)
    if plan is None:
        sidecar.unlink(missing_ok=True)
        return
    sidecar.write_text(
        json.dumps(
            {
                "upscaled_from": list(plan.source),
                "native_resolution_claim": claim_native_resolution_allowed(profile, plan),
            }
        )
    )


def _read_upscale(master_path: Path) -> dict:
    try:
        data = json.loads(_upscale_sidecar(master_path).read_text())
    except (OSError, ValueError):
        data = {}
    return {
        "upscaled_from": data.get("upscaled_from"),
        "native_resolution_claim": bool(data.get("native_resolution_claim", True)),
    }


def build_master_record(
    day_batch,
    master_path: Path,
    *,
    probe_fn=None,
) -> dict:
    """Build the ``per_day_completed[date]`` integrity metadata for a master.

    Captured ONCE at encode time so a later resume can validate the cached
    master after the raw episodes are purged: ``expected_duration``
    (= ``day_batch.total_duration``), ``master_duration`` (the master's own
    ffprobe), ``master_fingerprint`` (``size:mtime_ns``), and ``source_bytes``
    (the day's raw source size, so a resume's preflight disk gate can
    subtract this COMPLETED day from the peak-cache estimate); MERGED into the
    existing ``{master_path}`` record by the caller. RAISES
    ``BatchDurationIntegrityError`` (via ``assert_master_duration_integrity``)
    when a successfully-probed ``master_duration`` disagrees with
    ``expected_duration`` — best-effort on a probe failure (``0.0``), which
    the next resume's ``verify_cached_master`` partial-metadata gate catches.
    A fresh encode has already been gated on this comparison at promotion time
    inside ``encode_day_master``; this second, cheap (one ffprobe) assertion
    keeps the RECORDED metadata honest for callers that build a record for a
    master they did not just encode (e.g. a resume recovery record).
    """
    if probe_fn is None:
        from .validation.ffprobe_client import FFprobeClient

        probe_fn = FFprobeClient(subprocess).probe_duration
    try:
        master_duration = float(probe_fn(Path(master_path)))
    except Exception:
        master_duration = 0.0
    expected_duration = float(getattr(day_batch, "total_duration", 0.0) or 0.0)
    assert_master_duration_integrity(
        getattr(day_batch, "date", "?"), expected_duration, master_duration
    )
    try:
        fp = master_fingerprint(master_path)
    except OSError:
        fp = ""
    return {
        "expected_duration": expected_duration,
        "master_duration": master_duration,
        "master_fingerprint": fp,
        "source_bytes": _day_source_bytes(day_batch),
        # Encode-rule version — invalidates a cached master on any future
        # master-encode rule change (see MASTER_ENCODE_ALGO_VERSION).
        "encode_algo_version": MASTER_ENCODE_ALGO_VERSION,
        # Declared resampling upscale (source size, or None) and whether the
        # delivery may claim a native resolution.
        **_read_upscale(master_path),
    }


def verify_cached_master(
    master_path: Path,
    per_day_record: dict,
    *,
    probe_fn=None,
    tolerance_s: float = DURATION_INTEGRITY_TOLERANCE_S,
) -> bool:
    """Validate a cached master against recorded metadata (raw episodes gone).

    Validates against ``build_master_record`` metadata instead of the
    raw-episode ffprobe sum: the current ``size:mtime_ns`` fingerprint must
    equal the recorded one (catches any re-write) AND the current ffprobed
    duration must match the recorded ``master_duration`` within ``tolerance_s``
    (catches a truncated master). RELAXES "raw episodes must exist" WITHOUT
    relaxing the integrity guarantee — a mismatch RAISES
    ``BatchDurationIntegrityError``. Returns ``True`` on a clean validation OR a
    metadata-less record (written before fingerprints existed) so the caller falls back.

    A record whose ``encode_algo_version`` is absent or differs from the current
    ``MASTER_ENCODE_ALGO_VERSION`` (master encoded before a rule change — e.g.
    the attenuate-only audio change or the blur change, or any move in the frozen encode
    constants that version now fingerprints), OR whose metadata is PARTIAL
    (fingerprint present but ``master_duration`` never captured, from an
    encode-time ffprobe failure), is REFUSED — it RAISES so the operator
    re-encodes with ``--fresh`` instead of the stale master being stream-copied
    into every LF/short. Both gates run BEFORE any early return so the version
    stamp can't be skipped via a partial-metadata record.
    """
    recorded_fp = (per_day_record or {}).get("master_fingerprint") or ""
    recorded_duration = float((per_day_record or {}).get("master_duration", 0.0) or 0.0)
    recorded_version = (per_day_record or {}).get("encode_algo_version")
    if not recorded_fp:
        return True  # record without a fingerprint (metadata-less) — caller falls back to raw path

    # Encode-rule version gate — runs before the partial-metadata check so a
    # stale version can't slip through a duration<1.0 record.
    if recorded_version != MASTER_ENCODE_ALGO_VERSION:
        raise BatchDurationIntegrityError(
            f"Cached master {master_path} was encoded under encode-algo version "
            f"{recorded_version!r} != current {MASTER_ENCODE_ALGO_VERSION!r} (master "
            f"predates a master-encode rule change — its LFs/shorts would ship the "
            f"pre-fix artifact); re-encode with --fresh rather than reuse it."
        )
    # Partial metadata (fingerprint present but master_duration never
    # captured) can no longer early-return True and skip the fingerprint +
    # duration asserts — it is refused, not trusted.
    if recorded_duration < 1.0:
        raise BatchDurationIntegrityError(
            f"Cached master {master_path} has PARTIAL encode-time metadata "
            f"(master_duration={recorded_duration!r} < 1.0 from an ffprobe failure at "
            f"encode time); cannot validate the cached master — re-encode with --fresh."
        )

    if probe_fn is None:
        from .validation.ffprobe_client import FFprobeClient

        probe_fn = FFprobeClient(subprocess).probe_duration

    try:
        current_fp = master_fingerprint(master_path)
    except OSError as exc:
        raise BatchDurationIntegrityError(
            f"Cached master {master_path} could not be fingerprinted ({exc!r})"
        ) from exc
    if current_fp != recorded_fp:
        raise BatchDurationIntegrityError(
            f"Cached master {master_path} fingerprint changed since encode "
            f"(recorded={recorded_fp!r} current={current_fp!r}) — not the master "
            f"whose duration was validated; refusing to trust it."
        )

    try:
        actual = float(probe_fn(Path(master_path)))
    except Exception as exc:
        raise BatchDurationIntegrityError(
            f"Cached master {master_path} ffprobe failed ({exc!r})"
        ) from exc
    if abs(actual - recorded_duration) > tolerance_s:
        raise BatchDurationIntegrityError(
            f"Cached master {master_path} duration disagrees with recorded "
            f"metadata (actual={actual:.3f}s recorded={recorded_duration:.3f}s "
            f"tolerance={tolerance_s}s); re-encode rather than trust it."
        )
    return True


def _stable_cache_path(output_dir: Path, date: str) -> Path:
    """Path for the stable color cache file that survives pipeline restarts."""
    return output_dir / f"day_{date}_color_cache.json"


# Bumped 2->3: v3 adds ``grade_algo_version`` to the
# cache payload so a grade-algorithm change auto-invalidates the cache. The bump
# also clears any v2 cache written between the per-segment cache and
# the contrast_curve=0 dark-grade fix, which would otherwise replay
# the old dark S-curve on a same-source re-run.
# Bumped 3->4: v4 stores merged-span corrections +
# ``span_durations`` for per-span LUT baking; v3 files hold unmerged
# per-segment corrections that would explode the adaptive filtergraph.
COLOR_CACHE_SCHEMA_VERSION = 4


def _load_stable_color_cache(
    output_dir: Path,
    date: str,
    concat_paths: list,
) -> Optional[dict]:
    """Load stable color cache if it exists and source fingerprints match.

    The cache stores ``corrections`` (merged-span CorrectionParams as dicts)
    plus ``span_durations`` so the call site can rebuild per-span LUTs.
    A file whose version differs from ``COLOR_CACHE_SCHEMA_VERSION``
    invalidates cleanly and the next run re-analyzes.

    Returns the cached dict on hit, or None on miss (file missing,
    fingerprints changed, parse error, or another schema version).
    """
    cache_path = _stable_cache_path(output_dir, date)
    cached = load_cached_json(cache_path)
    if not cached:
        return None

    stored_fps = cached.get("source_fingerprints", [])
    current_fps = _source_fingerprints(concat_paths)
    if stored_fps != current_fps:
        logger.info(
            "  Day %s: color cache fingerprints changed, will re-analyze",
            date,
        )
        return None

    if cached.get("version") != COLOR_CACHE_SCHEMA_VERSION:
        logger.info(
            "  Day %s: color cache schema v%s != v%d, re-analyzing",
            date,
            cached.get("version"),
            COLOR_CACHE_SCHEMA_VERSION,
        )
        return None

    # Auto-invalidate when the grade algorithm itself changed since this cache
    # was written, even at the same schema version — the case that let the fixed
    # dark-grade replay from a stale cache.
    if cached.get("grade_algo_version") != GRADE_ALGO_VERSION:
        logger.info(
            "  Day %s: color grade algorithm changed (cache %s != %s), re-analyzing",
            date,
            cached.get("grade_algo_version"),
            GRADE_ALGO_VERSION,
        )
        return None

    if "corrections" not in cached or "method" not in cached:
        return None

    logger.info("  Day %s: stable color cache hit (%s)", date, cached["method"])
    return cached


def _preseed_transient_color_cache(
    raw_concat: Path,
    cached_data: dict,
) -> None:
    """Write the transient color_analysis.json alongside the raw concat so
    SinglePassProcessor._analyze_color finds a cache hit.

    Stamps the transient schema version the _analyze_color reader accepts
    (``TRANSIENT_COLOR_CACHE_SCHEMA_VERSION``) — NOT the stable day-cache's
    ``COLOR_CACHE_SCHEMA_VERSION``. The two schemas differ (the transient has
    no ``grade_algo_version``); stamping the wrong one made the reader reject
    every resume pre-seed and re-run the 35–90 min 4K analysis.

    v3 payload: ``corrections`` (list of CorrectionParams as dicts) +
    ``method`` + ``span_durations`` — matching what _analyze_color writes.
    """
    transient_path = raw_concat.parent / f"{raw_concat.stem}_color_analysis.json"
    save_cache_json(
        transient_path,
        {
            "version": TRANSIENT_COLOR_CACHE_SCHEMA_VERSION,
            "corrections": cached_data["corrections"],
            "method": cached_data["method"],
            "span_durations": cached_data.get("span_durations"),
            # The transient reader is keyed on the tonemap chain; omitting it
            # made every HDR-day resume miss and re-run the analysis.
            "tonemap": cached_data.get("tonemap", ""),
        },
    )
    logger.info("  Pre-seeded transient color cache at %s", transient_path.name)


def _save_stable_color_cache(
    output_dir: Path,
    date: str,
    raw_concat: Path,
    concat_paths: list,
) -> None:
    """After encoding, save the color analysis results to a stable cache.

    Reads the transient color_analysis.json that _analyze_color wrote next
    to the raw concat, and copies its data into the stable cache with source
    fingerprints attached. Written at ``COLOR_CACHE_SCHEMA_VERSION``.
    """
    transient_path = raw_concat.parent / f"{raw_concat.stem}_color_analysis.json"
    transient_data = load_cached_json(transient_path)
    if not transient_data or "corrections" not in transient_data:
        logger.debug("  Day %s: no transient color cache to persist", date)
        return

    stable_data = {
        "version": COLOR_CACHE_SCHEMA_VERSION,
        "grade_algo_version": GRADE_ALGO_VERSION,
        "corrections": transient_data["corrections"],
        "method": transient_data.get("method", "unknown"),
        "span_durations": transient_data.get("span_durations"),
        "tonemap": transient_data.get("tonemap", ""),
        "source_fingerprints": _source_fingerprints(concat_paths),
    }
    save_cache_json(_stable_cache_path(output_dir, date), stable_data)
    logger.info("  Day %s: saved stable color cache", date)


def _concat_raw_episodes(
    episode_paths: list,
    output_path: Path,
    work_dir: Path,
) -> bool:
    """Concatenate the day's raw episode files via ffmpeg concat demuxer
    (stream copy — no re-encode). Returns True on success.

    The concat list file is written to `work_dir` and referenced by the
    ffmpeg invocation via `-f concat -safe 0 -i list.txt`.
    """
    if not episode_paths:
        logger.warning("  _concat_raw_episodes called with no inputs")
        return False

    work_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Single-episode case: still use the concat demuxer for consistency
    # so downstream consumers see exactly one input format.
    list_file = work_dir / f"{output_path.stem}_concat_list.txt"
    list_lines = []
    for p in episode_paths:
        escaped = str(Path(p).resolve()).replace("'", "'\\''")
        list_lines.append(f"file '{escaped}'")
    list_file.write_text("\n".join(list_lines) + "\n")

    # No +faststart on the intermediate raw concat — it forces a full
    # file rewrite (doubling I/O) and the file is deleted after the
    # single-pass encode anyway. The final master gets +faststart via
    # SinglePassProcessor.
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-c",
        "copy",
        str(output_path),
    ]
    logger.info(
        "  Concatenating %d episodes → %s",
        len(episode_paths),
        output_path.name,
    )
    # 3600s ceiling — matches concat_episode and _split_episode_at_points
    # in pipeline.ingest.concatenator. The day-master raw concat reads
    # episode files from a cross-mount (network or drive-mount) path and
    # writes to the local-fs intermediate cache. A long multi-episode day
    # can reach ~200-220GB of source; a cross-mount read at ~100 MB/s
    # takes ~33-37 min, which would trip a shorter 1800s ceiling the same
    # way a short splitter ceiling can. The rationale sits alongside the
    # constant so future readers find it.
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=3600)
    except subprocess.TimeoutExpired:
        logger.error("  Day concat timed out after 60 min")
        return False
    except Exception as e:
        logger.error("  Day concat subprocess error: %s", e)
        return False

    if result.returncode != 0:
        logger.error(
            "  Day concat failed (rc=%d): %s",
            result.returncode,
            (result.stderr or b"").decode("utf-8", errors="replace")[-500:],
        )
        return False

    # Clean up the intermediate list file (not the concat output)
    try:
        list_file.unlink()
    except OSError:
        pass

    return True


# Schema for the mask-evidence sidecar (see `_build_day_blur_mask`). Bumping
# this version invalidates every cached mask on the next run — deliberate,
# the same way a source-fingerprint mismatch does.
_BLUR_EVIDENCE_SCHEMA = "gms.blur_mask_evidence/v2"

#: Schema of the publishable privacy attestation written beside the mask.
PRIVACY_ATTESTATION_SCHEMA = "gms.privacy_attestation/v1"
#: Its filename, one per day, beside the day's blur mask in the stage dir;
#: Phase D copies it beside each ``<project>_publish_plan.json`` sidecar,
#: which is how ``publishing/dispatch`` finds it from a delivery row.
PRIVACY_ATTESTATION_NAME = "day_{date}_privacy_attestation.json"

#: Every privacy class a publishable export must have VERIFIED (all four
#: are mandatory). A class switched OFF by config is not
#: thereby "not required" — it is unverified, and the attestation says so,
#: because a config flag must not be able to talk a required class away.
REQUIRED_PRIVACY_CLASSES = ("people", "faces", "plates", "vehicle_occupants")

#: Three states, not two. ``failed`` is "the detector ran and this class is
#: not covered"; ``inconclusive`` is "nothing ran, so nobody knows" — the
#: state a required-but-unresolvable model produces. Neither is a pass, and
#: collapsing them would let a missing model read like a clean detection.
PRIVACY_VERIFIED = "verified"
PRIVACY_FAILED = "failed"
PRIVACY_INCONCLUSIVE = "inconclusive"


def _blur_evidence_path(regions_sidecar: Path) -> Path:
    """Sidecar recording what privacy classes/models actually produced a
    cached mask, alongside (never inside) redactcam's own regions sidecar —
    redactcam's ``BlurTimeline`` schema is an external library contract we
    don't own and can't grow fields on."""
    return regions_sidecar.with_name(regions_sidecar.stem + "_evidence.json")


# Seconds of video between detection checkpoints on the day master. A day blur
# runs for hours; one GPU fault mid-pass otherwise discards all of it (one
# 132 GB day: ~8 h lost to a CUDA 999 at 54%). Five minutes of video bounds the
# loss to a few minutes of work while keeping snapshot writes rare.
BLUR_CHECKPOINT_EVERY_S = 300.0


def _blur_checkpoint_path(regions_sidecar: Path) -> Path:
    """redactcam's mid-pass detection checkpoint for this day."""
    return regions_sidecar.with_name(regions_sidecar.stem + "_detect_checkpoint.json")


def _blur_checkpoint_binding_path(regions_sidecar: Path) -> Path:
    """What the checkpoint was made under. redactcam only checks the source
    fingerprint, so a resume with a new model or detector setting would union
    stale boxes into the new pass; this sidecar is what refuses that."""
    return regions_sidecar.with_name(regions_sidecar.stem + "_detect_checkpoint_binding.json")


def _model_fingerprint(model_path) -> Optional[str]:
    """Cheap ``size:mtime`` identity of a resolved detector model file, or
    ``None`` when the model didn't resolve. Mirrors the concat fingerprint —
    good enough to detect "a different model file is now in play", not a
    tamper-proof hash."""
    if model_path is None:
        return None
    try:
        return file_fingerprint(Path(model_path))
    except OSError:
        return None


def _privacy_attestation_path(regions_sidecar: Path, date: str) -> Path:
    """Where this day's attestation lands: the run's output dir, beside the
    publish-plan sidecars publication already resolves per delivery row."""
    return regions_sidecar.with_name(PRIVACY_ATTESTATION_NAME.format(date=date))


def _stamp_master_fingerprint_into_attestation(
    output_dir: Path, date: str, master_path: Path
) -> None:
    """Record the PROMOTED master's identity INSIDE its own privacy attestation
    so a derived surface — LF, short, thumbnail — can prove it was cut
    from the SAME master the attestation covers, not a stale/different one
    (a resumed run with a changed mask, a hand-placed file, an older cached
    master). Best-effort and additive: runs on every successful return of
    ``encode_day_master`` (fresh encode AND the cached-master resume
    shortcut), no-ops when no attestation was written (``--skip-blur``), and
    never raises or gates — mirrors ``master_fingerprint``'s cheap
    ``size:mtime_ns`` identity, never a full hash of a 60 GB file."""
    regions_sidecar = Path(output_dir) / f"day_{date}_blur_regions.json"
    attestation_path = _privacy_attestation_path(regions_sidecar, date)
    payload = load_cached_json(attestation_path)
    if payload is None:
        return
    try:
        fp = master_fingerprint(master_path)
    except OSError:
        return
    payload["master_fingerprint"] = fp
    try:
        save_cache_json(attestation_path, payload)
    except OSError as exc:
        logger.warning(
            "  Day %s: failed to stamp master_fingerprint into attestation %s: %s",
            date,
            attestation_path,
            exc,
        )


#: Why every required class is recorded ``failed`` on a blur-bypassed day.
#: ``--skip-blur`` is the operator escape; ``blur_enabled: false`` in
#: config.yaml is the other way to the same place. ``blur_gate.blur_required``
#: is the single question, so both leave this same record.
SKIP_BLUR_REASON = (
    "blur was not required for this master (--skip-blur, or blur_enabled: false) "
    "-- no detector ran and the master shipped unblurred"
)


def write_skip_blur_attestation(output_dir: Path, date: str) -> None:
    """Write this day's attestation on the path where NO blur ran.

    ``_build_day_blur_mask`` was the file's only writer, and it is gated behind
    ``blur_gate.blur_required`` -- so a ``--skip-blur`` day produced no
    attestation at all. Publication was still held, because
    ``dispatch.privacy_hold_reason(None)`` is fail-closed; what was missing was
    the CAUSE. An absent file is indistinguishable from a crashed run, a
    permissions failure, or a directory nothing ever wrote to, and every
    derived surface's provenance cross-check (LF, short, thumbnail) had no
    document to bind to.

    **Records; never gates.** The publication outcome is identical before and
    after -- held either way. This is not a blur gate in disguise.

    Every required class is ``failed``, not ``inconclusive``. The distinction
    this module draws is that ``inconclusive`` means "nothing ran, so nobody
    knows"; here nothing ran but we are not in the dark -- blur was turned off
    deliberately, so the master is KNOWN to be uncovered, and the weaker state
    would understate a certainty.

    Never overwrites an existing document: a resume that reused a blurred
    master already carries the real detection record, and clobbering it with
    this one would turn a verified day into a failed one. Best-effort -- an
    unwritable output dir is logged, never raised, because by the time this
    runs the master is encoded and a record must not decide an outcome.
    """
    path = _privacy_attestation_path(Path(output_dir) / f"day_{date}_blur_regions.json", date)
    if load_cached_json(path) is not None:
        return
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        save_cache_json(
            path,
            {
                "schema": PRIVACY_ATTESTATION_SCHEMA,
                "date": date,
                "required_classes": list(REQUIRED_PRIVACY_CLASSES),
                "classes": {
                    name: {"state": PRIVACY_FAILED, "reason": SKIP_BLUR_REASON}
                    for name in REQUIRED_PRIVACY_CLASSES
                },
            },
        )
    except OSError as exc:
        logger.warning(
            "  Day %s: failed to write the skip-blur privacy attestation %s: %s",
            date,
            path,
            exc,
        )


def finalize_privacy_attestation(cfg, output_dir: Path, date: str, master_path: Path) -> None:
    """The one call every successful ``encode_day_master`` return makes.

    Both halves live behind a single name deliberately: when they were
    separate, a return path could stamp the master fingerprint and forget the
    document the stamp needs to live in -- which is exactly what the
    ``--skip-blur`` path did. Now a new return path cannot get one without the
    other. ``tests/test_skip_blur_privacy_attestation.py`` pins that no direct
    stamping call is left.
    """
    from . import blur_gate

    if not blur_gate.blur_required(cfg):
        write_skip_blur_attestation(output_dir, date)
    _stamp_master_fingerprint_into_attestation(output_dir, date, master_path)


def _detector_class(label: str, model) -> dict:
    """One required class covered by a detector model: verified when the model
    resolved, inconclusive when it did not. An unresolvable model is NOT a
    clean detection — nothing ran."""
    fingerprint = _model_fingerprint(model)
    if fingerprint is None:
        return {
            "state": PRIVACY_INCONCLUSIVE,
            "reason": f"{label} model did not resolve — no detector ran for this class",
        }
    return {
        "state": PRIVACY_VERIFIED,
        "reason": f"{label} model resolved and ran",
        "model_fingerprint": fingerprint,
    }


def _privacy_classes(
    drive,
    *,
    person_on: bool,
    person_model,
    vehicle_model,
    face_model,
    plate_model,
) -> dict:
    """Per required privacy class, which of the three states this run can
    honestly attest, and why. Consumed at PUBLICATION (the irreversible step),
    not here: a render costs GPU hours and a missing model file is fixed in
    seconds, so the encode proceeds and the master is retained."""
    classes = {
        "faces": _detector_class("face", face_model),
        "plates": _detector_class("plate", plate_model),
    }
    if person_on:
        classes["people"] = _detector_class("person", person_model)
    else:
        classes["people"] = {
            "state": PRIVACY_INCONCLUSIVE,
            "reason": (
                "person blur is off (blur_person_enabled=false); the class is "
                "required regardless, so this run cannot attest it"
            ),
        }
    if drive.blur_vehicle_cabin_frac > 0:
        classes["vehicle_occupants"] = _detector_class("vehicle", vehicle_model)
    else:
        classes["vehicle_occupants"] = {
            "state": PRIVACY_INCONCLUSIVE,
            "reason": (
                "occupant cabin blur is off (blur_vehicle_cabin_frac=0); the "
                "class is required regardless, so this run cannot attest it"
            ),
        }
    return classes


def _current_blur_evidence(
    drive,
    *,
    person_on: bool,
    person_model,
    vehicle_model,
    face_model,
    plate_model,
) -> dict:
    """What this run REQUIRES and would actually detect — the evidence a
    cached mask must match to be legally reused. Binds the required privacy
    classes and each detector model's identity, not just the source."""
    return {
        "schema": _BLUR_EVIDENCE_SCHEMA,
        "vehicle_required": drive.blur_vehicle_cabin_frac > 0,
        "vehicle_model_fingerprint": _model_fingerprint(vehicle_model),
        "person_required": person_on,
        # What would actually be tracked, not just the config flag — a
        # required-but-unresolvable person model degrades to WARN (faces
        # only), and that degraded mask must not pass as "person handled".
        "person_tracked": person_on and person_model is not None,
        "person_model_fingerprint": _model_fingerprint(person_model),
        "face_model_fingerprint": _model_fingerprint(face_model),
        "plate_model_fingerprint": _model_fingerprint(plate_model),
        # The same facts, shaped as the per-class attestation publication
        # reads. Inside the evidence dict so a change in what a run can
        # attest also invalidates the cached mask, exactly like a model
        # fingerprint change does.
        "privacy_classes": _privacy_classes(
            drive,
            person_on=person_on,
            person_model=person_model,
            vehicle_model=vehicle_model,
            face_model=face_model,
            plate_model=plate_model,
        ),
    }


def _build_day_blur_mask(
    raw_concat: Path,
    config: Config,
    mask_path: Path,
    regions_sidecar: Path,
    date: str,
) -> Path:
    """Detect faces ∪ plates ∪ vehicle-cabins ∪ persons on the day's raw concat and
    materialize the frame-accurate privacy mask the single-pass encode composites — so
    the day master (and every LF / short / thumbnail stream-cut from it) ships
    anonymized BY CONSTRUCTION, mirroring the single-video ``BlurDetectionStage`` →
    ``single_pass`` flow. Keyed on a cheap ``size:mtime`` fingerprint of the concat (a
    full sha of a 100 GB file is prohibitive) so a resume killed AFTER detection but
    before the encode reuses the mask instead of re-detecting a multi-hour master —
    AND on the ``_blur_evidence_path`` sidecar recording which privacy classes and
    detector models actually produced it, so a source-fingerprint match alone never
    authorizes reuse when this run's requirements (e.g. person blur newly resolvable)
    differ from what was recorded.

    Raises ``BlurGateError`` on ANY detection failure, or when the occupant cabin blur
    is on but its vehicle model can't resolve — never a silent fall-through to an
    unblurred encode. Person-blur model resolution does NOT raise here: the encode
    proceeds (faces still ship blurred) and the master is retained, because a render
    costs GPU hours and a missing model file is fixed in seconds. What it does instead
    is record ``people`` as ``inconclusive`` in the privacy attestation written beside
    the mask, which HOLDS every short cut from this master at publication — the
    irreversible step — until the model is restored and the day is re-run.
    """
    from redactcam import detect_and_track, materialize_mask_video
    from redactcam import timeline as br

    from ..analysis import (
        resolve_face_model,
        resolve_person_model,
        resolve_plate_model,
        resolve_vehicle_model,
    )

    drive = config.drive
    try:
        fingerprint = file_fingerprint(raw_concat)
    except OSError:
        fingerprint = ""

    # Occupant cabin blur is a signed-off privacy feature: enabled-but-unresolvable
    # means drivers would ship UNBLURRED — hard-fail rather than silently disable.
    vehicle_model = resolve_vehicle_model(drive)
    if drive.blur_vehicle_cabin_frac > 0 and vehicle_model is None:
        raise blur_gate.BlurGateError(
            "BLUR GATE: occupant cabin blur is on (blur_vehicle_cabin_frac>0) but the "
            "vehicle model could not be resolved — drivers would ship unblurred. Set "
            "vehicle_model_url/path, pre-cache the model, or disable with "
            "blur_vehicle_cabin_frac=0 (or --skip-blur)."
        )
    person_on = bool(getattr(drive, "blur_person_enabled", False))
    person_model = resolve_person_model(drive) if person_on else None
    face_model = resolve_face_model(drive)
    plate_model = resolve_plate_model(drive)
    current_evidence = _current_blur_evidence(
        drive,
        person_on=person_on,
        person_model=person_model,
        vehicle_model=vehicle_model,
        face_model=face_model,
        plate_model=plate_model,
    )

    # Written on EVERY path — fresh detection and cache reuse alike — because
    # this file, not the mask, is what publication reads. A resume that reused
    # the cached mask must not leave the publish gate with nothing to read.
    regions_sidecar.parent.mkdir(parents=True, exist_ok=True)
    save_cache_json(
        _privacy_attestation_path(regions_sidecar, date),
        {
            "schema": PRIVACY_ATTESTATION_SCHEMA,
            "date": date,
            "required_classes": list(REQUIRED_PRIVACY_CLASSES),
            "classes": current_evidence["privacy_classes"],
        },
    )

    existing = br.load_sidecar(regions_sidecar)
    existing_evidence = load_cached_json(_blur_evidence_path(regions_sidecar))
    if (
        existing is not None
        and fingerprint
        and existing.source_hash == fingerprint
        and existing_evidence == current_evidence
        and mask_path.exists()
        and mask_path.stat().st_size > 0
    ):
        logger.info(
            "  Day %s: reusing blur mask %s (concat fingerprint + evidence match)",
            date,
            mask_path.name,
        )
        return mask_path

    detect_kwargs = dict(
        sample_fps=drive.blur_sample_fps,
        conf=drive.blur_conf,
        face_conf=drive.blur_face_conf,
        face_sample_fps=drive.blur_face_sample_fps,
        face_confirm_min_detections=drive.blur_face_confirm_min_detections,
        pre_roll_frames=drive.blur_pre_roll_frames,
        plate_model_path=plate_model,
        face_model_path=face_model,
        tile_px=drive.blur_tile_px,
        tile_overlap=drive.blur_tile_overlap,
        plate_min_aspect=drive.blur_plate_min_aspect,
        plate_max_aspect=drive.blur_plate_max_aspect,
        plate_max_area_frac=drive.blur_plate_max_area_frac,
        plate_boost_speed_px=drive.blur_plate_boost_speed_px,
        plate_boost_max_run=drive.blur_plate_boost_max_run,
        track_face_margin_frac=drive.blur_face_margin_frac,
        track_face_motion_gain=drive.blur_face_motion_gain,
        track_cabin_frac=drive.blur_vehicle_cabin_frac,
        vehicle_model_path=vehicle_model,
        vehicle_conf=drive.blur_vehicle_conf,
        vehicle_min_height_frac=drive.blur_vehicle_min_height_frac,
        vehicle_ignore_bottom_frac=drive.blur_vehicle_ignore_bottom_frac,
        track_person=person_on and person_model is not None,
        person_model_path=person_model,
        person_conf=getattr(drive, "blur_person_conf", 0.35),
        person_min_height_frac=getattr(drive, "blur_person_min_height_frac", 0.0),
    )
    checkpoint_path = _blur_checkpoint_path(regions_sidecar)
    binding_path = _blur_checkpoint_binding_path(regions_sidecar)
    binding = {
        "evidence": current_evidence,
        "detect": {k: str(v) if isinstance(v, Path) else v for k, v in detect_kwargs.items()},
    }
    if load_cached_json(binding_path) != binding:
        checkpoint_path.unlink(missing_ok=True)
        save_cache_json(binding_path, binding)

    try:
        fps, frame_count, w, h = br.probe_video(raw_concat)
        frame_boxes, decoded_count, n_faces, n_plates = detect_and_track(
            raw_concat,
            **detect_kwargs,
            checkpoint_path=checkpoint_path,
            checkpoint_every_frames=max(1, math.ceil(fps * BLUR_CHECKPOINT_EVERY_S)),
        )
        # detect_and_track leaves frames between samples with no entry at all,
        # and the renderer reads "no entry" as "nothing to blur". Fill those
        # gaps before the timeline is built — see fill_detection_gaps.
        # A detector miss must not unmask an object still in frame: hold it per object.
        frame_boxes = blur_gate.hold_missed_detections(
            frame_boxes,
            sample_interval_frames=math.ceil(fps / max(drive.blur_sample_fps, 1e-6)),
        )
        frame_boxes = blur_gate.fill_detection_gaps(
            frame_boxes,
            max_gap_frames=math.ceil(fps / max(drive.blur_sample_fps, 1e-6)) + 1,
        )
        # A track's first detected frame is not its first VISIBLE frame; the
        # frames just before it are what ship unmasked. Runs after the gap
        # fill, so only births that survive interpolation are extended.
        frame_boxes = blur_gate.backfill_track_births(
            frame_boxes, pre_roll_frames=drive.blur_pre_roll_frames
        )
        # The decode loop counted every frame — authoritative over probe metadata.
        frame_count = decoded_count or frame_count
        timeline = br.build_timeline(
            frame_boxes,
            source_hash=fingerprint,
            source_fps=fps,
            source_frame_count=frame_count,
            frame_width=w or 0,
            frame_height=h or 0,
            box_dilation_px=drive.blur_box_dilation_px,
            sample_fps=drive.blur_sample_fps,
            conf=drive.blur_conf,
        )
        regions_sidecar.parent.mkdir(parents=True, exist_ok=True)
        br.write_sidecar(regions_sidecar, timeline)
        save_cache_json(_blur_evidence_path(regions_sidecar), current_evidence)
        materialize_mask_video(timeline, mask_path)
        # The finished mask is reused through the sidecar from here on.
        checkpoint_path.unlink(missing_ok=True)
        binding_path.unlink(missing_ok=True)
    except blur_gate.BlurGateError:
        raise
    except Exception as exc:
        raise blur_gate.BlurGateError(
            f"BLUR GATE: privacy detection failed for day {date} master ({exc}) — "
            "refusing to encode an unblurred master. Fix the detector/model and re-run "
            "--fresh, or bypass with --skip-blur."
        ) from exc

    logger.info(
        "  Day %s: blur mask built (%d face + %d plate detection(s)) → %s",
        date,
        n_faces,
        n_plates,
        mask_path.name,
    )
    return mask_path


# Seconds of video either side of a leaked frame that the cabin repair re-detects.
# The gate samples ~4 fps, so the frames between its samples carry the same exposed
# driver; an opposing car crosses the frame in about this long.
CABIN_REPAIR_WINDOW_S = 2.0
# Frames either side a repaired cabin box is also held for, so a frame the detector
# drops between two it caught still gets blurred.
CABIN_REPAIR_HOLD_FRAMES = 2


def _leak_windows(frames: list[int], half: int, frame_count: int) -> list[tuple[int, int]]:
    """Merged ``[lo, hi]`` frame windows ``half`` frames either side of each leak."""
    last = frame_count - 1 if frame_count > 0 else None
    windows: list[tuple[int, int]] = []
    for f in sorted(set(frames)):
        lo, hi = max(0, f - half), f + half
        if last is not None:
            hi = min(last, hi)
        if lo > hi:
            continue
        if windows and lo <= windows[-1][1] + 1:
            windows[-1] = (windows[-1][0], max(windows[-1][1], hi))
        else:
            windows.append((lo, hi))
    return windows


def _repair_cabin_leaks(
    raw_concat: Path, config: Config, mask_path: Path, regions_sidecar: Path
) -> int:
    """Add the cabin of every vehicle seen near each frame the cabin gate flagged,
    rebuild the mask, and return how many boxes were added (0 = nothing changed).

    The detection tracker blurs a car's cabin only once a plate track corroborates
    it, while the gate corroborates frame by frame, so the gate can see a driver the
    mask never covered. Re-detecting the whole day costs hours and repeats the same
    miss; this re-detects only ``CABIN_REPAIR_WINDOW_S`` around each leak, without
    the plate requirement (a few extra blurred billboards beat one exposed driver).

    Frames are read in order, never by seeking, so the frame numbers match the
    gate's. The new mask is written beside the old one and swapped in only once
    complete: the mask cache reuses whatever file is present, so a half-written one
    must never take its place. Any failure leaves the old mask and returns 0 —
    the caller then stops on the gate's original error."""
    import cv2
    from redactcam import materialize_mask_video
    from redactcam import timeline as br
    from redactcam.coverage import cabin_region
    from redactcam.detect import build_vehicle_detector, filter_vehicles

    from ..analysis import resolve_vehicle_model

    report = load_cached_json(blur_gate.cabin_leak_report_path(mask_path)) or {}
    leak_frames = [int(leak["frame"]) for leak in report.get("leaks", [])]
    if not leak_frames:
        return 0
    timeline = br.load_sidecar(regions_sidecar)
    drive = config.drive
    veh = resolve_vehicle_model(drive)
    if timeline is None or veh is None:
        return 0
    n = timeline.source_frame_count
    windows = _leak_windows(leak_frames, math.ceil(timeline.source_fps * CABIN_REPAIR_WINDOW_S), n)
    tmp_mask = mask_path.with_name(f"{mask_path.stem}.repair.tmp{mask_path.suffix}")
    tmp_sidecar = regions_sidecar.with_name(f"{regions_sidecar.stem}.repair.tmp.json")
    added = 0
    try:
        det = build_vehicle_detector(str(veh), 640, 0.45, 0)
        if det is None:
            return 0
        cap = cv2.VideoCapture(str(raw_concat))
        try:
            wi, idx = 0, -1
            while idx < windows[-1][1] and cap.grab():
                idx += 1
                while idx > windows[wi][1]:
                    wi += 1
                if idx < windows[wi][0]:
                    continue
                ok, frame = cap.retrieve()
                if not ok:
                    continue
                h, w = frame.shape[:2]
                for bx in filter_vehicles(
                    det.detect(frame, drive.blur_vehicle_conf),
                    h,
                    drive.blur_vehicle_min_height_frac,
                    drive.blur_vehicle_ignore_bottom_frac,
                ):
                    x0, y0, x1, y1 = cabin_region(bx, drive.blur_vehicle_cabin_frac, w, h)
                    if x1 <= x0 or y1 <= y0:
                        continue
                    box = br.dilate_box((x0, y0, x1 - x0, y1 - y0), timeline.box_dilation_px, w, h)
                    px0, py0, px1, py1 = blur_gate.plate_band_region(bx, w, h)
                    band = br.dilate_box(
                        (px0, py0, px1 - px0, py1 - py0), timeline.box_dilation_px, w, h
                    )
                    lo = max(0, idx - CABIN_REPAIR_HOLD_FRAMES)
                    hi = idx + CABIN_REPAIR_HOLD_FRAMES
                    for f in range(lo, (min(hi, n - 1) if n > 0 else hi) + 1):
                        boxes = timeline.frame_boxes.setdefault(f, [])
                        for new_box in (box, band):
                            if new_box not in boxes:
                                boxes.append(new_box)
                                added += 1
        finally:
            cap.release()
        if not added:
            return 0
        materialize_mask_video(timeline, tmp_mask)
        br.write_sidecar(tmp_sidecar, timeline)
        os.replace(tmp_mask, mask_path)
        os.replace(tmp_sidecar, regions_sidecar)
    except Exception as exc:
        logger.warning("cabin leak repair failed, keeping the original mask: %s", exc)
        return 0
    finally:
        tmp_mask.unlink(missing_ok=True)
        tmp_sidecar.unlink(missing_ok=True)
    blur_gate.cabin_verdict_path(mask_path).unlink(missing_ok=True)
    logger.info(
        "  cabin repair: %d box(es) added around %d leaked frame(s) → %s",
        added,
        len(set(leak_frames)),
        mask_path.name,
    )
    return added


def encode_day_master(
    day_batch,
    output_dir: Path,
    config: Optional[Config] = None,
    location: str = "",
    per_day_record: Optional[dict] = None,
    profile: Optional[object] = None,
) -> Optional[Path]:
    """Encode a day's master video file from a DayBatch.

    Steps:
      1. Concat the day's raw episode files (stream copy).
      2. Run SinglePassProcessor with color + audio in one pass (no OST;
         titles are burned later on each final).
      3. Clean up the raw concat intermediate.

    Args:
        day_batch: A DayBatch produced by `day_grouper.group_episodes_by_day`.
        output_dir: Directory where the master file will be written.
        config: Optional `gomoveshift.config.Config` — defaults to a fresh
            Config() if not provided.

    Returns:
        Path to the encoded `day_{date}_master.mp4` on success,
        or None on any failure (empty batch, concat failure, encode
        failure). No partial files are left behind on failure.

    Raises:
        BatchDurationIntegrityError: when the freshly-promoted master's own
            duration disagrees with ``day_batch.total_duration``. Checked HERE,
            immediately after the atomic promotion, so no LF / short /
            superlong part is ever cut from a truncated master.
    """
    # Lazy import to avoid a hard dependency cycle during module load
    from .single_pass import SinglePassProcessor

    episodes = list(getattr(day_batch, "episodes", None) or [])
    if not episodes:
        logger.info(
            "  Day %s: no episodes, skipping master encode", getattr(day_batch, "date", "?")
        )
        return None

    date = getattr(day_batch, "date", "unknown")
    concat_paths = [
        ep.concat_path for ep in episodes if getattr(ep, "concat_path", None) is not None
    ]
    if not concat_paths:
        logger.warning(
            "  Day %s: no concat_paths on any episode, cannot build master",
            date,
        )
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = config or Config()
    # Config-driven ffprobe timeout: under heavy concurrent encode load a
    # valid master/raw probe can exceed the 30s default. Resolve once here so
    # both resume-shortcut probes below honour the operator's floor.
    _ffprobe_timeout = cfg.drive.ffprobe_timeout_sec

    raw_concat = output_dir / f"day_{date}_raw.mp4"
    master_path = output_dir / f"day_{date}_master.mp4"
    work_dir = output_dir / f".day_{date}_work"

    # Resume shortcut: if a complete master already exists for this day,
    # skip the concat + single-pass re-encode. phase_bc_day_first has no
    # per-day granularity in its completion state, so on resume it walks
    # every day batch unconditionally — without this check we'd waste ~8h
    # per day re-encoding content we already have (and would overwrite it).
    expected_duration = float(getattr(day_batch, "total_duration", 0.0) or 0.0)
    if master_path.exists() and expected_duration >= 1.0:
        # Independent ground-truth check BEFORE the 99%-threshold compare.
        # When the raw episodes are ABSENT but the per-day record carries
        # encode-time metadata, validate the master against THAT (so the raw
        # episodes can be purged) instead of probing absent episodes. Otherwise
        # (episodes present, OR a record without a fingerprint) keep the raw-episode ffprobe
        # sum so a partial master can't pass the 99 % gate against an
        # under-reported ``batch.total_duration``. See ``verify_cached_master``.
        raw_episodes_present = all(Path(p).exists() for p in concat_paths)
        # Only when episodes are gone AND an integrity record exists do we validate the
        # master against recorded metadata (verify_cached_master RAISES on
        # mismatch / returns True on match); else fall to the raw-episode path.
        has_meta = bool((per_day_record or {}).get("master_fingerprint"))
        use_record = not raw_episodes_present and has_meta
        used_recorded_metadata = use_record and verify_cached_master(master_path, per_day_record)
        if used_recorded_metadata:
            logger.info(
                "  Day %s: validated cached master against recorded metadata "
                "(raw episodes absent) — skipping re-encode",
                date,
            )
        else:
            assert_batch_duration_integrity(day_batch, concat_paths)

        from .validation.ffprobe_client import FFprobeClient

        try:
            actual = FFprobeClient(subprocess, timeout=_ffprobe_timeout).probe_duration(master_path)
        except Exception:
            actual = 0.0
        if actual >= expected_duration * 0.99:
            # BLUR GATE (drive path): a blur-enabled run must never REUSE a cached
            # master that predates blur — its LFs/shorts/thumbnails would ship
            # unblurred. Raise upstream of the LF/short cuts (and so upstream of any
            # upload); no-op under --skip-blur or a marked (blurred) master.
            blur_gate.require_master_blurred(cfg, master_path)
            logger.info(
                "  Day %s: existing master is complete (%.1fs ≥ %.1fs "
                "expected), skipping re-encode",
                date,
                actual,
                expected_duration,
            )
            finalize_privacy_attestation(cfg, output_dir, date, master_path)
            return master_path
        logger.info(
            "  Day %s: existing master at %s is incomplete (%.1fs < %.1fs expected), re-encoding",
            date,
            master_path,
            actual,
            expected_duration,
        )
    elif master_path.exists() and expected_duration < 1.0:
        logger.warning(
            "  Day %s: expected_duration is %.2fs (missing/zero), "
            "cannot verify existing master — forcing re-encode",
            date,
            expected_duration,
        )

    # Preflight disk space check: require what is still to be written — the
    # raw concat (~source size) plus the master. A raw already on disk (a resume
    # after a crash) is reused below or freed when the concat rewrites it, so
    # only its missing bytes count. The master is sized at
    # MASTER_BYTES_PER_SOURCE_BYTE, an upper bound on measured masters.
    source_bytes = 0
    for p in concat_paths:
        try:
            source_bytes += Path(p).stat().st_size
        except OSError:
            pass
    if source_bytes > 0:
        try:
            raw_on_disk = min(raw_concat.stat().st_size, source_bytes)
        except OSError:
            raw_on_disk = 0
        required_bytes = (source_bytes - raw_on_disk) + int(
            source_bytes * MASTER_BYTES_PER_SOURCE_BYTE
        )
        check_disk_space(output_dir, required_bytes)

    # Resume shortcut for the raw concat: a WSL or encoder crash landing
    # between concat and single-pass leaves the 100+ GB raw on disk with
    # no master. Re-concatenating it costs ~20-30 min of pure I/O — skip
    # when the existing raw is intact (≥99% of expected duration).
    skip_concat = False
    if raw_concat.exists() and expected_duration >= 1.0:
        from .validation.ffprobe_client import FFprobeClient

        try:
            actual_raw = FFprobeClient(subprocess, timeout=_ffprobe_timeout).probe_duration(
                raw_concat
            )
        except Exception:
            actual_raw = 0.0
        if actual_raw >= expected_duration * 0.99:
            logger.info(
                "  Day %s: existing raw concat is complete (%.1fs ≥ %.1fs "
                "expected), skipping re-concat",
                date,
                actual_raw,
                expected_duration,
            )
            skip_concat = True
        else:
            logger.info(
                "  Day %s: existing raw concat at %s is incomplete "
                "(%.1fs < %.1fs expected), re-concatenating",
                date,
                raw_concat,
                actual_raw,
                expected_duration,
            )

    # Step 1: Concat raw episodes
    if not skip_concat and not _concat_raw_episodes(concat_paths, raw_concat, work_dir):
        # Clean up partial work directory and raw concat
        if raw_concat.exists():
            try:
                raw_concat.unlink()
            except OSError:
                pass
        return None

    # Step 1.4: opt-in picture steps (--deflicker / --stabilize), measured. They run on
    # the raw concat BEFORE the grade, blur mask and encode so all three see the same
    # frames. Intermediates live next to the raw concat under ``output_dir``. A
    # discarded / failed step falls back to the raw concat; never raises.
    graded_src = raw_concat
    if cfg.drive.deflicker or cfg.drive.stabilize:
        from .stabilize import apply_picture_steps

        steps = apply_picture_steps(
            raw_concat,
            output_dir,
            f"day_{date}_raw",
            deflicker=bool(cfg.drive.deflicker),
            stabilize=bool(cfg.drive.stabilize),
        )
        graded_src = steps.path
        try:
            (output_dir / f"day_{date}_picture_steps.json").write_text(
                json.dumps(steps.record, indent=2, sort_keys=True)
            )
        except OSError as e:
            logger.warning("  Day %s: could not write picture-steps record: %s", date, e)
    picture_changed = graded_src != raw_concat

    # Step 1.5: Pre-seed color cache from stable cache if available.
    # The raw concat is freshly created (new mtime), which would
    # invalidate the mtime-based color cache that _analyze_color
    # maintains. By pre-seeding the transient cache with verified
    # stable data, we skip the ~35 min color re-analysis on restart.
    stable_color = _load_stable_color_cache(output_dir, date, concat_paths)
    if stable_color and not picture_changed:
        _preseed_transient_color_cache(raw_concat, stable_color)

    # Step 2: Color + audio encode in a single pass. OST is deliberately NOT
    # burned on the day master (skip_ost=True): the master is never uploaded —
    # it is stream-cut into LF segments AND shorts — so a master-level overlay
    # is wasted work and can bleed into a clip whose source window overlaps the
    # master's burned region. The on-screen title is applied post-assembly by
    # ost_applier.apply_ost (LFs) and ffmpeg_filters.build_ost_filter (shorts),
    # each on its own final timeline. (The old skip_ost=not bool(location) burned
    # OST whenever a day-wide location was set; apply_ost re-encodes every LF
    # regardless, so the "avoid a wasteful re-encode" rationale no longer holds.)
    #
    # Atomic-rename pattern:
    # encode to a `.tmp` sidecar and only `os.replace` it onto the final
    # `master_path` after `SinglePassProcessor.process` returns success.
    # Without this, a SIGKILL or power loss mid-encode leaves a corrupt
    # mp4 (no `moov` atom) at the final filename, and the resume fast-path
    # silently trusts it via `Path.exists()`.
    processor = SinglePassProcessor(cfg, profile=profile)
    # ── Privacy blur (drive path) ─────────────────────────────────────
    # Detect faces/plates/occupants/persons on the raw concat and composite the
    # blur DURING this single existing encode pass (color+audio already re-encode
    # the whole day here) — so the master, and every LF/short/thumbnail cut from
    # it, is blurred by construction. Detection failure / a missing cabin model
    # RAISES BlurGateError (propagated by run_all_days), never a silent unblurred
    # ship. No-op under --skip-blur.
    blur_mask_path: Optional[Path] = None
    if blur_gate.blur_required(cfg):
        regions_sidecar = output_dir / f"day_{date}_blur_regions.json"
        blur_mask_path = _build_day_blur_mask(
            graded_src,
            cfg,
            output_dir / f"day_{date}_blur_mask.mkv",
            regions_sidecar,
            date,
        )
        # Occupant-coverage gate (drive path): a corroborated car whose cabin
        # the mask misses would ship its driver exposed — verify against the mask
        # BEFORE the irreversible master encode composites it, mirroring the
        # legacy per-episode gate (blur_gate.check_cabin_coverage). A failure gets
        # one targeted repair and a fresh check of the whole day; only a leak that
        # survives the repair stops the run.
        cabin_err = blur_gate.check_day_master_cabin_coverage(cfg, graded_src, blur_mask_path)
        if cabin_err and _repair_cabin_leaks(graded_src, cfg, blur_mask_path, regions_sidecar):
            cabin_err = blur_gate.check_day_master_cabin_coverage(cfg, graded_src, blur_mask_path)
        if cabin_err:
            raise blur_gate.BlurGateError(cabin_err)
    # Use ``<stem>.tmp<suffix>`` (e.g. ``day_2025-01-15_master.tmp.mp4``) rather
    # than ``<name><suffix>.tmp`` so the real container extension stays at the
    # tail. ffmpeg infers the output muxer from the trailing extension; with
    # ``...mp4.tmp`` it raised
    # ``Unable to choose an output format for '....mp4.tmp'`` and aborted the
    # day master encode.
    tmp_master_path = master_path.parent / f"{master_path.stem}.tmp{master_path.suffix}"
    # Clear any stale .tmp.<ext> from a prior killed run before we start writing.
    # `os.replace` later in this function will also overwrite it, but the
    # encoder may behave unpredictably when handed an existing output path.
    if tmp_master_path.exists():
        try:
            tmp_master_path.unlink()
        except OSError as e:
            logger.debug("  Could not unlink stale tmp master %s: %s", tmp_master_path, e)
    try:
        result = processor.process(
            video_path=graded_src,
            output_path=tmp_master_path,
            location=location,
            skip_ost=True,
            # --no-audio: encode the day master with NO audio stream (-an) so
            # every LF stream-cut from it ships truly silent. Default (attenuate
            # -only near-silent) keeps faint ambience; this is the "fully silent
            # LF too" switch. Shorts are always muted regardless.
            skip_audio=bool(cfg.drive.skip_audio),
            blur_mask_path=blur_mask_path,
        )
    except Exception as e:
        logger.error("  Day %s master encode raised: %s", date, e)
        # Clean up partial tmp master, raw concat on failure
        for partial in (tmp_master_path, raw_concat, graded_src):
            if partial.exists():
                try:
                    partial.unlink()
                except OSError:
                    pass
        return None

    if getattr(result, "tonemap", ""):
        logger.info("  Day %s master HDR tonemap: %s", date, result.tonemap)

    # Step 2.5: Persist color analysis to stable cache so it survives
    # pipeline restarts (the transient cache dies with the raw concat).
    if not picture_changed:
        _save_stable_color_cache(output_dir, date, raw_concat, concat_paths)

    # Step 3: Clean up the intermediate raw concat file regardless of
    # success so we don't leave ~multi-GB scratch files behind.
    for scratch in (raw_concat, graded_src):
        if scratch.exists():
            try:
                scratch.unlink()
            except OSError as e:
                logger.debug("  Could not unlink raw concat %s: %s", scratch, e)

    # Clean up the work directory if it's empty
    try:
        if work_dir.exists() and not any(work_dir.iterdir()):
            work_dir.rmdir()
    except OSError:
        pass

    if not result or not getattr(result, "success", False):
        # Encode failure — remove any partial tmp file. The final
        # `master_path` is never written to until success, so we only
        # need to scrub the .tmp here.
        if tmp_master_path.exists():
            try:
                tmp_master_path.unlink()
            except OSError:
                pass
        return None

    # Atomic promotion: only now does the final filename appear on disk.
    # `os.replace` is atomic on POSIX when src/dst are on the same
    # filesystem (which they are — both under output_dir).
    try:
        os.replace(tmp_master_path, master_path)
    except OSError as e:
        logger.error(
            "  Day %s: failed to promote %s → %s: %s",
            date,
            tmp_master_path,
            master_path,
            e,
        )
        if tmp_master_path.exists():
            try:
                tmp_master_path.unlink()
            except OSError:
                pass
        return None

    _record_upscale(master_path, getattr(result, "upscale_plan", None), profile)

    # UPSTREAM duration-integrity gate: the master exists at its final path and
    # NOTHING has been derived from it yet, so this is the earliest point the
    # check can run — and the cheapest place to fail (one ffprobe). It used to
    # run only from ``build_master_record`` at the tail of ``run_day_pipeline``,
    # i.e. after every LF/OST burn, short and superlong part had already been
    # NVENC-encoded from a master that was already known-bad by this line.
    # RAISES ``BatchDurationIntegrityError``; same comparison, same tolerance.
    from .validation.ffprobe_client import FFprobeClient

    try:
        promoted_duration = float(
            FFprobeClient(subprocess, timeout=_ffprobe_timeout).probe_duration(master_path)
        )
    except Exception:
        promoted_duration = 0.0
    assert_master_duration_integrity(date, expected_duration, promoted_duration)

    # Stamp the durable blur marker so a later resume can PROVE this master was
    # encoded with the privacy composite (blur_gate.require_master_blurred). A
    # blur-required run that reaches here always ran the composite above.
    if blur_gate.blur_required(cfg):
        blur_gate.mark_applied(master_path)

    finalize_privacy_attestation(cfg, output_dir, date, master_path)

    logger.info("  Day %s master: %s", date, master_path)
    return master_path
