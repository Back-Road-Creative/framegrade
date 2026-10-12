"""
Pipeline Cache — skip expensive operations when outputs are already valid.

Uses the classic `make` pattern: if output exists and is newer than input,
skip the operation. Also supports content-hash validation for critical outputs.

All pipeline stages should call `is_output_fresh()` before doing expensive work.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# Schema version for the transient per-video color-analysis cache
# (``{stem}_color_analysis.json``) that SinglePassProcessor._analyze_color
# writes and reads, and that day_master_encoder pre-seeds on resume. Reader,
# writer, and preseed all reference this ONE constant so they can't drift — a
# mismatch silently forces a 35–90 min 4K color re-analysis on every restart.
# (Distinct from the stable day-cache's COLOR_CACHE_SCHEMA_VERSION, which
# carries the grade-algorithm fingerprint the transient schema does not.)
# Bumped 2->3: v3 stores merged-span ``corrections`` +
# ``span_durations`` for per-span LUT baking. v2 files hold UNMERGED
# per-segment corrections (hundreds for a day master) that would explode the
# adaptive filtergraph, so they must invalidate and re-analyze once.
TRANSIENT_COLOR_CACHE_SCHEMA_VERSION = 3


def is_output_fresh(
    input_path: Path,
    output_path: Path,
    min_size_bytes: int = 1024,
) -> bool:
    """
    Check if an output file is fresh (doesn't need regeneration).

    Fresh means:
      1. Output exists
      2. Output is non-empty (above min_size_bytes)
      3. Output is newer than input (mtime comparison)

    Args:
        input_path: The source/input file
        output_path: The expected output file
        min_size_bytes: Minimum valid file size (default 1KB)

    Returns:
        True if output is fresh and can be reused
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not output_path.exists():
        return False

    output_size = output_path.stat().st_size
    if output_size < min_size_bytes:
        logger.debug(f"  Cache miss: {output_path.name} too small ({output_size} bytes)")
        return False

    if not input_path.exists():
        # Input gone but output exists — trust it
        return True

    input_mtime = input_path.stat().st_mtime
    output_mtime = output_path.stat().st_mtime

    if output_mtime >= input_mtime:
        logger.info(f"  Cache hit: {output_path.name} is fresh (skipping)")
        return True

    logger.debug(f"  Cache miss: {output_path.name} older than input")
    return False


def are_outputs_fresh(
    input_path: Path,
    output_paths: list[Path],
    min_size_bytes: int = 1024,
) -> bool:
    """
    Check if ALL output files are fresh relative to input.

    Useful for stages that produce multiple files (e.g., thumbnails, shorts).
    """
    if not output_paths:
        return False

    return all(is_output_fresh(input_path, out, min_size_bytes) for out in output_paths)


def is_analysis_cached(
    input_path: Path,
    cache_path: Path,
    *,
    expected_version: Optional[int] = None,
    version_key: str = "selector_version",
) -> bool:
    """
    Check if a cached analysis JSON is fresh.

    For analysis results (histogram stats, motion analysis, etc.)
    that are saved as JSON alongside the video.

    When ``expected_version`` is given (selection caches such as
    ``shorts_analysis.json``), the cache is additionally keyed on the stored
    ``version_key`` field AND the source-missing "trust the cache" shortcut is
    CLOSED: a selector-rule change re-analyzes, and a purged source can't prove
    the cache still reflects the current rule, so we re-analyze rather than ship
    pre-fix windows. With ``expected_version=None`` (default), behaviour is the
    legacy mtime+size freshness check (e.g. the color-analysis cache, which
    can't be recomputed once its source is gone).
    """
    if expected_version is None:
        return is_output_fresh(input_path, cache_path, min_size_bytes=10)

    input_path = Path(input_path)
    cache_path = Path(cache_path)
    if not cache_path.exists() or cache_path.stat().st_size < 10:
        return False
    # No source-missing trust for versioned selection caches — require the
    # source present and not newer than the cache.
    if not input_path.exists():
        logger.info("  Cache miss: %s source absent — re-analyzing", cache_path.name)
        return False
    if input_path.stat().st_mtime > cache_path.stat().st_mtime:
        return False
    data = load_cached_json(cache_path)
    if not data or data.get(version_key) != expected_version:
        logger.info(
            "  Cache miss: %s selector version %r != expected %r — re-analyzing",
            cache_path.name,
            (data or {}).get(version_key),
            expected_version,
        )
        return False
    return True


def load_cached_json(cache_path: Path) -> Optional[dict]:
    """Load a cached JSON file, returning None on any error."""
    try:
        with open(cache_path) as f:
            return json.load(f)
    except Exception as e:
        logger.debug("Cache load failed for %s: %s", cache_path, e)
        return None


def save_cache_json(cache_path: Path, data: dict):
    """Save analysis/results to a cache JSON file."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(data, f, indent=2)


def file_fingerprint(path: Path) -> str:
    """
    Quick fingerprint of a file using size + mtime.

    Much faster than hashing for large video files.
    For content-sensitive comparisons, use compute_file_hash instead.
    """
    stat = path.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"
