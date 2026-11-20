"""Failure fingerprint and class: what ``health heal`` keys its retry budget on.

A mirror commit says nothing about whether a failed drive can now succeed, so heal no longer
retries on a new SHA. It retries when the *inputs* of the failure changed (the fingerprint) and
as often as the failure's class allows. Pure functions only; heal injects every probe."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

DETERMINISTIC = "deterministic"
TRANSIENT = "transient"
PUBLISH_ONLY = "publish_only"
UNKNOWN = "unknown"

#: Launches of one (recipe, fingerprint) heal allows per class. ``deterministic`` is also gated
#: on the failure's own fingerprint (see ``refusal``); ``publish_only`` is never a render retry.
BUDGET: Dict[str, int] = {DETERMINISTIC: 1, TRANSIENT: 3, PUBLISH_ONLY: 0, UNKNOWN: 1}

#: ``(pattern, class)``, first match wins, matched against the failure text (error line + gate).
#: Each pattern names the raise site it was grounded on. Anything unmatched is ``unknown``.
CLASS_TABLE: Tuple[Tuple["re.Pattern[str]", str], ...] = tuple(
    (re.compile(p), c)
    for p, c in (
        # publish-only: a render retry cannot fix an upload/quota failure
        (
            r"YouTubeQuotaError|quotaExceeded|dailyLimitExceeded",
            PUBLISH_ONLY,
        ),  # pipeline/exceptions.py:141
        (r"DriveUploadError|DriveCredentialError", PUBLISH_ONLY),  # pipeline/uploader.py:52,56
        # deterministic: the same inputs fail the same way
        (r"BLUR GATE|BlurGateError", DETERMINISTIC),  # pipeline/blur_gate.py:486, inserts.py:271
        (
            r"final-text review refused|FinalTextReviewError|PublicationGateError",
            DETERMINISTIC,
        ),  # pipeline/final_text_record.py:116
        (
            r"InsufficientFootageError|is under full_drive_(?:part|short_day)_min_sec",
            DETERMINISTIC,
        ),  # pipeline/regular_lf_planner.py:312,373; orchestrator_preflight.py:243
        (r"BatchDurationIntegrityError", DETERMINISTIC),  # pipeline/day_master_encoder.py:111
        (r"GPS fix density .* below the", DETERMINISTIC),  # pipeline/title_decision.py:78
        # transient: the host or a remote source, not the inputs
        (
            r"No space left on device|ENOSPC",
            TRANSIENT,
        ),  # OS error text (pipeline/stop_processor.py:934 notes ENOSPC)
        (
            r"Cannot allocate memory|error code: -12|ENOMEM|[Oo]ut of memory|\bKilled\b|OOM",
            TRANSIENT,
        ),  # pipeline/ffmpeg_runner.py:478
        (
            r"DriveNotMountedError|is not accessible after \d+ probe|[Dd]rive offline|drive is offline",
            TRANSIENT,
        ),  # pipeline/wsl_mount.py:36; publishing/tiktok_pack.py:192
        (r"Input/output error|\[Errno 5\]", TRANSIENT),  # OS error text (EIO)
        (
            r"PlaceLookupUnavailableError|destination POI lookup",
            TRANSIENT,
        ),  # pipeline/exceptions.py:331
        (
            r"Connection (?:reset|refused|aborted)|ReadTimeout|timed out",
            TRANSIENT,
        ),  # network errors
    )
)


def classify_failure(error: Any = None, phase: Any = None, gate: Any = None) -> str:
    """``deterministic | transient | publish_only | unknown`` from the failure's text. Unknown is
    the default; ``phase`` is part of the fingerprint, not a signal (any phase can fail either way)."""
    text = " ".join(str(x) for x in (error, gate) if x)
    for pattern, klass in CLASS_TABLE:
        if pattern.search(text):
            return klass
    return UNKNOWN


def fingerprint(
    *, versions: Mapping[str, Optional[str]], manifest_sha: Optional[str],
    config_sha: Optional[str], phase: Optional[str],
) -> Dict[str, Any]:  # fmt: skip
    """The inputs a failure depends on. ``None`` = could not be read (stable, never a change)."""
    return {
        "versions": {k: versions.get(k) for k in sorted(versions)},
        "manifest_sha": manifest_sha, "config_sha": config_sha, "phase": phase,
    }  # fmt: skip


def fingerprint_hash(fp: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(fp, sort_keys=True).encode("utf-8")).hexdigest()


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def launched_count(attempts: Iterable[Mapping[str, Any]], fp_hash: str) -> int:
    """Launched, non-pre-flight attempts recorded under ``fp_hash``."""
    return sum(
        1 for a in attempts
        if a.get("action") == "launched" and a.get("fp_hash") == fp_hash
        and a.get("outcome") != "preflight_failed"
    )  # fmt: skip


def refusal(klass: str, fp_hash: str, *, launched: int, failed_under: bool) -> Optional[str]:
    """Why heal must not relaunch, else ``None``. ``launched``: launches under this fingerprint
    (all rows of the recipe); ``failed_under``: the failure itself was recorded under it."""
    short = f"{klass} failure, fingerprint {fp_hash[:12]}"
    if klass == PUBLISH_ONLY:
        return f"{short}: publish-only; heal never publishes, use `health publish`"
    if klass == DETERMINISTIC and (failed_under or launched):
        return (
            f"{short}: will fail the same way; waits for a changed version, manifest row or config"
        )
    if klass != DETERMINISTIC and launched >= BUDGET[klass]:
        return (
            f"{short}: {launched} of {BUDGET[klass]} attempts used; waits for a changed fingerprint"
        )
    return None
