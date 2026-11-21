"""Failure fingerprint and class: what ``health heal`` keys its retry budget on.

A mirror commit says nothing about whether a failed drive can now succeed, so heal no longer
retries on a new SHA. It retries when the *inputs* of the failure changed (the fingerprint) and
as often as the failure's class allows. Pure functions only; heal injects every probe."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

from .store import DIST_NAME

DETERMINISTIC = "deterministic"
TRANSIENT = "transient"
PUBLISH_ONLY = "publish_only"
UNKNOWN = "unknown"

#: Launches of one (recipe, fingerprint) heal allows per class. ``deterministic`` is also gated
#: on the failure's own fingerprint (see ``refusal``); ``publish_only`` is never a render retry.
BUDGET: Dict[str, int] = {DETERMINISTIC: 1, TRANSIENT: 3, PUBLISH_ONLY: 0, UNKNOWN: 1}

#: Interrupted heal runs (a host restart or killed runner left the job at "running" with a dead
#: nanny) per fingerprint that do not use the budget. Past it each counts like a failure, so a
#: crash-looping host cannot relaunch forever.
HEAL_INTERRUPTED_RETRIES = 3

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
        # an interrupted cabin-repair swap: the marker makes the next run re-detect
        (r"BLUR GATE \(repair swap\)", TRANSIENT),  # pipeline/blur_gate.py repair_swap_error
        # a privacy detection that FAILED (not a leak or a refusal) on a host cause: the
        # retry re-runs detection and every gate, so it can never encode unproven
        (
            r"(?s)BLUR GATE: privacy detection failed .*(?:No space left on device|ENOSPC|"
            r"Cannot allocate memory|error code: -12|ENOMEM|[Oo]ut of memory|\bKilled\b|"
            r"Input/output error|\[Errno 5\]|[Dd]rive offline|drive is offline)",
            TRANSIENT,
        ),  # pipeline/day_master_encoder.py _build_day_blur_mask
        (r"BLUR GATE|BlurGateError", DETERMINISTIC),  # pipeline/blur_gate.py:486, inserts.py:271
        (
            r"final-text review refused|FinalTextReviewError|PublicationGateError",
            DETERMINISTIC,
        ),  # pipeline/final_text_record.py:116
        (
            r"InsufficientFootageError|is under full_drive_(?:part|short_day)_min_sec"
            r"|under min_lf_duration=",
            DETERMINISTIC,
        ),  # pipeline/regular_lf_planner.py:312,371; orchestrator_preflight.py:241; drive_phases/phase_a.py:173
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


_VERSIONS_CODE = (
    "import importlib.metadata as m, json\n"
    "out = {}\n"
    "try:\n    out['redactcam'] = m.version('redactcam')\n"
    "except Exception:\n    out['redactcam'] = None\n"
    "try:\n    import seo_manager; out['seo_manager'] = seo_manager.__version__\n"
    "except Exception:\n    out['seo_manager'] = None\n"
    "print(json.dumps(out))"
)  # seo_manager ships no dist-info here, so its ``__version__`` is read instead


def default_installed_version(python: str) -> Optional[str]:
    """The ``gomoveshift-video`` version installed in ``python``'s environment, else ``None``
    (no interpreter, not installed, or the lookup failed)."""
    code = f"import importlib.metadata as m; print(m.version({DIST_NAME!r}))"
    try:
        done = subprocess.run(  # noqa: S603 -- the service interpreter, fixed code
            [python, "-c", code], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    out = done.stdout.strip()
    return out if done.returncode == 0 and out else None


def default_sibling_versions(python: str) -> Dict[str, Optional[str]]:
    """``seo_manager`` and ``redactcam`` versions in ``python``'s environment (``None`` each when
    unreadable): the dependencies a drive's output depends on besides ``gomoveshift``."""
    try:
        done = subprocess.run(  # noqa: S603 -- the service interpreter, fixed code
            [python, "-c", _VERSIONS_CODE], capture_output=True, text=True, timeout=30, check=False
        )
        got = json.loads(done.stdout) if done.returncode == 0 else {}
    except (OSError, ValueError, subprocess.SubprocessError):
        got = {}
    return {k: got.get(k) if isinstance(got, dict) else None for k in ("seo_manager", "redactcam")}


def default_manifest_sha(row: Dict[str, Any]) -> Optional[str]:
    """sha256 of the ``drive_backlog.yaml`` row the backlog launcher resolves for this drive
    (project = the innermost job name, minus ``gms-backlog-``), else ``None`` (not a backlog
    drive, or the manifest is unreadable)."""
    from . import drives  # noqa: PLC0415 -- drives imports store; keep the import graph acyclic

    name = str(row.get("drive") or "")
    name = min(drives.heal_origins(name) or {name}, key=len)
    project = name.removeprefix("gms-backlog-")
    try:
        import dataclasses  # noqa: PLC0415

        from seo_manager.scheduling import backlog  # noqa: PLC0415

        from ..paths import drive_backlog_path  # noqa: PLC0415

        found = next((r for r in backlog.load(drive_backlog_path()) if r.project == project), None)
        return sha256_json(dataclasses.asdict(found)) if found else None
    except Exception:  # noqa: BLE001 -- an unreadable manifest is "unknown", never a retry
        return None


def default_config_sha(config: Any) -> Optional[str]:
    """sha256 of the resolved drive config, else ``None``."""
    try:
        return sha256_json(config.drive.model_dump(mode="json"))
    except Exception:  # noqa: BLE001
        return None


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
    """Launched attempts recorded under ``fp_hash`` that used budget: not pre-flight failures and
    not the first ``HEAL_INTERRUPTED_RETRIES`` interrupted ones (a host restart is not a failure)."""
    used, interrupted = 0, 0
    for a in attempts:
        if a.get("action") != "launched" or a.get("fp_hash") != fp_hash:
            continue
        outcome = a.get("outcome")
        if outcome == "preflight_failed":
            continue
        if outcome == "interrupted":
            interrupted += 1
            if interrupted <= HEAL_INTERRUPTED_RETRIES:
                continue
        used += 1
    return used


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


def failure_stamp(
    row: Mapping[str, Any], *, python: str, config: Any,
    installed_version_fn: Optional[Callable[[str], Optional[str]]] = None,
    sibling_versions_fn: Optional[Callable[[str], Dict[str, Optional[str]]]] = None,
    manifest_sha_fn: Optional[Callable[[Any], Optional[str]]] = None,
    config_sha_fn: Optional[Callable[[Any], Optional[str]]] = None,
) -> Dict[str, Any]:  # fmt: skip
    """The ``failure_fingerprint`` a row carries: ``{hash, fingerprint, class}``. ``health check``
    writes it when it files the row, so it names the code the drive failed under, never the code
    heal later runs."""
    versions = {
        "gomoveshift": (installed_version_fn or default_installed_version)(python),
        **(sibling_versions_fn or default_sibling_versions)(python),
    }
    fp = fingerprint(
        versions=versions, manifest_sha=(manifest_sha_fn or default_manifest_sha)(row),
        config_sha=(config_sha_fn or default_config_sha)(config), phase=row.get("phase"),
    )  # fmt: skip
    klass = classify_failure(row.get("error"), row.get("phase"), row.get("gate"))
    return {"hash": fingerprint_hash(fp), "fingerprint": fp, "class": klass}


def service_python(code_root: Path) -> str:
    return str(Path(code_root) / ".venv" / "bin" / "python")
