"""YouTube uploads sidecar — persistent record of uploaded LFs.

Pipeline state can be reset (``fresh=True``, manual edits, empty-plan wipes).
When it is, the record of which LFs were already uploaded to YouTube is lost
and re-runs happily upload duplicates.

The sidecar lives at ``<output_base>/youtube_uploads.json`` and is independent
of pipeline state. It mirrors ``state["youtube_uploads"]`` but is written on
every state save and loaded on every run start. If its contents disagree with
state, sidecar wins (it is durable; state is mutable).

The file is written atomically (tempfile + ``os.replace``) so a crash during
write cannot corrupt the existing sidecar.

Integrity note
--------------

The sidecar is NOT a live mirror of YouTube. Operators can delete drafts in
YouTube Studio; aborted runs leave phantom slots; a re-planned drive can
shrink the LF count. :func:`assert_sidecar_integrity` is the local gate run
at Phase D entry — it only checks that the sidecar's LF-index set matches
``drive_plan.regular_lfs``. API-level reconciliation (querying YouTube for
each video_id) is out of scope for this module by design.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Optional

from ._atomic_write import atomic_write_json
from .exceptions import SidecarIntegrityError

logger = logging.getLogger(__name__)

SIDECAR_FILENAME = "youtube_uploads.json"
# Superlong parts get their OWN sidecar — see assert_superlong_sidecar_integrity.
SUPERLONG_SIDECAR_FILENAME = "superlong_uploads.json"
# Reserved key (never a valid "date:part_index" part key) recording which
# plan a superlong sidecar's progress was checkpointed against.
PLAN_FINGERPRINT_KEY = "__plan_fingerprint__"


def _sidecar_path(output_base: Path) -> Path:
    return Path(output_base) / SIDECAR_FILENAME


def _superlong_sidecar_path(output_base: Path) -> Path:
    return Path(output_base) / SUPERLONG_SIDECAR_FILENAME


def load_sidecar(output_base: Path) -> dict[str, Any]:
    """Load the YouTube uploads sidecar.

    Returns an empty dict if the sidecar is missing, unreadable, or
    contains invalid JSON. Never raises — a broken sidecar should not
    prevent the pipeline from starting.
    """
    path = _sidecar_path(output_base)
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(
            "youtube_uploads.json at %s is invalid JSON (%s) — treating as empty. "
            "Rename or delete the file to silence this warning.",
            path,
            e,
        )
        return {}
    except OSError as e:
        logger.warning("Failed to read youtube_uploads.json at %s: %s", path, e)
        return {}
    if not isinstance(data, dict):
        logger.warning(
            "youtube_uploads.json at %s is not a JSON object (got %s) — treating as empty",
            path,
            type(data).__name__,
        )
        return {}
    return data


def save_sidecar(output_base: Path, uploads: Optional[dict]) -> str:
    """Atomically write the YouTube uploads sidecar.

    Writes to a temp file in the same directory, then ``os.replace``s it
    over the target path. On failure the temp file is removed and the
    existing sidecar is left untouched.

    Returns the SHA-256 hex digest of the written file.
    """
    if uploads is None:
        uploads = {}
    path = _sidecar_path(output_base)
    # Stringify keys so int LF indices round-trip cleanly.
    serializable = {str(k): v for k, v in uploads.items()}
    atomic_write_json(path, serializable)
    # Compute SHA-256 of the file we just wrote.
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_superlong_sidecar_raw(output_base: Path) -> dict[str, Any]:
    """Raw file contents including the plan-fingerprint marker."""
    path = _superlong_sidecar_path(output_base)
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, ValueError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def load_superlong_sidecar(output_base: Path) -> dict[str, Any]:
    """Load ``superlong_uploads.json`` — same never-raises contract as ``load_sidecar``.

    Returns UPLOADS ONLY. The plan-fingerprint marker is metadata about the
    file, not a part, and every caller treats this mapping as
    ``part_key -> record`` — the resume skip-logic, the re-save round-trip
    and the integrity check all iterate its keys. Leaking the marker in
    would make it look like an uploaded part with no plan entry, i.e. a
    phantom key, which is the one condition that stays fatal. Read the
    marker with :func:`load_superlong_plan_fingerprint` instead.
    """
    return {
        k: v
        for k, v in _read_superlong_sidecar_raw(output_base).items()
        if k != PLAN_FINGERPRINT_KEY
    }


def load_superlong_plan_fingerprint(output_base: Path) -> Optional[str]:
    """The plan fingerprint this sidecar was last stamped with, if any."""
    value = _read_superlong_sidecar_raw(output_base).get(PLAN_FINGERPRINT_KEY)
    return value if isinstance(value, str) else None


def save_superlong_sidecar(output_base: Path, uploads: Optional[dict], *, state=None) -> str:
    """Atomically write ``superlong_uploads.json``; returns its SHA-256 digest.

    When *state* is given, stamps the write with the current plan's
    fingerprint (:data:`PLAN_FINGERPRINT_KEY`) so a later
    :func:`assert_superlong_sidecar_integrity` call — resuming mid-upload,
    with some plan keys not yet in the sidecar — can confirm the sidecar
    is progressing through the SAME plan rather than one that coincidentally
    shares a few keys with an unrelated re-plan.
    """
    if uploads is None:
        uploads = {}
    path = _superlong_sidecar_path(output_base)
    serializable = {str(k): v for k, v in uploads.items()}
    if state is not None:
        plan_keys = {key for key, _ in iter_superlong_parts(state)}
        serializable[PLAN_FINGERPRINT_KEY] = _plan_fingerprint(plan_keys)
    atomic_write_json(path, serializable)
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_superlong_parts(state) -> list[tuple[str, dict]]:
    """Yield ``(part_key, record)`` for every ``state.superlong_parts``
    entry, sorted by ``(date, part_index)``. Composite ``"date:part_index"``
    keys — a bare part_index alone repeats across a multi-day drive."""
    superlong_parts = getattr(state, "superlong_parts", None) or {}
    entries: list[tuple[str, int, dict]] = []
    for date, day_plan in superlong_parts.items():
        if not isinstance(day_plan, dict):
            continue
        for record in day_plan.get("records") or []:
            if not isinstance(record, dict) or not record.get("path"):
                continue
            try:
                part_index = int(record.get("part_index", 0))
            except (TypeError, ValueError):
                part_index = 0
            entries.append((str(date), part_index, record))
    entries.sort(key=lambda e: (e[0], e[1]))
    return [(f"{d}:{i:02d}", record) for d, i, record in entries]


def _plan_fingerprint(plan_keys) -> str:
    """Stable identity for a superlong plan's key-set: sha256 over the
    sorted composite ``date:part_index`` keys. Two plans that produce the
    same key-set hash identically; a re-plan that adds, removes, or
    renumbers parts changes it."""
    return hashlib.sha256("|".join(sorted(plan_keys)).encode("utf-8")).hexdigest()


def assert_superlong_sidecar_integrity(output_base: Path, state) -> None:
    """Part-sidecar analogue of :func:`assert_sidecar_integrity`, checked
    against ``state.superlong_parts`` (composite keys) — parts are never
    folded into ``drive_plan.regular_lfs``.

    Unlike the long-form sidecar (a complete manifest written once — see
    the intentional strict-subset contract in
    ``tests/test_resume_correctness_phase_d.py``), the superlong sidecar
    is an APPEND-AS-YOU-GO progress log: it is saved after every part
    upload, so a normal resume ALWAYS has plan keys the sidecar hasn't
    seen yet. Missing keys are therefore acceptable as long as the
    sidecar's recorded plan fingerprint (:func:`_plan_fingerprint`, see
    :func:`save_superlong_sidecar`) agrees with the plan being resumed —
    or no fingerprint was recorded at all (older sidecars; the subset
    check below already rules out the phantom-ID failure mode).

    Phantom keys — a sidecar entry the current plan never produced —
    stay fatal unconditionally, regardless of progress or fingerprint:
    that is the guard against dead video IDs baked into the
    Metricool CSV, and it is unaffected by write pattern.
    """
    entries = load_superlong_sidecar(output_base)
    path = _superlong_sidecar_path(output_base)
    stored_fingerprint = load_superlong_plan_fingerprint(output_base)
    if not entries:
        return
    plan_keys = {key for key, _ in iter_superlong_parts(state)}
    sidecar_keys = set(entries.keys())
    phantom = sorted(sidecar_keys - plan_keys)
    missing = sorted(plan_keys - sidecar_keys)

    if phantom:
        lines = [
            f"{SUPERLONG_SIDECAR_FILENAME}: {len(sidecar_keys)} entries vs {len(plan_keys)} parts",
            f"phantom part keys: {phantom}",
        ]
        if missing:
            lines.append(f"missing part keys: {missing}")
        raise SidecarIntegrityError(
            sidecar_path=str(path),
            phantom_indices=phantom,
            missing_indices=missing,
            message="\n  ".join(lines),
        )

    if not missing:
        return

    current_fingerprint = _plan_fingerprint(plan_keys)
    if stored_fingerprint is not None and stored_fingerprint != current_fingerprint:
        lines = [
            f"{SUPERLONG_SIDECAR_FILENAME}: plan fingerprint mismatch "
            f"({stored_fingerprint[:12]} recorded vs {current_fingerprint[:12]} current)",
            f"missing part keys: {missing}",
        ]
        raise SidecarIntegrityError(
            sidecar_path=str(path),
            phantom_indices=[],
            missing_indices=missing,
            message="\n  ".join(lines),
        )
    # No phantoms; missing keys with an agreeing (or absent) fingerprint —
    # normal in-progress upload, not corruption.


def assert_sidecar_integrity(output_base: Path, drive_plan) -> None:
    """Assert that the on-disk sidecar agrees with the current drive plan.

    Called from :class:`PhaseD.run` at entry, BEFORE any upload work, so
    stale ``youtube_uploads.json`` content cannot leak phantom video
    IDs into downstream CSV / first-comment generation (a sidecar whose
    slots referenced long-deleted videos once got baked into the
    Metricool CSV).

    Semantics:

      * **Empty sidecar** (no file on disk, or file containing ``{}``)
        — pass unconditionally. A fresh run has nothing to reconcile.
      * **Count + index agreement** — the set of integer LF indices
        derived from sidecar keys equals the set of
        ``lf.index`` values in ``drive_plan.regular_lfs``. Pass.
      * **Any disagreement** — raise :class:`SidecarIntegrityError`
        with the diff broken out into ``phantom_indices`` (in sidecar,
        not in plan) and ``missing_indices`` (in plan, not in
        sidecar). The operator sees the diff and decides: wipe via
        ``--force-sidecar-reset`` or reconcile manually.

    Sidecar keys on disk are strings (JSON); the gate coerces to int
    for comparison. Non-coercible keys are reported as phantoms
    (malformed files surface, rather than silently skipping entries).

    Does NOT contact YouTube; API-level reconciliation is out of
    scope for this gate. This gate is the cheapest, earliest signal that
    state and plan disagree.
    """
    sidecar = load_sidecar(output_base)
    path = _sidecar_path(output_base)
    if not sidecar:
        # Fresh run — nothing to validate.
        return

    plan_indices = {lf.index for lf in (drive_plan.regular_lfs or [])}

    sidecar_indices: set[int] = set()
    malformed_keys: list[str] = []
    for key in sidecar:
        try:
            sidecar_indices.add(int(key))
        except (TypeError, ValueError):
            malformed_keys.append(str(key))

    phantom = sorted(sidecar_indices - plan_indices)
    missing = sorted(plan_indices - sidecar_indices)

    if not phantom and not missing and not malformed_keys:
        return

    # Build an actionable operator message. The first line names the
    # counts; the next lines cite the diffs; the last gives the
    # escape-hatch command.
    plan_count = len(plan_indices)
    sidecar_count = len(sidecar)
    lines = [
        f"sidecar has {sidecar_count} entries, drive_plan has {plan_count} LFs",
    ]
    if phantom:
        lines.append(f"phantom LF indices (in sidecar, not in plan): {phantom}")
    if missing:
        lines.append(f"missing LF indices (in plan, not in sidecar): {missing}")
    if malformed_keys:
        lines.append(
            f"malformed (non-integer) sidecar keys — phantom by coercion: {malformed_keys}"
        )
    lines.append(
        "Operator action required: verify via YouTube Studio, then either "
        "(a) re-run with --force-sidecar-reset to wipe the sidecar and "
        "upload fresh, or (b) edit youtube_uploads.json manually to match "
        "the current plan."
    )
    # Extend phantoms with the malformed raw strings so err.phantom_indices
    # carries a complete picture for programmatic callers.
    phantom_for_err = list(phantom) + list(malformed_keys)
    raise SidecarIntegrityError(
        sidecar_path=str(path),
        phantom_indices=phantom_for_err,
        missing_indices=missing,
        message="\n  ".join(lines),
    )
