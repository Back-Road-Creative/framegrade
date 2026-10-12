"""Layer-2 end-of-Phase-D semantic quality gate.

Phase D emits many artifacts (LF metadata JSONs, ``drive_schedule``,
consolidated Metricool CSV, per-LF CSVs, thumbnails). Each can be
individually correct while the BATCH is semantically wrong — concrete
failure shapes:

  * All 7 LF titles identical (drive-wide ``--location`` leaked into
    every per-LF slot)
  * Metricool CSV's First Comment Text carried bare 11-char video IDs
    (phantom entries baked in from a stale sidecar)
  * ``publish_day='sat'`` on a Wed-only channel (schedule drift)

This module hosts two gates. The title-text subset (Rules 1 + 2) runs
UPSTREAM of the YouTube upload via :func:`assert_phase_d_title_quality`,
because titles exist the moment metadata is built and a violation must
block a bad-title video BEFORE it is published. Every other
rule runs after Phase D's per-LF loops but BEFORE ``PhaseD.run`` returns
via :func:`assert_phase_d_quality`, which validates upload-/CSV-derived
state. Each gate accumulates every violation it finds (not fail-on-first)
and raises :class:`~gomoveshift.pipeline.exceptions.PhaseQualityError`
once with the full list, so operators fix every problem in one round.

Entry points: :func:`assert_phase_d_title_quality` (upstream, Rules 1-2)
and :func:`assert_phase_d_quality` (end-of-Phase-D, the rest).

Rules (all checked except the removed Rule 4; accumulated and raised once):

  1. **LF titles all distinct.** (Upstream gate.) No collisions after
     ``ensure_unique_titles``. Operator-level intent: every LF tells
     viewers what's in it — same title means duplicate-looking library.
  2. **No ``seo_degraded: true``.** (Upstream gate.) Unless
     ``state["allow_seo_degraded"]`` is True. SEO fallback means title
     is derived from a worse source than the analytics-driven path, so
     it's a content-quality signal. Escape hatch:
     ``--allow-seo-degraded`` for runs where the SEO
     service is known-down and the operator has verified titles are
     still acceptable.
  3. **``publish_day`` ∈ ``LONG_FORM_DAYS``.** Catches schedule drift —
     e.g. a resume that carried ``"sat"`` from a prior run onto a
     Wednesday-only channel.
  4. **(Removed.)** The per-LF location != drive-wide ``location``
     check is no longer made here: the drive-wide fallback path was
     deleted, so the failure is unreachable by construction. The
     surviving assertion lives in ``day_pipeline.run_day_pipeline``.
  5. **CSV row count == Σ ``len(shorts_per_lf[i])``.** Row/post count
     mismatch means the CSV was built from the wrong source or
     partitioned incorrectly.
  6. **First Comment Text is URL OR placeholder.** Never a phantom
     bare video ID. The CSV's First Comment Text should be either:

        - empty (pre-upload / skip-upload path)
        - a ``#``-prefixed tag / non-URL caption (placeholder path), or
        - a proper YouTube URL

     A bare 11-char token is always a phantom ID leak from a stale
     sidecar.
  7. **``drive_schedule.lf_slots`` count == ``len(drive_plan.regular_lfs)``.**
     Planner skipped LFs or ate slots — ship time & date alignment is
     broken.
  8. **Every ``lf_slot.date`` weekday ∈ ``LONG_FORM_DAYS``.** Even if
     ``publish_day`` is right, individual slot dates can drift (e.g.
     resume-path miscomputation). Every date must land on a long-form
     day.
  9. **Shorts upload completeness.** When ``state["skip_upload"]`` is
     falsy, every LF in ``drive_plan.regular_lfs`` MUST have an entry
     in ``state["upload_manifests"]``. The failure this catches: every
     Drive upload silently failed (an error caught and logged as a
     warning in phase_d), so every ``Picture Url 1`` cell of the
     consolidated shorts CSV was empty while the run still reported
     success. Rule 9 turns that silent failure into a hard fail naming
     the missing LF indices.
 10. **YouTube LF upload completeness.** When YouTube upload is enabled
     (``state["youtube_upload"]`` truthy) and not skipped
     (``skip_youtube_upload`` / ``skip_upload``), every LF in
     ``drive_plan.regular_lfs`` MUST have a ``state["youtube_uploads"]``
     entry carrying a ``video_id``. The silent-drop class: the per-LF
     upload loop catches a transient ``_HttpError`` / ``Exception`` and
     ``continue``s past the failed LF, so the drive reports success with
     an LF whose YouTube video never uploaded. Rule 10 turns that into a
     hard fail naming the missing LF indices.

Each rule has a corresponding fail-case + pass-case test in
``tests/test_phase_d_quality_gates.py``.
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Any, Optional

from .exceptions import PhaseQualityError

logger = logging.getLogger(__name__)

# Grep-able prefix — operators can ``grep`` pipeline logs for this class of
# failure without knowing individual rule wording. Do NOT rename without
# flagging the break in a PR body.
PHASE_QUALITY_ERROR_PREFIX = "phase_d quality gate:"

# YouTube URL regex — accepts both youtu.be short-links and youtube.com/watch
# long-links with an 11-char alphanumeric ID (plus ``_-``). Matches the format
# emitted by ``csv_generator.build_consolidated_shorts_csv``
# (``▶ Full drive: https://youtu.be/<id>``) and legacy
# ``https://youtube.com/watch?v=<id>`` from older runs.
_YOUTUBE_URL_RE = re.compile(
    r"https?://(?:www\.)?(?:youtu\.be/|youtube\.com/watch\?v=)[A-Za-z0-9_-]{11}"
)

# Bare 11-char YouTube video ID: 11 chars from the YouTube alphabet AND
# nothing else. Rule 6 flags First Comment Text that is ONLY this pattern
# (no URL wrapping, no surrounding text) because that's the exact shape of
# a phantom leak from a stale sidecar. We do NOT flag longer captions that
# happen to contain an 11-char substring — those are human content, not
# state bugs.
_BARE_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

# Short day names → Python weekday int (Mon=0 … Sun=6). Matches
# ``state_invariants._DAY_NAME_TO_INT`` — kept as a local copy so
# quality_gates stays a leaf module (no cross-deps on state_invariants).
_DAY_NAME_TO_INT: dict[str, int] = {
    "mon": 0,
    "tue": 1,
    "wed": 2,
    "thu": 3,
    "fri": 4,
    "sat": 5,
    "sun": 6,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get(state: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a PipelineState dataclass OR a plain dict.

    Tests (and a couple of legacy callers) pass raw dicts directly, so
    this helper tolerates either shape. Maps a ``None`` attribute /
    missing key to the supplied default — the historical contract.
    """
    if state is None:
        return default
    if isinstance(state, dict):
        val = state.get(key, default)
    else:
        val = getattr(state, key, default)
    return default if val is None else val


def _metadata_items(state: Any) -> list[tuple[int, dict]]:
    """Return ``[(lf_index, metadata), ...]`` sorted by LF index."""
    meta_per_lf = _get(state, "metadata_per_lf", {}) or {}
    # Keys can be int (canonical) or str (resume-path drift — Layer 1 catches
    # that at write time, but we tolerate it here so a gate failure says
    # something useful instead of raising KeyError).
    items = []
    for k, v in meta_per_lf.items():
        try:
            idx = int(k) if not isinstance(k, bool) else k
        except (TypeError, ValueError):
            idx = k
        items.append((idx, v))
    try:
        items.sort(key=lambda kv: kv[0])
    except TypeError:
        # Mixed key types — stable by string form
        items.sort(key=lambda kv: str(kv[0]))
    return items


# ---------------------------------------------------------------------------
# Rule implementations — each returns a list of violation strings (empty = OK)
# ---------------------------------------------------------------------------


def check_titles_distinct(state: Any) -> list[str]:
    """Rule 1: every LF metadata's ``title`` value must be unique."""
    errors: list[str] = []
    items = _metadata_items(state)
    if not items:
        return errors
    titles = [str(m.get("title", "")) for _, m in items]
    seen: dict[str, list[int]] = {}
    for (idx, _), title in zip(items, titles):
        seen.setdefault(title, []).append(idx)
    duplicates = {t: idxs for t, idxs in seen.items() if len(idxs) > 1 and t.strip()}
    if duplicates:
        parts = [f"{t!r} on LFs {idxs}" for t, idxs in sorted(duplicates.items())]
        errors.append(f"LF titles are not all distinct: {'; '.join(parts)}")
    return errors


def check_seo_degraded(state: Any) -> list[str]:
    """Rule 2: reject ``seo_degraded: true`` unless allow_seo_degraded is set."""
    allow = bool(_get(state, "allow_seo_degraded", False))
    if allow:
        # Log-only note so operators see the flag was honoured on this run.
        degraded = [idx for idx, m in _metadata_items(state) if m.get("seo_degraded")]
        if degraded:
            logger.warning(
                "phase_d quality gate: allow_seo_degraded=True — %d LF(s) "
                "passed with seo_degraded=True (indices=%s)",
                len(degraded),
                degraded,
            )
        return []
    degraded = [idx for idx, m in _metadata_items(state) if m.get("seo_degraded")]
    if degraded:
        return [
            f"LFs {degraded} have seo_degraded=True but --allow-seo-degraded not set "
            "(SEO manager unavailable — re-run after fixing SEO service, or "
            "re-run with --allow-seo-degraded to accept degraded titles)"
        ]
    return []


def check_publish_day_in_long_form_days(state: Any) -> list[str]:
    """Rule 3: ``publish_day`` ∈ ``LONG_FORM_DAYS``."""
    # Lazy import so tests can monkeypatch ``schedule_constants.LONG_FORM_DAYS``
    from . import schedule_constants

    publish_day = _get(state, "publish_day")
    if not publish_day:
        # Empty / unset is OK at the gate — state_invariants Layer 1 handles
        # this (non-empty publish_day values must be in LONG_FORM_DAYS
        # already). Phase D with no publish_day is a different class of bug
        # and is not this gate's concern.
        return []
    if not isinstance(publish_day, str):
        return [f"publish_day must be a str, got {type(publish_day).__name__} {publish_day!r}"]
    pd_key = publish_day.lower()[:3]
    if pd_key not in _DAY_NAME_TO_INT:
        return [
            f"publish_day={publish_day!r} not a recognised day name "
            f"(expected one of mon/tue/wed/thu/fri/sat/sun)"
        ]
    pd_int = _DAY_NAME_TO_INT[pd_key]
    long_form_days = schedule_constants.LONG_FORM_DAYS
    if pd_int not in long_form_days:
        return [
            f"publish_day={publish_day!r} (weekday={pd_int}) not in "
            f"LONG_FORM_DAYS={sorted(long_form_days)}"
        ]
    return []


# ``check_per_lf_location_not_drive_wide`` was removed.
# Robust by construction: the drive-wide fallback path
# in ``derive_per_lf_locations`` was deleted, so the failure class this
# Phase D gate caught (after a full encode) is
# unreachable from a legitimate CLI invocation. The single contract
# assertion that survives lives at OST-burn entry in
# ``day_pipeline.run_day_pipeline`` — it fires only if construction
# logic regresses, not on operator input.


def check_csv_row_count(
    state: Any, csv_path: Optional[Path], expect_csv: bool = False
) -> list[str]:
    """Rule 5: CSV row count == Σ ``len(shorts_per_lf[i])``.

    ``expect_csv=True`` (threaded from ``not skip_csv``) means a CSV was
    supposed to be written; an absent file is then a genuine bug, not the
    intentional ``skip_csv`` skip (which the caller signals via
    ``csv_path is None``).
    """
    errors: list[str] = []
    if csv_path is None:
        return errors  # Intentional skip_csv — rule is cleanly vacuous.
    csv_path = Path(csv_path)
    if not csv_path.exists():
        if expect_csv:
            errors.append(
                f"consolidated shorts CSV expected but missing at {csv_path} "
                "— skip_csv was not set, so the CSV should have been written "
                "(a Phase D bug, not an intentional skip)"
            )
        # else: pre-skip_csv callers keep the historical vacuous pass.
        return errors
    shorts_per_lf = _get(state, "shorts_per_lf", {}) or {}
    expected = sum(len(lst) for lst in shorts_per_lf.values() if lst is not None)
    try:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            rows = list(reader)
    except OSError as e:
        errors.append(f"CSV read failed: {csv_path} — {e}")
        return errors
    if not rows:
        actual = 0
    else:
        actual = len(rows) - 1  # drop header
    if actual != expected:
        errors.append(
            f"CSV row count mismatch: expected {expected} "
            f"(Σ len(shorts_per_lf)), got {actual} at {csv_path}"
        )
    return errors


def check_csv_first_comment_text(csv_path: Optional[Path], expect_csv: bool = False) -> list[str]:
    """Rule 6: every First Comment Text is URL or placeholder, never phantom ID.

    ``expect_csv=True`` makes an absent-but-expected CSV a hard error (see
    :func:`check_csv_row_count`); ``csv_path is None`` is the intentional
    skip and always passes.
    """
    errors: list[str] = []
    if csv_path is None:
        return errors
    csv_path = Path(csv_path)
    if not csv_path.exists():
        if expect_csv:
            errors.append(
                f"consolidated shorts CSV expected but missing at {csv_path} "
                "— skip_csv was not set, so the CSV should have been written "
                "(a Phase D bug, not an intentional skip)"
            )
        return errors
    phantoms: list[tuple[int, str]] = []
    try:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader, start=1):
                val = (row.get("First Comment Text") or "").strip()
                if not val:
                    # Empty string is a valid placeholder (pre-upload).
                    continue
                if _YOUTUBE_URL_RE.search(val):
                    # URL present (even embedded in "Watch the full drive here: ...").
                    continue
                if _BARE_VIDEO_ID_RE.match(val):
                    # Bare 11-char ID — phantom.
                    phantoms.append((row_idx, val))
                    continue
                # Anything else (hashtags, free-form captions, etc.) is treated
                # as placeholder. Rule only rejects the specific phantom
                # signature — bare 11-char video ID.
    except OSError as e:
        errors.append(f"CSV read failed: {csv_path} — {e}")
        return errors
    if phantoms:
        sample = ", ".join(f"row {n}: {v!r}" for n, v in phantoms)
        errors.append(
            f"First Comment Text contains phantom video ID(s) at {csv_path} — "
            f"{sample} (these are bare 11-char YouTube IDs, not URLs — they "
            "leaked from a stale sidecar; re-run with --force-sidecar-reset)"
        )
    return errors


def check_schedule_lf_slot_count(state: Any, drive_plan: Any) -> list[str]:
    """Rule 7: ``len(drive_schedule.lf_slots) == len(drive_plan.regular_lfs)``."""
    drive_schedule = _get(state, "drive_schedule")
    if drive_schedule is None:
        return ["drive_schedule is None (expected populated after Phase D scheduler)"]
    lf_slots = getattr(drive_schedule, "lf_slots", None)
    if lf_slots is None:
        return ["drive_schedule has no lf_slots attribute"]
    regular_lfs = getattr(drive_plan, "regular_lfs", None) or []
    plan_count = len(regular_lfs)
    slot_count = len(lf_slots)
    if slot_count != plan_count:
        return [
            f"drive_schedule.lf_slots count mismatch: plan has {plan_count} "
            f"regular_lfs, schedule has {slot_count} lf_slots"
        ]
    return []


def check_schedule_dates_in_long_form_days(state: Any) -> list[str]:
    """Rule 8: every ``lf_slot.date`` weekday is in ``LONG_FORM_DAYS``."""
    # Lazy import — tests may monkeypatch.
    from . import schedule_constants

    drive_schedule = _get(state, "drive_schedule")
    if drive_schedule is None:
        return []  # covered by Rule 7
    lf_slots = getattr(drive_schedule, "lf_slots", None) or []
    from .seo_slots import anchor_errors

    return anchor_errors(
        (getattr(slot, "date", None) for slot in lf_slots), schedule_constants.LONG_FORM_DAYS
    )


def check_shorts_upload_complete(state: Any, drive_plan: Any) -> list[str]:
    """Rule 9: every regular_lf has a Drive upload manifest (unless skip_upload).

    Closes a silent-failure class: Phase D's per-LF upload loop
    catches ``FileNotFoundError`` as a generic ``Exception`` and logs a
    WARNING — leaving ``upload_manifests`` short of expected entries.
    Without this rule the pipeline reports success on a state where
    every shorts CSV row's ``Picture Url 1`` is empty.
    """
    # skip_upload is a back-compat alias for both Drive + YouTube
    # skips. The Drive-shorts completeness check honours either the alias
    # OR the granular skip_drive_upload flag.
    if _get(state, "skip_upload") or _get(state, "skip_drive_upload"):
        # Operator explicitly opted out of Drive uploads — empty manifests is expected.
        return []
    upload_manifests = _get(state, "upload_manifests", {}) or {}
    shorts_per_lf = _get(state, "shorts_per_lf", {}) or {}
    regular_lfs = getattr(drive_plan, "regular_lfs", None) or []
    expected_indices = [getattr(lf, "index", None) for lf in regular_lfs]
    missing: list[int] = []
    for idx in expected_indices:
        if idx is None:
            continue
        # An LF with zero shorts never produces a Drive manifest (the upload
        # loop skips it), so it is NOT expected here — requiring one would
        # false-fail any drive with a too-short LF (same failure class as the
        # empty-Picture-Url bug). When ``shorts_per_lf`` is absent (legacy
        # state) the guard is inert and all LFs are expected, as before.
        if shorts_per_lf and not (shorts_per_lf.get(idx) or shorts_per_lf.get(str(idx))):
            continue
        # Accept either int or str-keyed manifests (state JSON round-trip).
        if idx in upload_manifests or str(idx) in upload_manifests:
            continue
        missing.append(idx)
    if missing:
        try:
            missing_sorted = sorted(missing)
        except TypeError:
            missing_sorted = missing
        return [
            "shorts Drive upload incomplete: "
            f"{len(missing_sorted)}/{len(expected_indices)} LF(s) missing "
            f"upload_manifests entries — LF indices {missing_sorted} "
            "(see phase_d D-upload-shorts logs for the FileNotFoundError "
            "/ Drive errors that caused this; re-run after fixing, or "
            "pass --skip-upload if uploads are intentionally deferred)"
        ]
    return []


def check_youtube_uploads_complete(state: Any, drive_plan: Any) -> list[str]:
    """Rule 10: every regular_lf has a YouTube ``video_id`` (unless upload off).

    ``external_systems_coordinator._upload_lf_to_youtube`` catches a
    transient ``_HttpError`` / bare ``Exception`` and returns ``None``; the
    caller loop does ``if record is None: continue`` — so a per-LF YouTube
    upload that fails transiently is silently dropped, leaving
    ``state["youtube_uploads"]`` short of a ``video_id`` for that LF. Without
    this rule the drive reports success on a state where an LF's YouTube
    video never uploaded.

    No-op when YouTube upload was disabled (``youtube_upload`` falsy — no
    ``--youtube-upload``/``--publish`` this run) or skipped
    (``skip_youtube_upload`` / the ``skip_upload`` back-compat alias),
    mirroring ``_upload_lfs_to_youtube``'s ``skip_youtube_upload or not
    youtube_upload_enabled`` gate and Rule 9's skip semantics. Every LF that
    was uploaded needs its YouTube video — an LF with zero shorts still gets
    a long-form video, so (unlike Rule 9) there is no zero-shorts exemption.
    """
    if (
        not _get(state, "youtube_upload")
        or _get(state, "skip_youtube_upload")
        or _get(state, "skip_upload")
    ):
        # Upload disabled or explicitly skipped — empty youtube_uploads is expected.
        return []
    youtube_uploads = _get(state, "youtube_uploads", {}) or {}
    regular_lfs = getattr(drive_plan, "regular_lfs", None) or []
    expected_indices = [getattr(lf, "index", None) for lf in regular_lfs]
    missing: list[int] = []
    for idx in expected_indices:
        if idx is None:
            continue
        # Accept either int or str-keyed records (state JSON round-trip).
        record = youtube_uploads.get(idx)
        if record is None:
            record = youtube_uploads.get(str(idx))
        if isinstance(record, dict) and record.get("video_id"):
            continue
        missing.append(idx)
    if missing:
        try:
            missing_sorted = sorted(missing)
        except TypeError:
            missing_sorted = missing
        return [
            "YouTube LF upload incomplete: "
            f"{len(missing_sorted)}/{len(expected_indices)} LF(s) missing a "
            f"youtube_uploads video_id — LF indices {missing_sorted} "
            "(a per-LF upload failed transiently and was silently skipped; "
            "see phase_d D-upload-lfs logs for the YouTube HTTP/unexpected "
            "error that caused this; re-run to retry, or pass "
            "--skip-youtube-upload if uploads are intentionally deferred)"
        ]
    return []


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def assert_phase_d_title_quality(
    metadata_per_lf: Any,
    allow_seo_degraded: bool = False,
) -> None:
    """Title-text quality subset (Rules 1 + 2) — the UPSTREAM gate.

    Title distinctness and ``seo_degraded`` are computable from the LF
    metadata alone, which exists the moment ``_build_metadata_per_lf``
    runs — long before the irreversible YouTube upload. Running them here
    (instead of only at end-of-Phase-D) makes a duplicate-title /
    degraded-SEO video impossible to publish: a violation raises BEFORE the
    upload rather than after the LF is already public. The check runs before
    the irreversible step, not after it. These two rules are NOT repeated in
    :func:`assert_phase_d_quality`, which covers only the rules that need
    upload-/CSV-derived state.

    Args:
        metadata_per_lf: ``{lf_index: metadata}`` as built by Phase D's
            metadata loop (before any state write).
        allow_seo_degraded: ``--allow-seo-degraded`` escape hatch — downgrades
            Rule 2 to warning-only.

    Raises:
        PhaseQualityError: If titles collide or an LF is ``seo_degraded``
            without the escape hatch.
    """
    view = {
        "metadata_per_lf": metadata_per_lf or {},
        "allow_seo_degraded": allow_seo_degraded,
    }
    errors: list[str] = []
    errors.extend(check_titles_distinct(view))
    errors.extend(check_seo_degraded(view))
    if errors:
        joined = "\n  - ".join(errors)
        raise PhaseQualityError(f"{PHASE_QUALITY_ERROR_PREFIX}\n  - {joined}")


def assert_phase_d_quality(
    state: Any,
    drive_plan: Any,
    csv_path: Optional[Path] = None,
) -> None:
    """Run every Phase D quality gate and raise once with all violations.

    Args:
        state: Phase D's state dict / PipelineState (after per-LF loops).
        drive_plan: The ``DrivePlan`` being processed.
        csv_path: Path to the consolidated shorts Metricool CSV. Pass
            ``None`` if ``skip_csv`` was set — Rules 5 + 6 are skipped.

    Raises:
        PhaseQualityError: If ANY rule is violated. The exception message
            starts with :data:`PHASE_QUALITY_ERROR_PREFIX` and enumerates
            every failure as a separate line.
    """
    errors: list[str] = []

    # Accumulate all rule outputs. Order matches the rule list in the module docstring so
    # error messages read in the same order operators see in the docs.
    # Rules 1 (titles distinct) + 2 (seo_degraded) moved UPSTREAM of the
    # upload into ``assert_phase_d_title_quality`` — they are
    # computable from titles alone, so enforcing them here (after upload)
    # was a late gate that let a bad-title video ship before it raised.
    errors.extend(check_publish_day_in_long_form_days(state))
    # check_per_lf_location_not_drive_wide deleted —
    # the construction fix in phase_bc_day_first makes the failure
    # class unreachable from a legitimate CLI invocation.
    # A CSV is genuinely EXPECTED whenever skip_csv was not set — that's the
    # exact branch Phase D uses to build the consolidated CSV. An absent file
    # in that case is a bug (CSV never written), so Rules 5/6 must error
    # rather than pass vacuously. ``csv_path is None`` (the skip_csv signal)
    # still short-circuits to a clean pass inside each rule.
    expect_csv = not bool(_get(state, "skip_csv", False))
    errors.extend(check_csv_row_count(state, csv_path, expect_csv=expect_csv))
    errors.extend(check_csv_first_comment_text(csv_path, expect_csv=expect_csv))
    errors.extend(check_schedule_lf_slot_count(state, drive_plan))
    errors.extend(check_schedule_dates_in_long_form_days(state))
    errors.extend(check_shorts_upload_complete(state, drive_plan))
    errors.extend(check_youtube_uploads_complete(state, drive_plan))

    if errors:
        joined = "\n  - ".join(errors)
        raise PhaseQualityError(f"{PHASE_QUALITY_ERROR_PREFIX}\n  - {joined}")
