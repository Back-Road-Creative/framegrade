"""OST text audit sidecar — operator-visible artefact before any OST burn.

Burning OST pixels onto an LF without an operator-visible audit artefact
is unrepresentable. The gate at ``run_day_pipeline``'s per-LF loop
writes ``<output_base>/ost_audit_<date>.json`` listing each LF's
canonical OST text, YouTube title, thumbnail-text source (proxy =
canonical OST text — single source of truth for OST, title and
thumbnail), and recording date.

The operator either

* creates ``<output_base>/ost_audit_<date>.confirmed`` (manual review),
* sets ``state["ost_audit_passed"][date] = True`` (resume marker), or
* re-runs with ``--ost-audit-confirmed`` (non-interactive bypass).

Hash mismatch (operator edits per-LF labels and re-runs phase_bc) re-
raises :class:`OstAuditPendingError` so the operator re-confirms against
the new OST text.

This module is a pure formatter + writer + hash helper plus the
exception class. The gate logic lives in
``gomoveshift.pipeline.day_pipeline.run_day_pipeline``.

Mirrors :mod:`gomoveshift.pipeline.endscreen_sidecar`'s shape: pure
helpers separate from runtime gate logic.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from ._atomic_write import atomic_write_json
from .blur_gate import _gray_frames
from .encoding_constants import SHORTS_HEIGHT, SHORTS_WIDTH

logger = logging.getLogger(__name__)


class OstAuditPendingError(RuntimeError):
    """Raised by ``run_day_pipeline`` when the OST audit gate has not been
    confirmed for the day under encode.

    The orchestrator catches this and exits with ``success=True`` and a
    ``stages_completed`` list that includes phase_a but NOT
    phase_bc_day_first — so the next resume re-checks the gate.
    """


def _state_field(state, key: str):
    """Read ``key`` from a PipelineState dataclass OR a plain dict.

    Tests pass raw dicts here; production passes ``PipelineState``.
    Returns ``None`` for absent attributes / keys.
    """
    if state is None:
        return None
    if isinstance(state, dict):
        return state.get(key)
    return getattr(state, key, None)


def _resolve_ost_text(state, lf_index: int, location: str = "") -> str:
    """Mirror ``run_day_pipeline``'s OST-text priority cascade EXACTLY.

    Priority: ``per_lf_ost_text`` (canonical) > ``per_lf_locations``
    (county fallback) > drive-wide ``location`` > empty string. The
    drive-wide ``location`` fallback is included so the audited
    text equals the text the per-LF burn loop will actually composite
    (``day_pipeline.run_day_pipeline`` resolves ``lf_location`` with the
    same three-tier cascade) — review-text == burn-text. Omitting it (the
    prior behaviour) hid the real burn string for a GPS-less drive whose
    only label is the drive-wide ``--location``.
    """
    per_lf_ost = _state_field(state, "per_lf_ost_text") or {}
    if lf_index in per_lf_ost and per_lf_ost[lf_index]:
        return str(per_lf_ost[lf_index])
    per_lf_loc = _state_field(state, "per_lf_locations") or {}
    if lf_index in per_lf_loc and per_lf_loc[lf_index]:
        return str(per_lf_loc[lf_index])
    return str(location or "")


def confirmation_is_stale(prior_hash, current_hash) -> bool:
    """Whether a prior OST-audit confirmation no longer covers the live text.

    Returns ``True`` only when a ``prior_hash`` exists AND differs from the
    just-computed ``current_hash`` — i.e. the resolved label changed
    materially since the run the operator confirmed against. A ``None``
    ``prior_hash`` (a first-ever confirmation, pre-dating hash tracking)
    is NOT stale: there is nothing to invalidate. The hash already excludes
    ``generated_at`` (see :func:`compute_ost_audit_hash`), so wall-clock
    noise never marks a confirmation stale — only a real label change does.
    """
    if not prior_hash:
        return False
    return prior_hash != current_hash


def read_confirmed_hash(confirmed_file):
    """Audit hash a ``.confirmed`` marker is bound to, or ``None``.

    A confirmation is bound to the exact audit text it confirmed by stamping
    the audit hash into the marker (:func:`stamp_confirmed`) when the gate
    passes. A later run then detects a leftover marker that confirms a
    DIFFERENT, never-reviewed burn text — ``--fresh`` wipes the state-persisted
    prior hash, so ``exists()`` alone would otherwise pass. A marker whose
    content is not a sha256 hex digest — a hand-created / legacy ``.confirmed``,
    which the operator may fill with anything — carries no binding and returns
    ``None`` (a first-ever confirmation, treated as fresh, not stale).
    """
    try:
        text = Path(confirmed_file).read_text().strip()
    except OSError:
        return None
    if len(text) == 64 and all(c in "0123456789abcdef" for c in text):
        return text
    return None


def stamp_confirmed(confirmed_file, current_hash: str) -> None:
    """Bind a ``.confirmed`` marker to the audit hash it just confirmed.

    Written when the gate passes so a later run whose burn text changed reads a
    stale binding via :func:`read_confirmed_hash` and reopens the gate — a
    leftover confirmation can never validate a materially different burn text.
    """
    Path(confirmed_file).write_text(str(current_hash))


def format_ost_audit(
    date: str,
    state: dict,
    lf_indices: Iterable[int],
    location: str = "",
) -> dict:
    """Build the audit payload for one day.

    Args:
        date: The day's ISO date (``YYYY-MM-DD``) — both the partition
            key and the value of the JSON's ``date`` field.
        state: Pipeline state dict supplying ``per_lf_ost_text``,
            ``per_lf_titles``, ``per_lf_recording_date``, and
            ``per_lf_locations``.
        lf_indices: The LF indices on this day. Order is normalised
            (sorted ascending) so the operator-visible JSON is stable.
        location: The drive-wide ``--location`` default — the final tier
            of the burn cascade. Threaded through so the audited
            ``ost_text`` equals the text the per-LF burn loop composites
            even when an LF has neither canonical text nor a per-LF label.

    Returns:
        A dict with ``date``, ``generated_at`` (ISO-8601 UTC), and an
        ``lfs`` list where each entry carries ``lf_index``, ``ost_text``,
        ``youtube_title``, ``thumbnail_text``, ``recording_date``.

        ``ost_text`` is the REAL resolved burn string (full cascade,
        incl. drive-wide ``location``). ``thumbnail_text`` mirrors it —
        the same canonical text drives both surfaces (single source of
        truth for OST, title and thumbnail); it is the real burn text, not a divergent proxy.
    """
    titles = _state_field(state, "per_lf_titles") or {}
    recording = _state_field(state, "per_lf_recording_date") or {}
    indices_sorted = sorted({int(i) for i in lf_indices})
    lfs: list[dict] = []
    for idx in indices_sorted:
        ost_text = _resolve_ost_text(state, idx, location)
        lfs.append(
            {
                "lf_index": idx,
                "ost_text": ost_text,
                "youtube_title": str(titles.get(idx, "") or ""),
                # Real resolved burn text (= ost_text), not a proxy: the
                # same canonical string is what the thumbnail surfaces.
                "thumbnail_text": ost_text,
                "recording_date": str(recording.get(idx, "") or ""),
            }
        )
    return {
        "date": str(date),
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "lfs": lfs,
    }


def write_ost_audit(output_base: Path, date: str, payload: dict) -> Path:
    """Write the audit payload to ``<output_base>/ost_audit_<date>.json``.

    Creates ``output_base`` if missing. Returns the written path.
    """
    output_base = Path(output_base)
    path = Path(output_base) / f"ost_audit_{date}.json"
    atomic_write_json(path, payload, sort_keys=True)
    return path


def format_shorts_ost_audit(date: str, entries: Iterable[dict]) -> dict:
    """Build the per-short OST audit payload for one day.

    The per-LF :func:`format_ost_audit` is written BEFORE the shorts loop
    runs, so it can only enumerate LF text — the 14 shorts/LF burned text
    had zero audit coverage. This records each short's final (shortened +
    validated) burned text after the loop, the day's largest single-surface
    burn-risk blind spot.

    Each entry carries ``rank``, ``parent_lf_index``, ``ost_text``,
    ``start_sec``, ``end_sec`` and ``regions`` (the short's own clip-GPS vote
    ``"County, State"`` labels — the independent reference the per-short
    CSV region invariant checks the shipped surfaces against). The audit also adds
    ``median_speed_mps`` (this clip's GPS ground speed — a parked window reads
    ~0) so a reviewer can eyeball "this window is parked" before the
    irreversible burn; it is a diagnostic annotation EXCLUDED from the audit
    hash. There is deliberately no ``measured_lufs`` counterpart: shorts ship
    muted (``-an``), nothing measures a per-short loudness, and a permanently
    ``None`` field reads as "measured, and it was nothing". Order is normalised
    by rank for a stable operator-visible JSON.
    """

    def _num_or_none(v):
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    shorts = sorted(
        (
            {
                "rank": int(e.get("rank", 0)),
                "parent_lf_index": e.get("parent_lf_index"),
                "ost_text": str(e.get("ost_text", "") or ""),
                "start_sec": float(e.get("start_sec", 0.0) or 0.0),
                "end_sec": float(e.get("end_sec", 0.0) or 0.0),
                "regions": [str(r) for r in (e.get("regions") or []) if str(r).strip()],
                "median_speed_mps": _num_or_none(e.get("median_speed_mps")),
                # Which source won this short's burned text — "clip_gps"
                # (the clip's OWN GPS vote) or "lf_label" (inherited from the
                # parent LF / drive-wide fallback). Legacy entries predating
                # this field default to "lf_label", the pre-fix behaviour.
                "place_source": str(e.get("place_source") or "lf_label"),
                # Post-burn: the rendered short this entry describes. Empty
                # at the pre-burn gate (nothing exists yet); back-filled by the
                # render loop so the post-burn check has a file to open.
                "path": str(e.get("path", "") or ""),
            }
            for e in entries
        ),
        key=lambda s: s["rank"],
    )
    return {
        "date": str(date),
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "shorts": shorts,
    }


def write_shorts_ost_audit(output_base: Path, date: str, entries: Iterable[dict]) -> Path:
    """Write the shorts audit to ``<output_base>/shorts_ost_audit_<date>.json``.

    Separate from the per-LF ``ost_audit_<date>.json`` because the shorts
    burned text only exists after the per-LF audit gate has already run.
    Returns the written path.
    """
    output_base = Path(output_base)
    path = output_base / f"shorts_ost_audit_{date}.json"
    atomic_write_json(path, format_shorts_ost_audit(date, entries), sort_keys=True)
    return path


# Diagnostic annotations that must NOT invalidate a confirmation (same
# rationale as excluding ``generated_at``): a measured speed readout is
# observability, not burn text. ``path`` joins them — it records where the
# artefact landed, not what was burned, so binding a confirmation to it would
# reopen the reviewed gate on every ``--output-dir`` change. ``measured_lufs``
# is no longer emitted (nothing ever measured it) and is now inert here: the
# hash is always computed from a freshly formatted payload, never re-derived
# from an on-disk sidecar, so no current caller can supply the key. It stays
# listed because this tuple is a strip-list over caller-supplied entries —
# keeping a retired name costs nothing and removing one is what would risk
# admitting a stale key into a digest. ``place_source`` is the same
# kind of annotation — it records WHICH input won ``ost_text``, not a burn
# input of its own; a genuine change in the winning source always changes
# ``ost_text`` too, which already reopens the gate.
_HASH_EXCLUDED_ENTRY_KEYS = ("median_speed_mps", "measured_lufs", "path", "place_source")


def _strip_hash_excluded(entries) -> list:
    """Drop diagnostic-only keys from each audit entry before hashing."""
    return [
        {k: v for k, v in e.items() if k not in _HASH_EXCLUDED_ENTRY_KEYS} for e in (entries or [])
    ]


def compute_ost_audit_hash(payload: dict) -> str:
    """Return a stable hash of the audit content (excludes ``generated_at``).

    The hash IS the rebuild-detection signal: if the operator edits
    per-LF labels and re-runs phase_bc, the audit's LF list changes and
    the hash changes. ``generated_at`` is wall-clock noise that must NOT
    invalidate a confirmation, so it's excluded from the hash input.
    """
    content = {"date": payload.get("date"), "lfs": _strip_hash_excluded(payload.get("lfs", []))}
    serialised = json.dumps(content, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialised).hexdigest()


def compute_shorts_ost_audit_hash(payload: dict) -> str:
    """Stable hash of the SHORTS audit content (excludes ``generated_at``).

    Mirror of :func:`compute_ost_audit_hash` over the shorts payload
    (``{"date", "shorts"}``): a material change to any short's burned text
    reopens the pre-burn gate; ``generated_at`` noise is excluded.
    """
    content = {
        "date": payload.get("date"),
        "shorts": _strip_hash_excluded(payload.get("shorts", [])),
    }
    serialised = json.dumps(content, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialised).hexdigest()


def lf_gate_or_raise(
    output_dir: Path,
    date: str,
    lf_indices,
    state,
    location: str = "",
    per_lf_ost_text=None,
    per_lf_locations=None,
) -> None:
    """Pre-burn OST review gate for long-form LFs — LF analog of
    :func:`shorts_gate_or_raise`, extracted from ``run_day_pipeline`` so the
    SAME gate can also run UPFRONT for every day before the first master
    encode (front-load). Writes ``<output_dir>/ost_audit_<date>.json`` and
    raises :class:`OstAuditPendingError` unless confirmed: matching persisted
    ``ost_audit_hash`` (resume), an ``ost_audit_<date>.confirmed`` marker, or
    ``state["ost_audit_confirmed_this_run"]`` (``--ost-audit-confirmed`` passed
    on THIS run — a persisted-only confirmation no longer bypasses). A
    material burn-text change reopens the gate (``confirmation_is_stale``); empty
    ``lf_indices`` → no gate. A pass persists the markers a later per-day call
    fast-passes on. The ``audit_state`` assembly mirrors ``run_day_pipeline``
    EXACTLY so the front-load hash equals the per-day recompute.
    """
    lf_indices = [int(i) for i in (lf_indices or [])]
    if not lf_indices:
        return
    audit_state = {
        "per_lf_ost_text": _state_field(state, "per_lf_ost_text") or per_lf_ost_text or {},
        "per_lf_titles": _state_field(state, "per_lf_titles") or {},
        "per_lf_recording_date": _state_field(state, "per_lf_recording_date") or {},
        "per_lf_locations": _state_field(state, "per_lf_locations") or per_lf_locations or {},
    }
    payload = format_ost_audit(date, audit_state, lf_indices, location=location)
    current_hash = compute_ost_audit_hash(payload)

    passed_map = _state_field(state, "ost_audit_passed") or {}
    hash_map = _state_field(state, "ost_audit_hash") or {}
    # Only a confirmation supplied on THIS run bypasses without a stored
    # hash. A persisted ``ost_audit_confirmed`` no longer blanket-passes a
    # changed, never-reviewed burn after a kill lost the hash — a resume passes
    # via the hash-bound ``ost_audit_passed`` + ``ost_audit_hash`` markers below.
    cli_confirmed = bool(_state_field(state, "ost_audit_confirmed_this_run"))
    confirmed_file = Path(output_dir) / f"ost_audit_{date}.confirmed"

    # Bind the ``.confirmed`` marker to the hash it confirmed so
    # ``exists()`` alone can't pass a materially changed, never-reviewed burn.
    prior_hash = hash_map.get(date) or read_confirmed_hash(confirmed_file)
    stale = confirmation_is_stale(prior_hash, current_hash)

    gate_passes = False
    reason = ""
    if stale:
        gate_passes = False  # explicit: a material label change reopens the gate
    elif cli_confirmed:
        gate_passes = True
        reason = "--ost-audit-confirmed"
    elif confirmed_file.exists():
        gate_passes = True
        reason = f"{confirmed_file.name} on disk"
    elif passed_map.get(date) is True and hash_map.get(date) == current_hash:
        gate_passes = True
        reason = "state['ost_audit_passed'] + matching hash"

    if not gate_passes:
        sidecar_path = write_ost_audit(Path(output_dir), date, payload)
        logger.info(
            "OST audit sidecar written: %s. Review and create %s "
            "(or pass --ost-audit-confirmed) to proceed.",
            sidecar_path,
            confirmed_file,
        )
        raise OstAuditPendingError(
            f"OST audit pending for day {date}: review {sidecar_path} "
            f"and create {confirmed_file} (or re-run with "
            f"--ost-audit-confirmed) before OST burn-in."
        )

    if state is not None:
        pm = _state_field(state, "ost_audit_passed")
        if pm is None:
            pm = {}
            state["ost_audit_passed"] = pm
        pm[date] = True
        hm = _state_field(state, "ost_audit_hash")
        if hm is None:
            hm = {}
            state["ost_audit_hash"] = hm
        hm[date] = current_hash
    # Bind the marker so a later changed burn reads a stale binding.
    if confirmed_file.exists():
        stamp_confirmed(confirmed_file, current_hash)
    logger.info("OST audit gate passed for day %s (%s).", date, reason)


def shorts_gate_or_raise(output_base: Path, date: str, entries, state) -> None:
    """Pre-burn OST review gate for shorts — mirror of the per-LF gate.

    The per-LF gate runs before any short exists, so shorts burned text
    reached the irreversible 9:16 burn with no operator review. This writes
    the shorts audit for review and raises :class:`OstAuditPendingError`
    unless confirmed: a ``shorts_ost_audit_<date>.confirmed`` file, a matching
    persisted hash (resume short-circuit), or the shared
    ``state["ost_audit_confirmed_this_run"]`` (``--ost-audit-confirmed`` on THIS
    run) bypass — one operator signal covers both surfaces. A material change to
    any short's burned text reopens the gate (``confirmation_is_stale``); empty
    ``entries`` (no shorts) → no gate.
    """
    if not entries:
        return
    payload = format_shorts_ost_audit(date, entries)
    current_hash = compute_shorts_ost_audit_hash(payload)

    passed_map = _state_field(state, "shorts_ost_audit_passed") or {}
    hash_map = _state_field(state, "shorts_ost_audit_hash") or {}
    # Only a THIS-run confirmation bypasses without a stored hash (see the
    # LF gate). One flag covers both surfaces; a resume passes via the markers.
    cli_confirmed = bool(_state_field(state, "ost_audit_confirmed_this_run"))
    confirmed_file = Path(output_base) / f"shorts_ost_audit_{date}.confirmed"

    # Bind the ``.confirmed`` marker to the audit hash it confirmed so
    # ``exists()`` alone can't pass a materially changed, never-reviewed burn.
    prior_hash = hash_map.get(date) or read_confirmed_hash(confirmed_file)
    stale = confirmation_is_stale(prior_hash, current_hash)
    gate_passes = False
    if stale:
        gate_passes = False  # explicit: a material burn-text change reopens the gate
    elif cli_confirmed:
        gate_passes = True
    elif confirmed_file.exists():
        gate_passes = True
    elif passed_map.get(date) is True and hash_map.get(date) == current_hash:
        gate_passes = True

    if not gate_passes:
        sidecar_path = write_shorts_ost_audit(Path(output_base), date, entries)
        raise OstAuditPendingError(
            f"Shorts OST audit pending for day {date}: review {sidecar_path} "
            f"and create {confirmed_file} (or re-run with --ost-audit-confirmed) "
            "before the irreversible shorts burn."
        )

    # Persist the pass marker + hash so a resume short-circuits via state.
    if state is not None:
        passed = _state_field(state, "shorts_ost_audit_passed")
        if passed is None:
            passed = {}
            state["shorts_ost_audit_passed"] = passed
        passed[date] = True
        hashes = _state_field(state, "shorts_ost_audit_hash")
        if hashes is None:
            hashes = {}
            state["shorts_ost_audit_hash"] = hashes
        hashes[date] = current_hash
    # Bind the marker so a later changed burn reads a stale binding.
    if confirmed_file.exists():
        stamp_confirmed(confirmed_file, current_hash)


def format_superlong_ost_audit(date: str, parts: Iterable[dict]) -> dict:
    """Build the per-part OST audit payload for superlong day parts.

    The per-LF audit runs before any superlong part exists, so the day-level
    ``location`` string burned as every part's OST title, plus each part's
    own chapter-derived ``lower_thirds`` town labels, had zero pre-burn
    review — the only trace was the post-burn ``record["lower_thirds"]``,
    and the title was never recorded anywhere. Mirrors
    :func:`format_shorts_ost_audit`'s shape; ``parts`` entries carry
    ``part_index``, ``part_count``, ``ost_text`` (the burned title —
    identical for every part of a day) and ``lower_thirds`` (this part's
    own recurring place-label events). Order is normalised by
    ``part_index`` for a stable operator-visible JSON.
    """
    parts_sorted = sorted(
        (
            {
                "part_index": int(p.get("part_index", 0)),
                "part_count": int(p.get("part_count", 0)),
                "ost_text": str(p.get("ost_text", "") or ""),
                "lower_thirds": [
                    {"label": str(lt.get("label", "")), "time_sec": float(lt.get("time_sec", 0.0))}
                    for lt in (p.get("lower_thirds") or [])
                ],
            }
            for p in parts
        ),
        key=lambda e: e["part_index"],
    )
    return {
        "date": str(date),
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "parts": parts_sorted,
    }


def write_superlong_ost_audit(output_base: Path, date: str, parts: Iterable[dict]) -> Path:
    """Write the superlong-parts audit to
    ``<output_base>/superlong_ost_audit_<date>.json``. Returns the written path."""
    output_base = Path(output_base)
    path = output_base / f"superlong_ost_audit_{date}.json"
    atomic_write_json(path, format_superlong_ost_audit(date, parts), sort_keys=True)
    return path


def compute_superlong_ost_audit_hash(payload: dict) -> str:
    """Stable hash of the SUPERLONG audit content (excludes ``generated_at``).

    Mirror of :func:`compute_ost_audit_hash` — a material change to the
    burned title or any part's lower-thirds reopens the pre-burn gate.
    """
    content = {"date": payload.get("date"), "parts": payload.get("parts", [])}
    serialised = json.dumps(content, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialised).hexdigest()


def superlong_gate_or_raise(output_base: Path, date: str, parts, state) -> list:
    """Pre-burn OST review gate for superlong day parts — the THIRD
    burned-text surface, mirror of :func:`shorts_gate_or_raise`. The day's
    burned title (``location``) and each part's chapter-derived
    ``lower_thirds`` are built one statement before ``apply_ost``; this gate
    runs first, over EVERY part, so a wrong burn is caught upstream of the
    irreversible NVENC re-encode. Confirmed via a
    ``superlong_ost_audit_<date>.confirmed`` marker or the shared
    ``state["ost_audit_confirmed_this_run"]`` — one operator signal covers
    every burned-text surface. A material change to the title or any part's
    lower-thirds reopens the gate; empty ``parts`` -> no gate.

    Returns the SAME ``parts`` list back unchanged — the structural
    guarantee the caller's burn loop consumes (a gate-returns-burn-list
    pattern): a part can only reach ``apply_ost`` by having been
    in the list this gate just audited, never re-derived independently.
    """
    parts = list(parts or [])
    if not parts:
        return []
    payload = format_superlong_ost_audit(date, parts)
    current_hash = compute_superlong_ost_audit_hash(payload)

    cli_confirmed = bool(_state_field(state, "ost_audit_confirmed_this_run"))
    confirmed_file = Path(output_base) / f"superlong_ost_audit_{date}.confirmed"

    # Bind the ``.confirmed`` marker to the hash it confirmed so
    # ``exists()`` alone can't pass a materially changed, never-reviewed burn.
    prior_hash = read_confirmed_hash(confirmed_file)
    stale = confirmation_is_stale(prior_hash, current_hash)

    gate_passes = False
    if stale:
        gate_passes = False  # explicit: a material burn-text change reopens the gate
    elif cli_confirmed:
        gate_passes = True
    elif confirmed_file.exists():
        gate_passes = True

    if not gate_passes:
        sidecar_path = write_superlong_ost_audit(Path(output_base), date, parts)
        raise OstAuditPendingError(
            f"Superlong OST audit pending for day {date}: review {sidecar_path} "
            f"and create {confirmed_file} (or re-run with --ost-audit-confirmed) "
            "before the irreversible superlong part burn."
        )

    if confirmed_file.exists():
        stamp_confirmed(confirmed_file, current_hash)
    return parts


# ── Post-burn OST pixel proof ──────────────────────────────────────────
#: ``ffmpeg_filters.build_ost_filter`` draws the shorts title's first line at
#: this y in a 1080×1920 frame, white inside a 3px black border. Frames are read
#: at native width — downscaling smears the very strokes this check counts.
_OST_LINE_TOP_PX = 280
_OST_BAND_PAD_PX = 8
#: Control band — below a 3-line title block and clear of both CTAs, which share
#: the title's y and would otherwise read as burned title text. The title is
#: centre-anchored, so a centre window always holds it while excluding the frame
#: edges that never carry text.
_CONTROL_BAND_TOP_FRAC = 0.60
_BAND_X_MARGIN_FRAC = 0.2
_WHITE_LUMA = 235
_BLACK_LUMA = 40
#: Sampled inside the title's full-opacity plateau — the burn fades in over
#: [0, 0.5] and out over [4.5, 5.0] — and before any CTA can appear. A clip too
#: short to reach the last one simply reads no frames, which passes.
_POSTBURN_SAMPLE_TIMES = (1.0, 2.5, 4.0)
_POSTBURN_TIMEOUT_S = 60
#: A band carries text when this share of it is near-white AND this share is
#: near-black — white glyphs inside a black border is the one signature footage
#: does not counterfeit at density — and that much more than the control band,
#: so a high-contrast scene cannot pass a title that never burned. Loose on
#: purpose: a four-letter county measures ~4%, a wrapped arc ~18%, footage 0.00%.
_MIN_TEXT_SCORE = 0.01
_TEXT_SCORE_LIFT = 3.0


class OstPostBurnError(RuntimeError):
    """A rendered short's OST band provably carries no burned text.

    Raised from ``run_day_pipeline`` after the shorts burn but UPSTREAM of
    phase_d's upload — the last point a title-less short can be caught. A
    pipeline-control hard fail like :class:`OstAuditPendingError`, not a per-day
    skip: the shorts are on disk, so the operator fixes the burn and re-runs.
    """


def _text_score(gray, top: int, height: int) -> float:
    """How strongly a band shows the burn's white-on-black signature: the smaller
    of its near-white and near-black pixel shares. A bright sky supplies one, a
    deep shadow the other, only composited glyphs supply both at once."""
    band = gray[top : top + height, :]
    margin = int(band.shape[1] * _BAND_X_MARGIN_FRAC)
    band = band[:, margin : band.shape[1] - margin]
    if band.size == 0:
        return 0.0
    return min(float((band >= _WHITE_LUMA).mean()), float((band <= _BLACK_LUMA).mean()))


def ost_text_present_in_pixels(short_path) -> tuple[bool, str]:
    """Does this rendered short's title band actually carry burned OST text?

    Samples frames from inside the title's full-opacity window and measures the
    band the burn writes to against a control band in the SAME frame — same
    scene, same exposure, so only the composite separates them.

    Returns ``(True, reason)`` when the pixels corroborate the burn AND whenever
    the question is unanswerable: a missing or unreadable file, a clip too short
    to sample, a control band already saturated with the signature. Absence of
    measurement is not evidence of a title-less ship, and a gate that fails a
    fully-rendered day over a measurement it could not take is worse than the
    hole it closes. One clearly-burned frame is proof; it fails only when EVERY
    measurable frame shows the band bare."""
    from .ffmpeg_filters import LOC_FONT_SIZE

    short_path = Path(short_path)
    if not short_path.exists() or short_path.stat().st_size == 0:
        return True, "no rendered short to read"
    frames = _gray_frames(
        short_path, _POSTBURN_SAMPLE_TIMES, SHORTS_WIDTH, timeout=_POSTBURN_TIMEOUT_S
    )
    if not frames:
        return True, "unreadable, or too short to sample the title window"

    best = 0.0
    for gray in frames:
        scale = gray.shape[0] / SHORTS_HEIGHT
        height = max(1, int((LOC_FONT_SIZE + _OST_BAND_PAD_PX) * scale))
        title = _text_score(gray, int(_OST_LINE_TOP_PX * scale), height)
        control = _text_score(gray, int(gray.shape[0] * _CONTROL_BAND_TOP_FRAC), height)
        if control >= _MIN_TEXT_SCORE:
            return True, "scene is too high-contrast to distinguish burned text"
        if title >= max(_MIN_TEXT_SCORE, _TEXT_SCORE_LIFT * control):
            return True, f"title band carries burned text ({title:.1%} of the band)"
        best = max(best, title)
    return False, (
        f"no burned text in the title band across {len(frames)} sampled frame(s) "
        f"(strongest {best:.2%}, expected at least {_MIN_TEXT_SCORE:.0%})"
    )


def postburn_ost_gate_or_raise(date: str, entries) -> None:
    """Post-burn proof: every rendered short must show its title in PIXELS.

    The audit entries record what each short was ASKED to burn — an attestation
    that reads identically whether drawtext composited the title or the filter
    chain dropped it. This opens the artefacts and raises
    :class:`OstPostBurnError` for any short whose title band is provably bare,
    upstream of phase_d's upload. Escalates rather than warning, unlike the audit
    WRITE beside it: a lost audit file is a lost report, a title-less short is an
    irreversible publish. Safe only because :func:`ost_text_present_in_pixels` is
    one-sided — it passes everything it cannot measure, so no ffmpeg hiccup or
    short clip fails a day. Entries with no ``path`` (the pre-burn copy, written
    before any short exists) are skipped."""
    failures = []
    for entry in entries or []:
        path = str((entry or {}).get("path") or "")
        if not path:
            continue
        present, why = ost_text_present_in_pixels(path)
        if not present:
            failures.append(f"rank {entry.get('rank')} ({Path(path).name}): {why}")
    if not failures:
        return
    raise OstPostBurnError(
        f"Shorts OST post-burn check FAILED for day {date}: {len(failures)} rendered "
        f"short(s) carry no burned title where the audit says they do — "
        f"{'; '.join(failures)}. Do NOT upload; re-run the shorts burn (--fresh) "
        "after fixing the OST composite."
    )
