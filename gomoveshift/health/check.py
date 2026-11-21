"""``run_check``: run the detection rules, reconcile incident rows, push each new one once."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from . import cadence, drives
from .store import code_sha, load_all, utc_iso, write_row

logger = logging.getLogger(__name__)

_EVENTS = {
    "drive_failed": "health_drive_failed",
    "lf_slot_unbooked": "health_lf_slot_unbooked",
    "lf_slot_uncovered": "health_lf_slot_uncovered",
    "queue_stalled": "health_queue_stalled",
    "placeless_label_unverifiable": "health_placeless_label_unverifiable",
    "failure_unattended": "health_failure_unattended",
    "queued_short_refused": "health_queued_short_refused",
    "delivery_failed_permanent": "health_delivery_failed_permanent",
    "short_bad_run": "health_short_bad_run",
    "backlog_storage_refused": "health_backlog_storage_refused",
    "backlog_store_orphan": "health_backlog_store_orphan",
    "backlog_unlisted_footage": "health_backlog_unlisted_footage",
    "backlog_exclusion_invalid": "health_backlog_exclusion_invalid",
    "seo_feedback_stale": "health_seo_feedback_stale",
    "short_text_sync_refused": "health_short_text_sync_refused",
    "short_text_sync_crash": "health_short_text_sync_crash",
}

#: Kinds ``health sync-short-text`` owns end to end: it alone files, resolves and pushes them
#: (it holds a lock around its store write; ``run_check`` takes none, so it never writes, pushes
#: or resolves one). They are listed, and an open one keeps ``health check`` at exit 1.
_SYNC_OWNED = ("short_text_sync_refused", "short_text_sync_crash")

#: A queued Short the final-text review still refuses this close to its slot (or past it)
#: will fail at dispatch and its slot is never refilled; ``dispatch-queue`` already tried
#: to repair it, so a human must act.
QUEUED_REFUSED_WITHIN_H = 48

#: A YouTube row that ended ``failed_permanent`` this recently (``updated_at``) is "new": rule 8
#: raises for it once, and stops when it ages out, is recovered, or is one of JP's kept rows.
FAILED_PERMANENT_NEW_WITHIN_H = 72

#: ``health heal`` retries every open drive_failed row at least every 24 h (its backoff cap);
#: a row with no launch for 26 h (that cap plus two cron ticks) is one nothing is retrying.
UNATTENDED_AFTER_H = 26

#: Consecutive failed launches (no success between) that alarm a drive however recent the last
#: launch. Three matches the transient retry budget (``retry_policy.BUDGET``): a drive that has
#: relaunched three times and failed each time is not being healed, and the 26 h clock never
#: fires on it because every relaunch refreshes it (Courtmacsherry, 2026-11-17..19: ~10 failed
#: resumes across 9 SHAs with the alarm marked resolved).
UNATTENDED_AFTER_FAILURES = 3

_BACKFILL = ("resume_recipe", "log_path")
#: Fields of an already-open grouped ``seo_feedback_stale`` row that follow the live finding.
_REFRESH = ("detail", "count", "shorts", "decided_at")


#: Camera file names a grouped footage push lists before "and N more".
FOOTAGE_PUSH_NAMES = 10


def _footage_line(rows: List[Dict[str, Any]]) -> str:
    names = [str(r.get("name") or r["key"]) for r in rows]
    shown = ", ".join(names[:FOOTAGE_PUSH_NAMES])
    more = f" and {len(names) - FOOTAGE_PUSH_NAMES} more" if len(names) > FOOTAGE_PUSH_NAMES else ""
    return (
        f"{len(rows)} camera files are in no drive_backlog.yaml row and no "
        f"drive_backlog_excluded.yaml entry: {shown}{more}. Queue them as rows, or exclude "
        "them with a reason."
    )


def _line(row: Dict[str, Any]) -> str:
    if row["kind"] == "drive_failed":
        return drives.summary(row)
    if row["kind"] == "failure_unattended":
        return row["detail"]
    if row["kind"] == "queued_short_refused":
        return (
            f"Queued YouTube Short {row['delivery_id']} ({row['project']} #{row['short_index']}) "
            f"is slotted {row['scheduled_at']} and the final-text review refuses it: "
            f"{row['reason']}. dispatch-queue could not repair it; it will fail at its slot."
        )
    if row["kind"] == "delivery_failed_permanent":
        return (
            f"YouTube delivery {row['delivery_id']} ({row['project']} #{row['short_index']}, slot "
            f"{row['scheduled_at']}) failed permanently ({row['error_class']}); its slot is not "
            "refilled and it is not retried. dispatch-queue requeues a review refusal once the "
            "review passes."
        )
    if row["kind"] == "backlog_storage_refused":
        return (
            f"health backlog could not launch {row.get('project') or 'the next row'} for "
            f"{row['region']}: {row['reason']}. No {row['region']} drive launches until space is freed."
        )
    if row["kind"] == "lf_slot_unbooked":
        return f"LF slot {row['slot']} ({row['region']}) is inside 48 h with nothing booked and a drive ready."
    if row["kind"] == "lf_slot_uncovered":
        nxt = row.get("next_drive")
        todo = (
            f"next drive to audit: {nxt}"
            if nxt
            else f"drive_backlog.yaml has no {row['region']} row left to audit"
        )
        return f"LF slot {row['slot']} ({row['region']}) has no ready drive; {todo}."
    if row["kind"] == "placeless_label_unverifiable":
        why = f"claims {', '.join(row['claims'])}" if row.get("claims") else "no claim words"
        return (
            f"drive_backlog.yaml row {row['project']} carries placeless_label {row['label']!r} "
            f"({why}): a placeless label is a last resort, so the row never launches unattended. "
            "Give it bracket_gpx (the fixes files of the drives before and after) instead; run "
            "--placeless-label by hand only when no bracket exists."
        )
    if row["kind"] == "backlog_store_orphan":
        return (
            f"Store project {row['project']} is in no drive_backlog.yaml row and no "
            "drive_backlog_excluded.yaml entry's project: add the row, or exclude it with a reason."
        )
    if row["kind"] == "backlog_unlisted_footage":
        return (
            f"Camera file {row['name']} in {row['directory']} is in no drive_backlog.yaml row "
            "and no drive_backlog_excluded.yaml entry: queue it as a row, or exclude it with a reason."
        )
    if row["kind"] == "backlog_exclusion_invalid":
        where = "file" if row["entry"] is None else f"entry {row['entry']}"
        return f"drive_backlog_excluded.yaml {where} breaks the contract: {row['detail']}."
    if row["kind"] == "short_bad_run":
        runs = ", ".join(f"{s['project']}#{s['index']} ({s['views']})" for s in row["shorts"])
        return (
            f"{row['n']} settled Shorts in a row are under {row['threshold_views']:g} day-7 views "
            f"(channel median {row['median_views']:g}): {runs}; causes are in the incident row."
        )
    if row.get("detail"):
        return str(row["detail"])
    return f"{row['due_deliveries']} publishing deliveries are due and nothing has touched the queue in {row['quiet_hours']} h."


def _when(text: Any) -> Optional[datetime]:
    try:
        moment = datetime.fromisoformat(str(text))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _failed_streak(attempts: List[Dict[str, Any]]) -> int:
    """Consecutive launched attempts whose run failed, newest back to the last success. A launch
    still running, a refusal and a pre-flight failure (no work started) neither add nor reset."""
    timed = [(w, a) for a in attempts if (w := _when(a.get("at"))) is not None]
    streak = 0
    for _w, att in sorted(timed, key=lambda t: t[0], reverse=True):
        if att.get("action") != "launched":
            continue
        if att.get("outcome") == "succeeded":
            break
        if att.get("outcome") == "failed":
            streak += 1
    return streak


def _budget_spent(rows: List[Dict[str, Any]], attempts: List[Dict[str, Any]]) -> Optional[str]:
    """The retry policy's own refusal text when the drive's failure class has used its launches
    under the fingerprint it failed under, and the newest launch is over (so no relaunch is due);
    else ``None``. Publish-only is heal's refusal by design (``health publish`` owns it), and a
    row filed before fingerprints has no class: neither is a spent budget."""
    from . import retry_policy

    stamped = [r for r in rows if (r.get("failure_fingerprint") or {}).get("hash")]
    if not stamped:
        return None
    stamp = max(stamped, key=lambda r: str(r.get("failed_at") or ""))["failure_fingerprint"]
    klass, fph = stamp.get("class") or retry_policy.UNKNOWN, stamp["hash"]
    launches = [a for a in attempts if a.get("action") == "launched"]
    used = retry_policy.launched_count(launches, fph)
    newest = max(launches, key=lambda a: str(a.get("at") or ""), default=None)
    if klass == retry_policy.PUBLISH_ONLY or not used or (newest or {}).get("outcome") != "failed":
        return None
    return retry_policy.refusal(klass, fph, launched=used, failed_under=False)


def unattended_incidents(
    drive_rows: List[Dict[str, Any]], now: datetime, now_iso: str, notes: List[str]
) -> List[Dict[str, Any]]:
    """Rule 6: one ``failure_unattended`` incident per open drive_failed row nothing is retrying.

    Unattended: the newest ``launched`` attempt across every row sharing the ``resume_recipe``
    (siblings are one drive's stages) is over ``UNATTENDED_AFTER_H`` old -- or there is none and
    ``failed_at`` is -- or the row's newest attempt is ``refused`` (heal can never relaunch it).
    An operator hold (``no-auto-resume``) is deliberate: noted as held, never an alarm."""
    from . import resume
    from .store import new_incident, safe_name

    limit = timedelta(hours=UNATTENDED_AFTER_H)
    launched: Dict[Any, datetime] = {}
    by_group: Dict[Any, List[Dict[str, Any]]] = {}
    for row in drive_rows:
        by_group.setdefault(row.get("resume_recipe") or row["key"], []).append(row)
    for row in drive_rows:
        for att in row.get("resume_attempts") or []:
            when = _when(att.get("at"))
            if att.get("action") == "launched" and when:
                key = row.get("resume_recipe") or row["key"]
                launched[key] = max(when, launched.get(key, when))
    out: List[Dict[str, Any]] = []
    # one incident per drive: its heal resumes and its stage siblings are that drive
    emitted: set = set()
    for row in drive_rows:
        root = min(drives.heal_origins(str(row["drive"])) or {str(row["drive"])}, key=len)
        group = row.get("resume_recipe") or root
        if group in emitted or root in emitted:
            continue
        recipe = row.get("resume_recipe")
        attempts = row.get("resume_attempts") or []
        last = _when(attempts[-1].get("at")) if attempts else None
        newest = launched.get(recipe or row["key"])
        group_rows = by_group.get(recipe or row["key"], [row])
        group_atts = [a for r in group_rows for a in r.get("resume_attempts") or []]
        streak, spent = _failed_streak(group_atts), _budget_spent(group_rows, group_atts)
        if attempts and attempts[-1].get("action") == "refused":
            reason = attempts[-1].get("reason") or "no reason given"
            why = f"heal refused it ({reason}) and can never relaunch it"
        elif streak >= UNATTENDED_AFTER_FAILURES:
            why = (
                f"its last {streak} launches in a row failed (alarm at "
                f"{UNATTENDED_AFTER_FAILURES}) and heal is not fixing it"
            )
        elif spent:
            why = f"its retry budget is spent ({spent})"
        else:
            ref = newest or _when(row.get("failed_at"))
            if ref is None or now - ref <= limit:
                continue
            what = "last launched attempt" if newest else "failed, with no attempt,"
            why = f"{what} {ref.isoformat()} is over {UNATTENDED_AFTER_H} h ago and nothing is retrying it"
        if recipe and resume.is_held(Path(recipe)):
            notes.append(f"{row['drive']} is held (no-auto-resume): {why}")
            continue
        where = " / ".join(x for x in (row.get("phase"), row.get("gate")) if x) or "unknown gate"
        stamp = last or newest or _when(row.get("failed_at"))
        emitted.update((group, root))
        out.append(
            new_incident(
                "failure_unattended",
                f"failure_unattended:{root}",
                f"failure_unattended_{safe_name(root)}.json",
                now_iso,
                drive=root,
                last_attempt_at=stamp.isoformat() if stamp else None,
                detail=f"Drive {root} failed at {where} and is unattended: {why}.",
            )
        )
    return out


def queued_short_refused_incidents(
    db_path: Path, now: datetime, now_iso: str, notes: List[str]
) -> List[Dict[str, Any]]:
    """Rule 7: one ``queued_short_refused`` incident per queued, non-shadow YouTube row whose
    slot is within ``QUEUED_REFUSED_WITHIN_H`` of ``now`` (or past) and whose final-text review
    still refuses it. Read-only: the queue opens ``mode=ro`` and the review writes nothing.
    A row whose sidecar cannot be read proves nothing either way: noted, no incident."""
    import sqlite3

    from ..publishing.queue_repair import QUEUED_YOUTUBE_SQL, row_refusal
    from .store import new_incident

    limit = now + timedelta(hours=QUEUED_REFUSED_WITHIN_H)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(QUEUED_YOUTUBE_SQL).fetchall()
    finally:
        conn.close()
    out: List[Dict[str, Any]] = []
    for row in rows:
        slot = _when(row["scheduled_at"]) if row["scheduled_at"] else None
        if slot is None or slot > limit or not row["sidecar_path"]:
            continue
        try:
            reason = row_refusal(row)
        except Exception as exc:  # noqa: BLE001 -- unreadable sidecar: reported, never fatal
            notes.append(f"rule 7: delivery {row['id']} not reviewed: {type(exc).__name__}: {exc}")
            continue
        if reason is None:
            continue
        out.append(
            new_incident(
                "queued_short_refused",
                f"queued_short_refused:{row['id']}",
                f"queued_short_refused_{row['id']}.json",
                now_iso,
                delivery_id=row["id"],
                project=row["project_name"],
                short_index=row["short_index"],
                scheduled_at=slot.isoformat(),
                reason=" ".join(reason.split()),
            )
        )
    return out


def failed_permanent_incidents(
    db_path: Path, now: datetime, now_iso: str, notes: List[str]
) -> List[Dict[str, Any]]:
    """Rule 8: one ``delivery_failed_permanent`` incident per non-shadow YouTube delivery that
    ended ``failed_permanent`` within ``FAILED_PERMANENT_NEW_WITHIN_H`` of ``now`` and has not
    been recovered. JP's kept rows (``JP_KEPT_FAILED_PERMANENT_IDS``) never raise. Read-only
    (queue opened ``mode=ro``)."""
    import sqlite3

    from ..publishing.queue_slot_refill import JP_KEPT_FAILED_PERMANENT_IDS
    from .store import new_incident

    cutoff = now - timedelta(hours=FAILED_PERMANENT_NEW_WITHIN_H)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT d.* FROM deliveries d WHERE d.platform='youtube' AND d.shadow=0 "
            "AND d.status='failed_permanent' AND NOT EXISTS "
            "(SELECT 1 FROM deliveries r WHERE r.recovers_delivery_id=d.id) ORDER BY d.id"
        ).fetchall()
    finally:
        conn.close()
    out: List[Dict[str, Any]] = []
    for row in rows:
        ended = _when(row["updated_at"])
        if row["id"] in JP_KEPT_FAILED_PERMANENT_IDS or ended is None or ended < cutoff:
            continue
        out.append(
            new_incident(
                "delivery_failed_permanent",
                f"delivery_failed_permanent:{row['id']}",
                f"delivery_failed_permanent_{row['id']}.json",
                now_iso,
                delivery_id=row["id"],
                project=row["project_name"],
                short_index=row["short_index"],
                scheduled_at=row["scheduled_at"],
                error_class=row["error_class"] or "unknown",
            )
        )
    return out


def _next_drives(store: Any, regions: List[str], notes: List[str]) -> Dict[str, Optional[str]]:
    """Region -> the ``drive_backlog.yaml`` project to audit next (rule 2b names it)."""
    from seo_manager.scheduling import backlog

    from ..paths import drive_backlog_path

    path = drive_backlog_path()
    if not path.is_file():
        notes.append(f"{path} absent: rule 2b pages without naming a drive")
        return {}
    rows = backlog.load(path)
    return {
        region: (row.project if (row := backlog.next_row(rows, region, store)) else None)
        for region in regions
    }


def _backlog_rows(notes: List[str]) -> Optional[List[Any]]:
    """The ``drive_backlog.yaml`` rows, or ``None`` (noted) when unreadable: an unreadable
    backlog proves nothing either way, so it must not close an open incident."""
    try:
        from seo_manager.scheduling import backlog

        from ..paths import drive_backlog_path

        path = drive_backlog_path()
        if not path.is_file():
            notes.append(f"{path} absent: rule 5 not evaluated")
            return None
        return list(backlog.load(path))
    except Exception as exc:  # noqa: BLE001 -- a broken rule is reported, never fatal
        notes.append(f"rule 5 not evaluated: {type(exc).__name__}: {exc}")
        return None


def placeless_label_incidents(rows: List[Any], now_iso: str) -> List[Dict[str, Any]]:
    """Rule 5: one incident per backlog row carrying a ``placeless_label``, which the launcher
    always refuses (``launch.placeless_label_skip_reason``): a placeless label is a last resort
    and the row should carry ``bracket_gpx`` instead. It is surfaced here, open for as long as
    the yaml carries the label, instead of parking silently."""
    from ..pipeline.validation.content_gates import placeless_label_claimed_words
    from .launch import placeless_label_skip_reason
    from .store import new_incident, safe_name

    out: List[Dict[str, Any]] = []
    for row in rows:
        label = getattr(row, "placeless_label", "") or ""
        reason = placeless_label_skip_reason(label) if label else None
        if not reason:
            continue
        project = row.project
        out.append(
            new_incident(
                "placeless_label_unverifiable",
                f"placeless_label_unverifiable:{project}",
                f"placeless_label_{safe_name(str(project))}.json",
                now_iso,
                project=project,
                label=label,
                claims=placeless_label_claimed_words(label),
                reason=reason,
            )
        )
    return out


def _accounting_rows(
    config: Any, rows: List[Any], now_iso: str, notes: List[str]
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[str]]:
    """Rules 10 and 11 (and the exclusions file's own contract), for a readable manifest.

    Returns ``(found, held)``. A rule that could not be evaluated sets no key in ``found`` (so
    its open incidents stay open); ``held`` lists the folders (comparison form) rule 11 could not
    list, whose open ``backlog_unlisted_footage`` incidents stay live too: an unreadable input
    must not close an incident (D: offline then back must not re-open and re-notify it)."""
    from ..paths import drive_backlog_excluded_path
    from . import accounting

    found: Dict[str, List[Dict[str, Any]]] = {}
    held: List[str] = []
    try:
        entries, problems = accounting.load_exclusions(drive_backlog_excluded_path(), rows)
        found["backlog_exclusion_invalid"] = accounting.exclusion_problem_incidents(
            problems, now_iso
        )
    except Exception as exc:  # noqa: BLE001 -- a broken rule is reported, never fatal
        notes.append(f"rules 10 and 11 not evaluated: {type(exc).__name__}: {exc}")
        return found, held
    if any(p["entry"] is None for p in problems):
        notes.append(
            "drive_backlog_excluded.yaml unreadable or not a list: rules 10 and 11 not evaluated"
        )
        return found, held
    try:
        found["backlog_unlisted_footage"] = accounting.unlisted_footage_incidents(
            rows,
            entries,
            now_iso,
            notes,
            held,
            [src for p in problems for src in p.get("sources", ())],
        )
    except Exception as exc:  # noqa: BLE001
        notes.append(f"rule 11 not evaluated: {type(exc).__name__}: {exc}")
    storage = Path(config.seo.storage_path)
    if not storage.is_absolute():
        notes.append(
            f"seo.storage_path is relative ({config.seo.storage_path!r}); rule 10 not evaluated"
        )
        return found, held
    try:
        from ..pipeline.seo_slots import STORE_FILENAME, waypost_cadence

        store_path = storage / STORE_FILENAME
        if not store_path.exists():
            notes.append("schedule.sqlite3 absent: rule 10 (backlog_store_orphan) not evaluated")
            return found, held
        cad = waypost_cadence(config)
        projects = accounting.store_projects(store_path, cad.tz, cad.regions, notes)
        if projects is not None:
            found["backlog_store_orphan"] = accounting.store_orphan_incidents(
                projects, rows, entries, now_iso
            )
    except Exception as exc:  # noqa: BLE001
        notes.append(f"rule 10 not evaluated: {type(exc).__name__}: {exc}")
    return found, held


def _cadence_rows(
    config: Any, ledger_path: Optional[Path], now: datetime, now_iso: str, notes: List[str]
):
    """Rules 2 and 4. ``found`` maps kind -> rows, only for a rule that actually ran."""
    found: Dict[str, List[Dict[str, Any]]] = {}
    storage = Path(config.seo.storage_path)
    if not storage.is_absolute():
        notes.append(
            f"seo.storage_path is relative ({config.seo.storage_path!r}); rules 2 and 4 not evaluated"
        )
        return found
    try:
        from seo_manager.scheduling.store import ScheduleStore

        from ..pipeline.content_ledger import ContentLedger
        from ..pipeline.schedule_report import build_schedule_report
        from ..pipeline.seo_slots import STORE_FILENAME, waypost_cadence

        store_path = storage / STORE_FILENAME
        if store_path.exists():
            cad = waypost_cadence(config)
            store = ScheduleStore(store_path, cad.tz, cad.regions)
            ready = {r: store.ready_depth(r) for r in sorted(cad.regions)}
            report = build_schedule_report(ContentLedger.load(ledger_path), config, after=now)
            found["lf_slot_unbooked"] = cadence.unbooked_slots(report, ready, now, now_iso)
            found["lf_slot_uncovered"] = cadence.uncovered_slots(
                report, ready, _next_drives(store, sorted(cad.regions), notes), now, now_iso
            )
        else:
            notes.append("schedule.sqlite3 absent: no readiness signal, rule 2 not evaluated")
    except Exception as exc:  # noqa: BLE001 -- a broken rule is reported, never fatal to the others
        notes.append(f"rule 2 not evaluated: {type(exc).__name__}: {exc}")
    db = storage / "social_queue.sqlite3"
    if db.exists():
        try:
            row = cadence.queue_stalled(db, now, now_iso)
            found["queue_stalled"] = [row] if row else []
        except Exception as exc:  # noqa: BLE001
            notes.append(f"rule 4 not evaluated: {type(exc).__name__}: {exc}")
        try:
            found["queued_short_refused"] = queued_short_refused_incidents(db, now, now_iso, notes)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"rule 7 not evaluated: {type(exc).__name__}: {exc}")
        try:
            found["delivery_failed_permanent"] = failed_permanent_incidents(db, now, now_iso, notes)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"rule 8 not evaluated: {type(exc).__name__}: {exc}")
    else:
        notes.append("social_queue.sqlite3 absent: rule 4 not evaluated")
    return found


def _bad_run_rows(
    config: Any, tracking_root: Optional[Path], now: datetime, now_iso: str, notes: List[str]
) -> Optional[List[Dict[str, Any]]]:
    """Rule 9, or ``None`` (noted) when it could not run: an unreadable record proves nothing
    either way, so it must not close an open incident."""
    storage = Path(config.seo.storage_path)
    if not storage.is_absolute():
        notes.append("seo.storage_path is relative; rule 9 (short_bad_run) not evaluated")
        return None
    try:
        from .bad_run import bad_run_rows

        return bad_run_rows(storage, tracking_root, now.date(), now_iso)
    except Exception as exc:  # noqa: BLE001 -- a broken rule is reported, never fatal
        notes.append(f"rule 9 (short_bad_run) not evaluated: {type(exc).__name__}: {exc}")
        return None


def _seo_feedback_rows(
    config: Any,
    tracking_root: Optional[Path],
    now: datetime,
    now_iso: str,
    notes: List[str],
    open_rows: Mapping[str, Dict[str, Any]] = {},
) -> Tuple[Optional[List[Dict[str, Any]]], List[str]]:
    """Rule 13 (``seo_feedback_stale``): ``(rows, held)``. ``rows`` is ``None`` (noted) when no
    condition could run. ``held`` lists key prefixes (a condition's, or one row's) whose input
    could not be read: their open incidents stay live, never closed by an unreadable source.
    A missing decisions file is condition (a)'s finding only and holds nothing."""
    from . import seo_feedback as sf  # noqa: PLC0415

    storage = Path(config.seo.storage_path)
    if not storage.is_absolute():
        notes.append("seo.storage_path is relative; rule 13 (seo_feedback_stale) not evaluated")
        return None, []
    rows: List[Dict[str, Any]] = []
    held: List[str] = []
    try:
        rows += sf.decisions_incidents(storage, now, now_iso)
    except Exception as exc:  # noqa: BLE001 -- a broken rule is reported, never fatal
        notes.append(f"seo_feedback_stale decisions not evaluated: {type(exc).__name__}: {exc}")
        held.append(sf.KEY_DECISIONS)
    try:
        from ..tracking.store import TrackingStore  # noqa: PLC0415

        store = TrackingStore(tracking_root)
        records = [store.load(p) for p in store.list_projects()]
        metric_rows = sf.metrics_incidents(records, now.date(), now_iso)
        rows += metric_rows
        prior = open_rows.get(sf.KEY_METRICS)
        if prior is not None:
            # A Short past the lookback cannot be in the current row, so a listed one that is
            # now past it was dropped by ageing (the refresh removes it: said once).
            aged = sf.aged_out_count(records, now.date(), prior.get("shorts") or [])
            if aged:
                notes.append(
                    f"seo_feedback_stale metrics: {aged} listed Short(s) aged out of the "
                    f"{sf.METRICS_LOOKBACK_DAYS}-day lookback still missing metrics (not a fix)"
                )
    except Exception as exc:  # noqa: BLE001
        notes.append(f"seo_feedback_stale metrics not evaluated: {type(exc).__name__}: {exc}")
        held.append(sf.KEY_METRICS)
    return rows, held


def _reconcile_publishes(config: Any, jobs_dir: Path, notes: List[str]) -> List[Dict[str, str]]:
    """Close the Waypost claims of finished ``health publish`` jobs (idempotent)."""
    if not Path(config.seo.storage_path).is_absolute():
        return []
    try:
        from .publish import open_schedule_store, reconcile_publishes

        store, _cad = open_schedule_store(config)
        return reconcile_publishes(store, jobs_dir) if store else []
    except Exception as exc:  # noqa: BLE001 -- a broken reconcile is reported, never fatal
        notes.append(f"publish reconcile not run: {type(exc).__name__}: {exc}")
        return []


def _default_stamp(config: Any, code_root: Path) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    from . import retry_policy  # noqa: PLC0415

    return lambda row: retry_policy.failure_stamp(
        row, python=retry_policy.service_python(code_root), config=config
    )


def run_check(
    config: Any,
    *,
    incidents_dir: Path,
    jobs_dir: Path,
    code_root: Path,
    staging_sha: Optional[str] = None,
    ledger_path: Optional[Path] = None,
    tracking_root: Optional[Path] = None,
    now: Optional[datetime] = None,
    notify_fn: Optional[Callable[[str, Dict[str, Any], str], bool]] = None,
    stamp_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Returns ``{"incidents": open rows, "new": keys written, "pushed": keys pushed,
    "resolved": keys closed, "unreadable": int, "publish_reconciled": [claims closed], "notes": [str]}``. Never raises on bad input."""
    if notify_fn is None:
        from ..console.notify import notify as notify_fn  # noqa: PLC0415
    now = now or datetime.now(timezone.utc)
    now_iso = utc_iso(now.timestamp())
    notes: List[str] = []
    existing, bad_rows = load_all(incidents_dir)
    by_key = {r["key"]: r for r in existing}
    sha = staging_sha or code_sha(code_root)

    drive_rows, succeeded, bad_jobs = drives.detect(jobs_dir, sha, now_iso)
    # The fingerprint a failure is FILED under: heal retries by it, so it is taken here, at
    # detection, never by heal at a later tick on newer code. Rows already on file keep theirs
    # (or none: a legacy row, which heal treats as never fingerprinted).
    for new_row in (r for r in drive_rows if r["key"] not in by_key):
        new_row["failure_fingerprint"] = (stamp_fn or _default_stamp(config, code_root))(new_row)
    reconciled = _reconcile_publishes(config, jobs_dir, notes)
    found = {"drive_failed": drive_rows, **_cadence_rows(config, ledger_path, now, now_iso, notes)}
    held_dirs: List[str] = []
    backlog_rows = _backlog_rows(notes)
    if backlog_rows is not None:
        found["placeless_label_unverifiable"] = placeless_label_incidents(backlog_rows, now_iso)
        accounted, held_dirs = _accounting_rows(config, backlog_rows, now_iso, notes)
        found.update(accounted)

    bad_run = _bad_run_rows(config, tracking_root, now, now_iso, notes)
    if bad_run is not None:
        found["short_bad_run"] = bad_run

    seo_rows, seo_held = _seo_feedback_rows(
        config,
        tracking_root,
        now,
        now_iso,
        notes,
        {r["key"]: r for r in existing if not r.get("resolved_at")},
    )
    if seo_rows is not None:
        found["seo_feedback_stale"] = seo_rows

    live_drives = {r["key"]: r for r in drive_rows}
    open_drives = [
        r
        for r in existing
        if r.get("kind") == "drive_failed"
        and not r.get("resolved_at")
        and r["key"] in live_drives
        and r.get("job_name") not in succeeded
    ]
    seen = {r["key"] for r in open_drives}
    open_drives += [r for k, r in live_drives.items() if k not in seen and k not in by_key]
    found["failure_unattended"] = unattended_incidents(open_drives, now, now_iso, notes)

    resolved: List[str] = []
    new: List[str] = []
    live_keys = {r["key"] for rows in found.values() for r in rows}
    for row in existing:
        if row.get("resolved_at"):
            continue
        kind = row.get("kind")
        gone = kind in found and row["key"] not in live_keys
        if kind == "backlog_exclusion_invalid" and "backlog_exclusion_invalid:file" in live_keys:
            gone = False  # unreadable file: the per-entry findings could not be re-judged
        if kind == "backlog_unlisted_footage" and held_dirs:
            from .accounting import norm  # noqa: PLC0415

            gone = gone and norm(str(row.get("directory", ""))) not in held_dirs
        if kind == "seo_feedback_stale" and any(
            row["key"] == p or row["key"].startswith(p + ":") for p in seo_held
        ):
            gone = False  # that condition's input was unreadable: it could not be re-judged
        if kind == "drive_failed" and row.get("job_name") in succeeded:
            gone = True
        if gone:
            row["resolved_at"] = now_iso
            write_row(incidents_dir, row)
            resolved.append(row["key"])
    for rows in found.values():
        for row in rows:
            prior = by_key.get(row["key"])
            if prior is not None and not prior.get("resolved_at"):
                # Already on file: keeps its pushed state and resume_attempts, but takes the
                # recipe and log an older reader missed, so heal can reach the drive.
                fill = {k: row[k] for k in _BACKFILL if row.get(k) and not prior.get(k)}
                if row["kind"] == "seo_feedback_stale":
                    # A grouped row's count and listing move: refresh them, never `pushed`.
                    fill.update(
                        {k: row[k] for k in _REFRESH if k in row and row[k] != prior.get(k)}
                    )
                if fill:
                    prior.update(fill)
                    write_row(incidents_dir, prior)
                continue
            if prior is not None and prior.get("resolution_reason"):
                # ``health heal`` closed it with a recorded reason (a recipe-less publish/heal
                # job that is never resumable); re-filing it from the same failed job would
                # undo that closure every tick.
                continue
            write_row(incidents_dir, row)
            by_key[row["key"]] = row
            new.append(row["key"])

    pushed: List[str] = []
    open_rows = [r for r in by_key.values() if not r.get("resolved_at")]
    unpushed = [r for r in open_rows if not r.get("pushed")]
    footage = sorted(
        (r for r in unpushed if r["kind"] == "backlog_unlisted_footage"), key=lambda r: r["key"]
    )
    if len(footage) > 1:  # one push per tick, not one per file: each row keeps its own key
        if (
            notify_fn(
                _EVENTS["backlog_unlisted_footage"], {"count": len(footage)}, _footage_line(footage)
            )
            is not False
        ):
            for row in footage:
                row["pushed"] = True
                row["pushed_at"] = now_iso
                write_row(incidents_dir, row)
                pushed.append(row["key"])
        grouped = {r["key"] for r in footage}
    else:
        grouped = set()
    for row in open_rows:
        if row.get("pushed") or row["key"] in grouped or row["kind"] in _SYNC_OWNED:
            continue
        event = _EVENTS.get(row["kind"], "health_incident")
        if notify_fn(event, {"incident": row["key"]}, _line(row)) is False:
            continue  # attempted and failed: stay unrecorded, the next pass retries
        row["pushed"] = True
        row["pushed_at"] = now_iso
        write_row(incidents_dir, row)
        pushed.append(row["key"])
    return {
        "incidents": sorted(open_rows, key=lambda r: r["key"]),
        "new": new,
        "pushed": pushed,
        "resolved": resolved,
        "unreadable": bad_rows + bad_jobs,
        "publish_reconciled": reconciled,
        "notes": notes,
    }
