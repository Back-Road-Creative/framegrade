"""``health publish``: the one operator command that publishes an audited long-form drive through
Waypost, and the hourly reconcile that closes its claim.

Order, each a stop that leaves the store as it was: the drive must be ``auditing`` in
``schedule.sqlite3`` (or ``failed`` -- a publish job that died before uploading -- which is
reopened to ``auditing`` at the first write) with no upload receipt; no drive job running; its ``resume.json`` recipe must
rebind to the code root's venv and its checkpoint hold no phase the current code re-runs (Phase D
refuses a missing prerequisite, after the slot is booked); and no older ``ready`` drive may sit ahead of it (Waypost claims the
oldest ready drive of a region, never a named one), except one already uploaded outside a claim,
which is retired from its receipts first (Waypost's ``retire_published``). The long-form count comes from the
``drive_plan.json`` beside the recipe (its ``regular_lfs``), not the store, which ``health backlog`` registers
as 1; a corrected count is written back with the reopen, so the claim books one weekly slot per rendered
long-form (none readable -> refuse). Then ``set_status ready`` -> ``claim_or_reason``
-> a detached Phase D job (``--lf-publish-date`` = the claimed slot's local date, ``--youtube-upload
--phase D``, the recipe's own region/mode/GPX flags and ``GMS_INTERMEDIATE_DIR``). A claim of any
other drive, or a launch that fails, is undone. ``reconcile_publishes`` (called by ``health check``)
later calls ``finish_claim`` for each claimed drive whose ``gms-publish-<project>`` job has ended,
and retires each ``ready`` drive that is already uploaded."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from seo_manager.scheduling.claim import Claim, claim_or_reason
from seo_manager.scheduling.store import SlotTaken, StaleDrive

from .. import run_origin
from ..pipeline.replace_scheduled import record_incident, resolve_incidents
from ..publishing.mask_frame_gate import mask_frame_refusal
from . import resume
from .heal import _STALL_S, _job_records, bind_to_code_root, job_runner, stale_phases
from .launch import (
    SUPERSEDED_DIR,
    _uploaded,
    dead_publish_result,
    default_launcher,
    plan_total_lfs,
    read_replace_plan,
    replace_unfinished,
    upload_receipts,
)

JOB_PREFIX = "gms-publish-"
_DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
#: Recipe flags Phase D replaces (with a value) or drops (bare): the audit's no-upload and phase
#: switches, and the dates this publish takes from the claimed slot instead.
_VALUED = ("--publish", "-p", "--lf-day", "--lf-publish-date", "--phase")
_BARE = ("--skip-upload", "--no-youtube-upload", "--youtube-upload", "--fresh", "--force", "--no-tmux")  # fmt: skip


def find_recipe(stages_root: Path, project: str) -> Optional[Path]:
    """``<stages root>/<project>/<project>/_stages/resume.json`` (the layout ``health backlog``
    stages), else any recipe one level under the project's stage root."""
    root = Path(stages_root) / project
    found = sorted(root.glob("*/_stages/resume.json")) + sorted(root.glob("_stages/resume.json"))
    return found[0] if found else None


def phase_d_args(drive_argv: List[str], day: str, date: str) -> List[str]:
    """The recipe's drive argv rebound to Phase D: upload on, dates from the claimed slot."""
    keep: List[str] = []
    skip = False
    for tok in drive_argv:
        if skip:
            skip = False
        elif tok in _VALUED:
            skip = True
        elif tok in _BARE or tok.split("=", 1)[0] in _VALUED:
            continue
        else:
            keep.append(tok)
    return [*keep, "--publish", day, "--lf-day", day, "--lf-publish-date", date,
            "--youtube-upload", "--force", "--no-tmux", "--phase", "D"]  # fmt: skip


def rendered_lf_count(recipe_path: Path) -> Optional[int]:
    """How many long-forms the audit rendered: ``regular_lfs`` in the ``drive_plan.json`` beside
    the recipe. ``None`` when it is missing, unreadable or empty."""
    try:
        plan = json.loads((Path(recipe_path).parent / "drive_plan.json").read_text())
        lfs = plan.get("regular_lfs")
    except (OSError, ValueError, AttributeError):
        return None
    return len(lfs) if isinstance(lfs, list) and lfs else None


def known_video_ids(drives: Any) -> set:
    """Every video id any drive's own files name, whatever its ``publish_at``: its
    ``youtube_uploads.json`` receipts (a replace's new upload is saved with none), the ids in its
    ``replace_plan.json`` (``replace`` / ``kept`` ``video_id``, ``progress`` ``old_id`` /
    ``new_id``) and the receipts of its superseded runs. None of them is an orphan."""
    ids: set = set()
    for d in drives:
        if not d.output_dir:
            continue
        out = Path(d.output_dir)
        files = [out / "youtube_uploads.json", *sorted(out.glob("*/youtube_uploads.json")),
                 *sorted((out.parent / SUPERSEDED_DIR).glob(f"{d.project}-*/youtube_uploads.json"))]  # fmt: skip
        for path in files:
            try:
                held = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            ids.update(r["video_id"] for r in (held.values() if isinstance(held, dict) else ()) if isinstance(r, dict) and r.get("video_id"))  # fmt: skip
        plan = read_replace_plan(out)
        for part in ("replace", "kept"):
            ids.update(r["video_id"] for r in (plan.get(part) or {}).values() if isinstance(r, dict) and r.get("video_id"))  # fmt: skip
        ids.update(i for p in (plan.get("progress") or {}).values() if isinstance(p, dict) for i in (p.get("old_id"), p.get("new_id")) if i)  # fmt: skip
    return ids


def publish_job_times(jobs_dir: Path, project: str, now: datetime) -> tuple:
    """``(first attempt, when the newest dead attempt ended)`` of ``gms-publish-<project>`` jobs as
    aware UTC datetimes (``None`` for none): the window a catalog snapshot must cover. A dead job
    is one :func:`launch.dead_publish_result` calls dead (the backlog retries the same set). The
    window opens when it ENDED, not when it started: the upload happens inside the run, so a
    snapshot fetched mid-run cannot hold it. A dead job with no ``ended_at`` opens it ``now``."""
    made, dead = [], []
    for r in _job_records(Path(jobs_dir)):
        if r.get("name") != f"{JOB_PREFIX}{project}":
            continue
        at = float(r.get("created_at") or 0)
        made.append(at)
        if dead_publish_result(r.get("result")):
            dead.append((at, r.get("result") or {}))
    stamp = lambda t: datetime.fromtimestamp(t, timezone.utc)  # noqa: E731
    ended = None
    if dead:
        try:
            ended = stamp(float(max(dead, key=lambda d: d[0])[1].get("ended_at")))
        except (TypeError, ValueError, OverflowError, OSError):
            ended = now
    return (stamp(min(made)) if made else None, ended)


def once_per_tick(refresh: Callable[[], Any]) -> Callable[[], None]:
    """``refresh`` that runs at most once however many failed drives ask: the outcome, a failure
    included, is remembered, so a quota or network error costs one fetch attempt per tick."""
    outcome: List[Any] = []

    def run() -> None:
        if not outcome:
            try:
                refresh()
                outcome.append(None)
            except Exception as exc:  # noqa: BLE001 - re-raised to every caller, never swallowed
                outcome.append(exc)
        if outcome[0] is not None:
            raise outcome[0]

    return run


def catalog_refresher(config: Any) -> Callable[[], None]:
    """``refresh()`` that runs ``catalog-fetch --known-ids-from-queue`` (a read-only YouTube pull)
    and so rewrites ``live_catalog.latest.json``; raises when it fails."""

    def refresh() -> None:
        import argparse
        import contextlib
        import sys

        from ..cli.handlers.catalog_fetch import cmd_catalog_fetch

        out = Path(config.seo.storage_path) / "live_catalog.json"
        ns = argparse.Namespace(known_ids=None, known_ids_from_queue=True, out=str(out))
        # the fetch's summary lines go to stderr: stdout is `health backlog --json`'s
        with contextlib.redirect_stdout(sys.stderr):
            failed = cmd_catalog_fetch(ns, config)
        if failed:
            raise RuntimeError("catalog-fetch failed")

    return refresh


def _other_forms(store: Any, project: str, stages_root: Path) -> Dict[str, str]:
    """``{title: "drive, drive"}`` for every title the OTHER drives derive from their checkpoints
    (``expected_lf_titles``), so a shared title can name them."""
    from ..pipeline.longform_metadata import expected_lf_titles

    forms: Dict[str, set] = {}
    for other in store.all_drives():
        recipe = find_recipe(stages_root, other.project) if other.project != project else None
        try:
            saved = json.loads((recipe.parent / "pipeline_state.json").read_text()) if recipe else None  # fmt: skip
        except (OSError, ValueError):
            continue
        if isinstance(saved, dict):
            for ts in expected_lf_titles(saved).values():
                for t in ts:
                    forms.setdefault(t, set()).add(other.project)
    return {t: ", ".join(sorted(names)) for t, names in forms.items()}


def _refuse(res: Dict[str, Any], reason: str) -> Dict[str, Any]:
    return {**res, "action": "refused", "reason": reason}


def _back_to_auditing(store: Any, drive: Any, lf_count: Optional[int] = None) -> None:
    store.register_ready(
        drive.project, drive.region, lf_count or drive.lf_count, status="auditing",
        lf_mode=drive.lf_mode, ingest_mode=drive.ingest_mode,
    )  # fmt: skip


def run_publish(
    *,
    project: str,
    region: str,
    store: Any,
    cadence: Any,
    stages_root: Path,
    code_root: Path,
    jobs_dir: Path,
    now: datetime,
    dry_run: bool,
    drive_running: Callable[[Path], bool],
    launcher: Callable[..., Any] = default_launcher,
    lead_hours: Optional[int] = None,
    expect_sha: Optional[str] = None,
    refresh_catalog: Optional[Callable[[], Any]] = None,
) -> Dict[str, Any]:
    res: Dict[str, Any] = {"project": project, "region": region, "action": "noop", "steps": []}
    d = store.drive(project)
    if d is None:
        return _refuse(res, f"{project} is not in the schedule store")
    if d.region != region.strip().lower():
        return _refuse(res, f"{project} is a {d.region} drive, not {region}")
    reopen = d.status == "failed"
    if d.status != "auditing" and not reopen:
        return _refuse(res, f"{project} is {d.status}; only an auditing drive can be approved")
    if _uploaded(d.output_dir):
        return _refuse(res, f"{project} already has an upload receipt under {d.output_dir}")
    if drive_running(jobs_dir):
        return _refuse(res, "a drive is running; one heavy drive at a time")
    recipe_path = find_recipe(stages_root, project)
    recipe = resume.read_recipe(recipe_path) if recipe_path else None
    if recipe is None:
        return _refuse(res, f"no readable resume.json under {Path(stages_root) / project}")
    bound, why = bind_to_code_root(code_root, recipe["argv"])
    if bound is None:
        return _refuse(res, why)
    stale = stale_phases(recipe_path)
    if stale:
        return _refuse(res, f"{project}'s checkpoint is stale under the current code ({', '.join(stale)} "
                       "re-run); re-audit it (health heal) before publishing")  # fmt: skip
    masks = mask_frame_refusal(recipe_path.parent, drive=project, record=not dry_run)
    if masks:
        return _refuse(res, f"{project} blur mask does not match its source: {masks}")
    from ..pipeline.longform_metadata import orphan_upload_refusal

    first_try, last_failed = publish_job_times(jobs_dir, project, now)
    orphans = orphan_upload_refusal(
        project, recipe_path.parent, receipt_ids=known_video_ids(store.all_drives()),
        record=not dry_run, output_dir=d.output_dir, other_forms=_other_forms(store, project, stages_root),
        first_attempt=first_try,
        # a failed publish may have died after the upload: only a snapshot fetched since sees it
        # a reopened drive with no job record fetches too: nothing says when its upload ran
        fresh_since=last_failed or (now if reopen else None),
        refresh=None if dry_run else refresh_catalog,
    )  # fmt: skip
    if orphans:
        return _refuse(res, orphans)
    lf_count = rendered_lf_count(recipe_path)
    if lf_count is None:
        return _refuse(res, f"no long-form plan in {recipe_path.parent / 'drive_plan.json'}; re-audit it")  # fmt: skip
    plan = read_replace_plan(Path(d.output_dir)) if d.output_dir else {}
    if plan and lf_count != plan_total_lfs(plan):
        # a different split would upload other content under the old titles and slots (M5)
        return _refuse(res, f"{project} rendered {lf_count} long-forms but its replace plan covers "
                       f"{plan_total_lfs(plan)}; re-render it to the same split or supersede it again")  # fmt: skip
    older = [r for r in store.drives(region, "ready") if (r.created_at, r.project) < (d.created_at, d.project)]  # fmt: skip
    published = [r.project for r in older if upload_receipts(r.output_dir)]
    older = [r for r in older if r.project not in published]
    if older:
        return _refuse(res, f"{older[0].project} is ready and older: Waypost claims the oldest "
                       "ready drive first; publish or fail it, then retry")  # fmt: skip
    env = {
        **resume.recipe_env(recipe),
        run_origin.LAUNCHED_BY_ENV: run_origin.PUBLISH_LAUNCHER,
    }
    recount = lf_count != d.lf_count
    steps = [*([f"reopen: {project} failed -> auditing (no upload receipt)"] if reopen else []),
             *([f"count: {project} rendered {lf_count} long-forms (store said {d.lf_count})"] if recount else []),
             *(f"retire: {p} (uploaded outside a claim: Waypost books its receipt, complete)" for p in published),
             f"approve: {project} auditing -> ready",
             f"claim: Waypost {region} slot for {project}",
             f"launch: {JOB_PREFIX}{project} (Phase D, --youtube-upload)",
             "close: health check completes or fails the claim when the job ends"]  # fmt: skip
    res.update(steps=steps, env=env, recipe=str(recipe_path))
    if dry_run:  # Waypost's own dry run: what it would claim now (this drive is not ready yet)
        peek = claim_or_reason(store, cadence, region, now=now, lead_hours=lead_hours,
                               expect_sha=expect_sha, dry_run=True)  # fmt: skip
        res["next_claim"] = (
            {"project": peek.project, "slots": [s.isoformat() for s in peek.slots]}
            if isinstance(peek, Claim) else {"reason": peek.reason}
        )  # fmt: skip
        return {**res, "action": "would_publish", "reason": f"{project} (dry run: nothing changed)"}
    for p in published:
        try:
            store.retire_published(p, upload_receipts(store.drive(p).output_dir), now=now)
        except (SlotTaken, ValueError, KeyError) as exc:
            return _refuse(res, f"{p} is ready, older and uploaded, but Waypost refused to retire it: {exc}")  # fmt: skip
    if reopen or recount:
        _back_to_auditing(store, d, lf_count)
    d = store.drive(project)
    store.set_status(project, "ready")
    try:
        c = claim_or_reason(store, cadence, region, now=now, lead_hours=lead_hours,
                            expect_sha=expect_sha)  # fmt: skip
    except (SlotTaken, StaleDrive, ValueError) as exc:
        store.set_status(project, "auditing")
        return _refuse(res, f"claim refused: {exc}")
    if not isinstance(c, Claim):
        store.set_status(project, "auditing")
        later = f" (next slot {c.next_slot.isoformat()})" if c.next_slot else ""
        return _refuse(res, f"no claim: {c.reason}{later}")
    if c.already_claimed or c.project != project:
        if not c.already_claimed:  # we booked another drive's slot: free it, make it ready again
            other = store.drive(c.project)
            store.finish_claim(c.claim_id, ok=False)
            store.register_ready(other.project, other.region, other.lf_count, status="ready",
                                 lf_mode=other.lf_mode, ingest_mode=other.ingest_mode)  # fmt: skip
        if store.drive(project).status == "ready":
            store.set_status(project, "auditing")
        return _refuse(res, f"Waypost's claim for {region} is {c.project}, not {project}; undone")
    slot = c.slots[0]
    day, date = _DAYS[slot.weekday()], slot.date().isoformat()
    argv = [
        "python3", str(job_runner(code_root)), "--name", f"{JOB_PREFIX}{project}", "--heavy",
        "--detach", "--stall", _STALL_S, "--log-root", str(jobs_dir), "--",
        *phase_d_args(bound, day, date),
    ]  # fmt: skip
    res.update(argv=argv, slot=slot.isoformat(), claim_id=c.claim_id)
    ok, detail = launcher(argv, str(Path(code_root)), env)
    if not ok:
        store.finish_claim(c.claim_id, ok=False)
        _back_to_auditing(store, d)
        return _refuse(res, f"launch failed: {detail}")
    return {**res, "action": "launched", "reason": f"{project} for {slot.isoformat()}"}


def retire_uploaded(store: Any, *, now: Optional[datetime] = None) -> List[Dict[str, str]]:
    """Close each ``ready`` or ``auditing`` drive that was uploaded outside a claim, from its
    receipts: Waypost books each LF at its ``publish_at``, adopts a booking of the same video under
    a stray name (the run dir), and marks the drive ``complete``, so it no longer sits first in the
    claim line. An ``auditing`` drive is the same evidence (ace-basin, 2026-11-20: published, still
    ``auditing``). Never without a receipt: one Waypost cannot book (no ``publish_at``, a clash)
    leaves the drive as it is."""
    done: List[Dict[str, str]] = []
    for d in store.all_drives():
        held = upload_receipts(d.output_dir) if d.status in ("ready", "auditing") else {}
        if not held or replace_unfinished(d.output_dir):  # fmt: skip
            # (a replace still to finish is not published)
            continue
        try:
            done.append({"project": d.project, "status": store.retire_published(d.project, held, now=now)})  # fmt: skip
        except (SlotTaken, ValueError, KeyError):
            continue
    return done


def reconcile_publishes(
    store: Any, jobs_dir: Path, *, uploaded: Callable[[Optional[str]], bool] = _uploaded
) -> List[Dict[str, str]]:
    """Close each claimed drive whose newest ``gms-publish-<project>`` job (started after the
    claim) has ended: ``complete`` when its output holds an upload receipt (a receipt wins over
    the exit code, so a published video never loses its slot), else ``failed``, which frees the
    unconfirmed booking. A running job, or a claim with no publish job, is left alone.
    Then ``retire_uploaded``."""
    records = _job_records(Path(jobs_dir))
    done: List[Dict[str, str]] = []
    for d in store.all_drives():
        if d.status != "claimed" or d.claim_id is None or d.claimed_at is None:
            continue
        runs = [r for r in records if r.get("name") == f"{JOB_PREFIX}{d.project}"
                and float(r.get("created_at") or 0) >= d.claimed_at.timestamp()]  # fmt: skip
        if not runs:
            continue
        state = ((max(runs, key=lambda r: float(r.get("created_at") or 0))).get("result") or {}).get("state")  # fmt: skip
        if state in (None, "running"):
            continue
        left = replace_unfinished(d.output_dir)
        held = uploaded(d.output_dir)
        # a replace with an unfinished LF closes ``failed`` even over kept receipts: ``health
        # backlog`` republishes a failed drive, which resumes the LF; only done/abandoned is clean
        row = {"project": d.project, "status": store.finish_claim(d.claim_id, ok=held and not left)}  # fmt: skip
        if left:
            why = ", ".join(f"LF {lf} {st}" for lf, st in sorted(left.items()))
            row["partial"] = f"replace not finished: {why}"
            # one key (the project): the drive's incident is opened once, not once per alias
            record_incident(d.project, "partial_finish", f"claim closed {row['status']} but {row['partial']}; the publish is retried automatically")  # fmt: skip
        else:
            names = {d.project, Path(d.output_dir).name} if d.output_dir else {d.project}
            for name in names:
                resolve_incidents(name, drive_level=True)
        done.append(row)
    return done + retire_uploaded(store)


def backlog_publisher(
    config: Any,
    *,
    region: str,
    stages_root: Path,
    code_root: Path,
    jobs_dir: Path,
    now: datetime,
    drive_running: Callable[[Path], bool],
) -> Callable[[str, bool], Dict[str, Any]]:
    """``publish(project, dry_run)`` for ``health backlog``: ``run_publish`` with the tick's own
    roots and Waypost's default lead window; the store is opened only when a drive is due. One
    catalog refresher serves the whole tick: a failing fetch is tried once, not per failed drive."""
    refresh = once_per_tick(catalog_refresher(config))

    def publish(project: str, dry_run: bool) -> Dict[str, Any]:
        store, cadence = open_schedule_store(config)
        if store is None:
            return {"project": project, "action": "refused", "steps": [],
                    "reason": "schedule.sqlite3 does not exist; nothing to publish"}  # fmt: skip
        return run_publish(
            project=project, region=region, store=store, cadence=cadence, stages_root=stages_root,
            code_root=code_root, jobs_dir=jobs_dir, now=now, dry_run=dry_run,
            drive_running=drive_running, refresh_catalog=refresh,
        )  # fmt: skip

    return publish


def open_schedule_store(config: Any, *, create: bool = False):
    """``(store, cadence)`` for ``<seo.storage_path>/schedule.sqlite3``, or ``(None, cadence)``
    when it does not exist (a read never creates it)."""
    from seo_manager.scheduling.store import ScheduleStore

    from ..pipeline.seo_slots import STORE_FILENAME, waypost_cadence

    cad = waypost_cadence(config)
    path = Path(config.seo.storage_path) / STORE_FILENAME
    if not path.is_file() and not create:
        return None, cad
    return ScheduleStore(path, cad.tz, cad.regions), cad
