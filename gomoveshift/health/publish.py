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
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from seo_manager.scheduling.claim import Claim, claim_or_reason
from seo_manager.scheduling.store import SlotTaken, StaleDrive

from .. import run_origin
from . import resume
from .heal import _STALL_S, _job_records, bind_to_code_root, job_runner, stale_phases
from .launch import _uploaded, default_launcher, upload_receipts

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
    lf_count = rendered_lf_count(recipe_path)
    if lf_count is None:
        return _refuse(res, f"no long-form plan in {recipe_path.parent / 'drive_plan.json'}; re-audit it")  # fmt: skip
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
    """Close each ``ready`` drive that was uploaded outside a claim, from its receipts: Waypost
    books each LF at its ``publish_at``, adopts a booking of the same video under a stray name
    (the run dir), and marks the drive ``complete``, so it no longer sits first in the claim line.
    A receipt Waypost cannot book (no ``publish_at``, a clash) leaves the drive ``ready``."""
    done: List[Dict[str, str]] = []
    for d in store.all_drives():
        held = upload_receipts(d.output_dir) if d.status == "ready" else {}
        if not held:
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
        ok = uploaded(d.output_dir)
        done.append({"project": d.project, "status": store.finish_claim(d.claim_id, ok=ok)})
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
    roots and Waypost's default lead window; the store is opened only when a drive is due."""

    def publish(project: str, dry_run: bool) -> Dict[str, Any]:
        store, cadence = open_schedule_store(config)
        if store is None:
            return {"project": project, "action": "refused", "steps": [],
                    "reason": "schedule.sqlite3 does not exist; nothing to publish"}  # fmt: skip
        return run_publish(
            project=project, region=region, store=store, cadence=cadence, stages_root=stages_root,
            code_root=code_root, jobs_dir=jobs_dir, now=now, dry_run=dry_run,
            drive_running=drive_running,
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
