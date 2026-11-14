"""``health heal``: audit-only auto-resume of a failed drive from its own resume recipe.

Per open ``drive_failed`` incident, at most one resume per new mirror SHA, launched detached
through the job runner (``job_runner``). It never guesses an argv (no readable recipe = ``refused``) and never
publishes: it appends ``--skip-upload --no-youtube-upload``. A ``no-auto-resume`` file beside the
recipe refuses the drive on every SHA (an operator's stop). Guards, each a recorded skip: (a) this
recipe already has a launched attempt on this SHA, (b) a drive job is running, (c) the publish slot has passed
or is unknown, (d) a metadata-gate failure with no generator change since the failure.
A launch that failed in drive pre-flight (before any encode) is outcome ``preflight_failed``: it does
not count as launched on its SHA, so the next tick retries, at most ``HEAL_PREFLIGHT_RETRIES`` times.
Every attempt is appended to the row's ``resume_attempts``."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .. import run_origin
from . import drives, resume
from .store import code_sha, load_all, new_incident, safe_name, utc_iso, write_row

#: Files whose change can alter a metadata-gate verdict (the title/description generators and
#: their validator). Repo-root-relative; guard (d) diffs exactly these.
GENERATOR_PATHS: Tuple[str, ...] = (
    "gomoveshift/pipeline/title_builder.py",
    "gomoveshift/pipeline/title_kit.py",
    "gomoveshift/pipeline/longform_metadata.py",
    "gomoveshift/pipeline/content_validator.py",
)
_METADATA_GATE = re.compile(r"metadata", re.I)
_STALL_S = "86400"
#: Pre-flight failures a recipe may retry on one mirror SHA before it waits for a new SHA.
HEAL_PREFLIGHT_RETRIES = 8
#: What the drive logs when pre-flight blocks the run (``orchestrator_preflight``), and the
#: line that proves an encode phase ran (``orchestrator``): the latter means NOT a pre-flight failure.
_PREFLIGHT_MARK = "Pre-flight check failed:"
_PHASE_DONE_MARK = "PHASE DONE"
_BLOCKER_LINE = re.compile(r"^\s*\d+\.\s+\S")
_DETAIL_MAX = 300
_LOG_MAX_BYTES = 1_000_000

Launcher = Callable[[List[str], str], Tuple[bool, str]]


def stale_phases(recipe_path: Path) -> List[str]:
    """The completed phases of the recipe's checkpoint (``pipeline_state.json`` beside it) that
    the current code re-runs because their stamped code version changed, plus the phases built
    from them. ``[]`` for a missing or unreadable checkpoint (nothing to judge, the failed phase
    decides)."""
    from ..pipeline.orchestrator_fingerprint import (  # noqa: PLC0415 -- heavy, heal-only
        invalidate_phases_if_code_version_stale,
    )

    try:
        saved = json.loads((Path(recipe_path).parent / "pipeline_state.json").read_text())
    except (OSError, ValueError):
        return []
    if not isinstance(saved, dict) or not isinstance(saved.get("phases_completed"), list):
        return []
    done = [p for p in saved["phases_completed"] if isinstance(p, str)]
    kept = invalidate_phases_if_code_version_stale(saved, done)
    return [p for p in done if p not in kept]


def job_name(drive: str, sha: str) -> str:
    return f"gms-heal-{safe_name(drive)}-{sha[:7]}"


#: Overrides that described the writer's host, not the drive: a resume re-earns them.
_HOST_ONLY_FLAGS = ("--allow-cross-fs-source",)
#: The service directory inside a code root: the checkout itself.
SERVICE_DIR = Path(".")
#: Names the job-runner script (the process supervisor that writes the ``job/1`` records).
JOB_RUNNER_ENV = "GMS_JOB_RUNNER"
DEFAULT_JOB_RUNNER = Path("bin") / "job-runner.py"


def job_runner(code_root: Path) -> Path:
    """The job-runner script a detached resume or backlog drive goes through: ``$GMS_JOB_RUNNER``
    (``~`` expanded), else ``bin/job-runner.py`` in ``code_root``."""
    raw = os.environ.get(JOB_RUNNER_ENV, "").strip()
    return Path(raw).expanduser() if raw else Path(code_root) / DEFAULT_JOB_RUNNER


def bind_to_code_root(code_root: Path, recipe_argv: List[str]) -> Tuple[Optional[List[str]], str]:
    """``(argv, cwd)`` running the recipe's drive args on ``code_root``'s own service venv,
    else ``(None, reason)``. The recipe's interpreter and cwd belong to whatever tree wrote it
    (a pinned evidence worktree, say); running them would execute that tree's code while the
    attempt records ``code_sha(code_root)``."""
    if len(recipe_argv) < 3 or recipe_argv[1:3] != ["-m", "gomoveshift"]:
        return None, "recipe argv is not `<python> -m gomoveshift ...`; cannot rebind it"
    svc = Path(code_root) / SERVICE_DIR
    python = svc / ".venv" / "bin" / "python"
    if not python.is_file():
        return None, (
            f"no service interpreter at {python}; build the code root's virtualenv before a resume"
        )
    args = [a for a in recipe_argv[3:] if a not in _HOST_ONLY_FLAGS]
    return [str(python), "-m", "gomoveshift", *args], str(svc)


def register_recipe(incidents_dir: Path, recipe_path: Path, *, now_iso: str) -> Optional[str]:
    """Register one drive's ``resume.json`` as a ``drive_failed`` row so ``run_heal`` can target
    it (``only_key``) -- for a drive no ``health check`` saw fail, e.g. an evidence run launched
    outside the job runner. Re-registering keeps the row and its attempt history, and reopens a
    resolved row (``reopened_at``): an operator's ``--recipe`` is never a silent no-op. ``None``
    when the recipe is unreadable."""
    recipe_path = Path(recipe_path).resolve()
    recipe = resume.read_recipe(recipe_path)
    if recipe is None:
        return None
    drive = recipe_path.parent.parent.name  # <stages>/<drive>/_stages/resume.json
    key = f"drive_failed:{drive}:recipe"
    rows, _bad = load_all(incidents_dir)
    prior = next((r for r in rows if r.get("key") == key), None)
    if prior is not None:
        if prior.get("resolved_at"):
            prior.update(resolved_at=None, reopened_at=now_iso)
            write_row(incidents_dir, prior)
        return key
    write_row(
        incidents_dir,
        new_incident(
            "drive_failed", key, f"{safe_name(drive)}-recipe.json", now_iso,
            drive=drive, phase=recipe.get("phase_reached"), gate=None,
            error="registered by health heal --recipe", log_path=None, job_name=None,
            job_run_id=None, failed_at=None, staging_sha=recipe.get("sha"),
            resume_recipe=str(recipe_path), resume_attempts=[],
        ),
    )  # fmt: skip
    return key


def launch_argv(
    code_root: Path,
    jobs_dir: Path,
    name: str,
    recipe_argv: List[str],
    env: Optional[Dict[str, str]] = None,
):
    """The plain resume: every phase the checkpoint does not hold. Never ``--phase X``, the drive
    CLI's single-phase mode: it runs X alone, so a phase-A failure would pass A and then fail the
    DoD gate because B-D never ran."""
    # ``env K=V`` in the argv itself, so a dry run shows the staging the resume will use.
    prefix = ["env", *(f"{k}={v}" for k, v in sorted(env.items()))] if env else []
    return [
        *prefix,
        "python3", str(job_runner(code_root)),
        "--name", name, "--heavy", "--detach", "--stall", _STALL_S,
        "--log-root", str(jobs_dir),
        "--",
        *recipe_argv,
        # A heal of a heal reads a recipe that already carries them.
        *(f for f in ("--skip-upload", "--no-youtube-upload") if f not in recipe_argv),
    ]  # fmt: skip


def _launch_reason(stale: List[str]) -> str:
    if stale:
        return f"new mirror SHA; audit-only full resume (code version changed: {', '.join(stale)})"
    return "new mirror SHA; audit-only resume"


def _default_launcher(argv: List[str], cwd: str) -> Tuple[bool, str]:
    try:
        done = subprocess.run(  # noqa: S603 -- argv built from the drive's own recipe
            argv, cwd=cwd, capture_output=True, text=True, timeout=120, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return done.returncode == 0, (done.stderr or done.stdout).strip()[-200:]


def _default_git_changed(
    code_root: Path,
) -> Callable[[str, str, Tuple[str, ...]], Optional[List[str]]]:
    def changed(a: str, b: str, paths: Tuple[str, ...]) -> Optional[List[str]]:
        try:
            out = subprocess.run(  # noqa: S603,S607 -- fixed argv
                ["git", "-C", str(code_root), "diff", "--name-only", a, b, "--", *paths],
                capture_output=True, text=True, timeout=30, check=True,
            ).stdout  # fmt: skip
        except (OSError, subprocess.SubprocessError):
            return None  # unknown: do not hold a resume back on a failed lookup
        return [ln for ln in out.splitlines() if ln.strip()]

    return changed


def _job_records(jobs_dir: Path) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    for path in sorted(jobs_dir.glob("*/job.json")) if jobs_dir.is_dir() else []:
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(rec, dict) and rec.get("schema") == "job/1":
            recs.append(rec)
    return recs


def _proc_alive(ident: Any) -> bool:
    """The job runner's identity rule: the boot id and the
    ``/proc/<pid>/stat`` start-time field both match, so a reused pid never counts.
    On Windows psutil's boot time and process create time (as strings) play those roles."""
    if sys.platform == "win32":
        try:
            import psutil

            boot = str(psutil.boot_time())
            ticks = str(psutil.Process(int(ident["pid"])).create_time())
            return ident["boot_id"] == boot and ticks == str(ident["start_ticks"])
        except Exception:  # noqa: BLE001 - gone/denied/malformed identity: not alive
            return False
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        stat = Path(f"/proc/{int(ident['pid'])}/stat").read_text()
        ticks = stat.rsplit(")", 1)[1].split()[19]
        return ident["boot_id"] == boot and ticks == str(ident["start_ticks"])
    except (KeyError, TypeError, ValueError, OSError, IndexError):
        return False


# ``drive`` must be the subcommand (only top-level options may sit before it), so a ``health``
# call whose own arguments mention a drive (``--reason "queue next Ireland drive"``) never
# matches itself.
DRIVE_PROCESS_PATTERN = r"gomoveshift( -[^ ]+)* drive "


def _pgrep_drive() -> bool:
    try:
        done = subprocess.run(  # noqa: S603,S607 -- fixed argv
            ["pgrep", "-f", DRIVE_PROCESS_PATTERN], capture_output=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


def default_drive_running(
    jobs_dir: Path,
    *,
    alive_fn: Callable[[Any], bool] = _proc_alive,
    pgrep_fn: Callable[[], bool] = _pgrep_drive,
) -> bool:
    """An unfinished drive job record whose nanny or worker is still alive, or a live
    ``gomoveshift ... drive`` process. A record a killed nanny left at "running" does not count."""
    for rec in _job_records(jobs_dir):
        state = (rec.get("result") or {}).get("state")
        if state in (None, "running") and drives._is_drive_job(rec):
            if alive_fn(rec.get("worker")) or alive_fn(rec.get("nanny")):
                return True
    return pgrep_fn()


def _parse_when(text: Any) -> Optional[datetime]:
    try:
        moment = datetime.fromisoformat(str(text))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def default_slot(
    config: Any, now: datetime
) -> Callable[[Dict[str, Any], Dict[str, Any]], Optional[datetime]]:
    """The drive's LF publish slot: its ``--lf-publish-date`` in the recipe argv, else the soonest
    free slot of ``gomoveshift schedule`` (``build_schedule_report``) at launch time."""

    def slot(row: Dict[str, Any], recipe: Dict[str, Any]) -> Optional[datetime]:
        argv = recipe["argv"]
        if "--lf-publish-date" in argv and argv.index("--lf-publish-date") + 1 < len(argv):
            return _parse_when(argv[argv.index("--lf-publish-date") + 1])
        try:
            from ..pipeline.content_ledger import ContentLedger
            from ..pipeline.schedule_report import build_schedule_report

            report = build_schedule_report(ContentLedger.load(None), config, after=now)
            slots = [_parse_when(r.get("slot")) for r in report.get("next_available", [])]
            return min((s for s in slots if s), default=None)
        except Exception:  # noqa: BLE001 -- an unknown slot is a recorded skip, never a guess
            return None

    return slot


def preflight_detail(log_path: Path) -> Optional[str]:
    """The first blocker line of a drive log that failed in pre-flight, else ``None`` (no log, an
    unreadable or oversized one, no pre-flight failure, or a run that got as far as a phase)."""
    try:
        if log_path.stat().st_size > _LOG_MAX_BYTES:
            return None
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if _PREFLIGHT_MARK not in text or _PHASE_DONE_MARK in text:
        return None
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if _PREFLIGHT_MARK in ln)
    blocker = next((ln for ln in lines[start:] if _BLOCKER_LINE.match(ln)), lines[start])
    return blocker.strip()[:_DETAIL_MAX]


def _outcomes(rows, jobs_dir, notify_fn, incidents_dir, now_iso) -> List[str]:
    """Push once per launched attempt whose heal job has since succeeded or failed."""
    done: List[str] = []
    records = sorted(_job_records(jobs_dir), key=lambda r: float(r.get("created_at") or 0))
    for row in rows:
        for att in row.get("resume_attempts") or []:
            if att.get("action") != "launched" or att.get("outcome"):
                continue
            since = _parse_when(att.get("at"))
            # A retry reuses its SHA's job name: only a record made at or after this attempt is its run.
            runs = [
                r for r in records
                if r.get("name") == att.get("job_name")
                and (since is None or float(r.get("created_at") or 0) >= since.timestamp() - 60)
            ]  # fmt: skip
            if not runs:
                continue
            result = runs[-1].get("result") or {}
            state, code = result.get("state"), result.get("exit_code")
            if state == "succeeded":
                outcome = "succeeded"
            elif state == "failed" or (state not in (None, "running") and code not in (None, 0)):
                outcome = "failed"
            else:
                continue
            detail = None
            if outcome == "failed":
                detail = preflight_detail(jobs_dir / str(runs[-1].get("run_id")) / "output.log")
                if detail is not None:
                    outcome, att["outcome_detail"] = "preflight_failed", detail
            att["outcome"], att["outcome_at"] = outcome, now_iso
            write_row(incidents_dir, row)
            notify_fn(
                "health_heal_outcome",
                {"incident": row["key"], "job": att["job_name"]},
                f"Auto-resume of {row['drive']} failed pre-flight: {detail}."
                if detail is not None
                else f"Auto-resume of {row['drive']} {outcome}.",
            )
            done.append(f"{row['key']}:{outcome}")
    return done


def run_heal(
    *,
    incidents_dir: Path,
    jobs_dir: Path,
    code_root: Path,
    config: Any = None,
    mirror_sha: Optional[str] = None,
    dry_run: bool = False,
    now: Optional[datetime] = None,
    launcher: Optional[Launcher] = None,
    notify_fn: Optional[Callable[[str, Dict[str, Any], str], bool]] = None,
    git_changed: Optional[Callable[[str, str, Tuple[str, ...]], Optional[List[str]]]] = None,
    drive_running: Optional[Callable[[Path], bool]] = None,
    slot_fn: Optional[Callable[[Dict[str, Any], Dict[str, Any]], Optional[datetime]]] = None,
    only_key: Optional[str] = None,
) -> Dict[str, Any]:
    """``{"launched", "would_launch", "outcomes", "attempts", "notes"}``. Never raises on input."""
    if notify_fn is None:
        from ..console.notify import notify as notify_fn  # noqa: PLC0415
    now = now or datetime.now(timezone.utc)
    now_iso = utc_iso(now.timestamp())
    sha = mirror_sha or code_sha(code_root)
    launcher = launcher or _default_launcher
    git_changed = git_changed or _default_git_changed(code_root)
    drive_running = drive_running or default_drive_running
    slot_fn = slot_fn or default_slot(config, now)
    rows, _bad = load_all(incidents_dir)
    rows = [
        r for r in rows
        if r.get("kind") == "drive_failed" and (only_key is None or r.get("key") == only_key)
    ]  # fmt: skip
    out: Dict[str, Any] = {
        "launched": [], "would_launch": [], "outcomes": [], "attempts": [], "notes": [],
    }  # fmt: skip
    if not dry_run:
        out["outcomes"] = _outcomes(rows, jobs_dir, notify_fn, incidents_dir, now_iso)
    if not sha:
        out["notes"].append("mirror SHA unknown: nothing launched")
        return out

    # Guard (a) is per recipe, not per row: ``health check`` files a failed heal job as a row of
    # its own, and a per-row guard healed that row again on the same SHA, every tick.
    # A launch that died in pre-flight never started the work: it is retried, up to the cap.
    def _launched(a: Dict[str, Any]) -> bool:
        return (
            a.get("mirror_sha") == sha
            and a.get("action") == "launched"
            and a.get("outcome") != "preflight_failed"
        )

    launched_recipes = {
        r.get("resume_recipe") for r in rows for a in r.get("resume_attempts") or [] if _launched(a)
    }
    preflight_fails: Dict[Any, int] = {}
    for r in rows:
        for a in r.get("resume_attempts") or []:
            if a.get("mirror_sha") == sha and a.get("outcome") == "preflight_failed":
                preflight_fails[r.get("resume_recipe")] = (
                    preflight_fails.get(r.get("resume_recipe"), 0) + 1
                )
    for row in rows:
        if row.get("resolved_at"):
            continue
        attempts: List[Dict[str, Any]] = row.setdefault("resume_attempts", [])
        att: Dict[str, Any] = {
            "at": now_iso, "mirror_sha": sha, "slot": None, "action": "skipped",
            "reason": "", "job_name": None,
        }  # fmt: skip
        recipe_path = row.get("resume_recipe")
        recipe = resume.read_recipe(Path(recipe_path)) if recipe_path else None
        stale = stale_phases(Path(recipe_path)) if recipe else []
        if stale:
            att["stale_phases"] = stale
        slot = None
        if recipe is None:
            att.update(action="refused", reason="no readable resume.json for this incident")
        elif resume.is_held(Path(recipe_path)):
            att.update(
                action="refused", reason=f"operator hold: {resume.HOLD_NAME} beside the recipe"
            )
        elif recipe_path in launched_recipes or any(_launched(a) for a in attempts):
            continue  # guard (a): once per recipe per mirror SHA; silent so a pass is not noise
        elif preflight_fails.get(recipe_path, 0) >= HEAL_PREFLIGHT_RETRIES:
            att["reason"] = (
                f"pre-flight failed {HEAL_PREFLIGHT_RETRIES}x on this SHA; "
                "waiting for a new mirror SHA"
            )
        elif drive_running(jobs_dir):
            att["reason"] = "a drive job is running"
        else:
            slot = slot_fn(row, recipe)
            att["slot"] = slot.isoformat() if slot else None
            if slot is None:
                att["reason"] = "publish slot unknown"
            elif now >= slot:
                att["reason"] = f"publish slot {slot.isoformat()} has passed"
            elif (
                _METADATA_GATE.search(row.get("gate") or "")
                and row.get("staging_sha")
                and (git_changed(row["staging_sha"], sha, GENERATOR_PATHS) == [])
            ):
                att["reason"] = "metadata gate failed and no generator file changed"
        bound, cwd = (None, "")
        if att["action"] == "skipped" and not att["reason"]:
            bound, cwd = bind_to_code_root(code_root, recipe["argv"])
            if bound is None:
                att.update(action="refused", reason=cwd)
        if bound is not None:
            name = job_name(row["drive"], sha)
            # Always this launcher's own value: the recipe records only RECIPE_ENV_KEYS, never an
            # origin, and a recipe file must not be able to name its own launcher.
            env = {**resume.recipe_env(recipe), run_origin.LAUNCHED_BY_ENV: "health-heal"}
            argv = launch_argv(code_root, jobs_dir, name, bound, env=env)
            att["interpreter"] = bound[0]
            if dry_run:  # the exact launch, so the evidence is gradeable before it runs
                att.update(argv=argv, cwd=cwd, job_name=name, reason="dry run: would launch")
                out["would_launch"].append(row["key"])
            else:
                ok, detail = launcher(argv, cwd)
                att["job_name"] = name
                if ok:
                    att.update(action="launched", reason=_launch_reason(stale))
                    launched_recipes.add(recipe_path)
                else:
                    att["reason"] = f"launch failed: {detail}"
        if dry_run:
            out["attempts"].append({"key": row["key"], **att})
            continue
        if (
            att["action"] != "launched"
            and attempts
            and all(attempts[-1].get(k) == att[k] for k in ("mirror_sha", "action", "reason"))
        ):
            continue  # same skip as last pass: do not grow the list every cycle
        attempts.append(att)
        write_row(incidents_dir, row)
        out["attempts"].append({"key": row["key"], **att})
        if att["action"] == "launched":
            out["launched"].append(row["key"])
            notify_fn(
                "health_heal_launched",
                {"incident": row["key"], "job": att["job_name"]},
                f"Auto-resuming {row['drive']} ({att['job_name']}), audit only.",
            )
    return out
