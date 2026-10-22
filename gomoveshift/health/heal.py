"""``health heal``: audit-only auto-resume of a failed drive from its own resume recipe.

Per open ``drive_failed`` incident, at most one resume per new mirror SHA, launched detached
through the job runner (``job_runner``). It never guesses an argv (no readable recipe = ``refused``) and never
publishes: it appends ``--skip-upload --no-youtube-upload``. A ``no-auto-resume`` file beside the
recipe refuses the drive on every SHA (an operator's stop). Guards, each a recorded skip: (a) this
SHA already has a launched attempt, (b) a drive job is running, (c) the publish slot has passed
or is unknown, (d) a metadata-gate failure with no generator change since the failure.
Every attempt is appended to the row's ``resume_attempts``."""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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
_LETTER = {"phase_a": "A", "phase_d": "D"}

Launcher = Callable[[List[str], str], Tuple[bool, str]]


def phase_letter(phase: Optional[str]) -> Optional[str]:
    """Map a logged phase to a ``--phase`` choice (A, BC, D); None when it can't be."""
    text = (phase or "").strip().lower()
    if text in _LETTER:
        return _LETTER[text]
    if text.startswith(("phase_b", "phase_c")):
        return "BC"
    m = re.fullmatch(r"phase ([a-f])", text)
    if m:
        return {"a": "A", "b": "BC", "c": "BC", "d": "D"}.get(m.group(1))
    return None


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
    outside the job runner. Re-registering keeps the row and its attempt history. ``None`` when the
    recipe is unreadable."""
    recipe_path = Path(recipe_path).resolve()
    recipe = resume.read_recipe(recipe_path)
    if recipe is None:
        return None
    drive = recipe_path.parent.parent.name  # <stages>/<drive>/_stages/resume.json
    key = f"drive_failed:{drive}:recipe"
    rows, _bad = load_all(incidents_dir)
    if any(r.get("key") == key for r in rows):
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
    letter: str,
    env: Optional[Dict[str, str]] = None,
):
    # ``env K=V`` in the argv itself, so a dry run shows the staging the resume will use.
    prefix = ["env", *(f"{k}={v}" for k, v in sorted(env.items()))] if env else []
    return [
        *prefix,
        "python3", str(job_runner(code_root)),
        "--name", name, "--heavy", "--detach", "--stall", _STALL_S,
        "--log-root", str(jobs_dir),
        "--",
        *recipe_argv, "--phase", letter, "--skip-upload", "--no-youtube-upload",
    ]  # fmt: skip


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
    ``/proc/<pid>/stat`` start-time field both match, so a reused pid never counts."""
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        stat = Path(f"/proc/{int(ident['pid'])}/stat").read_text()
        ticks = stat.rsplit(")", 1)[1].split()[19]
        return ident["boot_id"] == boot and ticks == str(ident["start_ticks"])
    except (KeyError, TypeError, ValueError, OSError, IndexError):
        return False


def _pgrep_drive() -> bool:
    try:
        done = subprocess.run(  # noqa: S603,S607 -- fixed argv
            ["pgrep", "-f", r"gomoveshift.* drive "], capture_output=True, timeout=10, check=False
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


def _outcomes(rows, jobs_dir, notify_fn, incidents_dir, now_iso) -> List[str]:
    """Push once per launched attempt whose heal job has since succeeded or failed."""
    done: List[str] = []
    records = sorted(_job_records(jobs_dir), key=lambda r: float(r.get("created_at") or 0))
    for row in rows:
        for att in row.get("resume_attempts") or []:
            if att.get("action") != "launched" or att.get("outcome"):
                continue
            runs = [r for r in records if r.get("name") == att.get("job_name")]
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
            att["outcome"], att["outcome_at"] = outcome, now_iso
            write_row(incidents_dir, row)
            notify_fn(
                "health_heal_outcome",
                {"incident": row["key"], "job": att["job_name"]},
                f"Auto-resume of {row['drive']} {outcome}.",
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
        letter = phase_letter(row.get("phase"))
        slot = None
        if recipe is None:
            att.update(action="refused", reason="no readable resume.json for this incident")
        elif resume.is_held(Path(recipe_path)):
            att.update(
                action="refused", reason=f"operator hold: {resume.HOLD_NAME} beside the recipe"
            )
        elif letter is None:
            att.update(action="refused", reason=f"cannot map phase {row.get('phase')!r} to A/BC/D")
        elif any(a.get("mirror_sha") == sha and a.get("action") == "launched" for a in attempts):
            continue  # guard (a): once per mirror SHA; silent so a pass is not noise
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
            argv = launch_argv(
                code_root, jobs_dir, name, bound, letter, env=resume.recipe_env(recipe)
            )
            att["interpreter"] = bound[0]
            if dry_run:  # the exact launch, so the evidence is gradeable before it runs
                att.update(argv=argv, cwd=cwd, job_name=name, reason="dry run: would launch")
                out["would_launch"].append(row["key"])
            else:
                ok, detail = launcher(argv, cwd)
                att["job_name"] = name
                if ok:
                    att.update(action="launched", reason="new mirror SHA; audit-only resume")
                else:
                    att["reason"] = f"launch failed: {detail}"
        if dry_run:
            out["attempts"].append({"key": row["key"], **att})
            continue
        if attempts and all(
            attempts[-1].get(k) == att[k] for k in ("mirror_sha", "action", "reason")
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
