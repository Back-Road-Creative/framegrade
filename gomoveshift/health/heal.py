"""``health heal``: audit-only auto-resume of a failed drive from its own resume recipe.

An open ``drive_failed`` incident always gets another resume: a failed render is never abandoned.
Each resume is launched detached through the job runner (``job_runner``). Heal never guesses an
argv (no readable recipe = ``refused``) and never publishes: it appends
``--skip-upload --no-youtube-upload``. A ``no-auto-resume`` file beside the recipe refuses the drive
on every SHA (an operator's stop); that and an unreadable recipe are the only stops.
Retry is keyed on the failure, never the mirror SHA (``retry_policy``): each attempt records the
failure's fingerprint (installed ``gomoveshift-video`` / ``seo_manager`` / ``redactcam`` versions,
the drive's manifest row, the drive config, the failed phase) and class (``deterministic``,
``transient``, ``publish_only``, ``unknown``). Deterministic: no relaunch until the fingerprint
changes; transient: up to 3 attempts per fingerprint; unknown: 1; publish_only: refused (heal never
publishes). A refusal is a recorded ``refused`` attempt naming class, fingerprint and reason.
Guards, each a recorded skip that ends by itself: (a) retry backoff. With n launched attempts of
this recipe (counted across every row sharing it) under the current fingerprint, the next waits
``min(HEAL_RETRY_BASE_S * 2**(n-1), HEAL_RETRY_MAX_S)`` (1 h, 2 h, 4 h ... 24 h) after the newest.
A changed fingerprint retries at once. (b) a live drive on the same stages or project dir
(``drive_conflict``); a drive elsewhere is no reason, since a render queues on the GPU lock. (c) none: the publish slot is
recorded, never gating, because an audit-only drive keeps only the weekday check
(``_launch_anchor_refusal``); ``health backlog`` and ``health publish`` claim a fresh slot after
a passed audit. (d) a metadata-gate failure with no generator change since the failure retries once
per ``HEAL_RETRY_MAX_S``. (e) install pending: the checkout pins a ``gomoveshift-video`` version
the venv does not hold yet (the staging sync defers the install while a drive runs), so a launch
would run the old code.
A launch that failed in drive pre-flight (before any encode) is outcome ``preflight_failed``: it does
not count toward the backoff, so the next tick retries, ``HEAL_PREFLIGHT_RETRIES`` times on the same
code; then the retry waits ``HEAL_RETRY_MAX_S`` from the newest pre-flight failure, and so on.
A run killed by a host restart (its job record stays "running" but its nanny is gone) is outcome
``interrupted``: it does not count toward the budget or the backoff either, up to
``retry_policy.HEAL_INTERRUPTED_RETRIES`` per fingerprint; past that it counts like a failure.
A same-code retry reuses ``job_name(drive, sha)``; ``_outcomes`` matches a job record to an attempt by
creation time, so each attempt gets its own run's outcome.
Every attempt is appended to the row's ``resume_attempts``."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import tomllib

from .. import run_origin
from . import drives, resume, retry_policy
from .retry_policy import (  # noqa: F401 -- the probes live there; re-exported for callers
    default_config_sha,
    default_installed_version,
    default_manifest_sha,
    default_sibling_versions,
)
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
#: Pre-flight failures a recipe may retry on one code identity before it backs off to the cap.
HEAL_PREFLIGHT_RETRIES = 8
#: Retry backoff after a launched attempt on the same code: ``BASE * 2**(n-1)`` seconds after the
#: newest of n such attempts, capped at ``MAX``. A failed drive is never abandoned.
HEAL_RETRY_BASE_S = 3600
HEAL_RETRY_MAX_S = 86400
#: What the drive logs when pre-flight blocks the run (``orchestrator_preflight``), and the
#: line that proves an encode phase ran (``orchestrator``): the latter means NOT a pre-flight failure.
_PREFLIGHT_MARK = "Pre-flight check failed:"
_PHASE_DONE_MARK = "PHASE DONE"
_BLOCKER_LINE = re.compile(r"^\s*\d+\.\s+\S")
_DETAIL_MAX = 300
_LOG_MAX_BYTES = 1_000_000

Launcher = Callable[[List[str], str], Tuple[bool, str]]
logger = logging.getLogger(__name__)


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


def base_drive(drive: str) -> str:
    """The drive a heal job name stands for: ``gms-heal-<x>-<sha7>`` unwraps to ``<x>``, however
    deeply nested, so a heal of a heal reuses the base drive name instead of growing it."""
    return min(drives.heal_origins(str(drive)) or {str(drive)}, key=len)


def wrap_name(name: str, sha: str) -> str:
    """``gms-heal-<name>-<sha7>`` exactly as given: how a job record written before names stopped
    nesting still parses."""
    return f"gms-heal-{safe_name(name)}-{sha[:7]}"


def job_name(drive: str, sha: str) -> str:
    return wrap_name(base_drive(drive), sha)


#: Job-name prefixes whose incidents never have a resume recipe: a publish job (Phase D) and a
#: heal resume are not audits ``resume.json`` can restart.
_RECIPELESS_PREFIXES = ("gms-publish-", "gms-heal-")


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


#: Recipe flags whose values are files or directories the drive reads, with the values each takes.
_INPUT_FLAGS = {"--day-gpx": 1, "--bracket-gpx": 2, "--map-inset-bundle": 1, "--trail-profile": 1}
#: Of those, the ones a ``drive_backlog.yaml`` row owns (flag -> row attribute): only these have a
#: current replacement.
_ROW_INPUTS = {"--day-gpx": "day_gpx", "--bracket-gpx": "bracket_gpx"}


def _row_value(manifest_row: Any, flag: str) -> List[str]:
    raw = (getattr(manifest_row, _ROW_INPUTS[flag], None) or ()) if flag in _ROW_INPUTS else ()
    return [str(raw)] if isinstance(raw, (str, Path)) else [str(p) for p in raw]


def refresh_inputs(
    argv: List[str], manifest_row: Any, cwd: Optional[Path] = None
) -> Tuple[Optional[List[str]], Dict[str, List[str]], str]:
    """``(argv, refreshed, "")``, or ``(None, {}, reason)`` when a file the saved argv names is
    gone and nothing current replaces it: a replay that would only die in the drive's pre-flight
    is refused instead, saying which file.

    Inputs are the drive's source directory and each ``_INPUT_FLAGS`` value, judged from ``cwd``
    (where the drive will run). A gone ``--day-gpx`` / ``--bracket-gpx`` takes the drive's backlog
    row's value when every file the row names exists (a GPX rename, gomoveshiftsite#334, left
    Galley Head's recipe naming a gone file). An input that still exists is kept even when the
    row names another: the checkpoint was built from it. ``refreshed`` maps each replaced flag
    to its new value(s)."""
    base = Path(cwd) if cwd else Path.cwd()
    out, refreshed = list(argv), {}
    if "drive" in out[3:]:
        i = out.index("drive", 3) + 1
        if i < len(out) and not out[i].startswith("-") and not (base / out[i]).is_dir():
            return None, {}, f"the drive's source directory {out[i]} no longer exists"
    for k, tok in enumerate(list(out)):
        flag, eq, val = tok.partition("=")
        n = _INPUT_FLAGS.get(flag)
        if n is None or (eq and n != 1):
            continue
        old = [val] if eq else out[k + 1 : k + 1 + n]
        if all((base / p).exists() for p in old if p):
            continue
        new = _row_value(manifest_row, flag)
        if len(new) != n or not all((base / p).is_file() for p in new):
            where = "and its drive_backlog.yaml row names no existing replacement"
            return None, {}, f"recipe {flag} {' '.join(old)} no longer exists {where}"
        out[k : k + 1 + (0 if eq else n)] = [f"{flag}={new[0]}"] if eq else [flag, *new]
        refreshed[flag] = new
    return out, refreshed, ""


def refresh_recipe(
    recipe_path: Path, recipe: Dict[str, Any], manifest_row: Any, *, cwd: Path, dry_run: bool
) -> Tuple[Optional[List[str]], Dict[str, List[str]], str]:
    """``refresh_inputs`` on a recipe's own argv: the one step every replay of a ``resume.json``
    takes (heal, a confirmed backlog resume, ``health publish``, ``drive --from-recipe``). A
    replaced value is also written back into the recipe (``resume.rewrite_argv``; never in a dry
    run), so the next replay names the current file without depending on the backlog row again.
    That changes no checkpoint: only a file that no longer exists is replaced, which no resume
    could have read, and the drive itself rewrites the recipe from its own argv on every full
    start."""
    argv, refreshed, why = refresh_inputs(recipe["argv"], manifest_row, cwd)
    if argv is None:
        logger.warning("replay of %s refused: %s", recipe_path, why)
    elif refreshed and not dry_run:
        resume.rewrite_argv(Path(recipe_path), argv)
    return argv, refreshed, why


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


def _retry_wait_s(n: int) -> int:
    """Seconds to wait after the newest of ``n`` launched attempts on the same code."""
    return min(HEAL_RETRY_BASE_S * 2 ** (max(1, min(n, 32)) - 1), HEAL_RETRY_MAX_S)


def _launch_reason(stale: List[str], retry: int = 0) -> str:
    why = f"retry {retry + 1} on this code" if retry else "new mirror SHA"
    if stale:
        return f"{why}; audit-only full resume (code version changed: {', '.join(stale)})"
    return f"{why}; audit-only resume"


def _default_launcher(argv: List[str], cwd: str) -> Tuple[bool, str]:
    try:
        done = subprocess.run(  # noqa: S603 -- argv built from the drive's own recipe
            argv, cwd=cwd, capture_output=True, text=True, timeout=120, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return done.returncode == 0, (done.stderr or done.stdout).strip()[-200:]


def default_pinned_version(code_root: Path) -> Optional[str]:
    """The ``[project] version`` the code root's ``pyproject.toml`` pins, else ``None``."""
    try:
        data = tomllib.loads((Path(code_root) / SERVICE_DIR / "pyproject.toml").read_text("utf-8"))
    except (OSError, ValueError):
        return None
    version = (data.get("project") or {}).get("version")
    return version if isinstance(version, str) and version else None


def install_pending(installed: Optional[str], pinned: Optional[str]) -> Optional[str]:
    """Why no render may start now, else None: the code root pins one ``gomoveshift-video``
    version and the venv that would render holds another. The staging sync defers the install
    while any run of the service is live, so a launch here would run the old code and hold the
    install back again; refusing leaves the gap the deferred install lands in. Every launch path
    (heal's resume, ``health backlog``'s new drive and confirmed resume, ``health publish``) asks
    this. Both versions must be known: an unknown one is not a reason to hold a render."""
    if installed and pinned and installed != pinned:
        return f"install pending: checkout pins {pinned}, venv has {installed}"
    return None


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


def live_drive_cmdlines() -> List[str]:
    """The command line of each live ``gomoveshift ... drive`` process (``pgrep -af``); empty
    when there is none or ``pgrep`` cannot run."""
    try:
        done = subprocess.run(  # noqa: S603,S607 -- fixed argv
            ["pgrep", "-af", DRIVE_PROCESS_PATTERN],
            capture_output=True, text=True, timeout=10, check=False,
        )  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return []
    return [ln for ln in done.stdout.splitlines() if ln.strip()] if done.returncode == 0 else []


def _opt(argv: List[str], flag: str) -> Optional[str]:
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith(flag + "="):
            return tok.split("=", 1)[1]
    return None


def _run_dirs(recipe_path: Path, recipe: Dict[str, Any]) -> Tuple[List[Path], Optional[str]]:
    """``(run-lock dirs, --output-dir)`` of a run of this recipe: its stages dir (the recipe's own)
    and its project dir, derived as ``DriveOrchestrator._resolve_paths`` does. A dir the argv does
    not name is left out."""
    argv, base = [str(a) for a in recipe.get("argv") or []], Path(str(recipe.get("cwd") or "."))
    dirs, out = [Path(recipe_path).parent], _opt(argv, "--output-dir")
    if out and "drive" in argv[3:]:
        i = argv.index("drive", 3) + 1
        if i < len(argv) and not argv[i].startswith("-"):
            name = Path(argv[i]).stem.replace(" ", "_").lower()
            out_base = base / out
            dirs.append(out_base if out_base.name.lower() == name else out_base / name)
    return dirs, (os.path.normpath(str(base / out)) if out else None)


def drive_conflict(
    jobs_dir: Path,
    recipe_path: Path,
    recipe: Dict[str, Any],
    *,
    alive_fn: Callable[[Any], bool] = _proc_alive,
    holder_fn: Optional[Callable[[Path], Optional[Dict[str, Any]]]] = None,
) -> Optional[str]:
    """Why a run of this recipe would collide with a live drive, else ``None``: a live run holds
    the run lock of its stages dir or project dir (the locks ``ConcurrentRunError`` enforces), or
    a live drive job (a render or a Phase D publish) has the same ``--output-dir``, which covers
    a job still starting up. Any other live drive is never a reason: a render waits on the shared
    GPU lock, and an upload needs no GPU."""
    from ..pipeline.orchestrator_lock import (  # noqa: PLC0415 -- heal imports no pipeline at load
        PROJECT_RUN_LOCK_NAME,
        live_run_lock_holder,
    )

    holder_fn = holder_fn or live_run_lock_holder
    dirs, out = _run_dirs(recipe_path, recipe)
    for i, d in enumerate(dirs):
        held = holder_fn(d / ("pipeline_state.lock" if i == 0 else PROJECT_RUN_LOCK_NAME))
        if held:
            return f"a drive on the same {'stages' if i == 0 else 'project'} dir is running (pid {held.get('pid')})"
    for rec in _job_records(jobs_dir) if out else []:
        state = (rec.get("result") or {}).get("state")
        if state not in (None, "running") or not drives._is_drive_job(rec):
            continue
        argv = [str(a) for a in rec.get("argv") or []]
        theirs = _opt(argv, "--output-dir")
        cwd = str(rec.get("cwd") or recipe.get("cwd") or ".")
        same = theirs and os.path.normpath(os.path.join(cwd, theirs)) == out
        if same and (alive_fn(rec.get("worker")) or alive_fn(rec.get("nanny"))):
            return f"drive job {rec.get('name')} on the same output dir is running"
    return None


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


def _outcomes(
    rows,
    jobs_dir,
    notify_fn,
    incidents_dir,
    now_iso,
    proc_alive: Callable[[Any], bool] = _proc_alive,
) -> List[str]:
    """Push once per launched attempt whose heal job has since succeeded, failed, or was
    interrupted (left at "running" by a nanny and worker that no longer exist)."""
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
            elif (
                state in (None, "running")
                and isinstance(runs[-1].get("nanny"), dict)
                and runs[-1]["nanny"]
                and not proc_alive(runs[-1]["nanny"])
                and not (runs[-1].get("worker") and proc_alive(runs[-1]["worker"]))
            ):
                outcome = "interrupted"
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
                else f"Auto-resume of {row['drive']} was interrupted (host restart or killed "
                "runner); it will be retried."
                if outcome == "interrupted"
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
    conflict_fn: Optional[Callable[[Path, Path, Dict[str, Any]], Optional[str]]] = None,
    slot_fn: Optional[Callable[[Dict[str, Any], Dict[str, Any]], Optional[datetime]]] = None,
    only_key: Optional[str] = None,
    installed_version_fn: Optional[Callable[[str], Optional[str]]] = None,
    pinned_version_fn: Optional[Callable[[Path], Optional[str]]] = None,
    sibling_versions_fn: Optional[Callable[[str], Dict[str, Optional[str]]]] = None,
    manifest_sha_fn: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None,
    manifest_row_fn: Optional[Callable[[Dict[str, Any]], Optional[Any]]] = None,
    config_sha_fn: Optional[Callable[[Any], Optional[str]]] = None,
    proc_alive: Callable[[Any], bool] = _proc_alive,
) -> Dict[str, Any]:
    """``{"launched", "would_launch", "outcomes", "attempts", "notes"}``. Never raises on input."""
    if notify_fn is None:
        from ..console.notify import notify as notify_fn  # noqa: PLC0415
    now = now or datetime.now(timezone.utc)
    now_iso = utc_iso(now.timestamp())
    sha = mirror_sha or code_sha(code_root)
    launcher = launcher or _default_launcher
    git_changed = git_changed or _default_git_changed(code_root)
    conflict_fn = conflict_fn or drive_conflict
    slot_fn = slot_fn or default_slot(config, now)
    installed_version_fn = installed_version_fn or default_installed_version
    pinned_version_fn = pinned_version_fn or default_pinned_version
    sibling_versions_fn = sibling_versions_fn or default_sibling_versions
    manifest_sha_fn = manifest_sha_fn or default_manifest_sha
    manifest_row_fn = manifest_row_fn or retry_policy.default_manifest_row
    config_sha_fn = config_sha_fn or default_config_sha
    rows, _bad = load_all(incidents_dir)
    rows = [
        r for r in rows
        if r.get("kind") == "drive_failed" and (only_key is None or r.get("key") == only_key)
    ]  # fmt: skip
    out: Dict[str, Any] = {
        "launched": [], "would_launch": [], "outcomes": [], "attempts": [], "notes": [],
    }  # fmt: skip
    if not dry_run:
        out["outcomes"] = _outcomes(rows, jobs_dir, notify_fn, incidents_dir, now_iso, proc_alive)
    if not sha:
        out["notes"].append("mirror SHA unknown: nothing launched")
        return out

    # The code identity a resume runs: the SHA plus the version installed in the interpreter heal
    # launches. A deferred install leaves the SHA ahead of the venv, so the SHA alone is not it.
    python = Path(code_root) / SERVICE_DIR / ".venv" / "bin" / "python"
    installed = installed_version_fn(str(python))
    pinned = pinned_version_fn(Path(code_root))
    versions = {"gomoveshift": installed, **sibling_versions_fn(str(python))}
    config_sha = config_sha_fn(config)

    def _same_code(a: Dict[str, Any]) -> bool:
        # A pre-fingerprint attempt (older row): the SHA plus the installed version. An unknown
        # version matches: only two known, different versions are different code.
        had = a.get("installed_version")
        return a.get("mirror_sha") == sha and (had is None or installed is None or had == installed)

    def _same(a: Dict[str, Any], fph: str) -> bool:
        # The retry key is the failure's fingerprint, never the mirror SHA: an unrelated commit
        # to the mirror is not a reason to relaunch.
        return a.get("fp_hash") == fph if a.get("fp_hash") else _same_code(a)

    # The backoff is per recipe, not per row: ``health check`` files a failed heal job as a row of
    # its own, and a per-row count healed that row again every tick. A launch that died in
    # pre-flight never started the work: it is retried, up to the cap, then on the max backoff.
    launches: Dict[Any, List[Tuple[datetime, Dict[str, Any]]]] = {}
    seen_fp: Dict[Any, set] = {}  # recipe -> fingerprints a failure was recorded under
    last_launch_any: Dict[Any, datetime] = {}  # any code: guard (d)'s "same generators" clock
    for r in rows:
        rp = r.get("resume_recipe")
        if (r.get("failure_fingerprint") or {}).get("hash"):
            seen_fp.setdefault(rp, set()).add(r["failure_fingerprint"]["hash"])
        for a in r.get("resume_attempts") or []:
            when = _parse_when(a.get("at"))
            if when is None or a.get("action") != "launched":
                continue
            last_launch_any[rp] = max(when, last_launch_any.get(rp, when))
            launches.setdefault(rp, []).append((when, a))
    for row in rows:
        if row.get("resolved_at"):
            continue
        attempts: List[Dict[str, Any]] = row.setdefault("resume_attempts", [])
        att: Dict[str, Any] = {
            "at": now_iso, "mirror_sha": sha, "installed_version": installed, "slot": None, "action": "skipped",
            "reason": "", "job_name": None,
        }  # fmt: skip
        recipe_path = row.get("resume_recipe")
        klass = retry_policy.classify_failure(row.get("error"), row.get("phase"), row.get("gate"))
        fp = retry_policy.fingerprint(
            versions=versions, manifest_sha=manifest_sha_fn(row), config_sha=config_sha,
            phase=row.get("phase"),
        )  # fmt: skip
        fph = retry_policy.fingerprint_hash(fp)
        att.update(failure_class=klass, fp_hash=fph, fingerprint=fp)
        # A row with no ``failure_fingerprint`` is legacy (filed before fingerprints): heal never
        # stamps it from the current code, which would count the very release meant to recover it
        # as the one the failure happened under. ``health check`` stamps new rows when it files them.
        group = [a for _w, a in launches.get(recipe_path, ())]
        launched_at = []
        interrupted_n = 0
        for w, a in launches.get(recipe_path, ()):
            if not _same(a, fph) or a.get("outcome") == "preflight_failed":
                continue
            if a.get("outcome") == "interrupted":
                interrupted_n += 1
                if interrupted_n <= retry_policy.HEAL_INTERRUPTED_RETRIES:
                    continue  # a host restart is not a failed attempt: no backoff either
            launched_at.append(w)
        preflight_at = [
            w for w, a in launches.get(recipe_path, ())
            if _same(a, fph) and a.get("outcome") == "preflight_failed"
        ]  # fmt: skip
        refused = retry_policy.refusal(
            klass, fph, launched=retry_policy.launched_count(group, fph),
            failed_under=fph in seen_fp.get(recipe_path, ()),
        )  # fmt: skip
        recipe = resume.read_recipe(Path(recipe_path)) if recipe_path else None
        stale = stale_phases(Path(recipe_path)) if recipe else []
        if stale:
            att["stale_phases"] = stale
        slot = None
        if recipe is None and str(row.get("job_name") or "").startswith(_RECIPELESS_PREFIXES):
            # such a job never has a recipe: re-refusing it every tick can never end, so it closes
            why = (
                f"no resume recipe: {row.get('job_name')} is a publish or heal job, never resumable"
            )
            att.update(action="closed", reason=why)
            if not dry_run:
                row["resolved_at"], row["resolution_reason"] = now_iso, why
        elif recipe is None:
            att.update(action="refused", reason="no readable resume.json for this incident")
        elif resume.is_held(Path(recipe_path)):
            att.update(
                action="refused", reason=f"operator hold: {resume.HOLD_NAME} beside the recipe"
            )
        elif refused:
            att.update(action="refused", reason=refused)
        elif launched_at and now < (
            retry_at := max(launched_at) + timedelta(seconds=_retry_wait_s(len(launched_at)))
        ):
            att["reason"] = f"retry backoff: next attempt at {utc_iso(retry_at.timestamp())}"  # (a)
        elif len(preflight_at) >= HEAL_PREFLIGHT_RETRIES and now < (
            retry_at := max(preflight_at) + timedelta(seconds=HEAL_RETRY_MAX_S)
        ):
            att["reason"] = (
                f"pre-flight failed {HEAL_PREFLIGHT_RETRIES}x on this code; "
                f"retry at {utc_iso(retry_at.timestamp())}"
            )
        elif pending := install_pending(installed, pinned):
            att["reason"] = pending
        elif clash := conflict_fn(jobs_dir, Path(recipe_path), recipe):
            att["reason"] = clash  # (b) only the same drive's dirs; any other drive never waits it
        else:
            # Guard (c) is gone: an audit-only resume keeps only the drive CLI's weekday check
            # (``_launch_anchor_refusal``), so a passed or unknown slot refuses nothing. The slot
            # is still recorded; ``health backlog`` / ``health publish`` claim a fresh one.
            slot = slot_fn(row, recipe)
            att["slot"] = slot.isoformat() if slot else None
            if (
                _METADATA_GATE.search(row.get("gate") or "")
                and row.get("staging_sha")
                and (git_changed(row["staging_sha"], sha, GENERATOR_PATHS) == [])
            ):
                # Guard (d): the same failure on unchanged generators retries once per max wait,
                # counted from the newest launch (any code), else from the first such skip.
                first_skip = next(
                    (
                        _parse_when(a.get("at")) for a in attempts
                        if str(a.get("reason", "")).startswith("metadata gate failed")
                    ),
                    None,
                )  # fmt: skip
                since = last_launch_any.get(recipe_path) or first_skip or now
                retry_at = since + timedelta(seconds=HEAL_RETRY_MAX_S)
                if now < retry_at:
                    att["reason"] = (
                        "metadata gate failed and no generator file changed; "
                        f"retry at {utc_iso(retry_at.timestamp())}"
                    )
        bound, cwd = (None, "")
        if att["action"] == "skipped" and not att["reason"]:
            fresh, refreshed, why = refresh_recipe(
                Path(recipe_path), recipe, manifest_row_fn(row),
                cwd=Path(code_root) / SERVICE_DIR, dry_run=dry_run,
            )  # fmt: skip
            if refreshed:
                att["gpx_refreshed"] = refreshed
            bound, cwd = bind_to_code_root(code_root, fresh) if fresh else (None, why)
            if bound is None:
                att.update(action="refused", reason=cwd)
        if bound is not None:
            name = job_name(row["drive"], sha)
            # Always this launcher's own value: the recipe records only RECIPE_ENV_KEYS, never an
            # origin, and a recipe file must not be able to name its own launcher.
            env = {**resume.recipe_env(recipe), run_origin.LAUNCHED_BY_ENV: "health-heal"}
            argv = launch_argv(code_root, jobs_dir, name, bound, env=env)
            att["interpreter"] = bound[0]
            prior = len(launched_at)
            if dry_run:  # the exact launch, so the evidence is gradeable before it runs
                att.update(argv=argv, cwd=cwd, job_name=name, reason="dry run: would launch")
                out["would_launch"].append(row["key"])
            else:
                ok, detail = launcher(argv, cwd)
                att["job_name"] = name
                if ok:
                    att.update(action="launched", reason=_launch_reason(stale, prior))
                    launches.setdefault(recipe_path, []).append((now, att))
                    last_launch_any[recipe_path] = now
                else:
                    att["reason"] = f"launch failed: {detail}"
        if dry_run:
            out["attempts"].append({"key": row["key"], **att})
            continue
        if (
            att["action"] != "launched"
            and attempts
            and all(attempts[-1].get(k) == att[k] for k in ("fp_hash", "action", "reason"))
        ):
            continue  # same skip as last pass: do not grow the list every cycle
        attempts.append(att)
        if att["action"] == "refused" and refused:
            logger.info("heal %s: refused: %s", row["key"], att["reason"])
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
