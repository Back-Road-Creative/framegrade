"""``health backlog``: launch the next ``drive_backlog.yaml`` row for a region's first uncovered
LF slot, audit only (``--skip-upload --no-youtube-upload``), through the job runner (``heal.job_runner``).

Order, each a stop: the ``gms-backlog-launch`` lease (held = noop); a running drive (noop); a
region row already ``auditing`` (noop: that audit owns the next slot, so a repeat call never
launches a second drive); the region's free slots inside ``COVERAGE_HORIZON_H`` minus its ready
drives (none left = covered); the first manifest row not in the store, not in the content ledger,
not staged by a prior run's ``resume.json`` and not under a ``no-auto-resume`` hold; storage
(``storage_plan``); the subset dir of symlinks; ``run_preflight``; launch; ``register_ready``
as ``auditing``. Publication stays the operator's: nothing here uploads or books."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..pipeline.preflight import PreflightBlockedError, estimate_peak_cache_gb
from .cadence import COVERAGE_HORIZON_H, _aware
from .heal import _STALL_S, SERVICE_DIR, _job_records, job_runner
from .resume import HOLD_NAME

LEASE_NAME = "gms-backlog-launch"
#: Names the lease script (``acquire NAME TTL WHAT`` / ``release NAME``; exit 3 = held).
LEASE_SCRIPT_ENV = "GMS_LEASE_SCRIPT"
DEFAULT_LEASE_SCRIPT = Path("bin") / "lease.sh"
#: ``run_preflight``'s defaults: the staging floor and the output free-space floor.
STAGING_FLOOR_GB = 30.0
OUTPUT_FLOOR_GB = 50.0
STAGING_FACTOR = 1.5
STORAGE_BOUND = (
    f"staging = max(day source x {STAGING_FACTOR}, {STAGING_FLOOR_GB:g} GB) "
    f"(preflight.estimate_peak_cache_gb), output = {OUTPUT_FLOOR_GB:g} GB floor; summed per "
    "device. The x1.5 factor is preflight's existing contract, not an independently validated "
    "bound; retained artifacts are counted through measured free space."
)

Disk = Callable[[Path], Tuple[int, float]]


def exit_code(result: Dict[str, Any]) -> int:
    return 2 if result.get("action") == "refused" else 0


def staging_need_gb(sources: Iterable[str]) -> float:
    return estimate_peak_cache_gb(
        [Path(s) for s in sources], safety_factor=STAGING_FACTOR, floor_gb=STAGING_FLOOR_GB
    )


def default_disk(path: Path) -> Tuple[int, float]:
    """``(st_dev, free GB)`` of ``path``, or of its nearest existing ancestor."""
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return os.stat(probe).st_dev, shutil.disk_usage(probe).free / 1024**3


def storage_plan(
    sources: Sequence[str], staging_dir: Path, output_dir: Path, disk: Disk
) -> List[Dict[str, Any]]:
    """One row per device: what this run needs there against what is free. Staging and output on
    one device are summed, never checked against the same free space twice. Free == need passes
    (``run_preflight`` blocks only on ``free < need``)."""
    by_dev: Dict[int, Dict[str, Any]] = {}
    for role, path, need in (
        ("staging", staging_dir, staging_need_gb(sources)),
        ("output", output_dir, OUTPUT_FLOOR_GB),
    ):
        dev, free = disk(path)
        row = by_dev.setdefault(dev, {"device": dev, "free_gb": free, "need_gb": 0.0, "uses": []})
        row["need_gb"] += need
        row["uses"].append(f"{role} {path} {need:.1f} GB")
    for row in by_dev.values():
        row["need_gb"] = round(row["need_gb"], 3)
        row["ok"] = row["free_gb"] >= row["need_gb"]
    return list(by_dev.values())


def _recipe_sources(recipe: Path) -> Set[str]:
    """Real paths of the video files a prior run's recipe read (its drive source dir)."""
    try:
        argv = json.loads(recipe.read_text(encoding="utf-8"))["argv"]
        src = Path(argv[argv.index("drive") + 1])
        return {os.path.realpath(p) for p in src.iterdir() if p.is_file()}
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return set()


def staged_sources(stages_root: Path) -> List[Tuple[Path, Set[str], bool]]:
    """``(stages dir, source real paths, held)`` per prior run under ``stages_root``: one level
    (``<root>/<project>/_stages``) and two (``<root>/<run>/<project>/_stages``, the layout
    ``GMS_INTERMEDIATE_DIR=<root>/<run>`` gives)."""
    out = []
    for pattern in ("*/_stages", "*/*/_stages"):
        for stages in sorted(Path(stages_root).glob(pattern)):
            recipe = stages / "resume.json"
            out.append((stages, _recipe_sources(recipe), (stages / HOLD_NAME).exists()))
    return out


def _skip_reason(row, store_status, known, staged, stages_root) -> Optional[str]:
    if row.project in store_status:
        return f"in the schedule store ({store_status[row.project]})"
    if row.project in known:
        return "in the content ledger"
    missing = [s for s in row.sources if not Path(s).is_file()]
    if missing:
        return f"missing source {missing[0]}"
    for marker in sorted(Path(stages_root).glob(f"{row.project}/**/_stages/{HOLD_NAME}")):
        return f"operator hold: {marker}"
    mine = {os.path.realpath(s) for s in row.sources}
    hits = [(st, held) for st, srcs, held in staged if mine & srcs]
    for stages, held in hits:
        if held:
            return f"operator hold: {stages / HOLD_NAME}"
    if hits:
        return f"already staged by {hits[0][0] / 'resume.json'}"
    return None


def _subset(subset: Path, sources: Sequence[str], *, create: bool) -> Optional[str]:
    """Refusal text, or ``None`` once ``subset`` holds exactly one symlink per source."""
    want = {Path(s).name: os.path.realpath(s) for s in sources}
    if subset.exists():
        have = {p.name: os.path.realpath(p) for p in subset.iterdir()}
        if have != want:
            extra = sorted(set(have) - set(want)) or sorted(set(want) - set(have))
            return f"subset dir {subset} does not match the manifest row ({extra or 'targets'})"
        return None
    if create:
        subset.mkdir(parents=True)
        for name, target in want.items():
            (subset / name).symlink_to(target)
    return None


def _failed_audit(jobs_dir: Path, project: str) -> Optional[str]:
    """Refusal text when the newest ``gms-backlog-<project>`` job ended in anything but success.

    A failed audit still holds its slot, so without this every later call is a silent noop;
    the operator relaunches (the drive resumes from its stages) or clears the row."""
    name = f"gms-backlog-{project}"
    recs = [r for r in _job_records(Path(jobs_dir)) if r.get("name") == name]
    if not recs:
        return None
    rec = max(recs, key=lambda r: float(r.get("created_at") or 0))
    result = rec.get("result") or {}
    state = result.get("state")
    if state in (None, "running", "succeeded"):
        return None
    return (
        f"audit failed: {project} job {rec.get('run_id')} ended {state} "
        f"(exit {result.get('exit_code')}); relaunch it (stages resume) or clear the row"
    )


def _slot_target(slots, ready: int, now: datetime) -> Optional[Tuple[datetime, str]]:
    horizon = now + timedelta(hours=COVERAGE_HORIZON_H)
    free = sorted(s for s in slots if now <= s[0] <= horizon)
    return free[ready] if len(free) > ready else None


def report_slots(report: Dict[str, Any], region: str) -> List[Tuple[datetime, str]]:
    """``(slot, day)`` free LF slots for ``region`` from the schedule report; the day name
    is Waypost's own, never re-derived here."""
    out = []
    for r in report.get("next_available", []):
        if r.get("region") == region and (s := _aware(r.get("slot"))) and r.get("day"):
            out.append((s, r["day"]))
    return out


def run_backlog(
    *,
    region: str,
    rows: Sequence[Any],
    drives: Sequence[Any],
    free_slots: Sequence[Tuple[datetime, str]],
    known_projects: Set[str],
    stages_root: Path,
    output_root: Path,
    sources_root: Path,
    code_root: Path,
    jobs_dir: Path,
    mirror_sha: Optional[str],
    now: datetime,
    dry_run: bool,
    lease_acquire: Callable[[], bool],
    lease_release: Callable[[], None],
    drive_running: Callable[[], bool],
    preflight: Callable[..., Any],
    launcher: Callable[[List[str], str, Dict[str, str]], Tuple[bool, str]],
    register: Callable[..., None],
    disk: Disk = default_disk,
) -> Dict[str, Any]:
    res: Dict[str, Any] = {"region": region, "action": "noop", "skipped": []}
    if not dry_run and not lease_acquire():
        return {**res, "reason": f"lease {LEASE_NAME} is held by another launch"}
    try:
        return _run(res, region, rows, drives, free_slots, known_projects, stages_root,
                    output_root, sources_root, code_root, jobs_dir, mirror_sha, now, dry_run,
                    drive_running, preflight, launcher, register, disk)  # fmt: skip
    finally:
        if not dry_run:
            lease_release()


def _run(res, region, rows, drives, free_slots, known, stages_root, output_root, sources_root,
         code_root, jobs_dir, mirror_sha, now, dry_run, drive_running, preflight, launcher,
         register, disk):  # fmt: skip
    if drive_running():
        return {**res, "reason": "a drive is running; one heavy drive at a time"}
    mine = [d for d in drives if d.region == region]
    auditing = [d.project for d in mine if d.status == "auditing"]
    if auditing:
        failed = _failed_audit(jobs_dir, auditing[0])
        if failed:
            return {**res, "action": "refused", "project": auditing[0], "reason": failed}
        return {**res, "reason": f"audit in progress: {auditing[0]} holds the next slot"}
    slot = _slot_target(free_slots, sum(d.status == "ready" for d in mine), now)
    if slot is None:
        return {**res, "reason": f"covered: every free slot inside {COVERAGE_HORIZON_H} h has a ready drive"}  # fmt: skip
    res.update(slot=slot[0].isoformat(), lf_day=slot[1])
    status = {d.project: d.status for d in drives}
    staged = staged_sources(stages_root)
    row = None
    for cand in (r for r in rows if r.region == region):
        why = _skip_reason(cand, status, known, staged, stages_root)
        if why is None:
            row = cand
            break
        res["skipped"].append({"project": cand.project, "reason": why})
    if row is None:
        return {**res, "reason": f"no eligible backlog row for {region}"}
    res["project"] = row.project
    root = Path(stages_root) / row.project
    stages_dir, out_dir = root / row.project / "_stages", Path(output_root) / row.project
    subset = Path(sources_root) / row.project
    res["storage"] = storage_plan(row.sources, root, Path(output_root), disk)
    res["storage_bound"] = STORAGE_BOUND
    short = [s for s in res["storage"] if not s["ok"]]
    if short:
        text = "; ".join(f"{s['free_gb']:.1f} GB free < {s['need_gb']:.1f} GB ({', '.join(s['uses'])})" for s in short)  # fmt: skip
        return {**res, "action": "refused", "reason": f"storage: {text}"}
    python = Path(code_root) / SERVICE_DIR / ".venv" / "bin" / "python"
    if not python.is_file():
        return {**res, "action": "refused", "reason": f"no service interpreter at {python}"}
    bad = _subset(subset, row.sources, create=not dry_run)
    if bad:
        return {**res, "action": "refused", "reason": bad}
    day = res["lf_day"]
    drive_argv = [
        str(python), "-m", "gomoveshift", "drive", str(subset), "--region", region,
        "--publish", day, "--lf-day", day, "--ingest-mode", row.ingest_mode,
        "--lf-mode", row.lf_mode, "--output-dir", str(out_dir), "--skip-upload",
        "--no-youtube-upload", "--ost-system-review", "--no-review", "--force", "--no-tmux",
    ]  # fmt: skip
    argv = [
        "python3", str(job_runner(code_root)),
        "--name", f"gms-backlog-{row.project}", "--heavy", "--detach", "--stall", _STALL_S,
        "--expect", str(out_dir / "**" / "shorts" / "**" / "*.mp4"),
        "--log-root", str(jobs_dir), "--", *drive_argv,
    ]  # fmt: skip
    env = {"GMS_INTERMEDIATE_DIR": str(root)}
    res.update(argv=argv, env=env, stages_dir=str(stages_dir))
    if dry_run:
        return {**res, "action": "would_launch", "reason": f"{row.project} for {res['slot']}"}
    try:
        preflight(
            subset, out_dir, staging_dir=stages_dir, ingest_mode=row.ingest_mode, python=python
        )
    except PreflightBlockedError as exc:
        return {**res, "action": "refused", "reason": str(exc)}
    ok, detail = launcher(argv, str(Path(code_root) / SERVICE_DIR), env)
    if not ok:
        return {**res, "action": "refused", "reason": f"launch failed: {detail}"}
    register(
        project=row.project, region=region, lf_count=1, status="auditing",
        output_dir=str(out_dir), mirror_sha=mirror_sha, lf_mode=row.lf_mode,
        ingest_mode=row.ingest_mode,
    )  # fmt: skip
    return {**res, "action": "launched", "reason": f"{row.project} for {res['slot']}"}


def lease_script(code_root: Path) -> Path:
    """The lease script: ``$GMS_LEASE_SCRIPT`` (``~`` expanded), else ``bin/lease.sh`` in
    ``code_root``."""
    raw = os.environ.get(LEASE_SCRIPT_ENV, "").strip()
    return Path(raw).expanduser() if raw else Path(code_root) / DEFAULT_LEASE_SCRIPT


def lease_fns(code_root: Path, ttl_s: int = 600) -> Tuple[Callable[[], bool], Callable[[], None]]:
    """``lease_script`` acquire/release for ``LEASE_NAME`` (exit 0 = acquired, 3 = held)."""
    tool = str(lease_script(code_root))

    def acquire() -> bool:
        cmd = ["bash", tool, "acquire", LEASE_NAME, str(ttl_s), "health backlog"]
        done = subprocess.run(cmd, capture_output=True, timeout=30, check=False)  # noqa: S603,S607
        if done.returncode not in (0, 3):
            raise RuntimeError(f"lease acquire exited {done.returncode}: {done.stderr!r}")
        return done.returncode == 0

    def release() -> None:
        cmd = ["bash", tool, "release", LEASE_NAME]
        subprocess.run(cmd, capture_output=True, timeout=30, check=False)  # noqa: S603,S607

    return acquire, release


def default_launcher(argv: List[str], cwd: str, env: Dict[str, str]) -> Tuple[bool, str]:
    try:
        done = subprocess.run(  # noqa: S603 -- argv built from the manifest row
            argv, cwd=cwd, env={**os.environ, **env}, capture_output=True, text=True,
            timeout=120, check=False,
        )  # fmt: skip
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return done.returncode == 0, (done.stderr or done.stdout).strip()[-200:]


_PREFLIGHT_CHILD = """
import json, sys
from pathlib import Path
from gomoveshift.pipeline.preflight import PreflightBlockedError, run_preflight
src, out, staging, floor, staging_floor, mode = sys.argv[1:7]
try:
    run_preflight(Path(src), Path(out), min_free_gb=int(floor), staging_dir=Path(staging),
                  min_staging_gb=float(staging_floor), ingest_mode=mode)
except PreflightBlockedError as exc:
    print(json.dumps(exc.blockers))
    sys.exit(3)
"""


def default_preflight(source_dir, output_dir, *, staging_dir, ingest_mode, python) -> None:
    """The drive's own storage/tool preflight on this row's per-day inputs, before launch.

    Runs under the drive's interpreter (``python``), not this one: tool checks such as
    gopro-overlay belong to the venv that will render. Exit 3 carries the blockers as JSON;
    any other failure is a crash, never a pass."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    argv = [
        str(python), "-c", _PREFLIGHT_CHILD, str(source_dir), str(output_dir), str(staging_dir),
        str(int(OUTPUT_FLOOR_GB)), str(STAGING_FLOOR_GB), ingest_mode,
    ]  # fmt: skip
    done = subprocess.run(  # noqa: S603 -- fixed child script, paths from the manifest row
        argv, cwd=str(Path(python).parents[2]), capture_output=True, text=True, timeout=600,
        check=False,
    )  # fmt: skip
    if done.returncode == 3:
        raise PreflightBlockedError([tuple(b) for b in json.loads(done.stdout)])
    if done.returncode != 0:
        raise RuntimeError(f"preflight exited {done.returncode}: {done.stderr.strip()[-500:]}")


def load_inputs(config: Any, region: str, ledger_path: Optional[Path], now: datetime):
    """``(region, rows, drives, free_slots, known_projects, register)`` from the live stores:
    the manifest, ``schedule.sqlite3``, ``build_schedule_report`` and the content ledger."""
    from seo_manager.core.profile import canon_region
    from seo_manager.scheduling import backlog
    from seo_manager.scheduling.store import ScheduleStore

    from ..paths import drive_backlog_path
    from ..pipeline.content_ledger import ContentLedger
    from ..pipeline.schedule_report import build_schedule_report
    from ..pipeline.seo_slots import STORE_FILENAME, waypost_cadence

    region = canon_region(region)
    cad = waypost_cadence(config)
    drives, register = open_store(
        Path(config.seo.storage_path) / STORE_FILENAME,
        lambda path: ScheduleStore(path, cad.tz, cad.regions),
    )
    ledger = ContentLedger.load(ledger_path)
    report = build_schedule_report(ledger, config, after=now)
    slots = report_slots(report, region)
    known = {(e.project_name or "").strip().lower() for e in ledger.entries} - {""}
    rows = backlog.load(drive_backlog_path())
    return region, rows, drives, slots, known, register


def open_store(store_path: Path, make: Callable[[Path], Any]):
    """``(drives, register)`` for ``schedule.sqlite3``. An absent store has no drives (as in
    ``health check``) and is not created by reading it; only a launch's register creates it."""
    if not store_path.is_file():
        return [], lambda **kw: make(store_path).register_ready(**kw)
    store = make(store_path)
    return store.all_drives(), store.register_ready
