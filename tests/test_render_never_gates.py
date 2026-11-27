"""A running render holds back only what conflicts with it (note 191).

Renders run back to back for up to 45 hours (2026-11-26), so a path that waits on "some drive is
running" waits for ever. Each test here fails against a global render gate: publishing, a heal
resume, a backlog launch, a stage-dir reclaim, a day reel and a host-worker request all go ahead
while an unrelated drive renders, and only the same drive's dirs (the run locks) or the GPU itself
hold anything back. The last test scans the source so a global gate cannot come back unnoticed."""

from __future__ import annotations

import ast
import inspect
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from gomoveshift.health import heal, launch, publish
from gomoveshift.pipeline import orchestrator_lock as olock
from tests import test_health_backlog_launch as backlog_t
from tests import test_health_heal as heal_t
from tests import test_publish_command as publish_t

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="job identities read /proc")


def _own_identity() -> dict:
    """This test process's job-runner identity: alive by ``heal._proc_alive``'s own rule."""
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    ticks = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]
    return {"pid": os.getpid(), "boot_id": boot, "start_ticks": ticks}


def _rendering(jobs: Path, project: str = "ireland-unrelated", name: str = "gms-backlog-x") -> None:
    """A live render of another drive: a running ``job/1`` record whose worker is alive."""
    d = jobs / f"live-{project}"
    d.mkdir(parents=True)
    argv = ["/py", "-m", "gomoveshift", "drive", f"/src/{project}", "--output-dir", f"/out/{project}"]  # fmt: skip
    (d / "job.json").write_text(json.dumps({
        "schema": "job/1", "run_id": d.name, "name": name, "argv": argv, "created_at": 1.0,
        "result": {"state": "running"}, "worker": _own_identity(), "nanny": _own_identity(),
    }))  # fmt: skip


def _as_wired(fn, jobs: Path) -> dict:
    """The probe ``fn`` gets from ``cli/handlers/health.py`` if a global "is any drive running"
    parameter ever comes back, wired exactly as that handler wired it before note 191."""
    if "drive_running" not in inspect.signature(fn).parameters:
        return {}
    probe = getattr(heal, "default_drive_running")
    return {"drive_running": (lambda: probe(jobs)) if fn is launch.run_backlog else probe}


# ── health publish / backlog / heal never wait on another drive's render ──────────────────


def test_publish_launches_phase_d_while_another_drive_renders(tmp_path):
    rig = publish_t.Rig(tmp_path)
    rig.audit()
    _rendering(rig.jobs)
    res = publish.run_publish(
        project="day-a", region="usa", store=rig.store, cadence=publish_t.CAD,
        stages_root=rig.stages, code_root=rig.tmp, jobs_dir=rig.jobs, now=publish_t.NOW,
        dry_run=False, launcher=rig.launcher, manifest_row_fn=lambda row: None,
        **_as_wired(publish.run_publish, rig.jobs),
    )  # fmt: skip
    assert res["action"] == "launched", res


def test_the_phase_d_job_is_never_marked_heavy(tmp_path):
    rig = publish_t.Rig(tmp_path)
    rig.audit()
    res = publish.run_publish(
        project="day-a", region="usa", store=rig.store, cadence=publish_t.CAD,
        stages_root=rig.stages, code_root=rig.tmp, jobs_dir=rig.jobs, now=publish_t.NOW,
        dry_run=False, launcher=rig.launcher, manifest_row_fn=lambda row: None,
        **({"drive_running": lambda jobs: False} if _as_wired(publish.run_publish, rig.jobs) else {}),
    )  # fmt: skip
    argv = res["argv"]
    assert "--heavy" not in argv[: argv.index("--")], argv


def _backlog_kwargs(rig) -> dict:
    kw = rig.kwargs()
    kw.pop("drive_running", None)  # the pre-191 rig injected a fake; wire the real one instead
    return {**kw, **_as_wired(launch.run_backlog, rig.tmp / "jobs")}


def test_backlog_launches_the_next_row_while_another_drive_renders(tmp_path):
    rig = backlog_t.Rig(tmp_path)
    _rendering(rig.tmp / "jobs")
    res = launch.run_backlog(**_backlog_kwargs(rig))
    assert res["action"] == "launched" and len(rig.launches) == 1, res


def test_backlog_publishes_a_passed_audit_while_another_drive_renders(tmp_path):
    rig = backlog_t.Rig(tmp_path)
    backlog_t._audited(rig, action="launched", reason="day-a for slot")
    _rendering(rig.tmp / "jobs")
    res = launch.run_backlog(**_backlog_kwargs(rig))
    assert rig.publishes == [("day-a", False)] and res["action"] == "launched", res


def test_backlog_republishes_a_failed_publish_while_another_drive_renders(tmp_path):
    rig = backlog_t.Rig(tmp_path)
    backlog_t._failed_publish(rig)
    _rendering(rig.tmp / "jobs")
    res = launch.run_backlog(**_backlog_kwargs(rig))
    assert rig.publishes == [("day-a", False)] and res["action"] == "launched", res


def _heal(rig) -> dict:
    """``run_heal`` with the rig's injected world but heal's own default conflict probe."""
    if not (rig.legacy or rig.filed):
        rig.file_now()
    return heal.run_heal(
        incidents_dir=rig.inc, jobs_dir=rig.jobs, code_root=rig.tmp, mirror_sha=heal_t.SHA,
        now=heal_t.NOW, launcher=lambda argv, cwd: rig.launches.append((argv, cwd)) or (True, ""),
        notify_fn=lambda e, i, s: True, git_changed=lambda a, b, paths: [],
        slot_fn=lambda row, recipe: heal_t.FUTURE, installed_version_fn=lambda p: None,
        pinned_version_fn=lambda c: None, sibling_versions_fn=lambda p: {},
        manifest_sha_fn=lambda row: None, manifest_row_fn=lambda row: None,
        config_sha_fn=lambda config: "cfg",
    )  # fmt: skip


def test_heal_resumes_a_failed_drive_while_another_drive_renders(tmp_path):
    rig = heal_t.Rig(tmp_path)
    _rendering(rig.jobs)
    out = _heal(rig)
    assert out["launched"] == ["drive_failed:demo-drive-a:r1"], rig.row()["resume_attempts"]


def test_heal_still_waits_for_a_live_run_on_the_drives_own_stages_dir(tmp_path):
    rig = heal_t.Rig(tmp_path)
    stages = tmp_path / "stages"  # the recipe's own dir: a live run holds its run lock
    fd, lock = olock._acquire_run_lock(stages, tmp_path / "out")
    try:
        _heal(rig)
    finally:
        olock._release_run_lock(fd, lock)
    att = rig.row()["resume_attempts"][-1]
    assert rig.launches == [] and att["action"] == "skipped", att
    assert "same stages dir" in att["reason"] and str(os.getpid()) in att["reason"]


# ── cleanup --abandoned ───────────────────────────────────────────────────────────────────


def test_an_abandoned_stage_dir_is_reclaimable_while_another_drive_renders(tmp_path):
    from gomoveshift.pipeline.cleanup import abandoned_refusal

    stages = tmp_path / "nctopa-2026-03-05-rec0063" / "_stages"
    stages.mkdir(parents=True)
    other = "4242 /py -m gomoveshift drive /src/ireland-other --output-dir /out/ireland-other"
    with (
        patch("gomoveshift.health.heal._pgrep_drive", return_value=True, create=True),
        patch("gomoveshift.health.heal.live_drive_cmdlines", return_value=[other], create=True),
    ):
        assert abandoned_refusal(stages) is None


# ── the shared GPU lock: renders queue on it, uploads and reels never wait on it ──────────


def _holder(path: Path, owner: str = olock.GPU_LOCK_OWNER) -> int:
    import fcntl

    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.write(fd, json.dumps({"owner": owner, "pid": os.getpid()}).encode())
    return fd


def test_a_render_queues_behind_another_render_past_the_wait(tmp_path):
    path = tmp_path / "gpu.lock"
    fd = _holder(path)
    got: list = []

    def second_render():
        try:
            got.append(olock.acquire_gpu_lock(wait_s=0.2, path=path, queue_behind_renders=True))
        except Exception as exc:  # noqa: BLE001 -- the assertion names it
            got.append(exc)

    t = threading.Thread(target=second_render)
    t.start()
    time.sleep(1.2)  # six times the wait: a timed-out waiter would have failed by now
    assert got == []
    os.close(fd)  # the first render's GPU work ends
    t.join(timeout=10)
    assert len(got) == 1 and isinstance(got[0], tuple), got
    olock.release_gpu_lock(*got[0])


def test_a_foreign_holder_still_times_a_queued_render_out(tmp_path):
    path = tmp_path / "gpu.lock"
    fd = _holder(path, owner="local-inference")
    try:
        with pytest.raises(olock.GpuLockBusyError, match="local-inference"):
            olock.acquire_gpu_lock(wait_s=0.2, path=path, queue_behind_renders=True)
    finally:
        os.close(fd)


def test_a_day_reel_never_waits_for_a_render_holding_the_lock(tmp_path, monkeypatch):
    from gomoveshift.publishing import day_reels

    path = tmp_path / "gpu.lock"
    monkeypatch.setenv("GMS_GPU_LOCK_PATH", str(path))
    fd = _holder(path)
    start = time.monotonic()
    try:
        with pytest.raises(day_reels.GpuLockBusyError):
            with day_reels.shared_gpu_lock():
                pytest.fail("rendered without the lock")
    finally:
        os.close(fd)
    assert time.monotonic() - start < 5, "day-reels waited on the lock and stalled the tick"


class _TakenError(RuntimeError):
    pass


def test_a_phase_d_only_run_never_takes_the_gpu_lock(tmp_path, monkeypatch):
    from gomoveshift.config import Config
    from gomoveshift.pipeline import drive_orchestrator as mod

    def taken(*_a, **_k):
        raise _TakenError("acquire_gpu_lock reached")

    monkeypatch.setattr(mod, "acquire_gpu_lock", taken)
    source = tmp_path / "src"
    source.mkdir()
    orch = mod.DrivePipelineOrchestrator(config=Config())
    try:
        result = orch.run(source_dir=source, output_base=tmp_path / "out", phase="phase_d")
    except _TakenError:
        pytest.fail("a Phase D upload took the shared GPU lock")
    except Exception:  # noqa: BLE001 -- an empty source fails later; the lock is what is tested
        result = None
    assert result is None or not result.success  # nothing to publish here: it refuses


def test_a_full_run_drops_the_gpu_lock_before_phase_d(tmp_path, monkeypatch):
    import fcntl

    from gomoveshift.config import Config
    from gomoveshift.pipeline import drive_orchestrator as mod

    path = tmp_path / "gpu.lock"

    def held() -> bool:
        probe = os.open(path, os.O_RDWR)
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return False
        except BlockingIOError:
            return True
        finally:
            os.close(probe)

    seen: dict = {}
    config = Config()
    config.drive.no_review = True
    orch = mod.DrivePipelineOrchestrator(config=config)
    monkeypatch.setattr(orch, "_save_state", lambda *a, **k: None)
    monkeypatch.setattr(orch, "_phase_produced_required_artifacts", lambda *a: (False, "test"))
    monkeypatch.setattr(mod.ReviewGate, "gate_b", lambda *a, **k: None)
    orch._worktree_lease_handle = None  # set by ``run()``, which this test does not go through
    orch._gpu_lock = olock.acquire_gpu_lock(path=path)
    try:
        phases = [
            ("phase_bc_day_first", lambda s: seen.setdefault("phase_bc_day_first", held())),
            ("phase_d", lambda s: seen.setdefault("phase_d", held())),
        ]
        orch._execute_phases(
            state=type("S", (), {})(), state_file=tmp_path / "pipeline_state.json",
            phase_defs=phases, resumed_phases=[], source_dir=tmp_path, project_dir=tmp_path,
            location="", list_lfs=False, label_dry_run=False, start_time=time.time(),
            stages_completed=[], errors=[], timings={},
        )  # fmt: skip
    finally:
        olock.release_gpu_lock(*getattr(orch, "_gpu_lock", (None, None)))
    assert seen == {"phase_bc_day_first": True, "phase_d": False}, seen


# ── the guard: no health, launch, publish, heal or cleanup path asks "is any drive running" ─

#: Modules where a render must never gate unrelated work.
_GUARDED = (
    *sorted((ROOT / "gomoveshift" / "health").glob("*.py")),
    ROOT / "gomoveshift" / "pipeline" / "cleanup.py",
    ROOT / "gomoveshift" / "cli" / "handlers" / "health.py",
    ROOT / "gomoveshift" / "publishing" / "day_reels.py",
)
#: The names the removed global gates went by.
_GATE_NAMES = {
    "drive_running", "default_drive_running", "drive_running_fn", "_pgrep_drive",
    "_live_drive_probe", "any_drive_running", "drive_is_running",
}  # fmt: skip
#: The refusals they returned.
_GATE_TEXT = re.compile(
    r"one heavy drive at a time|a drive is running|a drive job is running|drive running anywhere",
    re.I,
)
#: A probe asked with no argument but the jobs dir, named like "is something running".
_PROBE_NAME = re.compile(r"running|busy|any_drive|live_drive|drive_live", re.I)


def _called_name(call: ast.Call) -> str:
    f = call.func
    return f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")


def _global_probe(node: ast.AST) -> bool:
    """``if probe():`` or ``if probe(jobs_dir):`` with a running-like name: a question about
    every drive, not about this one's dirs."""
    for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)):
        args = [a for a in call.args if not (isinstance(a, ast.Name) and a.id == "jobs_dir")]
        if not args and not call.keywords and _PROBE_NAME.search(_called_name(call)):
            return True
    return False


@pytest.mark.parametrize("path", _GUARDED, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_global_render_gate_comes_back(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        name = (
            node.id if isinstance(node, ast.Name)
            else node.attr if isinstance(node, ast.Attribute)
            else node.arg if isinstance(node, (ast.arg, ast.keyword))
            else node.name if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            else None
        )  # fmt: skip
        if name in _GATE_NAMES:
            found.append(f"line {node.lineno}: {name}")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _GATE_TEXT.search(node.value):
                found.append(f"line {node.lineno}: refusal text {node.value!r}")
        if isinstance(node, (ast.If, ast.IfExp, ast.While)) and _global_probe(node.test):
            found.append(f"line {node.lineno}: a no-argument 'is something running' probe gates it")
    assert found == [], (
        f"{path.name} gates on some unrelated drive running (note 191): a render may hold back "
        "only a run on its own project or stage dirs (the run locks) and the GPU (its lock). "
        + "; ".join(found)
    )
