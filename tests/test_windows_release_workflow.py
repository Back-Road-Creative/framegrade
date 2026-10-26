"""The Windows release workflow and its gate, read as data; nothing here runs CI."""

import importlib.util
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "windows-release.yml"
WIN = ROOT / "packaging" / "windows"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, WIN / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def wf() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _steps(job: dict) -> list[dict]:
    return job["steps"]


def _runs(job: dict) -> str:
    return "\n".join(s.get("run", "") for s in _steps(job))


def test_triggers_are_release_tags_and_manual_dry_run(wf):
    on = wf[True] if True in wf else wf["on"]  # PyYAML reads a bare `on` as True
    assert set(on) == {"push", "workflow_dispatch"}
    assert set(on["push"]) == {"tags"}, "a branch push must not build a release"
    assert on["push"]["tags"] == ["v[0-9]*.[0-9]*.[0-9]*"]


def test_runners_are_hosted_never_self_hosted(wf):
    for name, job in wf["jobs"].items():
        assert job["runs-on"] in {"windows-2025", "ubuntu-24.04"}, (name, job["runs-on"])
    assert wf["jobs"]["build"]["runs-on"] == "windows-2025"
    assert wf["jobs"]["proof"]["runs-on"] == "windows-2025"
    assert "self-hosted" not in yaml.safe_dump(wf)


def test_every_job_has_a_timeout(wf):
    for name, job in wf["jobs"].items():
        assert isinstance(job.get("timeout-minutes"), int), name


def test_only_the_release_job_can_write(wf):
    assert wf["permissions"] == {"contents": "read"}
    writers = [n for n, j in wf["jobs"].items() if "write" in j.get("permissions", {}).values()]
    assert writers == ["release"]
    assert wf["jobs"]["release"]["permissions"] == {"contents": "write"}


def test_release_is_a_draft_on_tag_runs_only(wf):
    release = wf["jobs"]["release"]
    cond = release["if"]
    assert "github.ref_type == 'tag'" in cond
    assert "workflow_dispatch" not in cond
    text = _runs(release)
    assert "gh release create" in text
    assert "--draft" in text
    assert "--verify-tag" in text
    # nothing but the release job touches releases
    for name, job in wf["jobs"].items():
        if name != "release":
            assert "gh release" not in _runs(job), name


def test_release_waits_for_gate_build_and_proof(wf):
    assert set(wf["jobs"]["release"]["needs"]) == {"gate", "build", "proof"}
    assert wf["jobs"]["build"]["needs"] == "gate"
    assert wf["jobs"]["proof"]["needs"] == "build"


def test_dry_run_keeps_the_installer_as_an_artifact(wf):
    ups = [s for s in _steps(wf["jobs"]["build"]) if s.get("uses", "").startswith("actions/upload")]
    assert len(ups) == 1
    assert "if" not in ups[0], "the tag run needs the artifact too"
    assert ups[0]["with"]["if-no-files-found"] == "error"
    assert "SHA256SUMS" in ups[0]["with"]["path"]


def test_actions_are_pinned_to_a_commit_with_a_version_comment():
    lines = [ln for ln in WORKFLOW.read_text(encoding="utf-8").splitlines() if "uses:" in ln]
    assert lines
    for ln in lines:
        assert re.search(r"uses: [\w.-]+/[\w.-]+@[0-9a-f]{40} # v\d+\.\d+\.\d+$", ln.strip()), ln


def test_checkouts_do_not_persist_credentials(wf):
    for name, job in wf["jobs"].items():
        for s in _steps(job):
            if s.get("uses", "").startswith("actions/checkout@"):
                assert s["with"]["persist-credentials"] is False, name


def test_gate_runs_the_gate_script_for_tags(wf):
    gate = wf["jobs"]["gate"]
    co = next(s for s in _steps(gate) if s.get("uses", "").startswith("actions/checkout@"))
    assert co["with"]["fetch-depth"] == 0, "the gate needs main's history"
    assert "packaging/windows/release_gate.py" in _runs(gate)


def test_build_stages_compiles_and_checksums(wf):
    text = _runs(wf["jobs"]["build"])
    assert "packaging/windows/build_release.py" in text
    assert "--skip-installer" not in text
    assert "packaging/windows/release_gate.py sums" in text
    assert "6.7.1" in text, "Inno Setup is installed pinned when the image lacks it"


def test_proof_runs_the_ps1_against_the_built_installer(wf):
    text = _runs(wf["jobs"]["proof"])
    assert "packaging/windows/windows_proof.ps1" in text
    assert "-Installer" in text


def test_proof_script_proves_the_installer_per_user_and_headless():
    ps1 = (WIN / "windows_proof.ps1").read_text(encoding="utf-8")
    for needle in ("/VERYSILENT", "/CURRENTUSER", "/DIR=", "Start-MpScan", "/healthz", "scan"):
        assert needle in ps1, needle
    assert "pröof" in ps1 or "pröof" in ps1, "install path holds a non-ASCII character"
    assert "unins000" in ps1


def test_no_unannotated_checksum_literal():
    for path in (WORKFLOW, WIN / "release_gate.py", WIN / "windows_proof.ps1"):
        for n, ln in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"\b[0-9a-f]{64}\b", ln):
                assert "pragma: allowlist secret" in ln, f"{path.name}:{n}"


# -- release_gate: pure logic, no network ------------------------------------

SHA = "a" * 40


def _ok_run(sha=SHA):
    return {"head_sha": sha, "status": "completed", "conclusion": "success"}


def test_tag_version():
    gate = _load("release_gate")
    assert gate.tag_version("v0.3.10") == "0.3.10"
    for bad in ("0.3.10", "v0.3", "v0.3.10-rc1", "vX", "", "refs/tags/v0.3.10"):
        assert gate.tag_version(bad) is None, bad


def test_gate_passes_a_good_tag():
    gate = _load("release_gate")
    assert gate.gate_problems(SHA, "v0.3.10", "0.3.10", on_main=True, runs=[_ok_run()]) == []


def test_gate_refuses_a_commit_off_main():
    gate = _load("release_gate")
    out = gate.gate_problems(SHA, "v0.3.10", "0.3.10", on_main=False, runs=[_ok_run()])
    assert len(out) == 1 and "not on main" in out[0]


def test_gate_refuses_a_tag_that_disagrees_with_pyproject():
    gate = _load("release_gate")
    out = gate.gate_problems(SHA, "v0.3.11", "0.3.10", on_main=True, runs=[_ok_run()])
    assert len(out) == 1 and "0.3.10" in out[0] and "v0.3.11" in out[0]


def test_gate_refuses_a_malformed_tag():
    gate = _load("release_gate")
    out = gate.gate_problems(SHA, "v0.3", "0.3.0", on_main=True, runs=[_ok_run()])
    assert any("vX.Y.Z" in p for p in out)


@pytest.mark.parametrize(
    "run",
    [
        {"head_sha": "b" * 40, "status": "completed", "conclusion": "success"},
        {"head_sha": SHA, "status": "completed", "conclusion": "failure"},
        {"head_sha": SHA, "status": "in_progress", "conclusion": None},
        {"head_sha": SHA, "status": "completed", "conclusion": "cancelled"},
    ],
)
def test_gate_needs_a_successful_ci_run_for_this_exact_commit(run):
    gate = _load("release_gate")
    out = gate.gate_problems(SHA, "v0.3.10", "0.3.10", on_main=True, runs=[run])
    assert len(out) == 1 and "ci.yml" in out[0]


def test_gate_reports_every_problem_at_once():
    gate = _load("release_gate")
    assert len(gate.gate_problems(SHA, "v0.3.11", "0.3.10", on_main=False, runs=[])) == 3


def test_sums_format_and_order(tmp_path):
    gate = _load("release_gate")
    a, b = tmp_path / "b.exe", tmp_path / "a.txt"
    a.write_bytes(b"abc")
    b.write_bytes(b"")
    text = gate.sums_text([a, b])
    assert (
        text
        == (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  a.txt\n"  # pragma: allowlist secret (public sha256 of empty input, not a credential)
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad  b.exe\n"  # pragma: allowlist secret (public sha256 of "abc", not a credential)
        )
    )


def test_assets_are_size_gated(tmp_path):
    gate = _load("release_gate")
    f = tmp_path / "x.bin"
    f.write_bytes(b"12345")
    gate.check_sizes([f], limit=5)
    with pytest.raises(SystemExit, match="x.bin"):
        gate.check_sizes([f], limit=4)
