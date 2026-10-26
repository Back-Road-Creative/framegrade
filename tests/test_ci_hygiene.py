"""CI hygiene: dependency updates, the advisory audit job, an honest type-check name.

These read the committed workflow files as data; nothing here runs CI.
"""

import re
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CI = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())
SHA_PIN = re.compile(r"^[^@\s]+@[0-9a-f]{40}$")


def _steps():
    for job_name, job in CI["jobs"].items():
        for step in job["steps"]:
            yield job_name, step


def test_ci_tests_use_every_logical_cpu():
    # pytest-xdist's "-n auto" counts PHYSICAL cores when psutil is installed
    # (it is, for preflight), so the 4-vCPU hosted runner got 2 workers and
    # the suite took ~20 min. CI passes "-n logical"; pyproject keeps "auto"
    # so a local run on this shared box does not take every thread.
    runs = [s["run"] for _, s in _steps() if s.get("name") == "Tests"]
    assert len(runs) == 1 and runs[0].startswith(".venv/bin/pytest -n logical"), runs
    addopts = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["pytest"]["ini_options"][
        "addopts"
    ]
    assert "-n auto" in addopts, addopts


def _addopts():
    return tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["pytest"]["ini_options"][
        "addopts"
    ]


def test_every_shard_runs_and_the_required_check_waits_for_all():
    # ruleset 24929613 requires a check named "test"; the shards are
    # "test-shard (n)". A matrix smaller than N in i/N silently drops tests.
    shard_job = CI["jobs"]["test-shard"]
    shards = shard_job["strategy"]["matrix"]["shard"]
    (tests,) = [s for s in shard_job["steps"] if s.get("name") == "Tests"]
    spec = tests["env"]["FRAMEGRADE_TEST_SHARD"]
    assert spec == "${{ matrix.shard }}/" + str(len(shards)), spec
    assert shards == list(range(1, len(shards) + 1)), shards
    assert shard_job["strategy"]["fail-fast"] is False
    gate = CI["jobs"]["test"]
    assert gate["needs"] == "test-shard"
    assert gate["if"] == "always()"
    assert "needs.test-shard.result" in str(gate["steps"][0]["env"])


def test_combined_coverage_floor_matches_pyproject():
    # Each shard turns the floor off (--cov-fail-under=0); the "test" job
    # combines the shards and must hold the same floor a local run does.
    (floor,) = re.findall(r"--cov-fail-under=(\d+)", _addopts())
    runs = "\n".join(s.get("run", "") for s in CI["jobs"]["test"]["steps"])
    assert f"coverage report --show-missing --fail-under={floor}" in runs, runs
    assert "coverage combine" in runs
    shard_steps = CI["jobs"]["test-shard"]["steps"]
    (upload,) = [s for s in shard_steps if s.get("uses", "").startswith("actions/upload-artifact@")]
    assert upload["with"]["path"] == ".coverage"
    assert upload["with"]["include-hidden-files"] is True


def test_dependabot_covers_pip_and_actions():
    path = ROOT / ".github" / "dependabot.yml"
    assert path.is_file(), ".github/dependabot.yml is missing"
    config = yaml.safe_load(path.read_text())
    assert config["version"] == 2
    by_eco = {u["package-ecosystem"]: u for u in config["updates"]}
    assert {"pip", "github-actions"} <= set(by_eco)
    for eco in ("pip", "github-actions"):
        assert by_eco[eco]["directory"] == "/"
        assert by_eco[eco]["schedule"]["interval"] in {"daily", "weekly", "monthly"}


def test_mypy_step_name_states_what_it_checks():
    # pyproject excludes gomoveshift/, tests/ and scripts/ from mypy, so the
    # step must not be called a plain "Type check" of the project.
    mypy = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["mypy"]
    steps = [s for _, s in _steps() if "mypy" in s.get("run", "")]
    runs = [s for s in steps if "-m mypy" in s["run"]]
    assert len(runs) == 1
    name = runs[0]["name"]
    assert name != "Type check"
    if "gomoveshift/" in mypy["exclude"]:
        assert "only" in name, name
        assert "tools" in name and "ci" in name and "bin" in name, name


def test_audit_job_is_wired_to_the_ignore_file_and_the_lock():
    jobs = [
        job
        for job in CI["jobs"].values()
        if any("pip-audit" in s.get("run", "") for s in job["steps"])
    ]
    assert len(jobs) == 1, "expected exactly one job that runs pip-audit"
    job = jobs[0]
    # Advisory until the locked versions are clean; see the comment in ci.yml.
    # The advisory lives on the audit step, so the job itself ends green: a
    # job-level continue-on-error still leaves a red check on every PR.
    assert not job.get("continue-on-error"), "job-level continue-on-error leaves a red check"
    audit_steps = [s for s in job["steps"] if "pip-audit" in s.get("run", "") and "-r " in s["run"]]
    assert len(audit_steps) == 1
    assert audit_steps[0].get("continue-on-error") is True
    text = "\n".join(s.get("run", "") for s in job["steps"])
    assert ".pip-audit-ignore" in text
    assert "requirements.lock" in text
    assert "ci/audit-requirements.txt" in text, (
        "the scanner and its dependencies come from a pinned file"
    )


def test_audit_tool_requirements_are_fully_pinned():
    pins = [
        line.strip()
        for line in (ROOT / "ci" / "audit-requirements.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert any(p.lower().startswith("pip-audit==") for p in pins)
    for p in pins:
        assert re.fullmatch(r"[A-Za-z0-9_.\-]+==[0-9][0-9A-Za-z.\-]*", p), p


def test_ignore_file_entries_are_parseable_and_justified():
    lines = (ROOT / ".pip-audit-ignore").read_text().splitlines()
    for i, line in enumerate(lines):
        if not line.strip() or line.startswith("#"):
            continue
        assert re.fullmatch(r"(PYSEC|GHSA|CVE)-[0-9A-Za-z-]+", line), line
        assert lines[i - 1].startswith("# reason:") or any(
            ln.startswith("# reason:") for ln in lines[max(0, i - 4) : i]
        ), f"{line} has no reason comment above it"


def test_every_action_is_pinned_by_full_sha():
    for job_name, step in _steps():
        uses = step.get("uses")
        if uses:
            assert SHA_PIN.match(uses), f"{job_name}: {uses} is not pinned by commit sha"


def test_ci_runs_once_per_commit():
    # A pull request run covers the PR head; the push run covers the merge
    # result on main. A push trigger on every branch would run each PR commit
    # twice.
    assert CI[True]["push"]["branches"] == ["main"]
    assert "pull_request" in CI[True]
