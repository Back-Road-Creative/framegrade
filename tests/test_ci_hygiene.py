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
    assert job.get("continue-on-error") is True
    text = "\n".join(s.get("run", "") for s in job["steps"])
    assert ".pip-audit-ignore" in text
    assert "requirements.lock" in text
    assert re.search(r"pip-audit==\d+\.\d+\.\d+", text), "pip-audit must be version-pinned"


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
