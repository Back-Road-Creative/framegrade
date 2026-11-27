"""The health loop launches and version-checks through the published venv.

The staging sync publishes a new venv by pointing ``.venvs/current`` at it and leaves the
``.venv`` link as it is while any process runs through it. On 2026-11-27 ``.venv`` stayed on
0.3.174 while ``.venvs/current`` held 0.3.177, and every heal, publish and backlog launch was
refused as "install pending: checkout pins 0.3.177, venv has 0.3.174".
"""

from __future__ import annotations

import re
from pathlib import Path

from gomoveshift.health import check, heal, retry_policy

HEALTH = Path(heal.__file__).resolve().parent


def _venv(root: Path, name: str) -> Path:
    py = root / ".venvs" / name / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("")
    return py


def _layout(tmp_path: Path) -> tuple[Path, Path]:
    """``.venv -> .venvs/old`` (held) and ``.venvs/current -> new`` (published)."""
    old, new = _venv(tmp_path, "old"), _venv(tmp_path, "new")
    (tmp_path / ".venv").symlink_to(Path(".venvs") / "old")
    (tmp_path / ".venvs" / "current").symlink_to("new")
    return old, new


def test_service_python_prefers_the_published_venv(tmp_path):
    _, new = _layout(tmp_path)
    assert retry_policy.service_python(tmp_path) == str(new)


def test_service_python_falls_back_to_dot_venv(tmp_path):
    old = _venv(tmp_path, "old")
    (tmp_path / ".venv").symlink_to(Path(".venvs") / "old")
    assert retry_policy.service_python(tmp_path) == str(old)


def test_service_python_ignores_a_current_link_with_no_interpreter(tmp_path):
    old = _venv(tmp_path, "old")
    (tmp_path / ".venv").symlink_to(Path(".venvs") / "old")
    (tmp_path / ".venvs" / "current").symlink_to("missing")
    assert retry_policy.service_python(tmp_path) == str(old)


def test_resume_binds_to_the_published_venv(tmp_path):
    _, new = _layout(tmp_path)
    argv, cwd = heal.bind_to_code_root(
        tmp_path, ["/elsewhere/python", "-m", "gomoveshift", "drive"]
    )
    assert argv == [str(new), "-m", "gomoveshift", "drive"]


def test_install_check_reads_the_published_venv(tmp_path, monkeypatch):
    _, new = _layout(tmp_path)
    seen = []
    monkeypatch.setattr(
        retry_policy, "default_installed_version", lambda py: seen.append(py) or "0.3.177"
    )
    installed, _ = check.default_versions(tmp_path)
    assert installed == "0.3.177" and seen == [str(new)]


def test_no_health_module_joins_dot_venv_itself():
    """One helper picks the interpreter; a second ``.venv`` join would bring the lag back."""
    joins = re.compile(r"""["']\.venv["']""")
    hits = [
        f"{p.name}:{n}"
        for p in sorted(HEALTH.glob("*.py"))
        if p.name != "retry_policy.py"
        for n, line in enumerate(p.read_text("utf-8").splitlines(), 1)
        if joins.search(line)
    ]
    assert hits == []
