"""``product_sha``: the commit of THIS product's code, never an enclosing repo's HEAD."""

from __future__ import annotations

import json
import subprocess
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from gomoveshift.health import store

_GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid"]


def _repo(root: Path) -> str:
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(root)], check=True)  # noqa: S603,S607
    (root / "f.txt").write_text("x")
    subprocess.run([*_GIT, "-C", str(root), "add", "."], check=True)  # noqa: S603
    subprocess.run([*_GIT, "-C", str(root), "commit", "-qm", "c"], check=True)  # noqa: S603
    return subprocess.run(  # noqa: S603
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def _fake_pkg(parent: Path) -> Path:
    pkg = parent / "gomoveshift"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("")
    return pkg


class _Dist:
    def __init__(self, text):
        self._text = text

    def read_text(self, name):
        return self._text if name == "direct_url.json" else None


def _direct_url(commit):
    return json.dumps(
        {"url": "https://example.invalid/x.git", "vcs_info": {"vcs": "git", "commit_id": commit}}
    )


def test_source_checkout_returns_its_head(tmp_path):
    head = _repo(tmp_path / "co")
    pkg = _fake_pkg(tmp_path / "co")
    assert store.product_sha(package_dir=pkg) == head


def test_package_nested_in_unrelated_repo_is_not_that_repos_head(tmp_path):
    _repo(tmp_path / "other")
    pkg = _fake_pkg(tmp_path / "other" / "venv" / "lib" / "site-packages")
    with mock.patch.object(store.importlib.metadata, "distribution", return_value=_Dist(None)):
        assert store.product_sha(package_dir=pkg) is None


_COMMIT = "1111111111111111111111111111111111111111"  # synthetic commit id


def test_installed_dist_commit_id_is_used(tmp_path):
    _repo(tmp_path / "other")
    pkg = _fake_pkg(tmp_path / "other" / "venv" / "site-packages")
    with mock.patch.object(
        store.importlib.metadata, "distribution", return_value=_Dist(_direct_url(_COMMIT))
    ):
        assert store.product_sha(package_dir=pkg) == _COMMIT


def test_dist_name_matches_pyproject():
    import tomllib

    root = Path(store.__file__).resolve().parents[2]
    meta = tomllib.loads((root / "pyproject.toml").read_text())
    assert store.DIST_NAME == meta["project"]["name"]


@pytest.mark.parametrize("text", [None, "{not json", json.dumps({"url": "u"}), "[]"])
def test_no_git_and_no_usable_direct_url_is_none(tmp_path, text):
    pkg = _fake_pkg(tmp_path / "plain")
    with mock.patch.object(store.importlib.metadata, "distribution", return_value=_Dist(text)):
        assert store.product_sha(package_dir=pkg) is None


def test_missing_distribution_is_none(tmp_path):
    pkg = _fake_pkg(tmp_path / "plain")
    err = store.importlib.metadata.PackageNotFoundError
    with mock.patch.object(store.importlib.metadata, "distribution", side_effect=err("x")):
        assert store.product_sha(package_dir=pkg) is None


def test_drive_recipe_records_product_sha(tmp_path):
    from gomoveshift.cli.handlers import drive

    args = Namespace(output_dir=tmp_path / "out", directory="/in/demo", phase=None)
    cfg = SimpleNamespace(paths=SimpleNamespace(intermediate_folder="_stages"))
    with (
        mock.patch.object(drive.sys, "argv", ["gomoveshift", "drive", "/in/x"]),
        mock.patch.object(store, "product_sha", return_value="feedface"),
    ):
        path = drive._write_resume_recipe(args, cfg)
    assert json.loads(path.read_text())["sha"] == "feedface"
