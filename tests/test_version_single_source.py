"""The package, the web app and pyproject.toml name one version."""

import tomllib
from pathlib import Path

import gomoveshift

ROOT = Path(__file__).resolve().parents[1]


def _pyproject_version() -> str:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]


def test_package_version_matches_pyproject():
    assert gomoveshift.__version__ == _pyproject_version()


def test_web_app_reports_the_package_version():
    source = (ROOT / "gomoveshift" / "web" / "app.py").read_text()
    assert "version=__version__," in source
