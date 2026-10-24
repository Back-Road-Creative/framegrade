"""Tooling config: the siblings pins, the mypy ratchet and the coverage scope.

These read pyproject.toml and ci/siblings.txt as data; nothing here runs mypy or CI.
"""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
SIBLING_NAMES = {"seo-manager", "gps-journey"}
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
# Modules that are clean under the per-module strict flags. Add a module here only after
# it passes; never remove one to make a change pass.
STRICT_MODULES = ["gomoveshift.paths", "gomoveshift.video_types"]
STRICT_FLAGS = [
    "disallow_untyped_defs",
    "disallow_incomplete_defs",
    "disallow_untyped_decorators",
    "disallow_untyped_calls",
    "disallow_any_generics",
    "disallow_subclassing_any",
    "no_implicit_optional",
    "no_implicit_reexport",
    "warn_return_any",
    "warn_unused_ignores",
    "strict_equality",
]


def _pin_lines(text: str) -> dict[str, str]:
    """Map a requirement name to its full line, skipping blanks and comments."""
    pins = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name = re.split(r"[\s@\[]", line, maxsplit=1)[0]
        pins[name] = line
    return pins


def siblings_drift(siblings_text: str, dependencies: list[str]) -> list[str]:
    """Name every way ci/siblings.txt disagrees with the pins in pyproject.toml."""
    installed = _pin_lines(siblings_text)
    declared = {
        n: ln for n, ln in _pin_lines("\n".join(dependencies)).items() if n in SIBLING_NAMES
    }
    problems = []
    for name in sorted(SIBLING_NAMES):
        if name not in installed:
            problems.append(f"{name}: not listed in ci/siblings.txt")
        elif name not in declared:
            problems.append(f"{name}: not declared in pyproject.toml")
        elif installed[name] != declared[name]:
            problems.append(
                f"{name}: siblings.txt {installed[name]!r} != pyproject {declared[name]!r}"
            )
    for name in sorted(set(installed) - SIBLING_NAMES):
        problems.append(f"{name}: listed in ci/siblings.txt but not a known sibling")
    return problems


def test_siblings_file_matches_pyproject_pins():
    text = (ROOT / "ci" / "siblings.txt").read_text(encoding="utf-8")
    assert siblings_drift(text, PYPROJECT["project"]["dependencies"]) == []


def test_siblings_file_pins_full_commit_shas():
    text = (ROOT / "ci" / "siblings.txt").read_text(encoding="utf-8")
    for name, line in _pin_lines(text).items():
        revision = line.rsplit("@", 1)[1]
        assert FULL_SHA.match(revision), f"{name} must pin a full commit SHA, got {revision!r}"


def test_siblings_drift_names_a_moved_revision_a_missing_pin_and_a_stray_line():
    sha = "a" * 40
    other = "b" * 40
    repo = "git+https://github.com/Back-Road-Creative"
    deps = [
        f"seo-manager @ {repo}/waypost@{sha}",
        f"gps-journey @ {repo}/gps-journey@{sha}",
        "numpy>=1.24.0",
    ]
    clean = f"# note\nseo-manager @ {repo}/waypost@{sha}\ngps-journey @ {repo}/gps-journey@{sha}\n"
    assert siblings_drift(clean, deps) == []

    moved = clean.replace(f"waypost@{sha}", f"waypost@{other}")
    assert [p.split(":")[0] for p in siblings_drift(moved, deps)] == ["seo-manager"]

    missing = f"seo-manager @ {repo}/waypost@{sha}\n"
    assert siblings_drift(missing, deps) == ["gps-journey: not listed in ci/siblings.txt"]

    stray = clean + f"redactcam @ {repo}/redactcam@{sha}\n"
    assert siblings_drift(stray, deps) == [
        "redactcam: listed in ci/siblings.txt but not a known sibling"
    ]


def test_mypy_exclude_hides_the_rest_of_gomoveshift_but_not_the_strict_modules():
    exclude = PYPROJECT["tool"]["mypy"]["exclude"]
    pattern = re.compile("|".join(exclude))
    for module in STRICT_MODULES:
        path = module.replace(".", "/") + ".py"
        assert (ROOT / path).is_file(), f"{path} is listed as strict but does not exist"
        assert not pattern.search(path), f"mypy exclude hides {path}"
    assert not pattern.search("gomoveshift/"), "mypy exclude hides the package directory"
    for path in ("gomoveshift/config.py", "gomoveshift/pipeline/quality.py", "tests/conftest.py"):
        assert pattern.search(path), f"mypy exclude should still skip {path}"


def test_mypy_strict_modules_turn_errors_back_on_with_every_strict_flag():
    overrides = PYPROJECT["tool"]["mypy"]["overrides"]
    blanket = [o for o in overrides if o["module"] == ["gomoveshift.*"]]
    assert blanket and blanket[0]["ignore_errors"] is True
    strict = [o for o in overrides if o["module"] == STRICT_MODULES]
    assert len(strict) == 1, "expected one override naming exactly the strict modules"
    assert strict[0]["ignore_errors"] is False
    for flag in STRICT_FLAGS:
        assert strict[0].get(flag) is True, f"{flag} must be on for the strict modules"


def test_coverage_measures_the_product_package_not_the_test_files():
    # Bare --cov in addopts takes its target from [tool.coverage.run]. Left unset,
    # the 150k statements in tests/ count as covered and pad the total.
    addopts = PYPROJECT["tool"]["pytest"]["ini_options"]["addopts"]
    assert re.search(r"--cov(?=\s)", addopts), "addopts should use a bare --cov"
    assert PYPROJECT["tool"]["coverage"]["run"]["source"] == ["gomoveshift"]
    assert re.search(r"--cov-fail-under=\d+", addopts)
