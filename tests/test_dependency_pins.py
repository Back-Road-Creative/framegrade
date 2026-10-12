"""Guard: what is INSTALLED must match the ``git+https://`` pins in ``pyproject.toml``.

What this prevents. A local environment can carry an older ``redactcam`` build than the commit
pinned in ``pyproject.toml``, so the installed function genuinely lacks a parameter that a test
asserts (for example ``KeyError: 'checkpoint_path'`` in
``tests/test_redactcam_parity.py::TestDetectAndTrackDefaults::test_unpassed_defaults_are_pinned``).
That is environment drift, not a code defect, and it looks like a pre-existing failure unless
something names it. CI installs straight from ``pyproject.toml``, so it never sees the drift.
This module makes the *diagnosis* explicit, so the next drift names itself.

A version string cannot catch this: ``redactcam`` is pinned to a COMMIT, and the stale install
reported the same ``0.1.1`` version as the pinned build. Pip does record exact provenance —
PEP 610 ``direct_url.json`` in the installed ``.dist-info`` — and that is the ground truth
compared here. Nothing below shells out to pip (slow, and pip may be absent) or touches the
network; installed metadata is read via ``importlib.metadata``.

What this can and cannot establish:

* A **40-hex commit pin** (today only ``redactcam``) is verified EXACTLY: ``vcs_info.commit_id``
  must equal the pinned SHA.
* A **tag or branch pin** (today ``sun-track`` / ``pycube-lut`` / ``keycut`` at ``v0.1.0``) is
  verified only as far as ``vcs_info.requested_revision`` — the revision NAME that was asked for.
  Which commit a tag points at cannot be resolved without a network call to the remote, so a
  moved tag is deliberately out of scope.
* An **editable or local-directory install** carries no VCS provenance at all. A
  developer checkout may install the tag-pinned siblings editable from local directories
  (siblings editable before the main project).
  It is reported as an intentional local override, never failed.
* A distribution with **no ``direct_url.json``** is reported as unverifiable, not failed — a
  false failure here would be the exact disease this module treats.

Why a pytest test, and why the live guard can never fail in CI: a test is the only home that runs
both locally and in CI with no extra wiring, and these checks are pure dict/string work, so they
cost an ordinary CI run milliseconds. But CI installs from ``pyproject.toml`` itself, so
``test_installed_vcs_pins_match_pyproject`` below is structurally incapable of failing there —
its entire value is local. The fixture-driven tests around it are what keep the logic honest in
CI, since they feed synthetic metadata rather than reading the live environment.
"""

from __future__ import annotations

import importlib.metadata
import json
import re
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 has no tomllib
    tomllib = None

# ``name[extra,...] @ git+https://host/org/repo@revision`` (PEP 508 direct reference;
# the optional extras are not part of the distribution name).
_VCS_DEP = re.compile(r"^(?P<name>[A-Za-z0-9._-]+)(?:\[[^\]]*\])?\s*@\s*(?P<url>git\+\S+)$")
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")

OK = "ok"
DRIFT = "drift"
LOCAL_OVERRIDE = "local-override"
UNVERIFIABLE = "unverifiable"


def parse_vcs_pins(pyproject_text: str) -> dict[str, tuple[str, str]]:
    """Return ``{normalized name: (git url, pinned revision)}`` for every ``git+`` requirement."""
    pins: dict[str, tuple[str, str]] = {}
    data = tomllib.loads(pyproject_text)
    for dep in data.get("project", {}).get("dependencies", []):
        match = _VCS_DEP.match(str(dep).strip())
        if match is None:
            continue
        url = match.group("url")[len("git+") :].split("#", 1)[0]
        base, separator, revision = url.rpartition("@")
        if not separator:
            base, revision = url, ""
        pins[match.group("name").lower().replace("_", "-")] = (base, revision)
    return pins


def classify_pin(name: str, url: str, revision: str, direct_url: dict | None) -> tuple[str, str]:
    """Compare one pin against its installed provenance; ``direct_url=None`` means not installed."""
    fix = f"Fix: pip install --force-reinstall --no-deps 'git+{url}@{revision}'"
    if direct_url is None:
        return DRIFT, f"{name}: pinned to {revision} ({url}) but NOT installed. {fix}"

    dir_info = direct_url.get("dir_info")
    if dir_info is not None:
        kind = "editable" if dir_info.get("editable") else "local directory"
        return LOCAL_OVERRIDE, (
            f"{name}: intentional local {kind} install from {direct_url.get('url')} — overrides "
            f"the {revision} pin (workspace installs siblings locally first), allowed"
        )

    vcs_info = direct_url.get("vcs_info")
    if not vcs_info:
        return UNVERIFIABLE, (
            f"{name}: installed from {direct_url.get('url')} with no VCS provenance recorded — "
            f"pin {revision} cannot be verified"
        )

    commit = vcs_info.get("commit_id") or "unknown"
    requested = vcs_info.get("requested_revision") or "unknown"
    if _COMMIT_SHA.match(revision):
        if commit == revision:
            return OK, f"{name}: installed commit {commit} matches the pin exactly"
        return DRIFT, f"{name}: pyproject pins commit {revision}, installed is {commit}. {fix}"
    if requested == revision:
        return OK, (
            f"{name}: installed from requested revision {revision} — matches the pin; its "
            f"commit {commit} is not checkable offline, so a moved tag is out of scope"
        )
    return DRIFT, (
        f"{name}: pyproject pins {revision}, installed was requested as {requested} "
        f"(commit {commit}). {fix}"
    )


def read_direct_url(name: str) -> dict | None:
    """Installed PEP 610 provenance: ``None`` if absent from the env, ``{}`` if unrecorded."""
    try:
        dist = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return None
    raw = dist.read_text("direct_url.json")
    return json.loads(raw) if raw else {}


def _pyproject_path() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "pyproject.toml"
        if candidate.is_file():
            return candidate
    return None


# Provenance shapes as pip records them in an installed dist-info directory.
_PINNED_SHA = "0123456789abcdef0123456789abcdef01234567"  # pragma: allowlist secret (synthetic commit id, not a credential)
_STALE_SHA = "1111111111111111111111111111111111111111"
_REDACTCAM_URL = "https://github.com/Back-Road-Creative/redactcam"
_VCS_INSTALL = {
    "url": _REDACTCAM_URL,
    "vcs_info": {
        "commit_id": _PINNED_SHA,
        "requested_revision": _PINNED_SHA,
        "vcs": "git",
    },
}
_EDITABLE_INSTALL = {
    "dir_info": {"editable": True},
    "url": "file:///src/sun-track",
}


class TestCommitPins:
    def test_matching_commit_install_is_ok(self):
        status, _ = classify_pin("redactcam", _REDACTCAM_URL, _PINNED_SHA, _VCS_INSTALL)
        assert status == OK

    def test_stale_commit_install_is_drift_with_an_actionable_message(self):
        stale = {**_VCS_INSTALL, "vcs_info": {**_VCS_INSTALL["vcs_info"], "commit_id": _STALE_SHA}}
        status, message = classify_pin("redactcam", _REDACTCAM_URL, _PINNED_SHA, stale)
        assert status == DRIFT
        assert "redactcam" in message
        assert _PINNED_SHA in message
        assert _STALE_SHA in message
        assert (
            f"pip install --force-reinstall --no-deps 'git+{_REDACTCAM_URL}@{_PINNED_SHA}'"
            in message
        )


class TestTagPins:
    def test_requested_tag_matching_the_pin_is_ok(self):
        install = {
            "url": "https://github.com/Back-Road-Creative/keycut",
            "vcs_info": {"commit_id": _STALE_SHA, "requested_revision": "v0.1.0", "vcs": "git"},
        }
        status, message = classify_pin(
            "keycut", "https://github.com/Back-Road-Creative/keycut", "v0.1.0", install
        )
        assert status == OK
        # A tag's commit is not checkable offline, so the commit id must not be treated as drift.
        assert _STALE_SHA not in message.split("commit ")[0]

    def test_a_different_requested_revision_is_drift(self):
        install = {
            "url": "https://github.com/Back-Road-Creative/keycut",
            "vcs_info": {"commit_id": _STALE_SHA, "requested_revision": "v0.0.9", "vcs": "git"},
        }
        status, message = classify_pin(
            "keycut", "https://github.com/Back-Road-Creative/keycut", "v0.1.0", install
        )
        assert status == DRIFT
        assert "v0.1.0" in message and "v0.0.9" in message


class TestNonVcsInstalls:
    def test_editable_local_install_is_an_allowed_override(self):
        status, message = classify_pin(
            "sun-track",
            "https://github.com/Back-Road-Creative/sun-track",
            "v0.1.0",
            _EDITABLE_INSTALL,
        )
        assert status == LOCAL_OVERRIDE
        assert "src/sun-track" in message

    def test_missing_install_is_drift_and_names_the_install_command(self):
        status, message = classify_pin("keycut", "https://x/keycut", "v0.1.0", None)
        assert status == DRIFT
        assert "pip install --force-reinstall --no-deps 'git+https://x/keycut@v0.1.0'" in message

    def test_absent_provenance_is_unverifiable_not_drift(self):
        status, _ = classify_pin("keycut", "https://x/keycut", "v0.1.0", {})
        assert status == UNVERIFIABLE


class TestPyprojectParsing:
    def test_parses_name_url_and_revision(self):
        text = '[project]\ndependencies = [\n  "Sun_Track @ git+https://h/o/sun-track@v0.1.0",\n  "numpy>=1.24.0",\n]\n'
        assert parse_vcs_pins(text) == {"sun-track": ("https://h/o/sun-track", "v0.1.0")}

    def test_reads_the_live_pyproject_pins(self):
        path = _pyproject_path()
        assert path is not None, "pyproject.toml must be locatable from tests/"
        pins = parse_vcs_pins(path.read_text(encoding="utf-8"))
        assert pins, f"no git+ requirements parsed from {path}"
        for name, (url, revision) in pins.items():
            assert url.startswith("https://"), f"{name} url: {url}"
            assert revision, f"{name} has a git+ url with no pinned revision"

    def test_parses_a_direct_reference_with_extras(self):
        """PEP 508 allows ``name[extra] @ url``; the extra must not leak into the
        distribution name the installed metadata is looked up under."""
        text = (
            '[project]\ndependencies = [\n  "redactcam[gpu] @ git+https://h/o/redactcam@'
            + "a" * 40
            + '",\n]\n'
        )
        assert parse_vcs_pins(text) == {"redactcam": ("https://h/o/redactcam", "a" * 40)}


def _live_project() -> dict:
    path = _pyproject_path()
    assert path is not None
    return tomllib.loads(path.read_text(encoding="utf-8"))["project"]


def _req_name(req: str) -> str:
    return re.split(r"[\s\[><=!~@;]", req.strip(), maxsplit=1)[0].lower()


class TestOnnxruntimeBuild:
    """The environment can carry ``onnxruntime-gpu`` while ``pip install -e .`` pulls the CPU
    ``onnxruntime`` wheel over it, if the core dependencies list ``onnxruntime``. Both wheels own
    ``onnxruntime/`` and the last install wins, so the blur pass silently runs on CPU with no
    mask produced. The dependency spec must make that downgrade impossible: the GPU build is THE
    runtime here, pinned exactly (1.27+ needs CUDA 13, which has no pip runtime wheels yet), and
    the CPU wheel appears nowhere -- not in core, not in any extra -- so no install path can
    request it."""

    def test_cpu_onnxruntime_is_requested_nowhere(self):
        project = _live_project()
        everywhere = list(project["dependencies"])
        for extra in project.get("optional-dependencies", {}).values():
            everywhere.extend(extra)
        offenders = [r for r in everywhere if _req_name(r) == "onnxruntime"]
        assert offenders == [], f"CPU onnxruntime would clobber the GPU build: {offenders}"

    def test_gpu_onnxruntime_is_a_core_dependency_pinned_exactly(self):
        core = {_req_name(r): r for r in _live_project()["dependencies"]}
        assert "onnxruntime-gpu" in core, "onnxruntime-gpu must be a core dependency, not an extra"
        # Extras (``[cuda,cudnn]``) only add the CUDA runtime wheels; the version
        # itself must still be exact.
        assert re.fullmatch(
            r"onnxruntime-gpu(\[[a-z,]+\])?==\d+\.\d+\.\d+", core["onnxruntime-gpu"].strip()
        ), (
            f"onnxruntime-gpu must be pinned exactly (CUDA-major coupling): {core['onnxruntime-gpu']}"
        )

    def test_gpu_onnxruntime_brings_its_own_cuda_runtime(self):
        """A resolver can pick a torch built for CUDA 13 next to a CUDA-12 build of
        onnxruntime-gpu. If nothing in the dependency set asks for the CUDA-12 runtime (an older
        torch used to pull it in as a side effect), ``libcudart.so.12`` is missing, the blur
        session falls back to CPU and a long run dies late. The ``cuda``/``cudnn`` extras make
        onnxruntime-gpu install the nvidia-*-cu12 wheels it links against."""
        core = {_req_name(r): r for r in _live_project()["dependencies"]}
        m = re.match(r"onnxruntime-gpu\[([^\]]*)\]", core["onnxruntime-gpu"].strip())
        extras = {e.strip() for e in m.group(1).split(",")} if m else set()
        assert {"cuda", "cudnn"} <= extras, (
            f"onnxruntime-gpu must request its CUDA runtime extras: {core['onnxruntime-gpu']}"
        )

    def test_redactcam_is_installed_with_its_gpu_extra(self):
        """redactcam ≥0.2 ships onnxruntime only via ``[cpu]``/``[gpu]`` extras;
        asking for ``[gpu]`` keeps the two specs in agreement, and asking for
        neither would leave a fresh install with no runtime at all."""
        core = [r for r in _live_project()["dependencies"] if _req_name(r) == "redactcam"]
        assert len(core) == 1
        assert core[0].startswith("redactcam[gpu] @ git+"), core[0]


def test_installed_vcs_pins_match_pyproject():
    """Live guard. Cannot fail in CI (CI installs from pyproject); its whole value is local."""
    if tomllib is None:
        pytest.skip("tomllib unavailable (Python < 3.11) — cannot parse pyproject.toml")
    path = _pyproject_path()
    if path is None:
        pytest.skip("pyproject.toml not locatable from this file — nothing to compare against")
    pins = parse_vcs_pins(path.read_text(encoding="utf-8"))
    if not pins:
        pytest.skip(f"no git+ requirements declared in {path}")

    drifted = []
    for name, (url, revision) in sorted(pins.items()):
        status, message = classify_pin(name, url, revision, read_direct_url(name))
        print(f"[{status}] {message}")
        if status == DRIFT:
            drifted.append(message)
    assert not drifted, (
        "Installed packages have drifted from the pyproject.toml pins:\n" + "\n".join(drifted)
    )
