"""Both provenance sidecars name the code revision that wrote them.

An inspector checking a corrected render must prove it came from code that
contains the fix. Without a revision in the artifact that proof needed an
out-of-band question; with it, ``git merge-base --is-ancestor <fix> <code_sha>``
answers it from the file alone. Absent git metadata records ``null`` with a
reason, never a guessed value.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from gomoveshift.pipeline import lf_provenance_sidecar as sidecar

from test_lf_provenance_sidecar import _state_full

_PKG = Path(sidecar.__file__).resolve().parent


def _head() -> str:
    return subprocess.run(
        ["git", "-C", str(_PKG), "rev-parse", "HEAD"],  # noqa: S603,S607 -- fixed argv
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _shorts_state(tmp_path):
    stages = tmp_path / "_stages"
    short = stages / "day_2026-02-24" / "shorts" / "short_01.mp4"
    return SimpleNamespace(
        stages_base=str(stages),
        shorts_per_lf={0: [str(short)]},
        per_day_completed={},
        per_lf_recording_date={},
    )


def test_lf_sidecar_records_the_code_revision(tmp_path) -> None:
    payload = json.loads(sidecar.write_lf_provenance_sidecar(tmp_path, _state_full()).read_text())
    assert payload["code_sha"] == _head()
    assert "code_sha" not in payload.get("null_reasons", {})


def test_shorts_sidecar_records_the_code_revision(tmp_path) -> None:
    path = sidecar.write_shorts_provenance_sidecar(tmp_path, _shorts_state(tmp_path))
    assert json.loads(path.read_text())["code_sha"] == _head()


def test_missing_git_metadata_is_null_with_a_reason(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sidecar, "product_sha", lambda: None)
    payload = json.loads(sidecar.write_lf_provenance_sidecar(tmp_path, _state_full()).read_text())
    assert payload["code_sha"] is None
    assert payload["null_reasons"]["code_sha"]
