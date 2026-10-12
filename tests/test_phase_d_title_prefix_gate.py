"""Phase D refuses a title set that shares a first-3-word prefix, pre-gate.

Composed (stamped) titles cannot collide, so the only titles this catches are
``titles.yaml`` hand overrides. It must fire right after the posts are built and
before ``_run_final_validation_gate`` and the later gates, not at enqueue.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from gomoveshift.pipeline.drive_phases.phase_d import PhaseD

from test_phase_d import (
    _install_fake_youtube_uploader,
    _phase_d_patches,
    _setup_mocks,
    get_config,
)
from test_phase_d import phase_d_state as phase_d_state  # noqa: F401 -- fixture re-export
from test_phase_d_verified_place_gate import TestPhaseDWiring as _Wiring
from test_phase_d_verified_place_gate import _sidecar, _verified

_PHASE_D = "gomoveshift.pipeline.drive_phases.phase_d"


def _post(idx: int, title: str):
    return SimpleNamespace(short_index=idx, youtube_title=title)


def test_colliding_hand_titles_refused_before_final_validation_gate(
    phase_d_state, tmp_path, monkeypatch
):
    _install_fake_youtube_uploader(monkeypatch)
    _Wiring._plan(phase_d_state, tmp_path, sidecar={r: _verified() for r in range(1, 7)})
    posts = {
        1: [_post(1, "Summer road trip one #Shorts"), _post(2, "Summer road trip two")],
        2: [_post(3, "Different opening words here")],
    }
    final_gate = MagicMock()
    gate_cd = MagicMock()
    with (
        patch(f"{_PHASE_D}._build_social_posts_per_lf", return_value=posts),
        patch(f"{_PHASE_D}._run_final_validation_gate", final_gate),
        _phase_d_patches()[0] as m0,
        _phase_d_patches()[1] as m1,
        _phase_d_patches()[2] as m2,
        _phase_d_patches()[3] as m3,
        _phase_d_patches()[4] as m4,
        _phase_d_patches()[5] as m5,
        _phase_d_patches()[6] as m6,
        _phase_d_patches()[7] as m7,
    ):
        _setup_mocks((m0, m1, m2, m3, m4, m5, m6, m7))
        with pytest.raises(RuntimeError, match="summer road trip") as exc_info:
            PhaseD(get_config()).run(phase_d_state, pre_upload_gate=gate_cd)
    assert "phase D" in str(exc_info.value)
    final_gate.assert_not_called()
    gate_cd.assert_not_called()
