"""``_build_metadata_per_lf`` must distinguish a literal trim_ratio=0
regression from the "not measured" case. A bug in phase_bc that produces
``per_lf_trim_ratio[idx] = 0`` would otherwise silently swap the title
verb (``"Highlights from"`` for ``trim_ratio < 0.5``) on what may be a
full-length drive.

The literal-zero case is NORMALISED to ``None`` at the source
(so the verb stays ``"Driving"`` and no Final-cut disclosure is implied) and
logs a greppable ``TRIM-RATIO-ZERO`` WARNING (was DEBUG).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline.drive_phases import phase_d_metadata
from gomoveshift.pipeline.drive_phases.phase_d import _build_metadata_per_lf

from _state_helpers import make_state


# _build_metadata_per_lf lives in phase_d_metadata.py; the
# source-guard reads that module's file.
PD_PATH = Path(phase_d_metadata.__file__)


def _lf(idx, *, total_duration=1800.0, segments=None):
    return SimpleNamespace(index=idx, total_duration=total_duration, segments=segments or [])


def _mk_meta_gen():
    gen = MagicMock()
    gen.generate.side_effect = lambda **kw: {"title": "t", "description": "d"}
    return gen


def test_source_guard_trim_ratio_zero_branch_logs_debug():
    """Source guard against silent removal of the elif branch."""
    src = PD_PATH.read_text()
    assert re.search(r"elif lf_trim == 0:", src), "missing 'elif lf_trim == 0' branch"
    assert "trim_ratio=0 for LF" in src, "missing debug-log payload for literal-zero case"


def test_trim_ratio_zero_normalises_to_none_and_warns(tmp_path, caplog):
    """Literal-zero ratio is normalised to None (verb stays 'Driving', no
    Final-cut disclosure) and a greppable TRIM-RATIO-ZERO WARNING fires."""
    gen = _mk_meta_gen()
    drive_plan = SimpleNamespace(regular_lfs=[_lf(1)])
    with patch(
        "gomoveshift.pipeline.validation.content_gates.check_description_redundancy",
        return_value=SimpleNamespace(passed=True, message=""),
    ):
        with caplog.at_level(logging.WARNING, logger="gomoveshift.pipeline.drive_phases.phase_d"):
            _build_metadata_per_lf(
                meta_gen=gen,
                drive_plan=drive_plan,
                regular_lf_paths=["/v/lf.mp4"],
                state=make_state(per_lf_trim_ratio={1: 0}),
                location="x",
                publish_day_full="Wednesday",
                project_base="p",
                output_base=tmp_path,
            )
    kwargs = gen.generate.call_args.kwargs
    assert kwargs["trim_ratio"] is None  # normalised from 0 → unmeasured
    assert kwargs["source_duration"] is None
    assert any("TRIM-RATIO-ZERO" in r.getMessage() for r in caplog.records)
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_trim_ratio_none_does_not_log_debug(tmp_path, caplog):
    """The "not measured" case must remain silent — DEBUG fires only for literal zero."""
    gen = _mk_meta_gen()
    drive_plan = SimpleNamespace(regular_lfs=[_lf(1)])
    with patch(
        "gomoveshift.pipeline.validation.content_gates.check_description_redundancy",
        return_value=SimpleNamespace(passed=True, message=""),
    ):
        with caplog.at_level(logging.DEBUG, logger="gomoveshift.pipeline.drive_phases.phase_d"):
            _build_metadata_per_lf(
                meta_gen=gen,
                drive_plan=drive_plan,
                regular_lf_paths=["/v/lf.mp4"],
                state=make_state(),  # no per_lf_trim_ratio entry
                location="x",
                publish_day_full="Wednesday",
                project_base="p",
                output_base=tmp_path,
            )
    assert not any("trim_ratio=0" in r.getMessage() for r in caplog.records)
    kwargs = gen.generate.call_args.kwargs
    assert kwargs["trim_ratio"] is None
    assert kwargs["source_duration"] is None


def test_positive_trim_ratio_computes_source_duration(tmp_path, caplog):
    gen = _mk_meta_gen()
    drive_plan = SimpleNamespace(regular_lfs=[_lf(1)])
    with patch(
        "gomoveshift.pipeline.validation.content_gates.check_description_redundancy",
        return_value=SimpleNamespace(passed=True, message=""),
    ):
        with caplog.at_level(logging.DEBUG, logger="gomoveshift.pipeline.drive_phases.phase_d"):
            _build_metadata_per_lf(
                meta_gen=gen,
                drive_plan=drive_plan,
                regular_lf_paths=["/v/lf.mp4"],
                state=make_state(per_lf_trim_ratio={1: 0.5}),
                location="x",
                publish_day_full="Wednesday",
                project_base="p",
                output_base=tmp_path,
            )
    kwargs = gen.generate.call_args.kwargs
    assert kwargs["trim_ratio"] == 0.5
    assert kwargs["source_duration"] == 3600.0
    assert not any("trim_ratio=0" in r.getMessage() for r in caplog.records)
