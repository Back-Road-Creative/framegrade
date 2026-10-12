"""A maskless --superlong run hard-fails instead of silently producing
nothing. Regression pin: the dispatch used to log a warning and skip when
``stop_mask is None``, so the planner's own ``ValueError`` for a missing
mask (``superlong_planner.plan_superlong_parts``) was never reached and the
run reported success with zero superlong output."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from gomoveshift.pipeline.day_grouper import DayBatch
from _state_helpers import make_state


def _config(*, stop_mask_enabled=True):
    return SimpleNamespace(
        drive=SimpleNamespace(
            superlong_enabled=True,
            stop_mask_enabled=stop_mask_enabled,
            superlong_part_max_duration_sec=1200,
            superlong_split_grace=0.08,
        )
    )


def _run(tmp_path, date, day_scores, config):
    from gomoveshift.pipeline import day_pipeline

    ep = SimpleNamespace(
        index=1, concat_path=Path("/fake/ep_000.mp4"), total_duration=3600.0, date=date
    )
    day_batch = DayBatch(date=date, episodes=[ep])
    master_path = tmp_path / "out" / f"day_{date}" / f"day_{date}_master.mp4"

    # lf_plans=[] skips regular-LF assembly entirely (no segments to encode)
    # so the run reaches the superlong dispatch directly after the master.
    with patch("gomoveshift.pipeline.day_pipeline.encode_day_master", return_value=master_path):
        return day_pipeline.run_day_pipeline(
            day_batch=day_batch,
            day_scores=day_scores,
            output_dir=tmp_path / "out",
            location="Test Drive, Test County",
            config=config,
            lf_plans=[],
            state=make_state(ost_audit_confirmed=True),
        )


class TestMasklessSuperlongHardFails:
    def test_no_day_scores_raises_naming_missing_scores(self, tmp_path):
        """The day has no scores (stop-mask detection couldn't run at
        all) — a real operator situation, but --superlong opted in and
        can't be honoured, so it must hard-fail, not skip silently."""
        with pytest.raises(ValueError, match="no drive scores"):
            _run(tmp_path, "2026-06-01", None, _config())

    def test_stop_mask_detection_disabled_raises_naming_config(self, tmp_path):
        """Stop-mask detection is explicitly off via config — also a real
        operator situation, but incompatible with --superlong, which
        REQUIRES a mask. The message must name the config knob so the
        operator knows which case they're in."""
        scores = np.zeros((10, 9), dtype=np.float64)
        with pytest.raises(ValueError, match="stop_mask_enabled"):
            _run(tmp_path, "2026-06-01", scores, _config(stop_mask_enabled=False))
