"""Contract guard: run_day_pipeline must reject 0-based ep.index.

Two ingest paths produce
`Episode.index`:

  - execute_ingest (ingest/__init__.py) — 1-based (enumerate(..., 1))
  - episode_grouper                      — 1-based (enumerate(..., 1))
  - _synthesize_ingest_plan_from_state   — 1-based

Any future regression that reintroduces 0-based indices must fail fast
with a clear error, not silently subtract to -1 and drop every LF.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gomoveshift.pipeline.day_pipeline import run_day_pipeline


def _make_day_batch(indices):
    episodes = [
        SimpleNamespace(
            index=i,
            concat_path=Path(f"/tmp/ep{i}.mp4"),
            date="2026-04-01",
            total_duration=60.0,
            gpx_path=None,
            creation_time=None,
        )
        for i in indices
    ]
    return SimpleNamespace(
        date="2026-04-01",
        episodes=episodes,
        file_boundaries=[(0.0, 60.0)] * len(indices),
    )


def test_rejects_zero_based_episode_indices(tmp_path):
    day_batch = _make_day_batch([0, 1, 2])
    with pytest.raises(ValueError, match="1-based"):
        run_day_pipeline(
            day_batch,
            day_scores=np.zeros((10, 2)),
            output_dir=tmp_path,
            location="Test",
            lf_plans=[SimpleNamespace(index=1, segments=[], montage_peaks=[])],
        )


def test_accepts_one_based_episode_indices_without_index_error(tmp_path):
    """The guard must not false-positive on valid 1-based input. The
    downstream pipeline will skip-with-warning on empty segments; what
    matters for this test is that the guard itself does not raise."""
    day_batch = _make_day_batch([1, 2, 3])
    result = run_day_pipeline(
        day_batch,
        day_scores=np.zeros((10, 2)),
        output_dir=tmp_path,
        location="Test",
        lf_plans=[],
    )
    assert result is None or hasattr(result, "regular_lf_paths")
