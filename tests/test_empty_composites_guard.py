"""Guard np.percentile against empty composite arrays.

An empty ``composites`` array reaching ``np.percentile`` raised an opaque
IndexError (numpy >= 1.22) / produced nan on unattended runs. Both
``_adaptive_threshold_search`` and ``build_drive_plan`` must return cleanly on
empty input, mirroring the existing guard in ``_build_full_drive_peak_pool``.

TDD: these tests are written before the guards exist.
"""

import numpy as np

from gomoveshift.config import DriveConfig
from gomoveshift.pipeline.longform_assembler import LongFormAssembler


def _empty_scores() -> np.ndarray:
    """Zero-row score array with the canonical 9-column layout."""
    return np.zeros((0, 9), dtype=np.float64)


class TestEmptyCompositesGuard:
    def test_adaptive_threshold_search_empty_returns_clean(self):
        """Empty day_scores → ([], 0.0) instead of IndexError from np.percentile."""
        assembler = LongFormAssembler(DriveConfig())

        segments, threshold = assembler._adaptive_threshold_search(
            day_scores=_empty_scores(),
            day_file_boundaries=[],
        )

        assert segments == []
        assert threshold == 0.0

    def test_build_drive_plan_empty_returns_empty_plan(self):
        """Empty scores → DrivePlan with no LFs instead of IndexError."""
        assembler = LongFormAssembler(DriveConfig())

        plan = assembler.build_drive_plan(
            scores=_empty_scores(),
            processed_files=[],
            file_boundaries=[],
            region="usa",
        )

        assert plan.regular_lfs == []
        assert plan.score_threshold == 0.0
        assert plan.total_source_duration == 0.0
        assert plan.total_highlight_duration == 0.0
