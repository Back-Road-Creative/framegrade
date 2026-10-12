"""Tests for `plan_regular_lfs` full-drive mode.

The planner branches on a required ``mode`` kwarg:

- ``mode="best_moments"`` keeps the legacy sliding-window algorithm.
- ``mode="full_drive"`` spans the day MINUS masked stop spans. With a
  ``stop_mask`` it returns one ``RegularLFRange`` per maximal contiguous
  UNMASKED span (parked spans cut out); with no mask it returns a single
  ``RegularLFRange`` over the whole timeline. Nothing upstream strips
  stops — the mask is the only thing that does.

The Insta360 use case: operator wants the whole drive, not highlights —
so the LF is the entire timeline minus parked spans, capped by an explicit
``full_drive_max_duration_sec`` ceiling (checked against the KEPT/unmasked
duration) that raises rather than silently truncates.

Robust-by-construction (the service contract): ``mode`` is a required
keyword argument — there is no default. Callers that forget it must
crash at the function boundary with ``TypeError``, not silently fall
back to a best-moments cut.
"""

import numpy as np
import pytest

NUM_COLUMNS = 9
COL_TIMESTAMP = 0
COL_COMPOSITE = 1


def _make_scores(duration_sec: int, score: float = 0.5) -> np.ndarray:
    arr = np.zeros((duration_sec, NUM_COLUMNS), dtype=np.float64)
    arr[:, COL_TIMESTAMP] = np.arange(duration_sec, dtype=np.float64)
    arr[:, COL_COMPOSITE] = score
    return arr


def _keep_all_mask(duration_sec: int) -> np.ndarray:
    """A stop_mask marking nothing as parked — the explicit equivalent of
    the old (now-disallowed) maskless single-span behavior."""
    return np.zeros(duration_sec, dtype=bool)


class TestFullDriveModeRequiresStopMask:
    """Maskless full_drive planning is disallowed. A wiring bug once
    let ``stop_mask=None`` reach the planner, which
    degraded to one whole-timeline span (parked time included) and measured
    the 7200s ceiling against wall-time instead of moving-time. Loud-fail
    instead of silently shipping a stops-included video."""

    def test_full_drive_without_stop_mask_raises(self):
        from gomoveshift.pipeline.regular_lf_planner import plan_regular_lfs

        scores = _make_scores(2700)
        with pytest.raises(ValueError, match="stop_mask"):
            plan_regular_lfs(scores, mode="full_drive")

    def test_best_moments_without_stop_mask_is_unaffected(self):
        """The stop_mask requirement is full_drive-only — best_moments
        never needed a mask and must not start requiring one."""
        from gomoveshift.pipeline.regular_lf_planner import plan_regular_lfs

        scores = _make_scores(7200)
        result = plan_regular_lfs(
            scores, mode="best_moments", min_duration_sec=480, max_duration_sec=1800
        )
        assert result


class TestFullDriveModeSingleLF:
    def test_full_drive_all_false_mask_spans_whole_timeline(self):
        """With a stop mask marking nothing as parked, full_drive returns
        exactly one LF spanning the entire timeline — the first scored second
        to the last. This is the explicit equivalent of the old (now
        disallowed) maskless single-span behavior."""
        from gomoveshift.pipeline.regular_lf_planner import plan_regular_lfs

        scores = _make_scores(2700)
        result = plan_regular_lfs(scores, mode="full_drive", stop_mask=_keep_all_mask(2700))
        assert len(result) == 1
        lf = result[0]
        # First timestamp == 0, last == 2699 (sample at every integer second).
        assert lf.start_sec == 0.0
        # The end is the timestamp of the last row + 1 (1s sample resolution).
        assert lf.end_sec == 2700.0
        assert lf.duration_sec == 2700.0

    def test_full_drive_excludes_masked_stop_span(self):
        """With a stop mask marking a parked MIDDLE span, full_drive cuts that
        span out — it emits the maximal contiguous UNMASKED ranges, so the LF
        excludes the masked seconds entirely (nothing upstream strips stops;
        the mask is the only thing that does)."""
        from gomoveshift.pipeline.regular_lf_planner import plan_regular_lfs

        scores = _make_scores(2700)
        # Mark [1200, 1500) as a STOP span to excise (True == parked second).
        mask = np.zeros(2700, dtype=bool)
        mask[1200:1500] = True

        result = plan_regular_lfs(scores, mode="full_drive", stop_mask=mask)

        # Two unmasked spans survive: [0, 1200) and [1500, 2700).
        assert len(result) == 2
        assert (result[0].start_sec, result[0].end_sec) == (0.0, 1200.0)
        assert (result[1].start_sec, result[1].end_sec) == (1500.0, 2700.0)
        # The masked span is physically excluded — no returned range covers it.
        for r in result:
            assert not (r.start_sec < 1500.0 and r.end_sec > 1200.0 and r.start_sec >= 1200.0)
        # Kept duration is the whole day minus the 300s parked span.
        assert sum(r.duration_sec for r in result) == 2700.0 - 300.0

    def test_full_drive_integrated_score_is_sum(self):
        from gomoveshift.pipeline.regular_lf_planner import plan_regular_lfs

        scores = _make_scores(1200, score=0.7)
        result = plan_regular_lfs(scores, mode="full_drive", stop_mask=_keep_all_mask(1200))
        # All 1200 rows at 0.7 → integrated score == 840.
        assert result[0].integrated_score == pytest.approx(1200 * 0.7)


class TestFullDriveModeCeiling:
    def test_full_drive_respects_max_duration_ceiling(self):
        """When the post-stop-removal timeline exceeds the configured
        ceiling, the planner raises ``ValueError`` rather than silently
        truncating to the ceiling. The operator must either raise the
        ceiling explicitly or split the source."""
        from gomoveshift.pipeline.regular_lf_planner import plan_regular_lfs

        # 3 hours of scores, ceiling at 2 hours → must raise.
        scores = _make_scores(3 * 3600)
        with pytest.raises(ValueError) as excinfo:
            plan_regular_lfs(
                scores,
                mode="full_drive",
                full_drive_max_duration_sec=2 * 3600,
                stop_mask=_keep_all_mask(3 * 3600),
            )
        # Error message must name the measured duration and the ceiling so
        # operators can pick a remediation without re-running.
        msg = str(excinfo.value)
        assert "10800" in msg or "3 hour" in msg.lower() or "10800.0" in msg
        assert "7200" in msg

    def test_full_drive_at_ceiling_does_not_raise(self):
        from gomoveshift.pipeline.regular_lf_planner import plan_regular_lfs

        # Exactly at the ceiling — boundary case, must succeed.
        scores = _make_scores(7200)
        result = plan_regular_lfs(
            scores,
            mode="full_drive",
            full_drive_max_duration_sec=7200,
            stop_mask=_keep_all_mask(7200),
        )
        assert len(result) == 1
        assert result[0].duration_sec == 7200.0

    def test_full_drive_below_min_duration_raises(self):
        """If the post-stop-removal timeline is shorter than the legacy
        8-minute floor, the source is misconfigured — refuse rather than
        ship a 90-second "LF"."""
        from gomoveshift.pipeline.regular_lf_planner import plan_regular_lfs

        # 5 minutes of scores; min_duration default = 8 minutes.
        scores = _make_scores(300)
        with pytest.raises(ValueError):
            plan_regular_lfs(scores, mode="full_drive", stop_mask=_keep_all_mask(300))


class TestFullDriveModeContract:
    def test_missing_mode_raises_typeerror(self):
        """``mode`` is a required keyword-only argument — calling without
        it must raise ``TypeError`` at the function boundary, NOT silently
        fall back to a default."""
        from gomoveshift.pipeline.regular_lf_planner import plan_regular_lfs

        scores = _make_scores(2700)
        with pytest.raises(TypeError):
            plan_regular_lfs(scores)  # type: ignore[call-arg]

    def test_invalid_mode_raises_valueerror(self):
        from gomoveshift.pipeline.regular_lf_planner import plan_regular_lfs

        scores = _make_scores(2700)
        with pytest.raises(ValueError):
            plan_regular_lfs(scores, mode="nonsense")  # type: ignore[arg-type]

    def test_best_moments_mode_unchanged(self):
        """The legacy sliding-window algorithm is the ``best_moments``
        branch. Passing ``mode="best_moments"`` must produce the same
        ranges as the original behavior."""
        from gomoveshift.pipeline.regular_lf_planner import plan_regular_lfs

        scores = _make_scores(7200)
        result = plan_regular_lfs(
            scores,
            mode="best_moments",
            min_duration_sec=480,
            max_duration_sec=1800,
            target_window_sec=1200,
            max_count=4,
        )
        # Best-moments mode produces multiple non-overlapping windows.
        assert len(result) >= 2
        starts = [r.start_sec for r in result]
        assert starts == sorted(starts)
