"""Phase A refuses a full_drive day the GPS-gap cut alone leaves under the part minimum.

The drive planner cuts every second with no usable fix within the route-claim
gap (``longform_assembler._with_gps_gap_mask``) and then refuses a day whose
kept footage is under ``full_drive_part_min_sec``. That cut needs only the
parsed GPX, so the refusal is knowable before the GPU scoring pass. #229: the
03-04 drive ran its whole scoring pass (about 35 minutes of the heavy slot)
on every attempt, only for the planner to refuse 2889 s against a 3600 s floor.
"""

from types import SimpleNamespace

import pytest

from gomoveshift.analysis.shorts_models import GPSData, GPSTrackpoint
from gomoveshift.pipeline.drive_phases.phase_a import _assert_days_clear_gps_gap_floor
from gomoveshift.pipeline.ingest.day_gpx_slicer import MARKER
from gomoveshift.pipeline.regular_lf_planner import InsufficientFootageError

DAY = "2026-03-04"


def _drive(mode="full_drive", part_min=3600, part_max=7200):
    return SimpleNamespace(
        lf_selection_mode=mode,
        full_drive_part_min_sec=part_min,
        full_drive_part_max_sec=part_max,
    )


def _gps(seconds, start=1_000_000.0, *, hole_len=0, every=120, creator=None):
    """A GPX with one usable fix per second for ``seconds`` footage seconds,
    silent for ``hole_len`` s every ``every`` s (a phone logger asleep)."""
    tps = [
        GPSTrackpoint(35.0 + 1e-5 * i, -80.0, None, start + i)
        for i in range(int(seconds))
        if i % every >= hole_len
    ]
    return (GPSData(trackpoints=tps, creator=creator), start)


def _layout(durations):
    bounds, offset = [], 0.0
    for d in durations:
        bounds.append((offset, offset + d))
        offset += d
    return bounds


def _check(drive, durations, dates, gps):
    _assert_days_clear_gps_gap_floor(drive, _layout(durations), durations, dates, gps)


def test_a_day_left_under_the_floor_by_its_gps_gap_is_refused_before_scoring():
    # 6000 s raw, but the second episode carries no fix: 3000 s survive the cut.
    with pytest.raises(InsufficientFootageError, match=DAY) as exc:
        _check(_drive(), [3000.0, 3000.0], [DAY, DAY], [_gps(3000), (None, 0.0)])
    msg = str(exc.value)
    assert "3000s" in msg and "full_drive_part_min_sec=3600s" in msg
    assert "GPS" in msg and "before scoring" in msg
    assert "--fresh" not in msg


def test_a_day_with_fixes_throughout_passes():
    _check(_drive(), [3000.0, 3000.0], [DAY, DAY], [_gps(3000), _gps(3000, 2_000_000.0)])


def test_a_gps_less_day_is_never_refused_here():
    # No usable fix anywhere: the gap cut masks nothing, as in the planner.
    _check(_drive(), [3000.0, 3000.0], [DAY, DAY], [(None, 0.0), (None, 0.0)])


def test_each_day_is_judged_on_its_own_footage():
    # 03-04 is whole; 03-05 is 4000 s of which only 1000 s has fixes.
    with pytest.raises(InsufficientFootageError, match="2026-03-05"):
        _check(
            _drive(),
            [4000.0, 1000.0, 3000.0],
            [DAY, "2026-03-05", "2026-03-05"],
            [_gps(4000), _gps(1000, 2_000_000.0), (None, 0.0)],
        )


@pytest.mark.parametrize(
    "drive",
    [_drive(mode="best_moments"), _drive(part_min=None, part_max=None)],
    ids=["best_moments", "no-part-bounds"],
)
def test_inactive_unless_full_drive_with_both_part_bounds(drive):
    _check(drive, [3000.0, 3000.0], [DAY, DAY], [_gps(3000), (None, 0.0)])


def test_a_day_track_with_phone_holes_is_judged_at_the_day_track_gap():
    # Clonakilty's shape: 4000 s with a 20 s hole every 120 s. At 5 s the holes
    # cut 660 s and the day falls under 3600 s; a slicer-written track is
    # judged at 30 s and passes.
    with pytest.raises(InsufficientFootageError, match=DAY):
        _check(_drive(), [4000.0], [DAY], [_gps(4000, hole_len=20)])
    _check(_drive(), [4000.0], [DAY], [_gps(4000, hole_len=20, creator=MARKER)])


def test_the_refusal_names_the_tolerance_it_applied():
    with pytest.raises(InsufficientFootageError) as exc:
        _check(_drive(), [4000.0], [DAY], [_gps(4000, hole_len=60, every=120, creator=MARKER)])
    assert "within 30.0s" in str(exc.value)
    with pytest.raises(InsufficientFootageError) as exc:
        _check(_drive(), [4000.0], [DAY], [_gps(4000, hole_len=60, every=120)])
    assert "within 5.0s" in str(exc.value)
