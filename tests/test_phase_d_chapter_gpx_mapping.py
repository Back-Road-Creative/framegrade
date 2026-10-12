"""LF -> GPX path mapping must be captured BEFORE
``_partition_lf_plans_by_date`` mutates segment.source_file_index
from global to day-local indices.

Pre-fix bug: phase_d.py chapter generation did
``gpx_files[lf.segments[0].source_file_index]`` AFTER the partition
had already remapped that index to a DAY-LOCAL value. On day 2+ of a
multi-day drive, day-local index 0 indexed into the global gpx_files
list and returned day 1's GPX file — wrong chapter timestamps in the
YouTube description.

Fix: build the LF -> GPX path map while segment.source_file_index is
still the GLOBAL episode index, stash it in state, and read from the
map in phase_d.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
    _build_per_lf_gpx_path_map,
    _partition_lf_plans_by_date,
)


@dataclass
class _Seg:
    source_file_index: int
    start_time: float = 0.0
    end_time: float = 60.0


@dataclass
class _LF:
    index: int
    segments: list
    montage_peaks: list = field(default_factory=list)


@dataclass
class _DrivePlan:
    regular_lfs: list


@dataclass
class _Episode:
    concat_path: Path


@dataclass
class _DayBatch:
    date: str
    episodes: list


def _make_plan_3day_2ep_per_day():
    """3 days, 2 episodes each, 6 total. LF i uses episode (i-1)
    so each LF is anchored to one global episode."""
    episode_files = [Path(f"/x/ep_{i:02d}.mp4") for i in range(6)]
    gpx_files = [f"/x/ep_{i:02d}.gpx" for i in range(6)]
    episode_dates = [
        "2026-03-01",
        "2026-03-01",
        "2026-03-02",
        "2026-03-02",
        "2026-03-03",
        "2026-03-03",
    ]
    lfs = [_LF(index=i + 1, segments=[_Seg(source_file_index=i)]) for i in range(6)]
    plan = _DrivePlan(regular_lfs=lfs)
    day_batches = [
        _DayBatch(
            date=episode_dates[i * 2],
            episodes=[
                _Episode(concat_path=episode_files[i * 2]),
                _Episode(concat_path=episode_files[i * 2 + 1]),
            ],
        )
        for i in range(3)
    ]
    return plan, episode_files, episode_dates, day_batches, gpx_files


def test_multi_day_drive_picks_correct_gpx_per_lf():
    plan, episode_files, episode_dates, day_batches, gpx_files = _make_plan_3day_2ep_per_day()

    # Build mapping BEFORE partition — segments still have global indices.
    mapping = _build_per_lf_gpx_path_map(plan, gpx_files)

    # Each LF picks its own per-episode GPX file.
    assert mapping == {
        1: "/x/ep_00.gpx",
        2: "/x/ep_01.gpx",
        3: "/x/ep_02.gpx",
        4: "/x/ep_03.gpx",
        5: "/x/ep_04.gpx",
        6: "/x/ep_05.gpx",
    }

    # Now run the partition (which mutates segment indices to day-local).
    _partition_lf_plans_by_date(
        plan,
        episode_files=episode_files,
        episode_dates=episode_dates,
        day_batches=day_batches,
    )

    # Segments are now day-local — sanity-check the mutation happened.
    # LF 3 / 4 (day 2) had global indices 2, 3; now they should be 0, 1.
    assert plan.regular_lfs[2].segments[0].source_file_index == 0
    assert plan.regular_lfs[3].segments[0].source_file_index == 1
    # LF 5 / 6 (day 3) had global indices 4, 5; now 0, 1.
    assert plan.regular_lfs[4].segments[0].source_file_index == 0
    assert plan.regular_lfs[5].segments[0].source_file_index == 1

    # The pre-fix bug: indexing gpx_files[segments[0].source_file_index]
    # after partition would give LF 3 = gpx_files[0] = ep_00.gpx (wrong).
    # Demonstrate the bug exists if we did that:
    assert gpx_files[plan.regular_lfs[2].segments[0].source_file_index] == "/x/ep_00.gpx"

    # The mapping captured before partition is still correct:
    assert mapping[3] == "/x/ep_02.gpx"
    assert mapping[5] == "/x/ep_04.gpx"


def test_single_day_drive_unaffected():
    """Single-day path: global == day-local, mapping unchanged
    before vs after partition. Regression guard."""
    episode_files = [Path("/x/ep_00.mp4"), Path("/x/ep_01.mp4")]
    gpx_files = ["/x/ep_00.gpx", "/x/ep_01.gpx"]
    episode_dates = ["2026-03-01", "2026-03-01"]
    lfs = [
        _LF(index=1, segments=[_Seg(source_file_index=0)]),
        _LF(index=2, segments=[_Seg(source_file_index=1)]),
    ]
    plan = _DrivePlan(regular_lfs=lfs)
    day_batches = [
        _DayBatch(
            date="2026-03-01",
            episodes=[
                _Episode(concat_path=episode_files[0]),
                _Episode(concat_path=episode_files[1]),
            ],
        )
    ]

    mapping = _build_per_lf_gpx_path_map(plan, gpx_files)
    assert mapping == {1: "/x/ep_00.gpx", 2: "/x/ep_01.gpx"}

    _partition_lf_plans_by_date(
        plan,
        episode_files=episode_files,
        episode_dates=episode_dates,
        day_batches=day_batches,
    )

    # Single-day: global == day-local, so segment indices match the
    # original gpx_files index. Mapping is still correct.
    assert mapping[1] == "/x/ep_00.gpx"
    assert mapping[2] == "/x/ep_01.gpx"


def test_lf_with_no_segments_yields_none_entry():
    """Defensive-fallback LFs (no segments) yield an explicit ``None`` entry.

    The per_lf_gpx_path map is total — every LF
    in ``drive_plan.regular_lfs`` gets an entry, with ``None`` for any
    LF that has no resolvable GPX. Phase D treats both ``None`` and
    missing-key the same way (skip chapters), but the explicit ``None``
    is required so the resume-time
    ``state_invariants._check_per_lf_completeness`` accepts the state.
    """
    lfs = [
        _LF(index=1, segments=[_Seg(source_file_index=0)]),
        _LF(index=2, segments=[]),
    ]
    plan = _DrivePlan(regular_lfs=lfs)
    mapping = _build_per_lf_gpx_path_map(plan, ["/x/ep_00.gpx"])
    assert mapping == {1: "/x/ep_00.gpx", 2: None}


def test_gpx_files_shorter_than_segment_index_yields_none():
    """If gpx_files is shorter than the segment's global index
    (legacy / partial GPX coverage), the LF gets an explicit ``None``
    entry — phase_d treats ``None`` as 'skip chapters', same as today."""
    lfs = [_LF(index=1, segments=[_Seg(source_file_index=5)])]
    plan = _DrivePlan(regular_lfs=lfs)
    mapping = _build_per_lf_gpx_path_map(plan, ["/x/ep_00.gpx"])
    assert mapping == {1: None}


def test_multi_segment_lfs_over_different_episode_sets_do_not_alias():
    """Regression: two LFs that START in the same episode but SPAN
    different later episodes must NOT resolve to the same GPX file.

    Pre-fix, ``_build_per_lf_gpx_path_map`` keyed off ``segments[0]``
    only, so lf_A (episodes [0, 1]) and lf_B (episodes [0, 2]) — both
    starting in episode 0 — both got episode 0's GPX, producing the
    byte-identical per-LF GPS stats published as the lf_04/lf_05 defect.
    Now each LF is attributed to the episode contributing the MOST of
    its segments, so the two LFs resolve to GPX files matching THEIR
    distinct spans.
    """
    gpx_files = ["/x/ep_00.gpx", "/x/ep_01.gpx", "/x/ep_02.gpx"]
    lfs = [
        # lf_A: one segment in ep0, two in ep1 -> ep1 is the plurality.
        _LF(
            index=1,
            segments=[
                _Seg(source_file_index=0),
                _Seg(source_file_index=1),
                _Seg(source_file_index=1),
            ],
        ),
        # lf_B: one segment in ep0, two in ep2 -> ep2 is the plurality.
        _LF(
            index=2,
            segments=[
                _Seg(source_file_index=0),
                _Seg(source_file_index=2),
                _Seg(source_file_index=2),
            ],
        ),
    ]
    plan = _DrivePlan(regular_lfs=lfs)
    mapping = _build_per_lf_gpx_path_map(plan, gpx_files)

    # The regression: they must NOT both alias to episode 0's GPX.
    assert mapping[1] != mapping[2]
    assert mapping[1] == "/x/ep_01.gpx"
    assert mapping[2] == "/x/ep_02.gpx"


def test_multi_segment_tie_breaks_on_lowest_episode_index():
    """When two episodes contribute equally, the lowest global episode
    index wins — deterministic and span-direction-independent."""
    gpx_files = ["/x/ep_00.gpx", "/x/ep_01.gpx"]
    # One segment in each episode -> tie; ep0 wins on lowest index.
    lfs = [
        _LF(
            index=1,
            segments=[_Seg(source_file_index=1), _Seg(source_file_index=0)],
        ),
    ]
    plan = _DrivePlan(regular_lfs=lfs)
    mapping = _build_per_lf_gpx_path_map(plan, gpx_files)
    assert mapping == {1: "/x/ep_00.gpx"}


def test_multi_segment_skips_gps_less_episodes_in_union():
    """An LF spanning a GPS-bearing episode and a GPS-less (``None``)
    episode is attributed to the GPS-bearing one — even if the GPS-less
    episode contributes more segments — rather than collapsing to
    ``None``."""
    gpx_files = ["/x/ep_00.gpx", None, "/x/ep_02.gpx"]
    lfs = [
        _LF(
            index=1,
            segments=[
                _Seg(source_file_index=1),  # GPS-less, dropped from the count
                _Seg(source_file_index=1),  # GPS-less, dropped from the count
                _Seg(source_file_index=2),  # the only GPS-bearing episode
            ],
        ),
    ]
    plan = _DrivePlan(regular_lfs=lfs)
    mapping = _build_per_lf_gpx_path_map(plan, gpx_files)
    assert mapping == {1: "/x/ep_02.gpx"}
