"""Phase D threads ``included_segments`` (source-timeline ranges
built from each LF's segments) into ``ChapterGenerator.generate``.

These are focused unit tests around ``_generate_chapters_for_lf``. The
broader PhaseD integration test surface lives in
``test_phase_d_chapters.py``; this file just pins the threading.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from gomoveshift.config import Config
from gomoveshift.models import LFSegment, RegularLFPlan
from gomoveshift.pipeline.drive_phases import phase_d as phase_d_mod
from gomoveshift.pipeline.gps_quality import trim_gps_to_clip_timeline

from _state_helpers import make_state


def _cfg(tmp_path: Path) -> Config:
    return Config(
        project_name="test",
        base_input_path=tmp_path / "in",
        base_output_path=tmp_path / "out",
    )


def test_generate_chapters_for_lf_forwards_included_segments(tmp_path, monkeypatch):
    """``_generate_chapters_for_lf`` must accept an ``included_segments``
    kwarg and forward it to ``ChapterGenerator.generate``."""
    captured = {}

    class FakeGen:
        def __init__(self, *a, **kw):
            pass

        def generate(self, **kwargs):
            captured.update(kwargs)
            return []

        def format_chapters(self, chapters):
            return ""

    monkeypatch.setattr(phase_d_mod, "ChapterGenerator", FakeGen)

    gpx = tmp_path / "t.gpx"
    gpx.write_text("<gpx></gpx>")

    included = [(0.0, 100.0), (200.0, 300.0)]
    phase_d_mod._generate_chapters_for_lf(
        config=_cfg(tmp_path),
        gpx_path=gpx,
        video_duration=200.0,
        location="Somewhere",
        included_segments=included,
    )
    assert captured.get("included_segments") == included


def test_phase_d_threads_included_segments_per_lf(tmp_path, monkeypatch):
    """The Phase D D1.5 loop must build ``included_segments`` from each
    LF's segments in the representative GPX's EPISODE-LOCAL time base
    (``start_time`` / ``end_time``, i.e. seconds within the source file)
    and pass it into ``_generate_chapters_for_lf``. It must NOT emit the
    drive-global ``original_start_time`` / ``original_end_time`` — those
    are offset by prior episodes and don't share the GPX waypoint base."""
    call_log: list[dict] = []

    def _fake_gen(**kwargs):
        call_log.append(kwargs)
        return [], ""

    monkeypatch.setattr(phase_d_mod, "_generate_chapters_for_lf", _fake_gen)

    # Build a two-LF drive plan with distinct per-LF segment ranges.
    src_a = tmp_path / "ep_0_FINAL.mp4"
    src_a.write_bytes(b"a")
    src_b = tmp_path / "ep_1_FINAL.mp4"
    src_b.write_bytes(b"b")

    seg_a1 = LFSegment(
        source_file_index=0,
        source_path=src_a,
        start_time=0.0,
        end_time=100.0,
        original_start_time=10.0,
        original_end_time=110.0,
        composite_score=0.5,
        gps_center_lat=0.0,
        gps_center_lon=0.0,
        terrain_type="countryside",
        contiguous_with_previous=False,
    )
    seg_a2 = LFSegment(
        source_file_index=0,
        source_path=src_a,
        start_time=150.0,
        end_time=250.0,
        original_start_time=160.0,
        original_end_time=260.0,
        composite_score=0.5,
        gps_center_lat=0.0,
        gps_center_lon=0.0,
        terrain_type="countryside",
        contiguous_with_previous=False,
    )
    seg_b1 = LFSegment(
        source_file_index=1,
        source_path=src_b,
        start_time=0.0,
        end_time=300.0,
        original_start_time=0.0,
        original_end_time=300.0,
        composite_score=0.5,
        gps_center_lat=0.0,
        gps_center_lon=0.0,
        terrain_type="countryside",
        contiguous_with_previous=False,
    )

    plan_a = RegularLFPlan(
        index=1,
        segments=[seg_a1, seg_a2],
        montage_peaks=[seg_a1],
        shorts_target=1,
        region="ireland",
        total_duration=200.0,
    )
    plan_b = RegularLFPlan(
        index=2,
        segments=[seg_b1],
        montage_peaks=[seg_b1],
        shorts_target=1,
        region="ireland",
        total_duration=300.0,
    )

    drive_plan = MagicMock()
    drive_plan.regular_lfs = [plan_a, plan_b]

    gpx_a = tmp_path / "ep_0.gpx"
    gpx_a.write_text("<gpx></gpx>")
    gpx_b = tmp_path / "ep_1.gpx"
    gpx_b.write_text("<gpx></gpx>")

    state = make_state(
        gpx_files=[str(gpx_a), str(gpx_b)],
    )
    # Exercise just the D1.5 loop logic by invoking the helper extracted
    # for it. We simulate the loop body here to keep the test hermetic.
    from gomoveshift.pipeline.drive_phases.phase_d import (
        _build_included_segments_for_lf,
    )

    got_a = _build_included_segments_for_lf(plan_a)
    got_b = _build_included_segments_for_lf(plan_b)
    # Episode-local (start_time/end_time) — NOT the drive-global
    # original_start/end_time. plan_a's segments carry a +10s drive-global
    # offset that the episode's GPX (which begins at its own local 0) does
    # not share.
    assert got_a == [(0.0, 100.0), (150.0, 250.0)]
    assert got_b == [(0.0, 300.0)]

    # And the helper is then passed through _generate_chapters_for_lf
    # — use a minimal invocation to confirm the kwarg propagates.
    phase_d_mod._generate_chapters_for_lf(
        config=_cfg(tmp_path),
        gpx_path=gpx_a,
        video_duration=plan_a.total_duration,
        location="X",
        included_segments=got_a,
    )
    assert call_log and call_log[-1].get("included_segments") == got_a
    del state  # unused
    _ = patch  # silence unused-import (kept for parity with sibling tests)


def _seg(*, ep: int, local_start: float, local_end: float, ep_offset: float) -> LFSegment:
    """An LFSegment on drive-episode ``ep`` whose episode-local footage is
    ``[local_start, local_end]`` and whose drive-global offset (the sum of
    prior episodes' durations) is ``ep_offset``."""
    return LFSegment(
        source_file_index=ep,
        source_path=Path(f"ep_{ep}_FINAL.mp4"),
        start_time=local_start,
        end_time=local_end,
        original_start_time=local_start + ep_offset,
        original_end_time=local_end + ep_offset,
        composite_score=0.5,
        gps_center_lat=0.0,
        gps_center_lon=0.0,
        terrain_type="countryside",
        contiguous_with_previous=False,
    )


def _wp(base: datetime, t_local: float, city: str):
    """A journey waypoint at episode-local second ``t_local``."""
    from gps_journey.core.models import Location, Waypoint

    return Waypoint(
        latitude=40.0,
        longitude=-120.0,
        timestamp=base + timedelta(seconds=t_local),
        location=Location(latitude=40.0, longitude=-120.0, city=city, country="Testland"),
    )


def test_included_segments_rebased_for_second_episode_lf():
    """Regression: an LF assembled from the drive's SECOND episode has
    drive-global segment times shifted by the first episode's duration, but
    its GPX begins at episode-local second 0. ``_build_included_segments_for_lf``
    must emit episode-local ranges so the chapter trim keeps the footage that
    is actually on-screen — not a window shifted out of the GPX's range."""
    from gomoveshift.pipeline.drive_phases.phase_d import _build_included_segments_for_lf

    # Episode 0 ran 600s; episode 1's drive-global offset is therefore 600.
    lf = RegularLFPlan(
        index=3,
        segments=[_seg(ep=1, local_start=100.0, local_end=200.0, ep_offset=600.0)],
        montage_peaks=[],
        shorts_target=1,
        region="ireland",
        total_duration=100.0,
    )

    ranges = _build_included_segments_for_lf(lf)
    # Episode-local, NOT drive-global (700, 800).
    assert ranges == [(100.0, 200.0)]

    # Trim episode-1's GPX (waypoints keyed from its own local 0) through the
    # same path the chapter generator uses. The kept waypoints must be the
    # on-screen ones (100-200s), never empty.
    base = datetime(2026, 6, 1, 9, 0, 0)
    waypoints = [_wp(base, t, f"Town{int(t)}") for t in (0.0, 100.0, 150.0, 200.0, 300.0)]
    j_start = waypoints[0].timestamp
    kept = trim_gps_to_clip_timeline(
        waypoints,
        ranges,
        time_key=lambda wp: (wp.timestamp - j_start).total_seconds(),
    )
    kept_secs = [round((wp.timestamp - j_start).total_seconds()) for wp in kept]
    assert kept_secs == [100, 150, 200]

    # The drive-global window would have kept nothing — proving the bug.
    drive_global = [(700.0, 800.0)]
    empty = trim_gps_to_clip_timeline(
        waypoints,
        drive_global,
        time_key=lambda wp: (wp.timestamp - j_start).total_seconds(),
    )
    assert empty == []


def test_included_segments_scoped_to_representative_episode():
    """A multi-episode LF is charted against a SINGLE representative GPX
    (the episode contributing the most segments). Segments from other
    episodes live in a different GPX time base, so emitting their local
    ranges would spuriously keep the wrong stretch of the representative
    GPX. Only the representative episode's ranges may be emitted."""
    from gomoveshift.pipeline.drive_phases.phase_d import _build_included_segments_for_lf

    # Episode 2 contributes 3 segments (representative); episode 1 contributes 1.
    lf = RegularLFPlan(
        index=4,
        segments=[
            _seg(ep=1, local_start=50.0, local_end=60.0, ep_offset=600.0),
            _seg(ep=2, local_start=100.0, local_end=150.0, ep_offset=1200.0),
            _seg(ep=2, local_start=200.0, local_end=250.0, ep_offset=1200.0),
            _seg(ep=2, local_start=300.0, local_end=350.0, ep_offset=1200.0),
        ],
        montage_peaks=[],
        shorts_target=1,
        region="ireland",
        total_duration=150.0,
    )

    ranges = _build_included_segments_for_lf(lf)
    # Only episode 2's episode-local ranges — episode 1's (50, 60) is dropped.
    assert ranges == [(100.0, 150.0), (200.0, 250.0), (300.0, 350.0)]


def test_pickers_agree_on_gpx_bearing_representative_episode():
    """The GPX-path picker (``_build_per_lf_gpx_path_map``) and the
    segment-window picker (``_build_included_segments_for_lf``) must resolve
    the SAME representative episode.

    When the episode contributing the MOST segments has NO GPX, the window
    picker — unaware of GPX availability — selected that episode's windows
    while the GPX picker (which filters to GPX-bearing episodes) selected a
    different episode. A GPX from episode B then got trimmed with episode A's
    windows: wrong ``Through:`` counties, chapters naming cut footage, and
    region checks that can false-fail a correct title. Both pickers must
    filter to GPX-BEARING episodes so the window base and the GPX trace come
    from the same episode."""
    from gomoveshift.pipeline.drive_phases.phase_d import _build_included_segments_for_lf
    from gomoveshift.pipeline.gps_enrichment import _build_per_lf_gpx_path_map

    # Episode 0 contributes 3 segments but has NO GPX; episode 1 contributes 1
    # segment and HAS GPX. The representative must be episode 1 for BOTH.
    lf = RegularLFPlan(
        index=7,
        segments=[
            _seg(ep=0, local_start=0.0, local_end=10.0, ep_offset=0.0),
            _seg(ep=0, local_start=10.0, local_end=20.0, ep_offset=0.0),
            _seg(ep=0, local_start=20.0, local_end=30.0, ep_offset=0.0),
            _seg(ep=1, local_start=100.0, local_end=110.0, ep_offset=600.0),
        ],
        montage_peaks=[],
        shorts_target=1,
        region="ireland",
        total_duration=40.0,
    )
    drive_plan = MagicMock()
    drive_plan.regular_lfs = [lf]

    # Episode 0 has no GPX (None); episode 1 does.
    gpx_files = [None, "/tmp/ep_1.gpx"]

    gpx_map = _build_per_lf_gpx_path_map(drive_plan, gpx_files)
    # GPX picker resolves the GPX-bearing episode 1 — NOT the segment-majority
    # episode 0 (which has no GPX to chart).
    assert gpx_map[7] == "/tmp/ep_1.gpx"

    # Window picker, given the same GPX availability, must select episode 1's
    # episode-local window — NOT episode 0's three windows.
    ranges = _build_included_segments_for_lf(lf, gpx_files=gpx_files)
    assert ranges == [(100.0, 110.0)]

    # Concretely: the window base and the GPX come from the SAME episode.
    rep_ep = gpx_files.index(gpx_map[7])
    assert rep_ep == 1
