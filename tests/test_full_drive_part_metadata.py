"""Per-part metadata for split full_drive days (part_index/part_count set).

Each part's numbers, location and chapter window come only from that part's
own segments, keyed by ``lf.index``; titles carry the superlong ``— Part k of
K`` suffix; an unsplit LF (part_* None) is untouched.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from _state_helpers import make_state

from gomoveshift.pipeline.drive_phases.phase_d_metadata import (
    _build_lf_multi_episode_journey,
    _build_metadata_per_lf,
    _generate_chapters_all_lfs,
)
from gomoveshift.pipeline.title_builder import append_full_drive_part_suffix


def _seg(idx, start, end, g_start=None, g_end=None):
    return SimpleNamespace(
        source_file_index=idx,
        start_time=start,
        end_time=end,
        original_start_time=start if g_start is None else g_start,
        original_end_time=end if g_end is None else g_end,
    )


def _lf(idx, part=None, count=None, segments=None, dur=3900.0):
    return SimpleNamespace(
        index=idx,
        total_duration=dur,
        segments=segments or [],
        part_index=part,
        part_count=count,
    )


def _gen(title="Driving Kerry, Ireland in 4K"):
    gen = MagicMock()
    gen.generate.side_effect = lambda **kw: {"title": title, "description": "d"}
    return gen


def _run(tmp_path, lfs, state, gen=None):
    gen = gen or _gen()
    with patch(
        "gomoveshift.pipeline.validation.content_gates.check_description_redundancy",
        return_value=SimpleNamespace(passed=True, message=""),
    ):
        out = _build_metadata_per_lf(
            meta_gen=gen,
            drive_plan=SimpleNamespace(regular_lfs=lfs),
            regular_lf_paths=[f"/v/lf{lf.index}.mp4" for lf in lfs],
            state=state,
            location="Ireland",
            publish_day_full="Sunday",
            project_base="p",
            output_base=tmp_path,
        )
    return out, gen


def _kw(gen, idx):
    return next(
        c.kwargs
        for c in gen.generate.call_args_list
        if c.kwargs["project_name"].endswith(f"lf{idx:02d}")
    )


class TestPartNumbers:
    def test_part_without_own_window_never_gets_whole_trace_stats(self, tmp_path):
        lfs = [_lf(1, 1, 2), _lf(2, 2, 2)]
        state = make_state(
            per_lf_distance_km={1: 40.0, 2: 80.0},
            per_lf_speed={1: {"avg_kmh": 30.0}, 2: {"avg_kmh": 50.0}},
        )
        _, gen = _run(tmp_path, lfs, state)
        for i in (1, 2):
            kw = _kw(gen, i)
            assert kw["distance_km"] is None
            assert kw["speed"] is None
            assert kw["full_drive_stats"] is None

    def test_part_with_own_window_keeps_only_its_own_numbers(self, tmp_path):
        lfs = [_lf(1, 1, 2), _lf(2, 2, 2)]
        state = make_state(
            per_lf_distance_km={1: 40.0, 2: 31.5},
            per_lf_gpx_episode_paths={1: {"a.gpx": [(0, 1)]}, 2: {"b.gpx": [(5, 9)]}},
        )
        _, gen = _run(tmp_path, lfs, state)
        kw = _kw(gen, 2)
        assert kw["distance_km"] == 31.5
        assert kw["full_drive_stats"] is None

    def test_unsplit_lf_whole_trace_flag_unchanged(self, tmp_path):
        state = make_state(per_lf_distance_km={1: 22.0})
        _, gen = _run(tmp_path, [_lf(1)], state)
        kw = _kw(gen, 1)
        assert kw["distance_km"] == 22.0
        assert sorted(kw["full_drive_stats"]) == ["distance", "speed"]

    def test_part_location_is_its_own_index(self, tmp_path):
        lfs = [_lf(1, 1, 2), _lf(2, 2, 2)]
        state = make_state(per_lf_locations={1: "Mizen, Cork", 2: "Schull, Cork"})
        _, gen = _run(tmp_path, lfs, state)
        assert _kw(gen, 1)["location"] == "Mizen, Cork"
        assert _kw(gen, 2)["location"] == "Schull, Cork"


class TestPartTitles:
    def test_k2_titles_carry_part_suffix(self, tmp_path):
        out, _ = _run(tmp_path, [_lf(1, 1, 2), _lf(2, 2, 2)], make_state())
        assert out[1]["title"].endswith(" — Part 1 of 2")
        assert out[2]["title"].endswith(" — Part 2 of 2")
        assert len(out[2]["title"]) <= 100

    def test_k1_and_none_titles_unchanged(self, tmp_path):
        out, _ = _run(tmp_path, [_lf(1), _lf(2, 1, 1)], make_state())
        assert out[1]["title"] == "Driving Kerry, Ireland in 4K"
        assert out[2]["title"] == "Driving Kerry, Ireland in 4K"

    def test_identical_part_labels_are_not_double_suffixed(self, tmp_path):
        state = make_state(per_lf_ost_text={1: "Schull, County Cork", 2: "Schull, County Cork"})
        out, _ = _run(tmp_path, [_lf(1, 1, 2), _lf(2, 2, 2)], state)
        for i, t in out.items():
            assert t["title"].count("Part") == 1, t["title"]

    def test_suffix_helper_idempotent_and_clamped(self):
        t = append_full_drive_part_suffix("Driving X in 4K", 2, 2)
        assert t == "Driving X in 4K — Part 2 of 2"
        assert append_full_drive_part_suffix(t, 2, 2) == t
        long = append_full_drive_part_suffix("word " * 40, 1, 3)
        assert len(long) <= 100 and long.endswith(" — Part 1 of 3")
        assert append_full_drive_part_suffix("T", None, None) == "T"
        assert append_full_drive_part_suffix("T", 1, 1) == "T"


class TestPartChapters:
    def _gps(self):
        pts = [
            SimpleNamespace(timestamp=float(t), lat=51.0 + t * 1e-5, lon=-9.0)
            for t in range(0, 3000, 10)
        ]
        entry = (SimpleNamespace(trackpoints=pts), 0.0)
        return [entry, entry]

    def test_part2_journey_and_ranges_restart_at_zero(self):
        # Part 2 spans two episodes; drive-global windows start late (2000s).
        segs = [_seg(0, 100.0, 400.0, 2000.0, 2300.0), _seg(1, 0.0, 300.0, 2600.0, 2900.0)]
        lf = _lf(2, 2, 2, segs)
        with patch(
            "gomoveshift.analysis.shorts_models.effective_offset", side_effect=lambda g, o: o
        ):
            journey, ranges = _build_lf_multi_episode_journey(lf, self._gps(), gpx_files=["a", "b"])
        assert ranges[0][0] >= 0.0
        first_wp = journey.waypoints[0].timestamp
        assert (journey.start_date - first_wp).total_seconds() == 0.0
        # Rebased onto the part's own journey: no day-global (2000s+) offset leaks.
        assert ranges[-1][1] < 1100.0

    def test_part2_chapters_get_only_its_own_ranges(self, tmp_path):
        gpx = tmp_path / "ep.gpx"
        gpx.write_text("<gpx></gpx>")
        lf = _lf(2, 2, 2, [_seg(0, 1800.0, 3600.0)])
        meta = {2: {"description": "d", "title": "T"}}
        seen = {}

        def fake(**kw):
            seen.update(kw)
            return ([{"time": "0:00", "title": "Start"}], "0:00 Start")

        with patch(
            "gomoveshift.pipeline.drive_phases.phase_d._generate_chapters_for_lf", side_effect=fake
        ):
            _generate_chapters_all_lfs(
                config=MagicMock(),
                drive_plan=SimpleNamespace(regular_lfs=[lf]),
                state=make_state(per_lf_gpx_path={2: str(gpx)}, gpx_files=[str(gpx)]),
                metadata_per_lf=meta,
                location="x",
                project_base="p",
                output_base=tmp_path,
                stages_dir=tmp_path,
            )
        assert seen["included_segments"] == [(1800.0, 3600.0)]
        assert seen["video_duration"] == lf.total_duration
