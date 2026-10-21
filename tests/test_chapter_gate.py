"""Drive-path chapter renderability hard gate.

The single-video path returns ``success=False`` when GPS is present but chapter
generation fails or yields too few chapters (``stages/chapters.py``). The drive
path used to swallow the failure at WARN and embed an empty / 2-chapter block
into the description — which YouTube silently refuses to render (needs
``>= YOUTUBE_MIN_CHAPTERS``). ``_generate_chapters_all_lfs`` now fails loud with
a greppable ``CHAPTER-GATE`` ERROR and skips the embed for a sub-threshold
block.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from gomoveshift.pipeline.chapters.generator import YOUTUBE_MIN_CHAPTERS
from gomoveshift.pipeline.drive_phases.phase_d import _generate_chapters_all_lfs

from _state_helpers import make_state


def _lf(idx, *, total_duration=1800.0, segments=None):
    return SimpleNamespace(index=idx, total_duration=total_duration, segments=segments or [])


def _run(tmp_path, *, chapters_return=None, side_effect=None, caplog=None):
    config = MagicMock()
    drive_plan = SimpleNamespace(regular_lfs=[_lf(1)])
    gpx = tmp_path / "lf1.gpx"
    gpx.write_text("<gpx></gpx>")
    meta_per_lf = {1: {"description": "Hook line"}}
    gen_patch = patch(
        "gomoveshift.pipeline.drive_phases.phase_d._generate_chapters_for_lf",
        **(
            {"side_effect": side_effect}
            if side_effect is not None
            else {"return_value": chapters_return}
        ),
    )
    with (
        gen_patch,
        patch(
            "gomoveshift.pipeline.drive_phases.phase_d._LFMeta._embed_chapters",
            return_value="Hook line\n\n0:00 Intro\n1:00 Mid\n2:00 End",
        ) as embed_mock,
        patch(
            "gomoveshift.pipeline.validation.content_gates.check_description_redundancy",
            return_value=SimpleNamespace(passed=True, message=""),
        ),
    ):
        _generate_chapters_all_lfs(
            config=config,
            drive_plan=drive_plan,
            state=make_state(per_lf_gpx_path={1: str(gpx)}),
            metadata_per_lf=meta_per_lf,
            location="x",
            project_base="p",
            output_base=tmp_path,
            stages_dir=tmp_path,
        )
    return meta_per_lf, embed_mock


def _enough():
    return [{"start": i * 60, "text": f"c{i}"} for i in range(YOUTUBE_MIN_CHAPTERS)]


class TestChapterRenderabilityGate:
    def test_renderable_block_is_embedded(self, tmp_path):
        meta, embed_mock = _run(tmp_path, chapters_return=(_enough(), "0:00 c0"))
        embed_mock.assert_called_once()
        assert "0:00 Intro" in meta[1]["description"]

    def test_subthreshold_block_not_embedded(self, tmp_path, caplog):
        # 2 chapters < YOUTUBE_MIN_CHAPTERS (3) → YouTube won't render → skip embed.
        two = [{"start": 0, "text": "a"}, {"start": 60, "text": "b"}]
        with caplog.at_level("ERROR"):
            meta, embed_mock = _run(tmp_path, chapters_return=(two, "0:00 a\n1:00 b"))
        embed_mock.assert_not_called()
        assert meta[1]["description"] == "Hook line"  # untouched
        assert any("CHAPTER-GATE" in r.getMessage() for r in caplog.records)

    def test_generation_failure_logs_chapter_gate_error(self, tmp_path, caplog):
        with caplog.at_level("ERROR"):
            meta, embed_mock = _run(tmp_path, side_effect=RuntimeError("gpx parse oops"))
        embed_mock.assert_not_called()
        assert meta[1]["chapters"] == []
        assert any(
            "CHAPTER-GATE" in r.getMessage() and r.levelname == "ERROR" for r in caplog.records
        )


# ── chapter names as search terms ──────────────────────────────────────


from gomoveshift.models import ResolvedLocation  # noqa: E402
from gomoveshift.pipeline.chapters.generator import (  # noqa: E402
    YOUTUBE_MIN_CHAPTER_GAP_SEC,
    ChapterGenerator,
)


def _secs(ts: str) -> int:
    s = 0
    for part in ts.split(":"):
        s = s * 60 + int(part)
    return s


def _pad_with(seq, chapters, video_duration, label=None):
    """Run _pad_to_min_chapters with a ResolvedLocation place sequence set."""
    gen = object.__new__(ChapterGenerator)
    if seq is not None:
        gen._resolved_location = ResolvedLocation(ordered_place_sequence=list(seq))
    return gen._pad_to_min_chapters(list(chapters), video_duration, label)


class TestChapterNamesSeo:
    def test_pads_carry_real_town_names_in_travel_order(self):
        # Two real chapters bunched at the start, one long tail gap → pads land
        # late and must pick towns from the END of the travel-ordered sequence.
        seq = ["Killarney", "Kenmare", "Sneem", "Waterville"]
        chapters = [
            {"timestamp": "0:00", "title": "Start - A"},
            {"timestamp": "0:20", "title": "Finish - B"},
        ]
        out = _pad_with(seq, chapters, video_duration=600.0, label="A")
        assert len(out) >= YOUTUBE_MIN_CHAPTERS
        assert out[0]["timestamp"] == "0:00"
        pad_titles = [c["title"] for c in out if c["title"] not in ("Start - A", "Finish - B")]
        assert pad_titles, [c["title"] for c in out]
        # Real towns, never the generic interior label.
        assert all("En route" not in t for t in pad_titles), pad_titles
        # Each pad's town is one of the late-sequence towns (tail-positioned).
        assert all(any(town in t for town in seq[-3:]) for t in pad_titles), pad_titles

    def test_back_compat_generic_name_without_resolver(self):
        chapters = [
            {"timestamp": "0:00", "title": "Start - A"},
            {"timestamp": "0:20", "title": "Finish - B"},
        ]
        # No resolver → unchanged "En route — {place}".
        out = _pad_with(None, chapters, video_duration=600.0, label="A")
        pads = [c["title"] for c in out if c["title"].startswith("En route")]
        assert pads and all(t == "En route — A" for t in pads), [c["title"] for c in out]
        # Empty sequence behaves identically (degraded resolver).
        out2 = _pad_with([], list(chapters), video_duration=600.0, label="A")
        pads2 = [c["title"] for c in out2 if c["title"].startswith("En route")]
        assert pads2 and all(t == "En route — A" for t in pads2)

    def test_invariants_hold_first_at_zero_min_gap_no_dupes(self):
        seq = ["Killarney", "Kenmare", "Sneem"]
        chapters = [
            {"timestamp": "0:00", "title": "Start - A"},
            {"timestamp": "5:00", "title": "Entering Mid"},
        ]
        out = _pad_with(seq, chapters, video_duration=900.0, label="A")
        assert len(out) >= YOUTUBE_MIN_CHAPTERS
        times = [_secs(c["timestamp"]) for c in out]
        assert times[0] == 0
        assert all(b - a >= YOUTUBE_MIN_CHAPTER_GAP_SEC for a, b in zip(times, times[1:])), times
        titles = [c["title"] for c in out]
        assert all(a != b for a, b in zip(titles, titles[1:])), titles

    def test_picks_different_town_over_duplicating_neighbour(self):
        # The big tail gap sits to the RIGHT of a real chapter named "Sneem".
        # The pad lands ~mid-tail (fraction ~0.6 → seq index 1 → "Sneem"),
        # which equals its left neighbour — a DIFFERENT town must be picked
        # rather than a timestamp-disambiguated "Sneem (..)" duplicate.
        seq = ["Kenmare", "Sneem"]
        chapters = [
            {"timestamp": "0:00", "title": "Start - A"},
            {"timestamp": "2:00", "title": "Sneem"},
        ]
        out = _pad_with(seq, chapters, video_duration=600.0, label="A")
        titles = [c["title"] for c in out]
        assert all(a != b for a, b in zip(titles, titles[1:])), titles
        pad_titles = [t for t in titles if t not in ("Start - A", "Sneem")]
        # The pad avoided "Sneem" (left neighbour) and chose the other town,
        # with no timestamp disambiguator.
        assert pad_titles == ["Kenmare"], titles

    def test_short_lf_left_untouched_with_sequence(self):
        seq = ["Killarney", "Kenmare"]
        chapters = [{"timestamp": "0:00", "title": "Start - A"}]
        out = _pad_with(seq, chapters, video_duration=5.0, label="A")
        assert out == chapters


# ── A chapter-floor shortfall is repaired in Phase D, not raised ─────────────

from gomoveshift.pipeline.validation.content_gates import check_metadata_quality  # noqa: E402

LONG_LF_SECONDS = 61 * 60  # a long LF that needs 4 middle chapters


def _chapters(middles):
    out = [{"timestamp": "0:00", "title": "Start - Cape Harrow"}]
    out += [{"timestamp": f"{5 + i * 10}:00", "title": f"Place {i}"} for i in range(middles)]
    out.append({"timestamp": "60:45", "title": "Finish - Harbor Town"})
    return out, "\n".join(f"{c['timestamp']} {c['title']}" for c in out)


class TestChapterFloorSelfHeal:
    def test_sparse_first_pass_is_regenerated_and_clears_the_gate(self, tmp_path, caplog):
        config = MagicMock()
        drive_plan = SimpleNamespace(regular_lfs=[_lf(1, total_duration=LONG_LF_SECONDS)])
        gpx = tmp_path / "lf1.gpx"
        gpx.write_text("<gpx></gpx>")
        meta_per_lf = {1: {"description": "Hook line", "video_duration": LONG_LF_SECONDS}}
        with (
            patch(
                "gomoveshift.pipeline.drive_phases.phase_d._generate_chapters_for_lf",
                side_effect=[_chapters(2), _chapters(4)],
            ) as gen_mock,
            patch(
                "gomoveshift.pipeline.validation.content_gates.check_description_redundancy",
                return_value=SimpleNamespace(passed=True, message=""),
            ),
            caplog.at_level("WARNING"),
        ):
            _generate_chapters_all_lfs(
                config=config,
                drive_plan=drive_plan,
                state=make_state(per_lf_gpx_path={1: str(gpx)}),
                metadata_per_lf=meta_per_lf,
                location="x",
                project_base="p",
                output_base=tmp_path,
                stages_dir=tmp_path,
            )
        assert gen_mock.call_count == 2
        assert len(meta_per_lf[1]["chapters"]) == 6
        gate = check_metadata_quality(meta_per_lf[1])
        assert "intermediate chapter" not in gate.message, gate.message
        assert any("first pass under the chapter floor" in r.getMessage() for r in caplog.records)

    def test_floor_met_on_first_pass_generates_once(self, tmp_path):
        config = MagicMock()
        drive_plan = SimpleNamespace(regular_lfs=[_lf(1, total_duration=LONG_LF_SECONDS)])
        gpx = tmp_path / "lf1.gpx"
        gpx.write_text("<gpx></gpx>")
        with (
            patch(
                "gomoveshift.pipeline.drive_phases.phase_d._generate_chapters_for_lf",
                return_value=_chapters(4),
            ) as gen_mock,
            patch(
                "gomoveshift.pipeline.validation.content_gates.check_description_redundancy",
                return_value=SimpleNamespace(passed=True, message=""),
            ),
        ):
            _generate_chapters_all_lfs(
                config=config,
                drive_plan=drive_plan,
                state=make_state(per_lf_gpx_path={1: str(gpx)}),
                metadata_per_lf={1: {"description": "Hook line"}},
                location="x",
                project_base="p",
                output_base=tmp_path,
                stages_dir=tmp_path,
            )
        assert gen_mock.call_count == 1


# ── The chapter floor is checked before the master encode starts ─────────────

import inspect  # noqa: E402

import pytest  # noqa: E402

from gomoveshift.pipeline.drive_phases import phase_bc_day_first  # noqa: E402
from gomoveshift.pipeline.drive_phases.phase_d_metadata import (  # noqa: E402
    assert_chapter_floor_before_encode,
)


def _preflight(tmp_path, side_effect, *, gpx=True):
    path = tmp_path / "lf1.gpx"
    path.write_text("<gpx></gpx>")
    state = make_state(per_lf_gpx_path={1: str(path)} if gpx else {})
    with patch(
        "gomoveshift.pipeline.drive_phases.phase_d._generate_chapters_for_lf",
        side_effect=side_effect,
    ) as gen_mock:
        assert_chapter_floor_before_encode(
            MagicMock(),
            SimpleNamespace(regular_lfs=[_lf(1, total_duration=LONG_LF_SECONDS)]),
            state,
            "x",
            tmp_path,
        )
    return gen_mock


class TestChapterFloorBeforeEncode:
    def test_unchapterable_trace_refuses_the_encode(self, tmp_path):
        with pytest.raises(RuntimeError, match=r"refusing to start the encode.*LF 1: 2 middle"):
            _preflight(tmp_path, [_chapters(2), _chapters(2)])

    def test_retry_that_clears_the_floor_lets_the_encode_start(self, tmp_path):
        assert _preflight(tmp_path, [_chapters(2), _chapters(4)]).call_count == 2

    def test_lf_without_gps_is_not_gated(self, tmp_path):
        assert _preflight(tmp_path, [], gpx=False).call_count == 0

    def test_runs_before_the_encode_in_phase_bc(self):
        src = inspect.getsource(phase_bc_day_first.PhaseBCDayFirst.run)
        assert "assert_chapter_floor_before_encode(" in src
        assert src.index("assert_chapter_floor_before_encode(") < src.index("run_all_days(")
