"""Unit tests for the per-segment adaptive color filtergraph builder."""

import re
from pathlib import Path

import pytest

from gomoveshift.pipeline.color_analysis import SegmentStats, merge_adjacent_segments
from gomoveshift.pipeline.color_correction import CorrectionParams, _average_params
from gomoveshift.pipeline.color_filters import (
    _build_adaptive_filtergraph,
    _build_single_filter,
    _params_all_equal,
    build_filter_chain,
)


def _params(shadow_lift: float = 5.0, **overrides) -> CorrectionParams:
    base = dict(
        shadow_lift=shadow_lift,
        highlight_pull=5.0,
        saturation_mult=1.05,
        contrast_curve=0.0,
        white_balance_r=0.0,
        white_balance_b=0.0,
        gamma=1.02,
        denoise_strength=0.0,
        sharpen_amount=0.0,
    )
    base.update(overrides)
    return CorrectionParams(**base)


def _lut_paths(n: int, tmp: Path) -> list[Path]:
    return [tmp / f"lut_{i:02d}.cube" for i in range(n)]


class TestBuilderShape:
    def test_three_segments_split_xfades_branches_and_offsets(self, tmp_path: Path) -> None:
        fg = _build_adaptive_filtergraph(
            _lut_paths(3, tmp_path), segment_duration=30.0, branch_filter="lut3d"
        )
        assert "split=3" in fg
        assert fg.count("lut3d=") == 3
        assert fg.count("xfade=") == 2  # N-1 transitions
        assert "duration=1" in fg  # 1.0s xfade (at-JND)
        # Offset T_n = n * segment_duration - xfade/2 → 29.5, 59.5
        assert "offset=29.5" in fg
        assert "offset=59.5" in fg

    def test_five_segments_chains_four_xfades(self, tmp_path: Path) -> None:
        fg = _build_adaptive_filtergraph(
            _lut_paths(5, tmp_path), segment_duration=30.0, branch_filter="lut3d"
        )
        assert "split=5" in fg
        assert fg.count("lut3d=") == 5
        assert fg.count("xfade=") == 4

    def test_lut_paths_appear_in_filtergraph(self, tmp_path: Path) -> None:
        paths = _lut_paths(2, tmp_path)
        fg = _build_adaptive_filtergraph(paths, segment_duration=30.0, branch_filter="lut3d")
        for p in paths:
            assert str(p) in fg

    def test_branches_trimmed_and_pts_reset(self, tmp_path: Path) -> None:
        """xfade restarts its 2nd input at the offset, so every branch must be
        trimmed to its span and PTS-reset or the seam REWINDS the content
        (verified: 6s source -> 8.5s output replaying from t=0)."""
        fg = _build_adaptive_filtergraph(
            _lut_paths(3, tmp_path), segment_duration=30.0, branch_filter="lut3d"
        )
        assert "trim=end=30.5,setpts=PTS-STARTPTS" in fg
        assert "trim=start=29.5:end=60.5,setpts=PTS-STARTPTS" in fg
        assert "trim=start=59.5,setpts=PTS-STARTPTS" in fg

    def test_graph_embeds_in_plain_vf_chain(self, tmp_path: Path) -> None:
        """Input and final output are unlabeled so the graph drops into -vf
        and composes with downstream scale/fade/drawtext via a comma."""
        fg = _build_adaptive_filtergraph(
            _lut_paths(2, tmp_path), segment_duration=30.0, branch_filter="lut3d"
        )
        assert not fg.startswith("[0:v]")
        assert not fg.rstrip().endswith("]")

    def test_variable_span_durations_set_boundaries(self, tmp_path: Path) -> None:
        """Merged spans have variable lengths; offsets follow cumulative sums."""
        fg = _build_adaptive_filtergraph(
            _lut_paths(3, tmp_path),
            span_durations=[60.0, 30.0, 90.0],
            branch_filter="lut3d",
        )
        assert "offset=59.5" in fg
        assert "offset=89.5" in fg

    def test_spatial_tail_applied_once_after_stitch(self, tmp_path: Path) -> None:
        fg = _build_adaptive_filtergraph(
            _lut_paths(3, tmp_path),
            segment_duration=30.0,
            branch_filter="lut3d",
            spatial_tail="hqdn3d=2.0:2.0:1.5:1.5,unsharp=5:5:0.5:5:5:0",
        )
        # Spatial filters appear exactly once (not per-branch), so
        # temporal-state filters never restart across seams.
        assert fg.count("hqdn3d=") == 1
        assert fg.count("unsharp=") == 1
        assert fg.rfind("xfade=") < fg.rfind("hqdn3d=")


class TestFinalXfadeEOFClamp:
    """A day whose final segment is shorter than segment_duration still gets a
    nominal (count * segment_duration) last span. Uncorrected, the final xfade
    is centered on the nominal boundary and its window ends PAST real EOF —
    a truncated fade / encode error. The final fade must land inside the real
    footage (offset + duration <= total). Interior seams and exact-length
    final spans stay byte-identical.
    """

    @staticmethod
    def _offsets(fg: str) -> list[float]:
        return [float(m) for m in re.findall(r"offset=([\d.]+)", fg)]

    def test_partial_final_span_keeps_last_fade_inside_footage(self, tmp_path: Path) -> None:
        # Real last span is 0.3s (a tiny partial trailing segment). Nominal
        # geometry would place the last fade at offset 59.5 → ends at 60.5,
        # but real EOF is 60.3 → the fade would run past EOF.
        spans = [30.0, 30.0, 0.3]
        fg = _build_adaptive_filtergraph(
            _lut_paths(3, tmp_path), span_durations=spans, xfade_duration=1.0
        )
        total = sum(spans)
        assert self._offsets(fg)[-1] + 1.0 <= total + 1e-9

    def test_exact_length_final_span_byte_identical(self, tmp_path: Path) -> None:
        # An exact-multiple final segment leaves every boundary untouched.
        fg = _build_adaptive_filtergraph(
            _lut_paths(3, tmp_path), span_durations=[30.0, 30.0, 30.0], xfade_duration=1.0
        )
        assert "offset=29.5" in fg
        assert "offset=59.5" in fg
        assert "trim=start=59.5,setpts=PTS-STARTPTS" in fg

    def test_merge_clamps_final_span_to_real_duration(self) -> None:
        # day/tunnel/day → 3 runs, nominal durations [60, 60, 30]. When the
        # real media is only 132s, the final span is clamped to the real
        # remaining (12s) so sum(durations) == the true media length.
        day = SegmentStats(black_point=20, white_point=220, mid_tone=140, saturation=0.35)
        tunnel = SegmentStats(black_point=2, white_point=80, mid_tone=30, saturation=0.15)
        segs = [day] * 2 + [tunnel] * 2 + [day] * 1
        _, durations = merge_adjacent_segments(segs, segment_duration=30.0, total_duration=132.0)
        assert durations[:-1] == [60.0, 60.0]
        assert durations[-1] == pytest.approx(12.0)
        assert sum(durations) == pytest.approx(132.0)

    def test_merge_without_total_duration_is_nominal(self) -> None:
        day = SegmentStats(black_point=20, white_point=220, mid_tone=140, saturation=0.35)
        tunnel = SegmentStats(black_point=2, white_point=80, mid_tone=30, saturation=0.15)
        segs = [day] * 2 + [tunnel] * 2 + [day] * 1
        _, durations = merge_adjacent_segments(segs, segment_duration=30.0)
        assert durations == [60.0, 60.0, 30.0]


class TestIdentityCollapse:
    """When all N params are numerically equal, the builder must produce
    a filtergraph byte-identical to ``_build_single_filter(_average_params(...))``
    — the "no behavior change in the single-segment case" guarantee."""

    def test_collapse_when_all_params_equal(self) -> None:
        p = _params(shadow_lift=8.0, saturation_mult=1.1)
        adaptive = build_filter_chain([p, p, p], segment_duration=30.0)
        ref = _build_single_filter(_average_params([p, p, p]))
        assert adaptive == ref

    def test_collapse_within_epsilon(self) -> None:
        p1 = _params(shadow_lift=8.0)
        p2 = _params(shadow_lift=8.0 + 1e-7)
        adaptive = build_filter_chain([p1, p2], segment_duration=30.0)
        ref = _build_single_filter(_average_params([p1, p2]))
        assert adaptive == ref


class TestDispatch:
    def test_single_params_unchanged(self) -> None:
        p = _params(shadow_lift=5.0)
        assert build_filter_chain([p], segment_duration=0.0) == _build_single_filter(p)

    def test_empty_returns_null(self) -> None:
        assert build_filter_chain([], segment_duration=0.0) == "null"

    def test_differing_params_with_lut_paths_emit_adaptive(self, tmp_path: Path) -> None:
        p1, p2 = _params(shadow_lift=2.0), _params(shadow_lift=12.0)
        out = build_filter_chain(
            [p1, p2],
            segment_duration=30.0,
            lut_paths=[tmp_path / "a.cube", tmp_path / "b.cube"],
            branch_filter="lut3d",
        )
        assert "split=2" in out
        assert "xfade=" in out
        assert out.count("lut3d=") == 2

    def test_differing_params_without_lut_paths_falls_back_to_averaged(self) -> None:
        # Call sites that have not opted into LUT baking keep current behavior.
        p1, p2 = _params(shadow_lift=2.0), _params(shadow_lift=12.0)
        out = build_filter_chain([p1, p2], segment_duration=30.0)
        assert out == _build_single_filter(_average_params([p1, p2]))


class TestParamsEqualHelper:
    def test_detects_exact_equality(self) -> None:
        p = _params()
        assert _params_all_equal([p, p, p])

    def test_rejects_difference(self) -> None:
        assert not _params_all_equal([_params(shadow_lift=2.0), _params(shadow_lift=12.0)])

    def test_tolerates_epsilon(self) -> None:
        p1, p2 = _params(), _params(shadow_lift=_params().shadow_lift + 1e-7)
        assert _params_all_equal([p1, p2])


@pytest.mark.slow
def test_one_second_xfade_smooths_boundary(tmp_path: Path) -> None:
    """Real-encode contract for the wired adaptive graph, via a plain -vf
    chain exactly as single_pass embeds it: (a) duration is preserved and
    content stays time-aligned (regression for the un-trimmed xfade rewind),
    (b) the second span's LUT actually reaches the encode,
    (c) the 1.0s xfade keeps the per-frame in-fade YAVG step sub-JND.
    Skipped without ffmpeg."""
    import shutil
    import subprocess

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not available")

    # Constant mid-gray source: any luma change is the LUTs' doing.
    src = tmp_path / "src.mp4"
    r = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=gray:s=64x64:d=6:r=10,format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            str(src),
        ],
        capture_output=True,
        timeout=30,
    )
    if r.returncode != 0:
        pytest.skip(f"ffmpeg synthetic src failed: {r.stderr[-200:]!r}")

    # Span A: identity LUT. Span B: 0.5x darkening LUT (size=2).
    for p, scale in ((tmp_path / "a.cube", 1.0), (tmp_path / "b.cube", 0.5)):
        lines = ["LUT_3D_SIZE 2", "DOMAIN_MIN 0.0 0.0 0.0", "DOMAIN_MAX 1.0 1.0 1.0", ""]
        for r_ in (0.0, 1.0):
            for g_ in (0.0, 1.0):
                for b_ in (0.0, 1.0):
                    lines.append(f"{r_ * scale:.6f} {g_ * scale:.6f} {b_ * scale:.6f}")
        p.write_text("\n".join(lines) + "\n")

    fg = _build_adaptive_filtergraph(
        [tmp_path / "a.cube", tmp_path / "b.cube"],
        segment_duration=3.0,
        branch_filter="lut3d",
        xfade_duration=1.0,
    )
    out = tmp_path / "out.mp4"
    r = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(src),
            "-vf",
            f"{fg},scale=in_range=full:out_range=full",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            str(out),
        ],
        capture_output=True,
        timeout=60,
    )
    if r.returncode != 0:
        pytest.skip(f"ffmpeg adaptive filtergraph failed: {r.stderr[-300:]!r}")

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(out),
        ],
        capture_output=True,
        timeout=15,
        text=True,
    )
    # (a) No rewind/stretch: the un-trimmed graph produced 8.5s from 6s.
    assert abs(float(probe.stdout.strip()) - 6.0) < 0.3, probe.stdout

    def _yavg(t: float) -> float:
        r = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-ss",
                f"{t:.2f}",
                "-i",
                str(out),
                "-vframes",
                "1",
                "-vf",
                "signalstats,metadata=print:key=lavfi.signalstats.YAVG:file=-",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            timeout=15,
            text=True,
        )
        for line in (r.stdout + r.stderr).splitlines():
            if "YAVG" in line and "=" in line:
                return float(line.split("=")[-1])
        pytest.skip("YAVG not parseable")
        return 0.0

    # (b) Per-span grades reach the encode, time-aligned: identity before
    # the fade window, darkened (~0.5x) after it.
    assert _yavg(1.0) > 100.0
    assert _yavg(5.0) < 80.0
    # (c) LUT step is ~64 luma over a 1.0s fade at 10fps → ~6.4 luma/frame.
    delta = abs(_yavg(2.7) - _yavg(2.6))
    assert delta <= 8.0, f"in-fade per-frame delta {delta:.2f} > 8.0"
