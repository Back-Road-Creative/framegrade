"""Thumbnail frame picking scores sharpness + exposure, not raw brightness.

Brightest-wins picks blown skies and motion-blurred frames on driving footage.
These pin the replacement: ONE picker, used by the drive path and the legacy
``ThumbnailGenerationStage`` alike.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gomoveshift.pipeline.drive_phases.thumbnail_generator import (
    FrameQuality,
    generate_lf_thumbnail,
    probe_frame_quality,
    quality_score,
    rank_by_quality,
    select_best_timestamps,
)


class TestQualityScore:
    def test_sharp_well_exposed_beats_brighter_blown_and_blurry(self):
        """The regression the picker exists for: brightest != best.

        ``blown`` is the frame brightest-wins would have shipped — a glare
        frame with a blown sky and motion blur. ``sharp`` is dimmer and the
        one a human would pick.
        """
        blown = FrameQuality(brightness=200.0, sharpness=40.0, blown_fraction=0.30)
        sharp = FrameQuality(brightness=120.0, sharpness=600.0, blown_fraction=0.01)

        assert blown.brightness > sharp.brightness  # brightest-wins picks `blown`
        assert quality_score(sharp) > quality_score(blown)

    def test_each_term_moves_the_score_on_its_own(self):
        best = FrameQuality(128.0, 500.0, 0.0)
        # soft / glare / crushed / washed out — one term degraded at a time.
        for worse in (
            FrameQuality(128.0, 10.0, 0.0),
            FrameQuality(128.0, 500.0, 0.5),
            FrameQuality(20.0, 500.0, 0.0),
            FrameQuality(236.0, 500.0, 0.0),
        ):
            assert quality_score(best) > quality_score(worse)


class TestRanking:
    def test_select_best_timestamps_returns_time_sorted_top_n(self):
        table = {
            10.0: FrameQuality(250.0, 5.0, 0.9),
            20.0: FrameQuality(130.0, 700.0, 0.0),
            25.0: None,  # probe failed
            30.0: FrameQuality(120.0, 650.0, 0.0),
            40.0: FrameQuality(200.0, 20.0, 0.4),
        }
        picked = select_best_timestamps(
            "/fake.mp4", sorted(table), count=3, probe=lambda _v, t: table[t]
        )
        assert picked == [20.0, 30.0, 40.0]
        assert rank_by_quality(list(table.items()))[-1] == 25.0

    def test_select_best_timestamps_empty_when_every_probe_fails(self):
        picked = select_best_timestamps("/fake.mp4", [1.0, 2.0], probe=lambda _v, _t: None)
        assert picked == []


class TestDrivePathUsesQuality:
    def test_generate_lf_thumbnail_returns_the_sharp_frame_not_the_bright_one(
        self, tmp_path, monkeypatch
    ):
        import gomoveshift.pipeline.drive_phases.thumbnail_generator as tg
        import gomoveshift.pipeline.thumbnail_text_overlay as tto

        # Body pick for a 600s clip is [30.0, 300.0, 570.0]; frame 2 is sharp.
        quality = {
            30.0: FrameQuality(250.0, 8.0, 0.8),
            300.0: FrameQuality(120.0, 700.0, 0.0),
            570.0: FrameQuality(210.0, 30.0, 0.5),
        }
        monkeypatch.setattr(tg, "probe_frame_quality", lambda _v, ts: quality[ts])
        monkeypatch.setattr(
            tto,
            "generate_thumbnails_from_video",
            lambda **kw: {f"frame{i}": {"bold": f"/out/frame{i}_bold.jpg"} for i in range(1, 4)},
        )
        out = generate_lf_thumbnail(Path("/fake.mp4"), "Title", None, tmp_path, duration=600.0)
        assert out == Path("/out/frame2_bold.jpg")


class TestLegacyStageUsesSharedPicker:
    def test_stage_timestamps_come_from_the_shared_picker(self, tmp_path, monkeypatch):
        import gomoveshift.pipeline.drive_phases.thumbnail_generator as tg
        import gomoveshift.pipeline.orchestrator as orch
        from gomoveshift.pipeline.stages.thumbnails import ThumbnailGenerationStage

        seen: dict = {}

        def fake_select(video, candidates, count=3, **kw):
            seen["candidates"] = list(candidates)
            seen["count"] = count
            return [111.0, 222.0, 333.0]

        monkeypatch.setattr(tg, "select_best_timestamps", fake_select)

        def fake_generate(**kw):
            seen["timestamps"] = kw["frame_timestamps"]
            styled = Path(kw["output_dir"])
            for i in range(9):
                (styled / f"t{i}.jpg").write_bytes(b"\xff" * (11 * 1024))
            return {"frame1": {"bold": "x"}}

        monkeypatch.setattr(orch, "generate_thumbnails_from_video", fake_generate)

        ctx = MagicMock()
        ctx.paths.project_dir = tmp_path
        ctx.video_properties = None
        ctx.current_video = Path("/fake.mp4")

        result = ThumbnailGenerationStage().run(ctx)
        assert result.success, result.error
        assert seen["timestamps"] == [111, 222, 333]
        assert seen["count"] == 3
        assert len(seen["candidates"]) == 20


def _clip(tmp_path: Path, name: str, vf: str) -> Path:
    out = tmp_path / name
    argv = (
        "ffmpeg -y -loglevel error -f lavfi -i testsrc2=duration=3:size=640x360:rate=30 "
        f"-vf {vf} -c:v libx264 -pix_fmt yuv420p"
    ).split() + [str(out)]
    r = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    if r.returncode != 0 or not out.exists():
        pytest.skip(f"ffmpeg could not build {name}: {r.stderr[:200]}")
    return out


def test_sharpness_metric_separates_blurred_from_sharp_real_frames(tmp_path):
    """Empirical grounding: the metric must actually measure blur."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg unavailable")
    sharp = probe_frame_quality(str(_clip(tmp_path, "sharp.mp4", "null")), 1.0)
    blurred = probe_frame_quality(str(_clip(tmp_path, "blur.mp4", "boxblur=10:2")), 1.0)
    assert sharp is not None and blurred is not None
    assert sharp.sharpness > blurred.sharpness * 2
