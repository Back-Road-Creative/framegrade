"""``config.drive.weather_chunk_*`` must actually reach ``WeatherDetector``.

The degraded-brightness log in ``gomoveshift/pipeline/weather.py`` tells the
operator to "raise ``config.drive.weather_chunk_max_sec`` if this repeats".
That remediation is only true if the construction site that produces the
message reads the field. It did not: every production ``WeatherDetector(...)``
call took the constructor defaults, so an operator who edited the config saw
no change at all.

The site that emits the message is the social stage (the only production
caller of ``detect_segment``), so that is the site these tests pin.
``check_weather_decode_path`` builds a detector too, but only ever asks it
for a cuvid decoder mapping — the chunk knobs cannot affect it, which the
last test here nails down.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.config import Config
from gomoveshift.pipeline.preflight import check_weather_decode_path
from gomoveshift.pipeline.stages.social import SocialGenerationStage


def _ctx(tmp_path: Path, config: Config) -> SimpleNamespace:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"\x00")
    return SimpleNamespace(
        config=config,
        paths=SimpleNamespace(project_dir=tmp_path),
        shorts_data={"segments": [{"start_time": 0.0, "end_time": 30.0}]},
        current_video=video,
        recording_date=None,
        metadata=None,
        project_name="episode_001",
        num_shorts=1,
        location="Test Drive",
        social_posts=None,
    )


def _detector_kwargs(tmp_path: Path, config: Config) -> dict:
    """Run the social stage and return the kwargs it built the detector with."""
    captured: dict = {}

    def _factory(**kwargs):
        captured.update(kwargs)
        det = MagicMock()
        det.detect_segment.return_value = None
        return det

    post = MagicMock()
    post.to_dict.return_value = {"platform": "instagram", "caption": "x"}
    generator = MagicMock()
    generator.generate_all.return_value = [post]

    with (
        patch("gomoveshift.pipeline.weather.WeatherDetector", _factory),
        patch("gomoveshift.pipeline.orchestrator.SocialGenerator", return_value=generator),
    ):
        result = SocialGenerationStage().run(_ctx(tmp_path, config))

    assert result.success, f"social stage failed: {result.error}"
    assert captured or "WeatherDetector was never constructed"
    return captured


class TestSocialStageThreadsChunkConfig:
    def test_non_default_chunk_max_sec_reaches_detector(self, tmp_path):
        config = Config()
        config.drive.weather_chunk_max_sec = 45.0
        kwargs = _detector_kwargs(tmp_path, config)
        assert kwargs.get("chunk_max_sec") == 45.0, (
            "config.drive.weather_chunk_max_sec did not reach WeatherDetector — "
            "the degraded-brightness remediation message is a dead end"
        )

    def test_non_default_chunk_timeout_cap_reaches_detector(self, tmp_path):
        config = Config()
        config.drive.weather_chunk_timeout_cap_sec = 600.0
        kwargs = _detector_kwargs(tmp_path, config)
        assert kwargs.get("chunk_timeout_cap_sec") == 600.0

    def test_default_config_preserves_current_behaviour(self, tmp_path):
        config = Config()
        kwargs = _detector_kwargs(tmp_path, config)
        assert kwargs.get("chunk_max_sec") == 120.0
        assert kwargs.get("chunk_timeout_cap_sec") == 180.0

    def test_weather_results_still_written(self, tmp_path):
        _detector_kwargs(tmp_path, Config())
        wf = tmp_path / "shorts" / "weather_results.json"
        assert wf.exists()
        assert json.loads(wf.read_text()) == []


class TestPreflightDetectorIsProbeOnly:
    """Pins why ``preflight.py`` deliberately keeps the constructor defaults.

    ``check_weather_decode_path`` uses its detector purely to map a source
    codec to a cuvid decoder; it never measures brightness, so the chunk
    knobs are inert there. Threading them would mean widening
    ``check_weather_decode_path`` *and* ``run_preflight`` *and* their caller
    in ``orchestrator_preflight.py`` for zero behavioural effect.
    """

    def test_detector_is_only_asked_for_a_cuvid_decoder(self):
        detector = MagicMock()
        detector._cuvid_decoder_for.return_value = None
        with patch("gomoveshift.pipeline.weather.WeatherDetector", return_value=detector):
            check_weather_decode_path([Path("/fake.mp4")], "auto")

        detector.detect_segment.assert_not_called()
        detector.detect_all_segments.assert_not_called()
        detector._get_avg_brightness_ffmpeg.assert_not_called()
        detector._cuvid_decoder_for.assert_called_once()


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
