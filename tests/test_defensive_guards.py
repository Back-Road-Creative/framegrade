"""Defensive guards bundle.

Five small independent guards, each with a focused red->green test:

- cap GMS_OST_ENCODE_TIMEOUT_SEC at 86400 (no unbounded int from env).
- check_metadata_quality coerces a present-but-None description to ""
       instead of raising AttributeError and aborting the whole gate.
- weather._classify guards its divisor against equal thresholds set
       after construction (the constructor guard can't see mutated attrs).
- apply_merge persists entry.publish_at into the journey-YAML videos dict.
- the hardcoded font-path constants contain no ffmpeg-special chars,
       validated at module load (robust-by-construction).
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# OST encode timeout is capped (no unbounded env value)
# ---------------------------------------------------------------------------


def test_ost_encode_timeout_is_capped(monkeypatch):
    from gomoveshift.pipeline import ost_applier

    monkeypatch.setenv("GMS_OST_ENCODE_TIMEOUT_SEC", "999999999")
    assert ost_applier._ost_encode_timeout_sec(817.0) == 86400


# ---------------------------------------------------------------------------
# None description does not abort the metadata-quality gate
# ---------------------------------------------------------------------------


def test_metadata_quality_handles_none_description():
    from gomoveshift.pipeline.validation.content_gates import check_metadata_quality

    result = check_metadata_quality({"description": None})
    assert result.gate == "metadata_quality"


# ---------------------------------------------------------------------------
# weather._classify survives equal thresholds set after construction
# ---------------------------------------------------------------------------


def test_weather_classify_equal_thresholds_no_zero_division():
    from gomoveshift.pipeline.weather import WeatherDetector

    det = WeatherDetector()
    # Mutate thresholds post-construction so the constructor guard can't help.
    det.rainy_threshold = 500.0
    det.sunny_threshold = 500.0
    # Midpoint lands in the overcast branch, which divides by range_half == 0.
    condition, confidence = det._classify(500.0)
    assert 0.0 <= confidence <= 1.0


# ---------------------------------------------------------------------------
# apply_merge persists publish_at into the videos dict
# ---------------------------------------------------------------------------


def test_apply_merge_persists_publish_at():
    from pathlib import Path

    from gomoveshift.pipeline.journal_publisher import MergePlan, VideoEntry, apply_merge
    from gomoveshift.pipeline.site_sync.videos_yaml import VideoLibrary

    entry = VideoEntry(
        lf_index=1,
        youtube_id="VID1",
        title="Some Title",
        duration="12:34",
        day_date="2025-09-03",
        publish_at="2025-09-03T14:00:00Z",
    )
    plan = MergePlan(inserts=[entry], skips=[], errors=[])
    doc = {"days": [{"day_number": 1, "date": "2025-09-03", "videos": []}]}

    apply_merge(doc, plan, library=VideoLibrary.for_site(Path("/nonexistent"), writable=False))

    video = doc["days"][0]["videos"][0]
    assert video["publish_at"] == "2025-09-03T14:00:00Z"


# ---------------------------------------------------------------------------
# font-path validation rejects ffmpeg-special chars (robust-by-construction)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["/fonts/a:b.ttf", "/fonts/a'b.ttf", "/fonts/a\\b.ttf"])
def test_validate_font_path_rejects_special_chars(bad):
    from gomoveshift.pipeline.ffmpeg_filters import _validate_font_path

    with pytest.raises(ValueError):
        _validate_font_path(bad)


def test_validate_font_path_accepts_clean_path():
    from gomoveshift.pipeline.ffmpeg_filters import FONT_PATH, _validate_font_path

    # The canonical constant must pass (it is loaded at import time).
    assert _validate_font_path(FONT_PATH) == FONT_PATH
