"""Chapter-density gate skips GPS-less videos.

Robust-by-construction: when metadata carries ``has_gps=False``, the
chapter-density branch of ``check_metadata_quality`` is unsatisfiable
(GPS-less full_drive has no source data for chapter generation). The
gate must skip both the "no chapters for {minutes}min" and the
"only N intermediate chapter(s)" checks for those videos.

Tests cover these contract corners:

  has_gps=True  + no chapters + duration > 600s  -> FAIL (preserved)
  has_gps=False + no chapters + duration > 600s  -> PASS (closes bug)
  has_gps=True  + 2 chapters  + 30min            -> FAIL "only 2 intermediate"
  has_gps absent (legacy)                        -> FAIL (preserved)
"""

from gomoveshift.pipeline.validation.content_gates import check_metadata_quality


def _base_metadata(**overrides):
    """Build a metadata dict that passes every other check by default."""
    meta = {
        "title": "Killarney to Dingle via N86 - POV Driving Kerry, Ireland 4K",
        "description": (
            "POV drive through Killarney to Dingle via N86, Kerry "
            "on the wild Atlantic coast.\n\n"
            "\U0001f4cd Location: Killarney to Dingle via N86, Kerry\n"
            "\U0001f3ac Final cut: 32m 10s"
        ),
        "location": "Killarney to Dingle via N86, Kerry",
        "tags": ["Kerry", "Ireland", "POV drive"],
    }
    meta.update(overrides)
    return meta


class TestChapterGateRespectsHasGps:
    def test_has_gps_true_no_chapters_long_video_fails(self):
        """Preserve existing behavior: GPS-bearing video with no chapters fails."""
        meta = _base_metadata(
            has_gps=True,
            chapters=[],
            video_duration=1800.0,  # 30 min
        )
        result = check_metadata_quality(meta)
        assert not result.passed
        assert "no chapters" in result.message

    def test_has_gps_false_no_chapters_long_video_passes(self):
        """The bug fix: GPS-less video should NOT trigger the chapter gate."""
        meta = _base_metadata(
            has_gps=False,
            chapters=[],
            video_duration=1800.0,  # 30 min
        )
        result = check_metadata_quality(meta)
        assert result.passed, (
            f"GPS-less long video should pass metadata_quality, got: {result.message}"
        )

    def test_has_gps_true_low_density_fails_with_intermediate_message(self):
        """Preserve existing behavior: low chapter density still fails."""
        meta = _base_metadata(
            has_gps=True,
            chapters=[
                {"timestamp": "0:00", "title": "Start - Killarney"},
                {"timestamp": "15:00", "title": "Entering Anascaul, County Kerry"},
                {"timestamp": "30:00", "title": "Finish - Dingle"},
            ],
            video_duration=1800.0,  # 30 min -> need 2+ intermediate; we have 1
        )
        result = check_metadata_quality(meta)
        # Only 1 intermediate; min_expected = max(1, 30/15) = 2
        assert not result.passed
        assert "intermediate" in result.message

    def test_has_gps_false_low_density_passes(self):
        """GPS-less video skips the intermediate-density check as well."""
        meta = _base_metadata(
            has_gps=False,
            chapters=[
                {"timestamp": "0:00", "title": "Start"},
                {"timestamp": "30:00", "title": "Finish"},
            ],
            video_duration=1800.0,
        )
        result = check_metadata_quality(meta)
        assert result.passed, f"GPS-less low-density should pass, got: {result.message}"

    def test_has_gps_absent_legacy_no_chapters_fails(self):
        """Legacy metadata without has_gps key must preserve existing strict behavior."""
        meta = _base_metadata(
            chapters=[],
            video_duration=1800.0,
        )
        assert "has_gps" not in meta
        result = check_metadata_quality(meta)
        assert not result.passed
        assert "no chapters" in result.message
