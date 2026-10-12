"""Tests for ScenicScorer piecewise time mapping and ExtractionResult.load()."""

import json
from pathlib import Path

import pytest

from gomoveshift.analysis.shorts_extractor import (
    ExtractionResult,
    ExtractedShort,
    GPSData,
    GPSTrackpoint,
    ScenicScorer,
    ShortsExtractor,
)


# ── Piecewise time mapping ────────────────────────────────────────────


class TestPiecewiseTimeMapping:
    """_map_from_edited_time must correctly invert stop removal."""

    def test_no_keep_segments_is_identity(self):
        result = ShortsExtractor._map_from_edited_time(100.0, None)
        assert result == 100.0

    def test_empty_keep_segments_is_identity(self):
        result = ShortsExtractor._map_from_edited_time(100.0, [])
        assert result == 100.0

    def test_single_segment_no_stops(self):
        # One continuous segment from 0-3600
        keep = [(0.0, 3600.0)]
        assert ShortsExtractor._map_from_edited_time(100.0, keep) == 100.0
        assert ShortsExtractor._map_from_edited_time(1800.0, keep) == 1800.0

    def test_two_segments_with_gap(self):
        # Segments: [0, 100) and [200, 500)
        # Gap at 100-200 (stop removed)
        # Edited video: 0-100 maps to original 0-100
        #               100-400 maps to original 200-500
        keep = [(0.0, 100.0), (200.0, 500.0)]

        # In first segment
        assert ShortsExtractor._map_from_edited_time(50.0, keep) == 50.0

        # At boundary
        assert ShortsExtractor._map_from_edited_time(100.0, keep) == 200.0

        # In second segment
        assert ShortsExtractor._map_from_edited_time(150.0, keep) == 250.0

        # End of second segment
        assert ShortsExtractor._map_from_edited_time(400.0, keep) == 500.0

    def test_three_segments_multiple_stops(self):
        # Original: [0,100], [200,300], [400,600]
        # Stops at 100-200, 300-400
        # Edited: 0-100, 100-200, 200-400
        keep = [(0.0, 100.0), (200.0, 300.0), (400.0, 600.0)]

        # First segment
        assert ShortsExtractor._map_from_edited_time(50.0, keep) == 50.0

        # Second segment (edited 100-200 = original 200-300)
        assert ShortsExtractor._map_from_edited_time(150.0, keep) == 250.0

        # Third segment (edited 200-400 = original 400-600)
        assert ShortsExtractor._map_from_edited_time(300.0, keep) == 500.0

    def test_past_all_segments_returns_last_end(self):
        keep = [(0.0, 100.0), (200.0, 300.0)]
        # Total edited duration = 200. Asking for 250 (past end) returns last segment end
        assert ShortsExtractor._map_from_edited_time(250.0, keep) == 300.0


class TestVideoToGpsTimePiecewise:
    """_video_to_gps_time must use piecewise mapping when keep_segments set."""

    def test_with_keep_segments(self):
        extractor = ShortsExtractor()
        gps_data = GPSData(
            trackpoints=[
                GPSTrackpoint(lat=51.0, lon=-10.0, elevation=100, timestamp=1000.0),
                GPSTrackpoint(lat=51.1, lon=-10.1, elevation=200, timestamp=4600.0),
            ]
        )
        extractor.set_gps_data(gps_data, gps_offset=1000.0)

        # Set time mapping: 2 segments, original 3600s, gps 3600s
        keep_segments = [(0.0, 1800.0), (2000.0, 3600.0)]
        extractor.set_time_mapping(keep_segments, gps_duration=3600.0, original_duration=3600.0)

        # Edited time 0 = original 0 = gps 0 + offset 1000 = 1000
        gps_t = extractor._video_to_gps_time(0.0)
        assert gps_t == pytest.approx(1000.0, abs=1)

        # Edited time 1800 = original 2000 = gps 2000 + offset 1000 = 3000
        gps_t = extractor._video_to_gps_time(1800.0)
        assert gps_t == pytest.approx(3000.0, abs=1)

    def test_without_keep_segments_uses_linear_offset(self):
        extractor = ShortsExtractor()
        gps_data = GPSData(
            trackpoints=[
                GPSTrackpoint(lat=51.0, lon=-10.0, elevation=100, timestamp=1000.0),
            ]
        )
        extractor.set_gps_data(gps_data, gps_offset=1000.0)

        # No set_time_mapping called
        gps_t = extractor._video_to_gps_time(500.0)
        assert gps_t == 1500.0  # simple linear offset


# ── POI proximity scoring ─────────────────────────────────────────────


class TestPOIProximityScoring:
    def test_poi_within_segment_adds_bonus(self):
        gps_data = GPSData(
            trackpoints=[
                GPSTrackpoint(lat=51.0, lon=-10.0, elevation=100, timestamp=0.0),
                GPSTrackpoint(lat=51.0, lon=-10.0, elevation=100, timestamp=10.0),
                GPSTrackpoint(lat=51.0, lon=-10.0, elevation=100, timestamp=20.0),
                GPSTrackpoint(lat=51.0, lon=-10.0, elevation=100, timestamp=30.0),
                GPSTrackpoint(lat=51.0, lon=-10.0, elevation=100, timestamp=40.0),
            ]
        )
        scorer = ScenicScorer(gps_data)

        # Score without POIs
        score_no_poi = scorer.score_segment(0.0, 40.0)

        # Score with POI in range
        scorer.set_poi_timestamps([20.0])
        score_with_poi = scorer.score_segment(0.0, 40.0)

        assert score_with_poi > score_no_poi, "POI proximity should increase score"

    def test_poi_outside_segment_no_bonus(self):
        gps_data = GPSData(
            trackpoints=[
                GPSTrackpoint(lat=51.0, lon=-10.0, elevation=100, timestamp=0.0),
                GPSTrackpoint(lat=51.0, lon=-10.0, elevation=100, timestamp=40.0),
            ]
        )
        scorer = ScenicScorer(gps_data)

        # POI at timestamp 100 — outside segment 0-40
        scorer.set_poi_timestamps([100.0])
        score = scorer.score_segment(0.0, 40.0)

        # Remove POIs and rescore
        scorer._poi_timestamps = []
        score_no_poi = scorer.score_segment(0.0, 40.0)

        assert score == score_no_poi


# ── ExtractionResult.load() ──────────────────────────────────────────


class TestExtractionResultLoad:
    def test_roundtrip_save_load(self, tmp_path):
        shorts = [
            ExtractedShort(
                index=1,
                rank=1,
                start_time=100.0,
                end_time=140.0,
                duration=40.0,
                avg_motion_energy=0.05,
                peak_motion_energy=0.12,
                source_video="/fake/video.mp4",
            ),
            ExtractedShort(
                index=2,
                rank=2,
                start_time=500.0,
                end_time=535.0,
                duration=35.0,
                avg_motion_energy=0.04,
                peak_motion_energy=0.09,
                source_video="/fake/video.mp4",
            ),
        ]
        result = ExtractionResult(
            source_video="/fake/video.mp4",
            source_duration=3600.0,
            num_shorts_requested=5,
            num_shorts_found=2,
            shorts=shorts,
            dead_spots_avoided=3,
            processing_time=12.5,
        )

        path = tmp_path / "analysis.json"
        result.save(path)

        loaded = ExtractionResult.load(path)
        assert loaded.source_video == "/fake/video.mp4"
        assert loaded.source_duration == 3600.0
        assert loaded.num_shorts_found == 2
        assert len(loaded.shorts) == 2
        assert loaded.shorts[0].start_time == 100.0
        assert loaded.shorts[1].rank == 2
        assert loaded.dead_spots_avoided == 3

    def test_load_preserves_all_fields(self, tmp_path):
        short = ExtractedShort(
            index=1,
            rank=1,
            start_time=10.0,
            end_time=45.0,
            duration=35.0,
            avg_motion_energy=0.05,
            peak_motion_energy=0.10,
            source_video="/fake/v.mp4",
            output_path="/fake/short_01.mp4",
            thumbnail_path="/fake/thumb_01.jpg",
            suggested_caption="Nice view",
        )
        result = ExtractionResult(
            source_video="/fake/v.mp4",
            source_duration=1800.0,
            num_shorts_requested=3,
            num_shorts_found=1,
            shorts=[short],
            dead_spots_avoided=0,
            processing_time=5.0,
        )

        path = tmp_path / "test.json"
        result.save(path)
        loaded = ExtractionResult.load(path)

        s = loaded.shorts[0]
        assert s.output_path == "/fake/short_01.mp4"
        assert s.thumbnail_path == "/fake/thumb_01.jpg"
        assert s.suggested_caption == "Nice view"


class TestScoringWeightsSum:
    """Default weights must sum to 1.0."""

    def test_default_weights_sum_to_one(self):
        from gomoveshift.analysis.shorts_extractor import ScoringWeights

        w = ScoringWeights()
        total = (
            w.motion_energy
            + w.peak_motion
            + w.duration_match
            + w.position
            + w.spread
            + w.visual_interest
            + w.scenic_location
        )
        assert abs(total - 1.0) < 0.01, f"Default weights sum to {total}, expected 1.0"
