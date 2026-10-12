"""``analysis.json`` must not advertise fields nothing can ever fill.

Two keys were emitted with a permanently dead writer:

* ``locations[].bbox`` — only ever set from ``OCREngine.process_frame(bbox=…)``,
  which the sole live caller (``AnalysisPipeline``) never passes; the one method
  that would have supplied one (``process_roi``) had no callers at all.
* ``processing_stats.claude_calls`` — initialised to 0 and never incremented,
  because the Claude OCR fallback is disabled in ``process_frame``.

Both are removed. A reader of an old artifact still has to load cleanly, so the
legacy-payload case is pinned too.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gomoveshift.analysis import pipeline as pipeline_mod
from gomoveshift.analysis.ocr import OCREngine
from gomoveshift.analysis.pipeline import AnalysisPipeline
from gomoveshift.cli.handlers import analysis as analysis_handler
from gomoveshift.config import Config
from gomoveshift.models import ConfidenceLevel, DetectedLocation, LocationType, VideoAnalysis


def _loc(timestamp: float = 12.0) -> DetectedLocation:
    return DetectedLocation(
        timestamp=timestamp,
        text="WELCOME TO DUBLIN",
        location_name="Dublin",
        location_type=LocationType.TOWN_ENTRY,
        confidence=ConfidenceLevel.HIGH,
        confidence_score=0.91,
        frame_path="/tmp/frame_000042.jpg",
    )


class TestLocationPayload:
    def test_emitted_location_has_no_bbox_key(self):
        payload = _loc().to_dict()

        assert "bbox" not in payload

    def test_rest_of_the_location_record_is_unchanged(self):
        payload = _loc().to_dict()

        assert payload == {
            "timestamp": 12.0,
            "timestamp_formatted": "0:00:12",
            "text": "WELCOME TO DUBLIN",
            "location_name": "Dublin",
            "location_type": "town_entry",
            "confidence": "high",
            "confidence_score": 0.91,
            "frame_path": "/tmp/frame_000042.jpg",
            "source": "easyocr",
        }

    def test_source_defaults_to_the_live_engine(self):
        """The dataclass default names the engine actually in use (EasyOCR),
        not the retired PaddleOCR one — the live caller (OCREngine) always
        passes ``source`` explicitly, so this default is only ever hit by
        callers that omit it, and it should describe the current pipeline."""
        loc = DetectedLocation(
            timestamp=1.0,
            text="x",
            location_name="X",
            location_type=LocationType.UNKNOWN,
            confidence=ConfidenceLevel.LOW,
            confidence_score=0.1,
        )

        assert loc.source == "easyocr"

    def test_legacy_analysis_json_missing_source_key_still_loads_as_paddleocr(self, tmp_path: Path):
        """Artifacts written before the ``source`` field existed ran PaddleOCR;
        the loader must keep attributing them correctly rather than relabel
        history to the current engine."""
        legacy_location = {k: v for k, v in _loc().to_dict().items() if k != "source"}
        legacy = {
            "video_path": "/drive.mp4",
            "duration": 60.0,
            "fps": 30.0,
            "width": 1920,
            "height": 1080,
            "locations": [legacy_location],
            "dead_spots": [],
            "suggested_cuts": [],
            "processing_stats": {},
        }
        path = tmp_path / "presource.analysis.json"
        path.write_text(json.dumps(legacy))

        loaded = VideoAnalysis.load(path)

        assert loaded.locations[0].source == "paddleocr"

    def test_bbox_cannot_be_constructed(self):
        """Robust by construction: no caller can reintroduce the dead field."""
        with pytest.raises(TypeError):
            DetectedLocation(
                timestamp=1.0,
                text="x",
                location_name="X",
                location_type=LocationType.UNKNOWN,
                confidence=ConfidenceLevel.LOW,
                confidence_score=0.1,
                bbox=(1, 2, 3, 4),
            )

    def test_legacy_analysis_json_with_bbox_still_loads(self, tmp_path: Path):
        """An artifact written before the removal must not break the reader."""
        legacy = {
            "video_path": "/drive.mp4",
            "duration": 60.0,
            "fps": 30.0,
            "width": 1920,
            "height": 1080,
            "locations": [dict(_loc().to_dict(), bbox=[10, 20, 30, 40])],
            "dead_spots": [],
            "suggested_cuts": [],
            "processing_stats": {"frames_processed": 120, "claude_calls": 0},
        }
        path = tmp_path / "old.analysis.json"
        path.write_text(json.dumps(legacy))

        loaded = VideoAnalysis.load(path)

        assert len(loaded.locations) == 1
        assert loaded.locations[0].location_name == "Dublin"
        assert not hasattr(loaded.locations[0], "bbox")

    def test_process_roi_is_gone(self):
        """The only method that would have supplied a bbox had zero callers."""
        assert not hasattr(OCREngine, "process_roi")


class TestProcessingStats:
    def _run_analyze(self, tmp_path: Path) -> VideoAnalysis:
        config = Config()
        config.paths.temp_folder = str(tmp_path / "temp")

        cap = MagicMock()
        caps = {
            5: 30.0,  # CAP_PROP_FPS
            7: 90.0,  # CAP_PROP_FRAME_COUNT
            3: 1920.0,  # CAP_PROP_FRAME_WIDTH
            4: 1080.0,  # CAP_PROP_FRAME_HEIGHT
        }
        cap.get.side_effect = lambda prop: caps[prop]

        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        frames = [(0, 0.0, frame), (15, 0.5, frame), (30, 1.0, frame)]

        with (
            patch.object(pipeline_mod.cv2, "VideoCapture", return_value=cap),
            patch.object(pipeline_mod, "extract_frames_at_intervals", return_value=iter(frames)),
            patch.object(pipeline_mod.MotionAnalyzer, "analyze", return_value=[]),
            patch.object(OCREngine, "process_frame", return_value=_loc()),
        ):
            return AnalysisPipeline(config).analyze(Path("/drive.mp4"), save_frames=False)

    def test_stats_have_no_claude_calls(self, tmp_path: Path):
        stats = self._run_analyze(tmp_path).processing_stats

        assert "claude_calls" not in stats

    def test_remaining_counters_still_report(self, tmp_path: Path):
        stats = self._run_analyze(tmp_path).processing_stats

        assert stats == {
            "frames_processed": 3,
            "locations_detected": 1,
            "dead_spots_detected": 0,
        }


class TestAnalyzeCommandSummary:
    def test_summary_drops_the_claude_line(self, tmp_path: Path, capsys):
        video = tmp_path / "drive.mp4"
        video.write_bytes(b"\x00")
        args = SimpleNamespace(video=video, output=tmp_path / "out.json", no_frames=True)

        fake = MagicMock()
        fake.processing_stats = {
            "frames_processed": 100,
            "locations_detected": 0,
            "dead_spots_detected": 0,
        }
        fake.locations = []
        fake.dead_spots = []
        fake.suggested_cuts = []

        with patch.object(analysis_handler, "AnalysisPipeline") as mock_pipe:
            mock_pipe.return_value.analyze.return_value = fake
            rc = analysis_handler.cmd_analyze(args, MagicMock())

        out = capsys.readouterr().out
        assert rc == 0
        assert "Claude" not in out
        assert "100 frames processed" in out
        assert "Locations found: 0" in out
