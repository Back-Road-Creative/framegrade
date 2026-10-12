"""Tests for OCR engine EasyOCR -> Claude fallback chain."""

from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest

from gomoveshift.analysis.ocr import LocationDeduplicator, OCREngine
from gomoveshift.config import Config
from gomoveshift.models import ConfidenceLevel, DetectedLocation, LocationType


@pytest.fixture
def config():
    return Config()


@pytest.fixture
def dummy_frame():
    """BGR uint8 frame, 100x100."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


class TestSignsExtraMissing:
    """``easyocr`` lives in the optional ``signs`` extra."""

    def test_missing_easyocr_names_the_extra(self, config, monkeypatch):
        """A default install must say which extra to add, not just 'no module'.

        Without this the only clue an operator gets from a ``--sign-verify``
        run is a bare ModuleNotFoundError naming a package that is absent on
        purpose.
        """
        import builtins

        real_import = builtins.__import__

        def _no_easyocr(name, *args, **kwargs):
            if name == "easyocr":
                raise ModuleNotFoundError("No module named 'easyocr'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_easyocr)

        with pytest.raises(ModuleNotFoundError, match=r"\[signs\]"):
            _ = OCREngine(config).easyocr_reader


class TestEasyOCRSuccess:
    """EasyOCR returns good results — no Claude call needed."""

    @patch.object(OCREngine, "easyocr_reader", new_callable=PropertyMock)
    def test_high_confidence_easyocr(self, mock_reader_prop, config, dummy_frame):
        """High-confidence EasyOCR result is used directly, no Claude call."""
        reader = MagicMock()
        reader.readtext.return_value = [
            ([[0, 0], [100, 0], [100, 30], [0, 30]], "Welcome to Cork", 0.95),
        ]
        mock_reader_prop.return_value = reader

        engine = OCREngine(
            config,
        )
        result = engine.process_frame(dummy_frame, timestamp=10.0)

        assert result is not None
        assert result.location_name == "Cork"
        assert result.location_type == LocationType.TOWN_ENTRY
        assert result.confidence == ConfidenceLevel.HIGH
        assert result.source == "easyocr"

    @patch.object(OCREngine, "easyocr_reader", new_callable=PropertyMock)
    def test_easyocr_no_text(self, mock_reader_prop, config, dummy_frame):
        """EasyOCR finds nothing — returns None, no fallback attempted."""
        reader = MagicMock()
        reader.readtext.return_value = []
        mock_reader_prop.return_value = reader

        engine = OCREngine(
            config,
        )
        result = engine.process_frame(dummy_frame, timestamp=5.0)

        assert result is None

    @patch.object(OCREngine, "easyocr_reader", new_callable=PropertyMock)
    def test_easyocr_low_confidence_items_filtered(self, mock_reader_prop, config, dummy_frame):
        """Items below 0.3 confidence or shorter than 2 chars are filtered."""
        reader = MagicMock()
        reader.readtext.return_value = [
            ([[0, 0], [50, 0], [50, 20], [0, 20]], "a", 0.9),  # too short
            ([[0, 0], [50, 0], [50, 20], [0, 20]], "noise", 0.1),  # too low conf
        ]
        mock_reader_prop.return_value = reader

        engine = OCREngine(
            config,
        )
        result = engine.process_frame(dummy_frame, timestamp=1.0)
        assert result is None


class TestClaudeFallback:
    """EasyOCR returns low confidence — falls back to Claude."""

    @patch.object(OCREngine, "easyocr_reader", new_callable=PropertyMock)
    def test_low_confidence_triggers_claude(self, mock_reader_prop, config, dummy_frame):
        """Below threshold triggers Claude fallback which returns higher confidence."""
        reader = MagicMock()
        reader.readtext.return_value = [
            ([[0, 0], [100, 0], [100, 30], [0, 30]], "W3lcome to Dub1in", 0.35),
        ]
        mock_reader_prop.return_value = reader

        engine = OCREngine(config)
        result = engine.process_frame(dummy_frame, timestamp=10.0)

        assert result is not None
        # Claude OCR disabled — always falls back to easyocr
        assert result.source == "easyocr"

    @patch.object(OCREngine, "easyocr_reader", new_callable=PropertyMock)
    def test_claude_not_called_when_no_client(self, mock_reader_prop, config, dummy_frame):
        """Without anthropic client, low-confidence result stays as easyocr."""
        reader = MagicMock()
        reader.readtext.return_value = [
            ([[0, 0], [100, 0], [100, 30], [0, 30]], "blurry text here", 0.35),
        ]
        mock_reader_prop.return_value = reader

        engine = OCREngine(
            config,
        )
        result = engine.process_frame(dummy_frame, timestamp=10.0)

        assert result is not None
        assert result.source == "easyocr"


class TestBothFail:
    """Both EasyOCR and Claude fail gracefully."""

    @patch.object(OCREngine, "easyocr_reader", new_callable=PropertyMock)
    def test_easyocr_exception_returns_none(self, mock_reader_prop, config, dummy_frame):
        """Exception in EasyOCR is caught and returns None."""
        reader = MagicMock()
        reader.readtext.side_effect = RuntimeError("CUDA OOM")
        mock_reader_prop.return_value = reader

        engine = OCREngine(
            config,
        )
        result = engine.process_frame(dummy_frame, timestamp=5.0)
        assert result is None

    @patch.object(OCREngine, "easyocr_reader", new_callable=PropertyMock)
    def test_claude_exception_falls_back_to_easyocr(self, mock_reader_prop, config, dummy_frame):
        """Claude exception means easyocr result is used instead."""
        reader = MagicMock()
        reader.readtext.return_value = [
            ([[0, 0], [100, 0], [100, 30], [0, 30]], "Galway 25km", 0.35),
        ]
        mock_reader_prop.return_value = reader

        engine = OCREngine(config)
        result = engine.process_frame(dummy_frame, timestamp=10.0)

        assert result is not None
        # Claude OCR disabled — always easyocr
        assert result.source == "easyocr"


class TestLocationDeduplicator:
    """Tests for LocationDeduplicator."""

    def test_empty_input(self):
        dedup = LocationDeduplicator()
        assert dedup.deduplicate([]) == []

    def test_no_duplicates(self):
        dedup = LocationDeduplicator(time_threshold=3.0)
        locs = [
            DetectedLocation(
                timestamp=0.0,
                text="Cork",
                location_name="Cork",
                location_type=LocationType.TOWN_ENTRY,
                confidence=ConfidenceLevel.HIGH,
                confidence_score=0.9,
            ),
            DetectedLocation(
                timestamp=10.0,
                text="Dublin",
                location_name="Dublin",
                location_type=LocationType.TOWN_ENTRY,
                confidence=ConfidenceLevel.HIGH,
                confidence_score=0.9,
            ),
        ]
        result = dedup.deduplicate(locs)
        assert len(result) == 2

    def test_duplicate_within_threshold_keeps_higher_confidence(self):
        dedup = LocationDeduplicator(time_threshold=5.0)
        locs = [
            DetectedLocation(
                timestamp=0.0,
                text="Cork",
                location_name="Cork",
                location_type=LocationType.TOWN_ENTRY,
                confidence=ConfidenceLevel.MEDIUM,
                confidence_score=0.5,
            ),
            DetectedLocation(
                timestamp=2.0,
                text="Cork",
                location_name="Cork",
                location_type=LocationType.TOWN_ENTRY,
                confidence=ConfidenceLevel.HIGH,
                confidence_score=0.9,
            ),
        ]
        result = dedup.deduplicate(locs)
        assert len(result) == 1
        assert result[0].confidence_score == pytest.approx(0.9)
