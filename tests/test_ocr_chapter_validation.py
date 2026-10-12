"""Unvalidated OCR sign-text must not become MP4-embedded chapters.

Two layers of defence, each tested at its own boundary:

1. The OCR county pattern must not match ``co`` *inside* a word — e.g.
   ``"tesco express"`` used to parse as a ``COUNTY_BOUNDARY`` named
   ``"Express"`` and ship an ``"Express County"`` chapter.
2. The legacy assembler's chapter builder is the single chokepoint before the
   irreversible MP4 chapter embed. Business names / OCR garbage that slip
   through parsing must be dropped there via the shared chapter validator.
"""

from gomoveshift.analysis.ocr import OCREngine
from gomoveshift.assembly.assembler import VideoAssembler
from gomoveshift.config import Config
from gomoveshift.models import ConfidenceLevel, DetectedLocation, LocationType


def _loc(name: str, loc_type: LocationType, timestamp: float = 5.0) -> DetectedLocation:
    return DetectedLocation(
        timestamp=timestamp,
        text=name,
        location_name=name,
        location_type=loc_type,
        confidence=ConfidenceLevel.HIGH,
        confidence_score=0.9,
    )


class TestOCRCountyPattern:
    """``co`` must be its own token to parse as a county boundary."""

    def test_co_inside_word_is_not_a_county(self):
        engine = OCREngine(Config())
        name, loc_type = engine._parse_location("tesco express")
        assert loc_type != LocationType.COUNTY_BOUNDARY
        assert name != "Express"

    def test_real_county_abbreviation_still_parses(self):
        engine = OCREngine(Config())
        name, loc_type = engine._parse_location("co. clare")
        assert loc_type == LocationType.COUNTY_BOUNDARY
        assert name == "Clare"


class TestChapterEmbedValidation:
    """The assembler drops invalid placenames before they become chapters."""

    def test_business_name_is_dropped(self):
        assembler = VideoAssembler(Config())
        # "Store" is a whole-word tourism keyword; the shared validator matches
        # keywords on word boundaries, so a settlement that merely embeds one
        # (e.g. "Superstore") is not caught — a business naming it as a token is.
        chapters = assembler._create_chapters(
            [
                _loc("Tesco Store", LocationType.TOWN_ENTRY, timestamp=5.0),
                _loc("Clifden", LocationType.TOWN_ENTRY, timestamp=10.0),
            ]
        )
        titles = [c.title for c in chapters]
        assert "Entering Clifden" in titles
        assert not any("Tesco" in t for t in titles)

    def test_garbage_raw_ocr_never_embeds(self):
        assembler = VideoAssembler(Config())
        chapters = assembler._create_chapters(
            [_loc("Xhqz Random Garbage", LocationType.UNKNOWN, timestamp=7.0)]
        )
        titles = [c.title for c in chapters]
        assert titles == ["Start"]


class TestOCRCountyFormAndStopWords:
    """A county capture must not emit a road token or flip the sign's form."""

    def _chapter_titles(self, sign_text: str) -> list[str]:
        name, loc_type = OCREngine(Config())._parse_location(sign_text)
        chapters = VideoAssembler(Config())._create_chapters([_loc(name, loc_type)])
        return [c.title for c in chapters]

    def test_county_road_does_not_become_road_county(self):
        # "County Road 12" captured "Road" -> a bogus "Road County" chapter.
        name, loc_type = OCREngine(Config())._parse_location("county road 12")
        assert not (loc_type == LocationType.COUNTY_BOUNDARY and name == "Road")
        assert not any("Road County" in t for t in self._chapter_titles("county road 12"))

    def test_irish_county_form_preserved(self):
        # An Irish "County Kerry" sign must stay "County Kerry", never flip to
        # the US "Kerry County" template, and never double to "County Kerry County".
        name, loc_type = OCREngine(Config())._parse_location("county kerry")
        assert loc_type == LocationType.COUNTY_BOUNDARY
        titles = self._chapter_titles("county kerry")
        assert "County Kerry" in titles
        assert "Kerry County" not in titles
        assert "County Kerry County" not in titles

    def test_us_name_first_county_keeps_us_form(self):
        # Regression: a US "Kerry County" sign keeps the appended-County form.
        assert "Kerry County" in self._chapter_titles("kerry county")

    def test_welcome_greedy_capture_truncated_at_sign_suffix(self):
        # "Welcome to X …" must not swallow trailing sign text into the name.
        name, loc_type = OCREngine(Config())._parse_location(
            "welcome to clifden twinned with boston"
        )
        assert loc_type == LocationType.TOWN_ENTRY
        assert name == "Clifden"
