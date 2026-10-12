"""Tests for ContentValidator — every blocked phrase and pattern must be caught."""

import pytest

from gomoveshift.pipeline.content_validator import (
    ContentValidationError,
    ContentValidator,
)


@pytest.fixture
def validator():
    return ContentValidator()


class TestBlockedPhrases:
    """Every blocked phrase must be detected."""

    @pytest.mark.parametrize(
        "phrase",
        [
            "Turn up the volume for this one",
            "turn it up",
            "Crank up the volume",
            "crank it up",
            "volume up",
            "blast the speakers",
            "Wait for the view",
            "wait for it",
            "wait till you see",
            "You won't believe this",
            "you wont believe",
            "never disappoints",
            "never fails to impress",
            "mind-blowing scenery",
            "mind blowing drive",
            "jaw-dropping views",
            "jaw dropping road",
            "this is insane",
            "this is unreal",
            "is unreal",
            "absolutely insane",
            "blows my mind",
            "blow your mind",
            "life-changing drive",
            "life changing route",
            "once in a lifetime",
            "you have to see this",
            "you need to see this",
            "trust me on this",
            "i promise you",
            "guaranteed to amaze",
            "will change your life",
            "best road ever",
            "best drive ever",
            "best route ever",
            "most beautiful road in the world",
            "most scenic road anywhere",
        ],
    )
    def test_blocked_phrase_detected(self, validator, phrase):
        result = validator.validate(phrase)
        assert not result.valid, f"Should have blocked: '{phrase}'"
        assert len(result.violations) > 0

    @pytest.mark.parametrize(
        "phrase",
        [
            "Turn up the volume for this one 🔊",
            "TURN IT UP!",
            "Wait for the view... 👀",
            "You Won't Believe This Road",
        ],
    )
    def test_case_insensitive(self, validator, phrase):
        result = validator.validate(phrase)
        assert not result.valid, f"Case-insensitive check failed for: '{phrase}'"


class TestBlockedPatterns:
    """Regex patterns must catch broader unsafe content."""

    @pytest.mark.parametrize(
        "text",
        [
            "Turn up the music",
            "Turn it up loud",
            "Wait for the scenery",
            "Wait for this moment",
            "You won't believe what happens",
            "Most beautiful drive anywhere",
            "Most scenic route ever",
            "Most amazing road trip",
            "Best incredible journey",
            "Insane road through the mountains",
            "Unreal drive along the coast",
            "Crazy road in Ireland",
        ],
    )
    def test_pattern_detected(self, validator, text):
        result = validator.validate(text)
        assert not result.valid, f"Pattern should have blocked: '{text}'"


class TestBlocksSubjectiveRoadClaims:
    """Unverifiable claims about road character must be blocked."""

    @pytest.mark.parametrize(
        "phrase",
        [
            "Only Port Royal Locals Know This Route",
            "Only locals know this path",
            "Hidden Gem Route in Highlands",
            "Sunday Drive Through Burton",
            "Quiet Roads of Connemara",
            "Quiet road through the valley",
            "A quiet ride through the countryside",
            "Lost in Burton — Best Kind of Lost",
            "Turn the sound on 🔊",
            "Turn on the sound for this one",
        ],
    )
    def test_subjective_road_claims_blocked(self, validator, phrase):
        result = validator.validate(phrase)
        assert not result.valid, f"Should have blocked: '{phrase}'"


class TestBlocksSubjectivePatterns:
    """Regex patterns for subtler quality violations."""

    @pytest.mark.parametrize(
        "text",
        [
            "Guess the Location — Port Royal 🚗",
            "Guess Where We're Going in Burton",
            "Lost in Burton to Yemassee — Best Kind of Lost",
            "This Road Is Incredible — Highlands to Brevard",
            "This Road Is Amazing — Highlands",
            "Found this road in Port Royal",
            "Discovered this route near Beaufort",
        ],
    )
    def test_subjective_pattern_detected(self, validator, text):
        result = validator.validate(text)
        assert not result.valid, f"Pattern should have blocked: '{text}'"


class TestSafeContent:
    """Legitimate content must pass validation."""

    @pytest.mark.parametrize(
        "text",
        [
            "POV: Driving Through Ireland",
            "Scenic Drive Through Beara Peninsula 🌿",
            "Road Less Traveled — County Clare",
            "Exploring County Kerry by Car 🚗",
            "Just Driving — Ireland Edition",
            "POV: Back Roads of Tipperary 🛤️",
            "Would you take this route?",
            "Where does this road go?",
            "POV driving through County Tipperary, Ireland",
            "Full scenic drive, 1h 15m long.",
            "4K Dashcam: Port Royal to Lady's Island",
            "Driving US-64 — Highlands to Franklin",
        ],
    )
    def test_safe_content_passes(self, validator, text):
        result = validator.validate(text)
        assert result.valid, f"Should have passed: '{text}' — violations: {result.violations}"


class TestValidateAll:
    """Batch validation must catch violations across all fields."""

    def test_catches_unsafe_title(self, validator):
        result = validator.validate_all(
            titles=["You Won't Believe This Road in Ireland"],
            descriptions=["Safe description"],
        )
        assert not result.valid

    def test_catches_unsafe_caption(self, validator):
        result = validator.validate_all(
            titles=["Safe title"],
            captions=["Turn up the volume for this one 🔊"],
        )
        assert not result.valid

    def test_catches_unsafe_hook(self, validator):
        result = validator.validate_all(
            hooks=["Wait for the view... 👀"],
        )
        assert not result.valid

    def test_all_safe_passes(self, validator):
        result = validator.validate_all(
            titles=["POV: Driving Through Ireland", "Scenic Drive — Cork"],
            descriptions=["A relaxing drive through beautiful countryside."],
            captions=["POV: driving through County Kerry"],
            hooks=["Would you take this route?"],
        )
        assert result.valid


class TestEnforce:
    """Enforce must raise on unsafe content."""

    def test_raises_on_violation(self, validator):
        with pytest.raises(ContentValidationError) as exc_info:
            validator.enforce(titles=["Turn up the volume for this one 🔊"])
        assert "violation" in str(exc_info.value).lower()

    def test_passes_safe_content(self, validator):
        # Should not raise
        validator.enforce(
            titles=["POV: Driving Through Ireland"],
            descriptions=["A scenic drive."],
        )


class TestFieldNameInViolations:
    """Violations must identify which field failed."""

    def test_title_field_identified(self, validator):
        result = validator.validate("This is unreal", field_name="youtube_title")
        assert not result.valid
        assert any("youtube_title" in v for v in result.violations)

    def test_caption_field_identified(self, validator):
        result = validator.validate("Wait for the view", field_name="caption")
        assert not result.valid
        assert any("caption" in v for v in result.violations)


class TestBlocksFalseClaimPhrases:
    """False claim phrases must be blocked."""

    @pytest.mark.parametrize(
        "phrase",
        [
            "County Kerry Complete Guide",
            "Ireland: Full Guide to Driving",
            "Full Tour of the Ring of Kerry",
            "Travel Guide to County Cork",
        ],
    )
    def test_blocks_false_claim_phrases(self, validator, phrase):
        result = validator.validate(phrase)
        assert not result.valid, f"Should have blocked: '{phrase}'"

    def test_scenic_drive_passes(self, validator):
        result = validator.validate("County Kerry: Scenic Drive")
        assert result.valid

    def test_full_drive_passes(self, validator):
        result = validator.validate("County Kerry: Full Drive")
        assert result.valid

    def test_pov_drive_passes(self, validator):
        result = validator.validate("Ring of Kerry | 4K POV Drive")
        assert result.valid


class TestCustomBlocklist:
    """Extra blocked phrases can be added at construction."""

    def test_extra_phrase_blocked(self):
        validator = ContentValidator(extra_blocked_phrases=["subscribe or else"])
        result = validator.validate("Subscribe or else you'll miss out!")
        assert not result.valid

    def test_extra_pattern_blocked(self):
        validator = ContentValidator(
            extra_blocked_patterns=[(r"\bsmash\s+that\b", "Engagement bait")]
        )
        result = validator.validate("Smash that like button!")
        assert not result.valid
