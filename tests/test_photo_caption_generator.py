"""Tests for photo caption generator (Claude CLI subprocess).

Covers: _call_llm via subprocess, prompt building, validation, retry loop.
"""

import json
import re
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline.photos.caption_generator import (
    CaptionBatch,
    CaptionGenerator,
    PhotoCaption,
    _build_prompt,
    _count_stats,
    _fallback_caption,
    _validate_caption,
    _validate_fields,
    _validate_stats_density,
)
from gomoveshift.pipeline.photos.selector import PhotoSelection
from gomoveshift.pipeline.photos.voice_styles import (
    VOICE_EXAMPLES,
    VOICE_PROMPTS,
    VoiceStyle,
)


@pytest.fixture
def sample_selections():
    return [
        PhotoSelection(
            journey_slug="example-2025-01",
            journey_title="Example Journey January 2025",
            day_number=3,
            day_date="2025-01-15",
            location="Fablemara",
            filename="photo-042.jpg",
            photo_path="/fake/path/photo-042.jpg",
            locations_visited=["Clifden", "Fablemara", "Roundstone"],
            day_summary="Exploring the rugged coast of Fablemara",
            statistics={"distance_km": 120, "duration_hours": 5.2},
        ),
        PhotoSelection(
            journey_slug="example-2025",
            journey_title="Example Road Trip - 2025",
            day_number=1,
            day_date="2025-05-10",
            location="Mockridge Pass",
            filename="photo-002.jpg",
            photo_path="/fake/path/photo-002.jpg",
            locations_visited=["Mockridge Pass", "Example Gulch"],
            day_summary="Through the mountains",
            statistics={"elevation_gain_m": 1200},
        ),
    ]


MOCK_LLM_RESPONSE = json.dumps(
    [
        {
            "main_caption": "The light hitting Fablemara that morning was something else entirely.",
            "tiktok_caption": "Fablemara morning light",
            "hashtags": ["ireland", "fablemara", "travel", "roadtrip", "wildatlanticway"],
        },
        {
            "main_caption": "Mockridge Pass took my breath away with those endless peaks.",
            "tiktok_caption": "Mockridge Pass views",
            "hashtags": ["colorado", "rockies", "roadtrip", "mountains", "travel"],
        },
    ]
)


class TestBuildPrompt:
    def test_includes_voice_instructions(self, sample_selections):
        prompt = _build_prompt(sample_selections, VoiceStyle.FIRST_PERSON)
        assert "recalling the moment from memory" in prompt

    def test_includes_photo_context(self, sample_selections):
        prompt = _build_prompt(sample_selections, VoiceStyle.MINIMAL_ZEN)
        assert "Fablemara" in prompt
        assert "Mockridge Pass" in prompt
        assert "Example Journey January 2025" in prompt

    def test_includes_examples(self, sample_selections):
        prompt = _build_prompt(sample_selections, VoiceStyle.ENGAGING_CONVERSATIONAL)
        assert "Example Gulch" in prompt

    def test_requests_json_format(self, sample_selections):
        prompt = _build_prompt(sample_selections, VoiceStyle.FIRST_PERSON)
        assert "JSON" in prompt
        assert "main_caption" in prompt
        assert "tiktok_caption" in prompt

    def test_prompt_contains_anti_fabrication_rules(self, sample_selections):
        """_build_prompt() includes anti-fabrication instructions."""
        prompt = _build_prompt(sample_selections, VoiceStyle.FIRST_PERSON)
        assert "Do NOT describe weather" in prompt or "DO NOT describe weather" in prompt
        assert "Do NOT invent" in prompt or "DO NOT invent" in prompt

    def test_unknown_location_handled(self):
        """Selection with 'Unknown Location' — prompt warns LLM."""
        selections = [
            PhotoSelection(
                journey_slug="test",
                journey_title="Test Journey",
                day_number=1,
                day_date="2025-01-01",
                location="Unknown Location",
                filename="photo-001.jpg",
                photo_path="/fake/path/photo-001.jpg",
                locations_visited=[],
                day_summary="A day of driving",
            ),
        ]
        prompt = _build_prompt(selections, VoiceStyle.FIRST_PERSON)
        assert "Unknown Location" in prompt
        # Anti-fabrication rules should cover this case
        assert "Do NOT invent" in prompt or "DO NOT invent" in prompt

    def test_prompt_excludes_elevation_data(self, sample_selections):
        """_build_prompt() PHOTO DETAILS section does NOT contain elevation stats."""
        prompt = _build_prompt(sample_selections, VoiceStyle.FIRST_PERSON)
        # Extract the PHOTO DETAILS section (between "PHOTO DETAILS:" and "For each photo")
        photo_details_match = re.search(
            r"PHOTO DETAILS:\n(.*?)(?=\nFor each photo)",
            prompt,
            re.DOTALL,
        )
        assert photo_details_match, "Could not find PHOTO DETAILS section"
        photo_details = photo_details_match.group(1)
        assert "elevation" not in photo_details.lower()
        assert "1200" not in photo_details  # The elevation_gain_m value


def _mock_subprocess_result(stdout_text, returncode=0):
    """Helper: build a fake Popen handle matching claude -p output.

    _call_llm runs claude in its own process group (Popen + communicate) so a
    timeout can reap MCP/node descendants. stderr is read lazily so a test can
    still override it after construction.
    """
    proc = MagicMock()
    proc.returncode = returncode
    proc.stderr = ""
    proc.communicate.side_effect = lambda *a, **k: (stdout_text, proc.stderr)
    return proc


class TestCaptionGenerator:
    @patch("subprocess.Popen")
    def test_generate_batch(self, mock_run, sample_selections, tmp_path):
        mock_run.return_value = _mock_subprocess_result(MOCK_LLM_RESPONSE)

        gen = CaptionGenerator(caption_dir=tmp_path)
        batch = gen.generate_batch(
            sample_selections,
            VoiceStyle.FIRST_PERSON,
            week_start="2026-02-23",
        )

        assert len(batch.captions) == 2
        assert batch.voice_style == "voice_first_person"
        assert batch.reviewed is False
        assert batch.captions[0].location == "Fablemara"
        assert batch.captions[1].location == "Mockridge Pass"
        # Verify claude -p was called with expected flags
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert cmd[0] == "claude"
        assert "-p" in cmd
        assert "--model" in cmd

    @patch("subprocess.Popen")
    def test_truncates_long_captions(self, mock_run, sample_selections, tmp_path):
        long_response = json.dumps(
            [
                {
                    "main_caption": "x" * 300,
                    "tiktok_caption": "y" * 100,
                    "hashtags": ["tag"],
                },
                {
                    "main_caption": "short",
                    "tiktok_caption": "short",
                    "hashtags": ["tag"],
                },
            ]
        )
        mock_run.return_value = _mock_subprocess_result(long_response)

        gen = CaptionGenerator(caption_dir=tmp_path)
        batch = gen.generate_batch(sample_selections, VoiceStyle.MINIMAL_ZEN)

        assert len(batch.captions[0].main_caption) <= 250
        assert len(batch.captions[0].tiktok_caption) <= 90

    @patch("subprocess.Popen")
    def test_empty_metadata_skipped(self, mock_run, tmp_path):
        """All-empty selection excluded from batch sent to LLM."""
        empty_sel = PhotoSelection(
            journey_slug="",
            journey_title="",
            day_number=0,
            day_date="",
            location="Unknown Location",
            filename="photo-001.jpg",
            photo_path="/fake/path/photo-001.jpg",
            locations_visited=[],
            day_summary="",
            statistics={},
        )
        good_sel = PhotoSelection(
            journey_slug="example-2025-01",
            journey_title="Example Journey January 2025",
            day_number=3,
            day_date="2025-01-15",
            location="Fablemara",
            filename="photo-042.jpg",
            photo_path="/fake/path/photo-042.jpg",
            locations_visited=["Clifden"],
            day_summary="Coast road",
        )

        mock_llm_resp = json.dumps(
            [
                {
                    "main_caption": "Coast road through Fablemara.",
                    "tiktok_caption": "Fablemara coast",
                    "hashtags": ["ireland"],
                }
            ]
        )
        mock_run.return_value = _mock_subprocess_result(mock_llm_resp)

        gen = CaptionGenerator(caption_dir=tmp_path)
        batch = gen.generate_batch(
            [empty_sel, good_sel],
            VoiceStyle.FIRST_PERSON,
            week_start="2026-02-23",
        )
        assert len(batch.captions) == 1
        assert batch.captions[0].location == "Fablemara"

    @patch("subprocess.Popen")
    def test_call_llm_retries_on_failure(self, mock_run, tmp_path):
        """_call_llm retries once on subprocess failure then raises."""
        mock_run.return_value = _mock_subprocess_result("", returncode=1)
        mock_run.return_value.stderr = "connection refused"

        gen = CaptionGenerator(caption_dir=tmp_path)
        with pytest.raises(RuntimeError, match="claude -p exit 1"):
            gen._call_llm("test prompt")

        assert mock_run.call_count == 2  # initial + 1 retry

    @patch("subprocess.Popen")
    def test_call_llm_returns_stdout(self, mock_run, tmp_path):
        """_call_llm returns stripped stdout from claude -p."""
        mock_run.return_value = _mock_subprocess_result('  [{"main_caption": "test"}]  ')

        gen = CaptionGenerator(caption_dir=tmp_path)
        result = gen._call_llm("test prompt")

        assert result == '[{"main_caption": "test"}]'


class TestCaptionValidation:
    """TDD tests for post-generation caption validation."""

    def test_validate_caption_catches_fabricated_location(self):
        """Caption mentions 'Cliffs of Moher' but metadata has 'Fablemara' -> caught."""
        sel = PhotoSelection(
            journey_slug="example-2025-01",
            journey_title="Example Journey January 2025",
            day_number=3,
            day_date="2025-01-15",
            location="Fablemara",
            filename="photo-042.jpg",
            photo_path="/fake/path/photo-042.jpg",
            locations_visited=["Clifden", "Fablemara", "Roundstone"],
            day_summary="Exploring the coast",
        )
        issues = _validate_caption(
            "The Cliffs of Moher were breathtaking today.",
            sel,
        )
        assert len(issues) > 0

    def test_validate_caption_passes_grounded_text(self):
        """Caption uses only metadata locations -> passes."""
        sel = PhotoSelection(
            journey_slug="example-2025-01",
            journey_title="Example Journey January 2025",
            day_number=3,
            day_date="2025-01-15",
            location="Fablemara",
            filename="photo-042.jpg",
            photo_path="/fake/path/photo-042.jpg",
            locations_visited=["Clifden", "Fablemara", "Roundstone"],
            day_summary="Exploring the coast of Fablemara",
        )
        issues = _validate_caption(
            "Day three on the Fablemara coast. Clifden and Roundstone.",
            sel,
        )
        assert len(issues) == 0

    def test_fallback_caption_uses_confirmed_metadata(self):
        """Failed validation -> fallback uses location when available."""
        sel = PhotoSelection(
            journey_slug="example-2025-01",
            journey_title="Example Journey January 2025",
            day_number=3,
            day_date="2025-01-15",
            location="Fablemara",
            filename="photo-042.jpg",
            photo_path="/fake/path/photo-042.jpg",
        )
        fallback = _fallback_caption(sel)
        assert fallback == "Fablemara. Example Journey January 2025, Day 3 (January 2025)."

    def test_fallback_caption_no_location_uses_locations_visited(self):
        """Fallback with Unknown Location but locations_visited."""
        sel = PhotoSelection(
            journey_slug="example-2025-01",
            journey_title="Example Journey January 2025",
            day_number=3,
            day_date="2025-01-15",
            location="Unknown Location",
            filename="photo-042.jpg",
            photo_path="/fake/path/photo-042.jpg",
            locations_visited=["Clifden", "Fablemara", "Roundstone"],
        )
        fallback = _fallback_caption(sel)
        assert (
            fallback
            == "Clifden, Fablemara, Roundstone. Example Journey January 2025, Day 3 (January 2025)."
        )

    def test_fallback_caption_bare_minimum(self):
        """Fallback with no location data falls back to journey — Day N."""
        sel = PhotoSelection(
            journey_slug="example-2025-01",
            journey_title="Example Journey January 2025",
            day_number=3,
            day_date="2025-01-15",
            location="Unknown Location",
            filename="photo-042.jpg",
            photo_path="/fake/path/photo-042.jpg",
        )
        fallback = _fallback_caption(sel)
        assert fallback == "Example Journey January 2025 \u2014 Day 3 (January 2025)"

    def test_validate_compound_place_name(self):
        """'Rock of Cashel' grounded in metadata passes."""
        sel = PhotoSelection(
            journey_slug="example-2018-06",
            journey_title="Example Journey June 2018",
            day_number=5,
            day_date="2018-06-15",
            location="Rock of Cashel",
            filename="photo-100.jpg",
            photo_path="/fake/path/photo-100.jpg",
            locations_visited=["Rock of Cashel", "Cahir"],
            day_summary="Visiting the Rock of Cashel and Cahir Castle",
        )
        issues = _validate_caption(
            "Standing at the Rock of Cashel, history all around.",
            sel,
        )
        assert len(issues) == 0

    def test_validate_single_proper_noun(self):
        """'Fablemara' grounded in metadata passes."""
        sel = PhotoSelection(
            journey_slug="example-2025-01",
            journey_title="Example Journey January 2025",
            day_number=3,
            day_date="2025-01-15",
            location="Fablemara",
            filename="photo-042.jpg",
            photo_path="/fake/path/photo-042.jpg",
            locations_visited=["Clifden", "Fablemara"],
            day_summary="Exploring Fablemara",
        )
        issues = _validate_caption(
            "Fablemara stretches out before you, wild and green.",
            sel,
        )
        assert len(issues) == 0

    def test_validate_fabricated_still_caught(self):
        """'Eiffel Tower' with Irish metadata fails."""
        sel = PhotoSelection(
            journey_slug="example-2025-01",
            journey_title="Example Journey January 2025",
            day_number=3,
            day_date="2025-01-15",
            location="Fablemara",
            filename="photo-042.jpg",
            photo_path="/fake/path/photo-042.jpg",
            locations_visited=["Clifden", "Fablemara"],
            day_summary="Exploring Fablemara",
        )
        issues = _validate_caption(
            "The Eiffel Tower sparkled in the Paris night.",
            sel,
        )
        assert len(issues) > 0

    def test_validate_single_ungrounded_noun_fails(self):
        """100% grounding required: one ungrounded proper noun fails the caption."""
        sel = PhotoSelection(
            journey_slug="example-2025-01",
            journey_title="Example Journey January 2025",
            day_number=3,
            day_date="2025-01-15",
            location="Fablemara",
            filename="photo-042.jpg",
            photo_path="/fake/path/photo-042.jpg",
            locations_visited=["Clifden", "Fablemara", "Roundstone"],
            day_summary="Coast road through Fablemara",
        )
        # 3 grounded (Clifden, Fablemara, Roundstone) + 1 ungrounded (Barcelona).
        # Under 100% grounding the lone hallucinated place must be flagged.
        issues = _validate_caption(
            "From Clifden to Fablemara via Roundstone, with a Barcelona vibe.",
            sel,
        )
        assert any("Barcelona" in issue for issue in issues)

    def test_caption_rejects_elevation_references(self):
        """Caption mentioning 'elevation gain' or 'elevation' -> validation fails."""
        sel = PhotoSelection(
            journey_slug="example-2025",
            journey_title="Example Road Trip",
            day_number=1,
            day_date="2025-05-10",
            location="Mockridge Pass",
            filename="photo-002.jpg",
            photo_path="/fake/path/photo-002.jpg",
            locations_visited=["Mockridge Pass"],
            day_summary="Mountain roads",
        )
        issues = _validate_caption(
            "1200m of elevation gain through the Rockies today.",
            sel,
        )
        assert any("elevation" in issue.lower() for issue in issues)


class TestAcronymValidation:
    """Verify all-caps acronyms (e.g. USA) don't cause false positives in _validate_caption."""

    def test_validate_all_caps_acronym_not_flagged(self):
        """'USA' is all-caps and doesn't match the [A-Z][a-z]{2,} regex, so it's ignored."""
        sel = PhotoSelection(
            journey_slug="example-2025",
            journey_title="Example Road Trip",
            day_number=1,
            day_date="2025-05-10",
            location="Mockridge Pass",
            filename="photo-002.jpg",
            photo_path="/fake/path/photo-002.jpg",
            locations_visited=["Mockridge Pass"],
            day_summary="Mountain roads in the USA",
        )
        issues = _validate_caption(
            "Driving through the USA on this Mockridge Pass stretch.",
            sel,
        )
        # "USA" is all-caps → not extracted as a proper noun → no false positive
        assert len(issues) == 0


class TestVoiceStyleAntiHallucination:
    """TDD tests for voice style prompt and example fixes."""

    def test_voice_examples_no_sensory_fabrication(self):
        """VOICE_EXAMPLES don't contain 'felt', 'heard', 'saw'."""
        sensory_words = {"felt", "heard", "saw"}
        for style, examples in VOICE_EXAMPLES.items():
            for example in examples:
                words = set(re.findall(r"\b\w+\b", example.lower()))
                found = words & sensory_words
                assert not found, (
                    f"Voice example for {style.value} contains sensory word(s): {found}\n"
                    f"Example: {example}"
                )

    def test_first_person_prompt_no_sensory_instruction(self):
        """FIRST_PERSON prompt doesn't instruct LLM to fabricate sensory details."""
        prompt = VOICE_PROMPTS[VoiceStyle.FIRST_PERSON]
        # Should not tell the LLM to include/imagine sensory details
        assert "what you saw" not in prompt
        assert "what you heard" not in prompt
        assert "what you felt" not in prompt
        assert "Include sensory details" not in prompt


class TestCaptionBatch:
    def test_save_and_load(self, tmp_path):
        batch = CaptionBatch(
            week_start="2026-02-23",
            voice_style="voice_first_person",
            captions=[
                PhotoCaption(
                    photo_index=0,
                    main_caption="Test caption",
                    tiktok_caption="Test",
                    hashtags=["ireland", "travel"],
                    journey_slug="example-2025-01",
                    filename="photo-042.jpg",
                    location="Fablemara",
                ),
            ],
            reviewed=False,
            generated_at="2026-02-16T10:00:00",
            model="claude-sonnet-4-5-20250929",
        )

        path = tmp_path / "test_captions.yaml"
        batch.save(path)
        assert path.exists()

        loaded = CaptionBatch.load(path)
        assert loaded.week_start == "2026-02-23"
        assert loaded.voice_style == "voice_first_person"
        assert len(loaded.captions) == 1
        assert loaded.captions[0].location == "Fablemara"
        assert loaded.reviewed is False

    def test_reviewed_flag_persists(self, tmp_path):
        batch = CaptionBatch(
            week_start="2026-02-23",
            voice_style="voice_minimal_zen",
            captions=[],
            reviewed=True,
        )
        path = tmp_path / "reviewed.yaml"
        batch.save(path)

        loaded = CaptionBatch.load(path)
        assert loaded.reviewed is True


class TestParseResponse:
    def test_parses_json_array(self):
        gen = CaptionGenerator()
        result = gen._parse_response(MOCK_LLM_RESPONSE)
        assert len(result) == 2
        assert "main_caption" in result[0]

    def test_strips_markdown_fences(self):
        gen = CaptionGenerator()
        fenced = f"```json\n{MOCK_LLM_RESPONSE}\n```"
        result = gen._parse_response(fenced)
        assert len(result) == 2

    def test_raises_on_invalid_json(self):
        gen = CaptionGenerator()
        with pytest.raises(ValueError, match="invalid JSON"):
            gen._parse_response("not json at all")


class TestMonthAndGeoValidation:
    """Month names, geographic terms, and directions should not trigger false positives."""

    def test_month_name_not_flagged_as_ungrounded(self):
        """'January' in caption should not be treated as a proper noun to validate."""
        sel = PhotoSelection(
            journey_slug="example-2025-01",
            journey_title="Example Journey January 2025",
            day_number=3,
            day_date="2025-01-15",
            location="Fablemara",
            filename="photo-042.jpg",
            photo_path="/fake/path/photo-042.jpg",
            locations_visited=["Clifden", "Fablemara"],
            day_summary="Exploring Fablemara",
        )
        issues = _validate_caption(
            "Fablemara in January. Clifden was quiet.",
            sel,
        )
        assert len(issues) == 0

    def test_geographic_term_not_flagged(self):
        """'Peninsula', 'Coast', etc. should not be treated as ungrounded proper nouns."""
        sel = PhotoSelection(
            journey_slug="example-2018",
            journey_title="Example Road Trip - 2018",
            day_number=1,
            day_date="2018-08-10",
            location="Avalon Peninsula",
            filename="photo-001.jpg",
            photo_path="/fake/path/photo-001.jpg",
            locations_visited=["Avalon Peninsula", "Cape Spear"],
            day_summary="Driving the Avalon Peninsula coast",
        )
        issues = _validate_caption(
            "The Avalon Peninsula stretches along the Coast. August 2018.",
            sel,
        )
        assert len(issues) == 0

    def test_direction_word_not_flagged(self):
        """'Northern', 'Southern', etc. should not be treated as ungrounded proper nouns."""
        sel = PhotoSelection(
            journey_slug="example-2025-01",
            journey_title="Example Journey January 2025",
            day_number=5,
            day_date="2025-01-17",
            location="Donegal",
            filename="photo-080.jpg",
            photo_path="/fake/path/photo-080.jpg",
            locations_visited=["Donegal"],
            day_summary="Northern Ireland coast",
        )
        issues = _validate_caption(
            "Northern Donegal. January 2025.",
            sel,
        )
        assert len(issues) == 0


class TestDayDateTokenization:
    """day_date month/year should be added to metadata_words for validation."""

    def test_month_year_from_day_date_grounds_caption(self):
        """Caption mentioning 'August' grounded via day_date='2018-08-10'."""
        sel = PhotoSelection(
            journey_slug="example-2018",
            journey_title="Example Road Trip - 2018",
            day_number=1,
            day_date="2018-08-10",
            location="St. John's",
            filename="photo-001.jpg",
            photo_path="/fake/path/photo-001.jpg",
            locations_visited=["St. John's"],
            day_summary="Arriving in Newfoundland",
        )
        # "August" is a month name (in _COMMON_WORDS), so it won't be extracted
        # as a proper noun. But "Newfoundland" needs grounding — it should match
        # via day_summary.
        issues = _validate_caption(
            "Newfoundland in August 2018.",
            sel,
        )
        assert len(issues) == 0


class TestPromptIncludesDurationAndPhotoCount:
    """_build_prompt should include duration_hours and photo_count from statistics."""

    def test_prompt_contains_duration(self, sample_selections):
        prompt = _build_prompt(sample_selections, VoiceStyle.FIRST_PERSON)
        assert "Duration: 5.2 hours" in prompt

    def test_prompt_contains_photo_count(self):
        selections = [
            PhotoSelection(
                journey_slug="test",
                journey_title="Test Journey",
                day_number=1,
                day_date="2025-01-01",
                location="Somewhere",
                filename="photo-001.jpg",
                photo_path="/fake/path/photo-001.jpg",
                locations_visited=["Somewhere"],
                day_summary="A day",
                statistics={"photo_count": 54},
            ),
        ]
        prompt = _build_prompt(selections, VoiceStyle.MINIMAL_ZEN)
        assert "Photos taken this day: 54" in prompt


class TestHookAlwaysPresent:
    """first_comment_hook should be requested in prompt for ALL voice styles."""

    @pytest.mark.parametrize("style", list(VoiceStyle))
    def test_hook_in_prompt_for_all_styles(self, style):
        selections = [
            PhotoSelection(
                journey_slug="test",
                journey_title="Test Journey",
                day_number=1,
                day_date="2025-01-01",
                location="Somewhere",
                filename="photo-001.jpg",
                photo_path="/fake/path/photo-001.jpg",
                locations_visited=["Somewhere"],
                day_summary="A day",
            ),
        ]
        prompt = _build_prompt(selections, style)
        assert "first_comment_hook" in prompt

    def test_hook_instruction_forbids_gimmicky_ctas(self):
        selections = [
            PhotoSelection(
                journey_slug="test",
                journey_title="Test Journey",
                day_number=1,
                day_date="2025-01-01",
                location="Somewhere",
                filename="photo-001.jpg",
                photo_path="/fake/path/photo-001.jpg",
                locations_visited=["Somewhere"],
                day_summary="A day",
            ),
        ]
        prompt = _build_prompt(selections, VoiceStyle.FIRST_PERSON)
        assert "Rate 1-10" in prompt  # Listed in the NEVER-use block
        assert "Tag someone" in prompt
        assert "Drop your answer" in prompt


class TestFallbackWithStats:
    """Fallback captions should include distance/duration when available."""

    def test_fallback_with_distance_and_duration(self):
        sel = PhotoSelection(
            journey_slug="example-2025-01",
            journey_title="Example Journey January 2025",
            day_number=3,
            day_date="2025-01-15",
            location="Fablemara",
            filename="photo-042.jpg",
            photo_path="/fake/path/photo-042.jpg",
            statistics={"distance_km": 75, "duration_hours": 3.8},
        )
        fallback = _fallback_caption(sel)
        assert "75 km" in fallback
        assert "hours" not in fallback
        assert fallback.startswith("Fablemara.")

    def test_fallback_with_distance_only(self):
        sel = PhotoSelection(
            journey_slug="example-2025-01",
            journey_title="Example Journey January 2025",
            day_number=3,
            day_date="2025-01-15",
            location="Fablemara",
            filename="photo-042.jpg",
            photo_path="/fake/path/photo-042.jpg",
            statistics={"distance_km": 120},
        )
        fallback = _fallback_caption(sel)
        assert "120 km" in fallback
        assert "hours" not in fallback

    def test_fallback_without_stats_unchanged(self):
        sel = PhotoSelection(
            journey_slug="example-2025-01",
            journey_title="Example Journey January 2025",
            day_number=3,
            day_date="2025-01-15",
            location="Fablemara",
            filename="photo-042.jpg",
            photo_path="/fake/path/photo-042.jpg",
        )
        fallback = _fallback_caption(sel)
        assert fallback == "Fablemara. Example Journey January 2025, Day 3 (January 2025)."


class TestStatsDensityValidation:
    """Tests for _count_stats() and _validate_stats_density()."""

    def test_two_stats_fails(self):
        """'198 km looping... Ten hours' has 2 stats → fails."""
        issue = _validate_stats_density("198 km looping through Inishowen. Ten hours, one county.")
        assert issue is not None
        assert "statistics" in issue

    def test_one_stat_at_end_passes(self):
        """'...75 km of Inishowen.' — one stat, not leading → passes."""
        issue = _validate_stats_density("The wild coast of Inishowen, 75 km of Atlantic roads.")
        assert issue is None

    def test_stat_leading_fails(self):
        """'106 photos in a single day...' — stat leads → fails."""
        issue = _validate_stats_density("106 photos in a single day of Fablemara.")
        assert issue is not None
        assert "leads with" in issue

    def test_zero_stats_passes(self):
        """Pure place caption with no stats → passes."""
        issue = _validate_stats_density("Fablemara stretches out, wild and green.")
        assert issue is None

    def test_year_not_counted(self):
        """'June 2018' — 2018 is a year, not a stat."""
        count, _ = _count_stats("Fablemara in June 2018.")
        assert count == 0

    def test_day_number_not_counted(self):
        """'Day 3' is not a stat."""
        count, _ = _count_stats("Day 3 on the Fablemara coast.")
        assert count == 0

    def test_written_number_with_unit(self):
        """'fourteen hours' is detected as a stat."""
        count, _ = _count_stats("Nearly fourteen hours on the road.")
        assert count == 1

    def test_tiktok_stats_checked(self):
        """TikTok caption with 2+ stats fails."""
        issue = _validate_stats_density("198 km in ten hours through Donegal")
        assert issue is not None

    def test_hook_leading_stat_fails(self):
        """Hook starting with number+unit → fails."""
        issue = _validate_stats_density("106 photos across Inishowen Peninsula")
        assert issue is not None
        assert "leads with" in issue


class TestRetryLoop:
    """Tests for the retry mechanism in generate_batch."""

    def _make_selection(self):
        return PhotoSelection(
            journey_slug="example-2025-01",
            journey_title="Example Journey January 2025",
            day_number=3,
            day_date="2025-01-15",
            location="Fablemara",
            filename="photo-042.jpg",
            photo_path="/fake/path/photo-042.jpg",
            locations_visited=["Clifden", "Fablemara", "Roundstone"],
            day_summary="Exploring the rugged coast of Fablemara",
            statistics={"distance_km": 120, "duration_hours": 5.2},
        )

    def _mock_llm_responses(self, mock_run, captions_list):
        """Set up mock to return different responses on successive calls."""
        results = []
        for caption_data in captions_list:
            content = json.dumps(caption_data)
            results.append(_mock_subprocess_result(content))
        mock_run.side_effect = results

    @patch("subprocess.Popen")
    def test_retry_on_stats_failure(self, mock_run, tmp_path):
        """LLM returns stats-heavy first, clean on retry → clean caption used."""
        sel = self._make_selection()
        stats_heavy = [
            {
                "main_caption": "198 km looping through Inishowen. Ten hours on the road.",
                "tiktok_caption": "Fablemara coast",
                "hashtags": ["ireland"],
                "first_comment_hook": "The coast road never ends.",
            }
        ]
        clean_retry = {
            "main_caption": "The rugged coast of Fablemara, Clifden to Roundstone.",
        }
        self._mock_llm_responses(mock_run, [stats_heavy, clean_retry])

        gen = CaptionGenerator(caption_dir=tmp_path)
        batch = gen.generate_batch([sel], VoiceStyle.FIRST_PERSON, week_start="2026-03-18")

        assert len(batch.captions) == 1
        assert "Fablemara" in batch.captions[0].main_caption
        assert "198 km" not in batch.captions[0].main_caption
        # Should have called LLM twice (batch + 1 retry)
        assert mock_run.call_count == 2

    @patch("subprocess.Popen")
    def test_max_retries_then_fallback(self, mock_run, tmp_path):
        """LLM always returns stats-heavy → fallback after 2 retries."""
        sel = self._make_selection()
        stats_heavy = [
            {
                "main_caption": "198 km looping through Inishowen. Ten hours on the road.",
                "tiktok_caption": "198 km in ten hours",
                "hashtags": ["ireland"],
                "first_comment_hook": "106 photos across the peninsula.",
            }
        ]
        bad_retry = {
            "main_caption": "120 km and five hours through Fablemara coast.",
            "tiktok_caption": "120 km in five hours",
            "first_comment_hook": "54 shots of Atlantic roads.",
        }
        self._mock_llm_responses(mock_run, [stats_heavy, bad_retry, bad_retry])

        gen = CaptionGenerator(caption_dir=tmp_path)
        batch = gen.generate_batch([sel], VoiceStyle.FIRST_PERSON, week_start="2026-03-18")

        assert len(batch.captions) == 1
        # main_caption should be fallback (starts with location)
        assert batch.captions[0].main_caption.startswith("Fablemara.")
        # tiktok fallback = location
        assert batch.captions[0].tiktok_caption == "Fablemara"
        # hook fallback = empty
        assert batch.captions[0].first_comment_hook == ""
        # 3 LLM calls: batch + 2 retries
        assert mock_run.call_count == 3

    @patch("subprocess.Popen")
    def test_no_retry_when_valid(self, mock_run, tmp_path):
        """Clean caption on first try → no retry call."""
        sel = self._make_selection()
        clean = [
            {
                "main_caption": "The rugged coast of Fablemara, Clifden to Roundstone.",
                "tiktok_caption": "Fablemara coast road",
                "hashtags": ["ireland"],
                "first_comment_hook": "The coast road never ends.",
            }
        ]
        self._mock_llm_responses(mock_run, [clean])

        gen = CaptionGenerator(caption_dir=tmp_path)
        batch = gen.generate_batch([sel], VoiceStyle.FIRST_PERSON, week_start="2026-03-18")

        assert len(batch.captions) == 1
        assert "Fablemara" in batch.captions[0].main_caption
        # Only 1 LLM call — no retries
        assert mock_run.call_count == 1


class TestPinterestDescriptionNotGenerated:
    """pinterest_description has no Metricool CSV column, so it is never requested.

    The 31-column golden schema exposes only ``Pinterest Board``; Pinterest reads
    the shared ``Text`` column. Generating the field burned LLM tokens on output
    that was discarded, so it is no longer asked for, validated, or retried.
    """

    def _make_selection(self):
        return PhotoSelection(
            journey_slug="example-2025-01",
            journey_title="Example Journey January 2025",
            day_number=3,
            day_date="2025-01-15",
            location="Fablemara",
            filename="photo-042.jpg",
            photo_path="/fake/path/photo-042.jpg",
            locations_visited=["Clifden", "Fablemara", "Roundstone"],
            day_summary="Exploring the rugged coast of Fablemara",
            statistics={"distance_km": 120, "duration_hours": 5.2},
        )

    def test_batch_prompt_does_not_request_pinterest_description(self, sample_selections):
        prompt = _build_prompt(sample_selections, VoiceStyle.FIRST_PERSON)
        assert "pinterest_description" not in prompt

    def test_validate_fields_never_flags_pinterest_description(self):
        """An ungrounded pinterest value must not open a retry — it is not shipped."""
        data = {
            "main_caption": "The rugged coast of Fablemara, Clifden to Roundstone.",
            "tiktok_caption": "Fablemara coast road",
            "first_comment_hook": "The coast road never ends.",
            "pinterest_description": "Sunset over Tuscany, Barcelona and Reykjavik.",
        }
        failed = _validate_fields(data, self._make_selection())
        assert "pinterest_description" not in failed

    @patch("subprocess.Popen")
    def test_no_retry_when_only_pinterest_would_fail(self, mock_run, tmp_path):
        """A bad pinterest value alone must not trigger a second LLM call."""
        payload = [
            {
                "main_caption": "The rugged coast of Fablemara, Clifden to Roundstone.",
                "tiktok_caption": "Fablemara coast road",
                "pinterest_description": "Sunset over Tuscany, Barcelona and Reykjavik.",
                "hashtags": ["ireland"],
                "first_comment_hook": "The coast road never ends.",
            }
        ]
        mock_run.return_value = _mock_subprocess_result(json.dumps(payload))

        gen = CaptionGenerator(caption_dir=tmp_path)
        batch = gen.generate_batch(
            [self._make_selection()], VoiceStyle.FIRST_PERSON, week_start="2026-03-18"
        )

        assert len(batch.captions) == 1
        assert mock_run.call_count == 1

    @patch("subprocess.Popen")
    def test_generated_caption_carries_no_pinterest_text(
        self, mock_run, sample_selections, tmp_path
    ):
        """Even if the LLM volunteers the field, it is not carried into the batch."""
        mock_run.return_value = _mock_subprocess_result(MOCK_LLM_RESPONSE)

        gen = CaptionGenerator(caption_dir=tmp_path)
        batch = gen.generate_batch(
            sample_selections, VoiceStyle.FIRST_PERSON, week_start="2026-02-23"
        )

        assert not any(hasattr(c, "pinterest_description") for c in batch.captions)


class TestCaptionDirDefault:
    """The cache directory follows the photo output dir, the root the orchestrator
    writes captions under; an explicit dir wins."""

    def test_default_is_under_photo_output_dir(self, tmp_path, monkeypatch):
        from gomoveshift.pipeline.photos.caption_generator import default_caption_dir

        monkeypatch.setenv("GMS_PHOTO_OUTPUT_DIR", str(tmp_path / "out"))
        assert default_caption_dir() == tmp_path / "out" / "photo_captions"
        assert CaptionGenerator().caption_dir == tmp_path / "out" / "photo_captions"

    def test_default_follows_data_home_without_override(self, tmp_path, monkeypatch):
        from gomoveshift.pipeline.photos.caption_generator import default_caption_dir

        monkeypatch.delenv("GMS_PHOTO_OUTPUT_DIR", raising=False)
        monkeypatch.setenv("GMS_DATA_HOME", str(tmp_path / "data"))
        assert default_caption_dir() == tmp_path / "data" / "photos" / "photo_captions"

    def test_explicit_caption_dir_overrides_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GMS_PHOTO_OUTPUT_DIR", str(tmp_path / "out"))
        explicit = tmp_path / "mine"
        assert CaptionGenerator(caption_dir=explicit).caption_dir == explicit
