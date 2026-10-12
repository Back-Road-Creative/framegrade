"""
TDD tests for longform metadata — fact-based description body.

Fact-only description body replaces generic "beautiful roads" line.
Hook quality gate — hooks must reference GPS-verified places.
No LLM calls — descriptions use only verifiable data (towns, routes, duration).
"""

from unittest.mock import patch

import pytest

from gomoveshift.pipeline.longform_metadata import (
    LongFormMetadataGenerator,
    _clamp_title_len,
    _select_verb,
)


class TestTrimRatioVerb:
    """A literal-zero trim_ratio is 'unmeasured', NOT a 100%-trimmed
    highlight — a full-length LF must never get the 'Highlights from' verb."""

    def test_zero_ratio_is_driving(self):
        assert _select_verb(0) == "Driving"
        assert _select_verb(0.0) == "Driving"

    def test_none_ratio_is_driving(self):
        assert _select_verb(None) == "Driving"

    def test_real_heavy_edit_is_highlights(self):
        assert _select_verb(0.3) == "Highlights from"

    def test_light_edit_is_driving(self):
        assert _select_verb(0.8) == "Driving"


class TestFinalCutDisclosure:
    """The '🎬 Final cut: X (edited from Y)' honesty disclosure must be
    gated on numeric seconds, not formatted-string equality (round-collapse)."""

    def test_material_edit_shows_final_cut(self, generator):
        desc = generator._build_gms_description(
            location="Marin County, CA",
            seo_description="POV drive.",
            duration_str="20m 0s",
            publish_day="Tuesday",
            source_duration=3600.0,  # 60 min source
            video_duration=1200.0,  # 20 min final — a real edit
        )
        assert "🎬 Final cut:" in desc

    def test_round_collapse_no_longer_hides_real_edit(self, generator):
        # Source 1850s and final 1805s could format to the same coarse
        # string; the numeric gate (Δ=45s ≥ 30s) still discloses the edit.
        desc = generator._build_gms_description(
            location="Marin County, CA",
            seo_description="POV drive.",
            duration_str="30m 5s",
            publish_day="Tuesday",
            source_duration=1850.0,
            video_duration=1805.0,
        )
        assert "🎬 Final cut:" in desc

    def test_trivial_edit_uses_plain_duration(self, generator):
        # Sub-30s delta → no disclosure noise, plain Duration line.
        desc = generator._build_gms_description(
            location="Marin County, CA",
            seo_description="POV drive.",
            duration_str="30m 0s",
            publish_day="Tuesday",
            source_duration=1810.0,
            video_duration=1800.0,
        )
        assert "🎬 Final cut:" not in desc
        assert "🎬 Duration:" in desc


class TestTitleLengthClamp:
    """The assembled YouTube title (verb + base + "(Part N of M)") must
    never exceed YouTube's 100-char hard limit, with the suffix preserved."""

    def test_short_title_unchanged(self):
        t = "Marin County, CA in 4K"
        assert _clamp_title_len(t) == t

    def test_over_limit_truncated_on_word_boundary(self):
        base = "Driving " + "Long Place Name " * 10  # well over 100
        out = _clamp_title_len(base)
        assert len(out) <= 100
        assert not out.endswith(" ")  # word-boundary, no dangling space
        assert out.startswith("Driving ")  # verb preserved

    def test_part_suffix_preserved_when_truncating(self):
        base = "Driving " + "Somewhereville to Anotherlongplace " * 5
        title = base.strip() + " (Part 2 of 7)"
        assert len(title) > 100
        out = _clamp_title_len(title)
        assert len(out) <= 100
        assert out.endswith("(Part 2 of 7)")
        assert out.startswith("Driving ")

    def test_highlights_verb_and_suffix_both_survive(self):
        title = "Highlights from " + ("Big County Region " * 6).strip() + " (Part 1 of 3)"
        out = _clamp_title_len(title)
        assert len(out) <= 100
        assert out.startswith("Highlights from ")
        assert out.endswith("(Part 1 of 3)")

    def test_build_title_clamps_end_to_end(self):
        # _build_title prepends the verb; a very long base must come back ≤100.
        long_base = "Somewhereville to Anotherlongplacename via The Very Long Scenic Parkway, NC"
        title = LongFormMetadataGenerator._build_title(location=long_base, trim_ratio=1.0)
        assert len(title) <= 100


@pytest.fixture
def generator():
    """Create generator with mocked SEO manager (no credentials needed)."""
    with patch.object(LongFormMetadataGenerator, "_init_seo_manager"):
        with patch.object(LongFormMetadataGenerator, "_init_ledger"):
            gen = LongFormMetadataGenerator()
            gen._seo_degraded = True  # skip SEO manager calls
            return gen


class TestBuildGmsDescription:
    """_build_gms_description body line must use only verifiable facts."""

    def test_description_contains_location(self, generator):
        """Basic: description must contain the location."""
        desc = generator._build_gms_description(
            location="South Carolina, United States",
            seo_description="POV drive along the roads of South Carolina.",
            duration_str="26m 59s",
            publish_day="Wednesday",
        )
        assert "South Carolina" in desc

    def test_description_no_visual_claims(self, generator):
        """Description must not contain subjective visual adjectives."""
        desc = generator._build_gms_description(
            location="South Carolina, United States",
            seo_description="POV drive.",
            duration_str="26m 59s",
            publish_day="Wednesday",
        )
        lower = desc.lower()
        for word in ["beautiful", "stunning", "breathtaking", "amazing", "gorgeous"]:
            assert word not in lower, (
                f"Description contains unverifiable adjective '{word}': {desc[:200]}"
            )

    def test_description_with_towns(self, generator):
        """When chapter towns are provided, they must appear in the body."""
        desc = generator._build_gms_description(
            location="South Carolina, United States",
            seo_description="POV drive.",
            duration_str="26m 59s",
            publish_day="Wednesday",
            chapter_towns=["Longwood", "Fort Fremont"],
        )
        assert "Longwood" in desc
        assert "Fort Fremont" in desc

    def test_description_with_named_route(self, generator):
        """When named routes are provided, they must appear in the body."""
        desc = generator._build_gms_description(
            location="North Carolina, United States",
            seo_description="POV drive.",
            duration_str="1h 15m",
            publish_day="Saturday",
            named_routes=["Blue Ridge Parkway"],
        )
        assert "Blue Ridge Parkway" in desc

    def test_description_with_terrain_type(self, generator):
        """Coastal/mountain/winding terrain should appear as factual descriptor."""
        desc = generator._build_gms_description(
            location="South Carolina, United States",
            seo_description="POV drive.",
            duration_str="26m 59s",
            publish_day="Wednesday",
            terrain_type="coastal",
        )
        assert "coastal" in desc.lower()

    def test_description_fallback_no_context(self, generator):
        """Without terrain/towns/routes, body line is generic but factual."""
        desc = generator._build_gms_description(
            location="South Carolina, United States",
            seo_description="POV drive.",
            duration_str="26m 59s",
            publish_day="Wednesday",
        )
        # Must contain location and duration, no visual claims
        assert "South Carolina" in desc
        assert "26m 59s" in desc
        lower = desc.lower()
        assert "beautiful" not in lower

    def test_description_countryside_terrain_neutral(self, generator):
        """Countryside terrain must not inject 'back roads' — GPS can't prove it."""
        desc = generator._build_gms_description(
            location="County Tipperary, Ireland",
            seo_description="POV drive.",
            duration_str="30m 0s",
            publish_day="Wednesday",
            terrain_type="countryside",
        )
        assert "back road" not in desc.lower()


class TestRebuildDescriptionBody:
    """After chapters, rebuild description body with route facts."""

    def test_body_replacement_with_towns(self, generator):
        """Rebuild inserts town names into the body paragraph."""
        # Build initial description (no towns)
        desc = generator._build_gms_description(
            location="South Carolina, United States",
            seo_description="POV drive.",
            duration_str="26m 59s",
            publish_day="Wednesday",
        )
        assert "Longwood" not in desc  # precondition

        # Rebuild with towns
        rebuilt = LongFormMetadataGenerator.rebuild_description_body(
            description=desc,
            location="South Carolina, United States",
            duration_str="26m 59s",
            chapter_towns=["Longwood", "Hunting Island State Park", "Fort Fremont"],
        )
        assert "Longwood" in rebuilt
        assert "Hunting Island State Park" in rebuilt
        assert "Fort Fremont" in rebuilt

    def test_body_replacement_preserves_location_line(self, generator):
        """Rebuild must not alter the Location line."""
        desc = generator._build_gms_description(
            location="South Carolina, United States",
            seo_description="POV drive along SC roads.",
            duration_str="26m 59s",
            publish_day="Wednesday",
        )

        rebuilt = LongFormMetadataGenerator.rebuild_description_body(
            description=desc,
            location="South Carolina, United States",
            duration_str="26m 59s",
            chapter_towns=["Longwood", "Fort Fremont"],
        )
        # Location line preserved
        assert "\U0001f4cd Location: South Carolina, United States" in rebuilt
        # Hook is updated to reference chapter towns (hook quality gate)
        hook = rebuilt.split("\n")[0]
        assert "Longwood" in hook or "Fort Fremont" in hook

    def test_body_replacement_noop_without_data(self, generator):
        """Without towns or routes, description is unchanged."""
        desc = generator._build_gms_description(
            location="South Carolina, United States",
            seo_description="POV drive.",
            duration_str="26m 59s",
            publish_day="Wednesday",
        )
        rebuilt = LongFormMetadataGenerator.rebuild_description_body(
            description=desc,
            location="South Carolina, United States",
            duration_str="26m 59s",
        )
        assert rebuilt == desc

    def test_body_replacement_with_named_route(self, generator):
        """Rebuild with named route includes route name in body."""
        desc = generator._build_gms_description(
            location="North Carolina, United States",
            seo_description="POV drive.",
            duration_str="1h 15m",
            publish_day="Saturday",
        )
        rebuilt = LongFormMetadataGenerator.rebuild_description_body(
            description=desc,
            location="North Carolina, United States",
            duration_str="1h 15m",
            named_routes=["Blue Ridge Parkway"],
        )
        assert "Blue Ridge Parkway" in rebuilt


class TestHookQualityGate:
    """Hook line must reference GPS-verified places, never generic/subjective."""

    def test_generic_relaxing_hook_replaced(self, generator):
        """'Join me on a relaxing POV drive...' must be replaced."""
        desc = generator._build_gms_description(
            location="South Carolina, United States",
            seo_description=(
                "Join me on a relaxing POV drive through South Carolina, United States!"
            ),
            duration_str="26m 59s",
            publish_day="Monday",
            chapters=[
                {"timestamp": "0:00", "title": "Start - South Carolina"},
                {"timestamp": "17:00", "title": "Entering Longwood, Beaufort County"},
                {"timestamp": "23:00", "title": "Entering Fort Fremont, Beaufort County"},
            ],
            chapter_towns=["Longwood", "Fort Fremont"],
        )
        hook = desc.split("\n")[0]
        assert "relaxing" not in hook.lower(), f"Generic hook leaked: {hook}"
        assert "Longwood" in hook or "Fort Fremont" in hook, (
            f"Hook must reference a GPS town, got: {hook}"
        )

    def test_hook_with_subjective_adjective_replaced(self, generator):
        """Hooks with 'stunning', 'beautiful', etc. must be replaced."""
        desc = generator._build_gms_description(
            location="South Carolina, United States",
            seo_description=("Experience the stunning beauty of South Carolina roads."),
            duration_str="26m 59s",
            publish_day="Monday",
            chapter_towns=["Longwood", "Fort Fremont"],
        )
        hook = desc.split("\n")[0]
        assert "stunning" not in hook.lower(), f"Subjective hook leaked: {hook}"

    def test_hook_without_towns_uses_region(self, generator):
        """Without chapter towns, hook should use region name only."""
        desc = generator._build_gms_description(
            location="South Carolina, United States",
            seo_description="Join me on a relaxing POV drive through SC!",
            duration_str="26m 59s",
            publish_day="Monday",
        )
        hook = desc.split("\n")[0]
        assert "relaxing" not in hook.lower(), f"Generic hook leaked: {hook}"
        assert "South Carolina" in hook, f"Hook should use region: {hook}"

    def test_rebuild_also_fixes_hook(self, generator):
        """rebuild_description_body must fix a generic hook when towns provided."""
        desc = generator._build_gms_description(
            location="South Carolina, United States",
            seo_description=("Join me on a relaxing POV drive through South Carolina!"),
            duration_str="26m 59s",
            publish_day="Monday",
        )
        rebuilt = LongFormMetadataGenerator.rebuild_description_body(
            description=desc,
            location="South Carolina, United States",
            duration_str="26m 59s",
            chapter_towns=["Longwood", "Fort Fremont"],
        )
        hook = rebuilt.split("\n")[0]
        assert "relaxing" not in hook.lower(), f"Generic hook survived rebuild: {hook}"
        assert "Longwood" in hook or "Fort Fremont" in hook, (
            f"Hook must reference a GPS town after rebuild, got: {hook}"
        )


class TestLocationPlaceholderRejection:
    """LongFormMetadataGenerator.generate() must reject
    placeholder/empty locations at entry. A blank or placeholder location
    cascades into titles like 'POV Drive Through  | Scenic Road Trip'
    which fail the non-negotiable SEO quality rule.
    """

    def test_empty_location_raises(self, generator):
        import pytest as _pytest

        with _pytest.raises(ValueError, match="(?i)location"):
            generator.generate(
                location="",
                publish_day="Wednesday",
                video_duration=1200.0,
                project_name="test_lf01",
            )

    def test_whitespace_only_location_raises(self, generator):
        import pytest as _pytest

        with _pytest.raises(ValueError, match="(?i)location"):
            generator.generate(
                location="   ",
                publish_day="Wednesday",
                video_duration=1200.0,
                project_name="test_lf01",
            )

    def test_literal_unknown_location_raises(self, generator):
        import pytest as _pytest

        for bad in ("UNKNOWN", "unknown", "LOCATION_PLACEHOLDER", "TODO", "TBD"):
            with _pytest.raises(ValueError, match="(?i)placeholder|unknown"):
                generator.generate(
                    location=bad,
                    publish_day="Wednesday",
                    video_duration=1200.0,
                    project_name="test_lf01",
                )

    def test_none_location_raises(self, generator):
        import pytest as _pytest

        with _pytest.raises(ValueError, match="(?i)location"):
            generator.generate(
                location=None,  # type: ignore[arg-type]
                publish_day="Wednesday",
                video_duration=1200.0,
                project_name="test_lf01",
            )
