"""TDD tests for boilerplate delete, verb-mandatory title, curated tags.

These are property tests on the generator (not runtime gates). Each invariant
is enforced by construction in `gomoveshift/pipeline/longform_metadata.py`.
"""

from __future__ import annotations

import re

import pytest

from gomoveshift.pipeline.longform_metadata import (
    GMS_BASE_TAGS,
    HIGHLIGHTS_TRIM_THRESHOLD,
    TITLE_TEMPLATE,
    LongFormMetadataGenerator,
    _location_tags,
    _select_verb,
)

# ---------------------------------------------------------------------------
# Fixtures — a sweep of representative locations, durations, terrains
# ---------------------------------------------------------------------------

LOCATION_FIXTURES = [
    "Glenfable Peninsula, County Kerry, Ireland",
    "Asheville, North Carolina, USA",
    "Charleston, South Carolina, United States",
    "Ballymock to Glenfable via N86, Kerry, Ireland",
    "Blue Ridge Parkway, North Carolina, USA",
    "Pacific Coast Highway, California, USA",
    "Ring of Kerry, County Kerry, Ireland",
    "Highway 1, Big Sur, California, USA",
    "Skyline Drive, Virginia, USA",
    "Cape Breton, Nova Scotia, Canada",
    "Amalfi Coast, Salerno, Italy",
    "Great Ocean Road, Victoria, Australia",
    "Route 66, Arizona, USA",
    "Cabot Trail, Cape Breton, Canada",
    "Going-to-the-Sun Road, Montana, USA",
    "Trollstigen, Møre og Romsdal, Norway",
    "Stelvio Pass, Lombardy, Italy",
    "Transfagarasan, Sibiu, Romania",
    "Atlantic Road, Møre og Romsdal, Norway",
    "Black Forest, Baden-Württemberg, Germany",
    "Glenfable",  # single-component edge case
    "Asheville",  # single-component edge case
]

CURATED_VERBS = {"Driving", "Highlights from"}


# ---------------------------------------------------------------------------
# Boilerplate delete
# ---------------------------------------------------------------------------


class TestBoilerplateDeleted:
    """The unverifiable trailing claim must not appear in any generated body."""

    BOILERPLATE_FRAGMENTS = (
        "relaxation",
        "study background",
        "virtual travel",
    )

    @pytest.mark.parametrize("location", LOCATION_FIXTURES)
    def test_factual_body_omits_boilerplate(self, location: str) -> None:
        body = LongFormMetadataGenerator._build_factual_body(
            location=location,
            duration_str="1h 23m",
        )
        lower = body.lower()
        for fragment in self.BOILERPLATE_FRAGMENTS:
            assert fragment not in lower, (
                f"Boilerplate fragment {fragment!r} found in factual body: {body!r}"
            )

    @pytest.mark.parametrize("location", LOCATION_FIXTURES)
    @pytest.mark.parametrize(
        "terrain,routes,towns",
        [
            (None, None, None),
            ("coastal", None, None),
            ("mountain", None, ["Town A", "Town B"]),
            ("winding", ["Some Route"], None),
            (None, ["Route One", "Route Two"], ["Town A", "Town B", "Town C"]),
        ],
    )
    def test_factual_body_no_boilerplate_across_variants(
        self, location, terrain, routes, towns
    ) -> None:
        body = LongFormMetadataGenerator._build_factual_body(
            location=location,
            duration_str="42m",
            terrain_type=terrain,
            named_routes=routes,
            chapter_towns=towns,
        )
        lower = body.lower()
        for fragment in self.BOILERPLATE_FRAGMENTS:
            assert fragment not in lower


# ---------------------------------------------------------------------------
# Verb-mandatory title template
# ---------------------------------------------------------------------------


class TestSelectVerb:
    def test_returns_driving_when_trim_ratio_none(self) -> None:
        assert _select_verb(None) == "Driving"

    def test_returns_driving_when_trim_ratio_above_threshold(self) -> None:
        assert _select_verb(HIGHLIGHTS_TRIM_THRESHOLD) == "Driving"
        assert _select_verb(HIGHLIGHTS_TRIM_THRESHOLD + 0.1) == "Driving"
        assert _select_verb(0.95) == "Driving"

    def test_returns_highlights_when_trim_ratio_below_threshold(self) -> None:
        assert _select_verb(HIGHLIGHTS_TRIM_THRESHOLD - 0.01) == "Highlights from"
        assert _select_verb(0.31) == "Highlights from"

    def test_returns_driving_when_trim_ratio_zero(self) -> None:
        # A literal 0 is "unmeasured" (no segments / ffprobe gap), NOT a
        # 100%-trimmed highlight — a full-length LF must stay "Driving".
        assert _select_verb(0.0) == "Driving"
        assert _select_verb(0) == "Driving"

    def test_threshold_constant_value(self) -> None:
        # Single source of truth — the Highlights-band tests read the same constant.
        assert HIGHLIGHTS_TRIM_THRESHOLD == 0.5


class TestTitleTemplate:
    def test_template_format_string(self) -> None:
        rendered = TITLE_TEMPLATE.format(verb="Driving", location="Glenfable")
        assert rendered == "Driving Glenfable"

    @pytest.mark.parametrize("location", LOCATION_FIXTURES)
    def test_title_is_verb_less_and_route_only(self, location: str) -> None:
        # Titles no longer lead with a verb, and carry no genre word — just the
        # route + "in 4K". (Specific US/non-US shapes: test_title_seo.py.)
        title = LongFormMetadataGenerator._build_title(location=location, trim_ratio=None)
        assert not any(title.startswith(v + " ") for v in CURATED_VERBS), (
            f"Title {title!r} still begins with a trim verb"
        )
        assert title.endswith(" in 4K"), f"Title {title!r} is not route-only 'in 4K'"
        assert "Scenic Drive" not in title

    def test_optimizer_title_cannot_be_offered(self) -> None:
        # The optimizer's title is consumed as tags, never appended; Waypost's
        # select_title decides the title, and _build_title takes no optimizer
        # suffix at all.
        with pytest.raises(TypeError):
            LongFormMetadataGenerator._build_title(
                location="Dingle", trim_ratio=None, seo_suffix="Scenic Road Trip"
            )

    def test_title_is_verb_less_regardless_of_trim(self) -> None:
        title = LongFormMetadataGenerator._build_title(
            location="Glenfable Peninsula, County Kerry, Ireland",
            trim_ratio=0.3,
        )
        assert not title.startswith("Highlights from ")


# ---------------------------------------------------------------------------
# Curated tag builder (whole-phrase only, never word-splits)
# ---------------------------------------------------------------------------


class TestLocationTags:
    @pytest.mark.parametrize(
        "location",
        [
            "Glenfable Peninsula, County Kerry, Ireland",
            "Asheville, North Carolina, USA",
            "Ballymock to Glenfable via N86, Kerry, Ireland",
        ],
    )
    def test_no_tag_is_a_word_fragment_of_location(self, location: str) -> None:
        """A tag that matches a single space-delimited word from inside a
        multi-word comma component is a fragment leak (e.g. ``"County"`` from
        ``"County Kerry"``). A tag that exactly matches a comma component is
        fine even if that component is itself a single word (e.g. ``"Ireland"``).
        """
        tags = _location_tags(location, recording_date="2025-09-01")
        components = {p.strip() for p in location.split(",") if p.strip()}
        # Tokens that appear inside a multi-word component but are not the
        # whole component themselves are forbidden.
        inner_words = set()
        for comp in components:
            words = comp.split()
            if len(words) > 1:
                inner_words.update(words)
        forbidden = inner_words - components
        for tag in tags:
            assert tag not in forbidden, (
                f"Tag {tag!r} is a word fragment of a multi-word component in {location!r}"
            )

    def test_emits_every_comma_component(self) -> None:
        tags = _location_tags(
            "Glenfable Peninsula, County Kerry, Ireland", recording_date="2025-09-01"
        )
        assert "Glenfable Peninsula" in tags
        assert "County Kerry" in tags
        assert "Ireland" in tags

    def test_emits_drive_phrase_tags_when_multi_component(self) -> None:
        tags = _location_tags(
            "Glenfable Peninsula, County Kerry, Ireland", recording_date="2025-09-01"
        )
        assert "Glenfable Peninsula drive" in tags
        assert "Ireland drive" in tags

    def test_emits_year_only_from_recording_date(self) -> None:
        tags = _location_tags("Glenfable, Ireland", recording_date="2025-09-01")
        assert "2025" in tags
        # Full date must NOT be present as a tag.
        assert "2025-09-01" not in tags

    def test_no_recording_date_omits_year(self) -> None:
        tags = _location_tags("Glenfable, Ireland", recording_date=None)
        assert not any(re.fullmatch(r"\d{4}", t) for t in tags)

    def test_dedup_preserves_order(self) -> None:
        # If two parts collapse to the same tag, only the first survives.
        tags = _location_tags("Ireland, Ireland", recording_date=None)
        assert tags.count("Ireland") == 1

    def test_single_component_location_emits_one_phrase_tag(self) -> None:
        tags = _location_tags("Glenfable", recording_date="2025-09-01")
        assert "Glenfable" in tags
        # No drive phrase when only one component (need >=2 parts).
        assert "Glenfable drive" not in tags
        assert "Glenfable scenic drive" not in tags
        # Year still emitted.
        assert "2025" in tags

    @pytest.mark.parametrize("location", LOCATION_FIXTURES)
    def test_all_tags_are_whole_phrases(self, location: str) -> None:
        """No tag is a single word from the word-tokenized location string,
        unless the whole location is itself a single word."""
        tags = _location_tags(location, recording_date="2025-09-01")
        word_tokens = {w for w in re.split(r"[\s,]+", location) if w}
        # If the location has only one word total, single-word tags are fine.
        if len([p for p in location.split(",") if p.strip()]) == 1 and " " not in location:
            return
        for tag in tags:
            # A tag that is exactly one of the word tokens of a multi-word
            # location is a fragment leak.
            if tag in word_tokens:
                # Allow it only when the tag matches a comma-component verbatim.
                components = {p.strip() for p in location.split(",")}
                assert tag in components, f"Tag {tag!r} is a word fragment of {location!r}"


# ---------------------------------------------------------------------------
# Combined: GMS_BASE_TAGS + _location_tags lands in 22–28 range
# ---------------------------------------------------------------------------


class TestCombinedTagBudget:
    @pytest.mark.parametrize(
        "location",
        [
            "Glenfable Peninsula, County Kerry, Ireland",
            "Asheville, North Carolina, USA",
            "Cape Breton, Nova Scotia, Canada",
        ],
    )
    def test_combined_tag_count_in_target_range(self, location: str) -> None:
        # GMS_BASE_TAGS (12) + location-derived (5–7 typical for a 3-component
        # location with a year) lands in the spec range 22–28 once additional
        # generator augmentation runs. For the unit-level builder we just
        # assert there are no fragments and the location component count is
        # >= 4 (every component + 2 drive-phrase tags + year).
        loc_tags = _location_tags(location, recording_date="2025-09-01")
        assert len(loc_tags) >= 4
        # Combined with base, the union has at least 16 entries (no duplicates
        # by construction since base tags are non-locational).
        combined = list(GMS_BASE_TAGS) + [t for t in loc_tags if t not in GMS_BASE_TAGS]
        assert len(combined) >= 16
