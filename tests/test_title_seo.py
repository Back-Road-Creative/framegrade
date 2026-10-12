"""Route-only, SEO-optimized YouTube title shape.

DECIDED shape: "{A} to {B}, {ST} in 4K" — route + "in 4K", NO genre word
("Scenic Drive" removed: not all footage is scenic drives). The route keeps
its region tail (state/country suffix); the via-road is demoted to tags/
description, and the trim-honesty verb moved OUT of the title into the
description's 🎬 Final cut line. Clamped to the 100-char YouTube limit.
"""

from __future__ import annotations

import pytest

from gomoveshift.models import ResolvedLocation
from gomoveshift.pipeline.longform_metadata import (
    LongFormMetadataGenerator,
    _clamp_title_len,
    _front_loaded_title,
)
from gomoveshift.pipeline.title_builder import ensure_unique_titles


class TestFrontLoadedShape:
    def test_us_two_town_drive(self) -> None:
        out = _front_loaded_title("Asheville to Boone via US-221, NC")
        assert out == "Asheville to Boone, NC in 4K"
        assert len(out) <= 100

    def test_non_us_two_town_drive_region_kept_in_tail(self) -> None:
        out = _front_loaded_title("Killarney to Dingle via N86, Kerry")
        # Route-only: region "Kerry" stays in the route tail (no genre word).
        assert out == "Killarney to Dingle, Kerry in 4K"

    def test_us_single_region_label(self) -> None:
        out = _front_loaded_title("Colleton County, SC")
        assert out == "Colleton County, SC in 4K"

    def test_via_road_dropped(self) -> None:
        out = _front_loaded_title("Asheville to Boone via The Long Scenic Parkway, NC")
        assert "via" not in out and "Parkway" not in out

    def test_single_part_location(self) -> None:
        out = _front_loaded_title("Some Lone Road")
        assert out == "Some Lone Road in 4K"

    def test_empty_location_degrades_gracefully(self) -> None:
        # No route/region → minimal factual token, never "Scenic Drive".
        assert _front_loaded_title("") == "4K POV"
        assert _front_loaded_title(None) == "4K POV"  # type: ignore[arg-type]

    def test_leading_verb_and_series_suffix_stripped(self) -> None:
        out = _front_loaded_title("Driving Killarney to Dingle via N86, Kerry — Part 2")
        assert out == "Killarney to Dingle, Kerry in 4K"


class TestResolverConsumed:
    def test_resolver_drives_us_verdict(self) -> None:
        # The resolver's US verdict drives the state-abbr normalization in the
        # route tail (route-only: no front-loaded full state name is emitted).
        resolved = ResolvedLocation(dominant_state="North Carolina", is_us=True, country_code="US")
        out = _front_loaded_title("Asheville to Boone via US-221, NC", resolved)
        assert out == "Asheville to Boone, NC in 4K"

    def test_degraded_resolver_falls_back_to_label_parse(self) -> None:
        # Empty/degraded resolver (no verdict) must not flip a non-US drive to US.
        out = _front_loaded_title("Killarney to Dingle via N86, Kerry", ResolvedLocation())
        assert out == "Killarney to Dingle, Kerry in 4K"


class TestPartSuffixSurvivesRebuild:
    """The ``(Part N of M)`` dedup suffix must survive the front-loaded
    rebuild so two same-county LFs keep DISTINCT published titles within the
    100-char clamp. ``_build_title`` is the single site both the SEO-manager and
    basic generators route ``title_override`` (already dedup-suffixed by
    ``ensure_unique_titles`` upstream in ``phase_d_metadata``) through.
    """

    def test_two_same_county_lfs_get_distinct_titles(self) -> None:
        # Two LFs resolving to the same county => identical canonical labels;
        # ensure_unique_titles disambiguates them with (Part N of M) BEFORE the
        # front-loaded rebuild runs (the live phase_d_metadata order).
        canonical = "Asheville to Boone via US-221, NC"
        deduped = ensure_unique_titles({0: canonical, 1: canonical})
        titles = {idx: LongFormMetadataGenerator._build_title(deduped[idx]) for idx in (0, 1)}

        # The dedup suffix survives to the actually-published string ...
        assert titles[0] != titles[1], "same-county LFs collapsed to one title"
        assert titles[0].endswith("(Part 1 of 2)")
        assert titles[1].endswith("(Part 2 of 2)")
        # ... and is NOT parsed into the route noun (clean route-only shape).
        assert titles[0].startswith("Asheville to Boone, NC in 4K")
        assert titles[1].startswith("Asheville to Boone, NC in 4K")
        # ... and both still respect the 100-char YouTube hard limit.
        assert all(len(t) <= 100 for t in titles.values())

    def test_part_suffix_survives_the_100_char_clamp(self) -> None:
        # A label long enough to force a mid-string truncation must still keep
        # its (Part N of M) tail (the clamp preserves it).
        long_label = (
            "Mooresville to Wilkesboro via a Very Long Scenic Highway Name Indeed, NC (Part 2 of 3)"
        )
        out = LongFormMetadataGenerator._build_title(long_label)
        assert out.endswith("(Part 2 of 3)")
        assert len(out) <= 100

    def test_no_part_suffix_is_a_noop(self) -> None:
        # The ordinary (unique-title) path carries no suffix => unchanged shape.
        out = LongFormMetadataGenerator._build_title("Asheville to Boone via US-221, NC")
        assert out == "Asheville to Boone, NC in 4K"
        assert "(Part" not in out


class TestFinalNamespaceDedup:
    """Dedup comparisons must run in the FINAL-title namespace.

    ``ensure_unique_titles`` receives pre-reshape labels ("A to B via US-221,
    NC") while the channel ledger holds assembled final titles ("A to B, NC
    in 4K"); the via-stripping reshape (``_build_title``) runs AFTER dedup.
    Both the channel comparison and the in-batch collision check therefore
    have to compare the labels' assembled final forms, not the raw labels —
    otherwise a byte-identical channel title ships.
    """

    def test_channel_dedup_fires_on_reshaped_final_title(self) -> None:
        # (i) The label never string-matches the channel's final title, but
        # its assembled form does — exact match is a hard failure,
        # not a silent "(Part N)" rename shipping a byte-identical title.
        from gomoveshift.pipeline.exceptions import PhaseBCExitContractError

        with pytest.raises(PhaseBCExitContractError, match="Asheville to Boone"):
            ensure_unique_titles(
                {0: "Asheville to Boone via US-221, NC"},
                published_titles={"Asheville to Boone, NC in 4K"},
            )

    def test_in_batch_via_only_difference_does_not_collide_post_reshape(self) -> None:
        # (ii) Two same-day labels differing only in via-road collapse to ONE
        # identical final title after via-stripping unless the in-batch
        # collision check keys on the assembled form.
        result = ensure_unique_titles(
            {
                0: "Boone to Blowing Rock via US-221, NC",
                1: "Boone to Blowing Rock via NC-105, NC",
            }
        )
        finals = {i: LongFormMetadataGenerator._build_title(t) for i, t in result.items()}
        assert finals[0].strip().lower() != finals[1].strip().lower(), (
            "via-only labels collapsed to one identical final title"
        )

    def test_bare_part_suffix_detaches_through_build_title(self) -> None:
        # (iii) The channel-dedup bump loop emits bare "(Part N)"; it must
        # detach before the front-loaded rebuild and re-append at the tail,
        # exactly like "(Part N of M)" — never fold into the route noun.
        out = LongFormMetadataGenerator._build_title("Asheville to Boone via US-221, NC (Part 2)")
        assert out == "Asheville to Boone, NC in 4K (Part 2)"

    def test_bare_part_suffix_survives_the_100_char_clamp(self) -> None:
        # (iii) _clamp_title_len must preserve a bare "(Part N)" tail inside
        # the 100-char budget instead of truncating it away.
        long_title = ("Wordy " * 18).strip() + " Valley in 4K (Part 2)"
        assert len(long_title) > 100
        clamped = _clamp_title_len(long_title)
        assert clamped.endswith("(Part 2)")
        assert len(clamped) <= 100
