"""Regression: longform tag builder must strip ` — Part N` series suffix.

A spot-check of a "Part 1" fixture showed that
`_location_tags` split the location by comma but did not strip
the trailing series suffix, leaking tags like `"Ireland — Part 1"` and
`"Ireland — Part 1 drive"` into the long-form YouTube tag list.
The social generator already had a sibling helper, but the longform path
was missed. This test pins the fix.
"""

from __future__ import annotations

from gomoveshift.pipeline.longform_metadata import (
    _location_tags,
    _strip_series_suffix,
)


class TestSharedSeriesUtils:
    """The regex + helper live in one module, re-used by all 3 callers."""

    def test_series_utils_is_single_source(self):
        from gomoveshift.pipeline import _series_utils
        from gomoveshift.pipeline import region_resolver, social_generator

        # All three call sites resolve to the SAME function object.
        assert (
            _series_utils._strip_series_suffix
            is _strip_series_suffix
            is social_generator._strip_series_suffix
            is region_resolver._strip_series_suffix
        )

    def test_canonical_helper_behavior(self):
        from gomoveshift.pipeline._series_utils import _strip_series_suffix as f

        assert f("Ireland — Part 1") == "Ireland"
        assert f("Dingle Peninsula, County Kerry, Ireland — Part 7") == (
            "Dingle Peninsula, County Kerry, Ireland"
        )
        assert f("Ireland") == "Ireland"
        # Internal "Part" preserved; only trailing suffix stripped.
        assert f("Foo — Part 1, Bar — Part 2") == "Foo — Part 1, Bar"


class TestStripSeriesSuffix:
    def test_strips_part_n_suffix(self):
        assert _strip_series_suffix("Ireland — Part 1") == "Ireland"
        assert _strip_series_suffix("Ireland — Part 12") == "Ireland"

    def test_no_suffix_unchanged(self):
        assert _strip_series_suffix("Ireland") == "Ireland"
        assert _strip_series_suffix("Dingle Peninsula, County Kerry, Ireland") == (
            "Dingle Peninsula, County Kerry, Ireland"
        )

    def test_trims_only_trailing(self):
        # Only the last ` — Part N` is stripped, never an internal one.
        assert _strip_series_suffix("Foo — Part 1, Bar — Part 2") == "Foo — Part 1, Bar"


class TestLocationTagsStripsSuffix:
    def test_no_part_n_in_emitted_tags(self):
        tags = _location_tags(
            "Dingle Peninsula, County Kerry, Ireland — Part 1",
            recording_date="2025-09-14",
        )
        assert "Ireland — Part 1" not in tags
        assert "Ireland — Part 1 drive" not in tags

    def test_country_tag_clean(self):
        tags = _location_tags(
            "Dingle Peninsula, County Kerry, Ireland — Part 1",
            recording_date="2025-09-14",
        )
        assert "Ireland" in tags
        assert "Ireland drive" in tags

    def test_no_part_n_in_any_tag_for_series_drive(self):
        # Property: for any input with a ` — Part N` suffix, no emitted tag
        # contains ` — Part`.
        for loc in [
            "Ireland — Part 1",
            "Foo, Bar — Part 9",
            "A, B, C — Part 12",
        ]:
            tags = _location_tags(loc, recording_date="2025-09-14")
            for t in tags:
                assert " — Part" not in t, f"tag {t!r} leaked suffix from {loc!r}"


class TestPartNofMSuffix:
    """The series-suffix leak, reintroduced in a second suffix shape.

    ``title_builder.build_superlong_part_title`` emits ``" — Part N of M"``
    (its own ``_SUPERLONG_PART_SUFFIX_RE`` matches exactly that), but the
    stripper's regex was anchored immediately after the digits, so it strips
    ``" — Part 1"`` and leaves ``" — Part 1 of 2"`` intact. Every consumer of
    ``_strip_series_suffix`` then sees the suffix as part of the location.

    Concrete production impact: ``region_resolver.derive_region`` takes the
    last comma-segment of the stripped string, so a multi-part title yields
    ``"Ireland — Part 1 of 2"``, which matches no country/state/county tail —
    region derivation returns empty for EVERY multi-part superlong title.
    """

    def test_strips_part_n_of_m_suffix(self):
        assert _strip_series_suffix("Ireland — Part 1 of 2") == "Ireland"
        assert _strip_series_suffix("Ireland — Part 12 of 15") == "Ireland"
        assert _strip_series_suffix("Dingle Peninsula, County Kerry, Ireland — Part 2 of 3") == (
            "Dingle Peninsula, County Kerry, Ireland"
        )

    def test_trims_only_trailing_of_m(self):
        # Same contract as the bare ` — Part N` form: internal survives.
        assert _strip_series_suffix("Foo — Part 1 of 2, Bar — Part 2 of 2") == (
            "Foo — Part 1 of 2, Bar"
        )

    def test_region_derives_from_a_multi_part_title(self):
        """The bug's real-world consequence, pinned at the consumer."""
        from gomoveshift.pipeline.region_resolver import derive_region

        assert derive_region("Dingle Peninsula, County Kerry, Ireland — Part 1 of 2") == "ireland"
        assert derive_region("Blue Ridge Parkway, North Carolina — Part 2 of 4") == "usa"

    def test_no_part_leaks_into_tags_for_of_m_form(self):
        for loc in [
            "Ireland — Part 1 of 2",
            "Foo, Bar — Part 9 of 10",
            "A, B, C — Part 12 of 12",
        ]:
            tags = _location_tags(loc, recording_date="2025-09-14")
            for t in tags:
                assert " — Part" not in t, f"tag {t!r} leaked suffix from {loc!r}"

    def test_no_suffix_input_unchanged_behavior(self):
        # No regression for the no-suffix case (the common case for
        # GPS-bearing drives).
        tags = _location_tags(
            "Dingle Peninsula, County Kerry, Ireland",
            recording_date="2025-09-14",
        )
        assert "Ireland" in tags
        assert "Dingle Peninsula" in tags
        assert "County Kerry" in tags
