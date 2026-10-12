"""Tests for drive-level OST derivation that reuses per-episode OSTs.

Problem: the old derive_drive_ost_text did its own reverse-geocode of the
first/last GPS coordinates, which landed on rural roads with no town and
fell through to "State to State". The per-episode derivation already figured
out the right town names for each episode — drive-level should reuse them.

New behavior: when episode_locations is provided, derive_drive_ost_text
parses the first parseable 'A to B' OST to extract the drive's start town,
and the last parseable 'A to B' OST to extract the end town.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch


class TestDriveFromEpisodes:
    """derive_drive_ost_text(..., episode_locations=...) parses episode OSTs."""

    def test_start_from_first_episode_end_from_last_episode(self):
        """Drive-level = first 'A to B' left side → last 'A to B' right side."""
        from gomoveshift.pipeline.ingest import derive_drive_ost_text

        episode_locations = [
            "Fort Fremont to Longwood via Seaside Road",
            "Frogmore to Port Royal via Overland Trail",
            "Port Royal to Beaufort via Trask Parkway",
            "Cashiers to Highlands via Highlands Road",
        ]
        result = derive_drive_ost_text(
            [Path("/fake/ep1.gpx"), Path("/fake/ep2.gpx")],
            episode_locations=episode_locations,
        )
        assert result == "Fort Fremont to Highlands"

    def test_skips_single_town_episodes_for_start(self):
        """'Road, Town, ST' shape is not parseable — skip until a real 'A to B'."""
        from gomoveshift.pipeline.ingest import derive_drive_ost_text

        episode_locations = [
            "Seaside Road, Longwood, SC",  # single-town shape — skip
            "Stocks Creek Road, SC (NW)",  # road+state shape — skip
            "Fort Fremont to Longwood via Seaside Road",  # first parseable
            "Cashiers to Highlands via Highlands Road",  # last parseable
        ]
        result = derive_drive_ost_text(
            [Path("/fake/ep1.gpx")] * 4,
            episode_locations=episode_locations,
        )
        assert result == "Fort Fremont to Highlands"

    def test_only_one_parseable_episode_uses_its_endpoints(self):
        """Single parseable episode → use its own start and end."""
        from gomoveshift.pipeline.ingest import derive_drive_ost_text

        episode_locations = [
            "Seaside Road, Longwood, SC",
            "Fort Fremont to Longwood via Seaside Road",
            "Seaside Road, Longwood, SC",
        ]
        result = derive_drive_ost_text(
            [Path("/fake/ep1.gpx")] * 3,
            episode_locations=episode_locations,
        )
        assert result == "Fort Fremont to Longwood"

    def test_same_start_and_end_returns_single_town(self):
        """Start == end → 'Town' (not 'Town to Town')."""
        from gomoveshift.pipeline.ingest import derive_drive_ost_text

        episode_locations = [
            "Longwood to Beaufort via Trask",
            "Beaufort to Longwood via Trask",
        ]
        result = derive_drive_ost_text(
            [Path("/fake/ep1.gpx")] * 2,
            episode_locations=episode_locations,
        )
        # Start = Longwood (from ep1), End = Longwood (from ep2, right side)
        assert result == "Longwood"

    def test_no_parseable_episodes_falls_back_to_geocode(self):
        """When NO episode has an 'A to B' shape, fall through to the existing
        GPX endpoint geocoding path (legacy behavior)."""
        from gomoveshift.pipeline.ingest import derive_drive_ost_text

        # Only non-route OSTs provided
        episode_locations = [
            "Seaside Road, Longwood, SC",
            "Stocks Creek Road, SC",
        ]
        # Empty gpx list means geocoder path also returns empty
        result = derive_drive_ost_text([], episode_locations=episode_locations)
        assert result == ""

    def test_empty_episode_locations_fall_back_to_geocode(self):
        """episode_locations=[] and no gpx_paths → empty."""
        from gomoveshift.pipeline.ingest import derive_drive_ost_text

        result = derive_drive_ost_text([], episode_locations=[])
        assert result == ""

    def test_differentiated_direction_suffix_stripped_for_parsing(self):
        """'OST (N)' direction differentiation suffix is stripped when
        parsing drive-level endpoints so the direction marker doesn't pollute
        the drive-level string."""
        from gomoveshift.pipeline.ingest import derive_drive_ost_text

        episode_locations = [
            "Fort Fremont to Longwood via Seaside Road",
            "Stocks Creek Road, SC (NW)",
            "Stocks Creek Road, SC (SE)",
            "Cashiers to Highlands via Highlands Road",
        ]
        result = derive_drive_ost_text(
            [Path("/fake/ep1.gpx")] * 4,
            episode_locations=episode_locations,
        )
        assert result == "Fort Fremont to Highlands"

    def test_missing_via_clause_still_parses(self):
        """'A to B' without 'via Road' is still parseable as a drive span."""
        from gomoveshift.pipeline.ingest import derive_drive_ost_text

        episode_locations = [
            "Springfield to Shelbyville",
            "Capital City to Ogdenville",
        ]
        result = derive_drive_ost_text(
            [Path("/fake/ep1.gpx")] * 2,
            episode_locations=episode_locations,
        )
        assert result == "Springfield to Ogdenville"
