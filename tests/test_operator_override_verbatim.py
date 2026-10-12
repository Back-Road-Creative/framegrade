"""An operator ``--per-lf-location`` override is AUTHORITATIVE.

``derive_per_lf_locations``' contract says "overrides[idx] wins over
centroid-derived labels", but the per-LF OST text was still produced by
``build_lf_title``, whose GPS cascade RESHAPED the override on a GPS-bearing LF
(an explicit override could come out as a different, GPS-derived label). The
override is now burned verbatim; a non-overridden LF still routes through the
GPS cascade.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from gomoveshift.pipeline import gps_enrichment as ge


def _run(overrides):
    lf_gps = SimpleNamespace(index=1, segments=[])
    lf_override = SimpleNamespace(index=2, segments=[])
    state = SimpleNamespace(
        gps_per_episode=None,
        gpx_files=[],
        episode_dates=[],
        per_lf_overrides=overrides,
    )
    with (
        patch.object(
            ge, "_build_gps_points_by_lf", return_value={1: [{"lat": 35.0, "lon": -80.0}]}
        ),
        patch.object(ge, "_build_per_lf_gpx_path_map", return_value={}),
        patch.object(ge, "_build_shared_geocoder", return_value=MagicMock()),
        patch.object(
            ge, "_partition_lf_plans_by_date", return_value={"2026-02-20": [lf_gps, lf_override]}
        ),
        patch.object(ge, "_assert_per_lf_locations_resolvable", return_value=None),
        patch.object(
            ge,
            "derive_per_lf_locations",
            return_value={1: "Jackson County, NC", 2: "Jackson County, NC"},
        ),
        patch.object(ge, "build_lf_title", return_value="GPS RESHAPED TITLE") as mock_build,
        patch.object(ge, "validate_burned_ost_text", side_effect=lambda c, loc, idx: c),
    ):
        result = ge._enrich_per_lf_gps(
            state=state,
            real_plan=SimpleNamespace(),
            day_batches=[],
            episode_files=[],
            per_day_locations={},
            drive_wide_location="Jackson County, NC",
        )
    return result, mock_build


class TestOperatorOverrideVerbatim:
    def test_override_burned_verbatim_not_reshaped(self):
        result, mock_build = _run({2: "Saluda to Lake Toxaway, NC"})
        ost = result.per_lf_ost_text_by_date["2026-02-20"]
        # LF 2 (override) is verbatim — the GPS cascade never touched it.
        assert ost[2] == "Saluda to Lake Toxaway, NC"
        # LF 1 (no override) still goes through build_lf_title.
        assert ost[1] == "GPS RESHAPED TITLE"
        # build_lf_title was invoked exactly once (LF 1), never for the
        # overridden LF 2.
        assert mock_build.call_count == 1

    def test_no_override_uses_cascade(self):
        result, mock_build = _run({})
        ost = result.per_lf_ost_text_by_date["2026-02-20"]
        assert ost[1] == "GPS RESHAPED TITLE"
        assert ost[2] == "GPS RESHAPED TITLE"
        assert mock_build.call_count == 2

    def test_blank_override_falls_through_to_cascade(self):
        result, _ = _run({2: "   "})
        ost = result.per_lf_ost_text_by_date["2026-02-20"]
        assert ost[2] == "GPS RESHAPED TITLE"


class TestDuplicateOverrideLocationVerbatim:
    """Two sibling LFs given the SAME --per-lf-location keep that exact label on
    the location surface — no '(Part N)' suffix. The operator was explicit (two
    parts of one continuous drive); YouTube title uniqueness is handled by
    ensure_unique_titles' '(Part N of M)'. Regression guard: identical overrides
    must not gain a '(Part N)' suffix on the location surface."""

    def test_identical_overrides_not_part_suffixed(self):
        lfs = [
            SimpleNamespace(index=1, segments=[]),
            SimpleNamespace(index=2, segments=[]),
        ]
        label = "Kilgarvan to Ballylickey via the Famine Tunnel"
        result = ge.derive_per_lf_locations(lfs, overrides={1: label, 2: label})
        assert result[1] == label
        assert result[2] == label  # was "<label> (Part 2)" before the fix
        assert "(Part" not in result[2]
