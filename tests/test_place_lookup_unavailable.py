"""A place lookup that FAILED measured nothing: the render stops for retry.

A POI source that raised (every Overpass endpoint down) must never let the title
cascade fall to the bare county; a source that answered with nothing keeps today's
fallback. Synthetic places only; the POI source is injected, never the network.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gomoveshift.pipeline import destination_resolver as dr
from gomoveshift.pipeline import gps_enrichment as ge
from gomoveshift.pipeline.drive_phases import lf_listing
from gomoveshift.pipeline.exceptions import PlaceLookupUnavailableError

from test_retained_gps_claims import _BandGeocoder, _episode, _lf

_PEAK = SimpleNamespace(gps_center_lat=9.5, gps_center_lon=20.0, composite_score=0.9)


def _down(_bbox):
    raise RuntimeError("Overpass query failed on every endpoint")


def _lf_with_peaks(index=2):
    lf = _lf(index, (0.0, 280.0))
    lf.montage_peaks = [_PEAK]
    return lf


def _enrich(lf, source, ost):
    state = SimpleNamespace(
        gps_per_episode=_episode(), gpx_files=[], episode_dates=[], per_lf_overrides={}
    )
    with (
        patch.object(dr, "_default_poi_source", source),
        patch.object(ge, "_build_per_lf_gpx_path_map", return_value={}),
        patch.object(ge, "_build_per_lf_gpx_episode_map", return_value={}),
        patch.object(ge, "_journey_endpoint_loss_by_lf", return_value={}),
        patch.object(ge, "_build_shared_geocoder", return_value=_BandGeocoder()),
        patch.object(ge, "_partition_lf_plans_by_date", return_value={"2026-01-02": [lf]}),
        patch.object(ge, "_assert_per_lf_locations_resolvable", return_value=None),
        patch.object(ge, "derive_per_lf_locations", return_value={}),
        patch.object(ge, "canonical_lf_ost_text", ost),
        patch.object(ge, "_differentiate_and_revalidate_siblings", return_value=None),
    ):
        return ge._enrich_per_lf_gps(
            state=state,
            real_plan=SimpleNamespace(regular_lfs=[lf]),
            day_batches=[],
            episode_files=[],
            per_day_locations={},
            drive_wide_location="",
        )


def test_failed_lookup_raises_and_builds_no_title():
    built = []

    def ost(lf, **_kw):
        built.append(lf.index)
        return "Harrow, Westmark"

    with pytest.raises(PlaceLookupUnavailableError) as exc:
        _enrich(_lf_with_peaks(2), _down, ost)
    msg = str(exc.value)
    assert "LF 2" in msg and "every endpoint" in msg
    assert "transient" in msg and "health heal" in msg
    assert built == []


def test_empty_lookup_keeps_the_fallback():
    out = _enrich(_lf_with_peaks(2), lambda _b: [], lambda lf, **_k: "Harrow, Westmark")
    assert out.per_lf_ost_text_by_date["2026-01-02"][2] == "Harrow, Westmark"


def test_error_is_a_drive_failure_not_a_clean_gate_exit():
    # drive_orchestrator treats OstAuditPendingError as a clean exit and
    # ReviewFailedError as a gate verdict; anything else reaching its generic
    # ``except Exception`` lands in ``errors`` (a failed drive, phase not recorded).
    from gomoveshift.pipeline.ost_audit_sidecar import OstAuditPendingError
    from gomoveshift.pipeline.review_gate import ReviewFailedError

    assert issubclass(PlaceLookupUnavailableError, RuntimeError)
    assert not issubclass(PlaceLookupUnavailableError, (OstAuditPendingError, ReviewFailedError))


def _dry_run(tmp_path, capsys, source):
    lf = _lf_with_peaks(1)
    state = SimpleNamespace(
        gps_per_episode=_episode(), per_lf_overrides={}, stages_base=tmp_path / "stages"
    )
    dom = SimpleNamespace(
        county="Ashby County", state="Westmark", name=None, votes={"Ashby County": 3},
        road=None, road_refs=[],
    )  # fmt: skip
    from gomoveshift.pipeline.drive_phases import phase_bc_day_first as phase_bc

    with (
        patch.object(dr, "_default_poi_source", source),
        patch.object(
            phase_bc, "_build_drive_plan_from_state", lambda c, s: SimpleNamespace(regular_lfs=[lf])
        ),
        patch.object(ge, "_build_gps_points_by_lf", lambda p, g: {1: [(0.0, 9.5, 20.0)]}),
        patch.object(ge, "_build_shared_geocoder", lambda *a, **k: _BandGeocoder()),
        patch.object(lf_listing, "_live_title_ids", lambda: {}),
        patch("gomoveshift.pipeline.title_builder.dominant_location", return_value=dom),
        patch.object(ge, "derive_per_lf_locations", return_value={}),
        patch.object(ge, "_differentiate_and_revalidate_siblings", return_value=None),
        patch.object(ge, "canonical_lf_ost_text", return_value="Harrow, Westmark"),
    ):
        lf_listing.print_label_dry_run(config=object(), state=state)
    return capsys.readouterr().out


def test_dry_run_marks_the_lf_instead_of_showing_a_fallback_title(tmp_path, capsys):
    out = _dry_run(tmp_path, capsys, _down)
    assert "place lookup unavailable (transient)" in out


def test_dry_run_is_silent_when_the_lookup_merely_found_nothing(tmp_path, capsys):
    out = _dry_run(tmp_path, capsys, lambda _b: [])
    assert "place lookup unavailable" not in out
