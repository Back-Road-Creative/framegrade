"""--label-dry-run verification harness.

Resolves each LF's corrected dominant-county label with the live reducer
and prints it (plus the county vote breakdown) without encoding — so an
operator can confirm labels after a mislabel report before re-running.
"""

from types import SimpleNamespace

import gomoveshift.pipeline.gps_enrichment as gps_enrichment
import gomoveshift.pipeline.drive_phases.phase_bc_day_first as phase_bc
from gomoveshift.cli import _build_parser
from gomoveshift.pipeline.drive_phases import lf_listing

_FRIPP = (32.3095, -80.6332)
_EDISTO = (32.6591, -80.6646)


class _Geo:
    def reverse_geocode_detailed(self, lat, lon):
        if abs(lat - _FRIPP[0]) < 0.2:
            return {
                "county": "Beaufort County",
                "state": "South Carolina",
                "hamlet": "Fort Fremont",
            }
        return {"county": "Colleton County", "state": "South Carolina"}


def test_parser_accepts_label_dry_run():
    args = _build_parser().parse_args(["drive", "/tmp/src", "--label-dry-run"])
    assert args.label_dry_run is True


def test_parser_label_dry_run_defaults_false():
    args = _build_parser().parse_args(["drive", "/tmp/src"])
    assert args.label_dry_run is False


def test_print_label_dry_run_shows_dominant_county(monkeypatch, capsys):
    lf = SimpleNamespace(index=1, segments=[])
    drive_plan = SimpleNamespace(regular_lfs=[lf])
    pts = [{"lat": _FRIPP[0], "lon": _FRIPP[1]}] * 8 + [{"lat": _EDISTO[0], "lon": _EDISTO[1]}] * 2

    monkeypatch.setattr(phase_bc, "_build_drive_plan_from_state", lambda c, s: drive_plan)
    monkeypatch.setattr(gps_enrichment, "_build_gps_points_by_lf", lambda p, g: {1: pts})
    monkeypatch.setattr(gps_enrichment, "_build_shared_geocoder", lambda *a, **k: _Geo())

    lf_listing.print_label_dry_run(config=object(), state=SimpleNamespace(gps_per_episode=[]))
    out = capsys.readouterr().out

    assert "Beaufort County" in out
    assert "Colleton" in out  # appears in the vote breakdown (minority)
    assert "NO encode" in out
    # The resolved label line names Beaufort, not Colleton.
    label_line = next(ln for ln in out.splitlines() if ln.strip().startswith("1 "))
    assert "Beaufort County" in label_line
    assert "Colleton" not in label_line


class _GeoRoad:
    """Geocoder whose samples all sit in one county on a single via-road, so the
    dominant-county reducer resolves a road + a road-vote tally."""

    def reverse_geocode_detailed(self, lat, lon):
        return {"county": "Ashe County", "state": "North Carolina", "road": "US 64"}


def test_print_label_dry_run_shows_road_and_road_vote(monkeypatch, capsys):
    # The ground-truth harness must surface the resolved ROAD and its vote
    # tally per LF, not just the county — otherwise via-road behaviour can't be
    # verified pre-encode.
    lf = SimpleNamespace(index=2, segments=[])
    drive_plan = SimpleNamespace(regular_lfs=[lf])
    pts = [{"lat": 36.3 + i * 0.01, "lon": -81.5 - i * 0.01} for i in range(6)]

    monkeypatch.setattr(phase_bc, "_build_drive_plan_from_state", lambda c, s: drive_plan)
    monkeypatch.setattr(gps_enrichment, "_build_gps_points_by_lf", lambda p, g: {2: pts})
    monkeypatch.setattr(gps_enrichment, "_build_shared_geocoder", lambda *a, **k: _GeoRoad())

    lf_listing.print_label_dry_run(config=object(), state=SimpleNamespace(gps_per_episode=[]))
    out = capsys.readouterr().out

    assert "via road: US-64" in out
    assert "road vote: US-64=" in out


def test_print_label_dry_run_no_lfs(monkeypatch, capsys):
    monkeypatch.setattr(phase_bc, "_build_drive_plan_from_state", lambda c, s: None)
    lf_listing.print_label_dry_run(config=object(), state=SimpleNamespace(gps_per_episode=[]))
    assert "nothing to list" in capsys.readouterr().out
