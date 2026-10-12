"""--list-lfs in full_drive: segments carry no gps_center (A3 scoring skipped),
so labels must resolve from the GPX points covering the LF span."""

from types import SimpleNamespace

import gomoveshift.pipeline.drive_phases.phase_bc_day_first as phase_bc
import gomoveshift.pipeline.gps_enrichment as gps_enrichment
from gomoveshift.pipeline.drive_phases import lf_listing

_PT = {"lat": 51.45, "lon": -9.82}


class _Geo:
    def reverse_geocode_detailed(self, lat, lon):
        return {"county": "County Cork", "state": "", "country": "Ireland"}


def _run(monkeypatch, capsys, points_by_lf):
    lfs = [
        SimpleNamespace(
            index=1, segments=[SimpleNamespace(gps_center_lat=0.0, gps_center_lon=0.0)]
        ),
        SimpleNamespace(
            index=2, segments=[SimpleNamespace(gps_center_lat=0.0, gps_center_lon=0.0)]
        ),
    ]
    plan = SimpleNamespace(regular_lfs=lfs)
    monkeypatch.setattr(phase_bc, "_build_drive_plan_from_state", lambda c, s: plan)
    monkeypatch.setattr(gps_enrichment, "_build_gps_points_by_lf", lambda p, g: points_by_lf)
    monkeypatch.setattr(gps_enrichment, "_build_shared_geocoder", lambda *a, **k: _Geo())
    lf_listing.print_lf_listing(object(), SimpleNamespace(gps_per_episode=[]))
    return capsys.readouterr().out


def test_lf_without_segment_gps_labelled_from_span_fixes(monkeypatch, capsys):
    out = _run(monkeypatch, capsys, {1: [_PT] * 6})
    line1 = next(ln for ln in out.splitlines() if ln.strip().startswith("1 "))
    assert "County Cork, Ireland" in line1
    assert "no GPS" not in line1


def test_truly_gpsless_lf_keeps_override_text(monkeypatch, capsys):
    out = _run(monkeypatch, capsys, {1: [_PT] * 6})
    line2 = next(ln for ln in out.splitlines() if ln.strip().startswith("2 "))
    assert "<no GPS — override required>" in line2
