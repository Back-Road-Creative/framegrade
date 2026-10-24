"""Dry-run outputs name the county each insert photo was taken in, and which
LF labels accept it, so an operator sees a county refusal before any encode."""

from types import SimpleNamespace

import gomoveshift.pipeline.drive_phases.phase_bc_day_first as phase_bc
import gomoveshift.pipeline.gps_enrichment as gps_enrichment
from gomoveshift.pipeline import inserts as ins_mod
from gomoveshift.pipeline.drive_phases import lf_listing
from gomoveshift.pipeline.inserts import describe_drive_inserts, describe_insert_label_match

_COLLETON = (32.9, -80.7)
_BEAUFORT = (32.3, -80.6)


def _county_of(lat, lon):
    if abs(lat - _COLLETON[0]) < 0.1:
        return "Colleton County", "South Carolina"
    if abs(lat - _BEAUFORT[0]) < 0.1:
        return "Beaufort County", "SC"
    return None, None


def _photos(tmp_path, monkeypatch):
    d = tmp_path / "photos"
    d.mkdir()
    for name in ("a_colleton.jpg", "b_beaufort.jpg", "c_nogps.jpg", "d_ocean.jpg"):
        (d / name).write_bytes(b"x")
    fixes = {
        "a_colleton.jpg": _COLLETON,
        "b_beaufort.jpg": _BEAUFORT,
        "d_ocean.jpg": (0.0, 0.0),
    }
    monkeypatch.setattr(ins_mod, "_photo_gps", lambda p: fixes.get(p.name))
    return SimpleNamespace(insert_photos=str(d), insert_clips=[])


def test_describe_drive_inserts_does_no_exif_or_geocode_work(tmp_path, monkeypatch):
    drive = _photos(tmp_path, monkeypatch)

    def boom(*a, **k):
        raise AssertionError("describe_drive_inserts must stay offline")

    monkeypatch.setattr(ins_mod, "_photo_gps", boom)
    monkeypatch.setattr(ins_mod, "_geocoder_county_of", boom)
    lines = describe_drive_inserts(drive)
    assert len(lines) == 4
    assert not any("County" in ln or "captured in" in ln for ln in lines)


def test_label_match_lists_the_lfs_that_accept_each_photo(tmp_path, monkeypatch):
    drive = _photos(tmp_path, monkeypatch)
    labels = {1: "Colleton County, SC", 2: "Beaufort County, SC"}
    lines = describe_insert_label_match(drive, labels, _county_of)
    text = "\n".join(lines)
    a = next(ln for ln in lines if "a_colleton.jpg" in ln)
    assert "Colleton County, SC" in a and "LF 1" in a and "LF 2" not in a
    b = next(ln for ln in lines if "b_beaufort.jpg" in ln)
    assert "LF 2" in b and "LF 1" not in b
    assert "outside every LF label" not in a
    assert "c_nogps.jpg" in text and "d_ocean.jpg" in text


def test_label_match_flags_a_photo_outside_every_label(tmp_path, monkeypatch):
    drive = _photos(tmp_path, monkeypatch)
    lines = describe_insert_label_match(drive, {1: "Colleton County, SC"}, _county_of)
    b = next(ln for ln in lines if "b_beaufort.jpg" in ln)
    assert "Beaufort County, SC" in b and "outside every LF label" in b


def test_label_match_is_empty_without_inserts():
    drive = SimpleNamespace(insert_photos="", insert_clips=[])
    assert describe_insert_label_match(drive, {1: "A County, SC"}, _county_of) == []


class _Geo:
    def reverse_geocode_detailed(self, lat, lon):
        county, state = _county_of(lat, lon)
        return {"county": county, "state": state} if county else {}


def test_label_dry_run_prints_insert_counties(tmp_path, monkeypatch, capsys):
    drive = _photos(tmp_path, monkeypatch)
    lf = SimpleNamespace(index=1, segments=[])
    drive_plan = SimpleNamespace(regular_lfs=[lf])
    pts = [{"lat": _COLLETON[0], "lon": _COLLETON[1]}] * 5
    monkeypatch.setattr(phase_bc, "_build_drive_plan_from_state", lambda c, s: drive_plan)
    monkeypatch.setattr(gps_enrichment, "_build_gps_points_by_lf", lambda p, g: {1: pts})
    monkeypatch.setattr(gps_enrichment, "_build_shared_geocoder", lambda *a, **k: _Geo())

    config = SimpleNamespace(drive=drive)
    lf_listing.print_label_dry_run(config=config, state=SimpleNamespace(gps_per_episode=[]))
    out = capsys.readouterr().out

    section = out.split("Insert photos vs the dry-run labels above", 1)[1]
    assert "a_colleton.jpg" in section and "Colleton County, SC" in section
    assert "b_beaufort.jpg" in section and "outside every LF label" in section
    assert "--per-lf-location" in section
