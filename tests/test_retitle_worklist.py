"""Tests for ``tools/retitle_worklist.py``."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools import retitle_worklist as rw  # noqa: E402

PA_DUP_TITLE = "Pennsylvania Scenic Drive in 4K | Millbrook, PA"
NC_ENDPOINTS_TITLE = "North Carolina Scenic Drive in 4K | Alderton to Brookdale, NC"

FIXTURE_VIDEOS = [
    {"id": "dup1", "title": PA_DUP_TITLE},
    {"id": "dup2", "title": PA_DUP_TITLE},
    {"id": "highlights1", "title": "Highlights from the North Carolina Mountains"},
    {"id": "county1", "title": "Marlow County, NC Scenic Drive"},
    {"id": "irish1", "title": "To Ballymore, County Cork in 4K"},
    {
        "id": "endpoints",
        "title": NC_ENDPOINTS_TITLE,
        "origin": "Alderton",
        "destination": "Brookdale",
        "region": "NC",
    },
    {"id": "endpoints2", "title": NC_ENDPOINTS_TITLE},
    {"id": "place1", "title": PA_DUP_TITLE, "place": "Millbrook", "region": "PA"},
    {"id": "clean1", "title": "Virginia Scenic Drive in 4K | Ashford to Coventry, VA"},
]

RSS_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry><title>To Seal Head</title><link href="https://youtube.com/watch?v=short1"/></entry>
  <entry><title>Slow Drive Through To Ballymore</title><link href="https://youtube.com/watch?v=short2"/></entry>
  <entry><title>Driving Solo, Pine County Loop</title><link href="https://youtube.com/watch?v=clean2"/></entry>
</feed>
"""


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


def _yaml_fixture(tmp_path: Path) -> Path:
    return _write(tmp_path, "videos.yaml", yaml.safe_dump({"videos": FIXTURE_VIDEOS}))


def test_reason_predicates():
    assert rw.is_highlights_from("Highlights from the North Carolina Mountains")
    assert not rw.is_highlights_from("Alderton to Brookdale, NC")
    assert rw.is_county_only_title("Marlow County, NC Scenic Drive")
    assert not rw.is_county_only_title("To Ballymore, County Cork in 4K")
    assert not rw.is_county_only_title("Alderton to Brookdale, NC")
    assert rw.has_preposition_led_label("To Ballymore, County Cork in 4K")
    assert rw.has_preposition_led_label("Slow Drive Through To Ballymore")
    assert rw.has_preposition_led_label("#ToBallymore")
    assert not rw.has_preposition_led_label("Ashford to Coventry, VA")


def test_suggested_title_format():
    endpoints = {"origin": "Alderton", "destination": "Brookdale", "region": "NC"}
    assert rw.build_suggested_title(endpoints) == "Alderton to Brookdale, NC"
    assert rw.build_suggested_title({"place": "Millbrook", "region": "PA"}) == "Millbrook, PA"
    assert rw.build_suggested_title({"title": "Ashford to Coventry, VA"}) == "needs-human"


def test_build_rows_end_to_end(tmp_path):
    """--no-compose (compose=False): byte-identical to the plain behaviour."""
    videos = rw.load_videos(_yaml_fixture(tmp_path))
    rows = rw.build_rows(videos, [], compose=False)
    by_id_reason = {(r["video_id"], r["reason"]) for r in rows}
    assert ("dup1", "duplicate-title") in by_id_reason
    assert ("dup2", "duplicate-title") in by_id_reason
    assert ("highlights1", "highlights-from") in by_id_reason
    assert ("county1", "county-only-title") in by_id_reason
    assert ("irish1", "preposition-led-label") in by_id_reason
    assert ("irish1", "county-only-title") not in by_id_reason  # "County Cork" != class-B
    assert not any(vid == "clean1" for vid, _ in by_id_reason)
    endpoints_row = next(r for r in rows if r["video_id"] == "endpoints")
    assert endpoints_row["suggested_title"] == "Alderton to Brookdale, NC"
    place_row = next(r for r in rows if r["video_id"] == "place1")
    assert place_row["suggested_title"] == "Millbrook, PA"
    dup1_row = next(r for r in rows if r["video_id"] == "dup1")
    assert dup1_row["suggested_title"] == "needs-human"
    assert dup1_row["platform"] == "youtube"


def test_compose_default_composes_duplicate_and_label_rows(tmp_path):
    """compose=True (the default): duplicate/highlights/preposition rows get
    a real, gate-passing suggestion pulled from their own title text."""
    videos = rw.load_videos(_yaml_fixture(tmp_path))
    rows = rw.build_rows(videos, [])
    by_id: dict[str, dict[str, str]] = {}
    for r in rows:
        by_id.setdefault(r["video_id"], {})[r["reason"]] = r["suggested_title"]

    dup1 = by_id["dup1"]["duplicate-title"]
    dup2 = by_id["dup2"]["duplicate-title"]
    assert dup1 != "needs-human" and dup2 != "needs-human"
    assert dup1 != dup2  # disambiguated, never two identical retitles
    assert dup1.startswith("Millbrook, PA (Part ")
    assert dup2.startswith("Millbrook, PA (Part ")

    highlights = by_id["highlights1"]["highlights-from"]
    assert highlights not in ("needs-human", "")
    assert "scenic drive" not in highlights.lower()

    irish = by_id["irish1"]["preposition-led-label"]
    assert irish == "Ballymore, County Cork"

    # No route/place text survives stripping "Marlow County, NC Scenic
    # Drive" -- insufficient data stays needs-human, never invented.
    assert by_id["county1"]["county-only-title"] == "needs-human"


def test_no_compose_flag_matches_plain_build_rows(tmp_path):
    yaml_path = _yaml_fixture(tmp_path)
    out_a = tmp_path / "a.csv"
    out_b = tmp_path / "b.csv"
    assert rw.main(["--videos-yaml", str(yaml_path), "--out", str(out_a), "--no-compose"]) == 0
    rw.write_csv(rw.build_rows(rw.load_videos(yaml_path), [], compose=False), out_b)
    assert out_a.read_text() == out_b.read_text()


def test_rss_entries_detected_and_deduplicated(tmp_path):
    yaml_path = _yaml_fixture(tmp_path)
    rss_path = _write(tmp_path, "feed.xml", RSS_FIXTURE)
    videos = rw.load_videos(yaml_path)
    rss_entries = rw.load_rss_entries(rss_path)
    rows = rw.build_rows(videos, rss_entries)
    by_id_reason = {(r["video_id"], r["reason"]) for r in rows}
    assert ("short1", "preposition-led-label") in by_id_reason
    assert ("short2", "preposition-led-label") in by_id_reason
    assert not any(vid == "clean2" for vid, _ in by_id_reason)
    ids_in_yaml = {v["id"] for v in videos}
    assert {e["id"] for e in rss_entries} - ids_in_yaml == {"short1", "short2", "clean2"}


def test_write_csv_and_cli(tmp_path):
    yaml_path = _yaml_fixture(tmp_path)
    videos = rw.load_videos(yaml_path)
    rows = rw.build_rows(videos, [])
    out_path = tmp_path / "out" / "worklist.csv"
    rw.write_csv(rows, out_path)
    with out_path.open() as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == rw.FIELDNAMES
        assert len(list(reader)) == len(rows)
    cli_out = tmp_path / "cli.csv"
    assert rw.main(["--videos-yaml", str(yaml_path), "--out", str(cli_out)]) == 0
    assert cli_out.exists()


# --- rewrite_description_place: swap the wrong place out of a description ----


def test_rewrite_description_place_swaps_text_and_place_hashtags():
    old = (
        "Rolling through Alder County 🚗\n\n🚗 Alder County\n\n"
        "Tag someone who needs this drive 👇\n\n"
        "#POVDriving #ScenicDrive #RoadTrip #AlderCounty #NorthCarolina"
    )
    assert rw.rewrite_description_place(old, "Alder County", "Birch County, Virginia") == (
        "Rolling through Birch County, Virginia 🚗\n\n🚗 Birch County, Virginia\n\n"
        "Tag someone who needs this drive 👇\n\n"
        "#POVDriving #ScenicDrive #RoadTrip #BirchCounty #Virginia"
    )


def test_rewrite_description_place_keeps_text_after_the_hashtags():
    old = (
        "📍 Elmsford to Fort Harlow via I-99\n\n#POVDriving #ScenicDrive #RoadTrip "
        "#ElmsfordtoFortHarlowviaI99 #VA Filmed January 2001.\n\n📹 Jan 15, 2001"
    )
    new = rw.rewrite_description_place(
        old, "Elmsford to Fort Harlow via I-99", "Willow Ford, Virginia"
    )
    assert new == (
        "📍 Willow Ford, Virginia\n\n#POVDriving #ScenicDrive #RoadTrip "
        "#WillowFord #Virginia Filmed January 2001.\n\n📹 Jan 15, 2001"
    )


def test_rewrite_description_place_matches_whole_words_only():
    old = "Millbrook roads, not Millbrookh.\n\n#POVDriving #Millbrook #PAScenicDrive #MountainRoads"
    new = rw.rewrite_description_place(old, "Millbrook", "Oakfield, Pennsylvania")
    assert new == (
        "Oakfield, Pennsylvania roads, not Millbrookh.\n\n"
        "#POVDriving #MountainRoads #Oakfield #Pennsylvania"
    )


def test_rewrite_description_place_refuses_when_old_place_is_absent():
    old = "No talking, just driving\n\n#POVDriving #RoadTrip"
    assert rw.rewrite_description_place(old, "Alder County", "Birch County, Virginia") is None


def test_rewrite_description_place_appends_tags_when_description_has_none():
    new = rw.rewrite_description_place("Driving Johnson City", "Johnson City", "Abingdon, Virginia")
    assert new == "Driving Abingdon, Virginia\n\n#Abingdon #Virginia"
