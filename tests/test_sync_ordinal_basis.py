"""Queue/bracket sync takes a Short's day ordinal from the plan itself, never by parsing the
"Nth stretch" out of the title it is about to replace, and refuses a title the plan's
generator did not write."""

from __future__ import annotations

import json

from gomoveshift.publishing.plan_bracket_sync import generated_title, ordinal_basis


def _short(path, title, *, lf=1, index=1, date="2026-03-05", regions=("Ashby County",)):
    return {
        "lf_index": lf,
        "short_index": index,
        "recording_date": date,
        "media": {"path": str(path)},
        "payloads": {"youtube": {"title": title}},
        "sampled_regions": list(regions),
        "sampled_roads": [],
        "route_record": None,
    }


def test_the_ordinal_is_the_day_sidecars_start_time_order(tmp_path):
    stages = tmp_path / "stages"
    day = stages / "day_2026-03-05"
    (day / "shorts").mkdir(parents=True)
    (stages / "shorts_ost_audit_2026-03-05.json").write_text(
        json.dumps({"shorts": [{"rank": 1, "start_sec": 50}, {"rank": 3, "start_sec": 10}]})
    )
    shorts = [
        _short(day / "shorts" / "short_01.mp4", "Harrow", index=1),
        _short(day / "shorts" / "short_03.mp4", "Ashby County", index=2),
    ]
    assert ordinal_basis(shorts) == [2, 1]  # rank 3 starts first, whatever the titles say


def test_without_a_sidecar_the_title_stamp_is_the_ordinal_if_it_verifies(tmp_path):
    shorts = [
        _short(tmp_path / "a.mp4", "Mar 5, 2nd stretch: 38 seconds, Ashby County #Shorts"),
        _short(tmp_path / "b.mp4", "Harrow", index=2),
    ]
    assert ordinal_basis(shorts) == [2, 2]  # b: no stamp, so its place among the day's Shorts


def test_without_either_the_ordinal_is_the_position_in_the_plans_day(tmp_path):
    shorts = [
        _short(tmp_path / "a.mp4", "Harrow", lf=1, index=1),
        _short(tmp_path / "b.mp4", "Ashby County", lf=1, index=2),
        _short(tmp_path / "c.mp4", "Harrow", lf=2, index=3, date="2026-03-06"),
    ]
    assert ordinal_basis(shorts) == [1, 2, 1]


def test_a_title_the_plan_generated_is_a_stamp_or_one_of_its_measured_places():
    stamped = _short("x.mp4", "Mar 5, 1st stretch: 38 seconds, Ashby County #Shorts")
    place = _short("x.mp4", "Ashby County #Shorts")
    foreign = _short("x.mp4", "Best road in America #Shorts")
    assert generated_title(stamped) and generated_title(place)
    assert not generated_title(foreign)
    label = _short("x.mp4", "Virginia to Pennsylvania #Shorts")
    assert generated_title(label, ("Virginia to Pennsylvania",))
