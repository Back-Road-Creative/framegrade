"""Mutation pins for the clip-road plumbing (synthetic places only).

Each test fails if one link of sidecar -> generator -> plan -> reader drops,
reorders or inherits the clip's own road names.
"""

from __future__ import annotations

import json

import pytest

from test_publishing_plan_builder import _build
from test_publishing_plan_io import _record, _write

from gomoveshift.pipeline.drive_phases.phase_d_scheduling import _per_short_sidecar_roads
from gomoveshift.pipeline.social_generator import SocialGenerator
from gomoveshift.publishing.models import PublishPlan
from gomoveshift.publishing.plan_io import load_publish_plan

REGION = "Ashby County, Ostry"


def _tags(post) -> list[str]:
    return [t.strip() for t in post.youtube_tags.split(",") if t.strip()]


def test_sidecar_roads_follow_each_shorts_own_rank_not_file_order(tmp_path):
    stages = tmp_path / "stages"
    shorts = stages / "day_2026-10-01" / "shorts"
    shorts.mkdir(parents=True)
    rows = [  # file order deliberately differs from rank order
        {"rank": 3, "roads": ["Quillhaven Road"]},
        {"rank": 1, "roads": ["Harrow Lane", "A1"]},
        {"rank": 2},
    ]
    (stages / "shorts_ost_audit_2026-10-01.json").write_text(json.dumps({"shorts": rows}))
    lf_shorts = [shorts / "short_02.mp4", shorts / "short_03.mp4", shorts / "short_01.mp4"]
    assert _per_short_sidecar_roads(lf_shorts) == [[], ["Quillhaven Road"], ["Harrow Lane", "A1"]]


def test_plan_without_the_sampled_roads_key_still_loads_as_empty(tmp_path):
    path = _write(tmp_path, PublishPlan(shorts=(_record(sampled_roads=("Harrow Lane",)),)))
    assert load_publish_plan(path).shorts[0].sampled_roads == ("Harrow Lane",)
    raw = json.loads(path.read_text())
    for rec in raw["shorts"]:
        rec.pop("sampled_roads")
    path.write_text(json.dumps(raw))
    assert load_publish_plan(path).shorts[0].sampled_roads == ()


def test_short_clone_never_inherits_the_parent_lfs_roads():
    parent = SocialGenerator(
        location="Harrow County, Ostry",
        project_name="p_lf01",
        series_part=1,
        roads=["Westmark Road", "Brindlemoor Way"],
    )
    clone = parent._per_short_variant("Pellmere, Ashby County", [REGION])
    assert clone._own_roads == []
    tags = _tags(clone.generate_post(1, None, global_index=1, title_facts=None))
    assert not {"Westmark Road", "Brindlemoor Way"} & set(tags)


def test_generated_short_roads_reach_the_publish_plan_record():
    gen = SocialGenerator(location="Harrow County, Ostry", project_name="p_lf01", series_part=1)
    posts = gen.generate_all(
        num_shorts=1,
        per_short_locations=["Pellmere, Ashby County"],
        per_short_regions=[[REGION]],
        per_short_roads=[["Harrow Lane"]],
    )
    assert posts[0].sampled_roads == ["Harrow Lane"]
    record = _build(posts=[posts[0]]).shorts[0]
    assert record.sampled_roads == ("Harrow Lane",)


class _ClipRegions(list):
    roads = ["Harrow Lane", "A1"]
    vote = None


def test_day_pipeline_writes_the_clips_roads_into_the_shorts_sidecar(tmp_path):
    from test_ost_system_review_surfaces import _full_day, _state

    state = _state()
    day = {}

    def regions_fn(points, short_range):
        return _ClipRegions([REGION])

    import gomoveshift.pipeline.day_pipeline as dp

    real = dp.run_day_pipeline

    def with_regions(**kw):
        return real(short_regions_fn=regions_fn, **kw)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(dp, "run_day_pipeline", with_regions)
        date, result, _render, out = _full_day(tmp_path, state)
    day = json.loads((out / f"shorts_ost_audit_{date}.json").read_text())
    assert result is not None and day["shorts"]
    assert all(row["roads"] == ["Harrow Lane", "A1"] for row in day["shorts"])
