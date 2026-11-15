"""Each Short records which caption opener template each platform's caption was built from.

Meta caption learning needs to know, per delivered post, which opener produced it: the
metrics collector joins ``caption_templates`` to the post's reach and plays by
``short_index`` and platform, the same join ``experiment_arms`` uses. The id is a hash
of the template text, so it survives reordering the pool and changes when the wording
does."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from gomoveshift.pipeline.social_generator import (
    _PLATFORM_CAPTION_TEMPLATES,
    SocialGenerator,
    caption_template_id,
)

sys.path.insert(0, str(Path(__file__).parent))

COMPOSED = ("youtube", "facebook", "instagram", "tiktok")


@pytest.fixture
def titles(tmp_path):
    path = tmp_path / "titles.yaml"
    path.write_text(
        "".join(f"{i}:\n  title: Rolling the Dingle loop number {i}\n" for i in range(1, 9))
    )
    return path


def _gen(titles):
    return SocialGenerator(
        titles_file=titles,
        location="Dingle Peninsula, County Kerry, Ireland",
        project_name="proj",
        series_part=1,
    )


def _opener_ids(platform):
    return {caption_template_id(t): t for t in _PLATFORM_CAPTION_TEMPLATES[platform]}


def test_the_id_is_a_short_stable_hash_of_the_template_text():
    a = caption_template_id("Rolling through {location}")
    assert a == caption_template_id("Rolling through {location}")
    assert a != caption_template_id("Rolling into {location}")
    assert len(a) == 12 and all(c in "0123456789abcdef" for c in a)


def test_pool_ids_never_collide():
    for platform in COMPOSED:
        pool = _PLATFORM_CAPTION_TEMPLATES[platform]
        assert len(_opener_ids(platform)) == len(set(pool)), platform


def test_every_composed_caption_names_the_opener_it_opens_with(titles):
    gen = _gen(titles)
    for post in gen.generate_all(num_shorts=4):
        assert set(post.caption_templates) == set(COMPOSED)
        for platform, tid in post.caption_templates.items():
            template = _opener_ids(platform)[tid]
            opener = template.format(
                location=gen.social_location, terrain_desc=gen.terrain_desc, weather_emoji=""
            ).strip()
            head = (opener[:1].upper() + opener[1:]).rstrip(".")
            assert post.captions[platform].startswith(head[:20]), (platform, template)


def test_a_hand_caption_records_no_opener(tmp_path):
    path = tmp_path / "titles.yaml"
    path.write_text(
        "1:\n  title: Rolling the Dingle loop\n"
        "  caption: Mist over Dingle Peninsula, County Kerry, Ireland\n"
    )
    post = _gen(path).generate_all(num_shorts=1)[0]
    assert post.caption_templates == {}


def test_the_ids_ride_the_social_post_dict(titles):
    post = _gen(titles).generate_all(num_shorts=1)[0]
    assert post.to_dict()["caption_templates"] == post.caption_templates


def test_the_plan_builder_carries_the_ids_onto_the_record():
    from test_publishing_plan_builder import _build, _post

    plan = _build(posts=[_post(caption_templates={"facebook": "abc123abc123"})])
    assert plan.shorts[0].caption_templates == {"facebook": "abc123abc123"}
    assert _build(posts=[_post()]).shorts[0].caption_templates == {}


def test_the_ids_round_trip_through_the_sidecar(tmp_path):
    from test_cli_enqueue_plan import _record, _sidecar

    from gomoveshift.publishing.plan_io import load_publish_plan

    tagged = replace(_record(), caption_templates={"instagram": "0123456789ab"})
    assert load_publish_plan(_sidecar(tmp_path, tagged)).shorts[0].caption_templates == {
        "instagram": "0123456789ab"
    }
    (tmp_path / "b").mkdir()
    assert load_publish_plan(_sidecar(tmp_path / "b", _record())).shorts[0].caption_templates == {}


def test_a_malformed_id_map_is_refused(tmp_path):
    from test_cli_enqueue_plan import _record, _sidecar

    from gomoveshift.publishing.plan_io import PublishPlanFormatError, load_publish_plan

    path = _sidecar(tmp_path, _record())
    doc = json.loads(path.read_text())
    doc["shorts"][0]["caption_templates"] = {"instagram": 7}
    path.write_text(json.dumps(doc))
    with pytest.raises(PublishPlanFormatError, match="caption_templates"):
        load_publish_plan(path)


def test_a_plan_without_ids_keeps_its_bytes_and_one_with_ids_carries_them():
    from test_cli_enqueue_plan import _record

    from gomoveshift.pipeline.drive_phases.phase_d import _serialize_publish_plan
    from gomoveshift.publishing.models import PublishPlan

    plain = json.loads(_serialize_publish_plan(PublishPlan(shorts=(_record(),))))
    assert "caption_templates" not in plain["shorts"][0]
    tagged = replace(_record(), caption_templates={"facebook": "abc123abc123"})
    doc = json.loads(_serialize_publish_plan(PublishPlan(shorts=(tagged,))))
    assert doc["shorts"][0]["caption_templates"] == {"facebook": "abc123abc123"}
