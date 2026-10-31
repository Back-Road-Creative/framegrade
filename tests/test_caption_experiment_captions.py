"""Hashtag modes in SocialGenerator and the sidecar arm tag."""

from datetime import date

import pytest

from gomoveshift.pipeline.social_generator import SocialGenerator
from gomoveshift.publishing.experiments import parse_experiments

EXPS = parse_experiments(
    [
        {
            "name": "fb_hashtags",
            "platforms": ["facebook"],
            "start": "2026-11-02",
            "min_n": 20,
            "stop_after_days": 45,
            "arms": {
                "A": {"hashtag_mode": "none"},
                "B": {"hashtag_mode": "location"},
                "C": {"hashtag_mode": "june"},
            },
        }
    ]
)


@pytest.fixture(autouse=True)
def _titles(tmp_path, monkeypatch):
    """Hand titles for every index: composed titles run out after a few shorts."""
    path = tmp_path / "titles.yaml"
    path.write_text(
        "".join(f"{i}:\n  title: Rolling the Dingle loop number {i}\n" for i in range(1, 31))
    )
    monkeypatch.setattr(SocialGenerator, "_TEST_TITLES", path, raising=False)


def _gen(experiments, today=date(2026, 11, 10)):
    return SocialGenerator(
        titles_file=SocialGenerator._TEST_TITLES,
        location="Dingle Peninsula, County Kerry, Ireland",
        project_name="proj",
        series_part=1,
        experiments=experiments,
        experiment_today=today,
    )


def _posts_by_arm():
    posts = _gen(EXPS).generate_all(num_shorts=30)
    by = {}
    for p in posts:
        by.setdefault(p.experiment_arms["fb_hashtags"], []).append(p)
    return by


def _tags(text):
    return [w for w in text.split() if w.startswith("#")]


def test_arm_a_has_no_hashtags_on_facebook():
    by = _posts_by_arm()
    assert by["A"] and all(_tags(p.captions["facebook"]) == [] for p in by["A"])


def test_arm_b_has_two_location_hashtags():
    by = _posts_by_arm()
    assert by["B"]
    for p in by["B"]:
        tags = _tags(p.captions["facebook"])
        assert len(tags) == 2
        assert not {"#POVDriving", "#RoadTrip", "#ScenicDrive"} & set(tags)


def test_arm_c_has_five_june_style_hashtags():
    # June 2026 posts (upstream plan Appendix A): "#POVDriving #ScenicDrive
    # #RoadTrip #<place> #<region>" -- three fixed generics, then the two
    # location tags arm B carries.
    by = _posts_by_arm()
    assert by["C"]
    loc = {tuple(_tags(p.captions["facebook"])) for p in by["B"]}
    for p in by["C"]:
        tags = _tags(p.captions["facebook"])
        assert tags[:3] == ["#POVDriving", "#ScenicDrive", "#RoadTrip"]
        assert len(tags) == 5 and tuple(tags[3:]) in loc


def test_other_platforms_unchanged_by_arm():
    base = SocialGenerator(
        titles_file=SocialGenerator._TEST_TITLES,
        location="Dingle Peninsula, County Kerry, Ireland",
        project_name="proj",
        series_part=1,
        experiments=(),
    ).generate_all(num_shorts=6)
    armed = _gen(EXPS).generate_all(num_shorts=6)
    for a, b in zip(base, armed):
        assert a.captions["instagram"] == b.captions["instagram"]
        assert a.captions["tiktok"] == b.captions["tiktok"]


def test_no_experiment_means_no_tag_and_legacy_caption():
    posts = _gen(()).generate_all(num_shorts=3)
    assert all(p.experiment_arms == {} for p in posts)
    assert all("experiment_arms" in p.to_dict() for p in posts)


def test_past_window_assigns_no_arm():
    posts = _gen(EXPS, today=date(2027, 1, 30)).generate_all(num_shorts=3)
    assert all(p.experiment_arms == {} for p in posts)


def test_arm_tag_round_trips_through_sidecar(tmp_path):
    from dataclasses import replace

    from test_cli_enqueue_plan import _record, _sidecar

    from gomoveshift.publishing.plan_io import load_publish_plan

    tagged = replace(_record(), experiment_arms={"fb_hashtags": "B"})
    assert load_publish_plan(_sidecar(tmp_path, tagged)).shorts[0].experiment_arms == {
        "fb_hashtags": "B"
    }
    assert load_publish_plan(_sidecar(tmp_path, _record())).shorts[0].experiment_arms == {}


def test_untagged_sidecar_keeps_its_bytes_and_tagged_one_carries_the_arm():
    import json
    from dataclasses import replace

    from test_cli_enqueue_plan import _record

    from gomoveshift.pipeline.drive_phases.phase_d import _serialize_publish_plan
    from gomoveshift.publishing.models import PublishPlan

    plain = json.loads(_serialize_publish_plan(PublishPlan(shorts=(_record(),))))
    assert "experiment_arms" not in plain["shorts"][0]
    tagged = replace(_record(), experiment_arms={"fb_hashtags": "C"})
    doc = json.loads(_serialize_publish_plan(PublishPlan(shorts=(tagged,))))
    assert doc["shorts"][0]["experiment_arms"] == {"fb_hashtags": "C"}
