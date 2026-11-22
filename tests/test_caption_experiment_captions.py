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


# --- enrollment: only a Short whose treatment differs from arm A's -----------


def _gen_at(location, titles_text, tmp_path, recording_date=None):
    path = tmp_path / "titles.yaml"
    path.write_text(titles_text)
    return SocialGenerator(
        titles_file=path,
        location=location,
        project_name="proj",
        series_part=1,
        experiments=EXPS,
        experiment_today=date(2026, 11, 10),
        recording_date=recording_date,
    )


def _titles_yaml(n, *, interactive=False, caption=None):
    out = ""
    for i in range(1, n + 1):
        out += f"{i}:\n  title: Rolling loop number {i}\n"
        if interactive:
            out += "  interactive: true\n"
        if caption:
            out += f"  caption: {caption}\n"
    return out


def test_a_short_with_no_location_tags_is_not_enrolled(tmp_path):
    # No location hashtag exists, so arm B would ship zero hashtags, identical to arm A.
    posts = _gen_at("", _titles_yaml(30), tmp_path).generate_all(num_shorts=30)
    arms = {p.experiment_arms.get("fb_hashtags") for p in posts}
    assert "B" not in arms
    assert arms & {"A", "C"}  # A and C differ from each other and stay enrolled
    unenrolled = [p for p in posts if "fb_hashtags" not in p.experiment_arms]
    assert unenrolled  # the would-be B Shorts, now outside the experiment
    legacy = _gen_at("", _titles_yaml(30), tmp_path)
    legacy.experiments = ()
    base = {p.short_index: p for p in legacy.generate_all(num_shorts=30)}
    for p in unenrolled:  # normal non-experiment hashtag behaviour
        assert p.captions["facebook"] == base[p.short_index].captions["facebook"]


def test_an_interactive_short_is_not_enrolled_in_arm_b(tmp_path):
    posts = _gen_at(
        "Dingle Peninsula, County Kerry, Ireland", _titles_yaml(30, interactive=True), tmp_path
    ).generate_all(num_shorts=30)
    assert "B" not in {p.experiment_arms.get("fb_hashtags") for p in posts}
    assert any(p.experiment_arms.get("fb_hashtags") == "C" for p in posts)  # generics differ


def test_every_enrolled_arm_b_short_ships_a_hashtag(tmp_path):
    posts = _gen(EXPS).generate_all(num_shorts=30)
    b = [p for p in posts if p.experiment_arms.get("fb_hashtags") == "B"]
    assert b and all(_tags(p.captions["facebook"]) for p in b)


# --- arm-keyed outcomes need no opener template ------------------------------


@pytest.mark.parametrize("caption", [None, "Filmed on a Tuesday."])
def test_enrolled_shorts_reach_arm_results_hand_caption_or_composed(tmp_path, caption):
    from types import SimpleNamespace as Ns

    from gomoveshift.publishing.experiments import arm_observations, arm_results
    from gomoveshift.tracking.collector import _FB_REEL_METRICS
    from gomoveshift.tracking.schema import PlatformMetrics

    assert _FB_REEL_METRICS["reel_initial_plays"] == "blue_reels_play_count"
    exp = parse_experiments(
        [
            {
                "name": "fb_hashtags",
                "platforms": ["facebook"],
                "start": "2026-11-02",
                "min_n": 3,
                "stop_after_days": 45,
                "arms": {
                    "A": {"hashtag_mode": "none"},
                    "B": {"hashtag_mode": "location"},
                    "C": {"hashtag_mode": "june"},
                },
            }
        ]
    )
    gen = _gen_at(
        "Dingle Peninsula, County Kerry, Ireland",
        _titles_yaml(30, caption=caption),
        tmp_path,
        recording_date="2026-11-01",
    )
    gen.experiments = exp
    posts = gen.generate_all(num_shorts=30)
    if caption:
        assert all(not p.caption_templates for p in posts)  # hand caption: no opener recorded
    tags = {p.short_index: p.experiment_arms for p in posts}
    shorts = [
        Ns(
            index=p.short_index,
            posted_at="2026-11-10T12:00:00",
            metrics={"facebook": PlatformMetrics(reel_initial_plays=100 + p.short_index)},
        )
        for p in posts
    ]
    rows = arm_observations(exp[0], [Ns(project="proj", shorts=shorts)], lambda _p: tags)
    results = {r.arm: r for r in arm_results(exp[0], rows, date(2026, 12, 1))}
    assert {a: results[a].n for a in "ABC"} == {
        a: sum(1 for t in tags.values() if t.get("fb_hashtags") == a) for a in "ABC"
    }
    assert all(r.n >= 3 and r.median_plays is not None for r in results.values())


def test_hand_caption_without_date_is_not_enrolled_where_arm_a_would_copy_youtube(tmp_path):
    # Arm A adds no tags, so a hand caption with no recording date would read the same on
    # Facebook and YouTube; the identical-caption check would raise. Such a Short stays out.
    gen = _gen_at(
        "Dingle Peninsula, County Kerry, Ireland",
        _titles_yaml(30, caption="Filmed on a Tuesday."),
        tmp_path,
    )
    posts = gen.generate_all(num_shorts=30)  # must not raise
    arms = {p.experiment_arms.get("fb_hashtags") for p in posts}
    assert "A" not in arms
    assert arms >= {"B", "C"}
