"""The Shorts experiment arms change the Short's text, deterministically, and
every variant passes the gates the legacy text passes. All data is synthetic."""

from __future__ import annotations

import dataclasses
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_publishing_plan_builder import _build  # noqa: E402
from test_sync_bracket_short_text import _short, _write  # noqa: E402
from test_sync_bracket_short_text import _records as _plan_records  # noqa: E402

from datetime import date  # noqa: E402

from gomoveshift.pipeline import longform_metadata as lm  # noqa: E402
from gomoveshift.pipeline.content_validator import ContentValidator  # noqa: E402
from gomoveshift.pipeline.drive_phases.phase_d import _serialize_publish_plan  # noqa: E402
from gomoveshift.pipeline.social_generator import (  # noqa: E402
    GENERIC_HASHTAGS,
    STAMPED_BODY_TEMPLATES,
    SocialGenerator,
    TitleFacts,
    title_stamp,
)
from gomoveshift.publishing.anti_spam_gate import shared_prefix_groups  # noqa: E402
from gomoveshift.publishing.experiments import (  # noqa: E402
    SHORTS_ARMS,
    SHORTS_FACTORS,
    SHORTS_HASHTAG_SET_ARMS,
    SHORTS_TITLE_BODY_ARMS,
    assign_shorts_arms,
    record_shorts_arms,
)
from gomoveshift.publishing.plan_bracket_sync import sync_bracket_short_text  # noqa: E402
from gomoveshift.publishing.plan_gates import (  # noqa: E402
    assert_plan_text_grammar,
    assert_shorts_metadata_in_region,
)
from gomoveshift.publishing.plan_io import load_publish_plan  # noqa: E402
from gomoveshift.publishing.youtube_adapter import short_review_refusal  # noqa: E402

LABEL = "Wythe County, Virginia"
REGIONS = ["Wythe County, Virginia"]
DATE = "2026-09-14"


@pytest.fixture(autouse=True)
def _no_published_titles(monkeypatch):
    monkeypatch.setattr(lm, "load_published_titles", lambda: set())


def _gen(**kw) -> SocialGenerator:
    return SocialGenerator(location=LABEL, project_name="drive", series_part=1, **kw)


def _facts(n=8, place_ok=True):
    return [TitleFacts(date=DATE, ordinal=k, secs=41, place_ok=place_ok) for k in range(1, n + 1)]


def _all(gen, facts, **kw):
    n = len(facts)
    return gen.generate_all(
        num_shorts=n,
        per_short_locations=[LABEL] * n,
        per_short_regions=[REGIONS] * n,
        per_short_title_facts=facts,
        **kw,
    )


def _dict(post):
    out = {k: v for k, v in post.to_dict().items() if k != "generated_at"}
    # The title decision stamps wall-clock evidence time; it is not post text.
    out["title_decision"] = {
        k: v for k, v in (out.get("title_decision") or {}).items() if k != "evidence_as_of"
    }
    return out


def _rendered(post):
    return (post.youtube_title, post.youtube_description, post.youtube_tags, post.captions)


# --- tracked config ----------------------------------------------------------


def test_arm_config_is_one_source_of_truth():
    assert tuple(SHORTS_ARMS) == SHORTS_FACTORS
    assert SHORTS_ARMS["yt_title_body"] == tuple(SHORTS_TITLE_BODY_ARMS)
    assert SHORTS_ARMS["yt_hashtag_set"] == tuple(SHORTS_HASHTAG_SET_ARMS)
    assert all(len(v) >= 2 for v in SHORTS_ARMS.values())


def test_every_title_body_is_an_existing_generator_body():
    for bodies in SHORTS_TITLE_BODY_ARMS.values():
        assert set(bodies) <= set(STAMPED_BODY_TEMPLATES)
    assert len({b for v in SHORTS_TITLE_BODY_ARMS.values() for b in v}) == sum(
        len(v) for v in SHORTS_TITLE_BODY_ARMS.values()
    ), "arms must not share a body"


def test_every_hashtag_is_an_existing_generic_hashtag():
    for tags in SHORTS_HASHTAG_SET_ARMS.values():
        assert set(tags) <= set(GENERIC_HASHTAGS)
    assert len(set(SHORTS_HASHTAG_SET_ARMS.values())) == len(SHORTS_HASHTAG_SET_ARMS)


# --- assignment --------------------------------------------------------------


def test_assignment_is_deterministic_and_stratified():
    def run():
        history: dict = {}
        out = []
        for k in range(1, 9):
            arms = assign_shorts_arms(("p", "slot1"), k, history, SHORTS_FACTORS)
            record_shorts_arms(("p", "slot1"), history, arms)
            out.append(arms)
        return out

    first = run()
    assert first == run()
    for factor, arms in SHORTS_ARMS.items():
        counts = [sum(1 for a in first if a[factor] == arm) for arm in arms]
        assert max(counts) - min(counts) <= 1, factor


def test_strata_are_balanced_independently():
    history: dict = {}
    for slot in ("a", "b"):
        for k in range(2):
            arms = assign_shorts_arms(("p", slot), k, history, ["yt_hashtag_set"])
            record_shorts_arms(("p", slot), history, arms)
    for slot in ("a", "b"):
        assert sorted(history[("yt_hashtag_set", "p", slot)]) == sorted(
            SHORTS_ARMS["yt_hashtag_set"]
        )


def test_generate_all_assigns_and_stores_arms_deterministically():
    posts = _all(_gen(), _facts(), shorts_arms_enabled=True)
    again = _all(_gen(), _facts(), shorts_arms_enabled=True)
    assert [p.experiment_arms for p in posts] == [p.experiment_arms for p in again]
    assert [_rendered(p) for p in posts] == [_rendered(p) for p in again]
    for post in posts:
        # a fact title has no body variants, so only the hashtag set is assigned
        assert "yt_title_body" not in post.experiment_arms
        assert post.experiment_arms["yt_hashtag_set"] in SHORTS_ARMS["yt_hashtag_set"]


# --- rendering ---------------------------------------------------------------


def _by_arm(posts, factor):
    out: dict = {}
    for post in posts:
        out.setdefault(post.experiment_arms[factor], []).append(post)
    return out


def test_title_body_arms_change_the_stamped_title_as_specified(monkeypatch):
    # the arm shapes only a STAMPED title's body; a fact title has no body to vary, so the
    # drive never assigns it. Every place already published leaves the stamped composer.
    monkeypatch.setattr(lm, "load_published_titles", lambda: {"wythe county"})
    facts = TitleFacts(DATE, 3, 41, True)
    for arm in SHORTS_TITLE_BODY_ARMS:
        gen = SocialGenerator(location="Wythe County, VA", project_name="drive", series_part=1)
        title = gen._compose_stamped_title(
            facts, "titles", 100, title_body_arm=arm, candidates=["Wythe County"]
        )
        body = title.split(": ", 1)[1]
        assert body.startswith("Wythe County") != (arm == "clip_led"), (arm, body)
        assert ("seconds" in body) == (arm == "clip_led"), (arm, body)
    posts = _all(_gen(), _facts(), shorts_arms_enabled=True)
    assert not any("yt_title_body" in p.experiment_arms for p in posts)


def test_hashtag_set_arms_change_the_description_as_specified():
    posts = _all(_gen(), _facts(), shorts_arms_enabled=True)
    by_arm = _by_arm(posts, "yt_hashtag_set")
    assert set(by_arm) == set(SHORTS_HASHTAG_SET_ARMS)
    for arm, members in by_arm.items():
        for post in members:
            tags = re.findall(r"#\w+", post.youtube_description)
            assert tags[0] == "#Shorts"
            for tag in SHORTS_HASHTAG_SET_ARMS[arm]:
                assert tag in tags
            for other, other_tags in SHORTS_HASHTAG_SET_ARMS.items():
                if other != arm:
                    assert not set(other_tags) & set(tags) - set(SHORTS_HASHTAG_SET_ARMS[arm])


GOLDEN = Path(__file__).parent / "fixtures" / "shorts_text_origin_main.json"


_HEX = re.compile(r"[0-9a-f]{8,}")


def _fake_hex(obj, seen=None):
    """``obj`` with every all-hex string (an evidence id, a caption template hash), key or
    value, replaced by ``hex-N`` (N = order of first appearance), so the fixture holds no
    hex the secret scan reads as a key while the test still checks which values repeat."""
    seen = {} if seen is None else seen
    if isinstance(obj, str):
        return seen.setdefault(obj, f"hex-{len(seen)}") if _HEX.fullmatch(obj) else obj
    if isinstance(obj, dict):
        return {_fake_hex(k, seen): _fake_hex(v, seen) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_fake_hex(v, seen) for v in obj]
    return obj


def test_no_arm_renders_byte_identical_to_origin_main():
    """``tests/fixtures/shorts_text_origin_main.json`` was generated from origin/main
    (e2eae14) before arms existed: 3 places x (no facts, place_ok, not place_ok) x 2 Shorts,
    spooled texts and the experiment date pinned, hex ids faked (``_fake_hex``).
    Arms off must reproduce it exactly. The one deliberate change since: a ``place_ok``
    Short 1's titles are its measured place, not the stamp (fact titles), and every
    caption is a measured-fact shape with no filler tail (fact captions)."""
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    places = {key.split("|")[0] for key in golden}
    assert len(places) == 3 and any("Llanfair" in p for p in places)
    assert any("St. Mary's" in p for p in places)
    modes = {"facts_none": None, "place_ok": True, "place_not_ok": False}
    for key, expected in golden.items():
        place, mode = key.split("|")
        ok = modes[mode]
        facts = (
            None
            if ok is None
            else [TitleFacts(date=DATE, ordinal=k, secs=41, place_ok=ok) for k in (1, 2)]
        )
        for kw in ({}, {"shorts_arms_enabled": False}):
            gen = SocialGenerator(
                location=place,
                project_name="golden",
                series_part=1,
                experiment_today=date(2026, 9, 14),
                spooled_texts={},
            )
            posts = gen.generate_all(
                num_shorts=2,
                per_short_locations=[place] * 2,
                per_short_regions=[[place]] * 2,
                per_short_title_facts=facts,
                **kw,
            )
            got = json.loads(json.dumps([_dict(p) for p in posts], sort_keys=True, default=str))
            got = _fake_hex(got)
            assert got == expected, key


def test_unknown_arm_renders_as_legacy_and_is_not_stored():
    legacy = _gen().generate_post(3, title_facts=_facts()[2])
    for arms in (
        {"yt_title_body": "nope", "yt_hashtag_set": "nope"},
        {"not_a_factor": "place_led"},
    ):
        post = _gen().generate_post(3, title_facts=_facts()[2], shorts_arms=arms)
        assert _dict(post) == _dict(legacy)


def test_pov_hashtag_arm_equals_the_legacy_description():
    legacy = _gen()._build_youtube_description(2)
    assert _gen()._build_youtube_description(2, hashtag_arm="pov") == legacy
    assert _gen()._build_youtube_description(2, hashtag_arm="road") != legacy


def test_a_factor_the_text_cannot_carry_is_not_assigned():
    # No verified place: no title-body arm, hashtag arm still assigned and rendered.
    posts = _all(_gen(), _facts(place_ok=False), shorts_arms_enabled=True)
    assert posts
    for post in posts:
        assert "yt_title_body" not in post.experiment_arms
        arm = post.experiment_arms["yt_hashtag_set"]
        assert set(SHORTS_HASHTAG_SET_ARMS[arm]) <= set(
            re.findall(r"#\w+", post.youtube_description)
        )
    assert {p.experiment_arms["yt_hashtag_set"] for p in posts} == set(SHORTS_HASHTAG_SET_ARMS)
    # No facts at all: the stamped title is not composed, so no title-body arm either.
    bare = _all(_gen(), [None] * 3, shorts_arms_enabled=True)
    assert all("yt_title_body" not in p.experiment_arms for p in bare)
    gen = _gen()
    assert gen._shorts_arm_factors(1, TitleFacts(DATE, 1, 41, False)) == ["yt_hashtag_set"]
    assert gen._shorts_arm_factors(1, None) == ["yt_hashtag_set"]
    # a verified place no longer adds the title-body factor: the title is the place itself
    assert gen._shorts_arm_factors(1, TitleFacts(DATE, 1, 41, True)) == ["yt_hashtag_set"]


# --- every variant passes the existing gates ---------------------------------


def _route_record():
    from gomoveshift.pipeline.route_claims import RouteEvidence, _Bin

    bins = [_Bin(60.0, "Wythe County", "Virginia", None)]
    return RouteEvidence(
        retained_seconds=60.0, gaps=[], places=[], unsettled_seconds=0.0, bins=bins,
        boundary_proofs={}, windows_key="k",
    ).record()  # fmt: skip


def test_every_variant_passes_the_existing_gates():
    posts = _all(_gen(), _facts(), shorts_arms_enabled=True)
    legacy = _all(_gen(), _facts())
    assert not any("yt_title_body" in p.experiment_arms for p in posts)
    assert {p.experiment_arms["yt_hashtag_set"] for p in posts} == set(SHORTS_HASHTAG_SET_ARMS)
    validator = ContentValidator()
    for post in posts:
        assert len(post.youtube_title) <= 100
        validator.enforce(
            titles=[post.youtube_title, post.tiktok_title],
            captions=[c for c in (post.caption, post.first_comment) if c],
            descriptions=[post.youtube_description],
        )
    assert_plan_text_grammar({1: posts})
    assert_shorts_metadata_in_region({1: posts})
    assert not shared_prefix_groups([p.youtube_title for p in posts])
    assert not shared_prefix_groups([p.youtube_title for p in legacy])

    plan = _build(
        posts=posts,
        parent_recording_date_by_short_index={p.short_index: DATE for p in posts},
    )
    for record in plan.shorts:
        record = dataclasses.replace(record, route_record=_route_record())
        assert short_review_refusal(record) is None, record.payloads.youtube.title


# --- persistence -------------------------------------------------------------


def test_arms_are_persisted_on_the_plan_and_survive_the_sidecar(tmp_path):
    posts = _all(_gen(), _facts(4), shorts_arms_enabled=True)
    plan = _build(
        posts=posts,
        parent_recording_date_by_short_index={p.short_index: DATE for p in posts},
    )
    assert [r.experiment_arms for r in plan.shorts] == [p.experiment_arms for p in posts]
    path = tmp_path / "p_publish_plan.json"
    path.write_text(_serialize_publish_plan(plan), encoding="utf-8")
    loaded = load_publish_plan(path)
    assert [r.experiment_arms for r in loaded.shorts] == [p.experiment_arms for p in posts]


# --- explicit arms: regeneration renders the stored arms, never assigns ------


def test_explicit_per_short_arms_are_rendered_verbatim_and_never_assigned():
    stored = [
        {"yt_title_body": "clip_led", "yt_hashtag_set": "road"},
        {},
        {"yt_hashtag_set": "pov"},
    ]
    posts = _all(_gen(), _facts(3), per_short_shorts_arms=stored)
    # a stored title-body arm is no longer rendered: the title is the measured place
    assert "seconds" not in posts[0].youtube_title
    assert "#RoadTrip #OpenRoad" in posts[0].youtube_description
    assert "yt_title_body" not in posts[0].experiment_arms
    assert not set(SHORTS_FACTORS) & set(posts[1].experiment_arms)
    assert posts[1].youtube_description == _all(_gen(), _facts(3))[1].youtube_description
    assert "yt_title_body" not in posts[2].experiment_arms
    assert posts[2].experiment_arms["yt_hashtag_set"] == "pov"
    with pytest.raises(ValueError):
        _all(_gen(), _facts(3), per_short_shorts_arms=stored[:2])
    with pytest.raises(ValueError):
        _all(_gen(), _facts(3), per_short_shorts_arms=stored, shorts_arms_enabled=True)


def _stored_arms_plan(tmp_path, arms):
    return _write(tmp_path, [_short(1, experiment_arms=arms)])


def test_bracket_sync_keeps_text_matching_the_stored_arms(tmp_path):
    arms = {"fb_hashtags": "A", "yt_hashtag_set": "road"}
    plan = _stored_arms_plan(tmp_path, arms)
    changes = sync_bracket_short_text(plan, apply=True)
    assert changes[0].status == "changed", changes[0].reason
    rec = _plan_records(plan)[0]
    assert rec.experiment_arms == arms
    assert "seconds" not in rec.payloads.youtube.title
    assert "#RoadTrip #OpenRoad" in rec.payloads.youtube.description
    assert "#POVDriving" not in rec.payloads.youtube.description


def test_bracket_sync_pov_arm_renders_its_own_text(tmp_path):
    arms = {"yt_hashtag_set": "pov"}
    plan = _stored_arms_plan(tmp_path, arms)
    assert sync_bracket_short_text(plan, apply=True)[0].status == "changed"
    rec = _plan_records(plan)[0]
    assert rec.experiment_arms == arms
    assert "seconds" not in rec.payloads.youtube.title
    assert "#POVDriving" in rec.payloads.youtube.description
    assert "#RoadTrip" not in rec.payloads.youtube.description


@pytest.mark.parametrize(
    "arms",
    [{"yt_hashtag_set": "nope"}, {"yt_title_body": "nope"}, {"yt_title_body": "clip_led"}],
)
def test_bracket_sync_refuses_a_stored_arm_it_can_no_longer_render(tmp_path, arms):
    plan = _stored_arms_plan(tmp_path, arms)
    before = plan.read_text()
    changes = sync_bracket_short_text(plan, apply=True)
    assert changes[0].status == "refused"
    assert "arm" in changes[0].reason
    assert plan.read_text() == before


# --- one history, one stratum per base project --------------------------------


def _drive_posts(n_lf=8, n=5, **gen_kw):
    """The Phase D call pattern: one ``generate_all`` per LF (``<base>_lfNN``), one shared
    history, day-wide title ordinals exactly as ``_per_short_title_facts`` gives them."""
    history: dict = {}
    posts = []
    for lf in range(1, n_lf + 1):
        gen = SocialGenerator(
            location=LABEL,
            project_name=f"drive_lf{lf:02d}",
            series_part=lf,
            spooled_texts={},
            **gen_kw,
        )
        facts = [
            TitleFacts(date=DATE, ordinal=(lf - 1) * n + k + 1, secs=41, place_ok=True)
            for k in range(n)
        ]
        posts += gen.generate_all(
            num_shorts=n,
            per_short_locations=[LABEL] * n,
            per_short_regions=[REGIONS] * n,
            per_short_title_facts=facts,
            global_indexes=[lf * 10 + k for k in range(n)],
            shorts_arms_enabled=True,
            shorts_arm_history=history,
        )
    return posts, history


def test_phase_d_pattern_balances_the_drive_with_day_wide_ordinals():
    posts, history = _drive_posts()
    assert len(posts) == 40
    assert not any("yt_title_body" in p.experiment_arms for p in posts)
    for factor in ("yt_hashtag_set",):
        arms = SHORTS_ARMS[factor]
        counts = [sum(1 for p in posts if p.experiment_arms[factor] == a) for a in arms]
        assert sum(counts) == 40 and max(counts) - min(counts) <= 1, (factor, counts)
    assert {key[1] for key in history} == {"drive"}, "strata use the base project"


def test_a_dropped_arm_is_not_counted_in_the_history():
    place = "Llanfairpwllgwyngyllgogerychwyrndrobwllllantysiliogogogoch Community Council"
    gen = SocialGenerator(location=place, project_name="p_lf01", series_part=1, spooled_texts={})
    history: dict = {}
    facts = [TitleFacts(DATE, k, 41, True) for k in range(1, 7)]
    posts = gen.generate_all(
        num_shorts=6,
        per_short_locations=[place] * 6,
        per_short_regions=[[place]] * 6,
        per_short_title_facts=facts,
        shorts_arms_enabled=True,
        shorts_arm_history=history,
    )
    for factor, arms in SHORTS_ARMS.items():
        kept = [p.experiment_arms[factor] for p in posts if factor in p.experiment_arms]
        recorded = history.get((factor, "p", ""), [])
        assert sorted(recorded) == sorted(kept), factor
        counts = [kept.count(a) for a in arms]
        if kept:
            assert max(counts) - min(counts) <= 1
    assert ("yt_title_body", "p", "") not in history


def test_phase_d_hands_one_shared_history_to_every_long_form(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from gomoveshift.pipeline.drive_phases.phase_d_scheduling import _build_social_posts_per_lf

    seen = []

    def gen_all(num_shorts, index_offset, **kw):
        seen.append(kw["shorts_arm_history"])
        return [
            SimpleNamespace(short_index=index_offset + i, to_dict=dict) for i in range(num_shorts)
        ]

    inst = MagicMock()
    inst.generate_all.side_effect = gen_all
    _build_social_posts_per_lf(
        social_gen_cls=MagicMock(return_value=inst),
        drive_plan=SimpleNamespace(
            regular_lfs=[SimpleNamespace(index=1), SimpleNamespace(index=2)]
        ),
        shorts_per_lf={1: ["a"], 2: ["b"]},
        per_lf_locations_override={1: "x", 2: "x"},
        project_base="p",
        output_base=tmp_path,
    )
    assert len(seen) == 2 and seen[0] is seen[1] and isinstance(seen[0], dict)


def test_phase_d_hands_one_shared_title_ledger_to_every_long_form(tmp_path):
    """A measured place titles one Short of the drive: every LF's ``generate_all`` gets the
    SAME ``used_titles`` dict, so a title cannot repeat across LFs."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from gomoveshift.pipeline.drive_phases.phase_d_scheduling import _build_social_posts_per_lf

    seen = []

    def gen_all(num_shorts, index_offset, **kw):
        seen.append(kw.get("used_titles"))
        return [
            SimpleNamespace(short_index=index_offset + i, to_dict=dict) for i in range(num_shorts)
        ]

    inst = MagicMock()
    inst.generate_all.side_effect = gen_all
    _build_social_posts_per_lf(
        social_gen_cls=MagicMock(return_value=inst),
        drive_plan=SimpleNamespace(
            regular_lfs=[SimpleNamespace(index=1), SimpleNamespace(index=2)]
        ),
        shorts_per_lf={1: ["a"], 2: ["b"]},
        per_lf_locations_override={1: "x", 2: "x"},
        project_base="p",
        output_base=tmp_path,
    )
    assert len(seen) == 2 and isinstance(seen[0], dict) and seen[0] is seen[1]


# --- the stored hashtag arm is always in the rendered description -------------

LONG_COUNTY = "Llanfairpwllgwyngyllgogerychwyrndrobwll County, Pennsylvania"
LONG_POI = "Snowdonia National Park Visitor Centre and Adventure Trailhead"


@pytest.mark.parametrize("arm", sorted(SHORTS_HASHTAG_SET_ARMS))
def test_hashtag_arm_tags_survive_a_long_place_description(arm):
    gen = SocialGenerator(location=LONG_COUNTY, project_name="p", series_part=1, spooled_texts={})
    gen.nearest_poi = LONG_POI
    description = gen._build_youtube_description(1, hashtag_arm=arm)
    assert len(description) <= 200
    tags = set(re.findall(r"#\w+", description))
    assert set(SHORTS_HASHTAG_SET_ARMS[arm]) <= tags, description


def test_long_place_arms_differ_and_stored_arm_matches_the_text():
    texts = {}
    for arm in SHORTS_HASHTAG_SET_ARMS:
        gen = SocialGenerator(
            location=LONG_COUNTY, project_name="p", series_part=1, spooled_texts={}
        )
        gen.nearest_poi = LONG_POI
        post = gen.generate_post(
            1,
            title_facts=TitleFacts(DATE, 1, 41, True),
            shorts_arms={"yt_hashtag_set": arm},
        )
        tags = set(re.findall(r"#\w+", post.youtube_description))
        recorded = post.experiment_arms.get("yt_hashtag_set")
        if recorded:
            assert set(SHORTS_HASHTAG_SET_ARMS[recorded]) <= tags
        texts[arm] = post.youtube_description
    assert len(set(texts.values())) == len(texts)


# --- title bodies: place-named only, never a duplicate, never cut mid-word ----


def test_a_title_arm_is_never_taken_over_place_free_text():
    gen = _gen()
    facts = TitleFacts(DATE, 2, 41, False)
    plain = gen._compose_stamped_title(facts, "titles", 100)
    for arm in SHORTS_TITLE_BODY_ARMS:
        assert gen._compose_stamped_title(facts, "titles", 100, title_body_arm=arm) == plain


def test_stamped_body_colliding_with_the_ledger_is_skipped_by_the_arm(monkeypatch):
    # The arm only shapes a stamped title (a place already published): every clip_led body
    # that collides leaves no body, a colliding body of a two-body arm still leaves the other.
    facts = TitleFacts(DATE, 1, 41, True)
    gen = _gen()
    stamp = title_stamp(DATE, 1)
    taken = {f"{stamp}: 41 seconds, {gen.location_short}".lower(), gen.location_short.lower()}
    monkeypatch.setattr(lm, "load_published_titles", lambda: taken)
    assert gen._arm_title_bodies(facts, None, "clip_led", 100) == []
    taken2 = {f"{stamp}: {gen.location_short}".lower(), gen.location_short.lower()}
    monkeypatch.setattr(lm, "load_published_titles", lambda: taken2)
    bodies = gen._arm_title_bodies(facts, None, "place_led", 100)
    assert bodies == [b for b in SHORTS_TITLE_BODY_ARMS["place_led"] if b != "{location}"]


def test_a_stamped_arm_body_that_would_be_cut_mid_word_is_skipped():
    place = "Llanfairpwllgwyngyllgogerychwyrndrobwllllantysiliogogogoch Community Council"
    gen = SocialGenerator(location=place, project_name="p", series_part=1, spooled_texts={})
    facts = TitleFacts(DATE, 1, 41, True)
    stamp_len = len("Sep 14, 1st stretch: ")
    assert stamp_len + len(place) <= 100 < stamp_len + len(place) + len(", on a Monday")
    assert gen._arm_title_bodies(facts, None, "place_led", 100) == ["{location}"]  # fits whole
    assert gen._arm_title_bodies(facts, None, "clip_led", 100) == []  # no clip_led body fits


def test_every_title_variant_passes_the_gates_with_a_long_place():
    county = "Llanfairpwllgwyngyllgogerychwyrndrobwll County"
    label = f"{county}, Pennsylvania"
    validator = ContentValidator()
    for arm in SHORTS_TITLE_BODY_ARMS:
        gen = SocialGenerator(location=label, project_name="p", series_part=1, spooled_texts={})
        posts = gen.generate_all(
            num_shorts=2,
            per_short_locations=[label] * 2,
            per_short_regions=[[label]] * 2,
            per_short_title_facts=_facts(2),
            per_short_shorts_arms=[{"yt_title_body": arm}] * 2,
        )
        for post in posts:
            assert len(post.youtube_title) <= 100
            assert not post.youtube_title.rstrip().endswith(("…", "..."))
            validator.enforce(
                titles=[post.youtube_title, post.tiktok_title],
                captions=[c for c in (post.caption, post.first_comment) if c],
                descriptions=[post.youtube_description],
            )
        plan = _build(
            posts=posts,
            parent_recording_date_by_short_index={p.short_index: DATE for p in posts},
        )
        from gomoveshift.pipeline.route_claims import RouteEvidence, _Bin

        bins = [_Bin(60.0, county, "Pennsylvania", None)]
        for record in plan.shorts:
            record = dataclasses.replace(
                record,
                route_record=RouteEvidence(
                    retained_seconds=60.0, gaps=[], places=[], unsettled_seconds=0.0,
                    bins=bins, boundary_proofs={}, windows_key="k",
                ).record(),
            )  # fmt: skip
            assert short_review_refusal(record) is None, (arm, record.payloads.youtube.title)


@pytest.mark.parametrize("arm", sorted(SHORTS_HASHTAG_SET_ARMS))
def test_a_kept_hashtag_arm_never_cuts_more_description_body_than_legacy(arm):
    def render(hashtag_arm=None):
        gen = SocialGenerator(
            location=LONG_COUNTY, project_name="p", series_part=1, spooled_texts={}
        )
        gen.nearest_poi = LONG_POI
        return gen.generate_post(
            1,
            title_facts=TitleFacts(DATE, 1, 41, True),
            shorts_arms={"yt_hashtag_set": hashtag_arm} if hashtag_arm else None,
        )

    legacy, post = render(), render(arm)
    body = lambda d: d.split(" #Shorts", 1)[0]  # noqa: E731
    assert len(legacy.youtube_description) > 150  # the long case
    if "yt_hashtag_set" in post.experiment_arms:
        assert body(post.youtube_description) == body(legacy.youtube_description)
    else:
        assert post.youtube_description == legacy.youtube_description
