"""Meta captions learn their opener order from Meta performance.

The nightly ``metrics collect`` joins each tracked Facebook/Instagram post's
day-7 snapshot to the opener template id its plan sidecar recorded and writes
``caption_evidence.json`` beside the queue. The caption generator hands that
evidence and its shuffled opener order to Waypost's ``rank_captions``, walks the
re-ranked order, and records the ranking on the post. The anti-spam freshness
split still runs after the ranking, so a winning opener is never reused inside
the gate's window."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from seo_manager.analytics.caption_ranking import (
    FACEBOOK_REEL_INITIAL_PLAYS_DAY7,
    INSTAGRAM_REEL_REACH_DAY7,
    RANKED,
    CaptionEvidence,
    CaptionObservation,
)

from gomoveshift.pipeline import longform_metadata
from gomoveshift.pipeline.social_generator import (
    _PLATFORM_CAPTION_TEMPLATES,
    SocialGenerator,
    _opener_keys,
    caption_template_id,
)
from gomoveshift.tracking.caption_evidence import (
    EVIDENCE_FILE,
    caption_evidence,
    load_caption_evidence,
    write_caption_evidence,
)
from gomoveshift.tracking.schema import ShortRecord, TrackingRecord

NOW = datetime(2026, 12, 1, 6, 0, tzinfo=timezone.utc)
EVENT = "2026-11-10T15:00:00+00:00"


def _obs(platform, counters, age=timedelta(days=7), status="measured", event=EVENT):
    return {
        "platform": platform,
        "status": status,
        "event_at": event,
        "fetched_at": (datetime.fromisoformat(event) + age).isoformat(),
        "cumulative": counters,
    }


def _short(index, fb=None, ig=None):
    s = ShortRecord(
        index=index, posted_at=EVENT, posted_at_source="platform_confirmed", platform_ids={}
    )
    s.observation_log = {}
    if fb is not None:
        s.platform_ids["facebook"] = f"fb{index}"
        s.observation_log["facebook"] = fb
    if ig is not None:
        s.platform_ids["instagram"] = f"ig{index}"
        s.observation_log["instagram"] = ig
    return s


def _evidence(shorts, tags):
    rec = TrackingRecord(project="p", shorts=shorts)
    return caption_evidence([rec], lambda project: tags if project == "p" else None, NOW)


def test_facebook_counts_initial_plays_and_instagram_counts_reach_per_opener():
    shorts = [
        _short(1, fb=[_obs("facebook", {"reel_initial_plays": 40, "reach": None})]),
        _short(2, fb=[_obs("facebook", {"reel_initial_plays": 60})]),
        _short(3, ig=[_obs("instagram", {"impressions": 103})]),
    ]
    tags = {1: {"facebook": "aaa"}, 2: {"facebook": "aaa"}, 3: {"instagram": "bbb"}}
    ev = _evidence(shorts, tags)
    assert ev.as_of == NOW.isoformat()
    by = {(o.platform, o.template_id): o for o in ev.observations}
    fb, ig = by[("facebook", "aaa")], by[("instagram", "bbb")]
    assert (fb.metric_source, fb.median, fb.n) == (FACEBOOK_REEL_INITIAL_PLAYS_DAY7, 50, 2)
    assert (ig.metric_source, ig.median, ig.n) == (INSTAGRAM_REEL_REACH_DAY7, 103, 1)
    assert fb.window_start == fb.window_end == "2026-11-17T15:00:00+00:00"


def test_only_a_measured_snapshot_near_day_7_counts_and_an_untagged_post_never_does():
    shorts = [
        _short(1, fb=[_obs("facebook", {"reel_initial_plays": 9}, age=timedelta(days=3))]),
        _short(2, fb=[_obs("facebook", {"reel_initial_plays": 9}, status="provider_error")]),
        _short(3, fb=[_obs("facebook", {"reel_initial_plays": None})]),
        _short(4, fb=[_obs("facebook", {"reel_initial_plays": 9})]),  # no tag
        _short(
            5,
            fb=[
                _obs("facebook", {"reel_initial_plays": 70}, age=timedelta(days=7, hours=10)),
                _obs("facebook", {"reel_initial_plays": 72}, age=timedelta(days=7, hours=1)),
            ],
        ),
    ]
    tags = {i: {"facebook": "aaa"} for i in (1, 2, 3, 5)}
    (only,) = _evidence(shorts, tags).observations
    assert (only.median, only.n) == (72, 1)


def test_an_unreadable_sidecar_costs_that_project_nothing_else():
    rec = TrackingRecord(project="gone", shorts=[_short(1, fb=[_obs("facebook", {"x": 1})])])
    assert caption_evidence([rec], lambda project: None, NOW).observations == ()


def test_the_evidence_file_round_trips(tmp_path):
    ev = _evidence(
        [_short(1, fb=[_obs("facebook", {"reel_initial_plays": 5})])], {1: {"facebook": "aaa"}}
    )
    path = write_caption_evidence(tmp_path, ev)
    assert path == tmp_path / EVIDENCE_FILE
    assert load_caption_evidence(tmp_path) == ev


def test_no_or_corrupt_evidence_file_is_no_evidence(tmp_path):
    assert load_caption_evidence(tmp_path) is None
    (tmp_path / EVIDENCE_FILE).write_text("{not json")
    assert load_caption_evidence(tmp_path) is None


# --- the generator walks the re-ranked opener order -----------------------------


@pytest.fixture
def titles(tmp_path):
    path = tmp_path / "titles.yaml"
    path.write_text(
        "".join(f"{i}:\n  title: Rolling the Dingle loop number {i}\n" for i in range(1, 5))
    )
    return path


@pytest.fixture
def storage(tmp_path, monkeypatch):
    seo = tmp_path / "seo"
    seo.mkdir()
    monkeypatch.setattr(longform_metadata, "DEFAULT_SEO_STORAGE", seo)
    return seo


def _gen(titles):
    return SocialGenerator(
        titles_file=titles,
        location="Dingle Peninsula, County Kerry, Ireland",
        project_name="proj",
        series_part=1,
    )


def _opener_of(post, platform):
    ids = {caption_template_id(t): t for t in _PLATFORM_CAPTION_TEMPLATES[platform]}
    return ids[post.caption_templates[platform]]


def _evidence_favouring(platform, winner, metric, others=3):
    """``winner`` far above the pool, ``others`` openers at the pool's middle."""
    pool = [caption_template_id(t) for t in _PLATFORM_CAPTION_TEMPLATES[platform]]
    rest = [t for t in pool if t != caption_template_id(winner)][:others]
    window = ("2026-11-01T00:00:00+00:00", "2026-11-28T00:00:00+00:00")
    obs = [CaptionObservation(platform, caption_template_id(winner), metric, 900.0, 12, *window)]
    obs += [CaptionObservation(platform, t, metric, 100.0, 12, *window) for t in rest]
    return CaptionEvidence(NOW.isoformat(), tuple(obs))


def test_without_evidence_the_opener_order_is_unchanged_and_no_ranking_is_recorded(titles, storage):
    post = _gen(titles).generate_all(num_shorts=1)[0]
    assert post.caption_rankings == {}


def test_a_proven_facebook_opener_moves_to_the_front_and_the_ranking_is_recorded(titles, storage):
    baseline = _gen(titles).generate_all(num_shorts=4)
    used = {_opener_of(p, "facebook") for p in baseline}
    winner = next(t for t in _PLATFORM_CAPTION_TEMPLATES["facebook"] if t not in used)
    write_caption_evidence(
        storage, _evidence_favouring("facebook", winner, FACEBOOK_REEL_INITIAL_PLAYS_DAY7)
    )

    post = _gen(titles).generate_all(num_shorts=1)[0]
    assert _opener_of(post, "facebook") == winner
    record = post.caption_rankings["facebook"]
    assert record["reason"] == RANKED and record["order"][0] == caption_template_id(winner)
    assert json.loads(json.dumps(post.to_dict()))["caption_rankings"] == post.caption_rankings
    # YouTube and TikTok openers are not Meta's to rank.
    assert set(post.caption_rankings) <= {"facebook", "instagram"}


def test_the_anti_spam_freshness_split_still_outranks_the_evidence(titles, storage, monkeypatch):
    winner = _PLATFORM_CAPTION_TEMPLATES["facebook"][0]
    write_caption_evidence(
        storage, _evidence_favouring("facebook", winner, FACEBOOK_REEL_INITIAL_PLAYS_DAY7)
    )
    gen = _gen(titles)
    opener = winner.format_map(
        {"location": gen.social_location, "terrain_desc": gen.terrain_desc, "weather_emoji": ""}
    ).strip()
    taken = _opener_keys({"facebook": [opener]})
    monkeypatch.setattr(SocialGenerator, "_spooled_opener_keys", lambda self: taken)
    post = gen.generate_all(num_shorts=1)[0]
    assert _opener_of(post, "facebook") != winner


def test_refused_evidence_leaves_the_rotation_alone(titles, storage):
    baseline = _opener_of(_gen(titles).generate_all(num_shorts=1)[0], "facebook")
    (Path(storage) / EVIDENCE_FILE).write_text(
        json.dumps({"as_of": "not a time", "observations": []})
    )
    post = _gen(titles).generate_all(num_shorts=1)[0]
    assert _opener_of(post, "facebook") == baseline and post.caption_rankings == {}


# --- the nightly collect writes the evidence file ---------------------------------


def test_nightly_collect_writes_the_evidence_file_from_the_plan_sidecars(
    tmp_path, monkeypatch, capsys
):
    from dataclasses import replace
    from types import SimpleNamespace

    from test_cli_enqueue_plan import _record, _sidecar

    from gomoveshift.cli.handlers import metrics

    _sidecar(tmp_path, replace(_record(short_index=1), caption_templates={"facebook": "aaa"}))
    rec = TrackingRecord(
        project="proj", shorts=[_short(1, fb=[_obs("facebook", {"reel_initial_plays": 41})])]
    )
    monkeypatch.setattr(
        metrics, "TrackingStore", lambda: SimpleNamespace(load_all_records=lambda: [rec])
    )
    metrics._write_caption_evidence(SimpleNamespace(seo=SimpleNamespace(storage_path=tmp_path)))

    (only,) = load_caption_evidence(tmp_path).observations
    assert (only.platform, only.template_id, only.median, only.n) == ("facebook", "aaa", 41, 1)
    out = capsys.readouterr().out
    assert (
        f"CAPTION_EVIDENCE platform=facebook metric={FACEBOOK_REEL_INITIAL_PLAYS_DAY7} "
        "openers=1 posts=1" in out
    )


def test_a_failed_evidence_write_never_fails_the_collection(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from gomoveshift.cli.handlers import metrics

    def boom():
        raise OSError("store unreadable")

    monkeypatch.setattr(metrics, "TrackingStore", boom)
    metrics._write_caption_evidence(SimpleNamespace(seo=SimpleNamespace(storage_path=tmp_path)))
    assert load_caption_evidence(tmp_path) is None
