# ruff: noqa: F811  (fixtures are imported from the adapter test module)
"""A Short's tags are limited to what Waypost can review -- its tag
templates, the clip's own county names and the clip's own GPS road names -- and
the clip's roads travel sidecar -> record -> ``kind=road`` evidence. Long-form
tags are untouched. Synthetic places only (Harrow, Pellmere, Ostry, Ashby County,
Brindlemoor, Quillhaven); only the YouTube client is a stub.
"""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from seo_manager.analytics.final_text_surfaces import all_templates
from test_publishing_youtube_adapter import (  # noqa: F401 -- fixtures
    _publish,
    _setup,
    mock_service,
    queue_conn,
)
from test_w9s_shorts_review import _inserts, _refuse

from gomoveshift.pipeline.drive_phases import youtube_uploader as youtube_uploader_module
from gomoveshift.pipeline.final_text_record import FinalTextReviewError, reviewable_short_tags
from gomoveshift.pipeline.ost_audit_sidecar import (
    compute_shorts_ost_audit_hash,
    format_shorts_ost_audit,
)
from gomoveshift.pipeline.social_generator import SocialGenerator
from gomoveshift.pipeline.validation import content_gates
from gomoveshift.publishing.youtube_adapter import (
    build_short_final_text_review,
    short_final_text_context,
)

TITLE = "Ashby County drive"
DESC = "A quiet drive through Ashby County."
TEMPLATE_TAGS = tuple(
    text for tid, text in sorted(all_templates().items()) if tid.startswith("tag.")
)[:3]


@pytest.fixture(autouse=True)
def _bind_open(monkeypatch):
    monkeypatch.setattr(youtube_uploader_module, "FINAL_TEXT_BIND_ON", True, raising=False)


def _with(record, *, tags, roads=("Harrow Lane",), regions=("Ashby County, Ostry",)):
    yt = replace(record.payloads.youtube, title=TITLE, description=DESC, tags=tuple(tags))
    return replace(
        record,
        payloads=replace(record.payloads, youtube=yt),
        sampled_regions=tuple(regions),
        sampled_roads=tuple(roads),
    )


def _prepared(mock_service, queue_conn, tmp_path, **over):
    adapter, record, did, vp = _setup(mock_service, queue_conn, tmp_path)
    return adapter, _with(record, **over), did, vp


# -- adapter: road evidence -------------------------------------------------


def test_clip_road_tag_is_reviewed_and_published(mock_service, queue_conn, tmp_path):
    tags = (*TEMPLATE_TAGS, "Ashby County", "Harrow Lane")
    adapter, record, did, vp = _prepared(mock_service, queue_conn, tmp_path, tags=tags)
    review = build_short_final_text_review(record)
    assert review["verdict"] == "agree", review["reasons"]

    _publish(adapter, record, queue_conn, did, vp)

    assert _inserts(mock_service) == 1
    sent = mock_service.videos.return_value.insert.call_args.kwargs["body"]["snippet"]
    assert "Harrow Lane" in sent["tags"]


def test_road_not_in_the_clip_record_is_refused(mock_service, queue_conn, tmp_path):
    adapter, record, did, vp = _prepared(
        mock_service, queue_conn, tmp_path, tags=("Pellmere Road",), roads=("Harrow Lane",)
    )
    err = _refuse(mock_service, queue_conn, tmp_path, record, adapter, did, vp)
    assert "tag_unreviewable:Pellmere Road" in err.reasons


def test_clip_without_gps_road_refuses_a_road_tag(mock_service, queue_conn, tmp_path):
    adapter, record, did, vp = _prepared(
        mock_service, queue_conn, tmp_path, tags=("Harrow Lane",), roads=()
    )
    err = _refuse(mock_service, queue_conn, tmp_path, record, adapter, did, vp)
    assert "tag_unreviewable:Harrow Lane" in err.reasons


def test_filler_tag_still_refuses_with_zero_inserts(mock_service, queue_conn, tmp_path):
    adapter, record, did, vp = _prepared(
        mock_service, queue_conn, tmp_path, tags=("dashcam footage",)
    )
    err = _refuse(mock_service, queue_conn, tmp_path, record, adapter, did, vp)
    assert "tag_unreviewable:dashcam footage" in err.reasons


def test_changed_roads_after_review_are_stale(mock_service, queue_conn, tmp_path):
    adapter, record, did, vp = _prepared(
        mock_service, queue_conn, tmp_path, tags=(*TEMPLATE_TAGS, "Harrow Lane")
    )
    review = build_short_final_text_review(record)
    assert (
        short_final_text_context(record)["plan_fingerprint"]
        != short_final_text_context(
            replace(record, sampled_roads=("Harrow Lane", "Quillhaven Road"))
        )["plan_fingerprint"]
    )
    resumed = replace(record, sampled_roads=("Brindlemoor Road",))
    err = _refuse(
        mock_service, queue_conn, tmp_path, resumed, adapter, did, vp, final_text_review=review
    )
    assert any(r.startswith("stale:") for r in err.reasons), err.reasons


# -- record round trip ------------------------------------------------------


def test_sampled_roads_round_trip_and_legacy_plans_load(tmp_path):
    from gomoveshift.publishing import plan_io

    assert hasattr(plan_io, "_str_tuple_or_empty")
    assert plan_io._str_tuple_or_empty({}, "sampled_roads", tmp_path, "x") == ()
    assert plan_io._str_tuple_or_empty(
        {"sampled_roads": ["Harrow Lane"]}, "sampled_roads", tmp_path, "x"
    ) == ("Harrow Lane",)


# -- upstream: sidecar and region vote carry the roads ----------------------


def test_sidecar_entry_keeps_roads_out_of_the_text_hash():
    entry = {"rank": 1, "ost_text": "Ashby County", "regions": ["Ashby County, Ostry"]}
    plain = format_shorts_ost_audit("2026-10-01", [entry])
    roaded = format_shorts_ost_audit("2026-10-01", [{**entry, "roads": ["Harrow Lane", " ", "A1"]}])
    assert roaded["shorts"][0]["roads"] == ["Harrow Lane", "A1"]
    assert plain["shorts"][0]["roads"] == []
    assert compute_shorts_ost_audit_hash(roaded) == compute_shorts_ost_audit_hash(plain)


def test_voted_regions_carry_the_clips_own_road_votes(monkeypatch):
    dom = SimpleNamespace(
        votes={"Ashby County|Ostry": 1.0},
        road_votes={"Harrow Lane": 7, "Pellmere Road": 2},
        road="Harrow Lane",
    )
    monkeypatch.setattr("gomoveshift.pipeline.title_builder.dominant_location", lambda *a, **k: dom)
    fn = content_gates._make_short_regions_fn(object())
    out = fn([object()], SimpleNamespace(rank=1))
    assert list(out) == ["Ashby County, Ostry"]
    assert out.roads == ["Harrow Lane", "Pellmere Road"]


def test_voted_regions_without_road_votes_have_no_roads(monkeypatch):
    dom = SimpleNamespace(votes={"Ashby County|Ostry": 1.0}, road_votes={}, road=None)
    monkeypatch.setattr("gomoveshift.pipeline.title_builder.dominant_location", lambda *a, **k: dom)
    out = content_gates._make_short_regions_fn(object())([object()], SimpleNamespace(rank=1))
    assert out.roads == []


# -- generator: Short tags narrowed, LF tags untouched ----------------------


def _gen():
    return SocialGenerator(location="Harrow County, Ostry", project_name="p_lf01", series_part=1)


def _tags(post):
    return [t.strip() for t in post.youtube_tags.split(",") if t.strip()]


def test_short_tags_are_templates_counties_and_clip_roads_only():
    variant = _gen()._per_short_variant(
        "Pellmere, Ashby County", ["Ashby County, Ostry"], roads=["Harrow Lane"]
    )
    tags = _tags(variant.generate_post(1, None, global_index=1, title_facts=None))
    templates = {t for k, t in all_templates().items() if k.startswith("tag.")}
    assert tags
    assert set(tags) <= templates | {"Ashby County", "Harrow Lane"}
    assert {"Ashby County", "Harrow Lane"} <= set(tags)
    assert reviewable_short_tags(tags, ["Ashby County, Ostry"], ["Harrow Lane"]) == tags


def test_short_without_clip_roads_emits_no_road_tag():
    variant = _gen()._per_short_variant("Pellmere, Ashby County", ["Ashby County, Ostry"])
    tags = _tags(variant.generate_post(1, None, global_index=1, title_facts=None))
    assert "Harrow Lane" not in tags and "Harrow" not in " ".join(tags)


def test_long_form_tags_are_unchanged():
    gen = _gen()
    before = _tags(gen.generate_post(1, None, global_index=1, title_facts=None))
    assert "dashcam footage" in before  # LF filler stays
    again = _tags(_gen().generate_post(1, None, global_index=1, title_facts=None))
    assert before == again


def test_reviewable_filter_drops_only_unreviewable():
    kept = reviewable_short_tags(
        [*TEMPLATE_TAGS, "dashcam footage", "Ashby County", "Elsewhere Road", "Harrow Lane"],
        ["Ashby County, Ostry"],
        ["Harrow Lane"],
    )
    assert kept == [*TEMPLATE_TAGS, "Ashby County", "Harrow Lane"]
