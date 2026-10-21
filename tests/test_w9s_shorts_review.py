# ruff: noqa: F811  (fixtures are imported from the adapter test module)
"""A Short's YouTube insert carries its final-text
review, and a Short whose text the review does not cover is refused with the
insert count at zero. Synthetic data only ("County Kerry", "Pellmere" are
invented here); only the YouTube client is a stub.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from seo_manager.analytics.final_text_review import _fingerprint_of
from test_publishing_youtube_adapter import (  # noqa: F401 -- fixtures
    _publish,
    _setup,
    mock_service,
    queue_conn,
)

from gomoveshift.pipeline.drive_phases import youtube_uploader as youtube_uploader_module
from gomoveshift.pipeline.final_text_record import FinalTextReviewError, PublicationGateError
from gomoveshift.publishing.queue import find_delivery
from gomoveshift.publishing.youtube_adapter import build_short_final_text_review

TITLE = "County Kerry drive"


@pytest.fixture(autouse=True)
def _bind_open(monkeypatch):
    # The publication gate refuses every insert while its bind is off; these
    # tests open it inside the test only.
    monkeypatch.setattr(youtube_uploader_module, "FINAL_TEXT_BIND_ON", True, raising=False)


DESC = "A quiet drive through County Kerry."


def _with(record, *, title=TITLE, description=DESC, regions=("County Kerry",), tags=()):
    yt = replace(record.payloads.youtube, title=title, description=description, tags=tags)
    return replace(
        record, payloads=replace(record.payloads, youtube=yt), sampled_regions=tuple(regions)
    )


def _prepared(mock_service, queue_conn, tmp_path, **over):
    adapter, record, delivery_id, video_path = _setup(mock_service, queue_conn, tmp_path)
    return adapter, _with(record, **over), delivery_id, video_path


def _inserts(mock_service):
    return mock_service.videos.return_value.insert.call_count


def _no_confirmed(tmp_path):
    assert not list(tmp_path.rglob("*.confirmed"))


def test_i3_happy_path_inserts_once_with_the_reviewed_text(mock_service, queue_conn, tmp_path):
    adapter, record, did, vp = _prepared(mock_service, queue_conn, tmp_path)
    review = build_short_final_text_review(record)
    assert review["verdict"] == "agree", review["reasons"]

    _publish(adapter, record, queue_conn, did, vp)

    assert _inserts(mock_service) == 1
    body = mock_service.videos.return_value.insert.call_args.kwargs["body"]
    assert body["snippet"]["title"] == review["final_title"] == TITLE
    _no_confirmed(tmp_path)


def _refuse(mock_service, queue_conn, tmp_path, record, adapter, did, vp, **kw):
    with pytest.raises(FinalTextReviewError) as err:
        adapter.publish(record, conn=queue_conn, delivery_id=did, video_path=vp, **kw)
    assert _inserts(mock_service) == 0
    row = find_delivery(queue_conn, record.project_name, record.short_index, "youtube")
    assert row["status"] == "queued"  # refused before the lease and SUBMITTING
    _no_confirmed(tmp_path)
    return err.value


def test_i4_unsupported_location_in_the_title(mock_service, queue_conn, tmp_path):
    adapter, record, did, vp = _prepared(mock_service, queue_conn, tmp_path, title="Pellmere drive")
    _refuse(mock_service, queue_conn, tmp_path, record, adapter, did, vp)


def test_i4_title_naming_no_place(mock_service, queue_conn, tmp_path):
    adapter, record, did, vp = _prepared(mock_service, queue_conn, tmp_path, title="Golden drive")
    _refuse(mock_service, queue_conn, tmp_path, record, adapter, did, vp)


@pytest.mark.parametrize("description", ["", "   "])
def test_i4_missing_description(mock_service, queue_conn, tmp_path, description):
    adapter, record, did, vp = _prepared(
        mock_service, queue_conn, tmp_path, description=description
    )
    err = _refuse(mock_service, queue_conn, tmp_path, record, adapter, did, vp)
    assert "description_missing" in err.reasons


def test_i4_one_byte_title_edit_after_review(mock_service, queue_conn, tmp_path):
    adapter, record, did, vp = _prepared(mock_service, queue_conn, tmp_path)
    review = build_short_final_text_review(record)
    edited = _with(record, title="County Kerry Drive")  # one byte: d -> D
    err = _refuse(
        mock_service, queue_conn, tmp_path, edited, adapter, did, vp, final_text_review=review
    )
    assert "stale:title" in err.reasons


def test_i4_stale_review_version(mock_service, queue_conn, tmp_path):
    adapter, record, did, vp = _prepared(mock_service, queue_conn, tmp_path)
    review = build_short_final_text_review(record)
    review["review_version"] = "final-text-review-2"
    review["fingerprint"] = _fingerprint_of(review)  # intact, only older
    err = _refuse(
        mock_service, queue_conn, tmp_path, record, adapter, did, vp, final_text_review=review
    )
    assert "stale:review_version" in err.reasons


@pytest.mark.parametrize("change", ["media", "regions"])
def test_i4_resumed_state_mismatch(mock_service, queue_conn, tmp_path, change):
    adapter, record, did, vp = _prepared(mock_service, queue_conn, tmp_path)
    review = build_short_final_text_review(record)  # saved before the resume
    if change == "media":
        resumed = replace(record, media=replace(record.media, sha256="b" * 64))
    else:
        resumed = replace(record, sampled_regions=("County Kerry", "County Cork"))
    err = _refuse(
        mock_service, queue_conn, tmp_path, resumed, adapter, did, vp, final_text_review=review
    )
    assert any(r.startswith("stale:") for r in err.reasons), err.reasons


def _sent(mock_service):
    return mock_service.videos.return_value.insert.call_args.kwargs["body"]["snippet"]


def test_i8_description_sent_is_the_reviewed_place_line(mock_service, queue_conn, tmp_path):
    """The generated prose (and its parent link) is not what ships: the sent
    description is Waypost's composed text, whatever the record carried."""
    adapter, record, did, vp = _prepared(
        mock_service, queue_conn, tmp_path, description="desc!\n\nhttps://youtu.be/abc"
    )
    _publish(adapter, record, queue_conn, did, vp)
    assert _inserts(mock_service) == 1
    assert _sent(mock_service)["description"] == "Drive through County Kerry"


def test_i4_tag_byte_edit_is_bound(mock_service, queue_conn, tmp_path):
    adapter, record, did, vp = _prepared(mock_service, queue_conn, tmp_path, tags=("County Kerry",))
    review = build_short_final_text_review(record)
    assert review["verdict"] == "agree", review["reasons"]
    edited = _with(record, tags=("County Kerrx",))
    err = _refuse(
        mock_service, queue_conn, tmp_path, edited, adapter, did, vp, final_text_review=review
    )
    assert "stale:tags" in err.reasons


@pytest.mark.parametrize("tag", ["kerry", "driving", "travel", "Bog Road"])
def test_i8_unreviewable_tag_refuses_the_short(mock_service, queue_conn, tmp_path, tag):
    adapter, record, did, vp = _prepared(mock_service, queue_conn, tmp_path, tags=("drive", tag))
    err = _refuse(mock_service, queue_conn, tmp_path, record, adapter, did, vp)
    assert f"tag_unreviewable:{tag}" in err.reasons


def test_i8_template_and_place_tags_ship_exactly(mock_service, queue_conn, tmp_path):
    tags = ("drive", "POV driving", "County Kerry")
    adapter, record, did, vp = _prepared(mock_service, queue_conn, tmp_path, tags=tags)
    _publish(adapter, record, queue_conn, did, vp)
    assert _sent(mock_service)["tags"] == list(tags)


def test_i8_description_naming_no_clip_place_refuses(mock_service, queue_conn, tmp_path):
    adapter, record, did, vp = _prepared(mock_service, queue_conn, tmp_path, title="Golden drive")
    _refuse(mock_service, queue_conn, tmp_path, record, adapter, did, vp)


def test_t1_flag_absent_never_reviews_and_holds_the_insert(
    mock_service, queue_conn, tmp_path, monkeypatch
):
    # An absent FINAL_TEXT_BIND_ON reads as off, so the publication
    # gate holds the Short before the lease (it stays queued) and no review runs.
    monkeypatch.delattr(youtube_uploader_module, "FINAL_TEXT_BIND_ON", raising=False)
    from gomoveshift.publishing import youtube_adapter

    def _no_review(record):
        raise AssertionError("the Short review ran with the bind flag absent")

    monkeypatch.setattr(youtube_adapter, "build_short_final_text_review", _no_review)
    adapter, record, did, vp = _prepared(mock_service, queue_conn, tmp_path, title="Pellmere drive")

    err = _refuse(mock_service, queue_conn, tmp_path, record, adapter, did, vp)

    assert isinstance(err, PublicationGateError)
    assert err.reasons == ["final_text_bind_off"]
    assert "publication_gate: final_text_bind_off" in str(err)


def test_gps_less_short_has_nothing_to_review_and_publishes(
    mock_service, queue_conn, tmp_path, monkeypatch
):
    # Parity with a GPS-less LF (``needs_review`` is False): a Short with no
    # clip GPS votes carries no place evidence, so no review runs; the publication
    # gate (open here) still decides the insert.
    from gomoveshift.publishing import youtube_adapter

    def _no_review(record):
        raise AssertionError("a GPS-less Short was reviewed")

    monkeypatch.setattr(youtube_adapter, "build_short_final_text_review", _no_review)
    adapter, record, did, vp = _prepared(
        mock_service, queue_conn, tmp_path, title="Golden hour drive", regions=()
    )

    _publish(adapter, record, queue_conn, did, vp)

    assert _inserts(mock_service) == 1
    _no_confirmed(tmp_path)


def test_i8_no_named_place_refuses_with_its_own_reason(mock_service, queue_conn, tmp_path):
    """No fallback text: a Short whose title names no clip place is refused for it."""
    adapter, record, did, vp = _prepared(mock_service, queue_conn, tmp_path, title="Golden drive")
    err = _refuse(mock_service, queue_conn, tmp_path, record, adapter, did, vp)
    assert err.reasons == ["description_no_clip_place"]


def test_i8_every_base_tag_is_a_waypost_tag_template():
    """A base tag without a template refuses every production Short."""
    from seo_manager.analytics.final_text_surfaces import all_templates

    from gomoveshift.pipeline.social_generator import BASE_TAGS

    templates = {t for tid, t in all_templates().items() if tid.startswith("tag.")}
    assert set(BASE_TAGS) <= templates, sorted(set(BASE_TAGS) - templates)
