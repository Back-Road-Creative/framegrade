# ruff: noqa: F811  (fixtures are imported from the adapter test module)
"""A Short's own generated title is reviewable.

The title comes from ``social_generator`` (a stamp -- "Feb 24, 2nd stretch" --
a body, closing hashtags) and the Short review refuses every word nothing backs.
So the review backs the generator's template words by the record: the stamp by
``recording_date``, a weekday by ``recording_date``, a hashtag by the generator's
own tag lists or the clip's region votes. A word the record does not back stays
refused. Synthetic clip data only.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from test_publishing_youtube_adapter import (  # noqa: F401 -- fixtures
    _publish,
    _setup,
    mock_service,
    queue_conn,
)

from gomoveshift.pipeline.final_text_record import FinalTextReviewError
from gomoveshift.publishing.youtube_adapter import build_short_final_text_review

#: A real refused title (queue row 178), its sidecar's date and region vote.
REAL_TITLE = "Feb 24, 2nd stretch: 38 seconds, Polk County #PolkCounty #NorthCarolina #POVDriving"
REGIONS = ("Polk County, North Carolina",)
DATE = "2026-02-24"  # a Tuesday


def _short(record, title=REAL_TITLE, *, date=DATE, regions=REGIONS, plan_hash=""):
    yt = replace(
        record.payloads.youtube,
        title=title,
        description="Drive through Polk County. Filmed February 2026.",
        tags=(),
    )
    return replace(
        record,
        payloads=replace(record.payloads, youtube=yt),
        recording_date=date,
        sampled_regions=tuple(regions),
        reviewed_plan_hash=plan_hash,
    )


@pytest.fixture
def base(mock_service, queue_conn, tmp_path):
    adapter, record, delivery_id, video_path = _setup(mock_service, queue_conn, tmp_path)
    return adapter, record, delivery_id, video_path


def test_the_real_refused_title_is_reviewed_agree(base):
    review = build_short_final_text_review(_short(base[1]))
    assert review["verdict"] == "agree", review["reasons"]


def test_an_empty_plan_hash_is_not_why_a_short_is_refused(base):
    review = build_short_final_text_review(_short(base[1], plan_hash=""))
    assert not any("approval_hash" in r for r in review["reasons"])
    assert review["approval_hash"]


def test_a_plan_hash_the_plan_carries_is_kept(base):
    review = build_short_final_text_review(_short(base[1], plan_hash="f" * 64))
    assert review["approval_hash"] == "f" * 64


@pytest.mark.parametrize(
    "title",
    [
        "Feb 24, 1st stretch: Polk County #PolkCounty",
        "Feb 24, 3rd stretch: Polk County, on a Tuesday #POVDriving",
        "Feb 24, 5th stretch: Polk County",
    ],
)
def test_every_stamped_body_template_is_backed(base, title):
    review = build_short_final_text_review(_short(base[1], title))
    unbacked = [r for r in review["reasons"] if r.startswith("name_unbacked")]
    assert unbacked == []


@pytest.mark.parametrize(
    ("title", "word"),
    [
        ("Mar 24, 2nd stretch: 38 seconds, Polk County", "mar"),  # stamp month != date
        ("Feb 24, 2nd stretch: Polk County, on a Monday", "monday"),  # weekday != date
        ("Feb 24, 2nd stretch: Polk County epic", "epic"),  # a word no template makes
        ("Feb 24, 2nd stretch: Polk County #Tryon", "tryon"),  # a place the clip did not vote
        ("Feb 24, 2nd stretch: Polk County #PolkCounty #SouthCarolina", "southcarolina"),
    ],
)
def test_a_word_the_record_does_not_back_is_still_refused(base, title, word):
    review = build_short_final_text_review(_short(base[1], title))
    assert review["verdict"] != "agree"
    assert any(r.endswith(f":{word}") and "name_unbacked" in r for r in review["reasons"]), review[
        "reasons"
    ]


def test_the_stamp_is_not_backed_by_a_different_day(base):
    review = build_short_final_text_review(_short(base[1], date="2026-02-25"))
    assert any("name_unbacked" in r and r.endswith(":feb") for r in review["reasons"])


def test_the_real_title_publishes_through_the_adapter(base, mock_service, queue_conn, tmp_path):
    from test_w9s_shorts_review import _bind_open  # noqa: F401

    adapter, record, delivery_id, video_path = base
    _publish(adapter, _short(record), queue_conn, delivery_id, video_path)
    assert mock_service.videos.return_value.insert.call_count == 1


def test_a_title_the_review_cannot_back_never_inserts(base, mock_service, queue_conn):
    adapter, record, delivery_id, video_path = base
    with pytest.raises(FinalTextReviewError):
        adapter.publish(
            _short(record, "Feb 24, 2nd stretch: Polk County epic"),
            conn=queue_conn,
            delivery_id=delivery_id,
            video_path=video_path,
        )
    assert mock_service.videos.return_value.insert.call_count == 0


def test_the_place_free_body_backs_its_words_by_the_date_only():
    from gomoveshift.publishing.short_title_review import short_title_filler

    title = "Feb 24, 4th stretch: 12 seconds on a Tuesday"
    assert {"feb", "4th", "stretch", "seconds", "on", "a", "tuesday"} <= set(
        short_title_filler(title, DATE, REGIONS)
    )
    assert "tuesday" not in short_title_filler(title, "2026-02-25", REGIONS)


def test_the_recovery_check_agrees_for_the_real_title_and_names_what_it_refuses(base):
    from gomoveshift.publishing.youtube_adapter import short_review_refusal

    assert short_review_refusal(_short(base[1])) is None
    refusal = short_review_refusal(_short(base[1], "Feb 24, 2nd stretch: Polk County epic"))
    assert refusal and "name_unbacked:short_title:epic" in refusal
