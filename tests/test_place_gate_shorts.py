# ruff: noqa: F811  (pytest fixtures imported then used as parameters)
"""Zero-tolerance place gate for a Short's title, description and tags.

A place named on any of those surfaces holds every retained second of the clip.
Synthetic places only (Harrow, Pellmere, Ashby County, Westmark).
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
from test_place_gate import Geo, _harrow_with_leak
from test_publishing_youtube_adapter import (  # noqa: F401 -- fixtures
    _setup,
    mock_service,
    queue_conn,
)

from gomoveshift.pipeline.final_text_record import FinalTextReviewError, check_final_text
from gomoveshift.pipeline.ost_audit_sidecar import (
    compute_shorts_ost_audit_hash,
    format_shorts_ost_audit,
)
from gomoveshift.pipeline.route_claims import make_short_route_fn, measure_route
from gomoveshift.publishing import plan_io
from gomoveshift.publishing.models import PublishPlan
from gomoveshift.publishing.youtube_adapter import short_final_text_context

_REVIEW = {"description_layout": None, "surfaces": []}


@pytest.fixture(autouse=True)
def _review_accepts(monkeypatch):
    monkeypatch.setattr(
        "gomoveshift.pipeline.final_text_record.verify_review", lambda *a, **k: None
    )


def _leaky_record():
    return measure_route([_harrow_with_leak()], Geo()).record()


def _short(mock_service, queue_conn, tmp_path, *, route_record, title="Drive", desc="Quiet drive"):
    _adapter, record, _did, _vp = _setup(mock_service, queue_conn, tmp_path)
    yt = replace(record.payloads.youtube, title=title, description=desc, tags=("drive",))
    return replace(
        record,
        payloads=replace(record.payloads, youtube=yt),
        sampled_regions=("Ashby County, Westmark",),
        route_record=route_record,
    )


def _check(record, *, title=None, description=None, tags=None):
    yt = record.payloads.youtube
    check_final_text(
        _REVIEW,
        short_final_text_context(record),
        title if title is not None else yt.title,
        description=description if description is not None else yt.description,
        tags=tags if tags is not None else list(yt.tags),
    )


def test_short_title_naming_a_place_with_leaked_seconds_is_refused(
    mock_service, queue_conn, tmp_path
):
    rec = _short(mock_service, queue_conn, tmp_path, route_record=_leaky_record(), title="Harrow")
    with pytest.raises(FinalTextReviewError) as exc:
        _check(rec)
    assert "place_outside_text" in exc.value.reasons
    assert "title:" in str(exc.value)


def test_short_description_naming_a_place_is_refused_before_upload(
    mock_service, queue_conn, tmp_path
):
    rec = _short(
        mock_service, queue_conn, tmp_path, route_record=_leaky_record(), desc="Driving Harrow"
    )
    with pytest.raises(FinalTextReviewError) as exc:
        _check(rec)
    assert "description:" in str(exc.value)


def test_short_tags_are_one_text_so_a_union_covers(mock_service, queue_conn, tmp_path):
    rec = _short(mock_service, queue_conn, tmp_path, route_record=_leaky_record())
    with pytest.raises(FinalTextReviewError):
        _check(rec, tags=["Harrow"])
    _check(rec, tags=["Harrow", "Pellmere"])


def test_short_surfaces_naming_every_place_pass(mock_service, queue_conn, tmp_path):
    rec = _short(mock_service, queue_conn, tmp_path, route_record=_leaky_record())
    _check(rec, title="Harrow and Pellmere", description="Harrow and Pellmere")


def test_legacy_short_without_persisted_bins_is_refused_as_unverifiable(
    mock_service, queue_conn, tmp_path
):
    for legacy in (None, {k: v for k, v in _leaky_record().items() if k != "bins"}):
        rec = _short(mock_service, queue_conn, tmp_path, route_record=legacy)
        with pytest.raises(FinalTextReviewError) as exc:
            _check(rec)
        assert "place_outside_text" in exc.value.reasons
        assert "re-measure" in str(exc.value)


def test_short_route_fn_exposes_the_measured_record():
    pts = [(float(t), 9.5, 20.0) for t in range(0, 300)]
    fn = make_short_route_fn(Geo())
    fn(pts, SimpleNamespace(rank=1, start_sec=0, end_sec=300), "Harrow")
    assert fn.last_record["bins"] and fn.last_record["windows_key"]


def test_sidecar_entry_keeps_the_route_record_out_of_the_text_hash():
    rec = _leaky_record()
    entry = {"rank": 1, "ost_text": "Harrow", "regions": ["Ashby County, Westmark"]}
    plain = format_shorts_ost_audit("2026-10-01", [entry])
    with_rec = format_shorts_ost_audit("2026-10-01", [{**entry, "route_record": rec}])
    assert with_rec["shorts"][0]["route_record"] == rec
    assert compute_shorts_ost_audit_hash(with_rec) == compute_shorts_ost_audit_hash(plain)


def test_route_record_round_trips_through_the_plan(mock_service, queue_conn, tmp_path):
    rec = _short(mock_service, queue_conn, tmp_path, route_record=_leaky_record())
    raw = json.loads(json.dumps(asdict(PublishPlan(shorts=(rec,))), default=str))
    path = tmp_path / "plan.json"
    assert plan_io._record(raw["shorts"][0], path, 0).route_record == rec.route_record
    del raw["shorts"][0]["route_record"]
    assert plan_io._record(raw["shorts"][0], path, 0).route_record is None
