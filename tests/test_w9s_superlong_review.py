"""Every superlong part's insert carries that
part's final-text review, and a part whose text the review does not cover is
refused with the insert count at zero. ``--superlong`` off (the default) uploads
no part. Synthetic places only ("Harrow", "Pellmere", "Ostry" are invented);
only the YouTube client, the slot clock and the encode are stubs.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from seo_manager.analytics.final_text_review import _fingerprint_of
from seo_manager.analytics.final_text_surfaces import compose_description, compose_tags
from test_retained_gps_claims import _BandGeocoder
from test_superlong_upload import _FrozenDatetime
from test_playlist_idempotency import (  # noqa: F401 -- fixture
    _build_mock_service,
    _make_uploader,
    fake_creds_dict,
)

from _state_helpers import make_state
from gomoveshift.config import Config
from gomoveshift.pipeline import final_text_record
from gomoveshift.pipeline.drive_phases import external_systems_coordinator as esc
from gomoveshift.pipeline.drive_phases import youtube_uploader as youtube_uploader_module
from gomoveshift.pipeline.final_text_record import FinalTextReviewError, final_text_context
from gomoveshift.pipeline.route_claims import RouteWindow, measure_route, route_claim_outcome

DATE = "2026-03-01"
TITLE = "Harrow to Pellmere"


@pytest.fixture(autouse=True)
def _place_gate_off(monkeypatch):
    # These fixtures' reviewed descriptions name only Harrow while 300 s of the
    # footage is in Pellmere; the zero-tolerance place gate (test_place_gate.py)
    # correctly refuses that, and these tests are about the reviewed-text bind.
    monkeypatch.setattr("gomoveshift.pipeline.final_text_record._place_gate", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def _bind_open(monkeypatch):
    # The publication gate refuses every insert while its bind is off; open it in-test only.
    monkeypatch.setattr(youtube_uploader_module, "FINAL_TEXT_BIND_ON", True, raising=False)


def _claim(title=TITLE):
    fixes = [(float(t), 9.5 if t < 300 else 10.2, 20.0) for t in range(600)]
    ev = measure_route([RouteWindow("part1.1", 0.0, 600.0, fixes)], _BandGeocoder())
    return route_claim_outcome(1, title, ev, chosen=title, reason="")


def _record(tmp_path, index=1, **over):
    path = tmp_path / f"sl_{index:02d}.mp4"
    path.write_bytes(b"\x00\x00\x00\x20ftypiso5")
    rec = {
        "path": str(path),
        "part_index": index,
        "part_count": 2,
        "duration_sec": 600.0,
        "title": TITLE,
        "description": "A whole-day drive.",
        "location": TITLE,
        "has_gps": True,
        "route_claim": _claim(),
    }
    rec.update(over)
    return rec


def _state(records, **over):
    fields = {
        "region": "usa",
        "episode_files": [Path("/footage/ep0.mp4")],
        "ost_audit_passed": {DATE: True},
        "ost_audit_hash": {DATE: "audit-h1"},
        "superlong_parts": {DATE: {"part_count": len(records), "records": records}}
        if records
        else None,
    }
    fields.update(over)
    return make_state(**fields)


def _run(tmp_path, fake_creds_dict, state):  # noqa: F811
    svc = _build_mock_service()
    svc.videos.return_value.insert.return_value.next_chunk.side_effect = [
        (None, {"id": "SLVID1"}),
        (None, {"id": "SLVID2"}),
    ]
    up, _ = _make_uploader(svc, fake_creds_dict)
    config = Config()
    config.seo.storage_path = str(tmp_path)
    with (
        patch.object(youtube_uploader_module, "MediaFileUpload", return_value=object()),
        patch.object(esc, "datetime", _FrozenDatetime),
        patch.object(
            esc, "_recheck_publish_at_after_upload", side_effect=lambda **kw: kw["record"]
        ),
        patch.object(esc, "_book_lf_upload_in_ledger"),
    ):
        esc._upload_superlong_parts_to_youtube(
            yt=up,
            state=state,
            output_base=tmp_path,
            config=config,
            save_state=None,
            project_base="proj",
        )
    assert not list(tmp_path.rglob("*.confirmed"))
    return svc


def _inserts(svc):
    return svc.videos.return_value.insert.call_count


def _review_for(state, record):
    """The review a part would carry, as the coordinator builds it."""
    meta = {"location": record["location"]}
    esc.superlong_final_text_inputs(state, f"{DATE}:{record['part_index']:02d}", record, meta)
    return meta["final_text_review"]


def test_i2_superlong_on_inserts_once_per_part_with_reviewed_text(tmp_path, fake_creds_dict):  # noqa: F811
    records = [_record(tmp_path, 1), _record(tmp_path, 2)]
    state = _state(records)
    assert _review_for(state, records[0])["verdict"] == "agree"

    svc = _run(tmp_path, fake_creds_dict, state)

    assert _inserts(svc) == 2
    titles = [
        c.kwargs["body"]["snippet"]["title"] for c in svc.videos.return_value.insert.call_args_list
    ]
    assert titles == [TITLE, TITLE]


def test_i2_superlong_off_by_default_uploads_no_part(tmp_path, fake_creds_dict):  # noqa: F811
    assert Config().drive.superlong_enabled is False
    svc = _run(tmp_path, fake_creds_dict, _state([]))
    assert _inserts(svc) == 0


def _refusal_reasons(caplog) -> set:
    """Every reason code the coordinator logged for a refused part."""
    found = set()
    for rec in caplog.records:
        msg = rec.getMessage()
        if "upload error" not in msg:
            continue
        head = msg.split("; next:", 1)[0]
        for group in re.findall(r"\(([^)]*)\)", head):
            found.update(p.strip() for p in group.split(","))
        for tail in re.findall(r"verdict:refused: ([^;]*)", head):
            found.update(p.strip() for p in tail.split(","))
    return found


def _refused(tmp_path, fake_creds_dict, caplog, record, reason, state=None):  # noqa: F811
    caplog.set_level(logging.ERROR)
    svc = _run(tmp_path, fake_creds_dict, state or _state([record]))
    assert _inserts(svc) == 0
    reasons = _refusal_reasons(caplog)
    assert reason in reasons, sorted(reasons)
    return reasons


def test_w9c_valid_part_carries_waypost_description_entries(tmp_path):
    rec = _record(tmp_path)
    meta = {"location": rec["location"]}
    esc.superlong_final_text_inputs(_state([rec]), f"{DATE}:01", rec, meta)
    assert meta["final_text_review"]["verdict"] == "agree"
    assert meta["final_text_review"]["description_layout"] == "gms-drive-1"
    assert "drive_text_refusal" not in meta
    assert "Location: Harrow" in compose_description(meta["final_text_description"])


def test_w9c_sent_text_is_the_composed_text(tmp_path, fake_creds_dict):  # noqa: F811
    rec = _record(tmp_path)
    meta = {"location": rec["location"]}
    esc.superlong_final_text_inputs(_state([rec]), f"{DATE}:01", rec, meta)
    entries = meta["final_text_description"]
    svc = _run(tmp_path, fake_creds_dict, _state([rec, _record(tmp_path, 2)]))
    body = svc.videos.return_value.insert.call_args_list[0].kwargs["body"]["snippet"]
    assert body["description"] == compose_description(entries)
    assert body["tags"] == compose_tags(entries)


def test_w9c_part_naming_no_evidenced_place_is_refused_before_insert(
    tmp_path,
    fake_creds_dict,  # noqa: F811
    caplog,
):
    rec = _record(tmp_path, location="Ostry")
    _refused(tmp_path, fake_creds_dict, caplog, rec, "verdict:refused")
    assert "layout_missing:hook" in caplog.text


def test_i4_unsupported_location_in_the_title(tmp_path, fake_creds_dict, caplog):  # noqa: F811
    rec = _record(tmp_path, title="Harrow to Ostry")
    _refused(tmp_path, fake_creds_dict, caplog, rec, "name_unbacked:title:ostry")


def test_i4_missing_description(tmp_path, fake_creds_dict, caplog):  # noqa: F811
    rec = _record(tmp_path, description="")
    _refused(tmp_path, fake_creds_dict, caplog, rec, "description_missing")


def test_i4_gps_day_part_without_a_route_claim(tmp_path, fake_creds_dict, caplog):  # noqa: F811
    rec = _record(tmp_path, route_claim=None)
    _refused(tmp_path, fake_creds_dict, caplog, rec, "route_record_missing")


def test_i4_one_byte_title_edit_after_review(tmp_path, fake_creds_dict, caplog):  # noqa: F811
    rec = _record(tmp_path)
    review = _review_for(_state([rec]), rec)
    edited = {**rec, "title": "Harrow to pellmere"}  # one byte: P -> p
    edited["final_text_review"] = review
    _refused(tmp_path, fake_creds_dict, caplog, edited, "stale:title")


def test_i4_stale_review_version(tmp_path, fake_creds_dict, caplog):  # noqa: F811
    rec = _record(tmp_path)
    review = dict(_review_for(_state([rec]), rec))
    review["review_version"] = "final-text-review-2"
    review["fingerprint"] = _fingerprint_of(review)
    carried = {**rec, "final_text_review": review}
    _refused(tmp_path, fake_creds_dict, caplog, carried, "stale:review_version")


@pytest.mark.parametrize(
    ("change", "reason"),
    [("text", "stale:text:ost"), ("source", "stale:source"), ("plan", "stale:plan")],
)
def test_i4_resumed_state_mismatch(tmp_path, fake_creds_dict, caplog, change, reason):  # noqa: F811
    rec = _record(tmp_path)
    state = _state([rec])
    carried = {**rec, "final_text_review": _review_for(state, rec)}  # saved, then resumed
    if change == "text":
        carried["location"] = "Harrow to Pellmere, Ashby"
    elif change == "plan":
        carried["duration_sec"] = 590.0
    else:
        state = _state([carried], episode_files=[Path("/footage/other.mp4")])
    state = state if change == "source" else None
    _refused(tmp_path, fake_creds_dict, caplog, carried, reason, state)


def _bound_check(tmp_path, monkeypatch, description=None, tags=None):
    """``check_final_text`` with the description/tag bind on, for the text a
    part would send (or the given byte-edited variant)."""
    monkeypatch.setattr(final_text_record, "BIND_DESCRIPTION_TAGS", True)
    rec = _record(tmp_path)
    meta = {"location": rec["location"]}
    ft_state, ft_lf = esc.superlong_final_text_inputs(_state([rec]), f"{DATE}:01", rec, meta)
    entries = meta["final_text_description"]
    ctx = final_text_context(ft_state, 1, ft_lf, meta)
    final_text_record.check_final_text(
        meta["final_text_review"],
        ctx,
        TITLE,
        compose_description(entries) if description is None else description,
        compose_tags(entries) if tags is None else tags,
    )
    return entries


def test_i4_bound_text_passes_unedited(tmp_path, monkeypatch):
    _bound_check(tmp_path, monkeypatch)


def test_i4_description_byte_edit_is_bound(tmp_path, monkeypatch):
    desc = compose_description(_bound_check(tmp_path, monkeypatch))
    with pytest.raises(FinalTextReviewError) as exc:
        _bound_check(tmp_path, monkeypatch, description=desc + "!")
    assert "stale:description" in exc.value.reasons


def test_i4_tag_byte_edit_is_bound(tmp_path, monkeypatch):
    tags = compose_tags(_bound_check(tmp_path, monkeypatch))
    with pytest.raises(FinalTextReviewError) as exc:
        _bound_check(tmp_path, monkeypatch, tags=[*tags[:-1], tags[-1] + "X"])
    assert "stale:tags" in exc.value.reasons


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(False, id="valid_inserts"),
        pytest.param(True, id="bad_part_refused"),
    ],
)
def test_t2_superlong_reviews_with_the_bind_flag_absent(
    tmp_path,
    fake_creds_dict,  # noqa: F811
    monkeypatch,
    caplog,
    bad,
):
    # The pre-insert review never reads the bind flag; the publication gate holds every insert
    # while the flag is off, so a valid part stays held until the flag is on.
    monkeypatch.delattr(youtube_uploader_module, "FINAL_TEXT_BIND_ON", raising=False)
    if bad:
        rec = _record(tmp_path, description="")
        _refused(tmp_path, fake_creds_dict, caplog, rec, "description_missing")
    else:
        # A reviewed, valid part is still HELD by the publication gate while the bind is off.
        caplog.set_level(logging.ERROR)
        svc = _run(tmp_path, fake_creds_dict, _state([_record(tmp_path)]))
        assert _inserts(svc) == 0
        assert "final_text_bind_off" in caplog.text
