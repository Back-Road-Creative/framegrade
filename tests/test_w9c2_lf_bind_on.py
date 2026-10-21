"""LF: with both byte-binds ON the LF sends exactly the text Waypost
composed and reviewed, not the GMS-built description and tags.

Stubs only the YouTube client; the review, the record and the pre-insert check
are the real code on the pinned seo_manager. Synthetic places only ("Harrow").
"""

from __future__ import annotations

import pytest
from seo_manager.analytics.final_text_surfaces import compose_description, compose_tags

from gomoveshift.pipeline import final_text_record as ftr
from gomoveshift.pipeline.drive_phases import youtube_uploader as yu
from gomoveshift.pipeline.final_text_record import FinalTextReviewError

from test_w9_final_text_publication_gate import (  # noqa: F401 — fixtures
    _inserts,
    _upload_lf,
    fake_creds_dict,
    video,
)
from test_w9c1_phase_d_wiring import _build as _build_c1


@pytest.fixture(autouse=True)
def _place_gate_off(monkeypatch):
    # These fixtures' reviewed descriptions name only Harrow while 300 s of the
    # footage is in Pellmere; the zero-tolerance place gate (test_place_gate.py)
    # correctly refuses that, and these tests are about the reviewed-text bind.
    monkeypatch.setattr("gomoveshift.pipeline.final_text_record._place_gate", lambda *a, **k: None)


def _build(tmp_path, label):
    state, lf, meta = _build_c1(tmp_path, label)
    meta.setdefault("tags", ["t"])
    return state, lf, meta


GMS_DESCRIPTION = "GMS-built description, not the reviewed text"
GMS_TAGS = ["gms-built-tag"]


def test_both_binds_ship_on():
    assert yu.FINAL_TEXT_BIND_ON is True
    assert ftr.BIND_DESCRIPTION_TAGS is True


def test_lf_sends_the_composed_description_and_tags(fake_creds_dict, video, tmp_path):  # noqa: F811
    state, lf, meta = _build(tmp_path, "Harrow")
    entries = meta["final_text_description"]
    svc, out = _upload_lf(
        fake_creds_dict, video, state, lf, meta, description=GMS_DESCRIPTION, tags=GMS_TAGS
    )
    assert not isinstance(out, Exception) and out["video_id"]
    assert _inserts(svc) == 1
    snippet = svc.videos.return_value.insert.call_args.kwargs["body"]["snippet"]
    assert snippet["description"] == compose_description(entries) != GMS_DESCRIPTION
    assert snippet["tags"] == compose_tags(entries) != GMS_TAGS


def test_lf_chapter_block_is_in_the_sent_bytes(fake_creds_dict, video, tmp_path):  # noqa: F811
    from unittest.mock import MagicMock, patch

    from gomoveshift.pipeline.drive_phases import phase_d_metadata as pdm
    from types import SimpleNamespace

    state, lf, meta = _build(tmp_path, "Harrow")
    state.per_lf_gpx_path = {1: "lf01.gpx"}
    chapters = [
        {"timestamp": "0:00", "title": "Harrow"},
        {"timestamp": "2:00", "title": "Pellmere"},
        {"timestamp": "4:00", "title": "Harrow"},
    ]
    redundancy = patch(
        "gomoveshift.pipeline.validation.content_gates.check_description_redundancy",
        return_value=SimpleNamespace(passed=True, message=""),
    )
    with (
        patch.object(pdm, "_lf_chapters", return_value=(chapters, "0:00 Harrow")),
        patch.object(pdm, "_below_chapter_floor", return_value=False),
        redundancy,
    ):
        pdm._generate_chapters_all_lfs(
            config=MagicMock(),
            drive_plan=SimpleNamespace(regular_lfs=[lf]),
            state=state,
            metadata_per_lf={1: meta},
            location="Harrow",
            project_base="p",
            output_base=tmp_path,
            stages_dir=tmp_path,
        )
    svc, out = _upload_lf(
        fake_creds_dict, video, state, lf, meta, description=GMS_DESCRIPTION, tags=GMS_TAGS
    )
    assert not isinstance(out, Exception), out
    sent = svc.videos.return_value.insert.call_args.kwargs["body"]["snippet"]["description"]
    assert "2:00" in sent and "Pellmere" in sent


@pytest.mark.parametrize("edit", ["description", "tags"])
def test_i4_edit_after_review_still_inserts_nothing(edit, fake_creds_dict, video, tmp_path):  # noqa: F811
    """The check runs on the composed bytes: an entry edited after the review
    (not the sent string) is what the refusal catches."""
    state, lf, meta = _build(tmp_path, "Harrow")
    meta["final_text_description"] = [
        {**e, "text": e["text"] + "x"} if _hit(e, edit) else e
        for e in meta["final_text_description"]
    ]
    with pytest.raises(FinalTextReviewError) as caught:
        _upload_lf(fake_creds_dict, video, state, lf, meta, description="d", tags=["t"])
    assert f"stale:{edit}" in caught.value.reasons


def _hit(entry, edit):
    from seo_manager.analytics.final_text_surfaces import is_description_surface, is_tag_surface

    return (is_description_surface if edit == "description" else is_tag_surface)(entry)


def test_lf_without_a_route_record_inserts_nothing(fake_creds_dict, video, tmp_path):  # noqa: F811
    """A GPS-less LF has no review to bind: the GMS-built text is never sent."""
    state, lf, meta = _build(tmp_path, "Harrow")
    meta.pop("final_text_description")
    meta.pop("final_text_review")
    state.per_lf_route_record = None
    with pytest.raises(FinalTextReviewError) as caught:
        _upload_lf(
            fake_creds_dict, video, state, lf, meta, description=GMS_DESCRIPTION, tags=GMS_TAGS
        )
    assert "route_record_missing" in caught.value.reasons
