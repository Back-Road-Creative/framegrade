"""A GPS-bearing long-form upload ships only under an agreeing
Waypost final-text review bound to its source, plan, route evidence, exact
title and visible text; GMS's own ``upload_video`` re-verifies it before
``videos().insert``.

Synthetic places only ("Harrow", "Pellmere", "Westmark" are invented); no
production trace, title or coordinate appears here.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from seo_manager.analytics.final_text_surfaces import compose_description, compose_tags

from gomoveshift.pipeline.drive_state_model import PipelineState
from gomoveshift.pipeline.final_text_record import (
    FinalTextReviewError,
    build_final_text_review,
    final_text_context,
    route_record,
)
from gomoveshift.pipeline.location_evidence import PlaceEvidence, _evidence_id
from gomoveshift.pipeline.route_claims import RouteWindow, measure_route
from gomoveshift.pipeline.title_decision import approve_retained_title, decide_title

from _state_helpers import make_state
from w9_drive_text import attach_drive_description
from test_retained_gps_claims import _BandGeocoder
from test_playlist_idempotency import (  # noqa: F401 — fixture
    _build_mock_service,
    _make_cfg,
    _make_uploader,
    fake_creds_dict,
)

DATE = "2026-01-02"
OST = "Harrow to Pellmere, Westmark"
TITLE = OST + " in 4K"
SUFFIX = " in 4K"


def _route(lat_of_t=lambda t: 9.5 if t < 300 else 10.2, end=600.0):
    fixes = [(float(t), lat_of_t(t), 20.0) for t in range(int(end))]
    return measure_route([RouteWindow("lf01-s0", 0.0, end, fixes)], _BandGeocoder())


def _location(names=(("Harrow", "in"), ("Pellmere", "in"))):
    places = [
        PlaceEvidence(
            name=n,
            county="Ashby County",
            state="Westmark",
            scope=scope,
            method="reverse_geocode",
            provenance="gps",
            segment_id="lf01-s0",
            evidence_id=_evidence_id("lf01-s0", n, "Ashby County", "Westmark", "reverse_geocode"),
        ).__dict__
        for n, scope in names
    ]
    return {
        "segment_id": "lf01-s0",
        "provenance": "gps",
        "method": "reverse_geocode",
        "places": places,
        "conflicts": [],
    }


def _plan_lf(end=600.0):
    return SimpleNamespace(
        index=1, segments=[SimpleNamespace(source_file_index=0, start_time=0.0, end_time=end)]
    )


def _state(**over):
    fields = {
        "episode_files": [Path("/footage/ep0.mp4")],
        "per_lf_ost_text": {1: OST},
        "per_lf_recording_date": {1: DATE},
        "ost_audit_hash": {DATE: "audit-h1"},
        "ost_audit_passed": {DATE: True},
        "per_lf_location_evidence": {1: _location()},
        "per_lf_route_record": {1: route_record(_route())},
        "per_lf_pre_render_title_decisions": {
            1: {**approve_retained_title(OST, _route(), _location()), "ost_text": OST}
        },
    }
    fields.update(over)
    return make_state(**fields)


def _metadata(state, title=TITLE, lf=None):
    loc = (state.per_lf_location_evidence or {}).get(1)
    meta = {
        "title": title,
        "title_decision": decide_title("longform", title, loc, suffix=SUFFIX),
    }
    if loc:
        attach_drive_description(state, meta)  # W9: drive reviews carry the gms-drive-1 layout
    meta["final_text_review"] = build_final_text_review(state, lf or _plan_lf(), meta)
    return meta


@pytest.fixture(autouse=True)
def _open_publication_gate(monkeypatch):
    """The publication gate holds every insert; these tests exercise the review, not the hold."""
    from gomoveshift.pipeline.drive_phases import youtube_uploader

    monkeypatch.setattr(youtube_uploader, "FINAL_TEXT_BIND_ON", True)


@pytest.fixture
def video(tmp_path):
    p = tmp_path / "lf_01.mp4"
    p.write_bytes(b"\x00\x00\x00\x20ftypiso5")
    return p


def _upload(
    fake_creds_dict,  # noqa: F811
    video,
    state,
    metadata,
    *,
    plan_lf=None,
    title=TITLE,
    description="d",
    tags=(),
):
    svc = _build_mock_service()
    up, yu = _make_uploader(svc, fake_creds_dict)
    with patch.object(yu, "MediaFileUpload", return_value=MagicMock()):
        try:
            return svc, up.upload_if_needed(
                state=state,
                lf_idx=1,
                video_path=video,
                title=title,
                description=description,
                tags=list(tags),
                metadata=metadata,
                config=_make_cfg(),
                plan_lf=plan_lf or _plan_lf(),
            )
        except Exception as exc:  # noqa: BLE001 — returned for the assertion
            return svc, exc


class TestBuild:
    def test_boundary_proof_reaches_waypost_review_unchanged(self):
        record = route_record(_route())
        proof = {
            "verified": True,
            "basis": "pinned_tiger_polygon_bounded_speed_and_error",
            "source": "TIGER2026/COUNTY/tl_2026_us_county.zip",
            "sha256": "a" * 64,
            "region": "Ashby County",
            "max_speed_m_s": 70.0,
            "gps_error_m": 100.0,
        }
        record["region_claims"][0]["boundary_proof"] = proof
        state = _state(per_lf_route_record={1: record})
        seen = {}

        def review(approved, decision, evidence, policy):
            seen.update(approved)
            return SimpleNamespace(verdict="agree", reasons=())

        with (
            patch("gomoveshift.pipeline.final_text_record.review_final_text", side_effect=review),
            patch(
                "gomoveshift.pipeline.final_text_record.review_to_dict", side_effect=lambda x: {}
            ),
        ):
            _metadata(state)
        assert seen["region_claims"][0]["boundary_proof"] == proof

    def test_supported_text_agrees(self):
        state = _state()
        review = _metadata(state)["final_text_review"]
        assert review["verdict"] == "agree", review

    def test_multi_part_suffix_agrees(self):
        # A split drive's parts are told apart by " — Part k of K"; "Part"
        # names nothing, so it must not refuse as name_unbacked:title:part.
        state = _state()
        review = _metadata(state, title=TITLE + " \u2014 Part 1 of 2")["final_text_review"]
        assert review["verdict"] == "agree", review

    def test_gps_less_lf_has_no_review(self):
        state = _state(per_lf_location_evidence=None, per_lf_route_record=None)
        assert _metadata(state)["final_text_review"] is None

    def test_unknown_place_in_the_text_is_not_agree(self):
        state = _state(per_lf_ost_text={1: "Longmere to Pellmere, Westmark"})
        title = "Longmere to Pellmere, Westmark in 4K"
        review = _metadata(state, title=title)["final_text_review"]
        assert review["verdict"] != "agree"

    def test_place_only_in_excluded_footage_is_not_agree(self):
        # The retained plan keeps 0-280 s (Harrow only); Pellmere is excluded.
        state = _state(per_lf_route_record={1: route_record(_route(end=280.0))})
        review = _metadata(state, lf=_plan_lf(end=280.0))["final_text_review"]
        assert review["verdict"] != "agree"

    def test_unapproved_ost_is_not_agree(self):
        state = _state(ost_audit_passed={DATE: False})
        assert _metadata(state)["final_text_review"]["verdict"] != "agree"


class TestUploadIfNeeded:
    def test_agreeing_review_uploads(self, fake_creds_dict, video):  # noqa: F811
        state = _state()
        meta = _metadata(state)
        surfaces = meta["final_text_review"]["surfaces"]
        svc, out = _upload(
            fake_creds_dict,
            video,
            state,
            meta,
            description=compose_description(surfaces),
            tags=compose_tags(surfaces),
        )
        assert not isinstance(out, Exception), out
        assert out["video_id"] == "VIDEO_ID_1"

    @pytest.mark.parametrize("marker", [None, {}, ""])
    def test_missing_or_empty_review_is_refused_with_no_insert(
        self,
        fake_creds_dict,  # noqa: F811
        video,
        marker,
    ):
        state = _state()
        meta = {**_metadata(state), "final_text_review": marker}
        svc, out = _upload(fake_creds_dict, video, state, meta)
        assert isinstance(out, FinalTextReviewError), out
        svc.videos.return_value.insert.assert_not_called()

    def test_refused_review_is_refused_with_no_insert(self, fake_creds_dict, video):  # noqa: F811
        state = _state(ost_audit_passed={DATE: False})
        svc, out = _upload(fake_creds_dict, video, state, _metadata(state))
        assert isinstance(out, FinalTextReviewError), out
        svc.videos.return_value.insert.assert_not_called()

    def test_changed_ost_after_review_is_stale(self, fake_creds_dict, video):  # noqa: F811
        state = _state()
        meta = _metadata(state)
        state.per_lf_ost_text = {1: "Harrow to Pellmere, Westmark County"}
        svc, out = _upload(fake_creds_dict, video, state, meta)
        assert isinstance(out, FinalTextReviewError)
        assert any(r.startswith("stale") for r in out.reasons), out.reasons
        svc.videos.return_value.insert.assert_not_called()

    def test_changed_plan_after_review_is_stale(self, fake_creds_dict, video):  # noqa: F811
        state = _state()
        meta = _metadata(state)
        svc, out = _upload(fake_creds_dict, video, state, meta, plan_lf=_plan_lf(end=590.0))
        assert isinstance(out, FinalTextReviewError)
        assert any("plan" in r for r in out.reasons), out.reasons
        svc.videos.return_value.insert.assert_not_called()

    def test_changed_source_after_review_is_stale(self, fake_creds_dict, video):  # noqa: F811
        state = _state()
        meta = _metadata(state)
        state.episode_files = [Path("/footage/other.mp4")]
        svc, out = _upload(fake_creds_dict, video, state, meta)
        assert isinstance(out, FinalTextReviewError)
        assert any("source" in r for r in out.reasons), out.reasons
        svc.videos.return_value.insert.assert_not_called()

    def test_gps_evidence_without_route_record_is_refused(
        self,
        fake_creds_dict,  # noqa: F811
        video,
    ):
        state = _state()
        meta = _metadata(state)
        state.per_lf_route_record = None
        svc, out = _upload(fake_creds_dict, video, state, meta)
        assert isinstance(out, FinalTextReviewError)
        assert out.reasons == ["route_record_missing"]
        svc.videos.return_value.insert.assert_not_called()

    def test_resumed_review_from_json_still_uploads(self, fake_creds_dict, video):  # noqa: F811
        state = _state()
        meta = json.loads(json.dumps(_metadata(state)))
        resumed = PipelineState.from_dict(json.loads(json.dumps(state.to_dict(), default=str)))
        surfaces = meta["final_text_review"]["surfaces"]
        svc, out = _upload(
            fake_creds_dict,
            video,
            resumed,
            meta,
            description=compose_description(surfaces),
            tags=compose_tags(surfaces),
        )
        assert not isinstance(out, Exception), out

    def test_gps_less_lf_uploads_without_review(self, fake_creds_dict, video):  # noqa: F811
        state = _state(per_lf_location_evidence=None, per_lf_route_record=None)
        svc, out = _upload(fake_creds_dict, video, state, _metadata(state))
        assert not isinstance(out, Exception), out


class TestLowerPaths:
    def test_upload_video_refuses_before_insert(self, fake_creds_dict, video):  # noqa: F811
        state = _state(ost_audit_passed={DATE: False})
        meta = _metadata(state)
        svc = _build_mock_service()
        up, yu = _make_uploader(svc, fake_creds_dict)
        with patch.object(yu, "MediaFileUpload", return_value=MagicMock()):
            with pytest.raises(FinalTextReviewError):
                up.upload_video(
                    video,
                    title=TITLE,
                    description="d",
                    tags=[],
                    final_text_review=meta["final_text_review"],
                    final_text_context=final_text_context(state, 1, _plan_lf(), meta),
                )
        svc.videos.return_value.insert.assert_not_called()

    def test_upload_video_with_playlist_refuses_before_insert(
        self,
        fake_creds_dict,  # noqa: F811
        video,
    ):
        state = _state()
        meta = {**_metadata(state), "final_text_review": None}
        svc = _build_mock_service()
        up, yu = _make_uploader(svc, fake_creds_dict)
        with patch.object(yu, "MediaFileUpload", return_value=MagicMock()):
            with pytest.raises(FinalTextReviewError):
                up.upload_video_with_playlist(
                    video,
                    title=TITLE,
                    description="d",
                    tags=[],
                    metadata=meta,
                    config=_make_cfg(),
                    final_text_context=final_text_context(state, 1, _plan_lf(), meta),
                )
        svc.videos.return_value.insert.assert_not_called()


class TestPhaseD:
    def _build(self, tmp_path, state):
        from gomoveshift.pipeline.drive_phases.phase_d import _build_metadata_per_lf

        gen = MagicMock()
        gen.generate.side_effect = lambda **kw: attach_drive_description(
            state,
            {
                "title": TITLE,
                "description": "d",
                "title_decision": state.per_lf_pre_render_title_decisions[1],
            },
        )
        lf = _plan_lf()
        lf.total_duration = 600.0
        with patch(
            "gomoveshift.pipeline.validation.content_gates.check_description_redundancy",
            return_value=SimpleNamespace(passed=True, message=""),
        ):
            return _build_metadata_per_lf(
                meta_gen=gen,
                drive_plan=SimpleNamespace(regular_lfs=[lf]),
                regular_lf_paths=["/v/lf.mp4"],
                state=state,
                location="Harrow, Westmark",  # Westmark holds the Pellmere half
                publish_day_full="Wednesday",
                project_base="p",
                output_base=tmp_path,
            )

    def test_phase_d_attaches_an_agreeing_review(self, tmp_path):
        out = self._build(tmp_path, _state())
        assert out[1]["final_text_review"]["verdict"] == "agree"

    def test_phase_d_review_for_unapproved_ost_is_refused(self, tmp_path):
        out = self._build(tmp_path, _state(ost_audit_passed={DATE: False}))
        assert out[1]["final_text_review"]["verdict"] != "agree"

    def test_phase_d_refuses_missing_pre_render_title_decision(self, tmp_path):
        from gomoveshift.pipeline.title_decision import PreRenderTitleError

        with pytest.raises(PreRenderTitleError, match="no pre-render Waypost title decision"):
            self._build(tmp_path, _state(per_lf_pre_render_title_decisions=None))

    def test_phase_d_refuses_changed_retained_route(self, tmp_path):
        from gomoveshift.pipeline.title_decision import PreRenderTitleError

        state = _state()
        state.per_lf_route_record[1]["record"]["retained_seconds"] += 1
        with pytest.raises(PreRenderTitleError, match="retained route changed"):
            self._build(tmp_path, state)

    def test_phase_d_refuses_lost_retained_route(self, tmp_path):
        from gomoveshift.pipeline.title_decision import PreRenderTitleError

        with pytest.raises(PreRenderTitleError, match="lost its retained route"):
            self._build(tmp_path, _state(per_lf_route_record=None))


class _CarolinaGeocoder(_BandGeocoder):
    """The same synthetic towns, placed in a real US state so the title can
    abbreviate it the way ``build_lf_title`` does."""

    def reverse_geocode_detailed(self, lat, lon):
        return {**super().reverse_geocode_detailed(lat, lon), "state": "South Carolina"}


def _carolina_state(ost, *, lat_of_t=lambda t: 9.5 if t < 300 else 10.2):
    fixes = [(float(t), lat_of_t(t), 20.0) for t in range(600)]
    ev = measure_route([RouteWindow("lf01-s0", 0.0, 600.0, fixes)], _CarolinaGeocoder())
    loc = _location()
    for p in loc["places"]:
        p["state"] = "South Carolina"
    return _state(
        per_lf_ost_text={1: ost},
        per_lf_location_evidence={1: loc},
        per_lf_route_record={1: route_record(ev)},
    )


class TestStateAbbreviation:
    """D6: "SC" is region evidence for the measured "South Carolina" claim;
    no road is cited (D9), and "Scenic" is never backed."""

    def _verdict(self, ost, **kw):
        state = _carolina_state(ost, **kw)
        return _metadata(state, title=ost + SUFFIX)["final_text_review"]

    def test_abbreviated_measured_state_agrees(self):
        review = self._verdict("Harrow to Pellmere, SC")
        assert review["verdict"] == "agree", review

    def test_wrong_state_abbreviation_is_not_agree(self):
        review = self._verdict("Harrow to Pellmere, NC")
        assert review["verdict"] != "agree"
        assert "name_unbacked:title:nc" in review["reasons"], review["reasons"]

    def test_unmeasured_via_road_is_not_agree(self):
        review = self._verdict("Harrow to Pellmere via Fern Road, SC")
        assert review["verdict"] != "agree"
        assert "name_unbacked:title:fern" in review["reasons"], review["reasons"]

    def test_scenic_loop_is_not_agree(self):
        review = self._verdict("Harrow Scenic Loop, SC", lat_of_t=lambda t: 9.5)
        assert review["verdict"] != "agree"
        assert "name_unbacked:title:scenic" in review["reasons"], review["reasons"]

    def test_circular_route_loop_title_agrees(self):
        review = self._verdict("Harrow Loop, SC", lat_of_t=lambda t: 9.5)
        assert review["verdict"] == "agree", review


def test_claim_id_missing_from_location_evidence_is_kept_not_crashed():
    """A chosen claim id with no location-evidence place stays as-is, so
    Waypost refuses it as evidence_missing (phase D used to die on
    'NoneType' object has no attribute 'get')."""
    from gomoveshift.pipeline.final_text_record import _evaluated_decision

    route = [
        {"evidence_id": "route-1", "name": "Beaufort", "county": "Beaufort County", "state": "SC"}
    ]
    known = {"evidence_id": "loc-1", "name": "Beaufort", "county": "Beaufort County", "state": "SC"}
    decision = {
        "chosen_index": 0,
        "candidates": [{"index": 0, "title": "Drive", "claim_evidence": ["loc-1", "loc-unknown"]}],
    }

    out = _evaluated_decision(decision, "Drive", route, [], [known])

    assert out["candidates"][0]["claim_evidence"] == ["route-1", "loc-unknown"]
