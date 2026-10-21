"""U5 (title-accuracy program): GMS consumes Waypost's title decision.

GMS supplies verified facts (the route-only baseline title and the places a
title may name, each with the evidence id of the GPS record behind it) and
ships whatever ``select_title`` decides. The decision record rides with the
metadata, and the upload refuses a title that differs from it, so no format
or resume path can silently replace the selected title.

Synthetic towns and traces only.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from seo_manager.analytics import metric_source  # noqa: E402
from seo_manager.analytics.title_selection import (  # noqa: E402
    EvidenceSnapshot,
    TemplateObservation,
)

from gomoveshift.pipeline import title_decision as td  # noqa: E402
from gomoveshift.pipeline.location_evidence import build_location_evidence  # noqa: E402


@pytest.fixture(autouse=True)
def _publication_gate_open(monkeypatch):
    # The publication gate holds every insert until the final text is bound;
    # these upload tests predate the gate, so they open it in-test only.
    monkeypatch.setattr(
        "gomoveshift.pipeline.drive_phases.youtube_uploader.FINAL_TEXT_BIND_ON", True
    )


_TOWNS = {
    0: ("Alder", "Ash County", "Virginia"),
    1: ("Birch", "Ash County", "Virginia"),
    2: ("Cedar", "Elm County", "Virginia"),
    3: ("Dogwood", "Elm County", "Virginia"),
}


class _Geo:
    def __init__(self):
        self.calls = 0

    def reverse_geocode_detailed(self, lat, lon):
        self.calls += 1
        town, county, state = _TOWNS[int(round((37.0 - lat) * 10))]
        return {"town": town, "county": county, "state": state}


def _trace(per_town=4):
    return [
        {"lat": 37.0 - band / 10, "lon": -81.0, "time": float(band * 100 + k)}
        for band in range(4)
        for k in range(per_town)
    ]


def _record():
    return build_location_evidence(_trace(), geocoder=_Geo(), segment_id="lf01")


BASELINE = "Alder to Dogwood, VA in 4K"


def _obs(template_id, rate, n=5000, fmt="longform", traffic="search"):
    return TemplateObservation(
        template_id=template_id,
        format=fmt,
        traffic_source=traffic,
        metric_source=metric_source.YOUTUBE_THUMBNAIL_IMPRESSION_CTR,
        rate=rate,
        n=n,
        window_start="2026-10-01T00:00:00+00:00",
        window_end="2026-10-10T00:00:00+00:00",
    )


def _snapshot(*obs):
    return EvidenceSnapshot(as_of="2026-10-14T00:00:00+00:00", observations=tuple(obs))


class TestFacts:
    def test_endpoints_and_region_carry_their_evidence_ids(self):
        rec = _record()
        facts = td.title_facts("longform", BASELINE, rec, suffix=" in 4K")
        assert facts is not None
        alder, dogwood = rec.place("Alder"), rec.place("Dogwood")
        assert (facts.origin.name, facts.origin.evidence_id) == ("Alder", alder.evidence_id)
        assert facts.destination.evidence_id == dogwood.evidence_id
        assert facts.region.name == "Virginia"
        assert facts.region.evidence_id == dogwood.evidence_id
        assert facts.suffix == " in 4K"
        assert facts.baseline_title == BASELINE

    @pytest.mark.parametrize(
        "title",
        [
            "Alder to Elmhurst, VA in 4K",  # destination never entered
            "Alder to Dogwood, OH in 4K",  # region the trace never entered
            "4K POV",  # no route at all
        ],
    )
    def test_unsupported_claims_give_no_facts(self, title):
        assert td.title_facts("longform", title, _record(), suffix=" in 4K") is None

    def test_no_record_gives_no_facts(self):
        assert td.title_facts("longform", BASELINE, None, suffix=" in 4K") is None


class TestDecide:
    def test_no_facts_keeps_the_baseline_and_says_so(self):
        d = td.decide_title("longform", BASELINE, None, suffix=" in 4K")
        assert d["chosen_title"] == BASELINE
        assert d["reason"] == td.FACTS_UNAVAILABLE
        assert d["candidates"] == []

    def test_no_evidence_is_an_explicit_baseline(self):
        d = td.decide_title("longform", BASELINE, _record(), suffix=" in 4K", snapshot=_snapshot())
        assert d["chosen_title"] == BASELINE
        assert d["reason"] == "insufficient_evidence"
        assert d["policy_version"]
        assert d["candidates"][0]["template_id"] == "route"

    def test_evidence_can_change_the_title(self):
        snap = _snapshot(_obs("route", 0.04), _obs("region_route", 0.06))
        d = td.decide_title("longform", BASELINE, _record(), suffix=" in 4K", snapshot=snap)
        assert d["reason"] == "lift_over_baseline"
        assert d["chosen_title"] == "Virginia Drive: Alder to Dogwood in 4K"

    def test_card_click_rate_cannot_manufacture_a_winner(self):
        bad = TemplateObservation(
            template_id="region_route",
            format="longform",
            traffic_source="search",
            metric_source=metric_source.YOUTUBE_CARD_CLICK_RATE,
            rate=0.9,
            n=5000,
            window_start="2026-10-01T00:00:00+00:00",
            window_end="2026-10-10T00:00:00+00:00",
        )
        snap = _snapshot(_obs("route", 0.04), bad)
        d = td.decide_title("longform", BASELINE, _record(), suffix=" in 4K", snapshot=snap)
        assert d["chosen_title"] == BASELINE

    def test_same_inputs_same_record(self):
        snap = _snapshot(_obs("route", 0.04), _obs("region_route", 0.06))
        a = td.decide_title("longform", BASELINE, _record(), suffix=" in 4K", snapshot=snap)
        b = td.decide_title("longform", BASELINE, _record(), suffix=" in 4K", snapshot=snap)
        assert a == b

    def test_record_accepts_the_persisted_dict_form(self):
        d = td.decide_title(
            "longform", BASELINE, _record().to_dict(), suffix=" in 4K", snapshot=_snapshot()
        )
        assert d["reason"] == "insufficient_evidence"


class TestFinalTitleGuard:
    def test_matching_title_passes(self):
        d = td.decide_title("longform", BASELINE, _record(), suffix=" in 4K")
        td.check_final_title(BASELINE, {"title_decision": d})

    def test_replaced_title_is_refused(self):
        d = td.decide_title("longform", BASELINE, _record(), suffix=" in 4K")
        with pytest.raises(td.TitleDecisionMismatchError):
            td.check_final_title("Driving the Old Road | 4K POV", {"title_decision": d})

    def test_metadata_without_a_decision_is_not_checked(self):
        td.check_final_title("anything", {"title": "anything"})
        td.check_final_title("anything", None)


@pytest.fixture
def generator():
    from gomoveshift.pipeline.longform_metadata import LongFormMetadataGenerator

    with patch.object(LongFormMetadataGenerator, "_init_seo_manager"):
        with patch.object(LongFormMetadataGenerator, "_init_ledger"):
            gen = LongFormMetadataGenerator()
            gen._seo_degraded = True
            gen._manager = None
            return gen


class TestGeneratorConsumesTheDecision:
    def test_title_is_the_decision_and_the_record_rides_along(self, generator):
        out = generator.generate("Alder to Dogwood, VA", location_evidence=_record().to_dict())
        assert out["title"] == BASELINE
        assert out["title_decision"]["chosen_title"] == out["title"]
        assert out["title_decision"]["reason"] == "insufficient_evidence"

    def test_evidence_snapshot_reaches_the_published_title(self, generator):
        snap = _snapshot(_obs("route", 0.04), _obs("region_route", 0.06))
        out = generator.generate(
            "Alder to Dogwood, VA",
            location_evidence=_record().to_dict(),
            title_evidence=snap,
        )
        assert out["title"] == "Virginia Drive: Alder to Dogwood in 4K"

    def test_part_suffix_is_inside_the_decision(self, generator):
        out = generator.generate(
            "Alder to Dogwood, VA",
            location_evidence=_record().to_dict(),
            title_part=(1, 2),
        )
        assert out["title"].endswith("— Part 1 of 2")
        assert out["title_decision"]["chosen_title"] == out["title"]

    def test_no_evidence_record_is_an_explicit_baseline(self, generator):
        out = generator.generate("Alder to Dogwood, VA")
        assert out["title"] == BASELINE
        assert out["title_decision"]["reason"] == td.FACTS_UNAVAILABLE

    def test_build_title_takes_no_optimizer_suffix(self):
        import inspect

        from gomoveshift.pipeline.longform_metadata import LongFormMetadataGenerator

        assert (
            "seo_suffix" not in inspect.signature(LongFormMetadataGenerator._build_title).parameters
        )


class TestUploadPayload:
    def _uploader(self):
        from gomoveshift.pipeline.drive_phases.youtube_uploader import YouTubeUploader

        up = YouTubeUploader.__new__(YouTubeUploader)
        service = MagicMock()
        service.videos.return_value.insert.return_value = MagicMock()
        up._build_service = lambda: service
        up._execute_resumable = lambda request: "vid123"
        return up, service

    def _call(self, up, title, metadata, tmp_path):
        video = tmp_path / "lf_00.mp4"
        video.write_bytes(b"")
        with patch(
            "gomoveshift.pipeline.drive_phases.youtube_uploader.MediaFileUpload", MagicMock()
        ):
            return up.upload_if_needed(
                state=SimpleNamespace(youtube_uploads={}),
                lf_idx=0,
                video_path=video,
                title=title,
                description="d",
                tags=["t"],
                metadata=metadata,
            )

    def test_final_payload_title_is_the_decision(self, tmp_path):
        up, service = self._uploader()
        d = td.decide_title("longform", BASELINE, _record(), suffix=" in 4K")
        with patch(
            "gomoveshift.pipeline.drive_phases.youtube_uploader._enforce_youtube_upload_media_limits",
            lambda p: None,
        ):
            rec = self._call(up, BASELINE, {"title_decision": d}, tmp_path)
        body = service.videos.return_value.insert.call_args.kwargs["body"]
        assert body["snippet"]["title"] == d["chosen_title"]
        assert rec["title_decision"] == d

    def test_a_replaced_title_never_reaches_insert(self, tmp_path):
        up, service = self._uploader()
        d = td.decide_title("longform", BASELINE, _record(), suffix=" in 4K")
        with pytest.raises(td.TitleDecisionMismatchError):
            self._call(up, "Driving the Old Road | 4K POV", {"title_decision": d}, tmp_path)
        service.videos.return_value.insert.assert_not_called()


class TestPhaseBcStoresTheEvidence:
    def test_each_lf_gets_a_record_from_a_bounded_sample(self):
        from gomoveshift.pipeline.drive_phases import phase_bc_day_first as bc

        geo = _Geo()
        pts = _trace(per_town=100)
        records = bc._build_location_evidence_per_lf([SimpleNamespace(index=3)], {3: pts}, geo)
        rec = records[3]
        names = [p["name"] for p in rec["places"]]
        assert names == ["Alder", "Birch", "Cedar", "Dogwood"]
        assert rec["segment_id"] == "lf03"
        assert geo.calls <= bc.EVIDENCE_SAMPLE_MAX

    def test_gps_less_lf_has_no_record(self):
        from gomoveshift.pipeline.drive_phases import phase_bc_day_first as bc

        assert bc._build_location_evidence_per_lf([SimpleNamespace(index=1)], {}, _Geo()) == {}

    def test_state_round_trips_the_records(self):
        from gomoveshift.pipeline.drive_state_model import PipelineState

        s = PipelineState()
        s.per_lf_location_evidence = {2: _record().to_dict()}
        back = PipelineState.from_dict(s.to_dict())
        assert back.per_lf_location_evidence[2]["segment_id"] == "lf01"
