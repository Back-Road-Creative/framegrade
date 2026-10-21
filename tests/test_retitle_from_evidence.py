"""U2 (title-accuracy program): retitles derive places from the GPS evidence
record, never from the existing title, and the applier re-checks that record
immediately before it pushes a title live. Synthetic places only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gomoveshift.pipeline.location_evidence import build_location_evidence
from tools import retitle_apply as ra
from tools import retitle_worklist as rw

_TOWNS = {
    0: ("Alder", "Ash County", "Virginia"),
    1: ("Birch", "Ash County", "Virginia"),
    2: ("Cedar", "Elm County", "Virginia"),
}


class _Geo:
    def reverse_geocode_detailed(self, lat, lon):
        town, county, state = _TOWNS[int(round((37.0 - lat) * 10))]
        return {"town": town, "county": county, "state": state}


def _evidence(bands=(0, 1, 2)) -> dict:
    pts = [
        {"lat": 37.0 - b / 10, "lon": -81.0, "time": float(i * 40 + k * 10)}
        for i, b in enumerate(bands)
        for k in range(4)
    ]
    return build_location_evidence(pts, geocoder=_Geo(), segment_id="lf01").to_dict()


def _rows_by_id(videos):
    return {r["video_id"]: r for r in rw.build_rows(videos, [])}


class TestWorklistDerivesFromEvidence:
    def test_title_comes_from_the_record_not_the_old_title(self):
        # The old title names a town the trace never entered.
        video = {"id": "v1", "title": "To Elkton in 4K", "location_evidence": _evidence()}
        row = _rows_by_id([video])["v1"]
        assert row["suggested_title"] == "Alder to Cedar, VA"
        assert "Elkton" not in row["suggested_title"]

    def test_no_record_means_needs_human_even_with_place_text_in_the_title(self):
        video = {"id": "v1", "title": "To Clonakilty, County Cork in 4K"}
        assert _rows_by_id([video])["v1"]["suggested_title"] == "needs-human"

    def test_structured_fields_the_record_contradicts_are_not_used(self):
        video = {
            "id": "v1",
            "title": "To Elkton in 4K",
            "origin": "Elkton",
            "destination": "Cedar",
            "region": "VA",
            "location_evidence": _evidence(),
        }
        assert _rows_by_id([video])["v1"]["suggested_title"] == "Alder to Cedar, VA"

    def test_single_place_record_titles_the_place(self):
        video = {"id": "v1", "title": "To Birch in 4K", "location_evidence": _evidence((1,))}
        assert _rows_by_id([video])["v1"]["suggested_title"] == "Birch, VA"

    def test_row_carries_the_record_for_the_applier(self):
        ev = _evidence()
        row = _rows_by_id([{"id": "v1", "title": "To Elkton in 4K", "location_evidence": ev}])["v1"]
        assert "location_evidence" in rw.FIELDNAMES
        assert json.loads(row["location_evidence"]) == ev

    def test_records_load_from_a_jsonl_sidecar(self, tmp_path):
        path = tmp_path / "evidence.jsonl"
        path.write_text(json.dumps({"video_id": "v1", "record": _evidence()}) + "\n")
        videos = rw.attach_evidence([{"id": "v1", "title": "To Elkton in 4K"}], path)
        assert _rows_by_id(videos)["v1"]["suggested_title"] == "Alder to Cedar, VA"

    def test_composed_rows_round_trip_as_approved(self):
        import io

        rows = rw.build_rows(
            [{"id": "v1", "title": "To Elkton in 4K", "location_evidence": _evidence()}], []
        )
        results = ra.dry_run(rows, out=io.StringIO())
        assert [r["verdict"] for r in results] == [ra.APPROVED]


class _Videos:
    def __init__(self, live):
        self.live, self.update_calls = live, []

    def list(self, part, id):
        snippet = self.live.get(id)
        items = [{"snippet": dict(snippet)}] if snippet else []
        return type("R", (), {"execute": lambda s: {"items": items}})()

    def update(self, part, body):
        calls, live = self.update_calls, self.live

        def execute(s):
            calls.append(body)
            live[body["id"]] = body["snippet"]  # a live update lands

        return type("R", (), {"execute": execute})()


class _Service:
    def __init__(self, live):
        self._v = _Videos(live)

    def videos(self):
        return self._v


class TestApplyRechecksGps:
    def test_approved_row_whose_record_no_longer_supports_it_is_not_pushed(self):
        row = {
            "video_id": "v1",
            "current_title": "To Elkton in 4K",
            "suggested_title": "Alder to Elkton, VA",
            "location_evidence": json.dumps(_evidence()),
        }
        service = _Service({"v1": {"title": "To Elkton in 4K"}})
        records = ra.apply_updates(service, [{"row": row, "verdict": ra.APPROVED}])
        assert service.videos().update_calls == []
        assert [r["result"] for r in records] == [ra.REFUSED]

    def test_supported_row_is_pushed(self):
        row = {
            "video_id": "v1",
            "current_title": "To Elkton in 4K",
            "suggested_title": "Alder to Cedar, VA",
            "location_evidence": json.dumps(_evidence()),
        }
        service = _Service({"v1": {"title": "To Elkton in 4K"}})
        records = ra.apply_updates(service, [{"row": row, "verdict": ra.APPROVED}])
        assert [r["result"] for r in records] == [ra.APPLIED]
