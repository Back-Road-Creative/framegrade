"""Tests for ``tools/retitle_apply.py``."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gomoveshift.pipeline.validation.place_corrections import PlaceCorrections
from tools import retitle_apply as ra


def _write_worklist(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "worklist.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _row(**kw):
    base = {
        "video_id": "vid1",
        "current_title": "To Ballymore in 4K",
        "reason": "preposition-led-label",
        "suggested_title": "Kilbarry to Ballymore, County Cork",
        "platform": "youtube",
    }
    base.update(kw)
    return base


def _req(payload):
    return SimpleNamespace(execute=lambda: payload)


class _FakeVideos:
    def __init__(self, snippets: dict[str, dict]):
        self.snippets = snippets
        self.update_calls: list[dict] = []

    def list(self, part, id):  # noqa: A002 - matches googleapiclient signature
        items = [{"id": id, "snippet": self.snippets[id]}] if id in self.snippets else []
        return _req({"items": items})

    def update(self, part, body):
        self.update_calls.append(body)
        return _req({})


class FakeService:
    def __init__(self, snippets):
        self._videos = _FakeVideos(snippets)

    def videos(self):
        return self._videos


def test_read_worklist_parses_rows(tmp_path):
    path = _write_worklist(tmp_path, [_row()])
    rows = ra.read_worklist(path)
    assert rows == [_row()]


@pytest.mark.parametrize(
    "title,corrections,expect_substr",
    [
        ("needs-human", None, "no usable suggested title"),
        ("X" * 101, None, "exceeds"),
        ("To Ballymore in 4K", None, "bare preposition"),
        ("Kingstown, County Cork", PlaceCorrections(denylist=("Kingstown",)), "denylisted"),
        ("Kilbarry to Ballymore, County Cork", None, None),  # clean -> no issues
    ],
)
def test_evaluate_row_gates(title, corrections, expect_substr):
    issues = ra.evaluate_row(_row(suggested_title=title), corrections)
    assert any(expect_substr in i for i in issues) if expect_substr else issues == []


def test_dry_run_marks_refused_rows(capsys):
    rows = [_row(), _row(video_id="vid2", suggested_title="needs-human")]
    results = ra.dry_run(rows, PlaceCorrections())
    assert [r["verdict"] for r in results] == [ra.APPROVED, ra.REFUSED]
    out = capsys.readouterr().out
    assert "vid1" in out and "APPROVED" in out and "REFUSED" in out


def test_main_dry_run_never_touches_service(tmp_path):
    path = _write_worklist(tmp_path, [_row()])
    service = FakeService({"vid1": {"title": "To Ballymore in 4K", "categoryId": "19"}})
    rc = ra.main(["--worklist", str(path)], service=service)
    assert rc == 0
    assert service.videos().update_calls == []


def _audit_records(path):
    audit_path = path.with_suffix(".audit.jsonl")
    return [json.loads(line) for line in audit_path.read_text().splitlines()]


def test_apply_preserves_snippet_fields_and_writes_audit(tmp_path):
    path = _write_worklist(tmp_path, [_row()])
    snippet = {
        "title": "To Ballymore in 4K",
        "description": "desc",
        "tags": ["ireland"],
        "categoryId": "19",
    }
    service = FakeService({"vid1": snippet})
    rc = ra.main(["--worklist", str(path), "--apply", "--yes-live"], service=service)
    assert rc == 0
    body = service.videos().update_calls[0]["snippet"]
    assert body == {**snippet, "title": "Kilbarry to Ballymore, County Cork"}
    records = _audit_records(path)
    assert records[0]["result"] == ra.APPLIED
    assert records[0]["video_id"] == "vid1"


def test_apply_skips_stale_row(tmp_path):
    path = _write_worklist(tmp_path, [_row()])
    service = FakeService({"vid1": {"title": "Some Other Live Title", "categoryId": "19"}})
    rc = ra.main(["--worklist", str(path), "--apply", "--yes-live"], service=service)
    assert rc == 0
    assert service.videos().update_calls == []
    assert _audit_records(path)[0]["result"] == ra.SKIPPED_STALE


def test_apply_requires_both_flags(tmp_path):
    path = _write_worklist(tmp_path, [_row()])
    rc = ra.main(["--worklist", str(path), "--apply"], service=FakeService({}))
    assert rc == 2


def test_composed_worklist_rows_round_trip_as_approved(tmp_path):
    """retitle_worklist's --compose output must clear retitle_apply's own
    gates -- a composed title stays needs-human (never REFUSED) at the source."""
    from tools import retitle_worklist as rw

    videos = [
        {"id": "dup1", "title": "Pennsylvania Scenic Drive in 4K | Millbrook, PA"},
        {"id": "dup2", "title": "Pennsylvania Scenic Drive in 4K | Millbrook, PA"},
        {"id": "irish1", "title": "To Ballymore, County Cork in 4K"},
    ]
    rows = rw.build_rows(videos, [])
    composed = [r for r in rows if r["suggested_title"] != "needs-human"]
    assert composed
    import io

    results = ra.dry_run(composed, out=io.StringIO())
    assert all(r["verdict"] == ra.APPROVED for r in results)


# --- descriptions: the wrong place lives in the description too -------------

OLD_DESC = "Rolling through Alder County 🚗\n\n#POVDriving #RoadTrip #AlderCounty #NorthCarolina"
NEW_DESC = (
    "Rolling through Birch County, Virginia 🚗\n\n#POVDriving #RoadTrip #BirchCounty #Virginia"
)


def _desc_row(**kw):
    return _row(**{"current_description": OLD_DESC, "suggested_description": NEW_DESC, **kw})


@pytest.mark.parametrize(
    "overrides,expect_substr",
    [
        ({"current_description": ""}, "needs current_description"),
        ({"suggested_description": "é" * 2501}, "exceeds 5000 bytes"),
        ({"suggested_description": "Rolling through <Birch County>"}, "angle bracket"),
        ({"suggested_description": "Rolling through Kingstown 🚗"}, "denylisted"),
        ({"suggested_description": "Drive through to to Elmsford"}, "doubled preposition"),
        ({}, None),  # clean -> no issues
    ],
)
def test_evaluate_row_gates_suggested_description(overrides, expect_substr):
    issues = ra.evaluate_row(_desc_row(**overrides), PlaceCorrections(denylist=("Kingstown",)))
    assert any(expect_substr in i for i in issues) if expect_substr else issues == []


def test_evaluate_row_without_description_columns_is_unchanged():
    assert ra.evaluate_row(_row(), PlaceCorrections()) == []


def test_apply_writes_title_and_description_in_one_update(tmp_path):
    path = _write_worklist(tmp_path, [_desc_row()])
    snippet = {"title": "To Ballymore in 4K", "description": OLD_DESC, "categoryId": "19"}
    service = FakeService({"vid1": snippet})
    rc = ra.main(["--worklist", str(path), "--apply", "--yes-live"], service=service)
    assert rc == 0
    [call] = service.videos().update_calls
    assert call["snippet"] == {
        **snippet,
        "title": "Kilbarry to Ballymore, County Cork",
        "description": NEW_DESC,
    }
    [record] = _audit_records(path)
    assert record["result"] == ra.APPLIED
    assert (record["old_description"], record["new_description"]) == (OLD_DESC, NEW_DESC)


def test_apply_skips_row_whose_live_description_drifted(tmp_path):
    path = _write_worklist(tmp_path, [_desc_row()])
    snippet = {
        "title": "To Ballymore in 4K",
        "description": "edited in Studio",
        "categoryId": "19",
    }
    service = FakeService({"vid1": snippet})
    rc = ra.main(["--worklist", str(path), "--apply", "--yes-live"], service=service)
    assert rc == 0
    assert service.videos().update_calls == []
    assert _audit_records(path)[0]["result"] == ra.SKIPPED_STALE
