"""Drive LF description chapters are timed against the montage-prefixed file.

An LF file is ``montage + body`` (then the OST burn), but the GPS chapters are
generated against the body alone. Phase D therefore shifts every chapter after
0:00 by the lead the file actually carries (``lf_NN.lead.json``, or the trail
opening's length once the montage was replaced), so they agree with the photo
markers, whose times already include the lead.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from gomoveshift.pipeline.drive_phases.phase_d import _generate_chapters_all_lfs
from gomoveshift.pipeline.media_api import Deliverables, deliverables_to_state

from _state_helpers import make_state

BODY = [
    {"timestamp": "0:00", "title": "Start - A"},
    {"timestamp": "5:00", "title": "B"},
    {"timestamp": "10:00", "title": "C"},
    {"timestamp": "29:45", "title": "End"},
]


def _run(tmp_path, *, lead_json, openings=(), inserts=None):
    lf_dir = tmp_path / "day_2026-05-01" / "regular_lfs"
    lf_dir.mkdir(parents=True)
    lf_file = lf_dir / "lf_01.mp4"
    lf_file.write_bytes(b"x")
    if lead_json is not None:
        (lf_dir / "lf_01.lead.json").write_text(json.dumps({"lead_s": lead_json}))
    gpx = tmp_path / "lf1.gpx"
    gpx.write_text("<gpx></gpx>")
    d = Deliverables(
        date="2026-05-01",
        master_path=None,
        regular_lf_paths=[lf_file],
        regular_lf_indices=[1],
        inserts=inserts or {},
        trail_openings=list(openings),
    )
    state = make_state(
        per_lf_gpx_path={1: str(gpx)},
        per_day_completed={"2026-05-01": {"deliverables": deliverables_to_state(d)}},
    )
    lf = SimpleNamespace(index=1, total_duration=1800.0, segments=[])
    meta = {1: {"description": "Hook"}}
    text = "\n".join(f"{c['timestamp']} {c['title']}" for c in BODY)
    with (
        patch(
            "gomoveshift.pipeline.drive_phases.phase_d._generate_chapters_for_lf",
            return_value=([dict(c) for c in BODY], text),
        ),
        patch(
            "gomoveshift.pipeline.drive_phases.phase_d._LFMeta._embed_chapters",
            return_value="Hook",
        ),
        patch(
            "gomoveshift.pipeline.validation.content_gates.check_description_redundancy",
            return_value=SimpleNamespace(passed=True, message=""),
        ),
    ):
        _generate_chapters_all_lfs(
            config=MagicMock(),
            drive_plan=SimpleNamespace(regular_lfs=[lf]),
            state=state,
            metadata_per_lf=meta,
            location="x",
            project_base="p",
            output_base=tmp_path,
            stages_dir=tmp_path,
        )
    return meta[1]["chapters"]


def _stamps(chapters):
    return [c["timestamp"] for c in chapters]


def test_gps_chapters_move_by_the_recorded_lead(tmp_path):
    chapters = _run(tmp_path, lead_json=15.0)
    assert _stamps(chapters) == ["0:00", "5:15", "10:15", "30:00"]
    assert chapters[0]["title"] == "Start - A"


def test_no_lead_record_leaves_chapters_unchanged(tmp_path):
    assert _stamps(_run(tmp_path, lead_json=None)) == _stamps(BODY)


def test_zero_lead_leaves_chapters_unchanged(tmp_path):
    assert _stamps(_run(tmp_path, lead_json=0.0)) == _stamps(BODY)


def test_trail_opening_length_replaces_the_planned_lead(tmp_path):
    openings = [{"lf_index": 1, "opening_s": 8.0, "lead_s": 15.0, "insert_delta_s": -7.0}]
    chapters = _run(tmp_path, lead_json=15.0, openings=openings)
    assert _stamps(chapters) == ["0:00", "5:08", "10:08", "29:53"]


def test_insert_markers_keep_their_output_time(tmp_path):
    inserts = {1: [{"kind": "still", "at_s": 400.0}]}
    chapters = _run(tmp_path, lead_json=15.0, inserts=inserts)
    assert "6:40" in _stamps(chapters)
    assert _stamps(chapters)[:2] == ["0:00", "5:15"]


def test_shift_chapters_is_pure_and_zero_is_a_noop():
    from gomoveshift.pipeline.chapters.marker_merger import shift_chapters

    original = [dict(c) for c in BODY]
    assert shift_chapters(original, 0) == original
    assert shift_chapters(original, 15)[0] == original[0]
    assert original == BODY
