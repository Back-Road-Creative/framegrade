"""Superlong OST audit gate — the third burned-text surface.

Long-form has a pre-burn audit gate; shorts has one; superlong parts had
none. ``day_pipeline._produce_superlong_parts`` builds each part's chapter-
derived ``lower_thirds`` one statement before calling ``apply_ost`` with the
day-level burned title (``location``) — with no operator-visible artefact
and no confirmation required. Mirrors ``ost_audit_sidecar.shorts_gate_or_raise``'s
shape.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from _state_helpers import make_state
from gomoveshift.pipeline.ost_audit_sidecar import OstAuditPendingError, stamp_confirmed

_CHAPTERS = [
    {"timestamp": "0:00", "title": "Start - Test Drive"},
    {"timestamp": "5:00", "title": "Entering Riverside, SC"},
    {"timestamp": "19:45", "title": "Finish - Test County"},
]

_LOCATION = "Test Drive, Test County"
_DATE = "2026-06-01"


def _make_config(part_max_sec=1200):
    return SimpleNamespace(
        drive=SimpleNamespace(
            brand_logo_path=None,
            superlong_enabled=True,
            superlong_part_max_duration_sec=part_max_sec,
            superlong_split_grace=0.08,
        )
    )


def _run(tmp_path, state, chapters=_CHAPTERS):
    """Call ``_produce_superlong_parts`` with a single fake part; returns
    ``(result_or_None, mock_ost)`` — ``result`` is ``None`` when the gate
    raised ``OstAuditPendingError`` (caught here so ``mock_ost`` stays
    inspectable after the raise)."""
    from gomoveshift.pipeline import day_pipeline

    master_path = tmp_path / "day_master.mp4"
    master_path.write_bytes(b"fake master")
    day_dir = tmp_path / f"day_{_DATE}"
    day_dir.mkdir(exist_ok=True)
    ep = SimpleNamespace(concat_path=Path("/fake/ep_000.mp4"))
    config = _make_config()

    def _fake_ost(input_path, output_path, location, layout, **kwargs):
        Path(output_path).write_bytes(b"fake ost")
        return output_path

    def _fake_assemble(master_path, segments, file_boundaries, output_path, work_dir, **kw):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"concat")
        return Path(output_path)

    with (
        patch("gomoveshift.pipeline.day_pipeline.apply_ost", side_effect=_fake_ost) as mock_ost,
        patch(
            "gomoveshift.pipeline.day_pipeline.extract_and_concat_segments",
            side_effect=_fake_assemble,
        ),
        patch(
            "gomoveshift.pipeline.day_pipeline._build_superlong_part_metadata",
            return_value={
                "title": "Driving Test Drive — Part 1 of 1",
                "description": "desc",
                "chapters": chapters,
                "video_duration": 1200.0,
                "has_gps": True,
                "location": _LOCATION,
            },
        ),
    ):
        try:
            result = day_pipeline._produce_superlong_parts(
                date=_DATE,
                stop_mask=np.zeros(1200, dtype=bool),
                episodes=[ep],
                file_boundaries=[(0.0, 1200.0)],
                master_path=master_path,
                day_dir=day_dir,
                config=config,
                location=_LOCATION,
                ost_crossfade_sec=0.5,
                state=state,
            )
        except OstAuditPendingError:
            result = None
    return result, mock_ost


class TestSuperlongOstAuditGate:
    def test_unconfirmed_run_raises_before_any_apply_ost(self, tmp_path):
        state = make_state()
        result, mock_ost = _run(tmp_path, state)
        assert result is None, "an unconfirmed superlong burn must be blocked, not produced"
        mock_ost.assert_not_called()

    def test_unconfirmed_run_writes_audit_with_per_part_text_and_title_location(self, tmp_path):
        state = make_state()
        _run(tmp_path, state)

        audit_path = tmp_path / f"superlong_ost_audit_{_DATE}.json"
        assert audit_path.is_file(), "gate must write the audit artifact before raising"
        payload = json.loads(audit_path.read_text())
        assert payload["parts"], "no parts recorded"
        entry = payload["parts"][0]
        # The burned title's location — the day-level string forwarded from
        # run_day_pipeline, previously absent from every audit artifact.
        assert entry["ost_text"] == _LOCATION
        # The per-part chapter-derived lower-thirds (varies per part).
        assert entry["lower_thirds"] == [{"label": "Riverside, SC", "time_sec": 300.0}]

    def test_confirmed_run_proceeds_to_apply_ost(self, tmp_path):
        state = make_state(ost_audit_confirmed=True)
        result, mock_ost = _run(tmp_path, state)
        assert mock_ost.called
        assert result["records"][0]["lower_thirds"] == [
            {"label": "Riverside, SC", "time_sec": 300.0}
        ]

    def test_stale_confirmed_marker_reopens_the_gate(self, tmp_path):
        confirmed_file = tmp_path / f"superlong_ost_audit_{_DATE}.confirmed"
        stamp_confirmed(confirmed_file, "0" * 64)  # bound to a hash that can never match
        state = make_state()
        result, mock_ost = _run(tmp_path, state)
        assert result is None, "a stale confirmation must not pass a never-reviewed burn"
        mock_ost.assert_not_called()
