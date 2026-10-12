"""Metricool shorts-CSV schedule gate.

A short with no schedule slot yields blank ``Date``/``Time`` cells. Metricool
imports a dateless/timeless row as an UNSCHEDULED (immediate) post — it fires
the instant the CSV lands instead of on the planned drop, and ``validate_csv``
only flags a *malformed* date/time, never a fully blank one, so the prior
warning-only path shipped the blank row.

These tests pin the hardening: ``build_consolidated_shorts_csv`` raises
``METRICOOL-CSV-NO-SCHEDULE`` when any shipped row lacks a Date/Time, mirroring
the ``Picture Url 1`` media gate — the slotless deliverable is never written.
"""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from gomoveshift.pipeline.csv_generator import build_consolidated_shorts_csv


def _make_post(short_index: int):
    return SimpleNamespace(
        short_index=short_index,
        caption=f"Short {short_index} caption",
        first_comment="",
        youtube_title=f"Short {short_index}",
        youtube_tags="",
        tiktok_title=f"Short {short_index}",
        pinterest_board="",
    )


def _make_slot(short_index: int, *, date="2026-04-29", time="16:00:00"):
    return SimpleNamespace(
        short_index=short_index,
        date=date,
        time=time,
        datetime_est="2026-04-29T16:00:00-04:00",
        day_of_week="Wednesday",
    )


def _build(tmp_path: Path, *, slots):
    # Full media for every short so the media gate can never fire first —
    # only the schedule gate is under test here.
    return build_consolidated_shorts_csv(
        output_path=tmp_path / "shorts.csv",
        shorts_by_lf={1: [_make_post(1), _make_post(2)]},
        slots_by_short_index=slots,
        drive_links_by_short_index={
            1: "https://drive.example/1",
            2: "https://drive.example/2",
        },
        parent_lf_video_id_by_short_index={1: "vid1", 2: "vid1"},
        parent_recording_date_by_short_index={1: "2025-09-14", 2: "2025-09-14"},
    )


class TestMetricoolScheduleGate:
    def test_missing_slot_raises_and_names_row(self, tmp_path: Path):
        # short_index 2 has no slot → its row (index 1) ships blank Date/Time.
        with pytest.raises(ValueError, match="METRICOOL-CSV-NO-SCHEDULE") as exc:
            _build(tmp_path, slots={1: _make_slot(1)})
        assert "[1]" in str(exc.value)
        assert not (tmp_path / "shorts.csv").exists()  # slotless CSV never written

    def test_blank_date_slot_raises(self, tmp_path: Path):
        # A slot object present but carrying an empty date is just as dead.
        with pytest.raises(ValueError, match="METRICOOL-CSV-NO-SCHEDULE"):
            _build(tmp_path, slots={1: _make_slot(1), 2: _make_slot(2, date="")})
        assert not (tmp_path / "shorts.csv").exists()

    def test_blank_time_slot_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="METRICOOL-CSV-NO-SCHEDULE"):
            _build(tmp_path, slots={1: _make_slot(1), 2: _make_slot(2, time="")})
        assert not (tmp_path / "shorts.csv").exists()

    def test_all_slots_present_passes(self, tmp_path: Path):
        out = _build(tmp_path, slots={1: _make_slot(1), 2: _make_slot(2)})
        with open(out) as f:
            rows = list(csv.DictReader(f))
        assert [r["Date"] for r in rows] == ["2026-04-29", "2026-04-29"]
        assert all(r["Time"] == "16:00:00" for r in rows)
