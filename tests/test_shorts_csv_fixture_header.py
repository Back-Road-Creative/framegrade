"""Fixture guard: the frozen redacted consolidated-shorts CSV fixture
used to document the CSV→platform-payload field mapping must never drift
from the live GOLDEN_COLUMNS header the writer actually emits.
"""

from __future__ import annotations

import csv as csvmod
from pathlib import Path

from gomoveshift.pipeline.csv_generator import GOLDEN_COLUMNS

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "consolidated_shorts_csv_row.csv"


def test_fixture_header_matches_golden_columns():
    assert FIXTURE_PATH.exists(), f"fixture missing: {FIXTURE_PATH}"
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        header = next(csvmod.reader(f))
    assert header == GOLDEN_COLUMNS


def test_fixture_has_exactly_one_data_row():
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        rows = list(csvmod.reader(f))
    assert len(rows) == 2, "expected header + exactly one redacted data row"
