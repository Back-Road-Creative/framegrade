"""CSV formula-injection sanitization + utf-8 encoding of the LF writer.

Formula injection: user/geocoder/LLM-derived text cells (Text, YouTube Video Title,
YouTube Video Tags, First Comment Text) are written into the Metricool CSV
verbatim. A value beginning ``= + - @`` (or tab/CR) executes as a formula when
the CSV is opened in Sheets/Excel (OWASP CSV-injection). The fix neutralizes
such cells with a leading apostrophe.

Encoding: ``generate_lf_row`` opened its CSV without an explicit ``encoding=``,
so an accented place name (e.g. ``Dún Laoghaire``) could raise on a non-UTF-8
default locale. The fix pins ``encoding="utf-8"``.
"""

from __future__ import annotations

import csv as csvmod
from pathlib import Path
from types import SimpleNamespace

from gomoveshift.pipeline.csv_generator import (
    CSVGenerator,
    build_consolidated_shorts_csv,
)


def _make_post(short_index: int, **overrides):
    base = dict(
        short_index=short_index,
        caption=f"caption {short_index}",
        youtube_title=f"title {short_index}",
        youtube_category="19",
        youtube_tags="driving,ireland",
        tiktok_title=f"tt {short_index}",
        pinterest_board="drives",
        first_comment=f"#tag{short_index}",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _make_slot(short_index: int):
    return SimpleNamespace(
        short_index=short_index,
        date="2026-04-01",
        time="16:00:00",
    )


def test_shorts_title_formula_is_neutralized(tmp_path: Path):
    """A ``=HYPERLINK(...)`` title is written with a leading apostrophe."""
    out = tmp_path / "shorts_metricool.csv"
    malicious = '=HYPERLINK("http://evil","click")'
    build_consolidated_shorts_csv(
        output_path=out,
        shorts_by_lf={1: [_make_post(1, youtube_title=malicious)]},
        slots_by_short_index={1: _make_slot(1)},
        drive_links_by_short_index={1: "https://drive.google.com/1"},
        parent_lf_video_id_by_short_index={1: "abc"},
        parent_recording_date_by_short_index={1: "2026-04-01"},
    )
    with open(out, encoding="utf-8") as f:
        row = next(csvmod.DictReader(f))
    assert row["YouTube Video Title"] == "'" + malicious


def test_lf_title_formula_is_neutralized(tmp_path: Path):
    """A ``=HYPERLINK(...)`` location/title in the LF writer is neutralized."""
    out = tmp_path / "lf_metricool.csv"
    malicious = '=HYPERLINK("http://evil")'
    CSVGenerator().generate_lf_row(
        title=malicious,
        tags="ireland,driving",
        category="19",
        drive_link="https://drive.google.com/lf",
        schedule_date="2026-04-01",
        schedule_time="15:00:00",
        output_path=out,
    )
    with open(out, encoding="utf-8") as f:
        row = next(csvmod.DictReader(f))
    assert row["YouTube Video Title"] == "'" + malicious
    # The Text column falls back to the (malicious) title when no description.
    assert row["Text"] == "'" + malicious


def test_lf_accented_name_roundtrips_utf8(tmp_path: Path):
    """An accented Irish place name writes and reads back as UTF-8."""
    out = tmp_path / "lf_accented_metricool.csv"
    title = "Dún Laoghaire to Wicklow"
    CSVGenerator().generate_lf_row(
        title=title,
        tags="ireland,wicklow",
        category="19",
        drive_link="https://drive.google.com/lf",
        schedule_date="2026-04-01",
        schedule_time="15:00:00",
        output_path=out,
    )
    # Decoding the bytes as UTF-8 must not raise and must preserve the accents.
    raw = out.read_bytes().decode("utf-8")
    assert "Dún Laoghaire" in raw
    with open(out, encoding="utf-8") as f:
        row = next(csvmod.DictReader(f))
    assert row["YouTube Video Title"] == title


def test_safe_values_unchanged(tmp_path: Path):
    """Non-leading-operator text and known-safe columns are not mangled."""
    out = tmp_path / "lf_safe_metricool.csv"
    CSVGenerator().generate_lf_row(
        title="Scenic coastal drive",
        tags="ireland,driving",
        category="19",
        drive_link="https://drive.google.com/lf",
        schedule_date="2026-04-01",
        schedule_time="15:00:00",
        output_path=out,
    )
    with open(out, encoding="utf-8") as f:
        row = next(csvmod.DictReader(f))
    assert row["YouTube Video Title"] == "Scenic coastal drive"
    assert row["YouTube Video Tags"] == "ireland,driving"
    # Known-safe / structured columns untouched.
    assert row["YouTube Video Category"] == "19"
    assert row["Date"] == "2026-04-01"
