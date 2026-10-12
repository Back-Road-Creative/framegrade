"""CSV writer defaults to plain UTF-8 (no BOM) for Metricool.

Metricool is the actual consumer of the consolidated shorts CSV and the
legacy ``utf-8-sig`` BOM may be breaking its
column-mapping parser (the first header column ends up as ``\\ufeffText``
instead of ``Text``). The default writer now emits plain UTF-8; the legacy
BOM behavior is opt-in via ``excel_compat=True`` for Excel users.
"""

from __future__ import annotations

import csv as csvmod
from pathlib import Path
from types import SimpleNamespace

from gomoveshift.pipeline.csv_generator import build_consolidated_shorts_csv

UTF8_BOM = b"\xef\xbb\xbf"


def _make_post(short_index: int):
    return SimpleNamespace(
        short_index=short_index,
        caption=f"caption {short_index}",
        youtube_title=f"title {short_index}",
        youtube_category="19",
        youtube_tags="driving,ireland",
        tiktok_title=f"tt {short_index}",
        pinterest_board="drives",
        first_comment=f"#tag{short_index}",
    )


def _make_slot(short_index: int):
    return SimpleNamespace(
        short_index=short_index,
        date="2026-04-29",
        time="16:00:00",
        datetime_est="2026-04-29T16:00:00-04:00",
        day_of_week="Wednesday",
    )


def _write(out: Path, **kwargs) -> Path:
    return build_consolidated_shorts_csv(
        output_path=out,
        shorts_by_lf={1: [_make_post(1)]},
        slots_by_short_index={1: _make_slot(1)},
        drive_links_by_short_index={1: "https://drive.google.com/1"},
        parent_lf_video_id_by_short_index={1: "abc"},
        parent_recording_date_by_short_index={1: "2026-04-29"},
        **kwargs,
    )


def test_default_writer_omits_bom(tmp_path: Path):
    """Default (Metricool target) — first bytes are the ASCII header, no BOM."""
    out = tmp_path / "shorts_metricool.csv"
    _write(out)

    raw = out.read_bytes()
    assert not raw.startswith(UTF8_BOM), (
        f"default CSV must not have UTF-8 BOM (Metricool parser may misread it); "
        f"got first 5 bytes: {raw[:5]!r}"
    )
    assert raw.startswith(b"Text,"), f"expected plain ASCII header; got: {raw[:8]!r}"


def test_excel_compat_writer_includes_bom(tmp_path: Path):
    """Opt-in excel_compat=True restores the legacy utf-8-sig BOM."""
    out = tmp_path / "shorts_metricool.csv"
    _write(out, excel_compat=True)

    raw = out.read_bytes()
    assert raw.startswith(UTF8_BOM), (
        f"excel_compat=True CSV must start with UTF-8 BOM; got first 5 bytes: {raw[:5]!r}"
    )
    assert raw[len(UTF8_BOM) :].startswith(b"Text,"), "header must follow the BOM"


def test_round_trip_default_and_excel_match(tmp_path: Path):
    """Both encodings round-trip identically through the csv module."""
    out_default = tmp_path / "default.csv"
    out_excel = tmp_path / "excel.csv"
    _write(out_default)
    _write(out_excel, excel_compat=True)

    # Read both with utf-8-sig — it accepts both with and without BOM.
    with open(out_default, encoding="utf-8-sig", newline="") as f:
        default_rows = list(csvmod.DictReader(f))
    with open(out_excel, encoding="utf-8-sig", newline="") as f:
        excel_rows = list(csvmod.DictReader(f))

    assert default_rows == excel_rows
    assert len(default_rows) == 1
    assert default_rows[0]["Text"] == "caption 1. Filmed April 2026."
    assert default_rows[0]["Picture Url 1"] == "https://drive.google.com/1"


def test_default_first_header_key_is_clean_text(tmp_path: Path):
    """Read with plain utf-8 — first column key must be ``Text`` not ``\\ufeffText``."""
    out = tmp_path / "shorts_metricool.csv"
    _write(out)

    with open(out, encoding="utf-8", newline="") as f:
        reader = csvmod.DictReader(f)
        first_key = list(reader.fieldnames or [])[0]

    assert first_key == "Text", (
        f"Metricool maps columns by header text; first key must be 'Text' not "
        f"{first_key!r} (BOM-prefixed key would cascade-misalign downstream columns)"
    )
