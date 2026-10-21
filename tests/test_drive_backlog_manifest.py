"""``drive_backlog.yaml`` is the ordered list of drives still to audit.

It is tenant data, so it lives in the operator's inputs directory
(``GMS_INPUTS_DIR``) and is read through ``gomoveshift.paths.drive_backlog_path``.
The manifest must parse with seo_manager's fail-closed reader, and no raw chapter
may be queued twice (two rows cutting the same footage would publish the same
drive twice). The rows below are synthetic."""

from __future__ import annotations

import re

import pytest
from seo_manager.scheduling import backlog

from gomoveshift.paths import drive_backlog_path

CHAPTER = re.compile(r"^/media/footage/(trip-a|trip-b)/GX\d{6}\.MP4$")

_ROWS = """\
- project: example-2026-01
  region: usa
  sources:
    - /media/footage/trip-a/GX010001.MP4
    - /media/footage/trip-a/GX010002.MP4
  ingest_mode: full
  lf_mode: full_drive
- project: example-2026-02
  region: usa
  sources:
    - /media/footage/trip-b/GX010003.MP4
  ingest_mode: full
  lf_mode: full_drive
"""


@pytest.fixture
def manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_INPUTS_DIR", str(tmp_path))
    path = drive_backlog_path()
    path.parent.mkdir(parents=True)
    path.write_text(_ROWS, encoding="utf-8")
    return path


def test_the_consumer_reads_the_inputs_dir_copy(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_INPUTS_DIR", str(tmp_path))
    assert drive_backlog_path() == (
        tmp_path / "gomoveshift/console/brands/gomoveshift/drive_backlog.yaml"
    )


def test_manifest_parses_with_the_seo_manager_reader(manifest):
    rows = backlog.load(manifest)
    assert rows
    assert {r.region for r in rows} == {"usa"}


def test_every_source_is_one_raw_gopro_chapter(manifest):
    for row in backlog.load(manifest):
        bad = [s for s in row.sources if not CHAPTER.match(s)]
        assert not bad, f"{row.project}: {bad}"


def test_no_chapter_is_queued_in_two_rows(manifest):
    seen: dict[str, str] = {}
    for row in backlog.load(manifest):
        for s in row.sources:
            assert s not in seen, f"{s} in {seen[s]} and {row.project}"
            seen[s] = row.project
