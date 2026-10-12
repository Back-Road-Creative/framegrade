"""Tests for OST audit sidecar formatter + writer.

The operator must see a per-day audit artefact listing every LF's OST
text, YouTube title, thumbnail-text source, and recording date BEFORE
any OST burn-in re-encode commits pixels.

``format_ost_audit`` is a pure formatter; ``write_ost_audit`` writes the
JSON sidecar and returns the path. Mirrors ``endscreen_sidecar``'s
shape (pure formatter + writer separated from gate logic).
"""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

from gomoveshift.pipeline import ost_audit_sidecar as sidecar
from gomoveshift.pipeline.encoding_constants import SHORTS_WIDTH
from gomoveshift.pipeline.ost_audit_sidecar import (
    compute_ost_audit_hash,
    format_ost_audit,
    write_ost_audit,
)

from _state_helpers import make_state


def test_format_ost_audit_full_data_single_lf():
    state = make_state(
        per_lf_ost_text={1: "Glenfable Peninsula toward Gortmore, County Kerry"},
        per_lf_titles={1: "Highlights from 30min — Glenfable Peninsula toward Gortmore, County Kerry"},
        per_lf_recording_date={1: "2025-09-14"},
        per_lf_locations={1: "County Kerry"},
    )
    payload = format_ost_audit("2025-09-14", state, [1])
    assert payload["date"] == "2025-09-14"
    assert "generated_at" in payload
    assert len(payload["lfs"]) == 1
    lf = payload["lfs"][0]
    assert lf["lf_index"] == 1
    assert lf["ost_text"] == "Glenfable Peninsula toward Gortmore, County Kerry"
    assert (
        lf["youtube_title"]
        == "Highlights from 30min — Glenfable Peninsula toward Gortmore, County Kerry"
    )
    # thumbnail_text mirrors ost_text by construction (single source of truth).
    assert lf["thumbnail_text"] == "Glenfable Peninsula toward Gortmore, County Kerry"
    assert lf["recording_date"] == "2025-09-14"


def test_format_ost_audit_missing_thumbnail_proxy_falls_back_to_per_lf_locations():
    """When per_lf_ost_text lacks an entry, per_lf_locations is the fallback
    for both ost_text and thumbnail_text — same priority cascade as the
    OST burn path in run_day_pipeline."""
    state = make_state(
        per_lf_ost_text={},  # canonical missing
        per_lf_titles={2: "Coastal drive — County Clare"},
        per_lf_recording_date={2: "2025-09-15"},
        per_lf_locations={2: "County Clare"},
    )
    payload = format_ost_audit("2025-09-15", state, [2])
    lf = payload["lfs"][0]
    assert lf["lf_index"] == 2
    assert lf["ost_text"] == "County Clare"
    assert lf["thumbnail_text"] == "County Clare"
    assert lf["youtube_title"] == "Coastal drive — County Clare"
    assert lf["recording_date"] == "2025-09-15"


def test_format_ost_audit_multi_lf_day_ordered_by_index():
    state = make_state(
        per_lf_ost_text={
            1: "Ballymock to Glenfable via N86, County Kerry",
            2: "Glenfable to Tralee via R559, County Kerry",
            3: "Tralee to Limerick via N69, County Kerry",
        },
        per_lf_titles={
            1: "Driving — Ballymock to Glenfable",
            2: "Driving — Glenfable to Tralee",
            3: "Driving — Tralee to Limerick",
        },
        per_lf_recording_date={1: "2025-09-14", 2: "2025-09-14", 3: "2025-09-14"},
        per_lf_locations={},
    )
    payload = format_ost_audit("2025-09-14", state, [3, 1, 2])
    indices = [lf["lf_index"] for lf in payload["lfs"]]
    assert indices == [1, 2, 3]
    assert payload["lfs"][0]["ost_text"].startswith("Ballymock")
    assert payload["lfs"][2]["ost_text"].startswith("Tralee")


def test_write_ost_audit_creates_parent_dir_and_returns_path(tmp_path):
    output_base = tmp_path / "out" / "project"
    payload = {
        "date": "2025-09-14",
        "generated_at": "2026-05-10T14:00:00Z",
        "lfs": [
            {
                "lf_index": 1,
                "ost_text": "x",
                "youtube_title": "x",
                "thumbnail_text": "x",
                "recording_date": "2025-09-14",
            }
        ],
    }
    path = write_ost_audit(output_base, "2025-09-14", payload)
    assert path == output_base / "ost_audit_2025-09-14.json"
    assert path.exists()
    loaded = json.loads(path.read_text())
    assert loaded["date"] == "2025-09-14"
    assert loaded["lfs"][0]["lf_index"] == 1


def test_compute_ost_audit_hash_is_stable_across_generated_at_changes():
    """Hash must reflect the LF content, not the timestamp — otherwise
    a re-prompt fires on every gate entry even when nothing changed."""
    p1 = {
        "date": "2025-09-14",
        "generated_at": "2026-05-10T14:00:00Z",
        "lfs": [
            {
                "lf_index": 1,
                "ost_text": "A",
                "youtube_title": "T",
                "thumbnail_text": "A",
                "recording_date": "2025-09-14",
            }
        ],
    }
    p2 = dict(p1, generated_at="2026-05-10T15:00:00Z")
    assert compute_ost_audit_hash(p1) == compute_ost_audit_hash(p2)


def test_compute_ost_audit_hash_changes_when_ost_text_changes():
    p1 = {
        "date": "2025-09-14",
        "generated_at": "2026-05-10T14:00:00Z",
        "lfs": [
            {
                "lf_index": 1,
                "ost_text": "A",
                "youtube_title": "T",
                "thumbnail_text": "A",
                "recording_date": "2025-09-14",
            }
        ],
    }
    p2 = {
        "date": "2025-09-14",
        "generated_at": "2026-05-10T14:00:00Z",
        "lfs": [
            {
                "lf_index": 1,
                "ost_text": "B",
                "youtube_title": "T",
                "thumbnail_text": "B",
                "recording_date": "2025-09-14",
            }
        ],
    }
    assert compute_ost_audit_hash(p1) != compute_ost_audit_hash(p2)


def test_the_postburn_proof_reads_frames_at_native_shorts_width(tmp_path, monkeypatch):
    """The post-burn check counts glyph strokes, so it must keep reading frames at
    the shorts' own width — a downscale smears the very strokes it counts. Pinned
    because the sampler is shared with ``blur_gate``, whose spot-check samples at a
    far smaller width."""
    short = tmp_path / "short_01.mp4"
    short.write_bytes(b"x")
    cmds = []

    def fake_run(cmd, **kwargs):
        cmds.append(cmd)
        return SimpleNamespace(returncode=0, stdout=b"\x80" * (SHORTS_WIDTH * 16))

    monkeypatch.setattr(subprocess, "run", fake_run)
    present, why = sidecar.ost_text_present_in_pixels(short)

    assert len(cmds) == 3  # one read per sample time inside the title's plateau
    assert f"scale={SHORTS_WIDTH}:-2,format=gray" in cmds[0]
    # Flat mid-grey carries neither near-white glyphs nor their black border.
    assert present is False and "no burned text" in why
