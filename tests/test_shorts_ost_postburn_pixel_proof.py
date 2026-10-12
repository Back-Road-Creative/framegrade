"""The shorts OST proof reads burned pixels, not the entry that claims them.

``shorts_ost_audit_<date>.json`` recorded the text each short was *asked* to
burn. That is an attestation: it reads the same whether drawtext composited the
title or the filter chain dropped it. Nothing ever opened the rendered short.
The post-burn check reads the FINAL's own pixels — the title band must carry the
white-glyph / black-border signature the burn puts there, and markedly more of it
than a control band from the same frame. One-sided like the blur proof: a clear
negative fails upstream of upload, and anything unmeasurable falls back to the
attestation rather than failing an already-rendered day.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from gomoveshift.pipeline import ost_audit_sidecar as sidecar
from gomoveshift.pipeline.ffmpeg_filters import FONT_PATH, LOC_FONT_SIZE

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

DATE = "2026-02-20"
# Flat mid-grey at real shorts geometry: neither near-white nor near-black, so
# every pixel the check counts came from the burn and nothing else.
_SRC = "color=c=gray:s=1080x1920:r=6:d=6"


def _render(path: Path, vf) -> Path:
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", _SRC]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-qp", "0", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True, timeout=120)
    return path


@pytest.fixture
def burned(tmp_path):
    """The real burn: same font, size, colours and y as `build_ost_filter` uses."""
    return _render(
        tmp_path / "short_01.mp4",
        f"drawtext=fontfile='{FONT_PATH}':text='Clare, Ireland':"
        f"fontsize={LOC_FONT_SIZE}:fontcolor=white:borderw=3:bordercolor=black:"
        f"x=(w-text_w)/2:y=280",
    )


@pytest.fixture
def unburned(tmp_path):
    """The title never composited — the failure a marker-only audit calls success."""
    return _render(tmp_path / "short_02.mp4", None)


@needs_ffmpeg
def test_a_burned_short_passes_and_the_gate_lets_it_through(burned):
    present, why = sidecar.ost_text_present_in_pixels(burned)
    assert present, why
    sidecar.postburn_ost_gate_or_raise(DATE, [{"rank": 1, "path": str(burned)}])


@needs_ffmpeg
def test_a_short_whose_title_never_burned_is_caught_before_upload(unburned):
    present, why = sidecar.ost_text_present_in_pixels(unburned)
    assert not present
    assert "no burned text" in why.lower()
    with pytest.raises(sidecar.OstPostBurnError) as excinfo:
        sidecar.postburn_ost_gate_or_raise(DATE, [{"rank": 3, "path": str(unburned)}])
    assert "rank 3" in str(excinfo.value)


def test_anything_unmeasurable_falls_back_to_the_attestation(tmp_path):
    """A missing file, an unreadable one, or an entry with no rendered path (the
    pre-burn copy) may not fail a day whose shorts are already encoded."""
    junk = tmp_path / "short_04.mp4"
    junk.write_bytes(b"not a video")
    assert sidecar.ost_text_present_in_pixels(junk)[0]
    assert sidecar.ost_text_present_in_pixels(tmp_path / "gone.mp4")[0]
    sidecar.postburn_ost_gate_or_raise(DATE, [{"rank": 1, "path": str(junk)}])
    sidecar.postburn_ost_gate_or_raise(DATE, [{"rank": 1, "ost_text": "Clare, Ireland"}])


def test_the_payload_carries_the_path_without_reopening_a_confirmed_gate():
    """The path is where the artefact landed, not what was burned — binding a
    confirmation to it would reopen the gate on every ``--output-dir`` change."""
    entry = {"rank": 1, "ost_text": "Clare, Ireland"}
    payload = sidecar.format_shorts_ost_audit(DATE, [{**entry, "path": "/out/short_01.mp4"}])
    assert payload["shorts"][0]["path"] == "/out/short_01.mp4"
    assert sidecar.compute_shorts_ost_audit_hash(payload) == sidecar.compute_shorts_ost_audit_hash(
        sidecar.format_shorts_ost_audit(DATE, [entry])
    )


def test_run_day_pipeline_binds_every_audit_entry_to_its_rendered_short(tmp_path):
    """The live call site: the entries the post-burn check reads are the ones
    ``run_day_pipeline`` writes, and each one names the short it describes."""
    import json

    from _state_helpers import make_state
    from test_day_pipeline_shorts_ost_gate import _patch_and_run

    result, exc, _, output_base = _patch_and_run(
        tmp_path, make_state(), lf_confirmed=True, shorts_confirmed=True
    )
    assert exc is None and result is not None
    written = json.loads((output_base / f"shorts_ost_audit_{DATE}.json").read_text())
    assert written["shorts"]
    for short in written["shorts"]:
        assert short["path"].endswith(f"short_{short['rank']:02d}.mp4")
    assert {s["path"] for s in written["shorts"]} == {str(p) for p in result.short_paths}
