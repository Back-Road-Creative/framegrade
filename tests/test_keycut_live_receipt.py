"""keycut's LIVE assembly receipt against framegrade's own loader.

``tests/fixtures/assembly_receipt_v1.json`` is the frozen receipt framegrade's
loader was built against; these tests run the real keycut on a real (tiny)
master and prove the receipt keycut actually writes is accepted by
``load_receipt`` and maps output time back to source time.
"""

import json
import logging
import shutil
import subprocess
from pathlib import Path

import pytest

from gomoveshift.pipeline import day_pipeline
from gomoveshift.pipeline.edit_timeline import EditTimeline, load_receipt

FIXTURE = Path(__file__).parent / "fixtures" / "assembly_receipt_v1.json"
needs_ffmpeg = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg/ffprobe not installed"
)


def _make_master(path: Path) -> None:
    """12 s, 10 fps, a keyframe every 2 s (scene-cut keyframes off)."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=12",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
            "-c:v", "libx264", "-g", "20", "-keyint_min", "20", "-sc_threshold", "0",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path),
        ],
        check=True,
    )  # fmt: skip


@needs_ffmpeg
def test_live_receipt_validates_and_maps_output_to_source(tmp_path):
    import keycut

    master = tmp_path / "master.mp4"
    _make_master(master)
    out, receipt = keycut.extract_and_concat_ranges_with_receipt(
        master, [(2.0, 5.0), (8.0, 11.0)], tmp_path / "out.mp4", tmp_path / "work", max_gap_sec=2.0
    )
    assert out is not None and receipt is not None
    live = receipt.to_dict()

    loaded = load_receipt(live)  # raises ReceiptError if keycut's output is not accepted
    assert [(s.source_start, s.source_end) for s in loaded.spans] == [(2.0, 5.0), (8.0, 11.0)]
    assert len(loaded.spans) == 2 and loaded.spans[1].output_start == pytest.approx(
        loaded.spans[0].output_end
    )

    tl = EditTimeline.from_receipt(loaded)
    t_out = loaded.spans[1].output_start + 1.0  # one second into the second span
    # keycut sets span lengths from the probed piece duration, which carries
    # container/audio-priming slack (~0.06 s here), so allow 0.15 s.
    (pos,) = tl.output_to_source(t_out)
    assert pos.span_index == 1 and pos.time == pytest.approx(9.0, abs=0.15)
    (first,) = tl.output_to_source(0.5)
    assert first.time == pytest.approx(2.5, abs=0.15)


def test_live_receipt_shape_matches_frozen_fixture():
    """keycut's receipt keys equal the frozen fixture's, at every level."""
    from keycut.receipt import AssemblyReceipt, ReceiptSpan

    frozen = json.loads(FIXTURE.read_text(encoding="utf-8"))
    span = ReceiptSpan(
        index=0, requested_indices=(0,), merged_start=2.0, merged_end=5.0,
        source_start=2.0, source_end=5.0, method="copy", reason="stream_copy",
        output_start=0.0, output_end=3.0, duration_source="probed",
    )  # fmt: skip
    live = AssemblyReceipt(
        keycut_version="0.2.0", master_path="m.mp4", master_size_bytes=1,
        max_gap_sec=2.0, requested=((2.0, 5.0),), spans=(span,),
        output_path="o.mp4", output_duration=3.0,
    ).to_dict()  # fmt: skip
    assert live["schema"] == frozen["schema"] == "keycut-assembly-receipt/1"
    assert set(live) == set(frozen)
    assert set(live["master"]) == set(frozen["master"])
    assert set(live["output"]) == set(frozen["output"])
    assert set(live["spans"][0]) == set(frozen["spans"][0])
    assert live["provenance"] == frozen["provenance"] and live["time_base"] == frozen["time_base"]


def test_unvalidated_receipt_makes_extract_and_concat_segments_return_none(
    tmp_path, monkeypatch, caplog
):
    master = tmp_path / "master.mp4"
    master.write_bytes(b"fake-master")
    out = tmp_path / "lf_01_segments.mp4"
    good = json.loads(FIXTURE.read_text(encoding="utf-8"))
    good["spans"][0]["method"] = "teleport"  # not a method the loader accepts

    class _BadReceipt:
        def to_dict(self):
            return good

    def _cut(*a, **kw):
        out.write_bytes(b"concat")
        return out, _BadReceipt()

    monkeypatch.setattr(day_pipeline, "extract_and_concat_ranges_with_receipt", _cut)
    seg = type("S", (), {"source_file_index": 0, "start_time": 1.0, "end_time": 5.0})()
    with caplog.at_level(logging.ERROR):
        result = day_pipeline.extract_and_concat_segments(
            master, [seg], [(0.0, 100.0)], out, tmp_path / "work"
        )
    assert result is None
    assert "spans[0].method" in caplog.text
    # The concat a bad receipt describes is removed, so a resume cannot reuse it.
    assert not out.exists()
