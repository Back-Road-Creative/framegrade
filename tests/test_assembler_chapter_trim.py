"""Chapter timestamps must accumulate every removed trim.

Two regressions in ``VideoAssembler``:

1. Single-file ``_adjust_chapters_for_trims`` overwrote the running trim total
   instead of accumulating it — with >=2 removed gaps before a chapter, only the
   last gap was subtracted, landing "Entering X" on the wrong footage.
2. The multi-file split path subtracted only the cut offset and never adjusted
   for dead-spots removed *inside* the segment.
"""

from pathlib import Path
from unittest.mock import MagicMock

from gomoveshift.assembly.assembler import AssemblyPlan, VideoAssembler
from gomoveshift.assembly.encoder import Chapter
from gomoveshift.config import Config


def test_adjust_accumulates_multiple_trims():
    assembler = VideoAssembler(Config())
    # keep 10-20, 30-40, 60-100 -> removed gaps 0-10, 20-30, 40-60 (30s total
    # before a chapter at src 70s).
    keep = [(10.0, 20.0), (30.0, 40.0), (60.0, 100.0)]
    adjusted = assembler._adjust_chapters_for_trims([Chapter(70.0, "Entering X")], keep)
    assert len(adjusted) == 1
    # 70 - (10 + 10 + 20) = 30, NOT the last-gap-only 70 - 20 = 50.
    assert adjusted[0].start_time == 30.0


def test_multifile_adjusts_for_in_segment_trims():
    assembler = VideoAssembler(Config())
    captured = []

    def fake_encode(source, output, keeps, chapters=None):
        captured.append((output, keeps, chapters))
        return True

    assembler.encoder = MagicMock()
    assembler.encoder.trim_and_encode.side_effect = fake_encode

    plan = AssemblyPlan(
        source_path=Path("/fake/source.mp4"),
        # dead-spots removed at 50-60 (inside file 0) and 100-110 (at the cut).
        segments_to_keep=[(0.0, 50.0), (60.0, 100.0), (110.0, 200.0)],
        cut_points=[100.0],
        chapters=[Chapter(0.0, "Start"), Chapter(70.0, "Entering X")],
        output_paths=[Path("/out/part01.mp4"), Path("/out/part02.mp4")],
    )
    assembler.execute(plan)

    # File 0 covers src [0,100): plays 0-50 then 60-100. A chapter at src 70 sits
    # 10s after the removed 50-60 gap, so it lands at edited 60s, not 70s.
    file0_chapters = captured[0][2]
    entering = [c for c in file0_chapters if c.title == "Entering X"]
    assert entering, "Entering X chapter should survive in file 0"
    assert entering[0].start_time == 60.0
