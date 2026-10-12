"""The split branch must cut each part from the right point in the source.

``VideoEncoder.extract_segment`` builds ``ffmpeg -ss <start> -i <source> -t <dur>``,
so the (start, end) pairs ``trim_and_encode`` receives are absolute timestamps into
whatever source it is handed. The multi-cut branch of ``VideoAssembler.execute``
rebased its keeps to the cut point while still passing the full, un-trimmed source,
so every part after the first cut from the *opening* of the drive instead of its own
footage — part02 shipped a duplicate of part01's first minutes.

Chapters are the opposite: they index the finished file, so they stay relative to
each part. Both are pinned here.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from gomoveshift.assembly.assembler import AssemblyPlan, VideoAssembler
from gomoveshift.assembly.encoder import Chapter
from gomoveshift.config import Config

SOURCE = Path("/fake/drive.mp4")
# Dead spot removed at 100-110s; split into two files at 200s.
#   part01 = source [0,200): plays 0-100 then 110-200
#   part02 = source [200,300)
SEGMENTS = [(0.0, 100.0), (110.0, 300.0)]
CUTS = [200.0]
CHAPTERS = [Chapter(0.0, "Start"), Chapter(250.0, "Entering X")]


def _plan(out_dir: Path) -> AssemblyPlan:
    return AssemblyPlan(
        source_path=SOURCE,
        segments_to_keep=list(SEGMENTS),
        cut_points=list(CUTS),
        chapters=list(CHAPTERS),
        output_paths=[out_dir / "part01.mp4", out_dir / "part02.mp4"],
    )


def test_split_parts_seek_to_absolute_source_offsets(tmp_path, monkeypatch):
    """Every ffmpeg seek must land on this part's own footage in the full source."""
    import keycut

    monkeypatch.setattr(keycut, "probe_keyframes", lambda *a, **k: [])
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return MagicMock(returncode=0, stdout="{}", stderr="")

    with patch("gomoveshift.assembly.encoder.subprocess.run", side_effect=fake_run):
        VideoAssembler(Config()).execute(_plan(tmp_path))

    seeks = [c for c in calls if c and c[0] == "ffmpeg" and "-ss" in c]
    cut_from = [(float(c[c.index("-ss") + 1]), float(c[c.index("-t") + 1])) for c in seeks]
    # part01: 0-100 then 110-200. part02: 200-300 — NOT 0-100 all over again.
    assert cut_from == [(0.0, 100.0), (110.0, 90.0), (200.0, 100.0)]
    # The offsets are absolute because the input stays the full, un-trimmed source.
    assert {c[c.index("-i") + 1] for c in seeks} == {str(SOURCE)}


def test_split_chapters_stay_relative_to_their_own_part(tmp_path):
    """Chapters index the finished file, so they keep the cut point subtracted."""
    captured: list[tuple[list, list]] = []

    def fake_encode(source, output, keeps, chapters=None):
        captured.append((keeps, chapters))
        return True

    assembler = VideoAssembler(Config())
    assembler.encoder = MagicMock()
    assembler.encoder.trim_and_encode.side_effect = fake_encode
    assembler.execute(_plan(tmp_path))

    part02_keeps, part02_chapters = captured[1]
    assert part02_keeps == [(200.0, 300.0)]
    # Source 250s is 50s into part02's own timeline (nothing trimmed inside it).
    assert [(c.start_time, c.title) for c in part02_chapters] == [(50.0, "Entering X")]
