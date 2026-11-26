"""A re-render of a day's shorts leaves only this plan's shorts on disk.

Galley Head 2026-11-25: the 11-22 plan rendered ``short_13.mp4``. The 11-25
plan dropped rank 13 (LF 1 drew 8 shorts for 7 publish slots) and refilled
rank 15, so it rendered 01-12, 14 and 15 and never touched the old 13. Phase D
then found 15 shorts for 14 attributed and failed ``shorts_per_lf SPLIT-BRAIN``
-- and would on every retry, since nothing ever removed the stale file.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import gomoveshift.pipeline.day_pipeline as dp

_PTS = [(float(i), 52.14 + i * 0.0002, -10.27) for i in range(80)]


def _run(monkeypatch, tmp_path, ranks, render=lambda **kw: True):
    def _render(**kw):
        ok = render(**kw)
        if ok:
            Path(kw["output_path"]).write_bytes(b"new")
        return ok

    monkeypatch.setattr(dp, "_render_short_ffmpeg", _render)
    monkeypatch.setattr(dp, "trim_gps_to_clip_timeline", lambda p, r: p)
    monkeypatch.setattr(
        dp, "_resolve_short_ost_text", lambda c, p, r, d, drive_location=None: (c, 0)
    )
    monkeypatch.setattr(dp, "compute_day_file_boundaries", lambda b: [])
    monkeypatch.setattr(
        dp, "assign_shorts_to_lfs_by_source_window", lambda **k: {1: range(len(ranks))}
    )
    return dp._render_day_shorts(
        day_dir=tmp_path,
        date="2026-01-01",
        day_batch=SimpleNamespace(episodes=[]),
        short_ranges=[
            SimpleNamespace(rank=r, start_sec=60.0 * i, end_sec=60.0 * i + 38.0)
            for i, r in enumerate(ranks)
        ],
        allow_short_count=True,
        gps_points_by_date={"2026-01-01": _PTS},
        lf_plans=[SimpleNamespace(index=1)],
        per_lf_ost_text={1: "Dingle to Gortmore via R559"},
        per_lf_locations={1: "County Kerry"},
        location="Drive Wide",
        short_ost_text_fn=lambda p, s: "Annascaul",
        master_source_for_extract=Path("/dev/null"),
        state={"ost_audit_confirmed": True, "ost_audit_confirmed_this_run": True},
    )


def _seed(shorts: Path, *names: str) -> None:
    shorts.mkdir(parents=True, exist_ok=True)
    for n in names:
        (shorts / n).write_bytes(b"old")


def test_a_rank_this_plan_dropped_is_removed_with_its_sidecar(monkeypatch, tmp_path):
    shorts = tmp_path / "shorts"
    _seed(shorts, "short_01.mp4", "short_13.mp4", "short_13.mp4.picture_gate.json", "notes.txt")
    paths, _, _ = _run(monkeypatch, tmp_path, [1, 15])
    assert sorted(p.name for p in paths) == ["short_01.mp4", "short_15.mp4"]
    assert sorted(p.name for p in shorts.glob("short_[0-9][0-9].mp4")) == [
        "short_01.mp4",
        "short_15.mp4",
    ]
    assert not (shorts / "short_13.mp4.picture_gate.json").exists()
    assert (shorts / "short_01.mp4").read_bytes() == b"new"
    assert (shorts / "notes.txt").exists()


def test_a_failed_render_never_leaves_the_earlier_plans_file_under_its_name(monkeypatch, tmp_path):
    shorts = tmp_path / "shorts"
    _seed(shorts, "short_02.mp4")
    paths, _, _ = _run(
        monkeypatch,
        tmp_path,
        [1, 2],
        render=lambda **kw: not kw["output_path"].name.endswith("02.mp4"),
    )
    assert [p.name for p in paths] == ["short_01.mp4"]
    assert not (shorts / "short_02.mp4").exists()
