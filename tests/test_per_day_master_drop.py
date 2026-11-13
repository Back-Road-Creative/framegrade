"""Master drops per-day inside on_day_complete, not at phase end."""

from types import SimpleNamespace
from unittest.mock import patch

from gomoveshift.pipeline.drive_phases.phase_bc_day_first import _maybe_drop_day_master

_STATE = SimpleNamespace(keep_masters=None, allow_short_count=None, superlong_parts=None)
_OK = {"ok": True, "reason": "", "missing": [], "bad": [], "count_actual": 1, "count_expected": 1}


def _day_with_finals(base, date):
    day_dir = base / f"day_{date}"
    [(day_dir / s).mkdir(parents=True) for s in ("regular_lfs", "shorts")]
    (day_dir / f"day_{date}_master.mp4").write_bytes(b"m" * 10)
    (day_dir / "regular_lfs" / "lf_01.mp4").write_bytes(b"lf")
    (day_dir / "shorts" / "short_01.mp4").write_bytes(b"s")
    return day_dir


def test_success_day_drops_its_own_master(tmp_path):
    master = _day_with_finals(tmp_path, "2026-02-20") / "day_2026-02-20_master.mp4"
    art = SimpleNamespace(date="2026-02-20", master_path=master)
    with patch("gomoveshift.pipeline.cleanup.verify_shorts_integrity", return_value=_OK):
        _maybe_drop_day_master(art, _STATE, tmp_path)
    assert not master.exists(), "master must drop inside the per-day callback"


def test_recovery_master_path_none_never_drops(tmp_path):
    master = _day_with_finals(tmp_path, "2026-02-20") / "day_2026-02-20_master.mp4"
    art = SimpleNamespace(date="2026-02-20", master_path=None)
    with patch("gomoveshift.pipeline.cleanup.drop_day_master") as mock_drop:
        _maybe_drop_day_master(art, _STATE, tmp_path)
    mock_drop.assert_not_called()
    assert master.exists(), "master_path=None (recovery) must never be dropped"


def test_shorts_count_shortfall_keeps_master(tmp_path):
    master = _day_with_finals(tmp_path, "2026-02-20") / "day_2026-02-20_master.mp4"
    art = SimpleNamespace(date="2026-02-20", master_path=master, planned_shorts_count=2)
    bad = {**_OK, "ok": False, "reason": "count mismatch", "count_actual": 1, "count_expected": 2}
    with patch("gomoveshift.pipeline.cleanup.verify_shorts_integrity", return_value=bad):
        _maybe_drop_day_master(art, _STATE, tmp_path)
    assert master.exists(), "shorts count shortfall must preserve the master"


def _gapped_day(base, date, numbers):
    day_dir = _day_with_finals(base, date)
    for f in (day_dir / "shorts").glob("short_*.mp4"):
        f.unlink()
    paths = []
    for n in numbers:
        p = day_dir / "shorts" / f"short_{n:02d}.mp4"
        p.write_bytes(b"s")
        paths.append(p)
    return day_dir, paths


_PROBE_OK = lambda _p, **_kw: SimpleNamespace(valid=True, duration=1.0, reason="")  # noqa: E731
_GAPPED = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26]


def _drop(day_dir, paths, planned):
    art = SimpleNamespace(
        date="2026-03-05",
        master_path=day_dir / "day_2026-03-05_master.mp4",
        planned_shorts_count=planned,
        short_paths=paths,
    )
    with patch("gomoveshift.pipeline.ffmpeg_runner.validate_video", _PROBE_OK):
        _maybe_drop_day_master(art, _STATE, day_dir.parent)
    return art.master_path


def test_gapped_numbering_all_present_drops_master(tmp_path):
    day_dir, paths = _gapped_day(tmp_path, "2026-03-05", _GAPPED)
    master = _drop(day_dir, paths, len(_GAPPED))
    assert not master.exists(), "dropped-and-refilled shorts (gapped numbers) are not missing"


def test_gapped_numbering_planned_short_absent_keeps_master(tmp_path):
    day_dir, paths = _gapped_day(tmp_path, "2026-03-05", _GAPPED)
    paths[3].unlink()
    master = _drop(day_dir, paths, len(_GAPPED))
    assert master.exists(), "a planned short absent from disk must still fail closed"


def test_verify_names_reports_the_absent_planned_short(tmp_path):
    from gomoveshift.pipeline.cleanup import verify_shorts_integrity

    day_dir, paths = _gapped_day(tmp_path, "2026-03-05", _GAPPED)
    names = [p.name for p in paths]
    paths[0].unlink()
    r = verify_shorts_integrity(
        day_dir, expected_count=len(names), expected_names=names, ffprobe_fn=_PROBE_OK
    )
    assert not r["ok"] and r["missing"] == ["short_01.mp4"]
