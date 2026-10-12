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
