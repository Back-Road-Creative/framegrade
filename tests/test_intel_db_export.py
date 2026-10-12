"""Tests for IntelDB integration in gomoveshift/intel_db.py."""

import os
import sqlite3
import sys
import types
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from gomoveshift.intel_db import _get_intel_db, _log_video_to_intel_db  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_intel_db(path: str) -> None:
    """Create a minimal IntelDB-compatible SQLite database."""
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE scrape_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            platform TEXT NOT NULL,
            source_name TEXT,
            started_at TEXT,
            finished_at TEXT,
            status TEXT DEFAULT 'running',
            items_found INTEGER DEFAULT 0,
            items_imported INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE news_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            summary TEXT,
            source_count INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()


class _SimpleIntelDB:
    """Minimal IntelDB backed by a real SQLite file."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    def start_scrape_run(self, platform: str, source_name: str = "") -> int:
        cur = self.conn.execute(
            "INSERT INTO scrape_runs (platform, source_name, started_at, status) "
            "VALUES (?, ?, datetime('now'), 'running')",
            (platform, source_name),
        )
        self.conn.commit()
        return cur.lastrowid

    def upsert_news_item(
        self, date: str, summary: str, article_links: list, source_count: int = 0
    ) -> None:
        self.conn.execute(
            "INSERT INTO news_items (date, summary, source_count) VALUES (?, ?, ?)",
            (date, summary, source_count),
        )
        self.conn.commit()

    def finish_scrape_run(self, run_id: int, items_found: int = 0, items_imported: int = 0) -> None:
        self.conn.execute(
            "UPDATE scrape_runs SET status='success', finished_at=datetime('now'), "
            "items_found=?, items_imported=? WHERE id=?",
            (items_found, items_imported, run_id),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


# ---------------------------------------------------------------------------
# Tests: _get_intel_db
# ---------------------------------------------------------------------------


def test_get_intel_db_returns_none_on_missing_module(tmp_path, monkeypatch):
    """Returns None without raising when intel_db is not importable."""
    bad_dir = str(tmp_path / "nonexistent")
    import gomoveshift.intel_db as gm

    # ``_get_intel_db`` inserts the real ``_INTEL_SCRIPTS_DIR`` onto sys.path and
    # never removes it. Under xdist work-stealing, a prior test that called it
    # can run first on this worker, leaving intel_db importable via that
    # lingering path — popping sys.modules alone then isn't enough and this
    # test spuriously finds an instance. Strip the real dir so the import
    # genuinely fails (monkeypatch restores sys.path afterward).
    real_dir = gm._INTEL_SCRIPTS_DIR
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != real_dir])
    sys.modules.pop("intel_db", None)

    with patch.object(gm, "_INTEL_SCRIPTS_DIR", bad_dir):
        result = _get_intel_db(str(tmp_path / "db.sqlite"))
    assert result is None


def test_scripts_dir_has_no_default_and_adds_nothing_to_sys_path(tmp_path, monkeypatch):
    """With ``INTEL_SCRIPTS_DIR`` unset the module adds no directory to ``sys.path``."""
    import importlib

    import gomoveshift.intel_db as gm

    monkeypatch.delenv("INTEL_SCRIPTS_DIR", raising=False)
    fresh = importlib.reload(gm)
    assert fresh._INTEL_SCRIPTS_DIR == ""
    before = list(sys.path)
    sys.modules.pop("intel_db", None)
    fresh._get_intel_db(str(tmp_path / "db.sqlite"))
    assert sys.path == before


def test_scripts_dir_env_override_is_added_to_sys_path(tmp_path, monkeypatch):
    """An ``INTEL_SCRIPTS_DIR`` override still reaches ``sys.path``."""
    import importlib

    import gomoveshift.intel_db as gm

    scripts = str(tmp_path / "intel-scripts")
    monkeypatch.setenv("INTEL_SCRIPTS_DIR", scripts)
    monkeypatch.setattr(sys, "path", list(sys.path))
    sys.modules.pop("intel_db", None)
    try:
        fresh = importlib.reload(gm)
        assert fresh._INTEL_SCRIPTS_DIR == scripts
        fresh._get_intel_db(str(tmp_path / "db.sqlite"))
        assert scripts in sys.path
    finally:
        monkeypatch.delenv("INTEL_SCRIPTS_DIR", raising=False)
        importlib.reload(gm)


def test_get_intel_db_returns_instance_when_available(tmp_path):
    """Returns a DB instance when IntelDB can be imported."""
    db_path = str(tmp_path / "intel.db")
    _make_intel_db(db_path)
    intel_mod = types.ModuleType("intel_db")
    intel_mod.IntelDB = _SimpleIntelDB
    with patch.dict(sys.modules, {"intel_db": intel_mod}):
        db = _get_intel_db(db_path)
    assert db is not None
    db.close()


# ---------------------------------------------------------------------------
# Tests: _log_video_to_intel_db
# ---------------------------------------------------------------------------


def test_log_video_noop_when_no_env(tmp_path):
    """No-op (no raise) when INTEL_DB_PATH is not set."""
    os.environ.pop("INTEL_DB_PATH", None)
    _log_video_to_intel_db("clip_001.mp4", "completed")


def test_log_video_writes_scrape_run(tmp_path):
    """A scrape_run row is created with platform='gomoveshift-video'."""
    db_path = str(tmp_path / "intel.db")
    _make_intel_db(db_path)
    intel_mod = types.ModuleType("intel_db")
    intel_mod.IntelDB = _SimpleIntelDB
    with (
        patch.dict(sys.modules, {"intel_db": intel_mod}),
        patch.dict(os.environ, {"INTEL_DB_PATH": db_path}),
    ):
        _log_video_to_intel_db("clip_001.mp4", "completed")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM scrape_runs").fetchall()
    conn.close()
    assert len(rows) == 1
    assert rows[0]["platform"] == "gomoveshift-video"
    assert rows[0]["status"] == "success"
    assert rows[0]["items_imported"] == 1


def test_log_video_source_name_is_video_name(tmp_path):
    """source_name in scrape_runs equals the video file name."""
    db_path = str(tmp_path / "intel.db")
    _make_intel_db(db_path)
    intel_mod = types.ModuleType("intel_db")
    intel_mod.IntelDB = _SimpleIntelDB
    with (
        patch.dict(sys.modules, {"intel_db": intel_mod}),
        patch.dict(os.environ, {"INTEL_DB_PATH": db_path}),
    ):
        _log_video_to_intel_db("drive_20250901.mp4", "completed")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT source_name FROM scrape_runs").fetchone()
    conn.close()
    assert row["source_name"] == "drive_20250901.mp4"


def test_log_video_imports_zero_when_not_completed(tmp_path):
    """items_imported is 0 for non-completed status."""
    db_path = str(tmp_path / "intel.db")
    _make_intel_db(db_path)
    intel_mod = types.ModuleType("intel_db")
    intel_mod.IntelDB = _SimpleIntelDB
    with (
        patch.dict(sys.modules, {"intel_db": intel_mod}),
        patch.dict(os.environ, {"INTEL_DB_PATH": db_path}),
    ):
        _log_video_to_intel_db("clip.mp4", "failed")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT items_imported FROM scrape_runs").fetchone()
    conn.close()
    assert row["items_imported"] == 0


def test_log_video_summary_contains_video_name(tmp_path):
    """News item summary includes the video name."""
    db_path = str(tmp_path / "intel.db")
    _make_intel_db(db_path)
    intel_mod = types.ModuleType("intel_db")
    intel_mod.IntelDB = _SimpleIntelDB
    with (
        patch.dict(sys.modules, {"intel_db": intel_mod}),
        patch.dict(os.environ, {"INTEL_DB_PATH": db_path}),
    ):
        _log_video_to_intel_db("my_video.mp4", "completed")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT summary FROM news_items").fetchone()
    conn.close()
    assert "my_video.mp4" in row["summary"]


def test_log_video_noop_on_bad_db_path(tmp_path):
    """Non-fatal when DB can't be opened."""
    intel_mod = types.ModuleType("intel_db")

    class _FailingDB:
        def __init__(self, path):
            raise RuntimeError("Cannot open")

    intel_mod.IntelDB = _FailingDB
    with (
        patch.dict(sys.modules, {"intel_db": intel_mod}),
        patch.dict(os.environ, {"INTEL_DB_PATH": "/nonexistent/db.sqlite"}),
    ):
        _log_video_to_intel_db("video.mp4", "completed")
