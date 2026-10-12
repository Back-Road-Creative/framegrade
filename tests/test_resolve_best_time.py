"""TDD for the best-time data path in ``resolve_shorts_times_for_region``.

Two sources must agree before the resolver moves a slot off the configured
default: our own ranked ``slot_performance`` (Source B, all platforms) and
the platform-native ``audience_online`` table (Source A, Meta only today).
The data only ever picks WHICH configured slot to prefer — it never invents
an hour outside ``config.drive.shorts_slot_pool_by_region``.
"""

import sqlite3

import pytest

from gomoveshift.config import Config
from gomoveshift.pipeline import schedule_constants
from gomoveshift.tracking.schema import ensure_slot_performance_schema
from gomoveshift.tracking.slot_performance import ensure_audience_online_schema


def _conn():
    conn = sqlite3.connect(":memory:")
    ensure_slot_performance_schema(conn)
    ensure_audience_online_schema(conn)
    return conn


def _insert_slot(conn, region, platform, weekday, hour_local, n, mean_views):
    conn.execute(
        "INSERT INTO slot_performance "
        "(region, platform, weekday, hour_local, n, mean_views, mean_engagement) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (region, platform, weekday, hour_local, n, mean_views, 0.0),
    )
    conn.commit()


def _insert_online(conn, platform, weekday, hour_local, n_online):
    conn.execute(
        "INSERT INTO audience_online (platform, weekday, hour_local, n_online, fetched_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (platform, weekday, hour_local, n_online, "2026-09-10T00:00:00"),
    )
    conn.commit()


@pytest.fixture
def cfg():
    c = Config()
    c.drive.shorts_slot_pool_by_region = {"ireland": ["08:00", "12:00"]}
    return c


class TestNoConnBackwardCompat:
    def test_existing_callers_unaffected(self, cfg):
        # No conn/platform passed -> identical to the default behaviour.
        assert schedule_constants.resolve_shorts_times_for_region(cfg, "ireland") == [
            "08:00",
            "12:00",
        ]


class TestSourcesAgree:
    def test_b_top_slot_in_a_top3_moves_the_slot(self, cfg):
        conn = _conn()
        # Source B: instagram's 12:00 (hour 12) wins on mean_views, n>=min_n.
        _insert_slot(conn, "ireland", "instagram", weekday=1, hour_local=12, n=8, mean_views=500)
        _insert_slot(conn, "ireland", "instagram", weekday=2, hour_local=8, n=8, mean_views=100)
        # Source A: audience is online most at hour 12 (top-3 includes 12).
        _insert_online(conn, "instagram", weekday=1, hour_local=12, n_online=900)
        _insert_online(conn, "instagram", weekday=2, hour_local=9, n_online=50)
        _insert_online(conn, "instagram", weekday=3, hour_local=20, n_online=10)

        result = schedule_constants.resolve_shorts_times_for_region(
            cfg, "ireland", conn=conn, platform="instagram"
        )
        assert result[0] == "12:00"
        assert set(result) == {"08:00", "12:00"}


class TestSourcesDisagree:
    def test_b_top_slot_not_in_a_top3_keeps_configured_window(self, cfg, caplog):
        conn = _conn()
        _insert_slot(conn, "ireland", "instagram", weekday=1, hour_local=12, n=8, mean_views=500)
        # Source A never sees hour 12 in its top-3 -> disagreement.
        _insert_online(conn, "instagram", weekday=1, hour_local=8, n_online=900)
        _insert_online(conn, "instagram", weekday=2, hour_local=9, n_online=800)
        _insert_online(conn, "instagram", weekday=3, hour_local=20, n_online=700)

        with caplog.at_level("INFO"):
            result = schedule_constants.resolve_shorts_times_for_region(
                cfg, "ireland", conn=conn, platform="instagram"
            )
        assert result == ["08:00", "12:00"]
        assert any("disagree" in r.message for r in caplog.records)


class TestBelowMinN:
    def test_n_below_min_n_keeps_configured_window(self, cfg, caplog):
        conn = _conn()
        _insert_slot(conn, "ireland", "instagram", weekday=1, hour_local=12, n=2, mean_views=500)
        _insert_online(conn, "instagram", weekday=1, hour_local=12, n_online=900)

        with caplog.at_level("INFO"):
            result = schedule_constants.resolve_shorts_times_for_region(
                cfg, "ireland", conn=conn, platform="instagram"
            )
        assert result == ["08:00", "12:00"]
        assert any("min_n" in r.message or "no ranked" in r.message for r in caplog.records)


class TestSourceAAbsentForPlatform:
    def test_youtube_has_no_audience_online_b_alone_decides(self, cfg, caplog):
        conn = _conn()
        _insert_slot(conn, "ireland", "youtube", weekday=1, hour_local=8, n=10, mean_views=1000)
        # No audience_online rows for youtube at all -- expected, not a fault.

        with caplog.at_level("INFO"):
            result = schedule_constants.resolve_shorts_times_for_region(
                cfg, "ireland", conn=conn, platform="youtube"
            )
        assert result[0] == "08:00"
        assert not any(
            "fault" in r.message.lower() or "error" in r.message.lower() for r in caplog.records
        )


class TestCandidateOutsidePool:
    def test_best_slot_not_in_configured_pool_keeps_configured_window(self, cfg, caplog):
        conn = _conn()
        # 16:00 is not in ireland's pool -- must never be invented.
        _insert_slot(conn, "ireland", "instagram", weekday=1, hour_local=16, n=10, mean_views=999)
        _insert_online(conn, "instagram", weekday=1, hour_local=16, n_online=900)

        with caplog.at_level("INFO"):
            result = schedule_constants.resolve_shorts_times_for_region(
                cfg, "ireland", conn=conn, platform="instagram"
            )
        assert result == ["08:00", "12:00"]
        assert any("pool" in r.message for r in caplog.records)


class TestBrokenDatabase:
    def test_broken_connection_keeps_configured_window_never_raises(self, cfg, caplog):
        conn = sqlite3.connect(":memory:")
        conn.close()  # any query against this now raises ProgrammingError

        with caplog.at_level("INFO"):
            result = schedule_constants.resolve_shorts_times_for_region(
                cfg, "ireland", conn=conn, platform="instagram"
            )
        assert result == ["08:00", "12:00"]
