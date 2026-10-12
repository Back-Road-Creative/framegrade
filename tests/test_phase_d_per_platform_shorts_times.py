"""Phase D resolves the shorts window PER
first-party platform and hands the platforms whose window
differs from the slot to ``build_publish_plan(shorts_times_by_platform=...)``,
so the sidecar's ``scheduled_at_utc_by_platform`` is finally populated by a
real drive rather than only by tests.
"""

import logging
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from gomoveshift.config import Config, DriveConfig
from gomoveshift.pipeline.drive_phases.phase_d import _emit_publish_plan_sidecar
from gomoveshift.pipeline.drive_phases.phase_d_scheduling import (
    _lf_day_key,
    _shorts_times_by_platform,
)
from gomoveshift.publishing.models import MediaFingerprint
from gomoveshift.publishing.plan_io import load_publish_plan
from gomoveshift.publishing.queue import open_queue_db
from gomoveshift.tracking.schema import ensure_slot_performance_schema
from gomoveshift.tracking.slot_performance import ensure_audience_online_schema

FP = MediaFingerprint(
    path="/x/short_01.mp4",
    size_bytes=1,
    duration_seconds=9.0,
    width=1080,
    height=1920,
    codec="h264",
    sha256="a" * 64,
)
FINGERPRINT = "gomoveshift.publishing.media_fingerprint.fingerprint_media"
WINDOW = ["12:00", "16:00"]


def _config(tmp_path, **route):
    """A real Config: a region whose flat window is WINDOW and whose pool is
    the same two hours, spooling to ``tmp_path`` (the queue database the
    platform-window resolver reads ``slot_performance`` from)."""
    c = Config()
    c.drive = DriveConfig(**route)
    c.drive.shorts_publish_times = list(WINDOW)
    c.drive.shorts_publish_times_by_region = {"ireland": list(WINDOW)}
    c.drive.shorts_slot_pool_by_region = {"ireland": list(WINDOW)}
    c.drive.long_form_publish_days = [1]  # tue -- what --lf-day tue sets
    c.seo.storage_path = str(tmp_path)
    return c


def _seed_instagram_prefers_16(tmp_path):
    """Source B (our ranking) and Source A (audience online) both put
    instagram's best hour at 16:00; nothing recorded for any other platform."""
    conn = open_queue_db(tmp_path)
    ensure_slot_performance_schema(conn)
    ensure_audience_online_schema(conn)
    conn.execute(
        "INSERT INTO slot_performance "
        "(region, platform, weekday, hour_local, n, mean_views, mean_engagement) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("ireland", "instagram", 1, 16, 8, 500.0, 0.0),
    )
    conn.execute(
        "INSERT INTO slot_performance "
        "(region, platform, weekday, hour_local, n, mean_views, mean_engagement) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("ireland", "instagram", 1, 12, 8, 100.0, 0.0),
    )
    conn.execute(
        "INSERT INTO audience_online (platform, weekday, hour_local, n_online, fetched_at) "
        "VALUES (?, ?, ?, ?, ?)",
        ("instagram", 1, 16, 900, "2026-09-10T00:00:00"),
    )
    conn.commit()
    conn.close()


class TestLfDayKey:
    def test_one_configured_day_inverts_to_its_key(self, tmp_path):
        assert _lf_day_key(_config(tmp_path)) == "tue"

    def test_anything_but_one_valid_day_is_none(self, tmp_path):
        cfg = _config(tmp_path)
        for days in ([], [1, 4], [7], ["tue"]):
            cfg.drive.long_form_publish_days = days
            assert _lf_day_key(cfg) is None


class TestShortsTimesByPlatform:
    def test_no_data_means_every_platform_keeps_the_slot(self, tmp_path):
        cfg = _config(tmp_path)
        assert _shorts_times_by_platform(cfg, "ireland", ("youtube", "instagram")) == {}

    def test_only_the_platform_the_data_moved_is_listed(self, tmp_path):
        cfg = _config(tmp_path)
        _seed_instagram_prefers_16(tmp_path)
        out = _shorts_times_by_platform(cfg, "ireland", ("youtube", "instagram"))
        assert out == {"instagram": ["16:00", "12:00"]}

    def test_lf_day_comes_from_long_form_publish_days(self, tmp_path):
        # A per-LF-day region map: tue's window differs from _default. The
        # resolver must be asked for the day the drive actually lands on.
        cfg = _config(tmp_path)
        cfg.drive.shorts_publish_times_by_region = {
            "ireland": {"_default": ["08:00", "09:00"], "tue": list(WINDOW)}
        }
        _seed_instagram_prefers_16(tmp_path)
        out = _shorts_times_by_platform(cfg, "ireland", ("instagram",))
        assert out == {"instagram": ["16:00", "12:00"]}

    def test_explicit_shorts_times_override_wins_over_the_data(self, tmp_path):
        # --shorts-times mutated the flat window upstream (the drift gate is
        # skipped for it); the data path must not reorder a window the
        # operator chose by hand.
        cfg = _config(tmp_path)
        cfg.drive.shorts_publish_times = ["07:00", "20:00"]
        _seed_instagram_prefers_16(tmp_path)
        assert _shorts_times_by_platform(cfg, "ireland", ("instagram",)) == {}

    def test_no_region_or_no_seo_config_is_a_noop(self, tmp_path):
        cfg = _config(tmp_path)
        assert _shorts_times_by_platform(cfg, None, ("instagram",)) == {}
        bare = SimpleNamespace(drive=cfg.drive)
        assert _shorts_times_by_platform(bare, "ireland", ("instagram",)) == {}

    def test_database_failure_keeps_the_slot_for_everyone(self, tmp_path, caplog):
        cfg = _config(tmp_path)
        _seed_instagram_prefers_16(tmp_path)
        caplog.set_level(logging.INFO, logger="gomoveshift.pipeline.drive_phases.phase_d")
        with patch(
            "gomoveshift.pipeline.drive_phases.phase_d_scheduling.open_queue_db",
            side_effect=OSError("disk gone"),
        ):
            out = _shorts_times_by_platform(cfg, "ireland", ("instagram",))
        assert out == {}
        assert "disk gone" in caplog.text


class TestSidecarCarriesThePlatformHours:
    def _kwargs(self, tmp_path, cfg):
        slots = [
            SimpleNamespace(short_index=10, date="2026-04-01", time="12:00:00"),
            SimpleNamespace(short_index=11, date="2026-04-01", time="16:00:00"),
        ]
        return dict(
            output_base=tmp_path,
            project_base="p",
            drive_plan=SimpleNamespace(regular_lfs=[SimpleNamespace(index=1)]),
            social_posts_per_lf={
                1: [SimpleNamespace(short_index=10), SimpleNamespace(short_index=11)]
            },
            drive_schedule=SimpleNamespace(shorts_slots=slots),
            youtube_uploads={1: {"video_id": "vid-1", "publish_at": "2026-03-31T08:00:00-05:00"}},
            per_lf_recording_date={1: "2026-03-30"},
            shorts_paths_by_lf={1: [str(tmp_path / "s1.mp4"), str(tmp_path / "s2.mp4")]},
            publish_pass=True,
            config=cfg,
            region="ireland",
        )

    def test_instagram_rows_move_youtube_rows_stay(self, tmp_path):
        cfg = _config(
            tmp_path,
            publishing_route_youtube="first_party",
            publishing_route_instagram="first_party",
        )
        _seed_instagram_prefers_16(tmp_path)
        with patch(FINGERPRINT, return_value=FP):
            plan_path = _emit_publish_plan_sidecar(**self._kwargs(tmp_path, cfg))
        by_short = {r.short_index: r.schedule for r in load_publish_plan(plan_path).shorts}
        # 12:00 EDT = 16:00Z; 16:00 EDT = 20:00Z. Instagram's window is
        # reversed, so its first short of the day takes the later hour.
        utc = timezone.utc
        assert by_short[10].scheduled_at_utc == datetime(2026, 4, 1, 16, tzinfo=utc)
        assert by_short[10].scheduled_at_utc_by_platform == {
            "instagram": datetime(2026, 4, 1, 20, tzinfo=utc)
        }
        assert by_short[11].scheduled_at_utc_by_platform == {
            "instagram": datetime(2026, 4, 1, 16, tzinfo=utc)
        }

    def test_no_region_writes_no_per_platform_times(self, tmp_path):
        cfg = _config(tmp_path, publishing_route_instagram="first_party")
        _seed_instagram_prefers_16(tmp_path)
        kwargs = self._kwargs(tmp_path, cfg)
        kwargs["region"] = None
        with patch(FINGERPRINT, return_value=FP):
            plan_path = _emit_publish_plan_sidecar(**kwargs)
        for r in load_publish_plan(plan_path).shorts:
            assert r.schedule.scheduled_at_utc_by_platform == {}
