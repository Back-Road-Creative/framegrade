"""Launch-time anchor gate and the schedule-store mirror.

``cmd_drive`` refuses a bad ``--lf-publish-date`` (or a saved one on resume) before any
phase, and a confirmed LF upload is also written to ``schedule.sqlite3`` when it exists.
"""

from __future__ import annotations

import ast
import json
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from freezegun import freeze_time

from gomoveshift.config import Config
from gomoveshift.pipeline import schedule_constants
from gomoveshift.pipeline.content_ledger import ContentEntry, ContentLedger

EST = schedule_constants.EST
PKG = Path(__file__).resolve().parents[1] / "gomoveshift"
NOW = datetime(2026, 10, 7, 12, 0, tzinfo=EST)  # the frozen "today" for the future-anchor tests
PAST_FRIDAY = date(2026, 9, 25)  # fixed, so it is in the past on any real clock


@pytest.fixture
def restore_schedule_module():
    saved = (
        schedule_constants.LONG_FORM_DAYS,
        schedule_constants.LONG_FORM_TIME,
        schedule_constants.SHORTS_TIMES,
    )
    yield
    (
        schedule_constants.LONG_FORM_DAYS,
        schedule_constants.LONG_FORM_TIME,
        schedule_constants.SHORTS_TIMES,
    ) = saved
    from gomoveshift.pipeline import scheduler as _sched

    _sched.LONG_FORM_DAYS, _sched.LONG_FORM_TIME, _sched.SHORTS_TIMES = saved


def _lf(date: str, region: str | None, project: str = "other") -> ContentEntry:
    return ContentEntry(
        date=date,
        time="16:00:00",
        content_type="longform",
        title="Booked",
        project_name=project,
        status="scheduled",
        source="pipeline",
        region=region,
    )


def _ledger(tmp_path: Path, *entries: ContentEntry) -> ContentLedger:
    ledger = ContentLedger(entries=list(entries))
    ledger.save(tmp_path / ".content_ledger.json")
    return ledger


# -- (b) launch-time anchor gate --------------------


def _launch(tmp_path, monkeypatch, *, persisted=None, **arg_overrides):
    """Run ``cmd_drive`` with a mocked orchestrator; return (rc, run_called, stderr)."""
    from test_cli_lf_day_propagation import _drive_args

    from gomoveshift.cli.handlers import drive as drive_handler

    monkeypatch.delenv("GMS_INTERMEDIATE_DIR", raising=False)
    stages = tmp_path / "out" / "src-run" / "_stages"
    if persisted is not None:
        stages.mkdir(parents=True, exist_ok=True)
        (stages / "pipeline_state.json").write_text(json.dumps(persisted))
    cfg = Config()
    cfg.seo.storage_path = str(tmp_path / "seo")
    Path(cfg.seo.storage_path).mkdir(exist_ok=True)
    opts = dict(
        directory=str(tmp_path / "src-run"),
        output_dir=str(tmp_path / "out"),
        publish=None,
        lf_day=None,
        lf_publish_date=None,
        region="usa",
    )
    opts.update(arg_overrides)
    args = _drive_args(**opts)
    with (
        patch.object(drive_handler, "is_in_tmux", return_value=True),
        patch.object(drive_handler, "configure_logging"),
        patch("gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator") as orch_cls,
    ):
        orch = orch_cls.return_value
        orch.run.return_value = SimpleNamespace(success=True, stages_completed=[], drive_plan=None)
        rc = drive_handler.cmd_drive(args, cfg)
    return rc, orch.run.called, cfg


def test_launch_refuses_persisted_tue_anchor_with_publish_fri(
    tmp_path, monkeypatch, restore_schedule_module, capsys
):
    rc, ran, _ = _launch(
        tmp_path,
        monkeypatch,
        persisted={"lf_publish_date": "2026-10-13", "publish_day": "tue", "region": "usa"},
        publish="fri",
    )
    assert rc == 2 and not ran
    err = capsys.readouterr().err
    assert "2026-10-13" in err and "weekday=1" in err and "--lf-publish-date" in err


def test_launch_refuses_cli_anchor_off_the_lf_day(
    tmp_path, monkeypatch, restore_schedule_module, capsys
):
    rc, ran, _ = _launch(tmp_path, monkeypatch, lf_day="fri", lf_publish_date="2026-10-13")
    assert rc == 2 and not ran
    assert "--lf-publish-date" in capsys.readouterr().err


@freeze_time(NOW)
def test_launch_refuses_anchor_booked_by_another_project_and_names_next_free(
    tmp_path, monkeypatch, restore_schedule_module, capsys
):
    future = (NOW + timedelta(days=30)).date()
    friday = future + timedelta(days=(4 - future.weekday()) % 7)
    (tmp_path / "seo").mkdir(exist_ok=True)
    _ledger(tmp_path / "seo", _lf(friday.isoformat(), "usa", project="rival"))
    rc, ran, _ = _launch(tmp_path, monkeypatch, lf_day="fri", lf_publish_date=friday.isoformat())
    assert rc == 2 and not ran
    err = capsys.readouterr().err
    assert "already booked" in err
    assert f"--lf-publish-date {(friday + timedelta(days=7)).isoformat()}" in err


@freeze_time(NOW)
def test_launch_accepts_anchor_this_project_already_holds(
    tmp_path, monkeypatch, restore_schedule_module
):
    future = (NOW + timedelta(days=30)).date()
    friday = future + timedelta(days=(4 - future.weekday()) % 7)
    (tmp_path / "seo").mkdir(exist_ok=True)
    _ledger(tmp_path / "seo", _lf(friday.isoformat(), "usa", project="src-run"))
    rc, ran, _ = _launch(tmp_path, monkeypatch, lf_day="fri", lf_publish_date=friday.isoformat())
    assert rc != 2 and ran


def test_launch_refuses_a_past_anchor(tmp_path, monkeypatch, restore_schedule_module, capsys):
    past = PAST_FRIDAY
    rc, ran, _ = _launch(tmp_path, monkeypatch, lf_day="fri", lf_publish_date=past.isoformat())
    assert rc == 2 and not ran
    assert "past" in capsys.readouterr().err


def _past_friday():
    return PAST_FRIDAY


def test_resume_of_an_uploaded_drive_past_its_anchor_is_not_refused(
    tmp_path, monkeypatch, restore_schedule_module
):
    """LF1 aired and LF2 needs a retry: the saved anchor is past, but moving it would
    move a drive whose LF1 is live. Only the weekday check applies (as in Phase D)."""
    past = _past_friday().isoformat()
    rc, ran, _ = _launch(
        tmp_path,
        monkeypatch,
        persisted={
            "lf_publish_date": past,
            "publish_day": "fri",
            "region": "usa",
            "youtube_uploads": {"1": {"video_id": "abc"}},
        },
        lf_day="fri",
    )
    assert rc != 2 and ran


def test_resume_without_uploads_still_refuses_a_past_saved_anchor(
    tmp_path, monkeypatch, restore_schedule_module, capsys
):
    rc, ran, _ = _launch(
        tmp_path,
        monkeypatch,
        persisted={
            "lf_publish_date": _past_friday().isoformat(),
            "publish_day": "fri",
            "region": "usa",
            "youtube_uploads": {},
        },
        lf_day="fri",
    )
    assert rc == 2 and not ran
    assert "past" in capsys.readouterr().err


def test_resume_of_an_uploaded_drive_still_refuses_an_off_day_saved_anchor(
    tmp_path, monkeypatch, restore_schedule_module, capsys
):
    rc, ran, _ = _launch(
        tmp_path,
        monkeypatch,
        persisted={
            "lf_publish_date": "2026-10-13",
            "publish_day": "tue",
            "region": "usa",
            "youtube_uploads": {"1": {"video_id": "abc"}},
        },
        publish="fri",
    )
    assert rc == 2 and not ran
    assert "weekday=1" in capsys.readouterr().err


def test_a_cli_anchor_in_the_past_is_refused_even_when_uploads_exist(
    tmp_path, monkeypatch, restore_schedule_module, capsys
):
    rc, ran, _ = _launch(
        tmp_path,
        monkeypatch,
        persisted={
            "publish_day": "fri",
            "region": "usa",
            "youtube_uploads": {"1": {"video_id": "a"}},
        },
        lf_day="fri",
        lf_publish_date=_past_friday().isoformat(),
    )
    assert rc == 2 and not ran


def test_a_refused_launch_creates_no_directories(tmp_path, monkeypatch, restore_schedule_module):
    from test_cli_lf_day_propagation import _drive_args

    from gomoveshift.cli.handlers import drive as drive_handler

    monkeypatch.delenv("GMS_INTERMEDIATE_DIR", raising=False)
    cfg = Config()
    cfg.seo.storage_path = str(tmp_path / "seo")
    Path(cfg.seo.storage_path).mkdir()
    out = tmp_path / "never-made"
    args = _drive_args(
        directory=str(tmp_path / "src-run"),
        output_dir=str(out),
        region="usa",
        lf_day="fri",
        lf_publish_date=_past_friday().isoformat(),
    )
    assert drive_handler._launch_anchor_refusal(args, cfg, "fri")
    assert not out.exists()


def test_todays_anchor_is_refused_once_its_slot_time_has_passed(
    monkeypatch, restore_schedule_module
):
    from gomoveshift.pipeline.seo_slots import anchor_refusal

    cfg = Config()
    cfg.drive.long_form_publish_days = [4]
    kw = dict(
        anchor="2031-01-03",  # a Friday
        publish_day="fri",
        region="usa",
        project_names=["p"],
    )
    late = datetime(2031, 1, 3, 23, 30, tzinfo=EST)
    early = datetime(2031, 1, 3, 0, 30, tzinfo=EST)
    assert "past" in anchor_refusal(cfg, ContentLedger(), now=late, **kw)
    assert anchor_refusal(cfg, ContentLedger(), now=early, **kw) is None


def test_launch_without_an_anchor_is_untouched(tmp_path, monkeypatch, restore_schedule_module):
    rc, ran, _ = _launch(tmp_path, monkeypatch, lf_day="fri")
    assert rc != 2 and ran


# -- (c) dual booking: a confirmed LF also lands in schedule.sqlite3 ----------


def _book(tmp_path, *, db_exists: bool):
    from gomoveshift.pipeline.drive_phases import external_systems_coordinator as esc

    seo = tmp_path / "seo"
    seo.mkdir(exist_ok=True)
    cfg = Config()
    cfg.seo.storage_path = str(seo)
    db = seo / "schedule.sqlite3"
    if db_exists:
        from seo_manager.scheduling.store import ScheduleStore

        ScheduleStore(db, EST, ["usa"]).init()
    record = {
        "video_id": "vid123",
        "publish_at": datetime(2031, 3, 4, 16, 0, tzinfo=EST).isoformat(),
        "title": "T",
    }
    esc._book_lf_upload_in_ledger(
        record=record,
        state=SimpleNamespace(region="usa"),
        config=cfg,
        project_base="proj-a",
        lf_index=2,
    )
    return db


def test_confirmed_lf_is_also_booked_in_the_schedule_store(tmp_path):
    db = _book(tmp_path, db_exists=True)
    rows = (
        sqlite3.connect(db)
        .execute("SELECT project, lf_index, region, video_id FROM lf_bookings")
        .fetchall()
    )
    assert rows == [("proj-a", 2, "usa", "vid123")]
    assert (tmp_path / "seo" / ".content_ledger.json").exists()  # the ledger stays


def test_dual_booking_never_creates_the_schedule_store(tmp_path):
    db = _book(tmp_path, db_exists=False)
    assert not db.exists()
    assert (tmp_path / "seo" / ".content_ledger.json").exists()


def test_a_store_failure_never_fails_the_ledger_booking(tmp_path, caplog):
    seo = tmp_path / "seo"
    seo.mkdir()
    (seo / "schedule.sqlite3").write_text("not a database")
    from gomoveshift.pipeline.drive_phases import external_systems_coordinator as esc

    cfg = Config()
    cfg.seo.storage_path = str(seo)
    esc._book_lf_upload_in_ledger(
        record={"video_id": "v", "publish_at": datetime(2031, 3, 4, 16, 0, tzinfo=EST).isoformat()},
        state=SimpleNamespace(region="usa"),
        config=cfg,
        project_base="p",
        lf_index=1,
    )
    assert "SCHEDULE-STORE-UNBOOKED" in caplog.text
    assert ContentLedger.load(seo / ".content_ledger.json").entries


# -- review fixes: unpadded date, hint over the real allowed set, mirror on ledger failure --


def test_unpadded_anchor_is_a_clean_refusal_not_a_traceback(restore_schedule_module):
    from gomoveshift.pipeline.seo_slots import anchor_refusal

    msg = anchor_refusal(
        Config(),
        ContentLedger(),
        anchor="2031-1-1",  # strptime accepts it; fromisoformat does not
        publish_day="wed",
        region="usa",
        project_names=["p"],
    )
    assert msg is not None and "not a YYYY-MM-DD date" in msg


def test_hint_never_names_a_date_the_gate_would_refuse(restore_schedule_module):
    """Default config LF day is Wed; ``--publish fri`` leaves no allowed day, so the
    refusal must say so instead of suggesting a Friday the same gate rejects."""
    from gomoveshift.pipeline.seo_slots import anchor_refusal

    msg = anchor_refusal(
        Config(),
        ContentLedger(),
        anchor="2031-01-03",  # a Friday
        publish_day="fri",
        region="usa",
        project_names=["p"],
    )
    assert msg is not None
    assert "next free slot: --lf-publish-date" not in msg
    assert "no LF day" in msg


def test_hint_suggests_a_date_on_the_allowed_day(restore_schedule_module):
    from gomoveshift.pipeline.seo_slots import anchor_refusal

    config = Config()
    config.drive.long_form_publish_days = [1]  # the run's pick: Tuesday, a Waypost usa day
    msg = anchor_refusal(
        config,
        ContentLedger(),
        anchor="2031-01-03",  # a Friday: not the run's day
        publish_day=None,
        region="usa",
        project_names=["p"],
        now=datetime(2030, 12, 1, tzinfo=EST),
    )
    assert msg is not None and "next free slot: --lf-publish-date 2030-12-03" in msg


def test_ledger_failure_still_mirrors_the_booking_into_the_store(tmp_path, caplog):
    from seo_manager.scheduling.store import ScheduleStore

    from gomoveshift.pipeline.drive_phases import external_systems_coordinator as esc

    seo = tmp_path / "seo"
    seo.mkdir()
    ScheduleStore(seo / "schedule.sqlite3", EST, ["usa"]).init()
    cfg = Config()
    cfg.seo.storage_path = str(seo)
    with patch("gomoveshift.pipeline.content_ledger.record_confirmed_upload", return_value=False):
        esc._book_lf_upload_in_ledger(
            record={
                "video_id": "v9",
                "publish_at": datetime(2031, 3, 4, 16, 0, tzinfo=EST).isoformat(),
            },
            state=SimpleNamespace(region="usa"),
            config=cfg,
            project_base="p",
            lf_index=1,
        )
    assert "LEDGER-UNBOOKED" in caplog.text
    rows = sqlite3.connect(seo / "schedule.sqlite3").execute("SELECT video_id FROM lf_bookings")
    assert rows.fetchall() == [("v9",)]


def _audit_only(**extra):
    return dict(skip_upload=True, no_youtube_upload=True, **extra)


def test_audit_only_run_is_not_refused_for_a_past_saved_anchor(
    tmp_path, monkeypatch, restore_schedule_module
):
    """The nightly argv (--skip-upload --no-youtube-upload, no --fresh) books nothing, so a
    saved anchor already in the past must not fail the backlog row every night."""
    rc, ran, _ = _launch(
        tmp_path,
        monkeypatch,
        persisted={
            "lf_publish_date": _past_friday().isoformat(),
            "publish_day": "fri",
            "region": "usa",
            "youtube_uploads": {},
        },
        lf_day="fri",
        **_audit_only(),
    )
    assert rc != 2 and ran


def test_audit_only_run_still_refuses_an_off_day_anchor(
    tmp_path, monkeypatch, restore_schedule_module, capsys
):
    off_day = _past_friday() + timedelta(days=1)  # a Saturday
    rc, ran, _ = _launch(
        tmp_path, monkeypatch, lf_day="fri", lf_publish_date=off_day.isoformat(), **_audit_only()
    )
    assert rc == 2 and not ran
    assert capsys.readouterr().err


# -- review fixes: --start-date, rival store row, structured anchor detail ---


def test_launch_refuses_a_start_date_off_the_lf_day(
    tmp_path, monkeypatch, restore_schedule_module, capsys
):
    rc, ran, _ = _launch(tmp_path, monkeypatch, publish="fri", start_date="2026-10-13")
    assert rc == 2 and not ran
    assert "2026-10-13" in capsys.readouterr().err


def test_a_rivals_store_row_is_not_hidden_by_the_runs_own_ledger_row(
    tmp_path, restore_schedule_module
):
    """Own ledger row and a rival's store row on one date+region: the rival still blocks."""
    from gomoveshift.pipeline import seo_slots
    from seo_manager.scheduling.store import ScheduleStore

    cfg = Config()
    cfg.drive.long_form_publish_days = [1, 4, 6]
    schedule_constants.apply_config(cfg)
    ledger = _ledger(tmp_path, _lf("2026-10-13", "usa", project="proj-a"))
    store = ScheduleStore(tmp_path / "schedule.sqlite3", EST, ["usa", "ireland"])
    store.init()
    store.book("proj-b", 1, "usa", datetime(2026, 10, 13, 16, 0, tzinfo=EST))
    refusal = seo_slots.anchor_refusal(
        cfg,
        ledger,
        anchor="2026-10-13",
        publish_day="tue",
        region="usa",
        project_names=["proj-a"],
        now=datetime(2026, 10, 8, 9, 0, tzinfo=EST),
    )
    assert refusal and "2026-10-13" in refusal


def test_the_same_lf_in_ledger_and_store_is_counted_once(tmp_path):
    from gomoveshift.pipeline import seo_slots
    from seo_manager.scheduling.store import ScheduleStore

    ledger = _ledger(tmp_path, _lf("2026-10-13", "usa", project="proj-a"))
    store = ScheduleStore(tmp_path / "schedule.sqlite3", EST, ["usa"])
    store.init()
    store.book("proj-a", 1, "usa", datetime(2026, 10, 13, 16, 0, tzinfo=EST))
    assert len(seo_slots.Occupancy(ledger).bookings()) == 1


def test_lf_anchor_problem_does_not_parse_seo_managers_message(monkeypatch):
    """A reworded seo_manager message must not break Rule 2b: the weekday comes structured."""
    from gomoveshift.pipeline.state_invariants import lf_anchor_problem
    from seo_manager.scheduling import slots

    real = slots.validate_anchor
    monkeypatch.setattr(
        slots,
        "validate_anchor",
        lambda *a, **k: [m.replace("weekday=", "wd:") for m in real(*a, **k)],
    )
    msg = lf_anchor_problem("2026-10-13", "fri", frozenset({4}))
    assert msg and "weekday=1" in msg


# -- review fix: the gate checks the date Phase D publishes on --------------------------


def test_a_resumed_saved_anchor_beats_start_date_like_phase_d(
    tmp_path, monkeypatch, restore_schedule_module, capsys
):
    """Phase D's backstop keeps the saved lf_publish_date over --start-date, so a saved
    Tuesday anchor on a Friday run is refused even when --start-date is a good Friday."""
    rc, ran, _ = _launch(
        tmp_path,
        monkeypatch,
        persisted={"lf_publish_date": "2026-10-13", "publish_day": "tue", "region": "usa"},
        publish="fri",
        start_date="2026-11-13",
    )
    assert rc == 2 and not ran
    assert "2026-10-13" in capsys.readouterr().err


def test_a_start_date_refusal_names_start_date_not_lf_publish_date(
    tmp_path, monkeypatch, restore_schedule_module, capsys
):
    rc, ran, _ = _launch(tmp_path, monkeypatch, publish="fri", start_date="2026-10-13")
    assert rc == 2 and not ran
    err = capsys.readouterr().err
    assert "--start-date" in err and "--lf-publish-date" not in err
