"""The long-form weekdays come from Waypost's cadence.

``seo_slots.cadence_for`` is ``seo_manager``'s effective cadence: the packaged brand seed
overlaid with the ``lf_days`` decision rows under ``config.seo.storage_path``. The launch
gate, the day gate and ``gomoveshift schedule`` all follow it, and an unreadable source
halts planning. Every test builds its own tmp store; none reads a real one.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from seo_manager.analytics import cadence_decisions as cd
from seo_manager.core.profile import load_cadence

from gomoveshift.config import Config
from gomoveshift.pipeline import schedule_constants, seo_slots
from gomoveshift.pipeline.content_ledger import ContentLedger, next_free_superlong_slot
from gomoveshift.pipeline.schedule_report import build_schedule_report

from test_launch_anchor_gate import restore_schedule_module  # noqa: F401  (fixture)

EST = schedule_constants.EST
pytestmark = pytest.mark.usefixtures("restore_schedule_module")
BRAND = Path(__file__).resolve().parents[1] / "gomoveshift/console/brands/gomoveshift/brand.yaml"
DB_NAME = "cadence_decisions.sqlite3"
NOW = datetime(2026, 10, 9, 12, 0)
TUE, WED, THU, FRI, SAT, SUN = 1, 2, 3, 4, 5, 6


def _config(tmp_path: Path, lf_days=(FRI,)) -> Config:
    cfg = Config()
    cfg.seo.storage_path = str(tmp_path / "seo")
    Path(cfg.seo.storage_path).mkdir(exist_ok=True)
    cfg.drive.long_form_publish_days = list(lf_days)
    return cfg


def _decide(config: Config, region: str, dow) -> int:
    """An ``lf_days`` decision row through the public append API."""
    return cd.append(
        Path(config.seo.storage_path) / DB_NAME,
        knob="lf_days",
        region=region,
        value=len(dow),
        prev=2,
        status="ok",
        source="policy",
        reason_code="TAIL_UP",
        basis="test",
        applied_at=NOW,
        evidence={"dow": sorted(dow)},
    )


ANCHOR_FRI = "2026-11-20"  # a Friday, six weeks after NOW (a fixed clock, never the host's)


def _gate(config, anchor, *, resumed=False, region="usa", publish_day="fri"):
    return seo_slots.anchor_refusal(
        config,
        ContentLedger(),
        anchor=anchor,
        publish_day=publish_day,
        region=region,
        project_names=["p"],
        now=NOW.replace(tzinfo=EST),
        resumed=resumed,
    )


def test_cadence_for_returns_seo_effective_cadence(tmp_path):
    config = _config(tmp_path)
    seed = seo_slots.cadence_for(config, "usa")
    assert seed.region("usa").dow == (TUE, FRI)
    assert seed.region("ireland").dow == (SUN,)
    row = _decide(config, "usa", (TUE, THU))
    moved = seo_slots.cadence_for(config, "usa")
    assert moved.region("usa").dow == (TUE, THU)
    assert moved.dow_source["usa"].row_id == row
    expected = load_cadence(BRAND, decisions_db=Path(config.seo.storage_path) / DB_NAME)
    assert dict(moved.regions) == dict(expected.regions)
    assert moved.region("usa").time == expected.region("usa").time


def test_cadence_for_refuses_an_unknown_region(tmp_path):
    with pytest.raises(ValueError, match="unknown region"):
        seo_slots.cadence_for(_config(tmp_path), "mars")


def test_changed_lf_days_decision_moves_launch_anchor_gate(tmp_path):
    config = _config(tmp_path)
    anchor = ANCHOR_FRI
    assert _gate(config, anchor) is None
    _decide(config, "usa", (TUE, THU))
    refusal = _gate(config, anchor)
    assert refusal and "weekday=4" in refusal and "tue" in refusal.lower()


def test_resumed_live_drive_on_the_old_day_is_not_refused_after_a_decision(tmp_path):
    config = _config(tmp_path)
    anchor = ANCHOR_FRI
    _decide(config, "usa", (TUE, THU))
    assert _gate(config, anchor, resumed=True) is None


def test_unreadable_store_halts_planning(tmp_path, caplog):
    config = _config(tmp_path)
    (Path(config.seo.storage_path) / DB_NAME).write_bytes(b"this is not a sqlite file" * 40)
    with pytest.raises(seo_slots.CadenceSourceUnreadableError) as exc:
        seo_slots.waypost_cadence(config)
    text = str(exc.value)
    assert text.startswith("CADENCE_SOURCE_UNREADABLE reason=")
    assert text.endswith("(planning halted)")
    assert sum("CADENCE_SOURCE_UNREADABLE" in r.getMessage() for r in caplog.records) == 1
    for call in (
        lambda: seo_slots.next_free_slot(ContentLedger(), config, "usa"),
        lambda: _gate(config, ANCHOR_FRI),
        lambda: build_schedule_report(ContentLedger(), config),
    ):
        with pytest.raises(seo_slots.CadenceSourceUnreadableError):
            call()


def test_schedule_report_shows_a_decided_day_and_drops_the_replaced_one(tmp_path):
    config = _config(tmp_path)
    ledger = ContentLedger()
    after = datetime(2026, 10, 12, 9, 0, tzinfo=EST)

    def usa_days():
        report = build_schedule_report(ledger, config, after=after)
        return report, {r["day"] for r in report["next_available"] if r["region"] == "usa"}

    report, days = usa_days()
    assert days == {"tue", "fri"} and report["designed_days"] == ["tue", "fri", "sun"]
    _decide(config, "usa", (TUE, THU))
    report, days = usa_days()
    assert days == {"tue", "thu"}
    assert report["designed_days"] == ["tue", "thu", "sun"]
    assert report["weekly_longform_capacity"] == 3
    ireland = [r["day"] for r in report["next_available"] if r["region"] == "ireland"]
    assert ireland == ["sun"]
    thu = next(r for r in report["next_available"] if r["day"] == "thu")
    assert thu["slot"].endswith("16:00:00-04:00") and thu["shorts_required"] >= 0


def test_lf_day_refusal_names_waypost_days_and_points_at_schedule(tmp_path):
    config = _config(tmp_path)
    assert seo_slots.lf_day_refusal(config, "usa", "fri") is None
    assert seo_slots.lf_day_refusal(config, "usa", "tue") is None
    message = seo_slots.lf_day_refusal(config, "usa", "wed")
    assert message and "wed" in message and "tue" in message and "fri" in message
    assert "gomoveshift schedule" in message
    assert seo_slots.lf_day_refusal(config, None, "sun") is None
    _decide(config, "usa", (TUE, THU))
    assert seo_slots.lf_day_refusal(config, "usa", "fri")


def test_a_saturday_superlong_slot_search_ignores_the_long_form_days(tmp_path):
    config = _config(tmp_path)
    after = datetime(2026, 10, 12, 9, 0, tzinfo=EST)
    slot = next_free_superlong_slot(ContentLedger(), config, "usa", after=after)
    assert slot.weekday() == SAT


# -- the launch path (cmd_drive) ------------------------------------------------


def _launch(tmp_path, monkeypatch, store=None, **overrides):
    from test_launch_anchor_gate import _launch as launch

    if store is not None:
        (tmp_path / "seo").mkdir(exist_ok=True)
        (tmp_path / "seo" / DB_NAME).write_bytes(store)
    return launch(tmp_path, monkeypatch, **overrides)


def test_launch_without_an_anchor_refuses_a_day_off_waypost_days(tmp_path, monkeypatch, capsys):
    rc, ran, _ = _launch(tmp_path, monkeypatch, lf_day="wed")
    assert rc == 2 and not ran
    err = capsys.readouterr().err
    assert "Waypost's days" in err and "gomoveshift schedule" in err


def test_launch_without_an_anchor_accepts_a_waypost_day(tmp_path, monkeypatch):
    rc, ran, _ = _launch(tmp_path, monkeypatch, lf_day="fri")
    assert rc != 2 and ran


def test_launch_with_an_unreadable_source_plans_nothing(tmp_path, monkeypatch, capsys):
    rc, ran, _ = _launch(tmp_path, monkeypatch, store=b"not sqlite" * 50, lf_day="fri")
    assert rc == 2 and not ran
    assert "CADENCE_SOURCE_UNREADABLE reason=" in capsys.readouterr().err
