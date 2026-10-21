"""B1: the console schedule read model keeps four volume figures apart.

selected target (Waypost's shorts-per-LF decision), slot capacity (shorts windows x LF interval),
constrained feasible output (derived min of the two) and what is actually scheduled (ledger rows)
are different quantities. Each is shown with its source, fallback reason and policy version, and
none is computed by a new cadence engine. Times render in America/New_York.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from seo_manager.analytics import cadence_decisions

from gomoveshift.console import schedule as console_schedule
from gomoveshift.console import schedule_model
from gomoveshift.console.app import create_app

REPORT = {
    "as_of": "2026-11-01T05:30:00+00:00",  # 01:30 EDT, before the fall-back hour repeats
    "weekly_longform_capacity": 3,
    "designed_days": ["tue", "fri", "sun"],
    "scheduled_longforms": [],
    "scheduled_shorts_count": 4,
    "scheduled_shorts_by_region": {"usa": 3, "unassigned": 1},
    "queue_ends": None,
    "next_available": [
        {
            "region": "usa",
            "day": "tue",
            "slot": "2026-11-03T16:00:00-05:00",
            "shorts_windows": ["10:00", "14:00"],
            "shorts_required": 5,
            "shorts_slots": 14,
            "earliest_for_region": True,
        }
    ],
    "next_available_superlong": [],
}


def _decision(knob, value, *, source="policy", rev=3, reason="steady", region=None):
    return cadence_decisions.StoredDecision(
        id=7,
        knob=knob,
        region=region,
        value=value,
        prev=None,
        reason_code=reason,
        basis="test",
        status="applied",
        source=source,
        policy_rev=rev,
        applied_at="2026-10-01T12:00:00+00:00",
    )


@pytest.fixture
def config(tmp_path):
    return SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path)))


def _stub(monkeypatch, selected, times=None):
    def current(db, knob, region=None, seed=None):
        if knob == "shorts_per_lf":
            if isinstance(selected, Exception):
                raise selected
            return selected
        return times or _decision("shorts_times", ["10:00", "14:00"], rev=5, region=region)

    monkeypatch.setattr(schedule_model.cadence_decisions, "current", current)


def test_four_figures_are_separate_and_each_carries_provenance(monkeypatch, config):
    _stub(monkeypatch, _decision("shorts_per_lf", 5, reason="retention-up"))
    model = schedule_model.volume_model(REPORT, config)
    assert model["selected"]["value"] == 5
    assert model["capacity"]["by_region"]["usa"]["value"] == 14
    assert model["feasible"]["by_region"]["usa"]["value"] == 5
    assert model["scheduled"]["total"] == 4
    assert model["scheduled"]["by_region"] == {"usa": 3, "unassigned": 1}
    sel = model["selected"]
    assert (sel["source"], sel["reason_code"], sel["policy_rev"]) == ("policy", "retention-up", 3)
    assert sel["fallback_reason"] is None
    cap = model["capacity"]["by_region"]["usa"]
    assert cap["windows"] == ["10:00", "14:00"] and cap["policy_rev"] == 5
    assert cap["fallback_reason"] is None


def test_feasible_is_the_lower_of_target_and_capacity_and_says_which_binds(monkeypatch, config):
    _stub(monkeypatch, _decision("shorts_per_lf", 20))
    feasible = schedule_model.volume_model(REPORT, config)["feasible"]["by_region"]["usa"]
    assert feasible["value"] == 14 and feasible["binding"] == "capacity"
    _stub(monkeypatch, _decision("shorts_per_lf", 5))
    feasible = schedule_model.volume_model(REPORT, config)["feasible"]["by_region"]["usa"]
    assert feasible["value"] == 5 and feasible["binding"] == "selected"


def test_a_seed_value_is_labelled_as_a_fallback(monkeypatch, config):
    _stub(
        monkeypatch,
        _decision("shorts_per_lf", 7, source="seed", rev=0, reason="seed"),
        times=_decision("shorts_times", ["10:00", "14:00"], source="seed", rev=0),
    )
    model = schedule_model.volume_model(REPORT, config)
    assert "seed" in model["selected"]["fallback_reason"]
    assert "seed" in model["capacity"]["by_region"]["usa"]["fallback_reason"]


def test_an_unreadable_store_blanks_the_target_but_keeps_the_other_figures(monkeypatch, config):
    _stub(monkeypatch, cadence_decisions.DecisionStoreUnreadable("db locked"))
    model = schedule_model.volume_model(REPORT, config)
    assert model["selected"]["value"] is None
    assert "unreadable" in model["selected"]["fallback_reason"].lower()
    assert model["feasible"]["by_region"]["usa"]["value"] is None
    assert model["scheduled"]["total"] == 4


def test_scheduled_comes_from_the_ledger_report_and_is_not_policy_versioned(monkeypatch, config):
    _stub(monkeypatch, _decision("shorts_per_lf", 5))
    scheduled = schedule_model.volume_model(REPORT, config)["scheduled"]
    assert scheduled["policy_rev"] is None
    assert "ledger" in scheduled["source"]
    assert scheduled["fallback_reason"]


def test_building_the_model_writes_nothing(monkeypatch, config, tmp_path):
    _stub(monkeypatch, _decision("shorts_per_lf", 5))
    schedule_model.volume_model(REPORT, config)
    assert list(tmp_path.iterdir()) == []


def test_times_render_in_new_york_across_the_dst_boundaries():
    eastern = schedule_model.eastern
    assert eastern("2026-03-08T06:59:00+00:00") == "2026-03-08 01:59 EST"
    assert eastern("2026-03-08T07:01:00+00:00") == "2026-03-08 03:01 EDT"
    assert eastern("2026-11-01T05:30:00+00:00") == "2026-11-01 01:30 EDT"
    assert eastern("2026-11-01T06:30:00+00:00") == "2026-11-01 01:30 EST"
    assert eastern("2026-11-03T16:00:00-05:00") == "2026-11-03 16:00 EST"
    assert eastern(None) == "-"
    assert eastern("not a time") == "not a time"


def test_the_schedule_tab_shows_all_four_with_their_versions(monkeypatch, config):
    _stub(monkeypatch, _decision("shorts_per_lf", 5, rev=3))
    monkeypatch.setattr(
        console_schedule, "build_schedule_report", lambda ledger, cfg, **_kw: REPORT
    )
    monkeypatch.setattr(schedule_model, "_config", lambda: config)
    html = TestClient(create_app()).get("/schedule").text
    for label in ("Selected target", "Slot capacity", "Feasible output", "Actually scheduled"):
        assert label in html
    assert "policy rev 3" in html and "policy rev 5" in html
    assert "2026-11-01 01:30 EDT" in html


@pytest.mark.parametrize("source", ["", None, "weird"])
def test_an_unknown_or_missing_source_is_labelled_not_shown_as_plain_provenance(
    monkeypatch, config, source
):
    _stub(monkeypatch, _decision("shorts_per_lf", 5, source=source))
    selected = schedule_model.volume_model(REPORT, config)["selected"]
    assert selected["fallback_reason"].startswith("unknown decision source")


@pytest.mark.parametrize("source", ["policy", "rollback", "blind_revert"])
def test_a_known_non_seed_source_has_no_fallback_reason(monkeypatch, config, source):
    _stub(monkeypatch, _decision("shorts_per_lf", 5, source=source))
    assert schedule_model.volume_model(REPORT, config)["selected"]["fallback_reason"] is None


def test_a_failing_model_shows_a_visible_unavailable_row_not_a_blank(monkeypatch, config):
    monkeypatch.setattr(
        console_schedule, "build_schedule_report", lambda ledger, cfg, **_kw: REPORT
    )
    monkeypatch.setattr(schedule_model, "_config", lambda: config)

    def boom(report, cfg):
        raise RuntimeError("store exploded")

    monkeypatch.setattr(schedule_model, "volume_model", boom)
    html = TestClient(create_app()).get("/schedule").text
    assert "Shorts volume" in html
    assert "volume model unavailable: store exploded" in html


def test_the_superlong_slot_is_shown_in_eastern_not_raw_iso(monkeypatch, config):
    _stub(monkeypatch, _decision("shorts_per_lf", 5, rev=3))
    report = {
        **REPORT,
        "next_available_superlong": [
            {"region": "usa", "day": "sat", "slot": "2026-08-15T15:00:00-04:00"}
        ],
    }
    monkeypatch.setattr(
        console_schedule, "build_schedule_report", lambda ledger, cfg, **_kw: report
    )
    monkeypatch.setattr(schedule_model, "_config", lambda: config)
    html = TestClient(create_app()).get("/schedule").text
    assert "2026-08-15 15:00 EDT" in html
    assert "2026-08-15T15:00:00-04:00" not in html
