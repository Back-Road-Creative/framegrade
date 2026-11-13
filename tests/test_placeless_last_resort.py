"""A placeless label is the LAST resort: never over footage that has GPS, and never launched
unattended where a bracket (the drives before and after, ``bracket_gpx``) could be given."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from _state_helpers import make_state

from gomoveshift.health import check as check_mod
from gomoveshift.health import launch

LABEL = "Somewhere On The Road"
DAY = "2025-09-16"


def _enrich(tmp_path, monkeypatch, gps_by_lf):
    from gomoveshift.pipeline import gps_enrichment as ge

    lfs = [SimpleNamespace(index=i, segments=[]) for i in (1, 2, 3)]
    monkeypatch.setattr(ge, "_partition_lf_plans_by_date", lambda *a, **k: {DAY: lfs})
    monkeypatch.setattr(ge, "_build_shared_geocoder", lambda **k: None)
    monkeypatch.setattr(ge, "_build_gps_points_by_lf", lambda *a, **k: gps_by_lf)
    monkeypatch.setattr(ge, "_assert_per_lf_locations_resolvable", lambda *a, **k: None)
    seen = {}

    def fake_derive(day_lfs, **kw):
        seen["overrides"] = dict(kw["overrides"])
        return {
            int(lf.index): kw["overrides"].get(int(lf.index), "Measured County, NC")
            for lf in day_lfs
        }

    monkeypatch.setattr(ge, "derive_per_lf_locations", fake_derive)
    state = make_state(placeless_label=LABEL, episode_dates=[DAY], output_base=str(tmp_path))
    out = ge._enrich_per_lf_gps(
        state=state,
        real_plan=SimpleNamespace(regular_lfs=lfs),
        day_batches=[],
        episode_files=[],
        per_day_locations={},
        drive_wide_location="",
    )
    return out, state, seen


def test_placeless_label_is_withheld_from_an_lf_with_gps(tmp_path, monkeypatch, caplog):
    with caplog.at_level(logging.WARNING):
        out, state, seen = _enrich(tmp_path, monkeypatch, {2: [object()]})
    assert out.per_lf_overrides == {1: LABEL, 3: LABEL}
    assert state.per_lf_overrides == {1: LABEL, 3: LABEL}
    assert seen["overrides"] == {1: LABEL, 3: LABEL}
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len([w for w in warnings if "placeless" in w]) == 1
    assert "[2]" in next(w for w in warnings if "placeless" in w)


def test_placeless_label_still_labels_every_gps_less_lf(tmp_path, monkeypatch, caplog):
    with caplog.at_level(logging.WARNING):
        out, _, _ = _enrich(tmp_path, monkeypatch, {})
    assert out.per_lf_overrides == {1: LABEL, 2: LABEL, 3: LABEL}
    assert not [r for r in caplog.records if "placeless" in r.getMessage()]


# -- launcher ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label, detail",
    [
        (LABEL, None),
        ("Somewhere Between", None),
        ("Somewhere In The Hills", "claims Hills"),
        ("Over The Ridges", "outside the placeless vocabulary"),
    ],
)
def test_every_placeless_label_is_refused_unattended(label, detail):
    reason = launch.placeless_label_skip_reason(label)
    assert reason and "bracket_gpx" in reason, reason
    assert "last resort" in reason and "--placeless-label" in reason
    if detail:
        assert detail in reason


def test_launcher_skips_a_claim_free_placeless_row():
    row = SimpleNamespace(placeless_label=LABEL)
    reason = launch.placeless_label_skip_reason(row.placeless_label)
    assert reason and reason.startswith(f"placeless_label {LABEL!r}")


# -- health check rule 5 -----------------------------------------------------------------


def test_rule_5_reports_a_claim_free_placeless_row_pointing_at_bracket_gpx():
    rows = check_mod.placeless_label_incidents(
        [
            SimpleNamespace(project="a", placeless_label=LABEL),
            SimpleNamespace(project="b", placeless_label=""),
        ],
        "2026-11-13T00:00:00+00:00",
    )
    assert [r["project"] for r in rows] == ["a"]
    line = check_mod._line(rows[0])
    assert "bracket_gpx" in line and "a" in line and LABEL in line
