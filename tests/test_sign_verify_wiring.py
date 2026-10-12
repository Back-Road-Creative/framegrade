"""Wiring for road-sign GPS verification: the ``--label-dry-run`` advisory
print and the opt-in strict pre-encode gate.

Both consumers are tested against a STUB ``gomoveshift.analysis.sign_verify``
module injected into ``sys.modules`` — the comparator's own semantics belong to
its own suite, and the wiring must be provable without the network, without
video decode, and without the comparator's internals.
"""

import logging
import sys
import types
from types import SimpleNamespace

import pytest

import gomoveshift.pipeline.drive_phases.phase_bc_day_first as phase_bc
import gomoveshift.pipeline.gps_enrichment as gps_enrichment
from gomoveshift.analysis.sign_evidence import (
    STATUS_CONFLICTS,
    STATUS_CORROBORATES,
    STATUS_NO_SIGNAL,
    SignHit,
    SignVerdict,
)
from gomoveshift.pipeline.drive_phases import lf_listing

_SIGN_MODULE = "gomoveshift.analysis.sign_verify"


class _Geo:
    """Single-county geocoder, so ``dominant_location`` resolves a dom."""

    def reverse_geocode_detailed(self, lat, lon):
        return {"county": "Beaufort County", "state": "South Carolina"}


def _hit(text="COLLETON COUNTY", date="2025-09-01"):
    return SignHit(
        episode_index=0,
        t_local=12.0,
        ts_day=12.0,
        date=date,
        raw_text=text,
        ocr_conf=0.9,
    )


def _install_fake_sign_verify(
    monkeypatch,
    *,
    bucket=None,
    verdict=None,
    verdicts=None,
    verify_lf_raises=False,
    bucket_raises=False,
):
    """Inject a stub ``sign_verify`` module and return its call log."""
    calls = {"bucket": [], "verify_lf": [], "verify_all": []}
    mod = types.ModuleType(_SIGN_MODULE)

    def bucket_hits_by_lf(hits_by_date, drive_plan):
        calls["bucket"].append({"hits_by_date": hits_by_date, "drive_plan": drive_plan})
        if bucket_raises:
            raise ValueError("bucketing exploded")
        return dict(bucket or {})

    def resolved_from_dominant(dom):
        return SimpleNamespace(
            dominant_county=getattr(dom, "county", None),
            dominant_state=getattr(dom, "state", None),
            regions=["Beaufort County"],
        )

    def verify_lf(lf_index, hits, resolved, *, min_corroborating_hits=2, geocode_fn=None):
        calls["verify_lf"].append(
            {
                "lf_index": lf_index,
                "hits": hits,
                "resolved": resolved,
                "min_corroborating_hits": min_corroborating_hits,
                "geocode_fn": geocode_fn,
            }
        )
        if verify_lf_raises:
            raise ValueError("comparator exploded")
        return verdict

    def verify_all(
        hits_by_date, drive_plan, resolved_by_lf, *, min_corroborating_hits=2, geocode_fn=None
    ):
        calls["verify_all"].append(
            {
                "hits_by_date": hits_by_date,
                "drive_plan": drive_plan,
                "resolved_by_lf": resolved_by_lf,
                "min_corroborating_hits": min_corroborating_hits,
                "geocode_fn": geocode_fn,
            }
        )
        return dict(verdicts or {})

    mod.bucket_hits_by_lf = bucket_hits_by_lf
    mod.resolved_from_dominant = resolved_from_dominant
    mod.verify_lf = verify_lf
    mod.verify_all = verify_all
    monkeypatch.setitem(sys.modules, _SIGN_MODULE, mod)
    return calls


def _wire_dry_run(monkeypatch, lf_index=1):
    lf = SimpleNamespace(index=lf_index, segments=[])
    drive_plan = SimpleNamespace(regular_lfs=[lf])
    pts = [{"lat": 32.3095 + i * 0.001, "lon": -80.6332} for i in range(6)]
    monkeypatch.setattr(phase_bc, "_build_drive_plan_from_state", lambda c, s: drive_plan)
    monkeypatch.setattr(gps_enrichment, "_build_gps_points_by_lf", lambda p, g: {lf_index: pts})
    monkeypatch.setattr(gps_enrichment, "_build_shared_geocoder", lambda *a, **k: _Geo())
    return drive_plan


def _config(*, sign_verify=False, strict=False, min_hits=2):
    return SimpleNamespace(
        drive=SimpleNamespace(
            sign_verify=sign_verify,
            sign_verify_strict=strict,
            sign_verify_min_corroborating_hits=min_hits,
        )
    )


def _state(hits_by_date=None, resolved=None):
    return SimpleNamespace(
        gps_per_episode=[],
        sign_hits_by_date=hits_by_date,
        per_lf_resolved_location=resolved,
    )


# ── --label-dry-run advisory print ────────────────────────────────────


def test_dry_run_off_prints_no_sign_lines_and_never_calls_comparator(monkeypatch, capsys):
    _wire_dry_run(monkeypatch)
    calls = _install_fake_sign_verify(monkeypatch, bucket={1: [_hit()]})

    lf_listing.print_label_dry_run(
        config=_config(sign_verify=False),
        state=_state({"2025-09-01": [_hit().to_dict()]}),
    )

    out = capsys.readouterr().out
    assert "sign evidence" not in out
    assert calls["bucket"] == []
    assert calls["verify_lf"] == []


def test_dry_run_prints_corroborating_evidence(monkeypatch, capsys):
    _wire_dry_run(monkeypatch)
    verdict = SignVerdict(
        lf_index=1,
        status=STATUS_CORROBORATES,
        hit_count=2,
        reasons=['sign "BEAUFORT COUNTY" names Beaufort — in sampled regions'],
        corroborating_counties=["Beaufort"],
    )
    calls = _install_fake_sign_verify(monkeypatch, bucket={1: [_hit(), _hit()]}, verdict=verdict)

    lf_listing.print_label_dry_run(
        config=_config(sign_verify=True, min_hits=3),
        state=_state({"2025-09-01": [_hit().to_dict()]}),
    )

    out = capsys.readouterr().out
    assert "sign evidence: corroborates" in out
    assert "2 hit" in out
    assert 'sign "BEAUFORT COUNTY" names Beaufort' in out
    # The operator's configured threshold and the repo's own forward geocoder
    # reach the comparator.
    assert calls["verify_lf"][0]["min_corroborating_hits"] == 3
    assert callable(calls["verify_lf"][0]["geocode_fn"])
    # Hits are bucketed from the typed state view, not raw dicts.
    assert isinstance(calls["bucket"][0]["hits_by_date"]["2025-09-01"][0], SignHit)


def test_dry_run_conflict_prints_warns_and_returns(monkeypatch, capsys, caplog):
    _wire_dry_run(monkeypatch)
    verdict = SignVerdict(
        lf_index=1,
        status=STATUS_CONFLICTS,
        hit_count=3,
        reasons=['sign "COLLETON COUNTY" names Colleton — absent from sampled regions'],
        conflicting_counties=["Colleton"],
    )
    _install_fake_sign_verify(monkeypatch, bucket={1: [_hit()] * 3}, verdict=verdict)
    caplog.set_level(logging.WARNING, logger=lf_listing.__name__)

    result = lf_listing.print_label_dry_run(
        config=_config(sign_verify=True),
        state=_state({"2025-09-01": [_hit().to_dict()]}),
    )

    out = capsys.readouterr().out
    assert result is None  # advisory only — the label gate still exits normally
    assert "sign evidence: conflicts" in out
    assert "COLLETON COUNTY" in out
    assert "SIGN-VERIFY-CONFLICT" in caplog.text
    assert "LF 1" in caplog.text
    assert "Colleton" in caplog.text
    assert "Beaufort County" in caplog.text  # the sampled regions it was absent from


def test_dry_run_no_hits_prints_explicit_no_signal(monkeypatch, capsys):
    _wire_dry_run(monkeypatch)
    calls = _install_fake_sign_verify(monkeypatch, bucket={})  # nothing bucketed into LF 1

    lf_listing.print_label_dry_run(
        config=_config(sign_verify=True),
        state=_state({"2025-09-01": [_hit(date="2025-09-02").to_dict()]}),
    )

    out = capsys.readouterr().out
    assert "sign evidence: no-signal" in out
    assert STATUS_NO_SIGNAL == "no-signal"
    # Silence is never reported as agreement, and no comparison is invented.
    assert "corroborates" not in out
    assert calls["verify_lf"] == []


@pytest.mark.parametrize("mode", ["verify", "bucket"])
def test_dry_run_sign_block_exception_does_not_propagate(monkeypatch, capsys, mode):
    _wire_dry_run(monkeypatch)
    _install_fake_sign_verify(
        monkeypatch,
        bucket={1: [_hit()]},
        verdict=SignVerdict(lf_index=1),
        verify_lf_raises=(mode == "verify"),
        bucket_raises=(mode == "bucket"),
    )

    lf_listing.print_label_dry_run(
        config=_config(sign_verify=True),
        state=_state({"2025-09-01": [_hit().to_dict()]}),
    )

    out = capsys.readouterr().out
    # The label gate itself still printed — an operator who cannot read their
    # labels cannot ship.
    assert "Beaufort County" in out
    assert "NO encode" in out
    assert "sign evidence: unavailable" in out


# ── strict pre-encode gate ────────────────────────────────────────────


def _resolved(county="Beaufort County", state="South Carolina", regions=None):
    return SimpleNamespace(
        dominant_county=county,
        dominant_state=state,
        regions=list(regions if regions is not None else [county]),
    )


def _conflict_verdict(lf_index=1):
    return SignVerdict(
        lf_index=lf_index,
        status=STATUS_CONFLICTS,
        hit_count=4,
        reasons=['sign "WELCOME TO COLLETON COUNTY" names Colleton'],
        conflicting_counties=["Colleton"],
    )


def _wire_strict(monkeypatch):
    monkeypatch.setattr(
        phase_bc,
        "_build_drive_plan_from_state",
        lambda c, s: SimpleNamespace(regular_lfs=[SimpleNamespace(index=1, segments=[])]),
    )


def test_strict_gate_noops_when_flag_off_even_on_conflict(monkeypatch):
    _wire_strict(monkeypatch)
    calls = _install_fake_sign_verify(monkeypatch, verdicts={1: _conflict_verdict()})

    phase_bc._assert_sign_verify(
        _state({"2025-09-01": [_hit().to_dict()]}, {1: _resolved()}),
        _config(sign_verify=True, strict=False),
    )

    assert calls["verify_all"] == []


def test_strict_gate_raises_on_conflict(monkeypatch):
    _wire_strict(monkeypatch)
    calls = _install_fake_sign_verify(monkeypatch, verdicts={1: _conflict_verdict()})

    with pytest.raises(RuntimeError) as exc:
        phase_bc._assert_sign_verify(
            _state({"2025-09-01": [_hit().to_dict()]}, {1: _resolved()}),
            _config(sign_verify=True, strict=True),
        )

    msg = str(exc.value)
    assert "SIGN-VERIFY-CONFLICT" in msg
    assert "LF 1" in msg
    assert "Colleton" in msg  # what the signs claimed
    assert "Beaufort County" in msg  # the sampled regions it was absent from
    assert "WELCOME TO COLLETON COUNTY" in msg  # the raw sign text
    # A hard gate must never depend on a live network service.
    assert calls["verify_all"][0]["geocode_fn"] is None


@pytest.mark.parametrize(
    "verdicts",
    [
        {1: SignVerdict(lf_index=1, status=STATUS_CORROBORATES, hit_count=2)},
        {1: SignVerdict(lf_index=1, status=STATUS_NO_SIGNAL)},
        {},
    ],
)
def test_strict_gate_passes_on_non_conflict(monkeypatch, verdicts):
    _wire_strict(monkeypatch)
    _install_fake_sign_verify(monkeypatch, verdicts=verdicts)

    phase_bc._assert_sign_verify(
        _state({"2025-09-01": [_hit().to_dict()]}, {1: _resolved()}),
        _config(sign_verify=True, strict=True),
    )


def test_strict_gate_noops_without_hits_or_resolved_model(monkeypatch):
    _wire_strict(monkeypatch)
    calls = _install_fake_sign_verify(monkeypatch, verdicts={1: _conflict_verdict()})
    cfg = _config(sign_verify=True, strict=True)

    # No hits recorded at all.
    phase_bc._assert_sign_verify(_state({}, {1: _resolved()}), cfg)
    # GPS-less / legacy LFs: no resolved model to compare against.
    phase_bc._assert_sign_verify(_state({"2025-09-01": [_hit().to_dict()]}, {1: None}), cfg)
    phase_bc._assert_sign_verify(_state({"2025-09-01": [_hit().to_dict()]}, None), cfg)

    assert calls["verify_all"] == []
