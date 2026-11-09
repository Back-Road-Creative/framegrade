"""``--label-dry-run`` suggests a sign-derived label for an LF the camera never located.

Until now the dry-run printed ``<no GPS — --per-lf-location override required>``
for a GPS-less LF and moved on, so the one piece of evidence the footage holds
about where it was (its road signs, recorded by the A6 pass) never reached the
operator who then typed a label from memory. The preview runs the real
``sign_label.explain_sign_label`` over the LF's bucketed hits and prints the
label it would allow — or every reason it refuses one — as a SUGGESTION beside
the override line. Nothing is written: the operator copies the suggestion into
``--per-lf-location`` and the review still needs a person.

The comparator module is stubbed the same way ``test_sign_verify_wiring`` does
(bucketing is its concern); the deriver runs for real, so the printed label is
the one the rules produce, never a string the wiring invented.
"""

import sys
import types
from types import SimpleNamespace

import pytest

import gomoveshift.pipeline.drive_phases.phase_bc_day_first as phase_bc
import gomoveshift.pipeline.gps_enrichment as gps_enrichment
from gomoveshift.analysis.sign_evidence import SignHit
from gomoveshift.analysis.sign_label import KIND_COUNTY, KIND_STATE
from gomoveshift.pipeline.drive_phases import lf_listing

_SIGN_MODULE = "gomoveshift.analysis.sign_verify"
COPY_HINT = "--per-lf-location '1=Colleton County, SC'"


class _Geo:
    def reverse_geocode_detailed(self, lat, lon):
        return {"county": "Beaufort County", "state": "South Carolina"}


def _hit(t_local, *claims, raw_text="Colleton County SC"):
    return SignHit(
        episode_index=0,
        t_local=t_local,
        ts_day=t_local,
        date="2026-03-05",
        raw_text=raw_text,
        ocr_conf=0.9,
        claims=[{"kind": k, "value": v, "raw_text": raw_text} for k, v in claims],
    )


def _colleton(t):
    return _hit(t, (KIND_COUNTY, "Colleton"), (KIND_STATE, "South Carolina"))


def _stub_sign_verify(monkeypatch, *, bucket=None, bucket_raises=False):
    mod = types.ModuleType(_SIGN_MODULE)

    def bucket_hits_by_lf(hits_by_date, drive_plan):
        if bucket_raises:
            raise ValueError("bucketing exploded")
        return dict(bucket or {})

    def _unused(*a, **k):  # pragma: no cover - a GPS-less LF never compares
        raise AssertionError("comparator called for a GPS-less LF")

    mod.bucket_hits_by_lf = bucket_hits_by_lf
    mod.resolved_from_dominant = _unused
    mod.verify_lf = _unused
    mod.verify_all = _unused
    monkeypatch.setitem(sys.modules, _SIGN_MODULE, mod)


def _wire(monkeypatch, *, gps_points_by_lf):
    lf = SimpleNamespace(index=1, segments=[])
    drive_plan = SimpleNamespace(regular_lfs=[lf])
    monkeypatch.setattr(phase_bc, "_build_drive_plan_from_state", lambda c, s: drive_plan)
    monkeypatch.setattr(
        gps_enrichment, "_build_gps_points_by_lf", lambda p, g: dict(gps_points_by_lf)
    )
    monkeypatch.setattr(gps_enrichment, "_build_shared_geocoder", lambda *a, **k: _Geo())
    monkeypatch.setattr(lf_listing, "_live_title_ids", lambda: {})
    monkeypatch.setattr(lf_listing, "_render_ost_text_by_lf", lambda *a, **k: {})
    monkeypatch.setattr(lf_listing, "_print_insert_counties", lambda *a, **k: None)


def _config(*, sign_verify=True, min_hits=2):
    return SimpleNamespace(
        drive=SimpleNamespace(
            sign_verify=sign_verify,
            sign_verify_strict=False,
            sign_verify_min_corroborating_hits=min_hits,
        )
    )


def _state():
    return SimpleNamespace(
        gps_per_episode=[],
        sign_hits_by_date={"2026-03-05": [_colleton(1.0).to_dict()]},
        per_lf_resolved_location=None,
        output_base=None,
        stages_base=None,
    )


def _run(monkeypatch, capsys, *, bucket=None, gps=None, sign_verify=True, bucket_raises=False):
    _wire(monkeypatch, gps_points_by_lf=gps or {})
    _stub_sign_verify(monkeypatch, bucket=bucket, bucket_raises=bucket_raises)
    lf_listing.print_label_dry_run(config=_config(sign_verify=sign_verify), state=_state())
    return capsys.readouterr().out


def test_a_gpsless_lf_whose_signs_clear_the_rules_gets_a_copyable_suggestion(monkeypatch, capsys):
    out = _run(monkeypatch, capsys, bucket={1: [_colleton(10.0), _colleton(11.0)]})

    # The override line stays: the suggestion is beside it, never instead of it.
    assert "<no GPS — --per-lf-location override required>" in out
    assert "sign-derived label (suggest only): 'Colleton County, SC'" in out
    assert COPY_HINT in out
    # The deriver's own reason line, so the operator sees WHY.
    assert "LF 1: 'Colleton County, SC' from 2 distinct second(s)" in out
    assert "Colleton County SC" in out


def test_one_hit_prints_no_label_and_the_refusing_rule(monkeypatch, capsys):
    out = _run(monkeypatch, capsys, bucket={1: [_colleton(10.0)]})

    assert "sign-derived label: none (1 sign hit(s))" in out
    assert "below the 2-hit floor" in out
    assert COPY_HINT not in out


def test_no_hits_prints_an_explicit_none_never_silence(monkeypatch, capsys):
    out = _run(monkeypatch, capsys, bucket={})

    assert "<no GPS — --per-lf-location override required>" in out
    assert "sign-derived label: none (0 sign hit(s))" in out
    assert "LF 1: no sign hits" in out


def test_feature_off_prints_no_suggestion(monkeypatch, capsys):
    out = _run(
        monkeypatch, capsys, bucket={1: [_colleton(10.0), _colleton(11.0)]}, sign_verify=False
    )

    assert "<no GPS — --per-lf-location override required>" in out
    assert "sign-derived" not in out


def test_bucketing_failure_degrades_to_unavailable_and_keeps_the_listing(monkeypatch, capsys):
    out = _run(monkeypatch, capsys, bucket_raises=True)

    assert "<no GPS — --per-lf-location override required>" in out
    assert "sign-derived label: unavailable (bucketing exploded)" in out
    assert "NO encode" in out


def test_an_lf_with_fixes_but_no_county_vote_is_refused_by_the_gps_rule(monkeypatch, capsys):
    # Two fixes is below the geocode quorum here (``_Geo`` is patched, but
    # ``dominant_location`` over an empty-vote sample returns None), so the LF
    # prints the override line — yet it HAS fixes, so signs may not label it.
    _wire(monkeypatch, gps_points_by_lf={1: [{"lat": 32.3, "lon": -80.6}]})
    _stub_sign_verify(monkeypatch, bucket={1: [_colleton(10.0), _colleton(11.0)]})
    import gomoveshift.pipeline.title_builder as title_builder

    monkeypatch.setattr(title_builder, "dominant_location", lambda pts, geo: None)

    lf_listing.print_label_dry_run(config=_config(), state=_state())
    out = capsys.readouterr().out

    assert "<no GPS — --per-lf-location override required>" in out
    assert "has 1 GPS fix(es); the GPS chain owns its label" in out
    assert COPY_HINT not in out


@pytest.mark.parametrize("min_hits", [1, 3])
def test_the_floor_follows_the_config(monkeypatch, capsys, min_hits):
    _wire(monkeypatch, gps_points_by_lf={})
    _stub_sign_verify(monkeypatch, bucket={1: [_colleton(10.0), _colleton(11.0)]})

    lf_listing.print_label_dry_run(config=_config(min_hits=min_hits), state=_state())
    out = capsys.readouterr().out

    assert (COPY_HINT in out) == (min_hits <= 2)
