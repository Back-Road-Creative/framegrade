"""``--label-dry-run`` prints each LF's assembled YouTube title and flags a
collision with a title already live on the channel, before any encode.

The Phase BC duplicate-title gate only refuses after the encode. The dry-run
compares against the on-disk ledger and the local live-catalog snapshot;
it never calls YouTube. Ids and titles below are synthetic.
"""

from types import SimpleNamespace

import gomoveshift.pipeline.drive_phases.phase_bc_day_first as phase_bc
import gomoveshift.pipeline.gps_enrichment as gps_enrichment
import gomoveshift.pipeline.longform_metadata as lm
from gomoveshift.pipeline.drive_phases import lf_listing

_PTS = [{"lat": 40.0, "lon": -75.0}] * 6


class _Geo:
    def reverse_geocode_detailed(self, lat, lon):
        return {"county": "Synth County", "state": "Pennsylvania"}


def _run(monkeypatch, capsys, *, ledger=(), snapshot=None):
    lf = SimpleNamespace(index=1, segments=[])
    monkeypatch.setattr(
        phase_bc, "_build_drive_plan_from_state", lambda c, s: SimpleNamespace(regular_lfs=[lf])
    )
    monkeypatch.setattr(gps_enrichment, "_build_gps_points_by_lf", lambda p, g: {1: _PTS})
    monkeypatch.setattr(gps_enrichment, "_build_shared_geocoder", lambda *a, **k: _Geo())
    monkeypatch.setattr(lm, "load_published_titles", lambda: set(ledger))
    monkeypatch.setattr(lm, "live_catalog_title_ids", lambda: dict(snapshot or {}))
    monkeypatch.setattr(
        lm, "refresh_channel_title_ledger", lambda *a, **k: (_ for _ in ()).throw(AssertionError)
    )
    lf_listing.print_label_dry_run(config=object(), state=SimpleNamespace(gps_per_episode=[]))
    return capsys.readouterr().out


def _title():
    return lm.assemble_final_title("Synth County, Pennsylvania")


def test_each_lf_prints_its_assembled_title(monkeypatch, capsys):
    out = _run(monkeypatch, capsys)
    assert f"title: {_title()}" in out
    assert "TITLE-COLLISION" not in out


def test_a_title_in_the_snapshot_is_flagged_with_its_live_id(monkeypatch, capsys):
    out = _run(monkeypatch, capsys, snapshot={_title().lower(): "SYNTHdup01"})
    assert "TITLE-COLLISION" in out
    assert "SYNTHdup01" in out


def test_a_title_in_the_on_disk_ledger_is_flagged(monkeypatch, capsys):
    out = _run(monkeypatch, capsys, ledger={_title().upper()})
    assert "TITLE-COLLISION" in out


def test_the_footer_counts_collisions(monkeypatch, capsys):
    out = _run(monkeypatch, capsys, ledger={_title()})
    assert "1 LF title(s) already live" in out


def _stub_render_canonical(monkeypatch, text="Synthville, PA"):
    """Pin the render's canonical OST text (``build_lf_title``) to a synthetic value."""
    monkeypatch.setattr(gps_enrichment, "build_lf_title", lambda *a, **k: text)
    return text


def test_title_is_assemble_final_title_of_the_render_canonical_ost_text(monkeypatch, capsys):
    canonical = _stub_render_canonical(monkeypatch)
    out = _run(monkeypatch, capsys)
    assert f"title: {lm.assemble_final_title(canonical)}" in out
    assert _title() not in out


def test_collision_check_uses_the_render_canonical_title(monkeypatch, capsys):
    canonical = _stub_render_canonical(monkeypatch)
    live = lm.assemble_final_title(canonical).lower()
    out = _run(monkeypatch, capsys, snapshot={live: "SYNTHrender1"})
    assert "SYNTHrender1" in out
    # the county-label title the render never uses must NOT drive the check
    out = _run(monkeypatch, capsys, snapshot={_title().lower(): "SYNTHcounty1"})
    assert "TITLE-COLLISION" not in out


def test_operator_override_is_the_printed_title_verbatim(monkeypatch, capsys):
    _stub_render_canonical(monkeypatch)
    lf = SimpleNamespace(index=1, segments=[])
    monkeypatch.setattr(
        phase_bc, "_build_drive_plan_from_state", lambda c, s: SimpleNamespace(regular_lfs=[lf])
    )
    monkeypatch.setattr(gps_enrichment, "_build_gps_points_by_lf", lambda p, g: {1: _PTS})
    monkeypatch.setattr(gps_enrichment, "_build_shared_geocoder", lambda *a, **k: _Geo())
    monkeypatch.setattr(lm, "load_published_titles", lambda: set())
    monkeypatch.setattr(lm, "live_catalog_title_ids", lambda: {})
    state = SimpleNamespace(gps_per_episode=[], per_lf_overrides={1: "Synth Ridge Loop"})
    lf_listing.print_label_dry_run(config=object(), state=state)
    assert f"title: {lm.assemble_final_title('Synth Ridge Loop')}" in capsys.readouterr().out
