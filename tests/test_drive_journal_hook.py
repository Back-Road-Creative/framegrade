"""Tests for the drive post-Phase-D journal auto-publish hook (default-on).

Covers slug/region/site-path resolution and the loud warn-on-unresolvable
path. ``run_publish_journal`` is patched to a spy so these stay unit-fast and
don't need a real journey/site.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest

from gomoveshift.cli.handlers.drive import _effective_region, _maybe_auto_publish_journal

_RPJ = "gomoveshift.cli.handlers.publish_journal.run_publish_journal"


class _FakeOrch:
    def __init__(self, project_dir, stages_base):
        self._paths = (project_dir, stages_base)

    def _resolve_paths(self, directory, output_dir):
        return self._paths


def _args(**kw):
    base = dict(
        region=None,
        auto_publish_journal=None,
        no_auto_publish_journal=False,
        site_path=None,
        publish=None,
        directory="/x",
        output_dir="/y",
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _cfg(mapping=None):
    return SimpleNamespace(drive=SimpleNamespace(active_journey_by_region=mapping or {}))


def _state(stages, region=None):
    stages.mkdir(parents=True, exist_ok=True)
    (stages / "pipeline_state.json").write_text(json.dumps({"region": region} if region else {}))


def test_effective_region_prefers_cli(tmp_path):
    sf = tmp_path / "pipeline_state.json"
    sf.write_text(json.dumps({"region": "usa"}))
    assert _effective_region(_args(region="ireland"), sf) == "ireland"


def test_effective_region_falls_back_to_state(tmp_path):
    sf = tmp_path / "pipeline_state.json"
    sf.write_text(json.dumps({"region": "usa"}))
    assert _effective_region(_args(region=None), sf) == "usa"


def test_no_flag_disables(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(_RPJ, lambda **k: called.append(k) or 0)
    orch = _FakeOrch(tmp_path / "proj", tmp_path / "stages")
    _maybe_auto_publish_journal(
        _args(
            no_auto_publish_journal=True,
            auto_publish_journal="x",
            site_path=tmp_path,
            publish="sun",
        ),
        orch,
        _cfg(),
    )
    assert called == []


def test_region_map_resolves_slug_and_routes_pending_dir(monkeypatch, tmp_path):
    stages = tmp_path / "stages"
    _state(stages, region="ireland")
    called = {}
    monkeypatch.setattr(_RPJ, lambda **k: called.update(k) or 0)
    orch = _FakeOrch(tmp_path / "proj", stages)
    _maybe_auto_publish_journal(
        _args(region="ireland", site_path=tmp_path, publish="sun"),
        orch,
        _cfg({"ireland": "example-2025-09"}),
    )
    assert called.get("journey_slug") == "example-2025-09"
    assert called.get("pending_dir") == tmp_path / "proj"


def test_explicit_slug_overrides_region_map(monkeypatch, tmp_path):
    stages = tmp_path / "stages"
    _state(stages, region="ireland")
    called = {}
    monkeypatch.setattr(_RPJ, lambda **k: called.update(k) or 0)
    orch = _FakeOrch(tmp_path / "proj", stages)
    _maybe_auto_publish_journal(
        _args(auto_publish_journal="explicit-trip", site_path=tmp_path, publish="sun"),
        orch,
        _cfg({"ireland": "example-2025-09"}),
    )
    assert called.get("journey_slug") == "explicit-trip"


def test_warns_when_unresolvable_on_publish_run(monkeypatch, tmp_path, caplog):
    stages = tmp_path / "stages"
    _state(stages, region="usa")  # 'usa' not in the map → no slug
    called = []
    monkeypatch.setattr(_RPJ, lambda **k: called.append(k) or 0)
    orch = _FakeOrch(tmp_path / "proj", stages)
    with caplog.at_level(logging.WARNING, logger="gomoveshift.cli.handlers.drive"):
        _maybe_auto_publish_journal(
            _args(region="usa", site_path=tmp_path, publish="sun"),
            orch,
            _cfg({"ireland": "example-2025-09"}),
        )
    assert called == []  # nothing published
    assert any("site will NOT update" in r.getMessage() for r in caplog.records)


def test_refuses_journal_merge_on_region_mismatch(monkeypatch, tmp_path, caplog):
    # A USA LF (ResolvedLocation -> usa) under a --region ireland run must
    # refuse the whole merge — nothing published, nothing routed.
    stages = tmp_path / "stages"
    stages.mkdir(parents=True, exist_ok=True)
    (stages / "pipeline_state.json").write_text(
        json.dumps(
            {
                "region": "ireland",
                "per_lf_resolved_location": {"1": {"is_us": True, "country_code": "US"}},
            }
        )
    )
    called = []
    monkeypatch.setattr(_RPJ, lambda **k: called.append(k) or 0)
    orch = _FakeOrch(tmp_path / "proj", stages)
    with caplog.at_level(logging.ERROR, logger="gomoveshift.cli.handlers.drive"):
        _maybe_auto_publish_journal(
            _args(region="ireland", site_path=tmp_path, publish="sun"),
            orch,
            _cfg({"ireland": "example-2025-09"}),
        )
    assert called == []  # merge refused, nothing routed
    assert not (tmp_path / "data" / "video_routing.yaml").exists()  # no durable route
    assert any("REFUSED" in r.getMessage() for r in caplog.records)


def test_matching_region_still_publishes(monkeypatch, tmp_path):
    stages = tmp_path / "stages"
    stages.mkdir(parents=True, exist_ok=True)
    (stages / "pipeline_state.json").write_text(
        json.dumps(
            {
                "region": "ireland",
                "per_lf_resolved_location": {"1": {"is_us": False, "country_code": "Ireland"}},
            }
        )
    )
    called = {}
    monkeypatch.setattr(_RPJ, lambda **k: called.update(k) or 0)
    orch = _FakeOrch(tmp_path / "proj", stages)
    _maybe_auto_publish_journal(
        _args(region="ireland", site_path=tmp_path, publish="sun"),
        orch,
        _cfg({"ireland": "example-2025-09"}),
    )
    assert called.get("journey_slug") == "example-2025-09"  # proceeded


def test_quiet_when_unresolvable_on_non_publish_run(monkeypatch, tmp_path, caplog):
    stages = tmp_path / "stages"
    _state(stages, region="usa")
    monkeypatch.setattr(_RPJ, lambda **k: 0)
    orch = _FakeOrch(tmp_path / "proj", stages)
    with caplog.at_level(logging.WARNING, logger="gomoveshift.cli.handlers.drive"):
        _maybe_auto_publish_journal(
            _args(region="usa", publish=None),  # smoke run, no --publish
            orch,
            _cfg({"ireland": "example-2025-09"}),
        )
    assert not any("site will NOT update" in r.getMessage() for r in caplog.records)


# --- Date-based journey resolution (no region-map entry) ---------------


def _journey_md(path, slug, start, end):
    path.write_text(
        f"---\ntitle: \"{slug}\"\ntrip:\n  start_date: '{start}'\n  end_date: '{end}'\n---\nbody\n"
    )
    return path


def _site_with_journeys(tmp_path, *journeys):
    site = tmp_path / "site"
    jdir = site / "content" / "journeys"
    jdir.mkdir(parents=True)
    for slug, start, end in journeys:
        _journey_md(jdir / f"{slug}.md", slug, start, end)
    return site


def _state_with_date(stages, region, run_date):
    stages.mkdir(parents=True, exist_ok=True)
    (stages / "pipeline_state.json").write_text(
        json.dumps({"region": region, "per_lf_recording_date": {"1": run_date}})
    )


def test_date_resolution_single_match(monkeypatch, tmp_path):
    stages = tmp_path / "stages"
    _state_with_date(stages, "usa", "2026-03-04")
    site = _site_with_journeys(
        tmp_path,
        ("journey-a-2026", "2026-03-04", "2026-03-05"),
        ("journey-b-2026", "2026-02-20", "2026-02-20"),
    )
    called = {}
    monkeypatch.setattr(_RPJ, lambda **k: called.update(k) or 0)
    orch = _FakeOrch(tmp_path / "proj", stages)
    _maybe_auto_publish_journal(
        _args(region="usa", site_path=site, publish="tue"),
        orch,
        _cfg({}),  # no region-map entry -> date resolution
    )
    assert called.get("journey_slug") == "journey-a-2026"


def test_date_resolution_ambiguous_refuses(monkeypatch, tmp_path, caplog):
    stages = tmp_path / "stages"
    _state_with_date(stages, "usa", "2026-03-04")
    site = _site_with_journeys(
        tmp_path,
        ("journey-a-2026", "2026-03-01", "2026-03-10"),
        ("overlap-trip", "2026-03-04", "2026-03-04"),
    )
    called = []
    monkeypatch.setattr(_RPJ, lambda **k: called.append(k) or 0)
    orch = _FakeOrch(tmp_path / "proj", stages)
    with caplog.at_level(logging.ERROR, logger="gomoveshift.cli.handlers.drive"):
        _maybe_auto_publish_journal(
            _args(region="usa", site_path=site, publish="tue"),
            orch,
            _cfg({}),
        )
    assert called == []
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "journey-a-2026" in messages
    assert "overlap-trip" in messages


def test_date_resolution_zero_matches_creates_draft_journey(monkeypatch, tmp_path, caplog):
    stages = tmp_path / "stages"
    _state_with_date(stages, "usa", "2026-05-01")
    site = _site_with_journeys(tmp_path, ("journey-a-2026", "2026-03-04", "2026-03-05"))
    called = {}
    monkeypatch.setattr(_RPJ, lambda **k: called.update(k) or 0)
    orch = _FakeOrch(tmp_path / "proj", stages)
    with caplog.at_level(logging.INFO, logger="gomoveshift.cli.handlers.drive"):
        _maybe_auto_publish_journal(
            _args(region="usa", site_path=site, publish="tue"),
            orch,
            _cfg({}),
        )
    assert called.get("journey_slug") == "usa-2026-05-01"  # merge proceeded into the new draft
    journey_path = site / "content" / "journeys" / "usa-2026-05-01.md"
    assert journey_path.is_file()
    assert "draft: true" in journey_path.read_text()
    assert any("created" in r.getMessage() for r in caplog.records)


def test_date_resolution_zero_matches_never_overwrites_existing(monkeypatch, tmp_path, caplog):
    # The derived slug already has a file on disk that resolve_journey_by_date
    # did NOT match (its trip range excludes the run date) — refuse rather
    # than clobber it.
    stages = tmp_path / "stages"
    _state_with_date(stages, "usa", "2026-05-01")
    site = _site_with_journeys(tmp_path, ("journey-a-2026", "2026-03-04", "2026-03-05"))
    collision = site / "content" / "journeys" / "usa-2026-05-01.md"
    collision.write_text("---\ntitle: pre-existing\n---\nbody\n")
    called = []
    monkeypatch.setattr(_RPJ, lambda **k: called.append(k) or 0)
    orch = _FakeOrch(tmp_path / "proj", stages)
    with caplog.at_level(logging.ERROR, logger="gomoveshift.cli.handlers.drive"):
        _maybe_auto_publish_journal(
            _args(region="usa", site_path=site, publish="tue"),
            orch,
            _cfg({}),
        )
    assert called == []
    assert collision.read_text() == "---\ntitle: pre-existing\n---\nbody\n"
    assert any("REFUSED" in r.getMessage() for r in caplog.records)


def test_date_resolution_single_match_creates_no_draft(monkeypatch, tmp_path):
    stages = tmp_path / "stages"
    _state_with_date(stages, "usa", "2026-03-04")
    site = _site_with_journeys(tmp_path, ("journey-a-2026", "2026-03-04", "2026-03-05"))
    monkeypatch.setattr(_RPJ, lambda **k: 0)
    orch = _FakeOrch(tmp_path / "proj", stages)
    _maybe_auto_publish_journal(
        _args(region="usa", site_path=site, publish="tue"),
        orch,
        _cfg({}),
    )
    assert list((site / "content" / "journeys").glob("usa-*.md")) == []


def test_region_map_still_wins_over_date_inference(monkeypatch, tmp_path):
    stages = tmp_path / "stages"
    _state_with_date(stages, "ireland", "2026-03-04")
    # A journey exists whose range contains the date but the region map is set
    # for "ireland" -> a different slug; the map must win, no date scan needed.
    site = _site_with_journeys(tmp_path, ("journey-a-2026", "2026-03-04", "2026-03-05"))
    called = {}
    monkeypatch.setattr(_RPJ, lambda **k: called.update(k) or 0)
    orch = _FakeOrch(tmp_path / "proj", stages)
    _maybe_auto_publish_journal(
        _args(region="ireland", site_path=site, publish="sun"),
        orch,
        _cfg({"ireland": "example-2025-09"}),
    )
    assert called.get("journey_slug") == "example-2025-09"


def test_explicit_slug_wins_over_date_inference(monkeypatch, tmp_path):
    stages = tmp_path / "stages"
    _state_with_date(stages, "usa", "2026-03-04")
    site = _site_with_journeys(tmp_path, ("journey-a-2026", "2026-03-04", "2026-03-05"))
    called = {}
    monkeypatch.setattr(_RPJ, lambda **k: called.update(k) or 0)
    orch = _FakeOrch(tmp_path / "proj", stages)
    _maybe_auto_publish_journal(
        _args(auto_publish_journal="explicit-trip", region="usa", site_path=site, publish="tue"),
        orch,
        _cfg({}),
    )
    assert called.get("journey_slug") == "explicit-trip"


# --- upload-off runs have nothing to merge: the hook must not spawn it ------


def _hook_logs(caplog, level):
    return [r for r in caplog.records if r.levelno >= level]


def test_no_upload_run_skips_journal_hook(monkeypatch, tmp_path, caplog):
    called = []
    monkeypatch.setattr(_RPJ, lambda **k: called.append(k) or 0)
    orch = _FakeOrch(tmp_path / "proj", tmp_path / "stages")
    with caplog.at_level(logging.INFO, logger="gomoveshift.cli.handlers.drive"):
        _maybe_auto_publish_journal(
            _args(no_youtube_upload=True, auto_publish_journal="x", site_path=tmp_path),
            orch,
            _cfg(),
        )
    assert called == []
    assert any("YouTube upload was off" in r.getMessage() for r in caplog.records)
    assert not _hook_logs(caplog, logging.WARNING)


def test_default_smoke_run_skips_journal_hook(monkeypatch, tmp_path, caplog):
    called = []
    monkeypatch.setattr(_RPJ, lambda **k: called.append(k) or 0)
    orch = _FakeOrch(tmp_path / "proj", tmp_path / "stages")
    with caplog.at_level(logging.INFO, logger="gomoveshift.cli.handlers.drive"):
        _maybe_auto_publish_journal(
            _args(auto_publish_journal="x", site_path=tmp_path), orch, _cfg()
        )
    assert called == []
    assert any("YouTube upload was off" in r.getMessage() for r in caplog.records)
    assert not _hook_logs(caplog, logging.WARNING)


def test_upload_run_still_publishes(monkeypatch, tmp_path):
    stages = tmp_path / "stages"
    _state(stages)
    called = {}
    monkeypatch.setattr(_RPJ, lambda **k: called.update(k) or 0)
    orch = _FakeOrch(tmp_path / "proj", stages)
    _maybe_auto_publish_journal(
        _args(youtube_upload=True, auto_publish_journal="x", site_path=tmp_path),
        orch,
        _cfg(),
    )
    assert called.get("journey_slug") == "x"


# --- discovery runs (--dry-run/--list-lfs/--label-dry-run) upload nothing ---


@pytest.mark.parametrize("flag", ["dry_run", "list_lfs", "label_dry_run"])
def test_discovery_run_skips_journal_hook(monkeypatch, tmp_path, flag):
    # A --youtube-upload --dry-run plan render must neither spawn
    # publish-journal nor create a draft journey on the site.
    stages = tmp_path / "stages"
    _state_with_date(stages, "usa", "2026-05-01")
    site = _site_with_journeys(tmp_path, ("journey-a-2026", "2026-03-04", "2026-03-05"))
    called = []
    monkeypatch.setattr(_RPJ, lambda **k: called.append(k) or 0)
    orch = _FakeOrch(tmp_path / "proj", stages)
    _maybe_auto_publish_journal(
        _args(youtube_upload=True, region="usa", site_path=site, publish="fri", **{flag: True}),
        orch,
        _cfg({}),
    )
    assert called == []
    assert list((site / "content" / "journeys").glob("usa-*.md")) == []
