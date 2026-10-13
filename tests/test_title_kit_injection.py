"""The day render and Phase B/C take their title, chapter and brand inputs from a kit.

``pipeline.day_pipeline`` and ``drive_phases.phase_bc_day_first`` no longer import
``console``, ``publishing``, ``chapters`` or ``title_builder``/``longform_metadata``.
The GMS composition root builds a ``TitleKit`` (``publishing/title_kit.py``) and
threads it through ``PhaseBCDayFirst`` -> ``run_all_days`` -> ``RunSpec`` ->
``run_day_pipeline``; a run without one fails loudly.
"""

from __future__ import annotations

import dataclasses
import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gomoveshift.pipeline import day_pipeline, forward_geocode, media_api
from gomoveshift.pipeline.drive_phases import phase_bc_day_first
from gomoveshift.pipeline.drive_phases.phase_bc_day_first import PhaseBCDayFirst
from gomoveshift.pipeline.title_kit import TitleKit, require_title_kit
from gomoveshift.publishing.title_kit import build_title_kit

pytestmark = pytest.mark.no_default_title_kit


def test_require_title_kit_raises_a_pointed_error_without_a_kit():
    with pytest.raises(RuntimeError, match="title_kit"):
        require_title_kit(None)


def test_require_title_kit_returns_the_injected_kit():
    kit = MagicMock()
    assert require_title_kit(kit) is kit


def test_build_title_kit_fills_every_field():
    kit = build_title_kit()
    for f in dataclasses.fields(TitleKit):
        assert getattr(kit, f.name) is not None, f.name
    assert kit.min_chapters >= 1


def test_run_day_pipeline_without_a_kit_fails_at_the_first_kit_step(tmp_path):
    batch = SimpleNamespace(date="2026-01-01", episodes=[SimpleNamespace(path="x", index=1)])
    with pytest.raises(RuntimeError, match="no title kit was injected"):
        day_pipeline.run_day_pipeline(
            day_batch=batch,
            day_scores=None,
            output_dir=tmp_path,
            location="X",
            lf_plans=[],
        )


def test_superlong_part_metadata_without_a_kit_fails_loudly():
    with pytest.raises(RuntimeError, match="no title kit was injected"):
        day_pipeline._build_superlong_part_metadata(
            part=SimpleNamespace(part_index=1, part_count=1, kept_duration_sec=1.0, ranges=[]),
            date="2026-01-01",
            location="X",
            gpx_path=None,
            resolved_location=None,
            config=None,
            stages_dir=None,
        )


def test_phase_bc_without_a_kit_has_none_and_the_constructor_stores_one():
    assert PhaseBCDayFirst(MagicMock()).title_kit is None
    kit = MagicMock()
    assert PhaseBCDayFirst(MagicMock(), title_kit=kit).title_kit is kit


def test_run_spec_carries_the_kit_to_run_day_pipeline(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(media_api, "run_day_pipeline", lambda **kw: seen.update(kw))
    kit = MagicMock()
    media_api.render_day(
        media_api.RunSpec(
            day_batch=None,
            day_scores=None,
            output_dir=tmp_path,
            location="X",
            lf_plans=[],
            kit=kit,
        )
    )
    assert seen["kit"] is kit


def test_forward_geocoders_degrade_instead_of_raising(monkeypatch):
    def boom(*_a, **_k):
        raise OSError("down")

    monkeypatch.setattr(forward_geocode, "_nominatim_search", boom)
    monkeypatch.setattr(forward_geocode, "_nominatim_places", boom)
    assert forward_geocode.forward_geocode("Goleen") is None
    assert forward_geocode.forward_geocode_places("Goleen") == []


def test_engine_modules_do_not_import_publish_side_helpers():
    for mod in (day_pipeline, phase_bc_day_first):
        src = inspect.getsource(mod)
        for banned in (
            "console.settings",
            "publishing",
            "chapters",
            "longform_metadata",
            "import title_builder",
            "..title_builder",
        ):
            assert f"import {banned}" not in src and f"from {banned}" not in src, (mod, banned)
            assert f"from ..{banned}" not in src and f"from .{banned}" not in src, (mod, banned)


def test_drive_orchestrator_builds_the_kit_and_hands_it_to_phase_bc(monkeypatch):
    """The production composition root threads a kit into Phase B/C.

    The conftest autouse fixture supplies the real kit whenever a test passes none,
    so only this opted-out test would catch ``DrivePipelineOrchestrator`` dropping
    the ``title_kit=`` argument.
    """
    from gomoveshift.pipeline import drive_orchestrator

    sentinel = MagicMock(name="kit")
    built = []

    def fake_build():
        built.append(1)
        return sentinel

    monkeypatch.setattr(drive_orchestrator, "build_title_kit", fake_build)
    orch = drive_orchestrator.DrivePipelineOrchestrator(MagicMock())
    assert built == [1]
    assert orch._phases["phase_bc_day_first"].title_kit is sentinel


def test_drive_orchestrator_default_kit_is_the_real_gms_kit():
    from gomoveshift.pipeline import drive_orchestrator

    kit = (
        drive_orchestrator.DrivePipelineOrchestrator(MagicMock())
        ._phases["phase_bc_day_first"]
        .title_kit
    )
    assert isinstance(kit, TitleKit)
