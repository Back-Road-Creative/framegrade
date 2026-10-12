"""Tests for the ``route-story-proof`` CLI: parser flags and the handler's
bundle resolution.

The handler resolves its bundle through ``prepare.find_prepared_bundle`` --
the exact same day-selection/footprint/source-version key
``route-story-prepare`` writes under -- rather than re-deriving its own,
which is the bug this file exists to catch (a re-derived key silently
drifted from what prepare actually wrote).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from gomoveshift.cli.parser import _build_parser


def _parse(argv: list[str]):
    return _build_parser().parse_args(argv)


def test_route_story_proof_requires_recorded_on():
    with pytest.raises(SystemExit):
        _parse(["route-story-proof", "trip.gpx", "out.mp4"])


def test_route_story_proof_accepts_workers_and_exaggeration():
    args = _parse(
        [
            "route-story-proof",
            "trip.gpx",
            "out.mp4",
            "--recorded-on",
            "2026-09-16",
            "--workers",
            "4",
            "--exaggeration",
            "1.1",
        ]
    )
    assert args.gpx == Path("trip.gpx")
    assert args.output == Path("out.mp4")
    assert args.recorded_on == date(2026, 9, 16)
    assert args.workers == 4
    assert args.exaggeration == pytest.approx(1.1)


def test_route_story_proof_workers_defaults_to_none():
    args = _parse(["route-story-proof", "trip.gpx", "out.mp4", "--recorded-on", "2026-09-16"])
    assert args.workers is None


def test_route_story_proof_trip_date_imagery_defaults_off():
    args = _parse(["route-story-proof", "trip.gpx", "out.mp4", "--recorded-on", "2026-09-16"])
    assert args.trip_date_imagery is False


def test_route_story_proof_accepts_trip_date_imagery_flag():
    args = _parse(
        [
            "route-story-proof",
            "trip.gpx",
            "out.mp4",
            "--recorded-on",
            "2026-09-16",
            "--trip-date-imagery",
        ]
    )
    assert args.trip_date_imagery is True


def test_cmd_route_story_proof_resolves_bundle_via_prepare_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gomoveshift.cli.handlers import route_story_proof as handler_mod

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    calls: dict = {}

    def fake_find_prepared_bundle(
        gpx_path,
        *,
        recorded_on,
        margin_m,
        terrain_exaggeration,
        regional_margin_km,
        trip_date_imagery,
    ):
        calls["find"] = {
            "gpx_path": gpx_path,
            "recorded_on": recorded_on,
            "margin_m": margin_m,
            "terrain_exaggeration": terrain_exaggeration,
            "regional_margin_km": regional_margin_km,
            "trip_date_imagery": trip_date_imagery,
        }
        return bundle_dir

    def fake_render_proof_clip(
        gpx_path, bundle_root, output_path, *, recorded_on, fps, width, height, workers
    ):
        calls["render"] = {
            "gpx_path": gpx_path,
            "bundle_root": bundle_root,
            "output_path": output_path,
            "recorded_on": recorded_on,
            "workers": workers,
        }
        return {"schema": "route-story-proof-receipt/2"}

    monkeypatch.setattr(handler_mod.prepare, "find_prepared_bundle", fake_find_prepared_bundle)
    monkeypatch.setattr(handler_mod.proof, "render_proof_clip", fake_render_proof_clip)

    output_path = tmp_path / "out.mp4"
    args = SimpleNamespace(
        gpx=tmp_path / "trip.gpx",
        output=output_path,
        recorded_on=date(2026, 9, 16),
        margin_m=None,
        exaggeration=handler_mod.prepare.DEFAULT_EXAGGERATION,
        regional_margin_km=handler_mod.prepare.DEFAULT_REGIONAL_MARGIN_KM,
        fps=30.0,
        width=1920,
        height=1080,
        workers=4,
        trip_date_imagery=True,
    )
    exit_code = handler_mod.cmd_route_story_proof(args, config=None)

    assert exit_code == 0
    assert calls["find"]["gpx_path"] == args.gpx
    assert calls["find"]["recorded_on"] == date(2026, 9, 16)
    assert calls["find"]["trip_date_imagery"] is True
    assert calls["render"]["bundle_root"] == bundle_dir
    assert calls["render"]["workers"] == 4
    receipt_path = Path(str(output_path) + ".receipt.json")
    assert json.loads(receipt_path.read_text())["schema"] == "route-story-proof-receipt/2"


def test_cmd_route_story_proof_fails_loudly_when_nothing_prepared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from gomoveshift.cli.handlers import route_story_proof as handler_mod

    monkeypatch.setattr(handler_mod.prepare, "find_prepared_bundle", lambda *a, **k: None)

    def _unreachable(*a, **k):
        raise AssertionError("render_proof_clip must not run with no bundle resolved")

    monkeypatch.setattr(handler_mod.proof, "render_proof_clip", _unreachable)

    args = SimpleNamespace(
        gpx=tmp_path / "trip.gpx",
        output=tmp_path / "out.mp4",
        recorded_on=date(2026, 9, 16),
        margin_m=None,
        exaggeration=handler_mod.prepare.DEFAULT_EXAGGERATION,
        regional_margin_km=handler_mod.prepare.DEFAULT_REGIONAL_MARGIN_KM,
        fps=30.0,
        width=1920,
        height=1080,
        workers=None,
        trip_date_imagery=False,
    )
    exit_code = handler_mod.cmd_route_story_proof(args, config=None)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "2026-09-16" in captured.err
    assert "route-story-prepare" in captured.err
