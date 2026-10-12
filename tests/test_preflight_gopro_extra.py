"""Preflight gate: a GoPro-ingest drive needs the ``gopro`` extra (gopro-overlay)
to read GPMF GPS, and must say so at minute 0.

Failure mode: a venv installed without the ``gopro`` extra.
``extract_gopro_gpx`` logs "gopro-overlay not installed" at ERROR and returns
``None``, the concatenator only warns "no GPS data extracted", and the run
keeps going into a long audit that can only end in a GPS-less LF and shorts.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from gomoveshift.pipeline import preflight
from gomoveshift.pipeline.preflight import check_gopro_telemetry_parser, run_preflight


class TestGoproTelemetryParserGate:
    def test_gopro_ingest_without_the_parser_fails_with_the_repair(self):
        with pytest.raises(RuntimeError) as exc:
            check_gopro_telemetry_parser("gopro", importable=False)
        msg = str(exc.value)
        assert "gopro-overlay" in msg
        assert ".[gopro]" in msg
        assert "pip install -e '.[gopro]'" in msg
        # gopro-overlay pins gpxpy==1.6.1 (gps-journey needs >=1.6.2), so it is
        # not in the extra: the repair has to name its --no-deps install.
        assert "pip install --no-deps gopro-overlay==0.134.0" in msg
        assert "gpxpy==1.6.1" in msg
        assert "_ops/" not in msg
        assert "--ingest-mode generic" in msg

    def test_gopro_ingest_with_the_parser_passes(self):
        check_gopro_telemetry_parser("gopro", importable=True)

    def test_generic_ingest_never_needs_the_parser(self):
        check_gopro_telemetry_parser("generic", importable=False)

    def test_default_probe_looks_up_gopro_overlay(self, monkeypatch):
        seen: list[str] = []

        def _find_spec(name):
            seen.append(name)
            return None

        monkeypatch.setattr(preflight.importlib.util, "find_spec", _find_spec)
        with pytest.raises(RuntimeError):
            check_gopro_telemetry_parser("gopro")
        assert seen == ["gopro_overlay"]


class TestRunPreflightWiring:
    def test_run_preflight_takes_ingest_mode_defaulting_to_skip(self):
        param = inspect.signature(run_preflight).parameters["ingest_mode"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default is None

    def test_run_preflight_refuses_a_gopro_run_without_the_parser(self, tmp_path, monkeypatch):
        """Fails before the disk/encode checks: nothing heavy is worth measuring
        for a run that cannot read its own GPS."""
        monkeypatch.setattr(preflight.importlib.util, "find_spec", lambda _name: None)
        src = tmp_path / "src"
        src.mkdir()
        with pytest.raises(RuntimeError, match="gopro-overlay"):
            run_preflight(src, tmp_path / "out", require_tools=None, ingest_mode="gopro")

    def test_orchestrator_passes_the_configured_ingest_mode(self):
        src = Path(preflight.__file__).with_name("orchestrator_preflight.py").read_text("utf-8")
        assert "ingest_mode=config.drive.ingest_mode" in src
