"""A keyframe-probe failure must never silently downgrade to a CPU
re-encode — the operator has to see it, every time it happens, not
just the first.

Two sites, both falling back to a (safe, deliberate) re-encode when
the probe can't answer, but neither one telling the operator so at a
level that survives default logging config:

* ``montage_builder._trim_peak_to`` — ``probe_keyframes`` returning
  ``None`` falls straight into ``reencode_range`` with no log call at
  all.
* ``assembly.encoder.VideoEncoder._starts_land_on_keyframes`` — a
  probe exception is logged at DEBUG, invisible unless an operator
  has DEBUG logging on, then the caller only reports "re-encoding" at
  INFO with no indication *why*.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from gomoveshift.assembly.encoder import VideoEncoder
from gomoveshift.pipeline import montage_builder

MONTAGE_LOGGER = "gomoveshift.pipeline.montage_builder"
ENCODER_LOGGER = "gomoveshift.assembly.encoder"


class TestMontageKeyframeProbeSignal:
    """Site 1: ``montage_builder._trim_peak_to``."""

    def _call(self, tmp_path, name="master.mp4"):
        master = tmp_path / name
        master.write_bytes(b"fake")
        peak = SimpleNamespace(source_file_index=0, start_time=100.0, end_time=140.0)
        boundaries = [(0.0, 600.0)]
        out = tmp_path / f"{name}.out.mp4"
        with patch("gomoveshift.pipeline.montage_builder.probe_keyframes", return_value=None):
            with patch("gomoveshift.pipeline.montage_builder.reencode_range", return_value=True):
                montage_builder._trim_peak_to(master, peak, boundaries, out, clip_duration=5.0)

    def test_probe_failure_logs_warning(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING, logger=MONTAGE_LOGGER):
            self._call(tmp_path, "a.mp4")
        warnings = [
            r for r in caplog.records if r.name == MONTAGE_LOGGER and r.levelno >= logging.WARNING
        ]
        assert warnings, "keyframe probe failure must log at WARNING or above"

    def test_probe_failure_logs_warning_on_second_occurrence_too(self, tmp_path, caplog):
        """Not cached: a second peak on a second probe failure must ALSO warn."""
        with caplog.at_level(logging.WARNING, logger=MONTAGE_LOGGER):
            self._call(tmp_path, "a.mp4")
            self._call(tmp_path, "b.mp4")
        warnings = [
            r for r in caplog.records if r.name == MONTAGE_LOGGER and r.levelno >= logging.WARNING
        ]
        assert len(warnings) >= 2, (
            f"expected a WARNING on every occurrence, not just the first; got {len(warnings)}"
        )


class TestAssemblyEncoderKeyframeProbeSignal:
    """Site 2: ``VideoEncoder._starts_land_on_keyframes``."""

    def _call(self):
        enc = VideoEncoder()
        with patch("keycut.probe_keyframes", side_effect=RuntimeError("probe boom")):
            return enc._starts_land_on_keyframes(Path("/src.mp4"), [0.0, 10.0])

    def test_probe_exception_logs_warning_not_debug(self, caplog):
        with caplog.at_level(logging.DEBUG, logger=ENCODER_LOGGER):
            result = self._call()
        assert result is False
        matching = [r for r in caplog.records if r.name == ENCODER_LOGGER]
        assert matching, "expected a log record for the probe failure"
        assert all(r.levelno >= logging.WARNING for r in matching), (
            f"probe failure must not be logged below WARNING; got levels "
            f"{[r.levelname for r in matching]}"
        )

    def test_probe_exception_logs_warning_on_second_occurrence_too(self, caplog):
        with caplog.at_level(logging.DEBUG, logger=ENCODER_LOGGER):
            self._call()
            self._call()
        matching = [
            r for r in caplog.records if r.name == ENCODER_LOGGER and r.levelno >= logging.WARNING
        ]
        assert len(matching) >= 2, (
            f"expected a WARNING on every occurrence, not just the first; got {len(matching)}"
        )
