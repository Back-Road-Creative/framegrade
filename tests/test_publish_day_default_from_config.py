"""Regression test: orchestrator publish_day default must come from
config, not a stale ``"sat"``/``"wed"`` literal.

Config sets Wed for BOTH regions (``long_form_publish_days=[2]``) and
state-invariant Rule 2 rejects any weekday not in ``LONG_FORM_DAYS``. The old
default ``"sat" if region == "usa" else "wed"`` therefore resolved a USA drive
started without ``--lf-day`` to ``"sat"`` (dow=5), which is not in
``LONG_FORM_DAYS={2}`` — a ``StateInvariantError`` waiting to fire (masked only
because operators always pass ``--lf-day``).

This mirrors the existing source-of-truth pattern already used in
``scheduler.generate_drive_schedule`` (``min(self.long_form_days)``).
"""

from unittest.mock import patch

import pytest

from gomoveshift.config import Config
from gomoveshift.pipeline import schedule_constants
from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator
from gomoveshift.pipeline.schedule_constants import default_publish_day, long_form_days
from gomoveshift.pipeline.state_invariants import _DAY_NAME_TO_INT


class _StopRunError(Exception):
    """Sentinel raised from the _build_initial_state spy to short-circuit
    run() right after publish_day is resolved — before any heavy state IO."""


def _resolved_publish_day(orch, *, region, publish_day, tmp_path):
    """Drive orch.run() far enough to capture the publish_day it resolved
    and threaded into _build_initial_state, then abort."""
    captured = {}

    def _spy(*args, **kwargs):
        captured["publish_day"] = kwargs.get("publish_day")
        raise _StopRunError

    source = tmp_path / "src"
    source.mkdir()
    with patch.object(DrivePipelineOrchestrator, "_build_initial_state", _spy):
        with pytest.raises(_StopRunError):
            orch.run(
                source_dir=source,
                output_base=tmp_path / "out",
                region=region,
                publish_day=publish_day,
                force=True,
            )
    return captured["publish_day"]


class TestDefaultPublishDayFromConfig:
    def test_default_lands_in_long_form_days(self):
        """The config-derived default must be a weekday in LONG_FORM_DAYS —
        otherwise it trips state-invariant Rule 2 on the first save."""
        config = Config()
        day = default_publish_day(config)
        pd_int = _DAY_NAME_TO_INT[day.lower()[:3]]
        assert pd_int in long_form_days(config)

    def test_usa_default_is_wed_not_sat(self):
        """USA must default to Wed under the current cadence, never the
        pre-fix Sat literal (dow=5 ∉ LONG_FORM_DAYS={2})."""
        config = Config()
        assert config.drive.long_form_publish_days == [2]
        assert default_publish_day(config) == "wed"

    def test_honors_min_of_configured_days(self):
        """When config carries multiple long-form days, the default is the
        smallest dow (matching scheduler.generate_drive_schedule)."""
        config = Config()
        config.drive.long_form_publish_days = [4, 2]  # Fri, Wed
        assert default_publish_day(config) == "wed"  # min(dow) -> Wed

    def test_falls_back_to_module_attr_when_config_none(self):
        """``config=None`` mirrors the other schedule_constants accessors and
        reads the module-level (monkeypatchable) LONG_FORM_DAYS."""
        day = default_publish_day(None)
        assert _DAY_NAME_TO_INT[day.lower()[:3]] in schedule_constants.LONG_FORM_DAYS


class TestOrchestratorResolution:
    def test_usa_no_lf_day_resolves_to_valid_long_form_day(self, tmp_path):
        """USA drive with no --lf-day must resolve to a weekday in
        LONG_FORM_DAYS (was "sat" -> StateInvariantError)."""
        orch = DrivePipelineOrchestrator()
        resolved = _resolved_publish_day(orch, region="usa", publish_day=None, tmp_path=tmp_path)
        assert resolved == "wed"
        assert _DAY_NAME_TO_INT[resolved] in long_form_days(orch.config)

    def test_explicit_publish_day_still_wins(self, tmp_path):
        """An explicit publish_day (--lf-day) bypasses the config default."""
        orch = DrivePipelineOrchestrator()
        resolved = _resolved_publish_day(orch, region="usa", publish_day="fri", tmp_path=tmp_path)
        assert resolved == "fri"
