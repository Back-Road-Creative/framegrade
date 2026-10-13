"""Regression test for schedule_constants.apply_config()
must reach LongFormMetadataGenerator.get_next_publish_slot AND
ContentLedger.get_next_available_slot.

Bug: both modules originally did `from .schedule_constants import LONG_FORM_DAYS`
which captured an import-time snapshot. apply_config()'s globals() rebind on
schedule_constants couldn't reach across module boundaries, so --lf-day mon
shipped LF metadata with next_publish.date on Wednesday (the config default).

Fix: import the module (`from . import schedule_constants as _schedule_constants`)
and reference attributes at call time, so the apply_config mutation is visible.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from gomoveshift.config import get_config
from gomoveshift.pipeline import schedule_constants as _schedule_constants
from gomoveshift.pipeline.content_ledger import ContentLedger
from gomoveshift.pipeline.longform_metadata import LongFormMetadataGenerator


def _next_monday_after(now: datetime) -> str:
    days_ahead = (0 - now.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")


def test_longform_metadata_get_next_publish_slot_reflects_apply_config(monkeypatch):
    """apply_config() with --lf-day mon must surface as Monday in LongForm slot."""
    config = get_config()
    config.drive.long_form_publish_days = [0]  # Mon
    _schedule_constants.apply_config(config)

    # No ledger → use the naive fallback path that reads LONG_FORM_DAYS.
    gen = LongFormMetadataGenerator.__new__(LongFormMetadataGenerator)
    gen._ledger = None

    slot = gen.get_next_publish_slot("longform")
    assert slot["day_name"] == "Monday", f"expected Monday, got {slot}"


def test_content_ledger_get_next_available_slot_reflects_apply_config(monkeypatch, waypost_lf_days):
    """A Monday long-form day must surface as Monday in the ledger slot. The day is
    Waypost's now (``lf_days`` decisions); apply_config still carries the run's pick."""
    waypost_lf_days({"ireland": (0,), "usa": (0,)})
    config = get_config()
    config.drive.long_form_publish_days = [0]  # Mon
    _schedule_constants.apply_config(config)

    ledger = ContentLedger()

    slot = ledger.get_next_available_slot("longform")
    assert slot is not None
    assert slot["day_name"] == "Monday", f"expected Monday, got {slot}"


def test_apply_config_then_revert_to_default(monkeypatch):
    """Verify the rebind is reversible — apply_config to Wed restores default."""
    config = get_config()
    config.drive.long_form_publish_days = [2]  # Wed (default)
    _schedule_constants.apply_config(config)

    gen = LongFormMetadataGenerator.__new__(LongFormMetadataGenerator)
    gen._ledger = None

    slot = gen.get_next_publish_slot("longform")
    assert slot["day_name"] == "Wednesday", f"expected Wednesday, got {slot}"
