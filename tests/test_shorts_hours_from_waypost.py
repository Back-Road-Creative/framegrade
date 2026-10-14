"""GMS asks Waypost for its shorts hours; the hand-written tables are not consulted.

A region's Waypost list is every hour its shorts post at (packaged seed in ``brand.yaml``,
replaced by the latest ``shorts_times`` decision row). Each long-form day's drive takes its
share of it (``seo_manager.scheduling.shorts_times.drive_share``).
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pytest
from seo_manager.analytics import cadence_decisions

from gomoveshift.config import Config
from gomoveshift.pipeline import schedule_constants as sc
from gomoveshift.pipeline import seo_slots
from gomoveshift.pipeline.volume_source import DB_NAME

SEED_CADENCE = True  # opt out of conftest.legacy_test_cadence
APPLIED_AT = datetime(2026, 10, 9, 12, 0)


def _config(tmp_path: Path) -> Config:
    config = Config()
    config.seo.storage_path = str(tmp_path)
    return config


def _decide_shorts(tmp_path: Path, region: str, hours: list[str]) -> None:
    cadence_decisions.append(
        tmp_path / DB_NAME,
        knob="shorts_times",
        region=region,
        value=hours,
        prev=hours,
        status="ok",
        source="policy",
        reason_code="HOUR_SWAP",
        basis="test fixture",
        applied_at=APPLIED_AT,
    )


def _decide_days(tmp_path: Path, region: str, dow: list[int]) -> None:
    cadence_decisions.append(
        tmp_path / DB_NAME,
        knob="lf_days",
        region=region,
        value=len(dow),
        prev=len(dow),
        status="ok",
        source="policy",
        reason_code="TAIL_UP",
        basis="test fixture",
        applied_at=APPLIED_AT,
        evidence={"dow": sorted(dow)},
    )


def test_seed_reproduces_the_retired_table(tmp_path):
    config = _config(tmp_path)
    assert sc.resolve_shorts_times_for_region(config, "usa", "tue") == ["14:00", "18:00"]
    assert sc.resolve_shorts_times_for_region(config, "usa", "fri") == ["10:00", "16:00"]
    assert sc.resolve_shorts_times_for_region(config, "usa", None) == ["14:00", "18:00"]
    assert sc.resolve_shorts_times_for_region(config, "ireland", "sun") == ["08:00", "12:00"]
    assert sc.resolve_shorts_times_for_region(config, "ireland") == ["08:00", "12:00"]
    assert sc.resolve_shorts_pool_for_region(config, "usa") == [
        "10:00",
        "14:00",
        "16:00",
        "18:00",
    ]
    assert sc.resolve_shorts_pool_for_region(config, "ireland") == ["08:00", "12:00"]


def test_resolve_shorts_times_reads_decision(tmp_path):
    config = _config(tmp_path)
    _decide_shorts(tmp_path, "usa", ["09:00", "11:00", "13:00", "17:00"])
    # usa days are tue + fri: the last LF day (fri) takes the earliest windows
    assert sc.resolve_shorts_times_for_region(config, "usa", "fri") == ["09:00", "13:00"]
    assert sc.resolve_shorts_times_for_region(config, "usa", "tue") == ["11:00", "17:00"]
    assert sc.resolve_shorts_pool_for_region(config, "usa") == [
        "09:00",
        "11:00",
        "13:00",
        "17:00",
    ]
    # ireland has no decision: its seed stands
    assert sc.resolve_shorts_times_for_region(config, "ireland") == ["08:00", "12:00"]


def test_changed_lf_days_changes_each_drives_share(tmp_path):
    config = _config(tmp_path)
    assert sc.resolve_shorts_times_for_region(config, "usa", "tue") == ["14:00", "18:00"]
    _decide_days(tmp_path, "usa", [0, 2])  # mon + wed
    assert sc.resolve_shorts_times_for_region(config, "usa", "mon") == ["14:00", "18:00"]
    assert sc.resolve_shorts_times_for_region(config, "usa", "wed") == ["10:00", "16:00"]
    # tue is off the new cadence: it takes the first LF day's share
    assert sc.resolve_shorts_times_for_region(config, "usa", "tue") == ["14:00", "18:00"]


def test_unreadable_store_halts_with_cadence_source_unreadable(tmp_path, caplog):
    config = _config(tmp_path)
    (tmp_path / DB_NAME).write_bytes(b"this is not a sqlite database" * 50)
    with caplog.at_level(logging.ERROR):
        with pytest.raises(
            seo_slots.CadenceSourceUnreadableError, match="CADENCE_SOURCE_UNREADABLE"
        ):
            sc.resolve_shorts_times_for_region(config, "usa", "tue")
        with pytest.raises(
            seo_slots.CadenceSourceUnreadableError, match="CADENCE_SOURCE_UNREADABLE"
        ):
            sc.resolve_shorts_pool_for_region(config, "usa")


def test_explicit_shorts_times_flag_still_wins(tmp_path):
    config = _config(tmp_path)
    assert sc.resolve_shorts_times(config, "tue", "07:00, 21:00", "usa") == ["07:00", "21:00"]
    assert sc.resolve_shorts_times(config, "tue", None, "usa") == ["14:00", "18:00"]


def test_an_unknown_region_halts_instead_of_borrowing_another_list(tmp_path):
    config = _config(tmp_path)
    with pytest.raises(
        seo_slots.CadenceSourceUnreadableError, match="region mars has no shorts hours"
    ):
        seo_slots.shorts_windows(config, "mars")
    with pytest.raises(
        seo_slots.CadenceSourceUnreadableError, match="region mars has no shorts hours"
    ):
        seo_slots.shorts_pool(config, "mars")


def test_no_region_keeps_the_flat_list(tmp_path):
    config = _config(tmp_path)
    assert sc.resolve_shorts_times(config, None, None, None) == list(
        config.drive.shorts_publish_times
    )
