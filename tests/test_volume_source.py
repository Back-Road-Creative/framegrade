"""The pipeline's shorts-per-LF number is Waypost's decision, nothing else."""

import logging
from pathlib import Path

import pytest
from seo_manager.analytics import cadence_decisions as cd
from seo_manager.analytics import volume_policy as policy

from gomoveshift.pipeline import volume_source


def _write(db: Path, value: int) -> None:
    from datetime import datetime, timezone

    cd.append(
        db,
        knob="shorts_per_lf",
        region=None,
        value=value,
        prev=policy.SHORTS_FLOOR,
        status="ok",
        source="policy",
        reason_code="TEST",
        basis="test",
        applied_at=datetime(2026, 10, 1, tzinfo=timezone.utc),
    )


def test_empty_store_returns_seed_floor_source_seed(tmp_path):
    d = volume_source.shorts_per_lf(tmp_path / "cadence_decisions.sqlite3")
    assert d.value == policy.SHORTS_FLOOR
    assert d.source == "seed"


def test_decided_value_is_returned(tmp_path):
    db = tmp_path / "cadence_decisions.sqlite3"
    _write(db, 9)
    d = volume_source.shorts_per_lf(db)
    assert (d.value, d.source) == (9, "policy")


def test_unreadable_decision_store_raises_and_plans_nothing(tmp_path, caplog):
    db = tmp_path / "cadence_decisions.sqlite3"
    db.write_bytes(b"not a sqlite database at all" * 50)
    with caplog.at_level(logging.ERROR, logger=volume_source.logger.name):
        with pytest.raises(volume_source.VolumeSourceUnreadableError):
            volume_source.shorts_per_lf(db)
    assert any("VOLUME_SOURCE_UNREADABLE" in m and "planned=0" in m for m in caplog.messages)


def test_shorts_target_flag_only_lowers_and_logs_VOLUME_OVERRIDE_LOWER(caplog):
    with caplog.at_level(logging.INFO, logger=volume_source.logger.name):
        assert volume_source.apply_flag(9, 4) == 4
        assert any("VOLUME_OVERRIDE_LOWER decided=9 flag=4" in m for m in caplog.messages)
        caplog.clear()
        assert volume_source.apply_flag(9, 20) == 9
        assert volume_source.apply_flag(9, None) == 9
        assert volume_source.apply_flag(9, 9) == 9
    assert not any("VOLUME_OVERRIDE_LOWER" in m for m in caplog.messages)
