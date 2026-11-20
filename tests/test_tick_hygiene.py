"""Hourly ``health backlog`` tick hygiene: a retired config key logs once per process, and the
release-list network read is cached for a day."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

import pytest
import yaml

from gomoveshift.config import Config
from gomoveshift.health import launch

T0 = datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc)


def test_retired_key_warns_once_per_process_across_loads(tmp_path, caplog):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"drive": {"shorts_per_lf_count": 7}}))
    with caplog.at_level(logging.WARNING, logger="gomoveshift.config"):
        Config.load(path)
        Config.load(path)
    hits = [
        r for r in caplog.records if "RETIRED_CONFIG_KEY key=shorts_per_lf_count" in r.getMessage()
    ]
    assert len(hits) == 1


def _cached(tmp_path, fetch, clock):
    return launch.cached_latest_version(tmp_path / "release-latest.json", fetch=fetch, clock=clock)


def test_one_fetch_across_two_ticks_inside_ttl(tmp_path):
    calls = []
    now = [T0]

    def fetch():
        calls.append(1)
        return "1.2.3"

    assert _cached(tmp_path, fetch, lambda: now[0])() == "1.2.3"
    now[0] = T0 + launch.LATEST_VERSION_TTL - timedelta(minutes=1)
    assert _cached(tmp_path, fetch, lambda: now[0])() == "1.2.3"
    assert len(calls) == 1


def test_fetch_again_after_ttl(tmp_path):
    seen = iter(["1.2.3", "1.3.0"])
    now = [T0]
    assert _cached(tmp_path, lambda: next(seen), lambda: now[0])() == "1.2.3"
    now[0] = T0 + launch.LATEST_VERSION_TTL + timedelta(minutes=1)
    assert _cached(tmp_path, lambda: next(seen), lambda: now[0])() == "1.3.0"
    assert json.loads((tmp_path / "release-latest.json").read_text())["version"] == "1.3.0"


def test_network_failure_falls_back_to_cache(tmp_path):
    now = [T0]
    _cached(tmp_path, lambda: "1.2.3", lambda: now[0])()
    now[0] = T0 + launch.LATEST_VERSION_TTL + timedelta(hours=1)

    def down():
        raise OSError("offline")

    assert _cached(tmp_path, down, lambda: now[0])() == "1.2.3"


def test_failure_without_cache_raises_as_before(tmp_path):
    def down():
        raise OSError("offline")

    with pytest.raises(OSError):
        _cached(tmp_path, down, lambda: T0)()
    # and stale_pin_reason still turns it into the same refusal
    reason = launch.stale_pin_reason(
        tmp_path, lambda _p: "1.0", _cached(tmp_path, down, lambda: T0)
    )
    assert reason and "cannot determine the latest framegrade release" in reason


def test_failure_with_cache_older_than_grace_raises(tmp_path):
    now = [T0]
    _cached(tmp_path, lambda: "1.2.3", lambda: now[0])()
    now[0] = T0 + launch.LATEST_VERSION_MAX_STALE + timedelta(hours=1)

    def down():
        raise OSError("offline")

    with pytest.raises(OSError):
        _cached(tmp_path, down, lambda: now[0])()


def test_corrupt_cache_is_ignored(tmp_path):
    (tmp_path / "release-latest.json").write_text("not json")
    assert _cached(tmp_path, lambda: "2.0.0", lambda: T0)() == "2.0.0"
