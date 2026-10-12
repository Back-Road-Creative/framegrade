"""Shared, cross-run Overpass response cache.

The chapter path already cached Overpass responses, but into the RUN's
``stages_dir`` — so re-processing the same footage into a fresh
``--output-dir`` started cold — and ``destination_resolver``'s POI source
called ``_query_overpass`` directly with no cache at all. Re-running one
drive for LFs, shorts and superlong therefore re-queried the same bboxes
every time, and the public endpoint answered with HTTP 429 (observed on a long run).

The cache also keyed on the BBOX alone while the query text embeds
``named_road_patterns`` — two different pattern sets over one bbox
collided on a single cache file. Keying on the full query string closes
that with the same change.

Every test here patches ``urllib.request.urlopen`` and so reaches the real
``_query_overpass``; that bypasses the module-attr lookup the conftest
``hermetic_overpass`` stub replaces, hence ``@pytest.mark.real_overpass``.
"""

import json
import os
import time
from unittest.mock import patch

import pytest

from gomoveshift.pipeline.chapters.overpass_source import _query_overpass

pytestmark = pytest.mark.real_overpass

BBOX = (51.8, -9.6, 52.1, -9.2)
_RESPONSE = {"elements": [{"type": "node", "id": 1, "tags": {"name": "Kenmare"}}]}


class _FakeResponse:
    """Minimal ``urlopen`` context-manager stand-in."""

    status = 200

    def __init__(self, payload: dict):
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """Point the shared Overpass cache at a throwaway directory."""
    path = tmp_path / "overpass-cache"
    monkeypatch.setenv("GMS_OVERPASS_CACHE_DIR", str(path))
    return path


def _fake_urlopen(_req, timeout=None):
    return _FakeResponse(_RESPONSE)


class TestSharedOverpassCache:
    def test_successful_query_writes_a_cache_entry(self, cache_dir):
        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            assert _query_overpass(BBOX) == _RESPONSE
        entries = list(cache_dir.glob("*.json"))
        assert len(entries) == 1, f"expected one cache entry, got {entries}"

    def test_second_query_is_served_from_cache_without_network(self, cache_dir):
        with patch("urllib.request.urlopen", side_effect=_fake_urlopen) as mocked:
            _query_overpass(BBOX)
            _query_overpass(BBOX)
        assert mocked.call_count == 1, "second identical query must not hit the network"

    def test_different_road_patterns_do_not_share_a_cache_entry(self, cache_dir):
        """The pre-fix key was the bbox alone, so these two collided."""
        with patch("urllib.request.urlopen", side_effect=_fake_urlopen) as mocked:
            _query_overpass(BBOX, ["Ring of Kerry"])
            _query_overpass(BBOX, ["Healy Pass"])
        assert mocked.call_count == 2, "distinct queries must not share one cache entry"
        assert len(list(cache_dir.glob("*.json"))) == 2

    def test_expired_entry_is_refetched(self, cache_dir):
        from gomoveshift.pipeline.chapters import overpass_source

        with patch("urllib.request.urlopen", side_effect=_fake_urlopen) as mocked:
            _query_overpass(BBOX)
            entry = next(iter(cache_dir.glob("*.json")))
            stale = time.time() - (overpass_source.OVERPASS_CACHE_TTL_SEC + 60)
            os.utime(entry, (stale, stale))
            _query_overpass(BBOX)
        assert mocked.call_count == 2, "an entry past its TTL must be refetched"

    def test_corrupt_entry_falls_back_to_the_network(self, cache_dir):
        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            _query_overpass(BBOX)
        entry = next(iter(cache_dir.glob("*.json")))
        entry.write_text("{ this is not json")
        with patch("urllib.request.urlopen", side_effect=_fake_urlopen) as mocked:
            assert _query_overpass(BBOX) == _RESPONSE
        assert mocked.call_count == 1, "a corrupt entry must degrade to a live query"

    def test_empty_cache_dir_setting_disables_caching(self, monkeypatch):
        """Escape hatch: an explicitly empty setting means "never cache"."""
        monkeypatch.setenv("GMS_OVERPASS_CACHE_DIR", "")
        with patch("urllib.request.urlopen", side_effect=_fake_urlopen) as mocked:
            _query_overpass(BBOX)
            _query_overpass(BBOX)
        assert mocked.call_count == 2

    def test_failed_query_is_never_cached(self, cache_dir):
        """A raising query must leave no entry to poison the next run."""
        with patch("urllib.request.urlopen", side_effect=OSError("boom")):
            with pytest.raises(RuntimeError):
                _query_overpass(BBOX)
        assert list(cache_dir.glob("*.json")) == []
