"""Guards the autouse fixture that keeps ``Config.load()`` off the live
publishing spool (see ``tests/conftest.py::isolate_publishing_spool_from_workspace``).

A deployed ``config.yaml`` can pin ``seo.storage_path`` to a real
publishing spool -- any test reaching ``Config.load()`` without its own
override would open that live SQLite file and its schema lock (which a
different OS user may not be able to write).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from gomoveshift.config import Config
from gomoveshift.console import pubqueue
from gomoveshift.console.app import create_app


def test_config_load_never_resolves_the_live_seo_spool(tmp_path):
    config = Config.load()
    storage_path = config.seo.storage_path
    assert storage_path == str(tmp_path)


def test_publishing_tab_opens_the_isolated_spool_without_its_own_monkeypatch(tmp_path, monkeypatch):
    # Deliberately no ``monkeypatch.setattr(pubqueue.Config, "load", ...)`` --
    # this is the code path the fixture must make safe on its own.
    monkeypatch.setenv("GMS_CONSOLE_BRAND", "examplebrand")
    assert TestClient(create_app()).get("/publishing").status_code == 200
    db_path = pubqueue.queue_db_path()
    assert str(db_path).startswith(str(tmp_path))
