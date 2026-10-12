"""Supervision releases its SQLite descriptor on successful and failed reads."""

import json
import sqlite3

import pytest

from gomoveshift.console import night_workers


@pytest.mark.parametrize("outcome", ["success", "query_error", "decode_error"])
def test_project_metadata_closes_connection_without_gc(tmp_path, monkeypatch, outcome):
    database = tmp_path / "projects.sqlite3"
    with sqlite3.connect(database) as setup:
        if outcome != "query_error":
            setup.execute("CREATE TABLE night_projects (id TEXT, brand TEXT, data TEXT)")
            payload = json.dumps({"id": "project", "brand": "brand"})
            setup.execute(
                "INSERT INTO night_projects VALUES (?, ?, ?)",
                ("project", "brand", "broken" if outcome == "decode_error" else payload),
            )
    setup.close()
    original = sqlite3.connect
    connections = []

    def connect(*args, **kwargs):
        connection = original(*args, **kwargs)
        connections.append(connection)  # Retain it so GC cannot hide missing close.
        return connection

    monkeypatch.setattr(night_workers.sqlite3, "connect", connect)
    try:
        if outcome == "success":
            assert night_workers.project_metadata(tmp_path, "brand", "project")["id"] == "project"
        else:
            with pytest.raises(
                sqlite3.OperationalError if outcome == "query_error" else ValueError
            ):
                night_workers.project_metadata(tmp_path, "brand", "project")
        assert len(connections) == 1
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connections[0].execute("SELECT 1")
    finally:
        for connection in connections:
            connection.close()
