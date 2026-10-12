"""``TrackingStore()``'s default root must track ``GMS_OUTPUT_DIR``.

A hardcoded host-path default in ``store.py`` would be a path nothing writes
to; the pipeline writes tracking data under
``gomoveshift.paths.gms_output_dir()``. The default must resolve lazily
(read the env var at construction time, not at import time) so
``monkeypatch.setenv`` in tests — and a real operator's env — take effect.
"""

from __future__ import annotations

from gomoveshift.paths import gms_output_dir
from gomoveshift.tracking.store import TrackingStore


def test_default_root_tracks_gms_output_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_OUTPUT_DIR", str(tmp_path))
    store = TrackingStore()
    assert store.root == gms_output_dir() == tmp_path


def test_default_root_reflects_env_set_after_import(tmp_path, monkeypatch):
    """Not resolved at import time: two different env values in the same
    process produce two different defaults."""
    first = tmp_path / "first"
    second = tmp_path / "second"

    monkeypatch.setenv("GMS_OUTPUT_DIR", str(first))
    assert TrackingStore().root == first

    monkeypatch.setenv("GMS_OUTPUT_DIR", str(second))
    assert TrackingStore().root == second


def test_write_then_default_store_round_trip_finds_the_record(tmp_path, monkeypatch):
    """A record written under ``gms_output_dir()`` (the pipeline's real
    output root) is found by a default-constructed store — the read path
    every ``metrics``/``velocity``/``report`` caller uses."""
    monkeypatch.setenv("GMS_OUTPUT_DIR", str(tmp_path))

    writer = TrackingStore(root=gms_output_dir())
    writer.register_short(project="my_project", index=1, file="short_01.mp4")

    reader = TrackingStore()
    assert reader.exists("my_project")
    record = reader.load("my_project")
    assert record.shorts[0].file == "short_01.mp4"
