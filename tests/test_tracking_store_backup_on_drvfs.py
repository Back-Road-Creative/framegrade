"""``TrackingStore.save`` must back up the previous record on a mount that
refuses ``copystat`` (drvfs under ``/mnt/<letter>``).

``save`` once backed the old
record up with ``shutil.copy2``, whose ``copystat`` raised
``PermissionError: [Errno 1] Operation not permitted`` on a drvfs mount
and the refreshed record was never written, so a scheduled metrics run
collected nothing. A recovery backup needs the
bytes, not preserved timestamps or mode bits — the same lesson
``pipeline/redrive_ost.py`` already records for the drive-state backup.
"""

from __future__ import annotations

import shutil

from gomoveshift.tracking.schema import TrackingRecord
from gomoveshift.tracking.store import TrackingStore


def _refuse_stat(*_args, **_kwargs):
    raise PermissionError(1, "Operation not permitted")


def test_second_save_succeeds_when_copystat_is_refused(tmp_path, monkeypatch):
    store = TrackingStore(root=tmp_path)
    record = TrackingRecord(project="proj", location="Ireland")
    store.save(record)

    monkeypatch.setattr(shutil, "copystat", _refuse_stat)
    monkeypatch.setattr(shutil, "copymode", _refuse_stat)

    record.location = "Ireland (refreshed)"
    store.save(record)

    assert store.load("proj").location == "Ireland (refreshed)"
    backups = list((tmp_path / "proj" / "tracking" / "backups").glob("proj_*.json"))
    assert len(backups) == 1
    assert '"Ireland"' in backups[0].read_text()
