"""The tenant stamp covers the job store, not just the delivery queue.

`claim_storage` guarded one directory: the delivery-queue storage. The console
keeps a second one -- `console_data_dir()`, holding `jobs.sqlite3`, the job
logs and each brand's saved settings -- and nothing checked whose it was. Two
brands pointed at one data dir shared every launch, every log and every saved
setting, while their queues stayed properly separate and the existing guard
reported everything fine.

It is the same blind spot the module was written for, one directory over: the
container-side path is `/data` for every brand by design
(`docker/console-compose.yaml`), so only the host
side distinguishes them and no in-container path comparison can see a
collision. The stamp crosses the mount boundary; a path comparison does not.
"""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from gomoveshift.console import pubqueue, tenant
from gomoveshift.console.app import create_app

_STAMP = ".waypost-tenant"


def _start(tmp_path, monkeypatch, brand, *, data_dir, queue_dir):
    """Bring the console up the way a container does: brand from the
    environment, its two storage directories bound independently."""
    data_dir.mkdir(parents=True, exist_ok=True)
    queue_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("GMS_CONSOLE_BRAND", brand)
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(data_dir))
    monkeypatch.setattr(
        pubqueue.Config,
        "load",
        staticmethod(lambda: SimpleNamespace(seo=SimpleNamespace(storage_path=str(queue_dir)))),
    )
    with TestClient(create_app()) as client:
        return client.get("/healthz")


def test_starting_the_console_claims_its_job_store(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"

    _start(tmp_path, monkeypatch, "gomoveshift", data_dir=data_dir, queue_dir=tmp_path / "queue")

    assert (data_dir / _STAMP).read_text(encoding="utf-8") == "gomoveshift"


def test_a_second_brand_on_one_data_dir_is_refused(tmp_path, monkeypatch):
    data_dir = tmp_path / "shared-data"
    _start(
        tmp_path, monkeypatch, "gomoveshift", data_dir=data_dir, queue_dir=tmp_path / "queue-gms"
    )

    # The queues ARE separate here -- this is the misconfiguration the
    # delivery-queue stamp is structurally unable to see.
    with pytest.raises(tenant.TenantMismatchError) as excinfo:
        _start(
            tmp_path,
            monkeypatch,
            "examplebrand",
            data_dir=data_dir,
            queue_dir=tmp_path / "queue-other",
        )

    message = str(excinfo.value)
    assert "examplebrand" in message and "gomoveshift" in message


def test_the_same_brand_restarts_without_complaint(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    queue_dir = tmp_path / "queue"
    _start(tmp_path, monkeypatch, "gomoveshift", data_dir=data_dir, queue_dir=queue_dir)

    response = _start(tmp_path, monkeypatch, "gomoveshift", data_dir=data_dir, queue_dir=queue_dir)

    assert response.status_code == 200
    assert (data_dir / _STAMP).read_text(encoding="utf-8") == "gomoveshift"


def test_a_refused_brand_never_touches_the_jobs_table(tmp_path, monkeypatch):
    data_dir = tmp_path / "shared-data"
    _start(
        tmp_path, monkeypatch, "gomoveshift", data_dir=data_dir, queue_dir=tmp_path / "queue-gms"
    )
    (data_dir / "jobs.sqlite3").unlink(missing_ok=True)

    with pytest.raises(tenant.TenantMismatchError):
        _start(
            tmp_path,
            monkeypatch,
            "examplebrand",
            data_dir=data_dir,
            queue_dir=tmp_path / "queue-other",
        )

    # `sweep_orphaned_jobs` creates the store on first touch. A brand that was
    # refused must not have swept another tenant's jobs to "interrupted" on the
    # way out.
    assert not (data_dir / "jobs.sqlite3").exists()


def test_the_delivery_queue_is_still_claimed_too(tmp_path, monkeypatch):
    queue_dir = tmp_path / "queue"

    _start(tmp_path, monkeypatch, "gomoveshift", data_dir=tmp_path / "data", queue_dir=queue_dir)

    # The new claim is an addition, not a move: both directories are stamped.
    assert (queue_dir / _STAMP).read_text(encoding="utf-8") == "gomoveshift"
