"""Durable requested products bind prerequisites and source identity before work."""

import datetime as dt
import json

import pytest

from gomoveshift.pipeline.photos.night_collection import (
    NightCollection,
    NightFrame,
    NightProduct,
    NightSequence,
    ProductRequest,
)
from gomoveshift.pipeline.photos.night_project import NightStore


def _linear(root, name):
    path = root / f"{name}.linear"
    path.mkdir()
    (path / "manifest.json").write_text(json.dumps({"schema": 4, "payloads": {}}))
    return path


def _collection(tmp_path, *, linear_count=2, frame_count=5):
    root = tmp_path / "card"
    root.mkdir()
    started = dt.datetime(2026, 9, 14, 23, 59)
    frames = []
    for index in range(frame_count):
        path = root / f"DSC_{index:04d}.NEF"
        path.write_bytes(f"raw-{index}".encode())
        frames.append(NightFrame(path, started + dt.timedelta(seconds=20 * index), 3200, 15.0))
    dark = root / "darks" / "dark.NEF"
    dark.parent.mkdir()
    dark.write_bytes(b"dark")
    linears = tuple(_linear(root, f"stack-{index}") for index in range(linear_count))
    sequence = NightSequence(tuple(frames))
    return NightCollection(
        root=root,
        lights=tuple(frames),
        likely_darks=(dark,),
        unreadable=(),
        sequences=(sequence,),
        existing_linear=linears,
        invalid_linear=(),
        refused=(),
    )


def test_product_request_persists_typed_tasks_selected_sequence_and_input_identity(tmp_path):
    collection = _collection(tmp_path)
    store = NightStore(tmp_path / "projects")
    request = ProductRequest.create(["best_stills", "timelapse", "star_trails", "deep_still"])

    identifier = store.create_product_request(
        "brand", collection, [tmp_path], request, "natural", sequence_index=0
    )
    duplicate = store.create_product_request(
        "brand", collection, [tmp_path], request, "natural", sequence_index=0
    )
    row = store.get("brand", identifier)

    assert duplicate == identifier
    assert row["schema"] == 2 and row["kind"] == "night_products"
    assert row["requested_products"] == [
        "best_stills",
        "timelapse",
        "star_trails",
        "deep_still",
    ]
    assert [task["product"] for task in row["tasks"]] == row["requested_products"]
    assert all(task["status"] == "queued" for task in row["tasks"])
    assert row["capture"]["started_at"].startswith("2026-09-14")
    assert row["capture"]["ended_at"].startswith("2026-09-15")
    assert len(row["inputs"]["lights"]) == 5
    assert len(row["inputs"]["darks"]) == 1
    assert len(row["inputs"]["linear"]) == 2
    assert set(row["inputs"]["lights"][0]) >= {
        "path",
        "size",
        "mtime_ns",
        "device",
        "inode",
    }


@pytest.mark.parametrize(
    ("products", "linear_count", "frame_count", "message"),
    [
        ([NightProduct.BEST_STILLS], 0, 5, "best_stills requires"),
        ([NightProduct.DEEP_STILL], 1, 5, "deep_still requires at least two"),
        ([NightProduct.TIMELAPSE], 2, 0, "sequence"),
        ([NightProduct.STAR_TRAILS], 2, 1, "sequence"),
    ],
)
def test_product_request_rejects_missing_prerequisites(
    tmp_path, products, linear_count, frame_count, message
):
    collection = _collection(tmp_path, linear_count=linear_count, frame_count=frame_count)
    store = NightStore(tmp_path / "projects")
    with pytest.raises(ValueError, match=message):
        store.create_product_request(
            "brand", collection, [tmp_path], ProductRequest.create(products), "natural"
        )
    assert store.list("brand") == []


def test_product_request_identity_changes_when_selected_raw_changes(tmp_path):
    collection = _collection(tmp_path)
    store = NightStore(tmp_path / "projects")
    request = ProductRequest.create([NightProduct.TIMELAPSE])
    first = store.create_product_request("brand", collection, [tmp_path], request, "accent")
    selected = collection.sequences[0].frames[0].path
    selected.write_bytes(b"changed-selected-input")
    second = store.create_product_request("brand", collection, [tmp_path], request, "accent")
    assert second != first


def test_product_request_refuses_cross_root_and_symlink_inputs(tmp_path):
    collection = _collection(tmp_path)
    store = NightStore(tmp_path / "projects")
    request = ProductRequest.create([NightProduct.TIMELAPSE])
    with pytest.raises(ValueError, match="configured roots"):
        store.create_product_request("brand", collection, [tmp_path / "elsewhere"], request)

    target = collection.sequences[0].frames[0].path
    link = collection.root / "linked.NEF"
    link.symlink_to(target)
    linked = NightFrame(link, collection.lights[0].captured_at, 3200, 15.0)
    unsafe = NightCollection(
        root=collection.root,
        lights=(linked,),
        likely_darks=collection.likely_darks,
        unreadable=(),
        sequences=(NightSequence((linked, collection.lights[1])),),
        existing_linear=collection.existing_linear,
        invalid_linear=(),
        refused=(),
    )
    with pytest.raises(ValueError, match="regular files"):
        store.create_product_request("brand", unsafe, [tmp_path], request)


def test_private_export_record_is_idempotent_and_brand_scoped(tmp_path):
    collection = _collection(tmp_path)
    store = NightStore(tmp_path / "projects")
    identifier = store.create_product_request(
        "brand", collection, [tmp_path], ProductRequest.create([NightProduct.BEST_STILLS])
    )
    row = store.get("brand", identifier)
    row["tasks"][0].update(status="complete", output=str(store.root / identifier / "attempt"))
    store.save(row)
    record = {
        "export_id": "e" * 64,
        "project_id": identifier,
        "task_index": 0,
        "product": "best_stills",
        "path": "exports/keeper/selected.jpg",
        "receipt": "exports/keeper/export-receipt.json",
        "publishable": False,
    }

    store.add_private_export("brand", identifier, record)
    store.add_private_export("brand", identifier, record)

    assert store.get("brand", identifier)["private_exports"] == [record]
    with pytest.raises(KeyError):
        store.add_private_export("foreign", identifier, record)


def test_stale_worker_save_preserves_concurrent_export_and_request(tmp_path):
    collection = _collection(tmp_path)
    store = NightStore(tmp_path / "projects")
    identifier = store.create_product_request(
        "brand", collection, [tmp_path], ProductRequest.create([NightProduct.TIMELAPSE])
    )
    row = store.get("brand", identifier)
    row["tasks"][0].update(status="complete", output=str(store.root / identifier / "attempt"))
    store.save(row)
    stale = store.get("brand", identifier)
    record = {
        "export_id": "e" * 64,
        "project_id": identifier,
        "task_index": 0,
        "product": "timelapse",
        "path": "exports/keeper/selected.mp4",
        "receipt": "exports/keeper/export-receipt.json",
        "publishable": False,
    }

    store.add_private_export("brand", identifier, record)
    request = store.request_private_export("brand", identifier, 0)
    stale["status"] = "complete"
    store.save(stale)

    saved = store.get("brand", identifier)
    assert saved["private_exports"] == [record]
    assert saved["private_export_requests"] == [request]


def test_progress_updates_only_the_exact_running_worker(tmp_path):
    collection = _collection(tmp_path)
    store = NightStore(tmp_path / "projects")
    identifier = store.create_product_request(
        "brand", collection, [tmp_path], ProductRequest.create([NightProduct.BEST_STILLS])
    )
    row = store.get("brand", identifier)
    worker = {"pid": 1, "started_at": "2", "boot_id": "3"}
    row.update(status="running", worker=worker, current_stage="best_stills")
    store.save(row)

    assert store.progress("brand", identifier, worker, "best_stills", 1, 2, now=100)
    assert not store.progress("brand", identifier, worker | {"pid": 9}, "best_stills", 2, 2)

    current = store.get("brand", identifier)
    assert current["heartbeat_at"] == current["last_progress_at"] == 100
    assert current["stage_progress"] == {"completed": 1, "total": 2}


def test_private_video_request_is_durable_deduplicated_and_explicitly_retryable(tmp_path):
    collection = _collection(tmp_path)
    store = NightStore(tmp_path / "projects")
    identifier = store.create_product_request(
        "brand", collection, [tmp_path], ProductRequest.create([NightProduct.TIMELAPSE])
    )
    row = store.get("brand", identifier)
    row["tasks"][0].update(status="complete", output=str(store.root / identifier / "attempt"))
    row["status"] = "complete"
    store.save(row)

    first = store.request_private_export("brand", identifier, 0)
    duplicate = store.request_private_export("brand", identifier, 0)

    assert duplicate == first
    queued = store.get("brand", identifier)
    assert queued["status"] == "queued"
    assert queued["private_export_requests"] == [first]

    first.update(status="failed", error="detector unavailable", attempts=1)
    queued["private_export_requests"] = [first]
    queued["status"] = "complete"
    store.save(queued)
    retried = store.request_private_export("brand", identifier, 0)
    assert retried["status"] == "queued" and retried["attempts"] == 1
    assert "error" not in retried
