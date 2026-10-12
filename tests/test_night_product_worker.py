"""Typed product tasks run durably, adopt receipts, and preserve partial truth."""

import datetime as dt
import json
import sqlite3
from contextlib import contextmanager
from unittest.mock import Mock

import gomoveshift.pipeline.photos.night_products as night_products
from gomoveshift.pipeline.photos.night_collection import (
    NightCollection,
    NightFrame,
    NightSequence,
    ProductRequest,
)
from gomoveshift.pipeline.photos.night_products import ProductWaitError, process_product_project
from gomoveshift.pipeline.photos.night_project import NightStore, process_project


@contextmanager
def _admitted():
    yield True


def _store(tmp_path, products, *, budget=None):
    card = tmp_path / "card"
    card.mkdir(parents=True)
    frames = []
    for index in range(5):
        path = card / f"light-{index}.NEF"
        path.write_bytes(f"raw-{index}".encode())
        frames.append(
            NightFrame(
                path, dt.datetime(2026, 9, 14, 23, 59) + dt.timedelta(seconds=20 * index), 3200, 15
            )
        )
    dark = card / "dark.NEF"
    dark.write_bytes(b"dark")
    linears = []
    for index in range(2):
        path = card / f"stack-{index}.linear"
        path.mkdir()
        (path / "manifest.json").write_text(
            json.dumps({"schema": 4, "payloads": {"sky_sum.npy": {"shape": [10, 20, 3]}}})
        )
        linears.append(path)
    collection = NightCollection(
        root=card,
        lights=tuple(frames),
        likely_darks=(dark,),
        unreadable=(),
        sequences=(NightSequence(tuple(frames)),),
        existing_linear=tuple(linears),
        invalid_linear=(),
        refused=(),
    )
    store = NightStore(tmp_path / "projects")
    identifier = store.create_product_request(
        "brand",
        collection,
        [tmp_path],
        ProductRequest.create(products),
        storage_budget_bytes=budget,
    )
    return store, identifier


def _fake_receipt(row, task, destination, *, state="complete"):
    destination.mkdir(parents=True)
    suffix = ".mp4" if task["product"] == "timelapse" else ".tif"
    output = destination / f"{task['product']}{suffix}"
    output.write_bytes(b"product")
    receipt = {
        "state": state,
        "product": task["product"],
        "outputs": [{"path": output.name, "primary": True}],
    }
    (destination / "product-receipt.json").write_text(json.dumps(receipt))
    return receipt


def _validate(_row, task, destination):
    try:
        receipt = json.loads((destination / "product-receipt.json").read_text())
    except OSError:
        return None
    return receipt if receipt.get("product") == task["product"] else None


def test_worker_completes_each_task_once_and_resume_reuses_receipts(tmp_path):
    store, identifier = _store(tmp_path, ["best_stills", "timelapse"])
    calls = []

    def render(row, task, destination):
        calls.append(task["product"])
        return _fake_receipt(row, task, destination)

    kwargs = dict(
        render=render,
        validate=_validate,
        estimate=lambda _row, _task: 0,
        admission_context=_admitted,
    )
    assert process_product_project(store, "brand", identifier, **kwargs) == 0
    assert process_product_project(store, "brand", identifier, **kwargs) == 0

    row = store.get("brand", identifier)
    assert row["status"] == "complete"
    assert [task["status"] for task in row["tasks"]] == ["complete", "complete"]
    assert [task["attempts"] for task in row["tasks"]] == [1, 1]
    assert calls == ["best_stills", "timelapse"]


def test_worker_adopts_receipt_published_before_database_commit(tmp_path):
    store, identifier = _store(tmp_path, ["best_stills"])
    row = store.get("brand", identifier)
    orphan = store.root / identifier / "0-best_stills-orphan"
    _fake_receipt(row, row["tasks"][0], orphan)
    calls = []

    assert (
        process_product_project(
            store,
            "brand",
            identifier,
            render=lambda *_args: calls.append(True),
            validate=_validate,
            estimate=lambda _row, _task: 0,
            admission_context=_admitted,
        )
        == 0
    )
    adopted = store.get("brand", identifier)["tasks"][0]
    assert adopted["status"] == "complete" and adopted["output"] == str(orphan)
    assert calls == []


def test_partial_product_and_failed_sibling_never_report_overall_success(tmp_path):
    store, identifier = _store(tmp_path, ["best_stills", "timelapse"])

    def render(row, task, destination):
        if task["product"] == "best_stills":
            raise ValueError("one still stage failed")
        return _fake_receipt(row, task, destination, state="partial")

    assert (
        process_product_project(
            store,
            "brand",
            identifier,
            render=render,
            validate=_validate,
            estimate=lambda _row, _task: 0,
            admission_context=_admitted,
        )
        == 1
    )
    row = store.get("brand", identifier)
    assert row["status"] == "partial"
    assert [task["status"] for task in row["tasks"]] == ["failed", "partial"]
    assert "one still stage failed" in row["tasks"][0]["error"]


def test_immutable_budget_waits_before_render_and_pause_starts_nothing(tmp_path):
    store, identifier = _store(tmp_path, ["best_stills"], budget=10)
    calls = []
    kwargs = dict(
        render=lambda *_args: calls.append(True),
        validate=_validate,
        estimate=lambda _row, _task: 11,
        admission_context=_admitted,
    )
    assert process_product_project(store, "brand", identifier, **kwargs) == 2
    row = store.get("brand", identifier)
    assert row["status"] == "waiting" and row["wait_kind"] == "budget"
    assert row["tasks"][0]["attempts"] == 0 and calls == []

    store2, identifier2 = _store(tmp_path / "paused", ["best_stills"])
    store2.pause("brand", identifier2)
    assert (
        process_product_project(
            store2, "brand", identifier2, **(kwargs | {"estimate": lambda *_: 0})
        )
        == 2
    )
    assert store2.get("brand", identifier2)["status"] == "paused" and calls == []


def test_schema_two_dispatches_from_supported_project_processor(tmp_path, monkeypatch):
    store, identifier = _store(tmp_path, ["best_stills"])
    seen = []

    monkeypatch.setattr(
        "gomoveshift.pipeline.photos.night_products.process_product_project",
        lambda actual_store, brand, project: seen.append((actual_store, brand, project)) or 2,
    )
    assert process_project(store, "brand", identifier) == 2
    assert seen == [(store, "brand", identifier)]


def test_transient_resource_shortage_waits_without_consuming_an_attempt(tmp_path):
    store, identifier = _store(tmp_path, ["best_stills"])

    def short(_row, _task):
        raise ProductWaitError("Waiting for memory", kind="memory")

    assert (
        process_product_project(
            store,
            "brand",
            identifier,
            render=lambda *_args: None,
            validate=_validate,
            estimate=lambda *_args: 0,
            resource_check=short,
            admission_context=_admitted,
        )
        == 2
    )
    row = store.get("brand", identifier)
    assert row["status"] == "waiting" and row["wait_kind"] == "memory"
    assert row["tasks"][0]["attempts"] == 0


def test_running_stage_persists_worker_heartbeat_and_stage_start(tmp_path):
    store, identifier = _store(tmp_path, ["best_stills"])

    def render(row, task, destination):
        current = store.get("brand", identifier)
        assert current["heartbeat_at"] >= current["stage_started_at"]
        assert current["current_stage"] == "best_stills"
        return _fake_receipt(row, task, destination)

    assert (
        process_product_project(
            store,
            "brand",
            identifier,
            render=render,
            validate=_validate,
            estimate=lambda *_args: 0,
            admission_context=_admitted,
        )
        == 0
    )
    complete = store.get("brand", identifier)
    assert complete["current_stage"] is None
    assert complete["heartbeat_at"] <= complete["last_progress_at"]


def test_default_renderer_progress_is_persisted_and_cleared_on_completion(tmp_path, monkeypatch):
    store, identifier = _store(tmp_path, ["best_stills"])
    seen = []

    def render(row, task, destination, *, progress):
        progress(1, 2)
        current = store.get("brand", identifier)
        seen.append(current["stage_progress"])
        progress(2, 2)
        return _fake_receipt(row, task, destination)

    monkeypatch.setattr(night_products, "render_product", render)
    assert (
        process_product_project(
            store,
            "brand",
            identifier,
            render=render,
            validate=_validate,
            estimate=lambda *_args: 0,
            admission_context=_admitted,
        )
        == 0
    )
    complete = store.get("brand", identifier)
    assert seen == [{"completed": 1, "total": 2}]
    assert "stage_progress" not in complete


def test_progress_telemetry_failure_does_not_discard_completed_product(tmp_path, monkeypatch):
    store, identifier = _store(tmp_path, ["best_stills"])

    def render(row, task, destination, *, progress):
        progress(1, 1)
        return _fake_receipt(row, task, destination)

    monkeypatch.setattr(night_products, "render_product", render)
    monkeypatch.setattr(store, "progress", Mock(side_effect=sqlite3.OperationalError("busy")))

    assert (
        process_product_project(
            store,
            "brand",
            identifier,
            render=render,
            validate=_validate,
            estimate=lambda *_args: 0,
            admission_context=_admitted,
        )
        == 0
    )
    assert store.get("brand", identifier)["status"] == "complete"


def test_worker_durably_processes_requested_private_video_after_product_completion(tmp_path):
    store, identifier = _store(tmp_path, ["timelapse"])

    def render(row, task, destination):
        return _fake_receipt(row, task, destination)

    common = {
        "render": render,
        "validate": _validate,
        "estimate": lambda *_args: 0,
        "admission_context": _admitted,
    }
    assert process_product_project(store, "brand", identifier, **common) == 0
    store.request_private_export("brand", identifier, 0)
    calls = []

    def prepare(row, task_index, root):
        calls.append((task_index, root, store.get("brand", identifier)["current_stage"]))
        return {
            "export_id": "e" * 64,
            "project_id": identifier,
            "task_index": 0,
            "product": "timelapse",
            "path": "exports/keeper/selected.mp4",
            "receipt": "exports/keeper/export-receipt.json",
            "publishable": False,
        }

    assert (
        process_product_project(
            store,
            "brand",
            identifier,
            prepare_video=prepare,
            estimate_video=lambda *_args: 0,
            **common,
        )
        == 0
    )

    complete = store.get("brand", identifier)
    assert calls == [(0, store.root, "private_video")]
    assert complete["private_export_requests"][0]["status"] == "complete"
    assert complete["private_exports"][0]["export_id"] == "e" * 64


def test_product_worker_honors_durable_cancel_between_products(tmp_path):
    store, identifier = _store(tmp_path, ["best_stills", "deep_still"])

    def render(row, task, destination):
        receipt = _fake_receipt(row, task, destination)
        store.cancel("brand", identifier)
        return receipt

    assert (
        process_product_project(
            store,
            "brand",
            identifier,
            render=render,
            validate=_validate,
            estimate=lambda *_args: 0,
            admission_context=_admitted,
        )
        == 1
    )
    row = store.get("brand", identifier)
    assert row["status"] == "cancelled"
    assert row["tasks"][0]["status"] == "complete"
    assert row["tasks"][1]["attempts"] == 0
