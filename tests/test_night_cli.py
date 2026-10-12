"""Private night worker exit codes and explicit control operations."""

import json
from argparse import Namespace
from unittest.mock import Mock

from gomoveshift.cli.handlers import night


def args(tmp_path, **overrides):
    values = dict(
        workspace=str(tmp_path),
        brand="test",
        project="project",
        sources=None,
        scan=None,
        collection=None,
        products=None,
        sequence_index=0,
        source_root=[],
        recipe="natural",
        pause=False,
        cancel=False,
        resume=False,
        status=False,
        wait_seconds=0,
        poll_seconds=15,
    )
    return Namespace(**(values | overrides))


def test_scan_outputs_collection_without_opening_project_store(tmp_path, monkeypatch, capsys):
    collection = Mock()
    collection.as_dict.return_value = {"light_count": 5, "sequences": []}
    scan = Mock(return_value=collection)
    monkeypatch.setattr(night, "scan_collection", scan)
    store = Mock()
    monkeypatch.setattr(night, "NightStore", store)

    assert night.cmd_night(args(tmp_path, project=None, scan=str(tmp_path))) == 0

    assert json.loads(capsys.readouterr().out) == {"light_count": 5, "sequences": []}
    scan.assert_called_once_with(tmp_path)
    store.assert_not_called()


def test_collection_products_create_typed_request_and_use_supported_worker(tmp_path, monkeypatch):
    collection = Mock()
    scan = Mock(return_value=collection)
    store = Mock()
    store.create_product_request.return_value = "typed-project"
    monkeypatch.setattr(night, "scan_collection", scan)
    monkeypatch.setattr(night, "NightStore", lambda _root: store)
    process = Mock(return_value=0)
    monkeypatch.setattr(night, "process_project", process)

    assert (
        night.cmd_night(
            args(
                tmp_path,
                project=None,
                collection=str(tmp_path / "card"),
                products=["best_stills", "timelapse"],
                source_root=[str(tmp_path)],
                sequence_index=2,
            )
        )
        == 0
    )
    scan.assert_called_once_with(tmp_path / "card")
    request = store.create_product_request.call_args.args[3]
    assert [product.value for product in request.products] == ["best_stills", "timelapse"]
    assert store.create_product_request.call_args.kwargs["sequence_index"] == 2
    process.assert_called_once_with(store, "test", "typed-project")


def test_collection_products_are_required_and_refused_with_legacy_sources(tmp_path, monkeypatch):
    store = Mock()
    monkeypatch.setattr(night, "NightStore", lambda _root: store)
    assert (
        night.cmd_night(
            args(tmp_path, project=None, collection=str(tmp_path), source_root=[str(tmp_path)])
        )
        == 1
    )
    assert (
        night.cmd_night(
            args(
                tmp_path,
                project=None,
                sources=[str(tmp_path)],
                products=["best_stills"],
                source_root=[str(tmp_path)],
            )
        )
        == 1
    )
    store.create_product_request.assert_not_called()


def test_waiting_is_not_reported_as_success(tmp_path, monkeypatch):
    store = Mock()
    store.get.return_value = {"pause_requested": False, "status": "waiting"}
    monkeypatch.setattr(night, "NightStore", lambda root: store)
    process = Mock(return_value=2)
    monkeypatch.setattr(night, "process_project", process)
    assert night.cmd_night(args(tmp_path)) == 2
    process.assert_called_once()


def test_pause_and_status_never_render(tmp_path, monkeypatch):
    store = Mock()
    store.get.return_value = {"status": "paused"}
    monkeypatch.setattr(night, "NightStore", lambda root: store)
    process = Mock()
    monkeypatch.setattr(night, "process_project", process)
    assert night.cmd_night(args(tmp_path, pause=True)) == 0
    store.pause.assert_called_once_with("test", "project")
    assert night.cmd_night(args(tmp_path, status=True)) == 0
    process.assert_not_called()


def test_cancel_is_durable_and_resume_clears_both_control_flags(tmp_path, monkeypatch):
    store = Mock()
    store.get.return_value = {"pause_requested": False, "status": "cancelled"}
    monkeypatch.setattr(night, "NightStore", lambda root: store)
    process = Mock(return_value=0)
    monkeypatch.setattr(night, "process_project", process)

    assert night.cmd_night(args(tmp_path, cancel=True)) == 0
    store.cancel.assert_called_once_with("test", "project")
    process.assert_not_called()

    assert night.cmd_night(args(tmp_path, resume=True)) == 0
    store.pause.assert_called_once_with("test", "project", requested=False)
    store.cancel.assert_called_with("test", "project", requested=False)
    process.assert_called_once()


def test_missing_source_roots_refused_before_create(tmp_path, monkeypatch):
    store = Mock()
    monkeypatch.setattr(night, "NightStore", lambda root: store)
    assert night.cmd_night(args(tmp_path, project=None, sources=["/untrusted"])) == 1
    store.create.assert_not_called()


def test_resume_rechecks_paused_after_waiting_pass(tmp_path, monkeypatch):
    store = Mock()
    store.get.return_value = {"pause_requested": True}
    monkeypatch.setattr(night, "NightStore", lambda root: store)
    monkeypatch.setattr(night, "process_project", Mock(return_value=2))
    sleep = Mock()
    monkeypatch.setattr(night.time, "sleep", sleep)
    assert night.cmd_night(args(tmp_path, resume=True, wait_seconds=60)) == 2
    store.pause.assert_called_once_with("test", "project", requested=False)
    sleep.assert_not_called()


def test_invalid_poll_and_conflicting_control_flags_fail(tmp_path):
    assert night.cmd_night(args(tmp_path, poll_seconds=0)) == 1
    assert night.cmd_night(args(tmp_path, pause=True, resume=True)) == 1


def test_new_project_passes_gib_budget_as_bytes(tmp_path, monkeypatch):
    store = Mock()
    monkeypatch.setattr(night, "NightStore", lambda root: store)
    monkeypatch.setattr(night, "process_project", Mock(return_value=0))
    assert (
        night.cmd_night(
            args(
                tmp_path,
                project=None,
                sources=[str(tmp_path)],
                source_root=[str(tmp_path)],
                storage_budget_gib="1.5",
            )
        )
        == 0
    )
    assert store.create.call_args.kwargs["storage_budget_bytes"] == 1610612736


def test_existing_project_rejects_budget_change_before_work(tmp_path, monkeypatch):
    process = Mock(return_value=0)
    monkeypatch.setattr(night, "process_project", process)
    assert night.cmd_night(args(tmp_path, storage_budget_gib="1")) == 1
    process.assert_not_called()


def test_invalid_creation_budget_fails_before_store_or_worker(tmp_path, monkeypatch):
    store, process = Mock(), Mock()
    monkeypatch.setattr(night, "NightStore", store)
    monkeypatch.setattr(night, "process_project", process)
    assert (
        night.cmd_night(
            args(
                tmp_path,
                project=None,
                sources=[str(tmp_path)],
                source_root=[str(tmp_path)],
                storage_budget_gib="nan",
            )
        )
        == 1
    )
    store.assert_not_called()
    process.assert_not_called()
