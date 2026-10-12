"""Storage admission reserves new outputs without deleting or blocking valid reuse."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gomoveshift.pipeline.photos import night_project as module

MIB = 1024**2


def setup_project(tmp_path, budget=None, shape=None):
    source = tmp_path / "source.linear"
    source.mkdir(exist_ok=True)
    manifest = {"payloads": {"sky_sum.npy": {"shape": shape or [10, 20, 3]}}}
    (source / "manifest.json").write_text(json.dumps(manifest))
    store = module.NightStore(tmp_path / "projects")
    identifier = store.create("brand", [source], [tmp_path], "natural", storage_budget_bytes=budget)
    return store, identifier


def run(store, identifier, **kwargs):
    return module.process_project(store, "brand", identifier, admitted=lambda _: True, **kwargs)


def test_budget_is_immutable_identity_and_none_preserves_default(tmp_path):
    store, first = setup_project(tmp_path)
    source = tmp_path / "source.linear"
    assert store.create("brand", [source], [tmp_path], "natural") == first
    _, second = setup_project(tmp_path, 50 * MIB)
    _, third = setup_project(tmp_path, 60 * MIB)
    assert len({first, second, third}) == 3
    assert store.get("brand", second)["storage_budget_bytes"] == 50 * MIB


@pytest.mark.parametrize("budget", [0, -1, True, 1.5, "10"])
def test_invalid_budget_refused(tmp_path, budget):
    with pytest.raises(ValueError, match="budget"):
        setup_project(tmp_path, budget)


def test_cap_shortage_waits_before_decode_or_write(tmp_path):
    store, identifier = setup_project(tmp_path, 1)
    render, validate, recover = Mock(), Mock(), Mock()
    assert run(store, identifier, render=render, validate=validate, recover=recover) == 2
    row = store.get("brand", identifier)
    assert row["status"] == "waiting"
    assert "budget" in row["reason"]
    assert row["wait_kind"] == "budget"
    assert row["items"][0]["attempts"] == 0
    render.assert_not_called()
    validate.assert_not_called()
    recover.assert_not_called()


def test_free_space_reserves_dimension_sized_outputs(tmp_path, monkeypatch):
    store, identifier = setup_project(tmp_path, shape=[10000, 10000, 3])
    monkeypatch.setattr(module.shutil, "disk_usage", lambda _: SimpleNamespace(free=1100 * MIB))
    render, validate = Mock(), Mock()
    assert run(store, identifier, render=render, validate=validate) == 2
    assert "free space" in store.get("brand", identifier)["reason"]
    assert store.get("brand", identifier)["wait_kind"] == "disk"
    render.assert_not_called()
    validate.assert_not_called()


def test_completed_receipt_reuses_at_cap_without_free_space(tmp_path, monkeypatch):
    store, identifier = setup_project(tmp_path, 10)
    attempt = store.root / identifier / "0-complete"
    attempt.mkdir(parents=True)
    (attempt / "receipt.json").write_bytes(b"1234567890")
    monkeypatch.setattr(module.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    render, validate = Mock(), Mock(return_value=True)
    assert run(store, identifier, render=render, validate=validate) == 0
    validate.assert_called_once()
    render.assert_not_called()


def test_partial_attempts_consume_budget_and_remain_preserved(tmp_path):
    store, identifier = setup_project(tmp_path, 20 * MIB)
    attempt = store.root / identifier / "0-partial"
    attempt.mkdir(parents=True)
    master = attempt / "master.tif"
    with master.open("wb") as stream:
        stream.truncate(10 * MIB)
    render = Mock()
    assert run(store, identifier, render=render, validate=lambda *a: False) == 2
    assert master.stat().st_size == 10 * MIB
    render.assert_not_called()


def test_preview_recovery_reserves_only_jpeg_and_preserves_master(tmp_path, monkeypatch):
    store, identifier = setup_project(tmp_path, 17 * MIB, shape=[1000, 1000, 3])
    attempt = store.root / identifier / "0-pending"
    attempt.mkdir(parents=True)
    (attempt / "preview-pending.json").write_text("{}")
    with (attempt / "master.tif").open("wb") as stream:
        stream.truncate(6 * MIB)
    # New master reservation cannot fit; preview-only reservation can.
    monkeypatch.setattr(module.shutil, "disk_usage", lambda _: SimpleNamespace(free=2**40))
    recovered = []
    render = Mock()
    assert (
        run(
            store,
            identifier,
            render=render,
            recover=lambda *a: recovered.append(a),
            validate=lambda *a: bool(recovered),
        )
        == 0
    )
    assert len(recovered) == 1
    render.assert_not_called()


@pytest.mark.parametrize("shape", [[0, 20, 3], [-1, 20, 3], [10, True, 3], [10, 20], [10, 20, 4]])
def test_malformed_dimensions_fail_without_render(tmp_path, shape):
    store, identifier = setup_project(tmp_path, shape=shape)
    render = Mock()
    assert run(store, identifier, render=render, validate=lambda *a: False) == 1
    assert "dimensions" in store.get("brand", identifier)["items"][0]["error"]
    render.assert_not_called()


def test_usage_excludes_protected_sources_and_refuses_symlinks(tmp_path):
    target = tmp_path / "project"
    source = target / "source.linear"
    source.mkdir(parents=True)
    (source / "ground.npy").write_bytes(b"source")
    (target / "partial.tif").write_bytes(b"master")
    assert module.generated_bytes(target, [source]) == 6
    (target / "outside").symlink_to(tmp_path / "private")
    with pytest.raises(ValueError, match="symlink"):
        module.generated_bytes(target, [source])


def test_existing_over_budget_blocks_receipt_decode(tmp_path):
    store, identifier = setup_project(tmp_path, 1)
    attempt = store.root / identifier / "0-complete"
    attempt.mkdir(parents=True)
    (attempt / "receipt.json").write_text("complete")
    validate = Mock(return_value=True)
    assert run(store, identifier, render=Mock(), validate=validate) == 2
    validate.assert_not_called()


def test_preview_shortage_waits_without_recovering(tmp_path):
    store, identifier = setup_project(tmp_path, 1)
    attempt = store.root / identifier / "0-pending"
    attempt.mkdir(parents=True)
    (attempt / "preview-pending.json").write_text("{}")
    recover = Mock()
    assert run(store, identifier, render=Mock(), validate=Mock(), recover=recover) == 2
    recover.assert_not_called()


def test_unexpected_output_growth_stops_with_partial_preserved(tmp_path):
    store, identifier = setup_project(tmp_path, 20 * MIB)

    def render(source, destination, recipe):
        destination.mkdir()
        with (destination / "master.tif").open("wb") as stream:
            stream.truncate(21 * MIB)

    validate = Mock(return_value=True)
    assert run(store, identifier, render=render, validate=validate) == 2
    validate.assert_not_called()
    row = store.get("brand", identifier)
    assert row["status"] == "waiting"
    assert next((store.root / identifier).glob("*/master.tif")).stat().st_size == 21 * MIB


def test_completed_receipt_wins_over_pending_recovery_at_capacity(tmp_path, monkeypatch):
    store, identifier = setup_project(tmp_path, 3)
    completed = store.root / identifier / ("0-" + "a" * 32)
    completed.mkdir(parents=True)
    (completed / "receipt.json").write_bytes(b"x")
    pending = store.root / identifier / ("0-" + "f" * 32)
    pending.mkdir()
    (pending / "preview-pending.json").write_text("{}")
    monkeypatch.setattr(module.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    render, recover = Mock(), Mock()
    assert (
        run(
            store,
            identifier,
            render=render,
            recover=recover,
            validate=lambda source, candidate, recipe: candidate == completed,
        )
        == 0
    )
    assert store.get("brand", identifier)["items"][0]["output"] == str(completed)
    recover.assert_not_called()
    render.assert_not_called()


def test_invalid_completed_receipt_does_not_prevent_pending_recovery(tmp_path):
    store, identifier = setup_project(tmp_path, 30 * MIB)
    invalid = store.root / identifier / ("0-" + "f" * 32)
    invalid.mkdir(parents=True)
    (invalid / "receipt.json").write_text("broken")
    pending = store.root / identifier / ("0-" + "a" * 32)
    pending.mkdir()
    (pending / "preview-pending.json").write_text("{}")
    recovered = []

    def validate(source, candidate, recipe):
        if candidate == invalid:
            raise ValueError("invalid receipt")
        return candidate in recovered

    render = Mock()
    assert (
        run(
            store,
            identifier,
            render=render,
            recover=lambda source, candidate, recipe: recovered.append(candidate),
            validate=validate,
        )
        == 0
    )
    assert recovered == [pending]
    assert store.get("brand", identifier)["items"][0]["output"] == str(pending)
    render.assert_not_called()
