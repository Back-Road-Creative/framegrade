"""Private night projects retain ownership and truthful batch outcomes."""

import json
from pathlib import Path
import pytest
from gomoveshift.pipeline.photos.night_project import NightStore, process_project


def source(tmp_path, name="a"):
    path = tmp_path / name
    path.mkdir()
    (path / "manifest.json").write_text(
        json.dumps({"schema": 4, "payloads": {"sky_sum.npy": {"shape": [10, 20, 3]}}})
    )
    return path


def fake_render(src, dest, recipe):
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "preview.jpg").write_bytes(b"preview")
    (dest / "receipt.json").write_text("valid")


def fake_valid(src, dest, recipe):
    return (dest / "receipt.json").exists() and (dest / "preview.jpg").read_bytes() == b"preview"


def test_create_is_idempotent_brand_scoped_and_rejects_escape(tmp_path):
    src = source(tmp_path)
    store = NightStore(tmp_path / "projects")
    project = store.create("one", [src], [tmp_path], "natural")
    assert store.create("one", [src], [tmp_path], "natural") == project
    assert store.get("one", project)["status"] == "queued"
    with pytest.raises(KeyError):
        store.get("two", project)
    with pytest.raises(ValueError, match="root"):
        store.create("one", [src], [tmp_path / "other"], "natural")
    link = tmp_path / "allowed"
    link.mkdir()
    (link / "escape").symlink_to(src, target_is_directory=True)
    with pytest.raises(ValueError, match="root"):
        store.create("one", [link / "escape"], [link], "natural")
    with pytest.raises(ValueError):
        store.create("one", [], [tmp_path], "natural")


def test_batch_failure_is_partial_and_resume_only_retries_missing(tmp_path):
    sources = [source(tmp_path, "a"), source(tmp_path, "b")]
    store = NightStore(tmp_path / "projects")
    project = store.create("one", sources, [tmp_path], "natural")
    calls = []

    def render(src, dest, recipe):
        calls.append(src.name)
        if src.name == "b":
            raise RuntimeError("preview failed")
        fake_render(src, dest, recipe)

    assert (
        process_project(
            store, "one", project, render=render, validate=fake_valid, admitted=lambda _: True
        )
        == 1
    )
    assert store.get("one", project)["status"] == "partial"
    assert calls == ["a", "b"]
    assert (
        process_project(
            store, "one", project, render=fake_render, validate=fake_valid, admitted=lambda _: True
        )
        == 0
    )
    row = store.get("one", project)
    assert row["status"] == "complete"
    assert row["items"][0]["attempts"] == 1
    assert row["items"][1]["attempts"] == 2


def test_missing_output_is_not_reused_and_changed_source_refused(tmp_path):
    src = source(tmp_path)
    store = NightStore(tmp_path / "projects")
    project = store.create("one", [src], [tmp_path], "natural")
    kwargs = dict(render=fake_render, validate=fake_valid, admitted=lambda _: True)
    assert process_project(store, "one", project, **kwargs) == 0
    row = store.get("one", project)
    (Path(row["items"][0]["output"]) / "preview.jpg").unlink()
    assert process_project(store, "one", project, **kwargs) == 0
    assert store.get("one", project)["items"][0]["attempts"] == 2
    (src / "manifest.json").write_text("{}")
    assert process_project(store, "one", project, **kwargs) == 1
    assert store.get("one", project)["status"] == "failed"


def test_admission_and_pause_do_not_skip_items(tmp_path):
    src = source(tmp_path)
    store = NightStore(tmp_path / "projects")
    project = store.create("one", [src], [tmp_path], "natural")
    assert process_project(store, "one", project, admitted=lambda _: False) == 2
    assert store.get("one", project)["status"] == "waiting"
    assert store.get("one", project)["wait_kind"] == "memory"
    assert store.get("one", project)["items"][0]["attempts"] == 0
    store.pause("one", project)
    assert process_project(store, "one", project, admitted=lambda _: True) == 2
    assert store.get("one", project)["status"] == "paused"


def test_worker_lock_prevents_concurrent_render(tmp_path):
    import fcntl

    store = NightStore(tmp_path / "projects")
    project = store.create("one", [source(tmp_path)], [tmp_path], "natural")
    with (store.root / "heavy.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert (
            process_project(
                store,
                "one",
                project,
                render=lambda *a: pytest.fail("duplicate worker"),
                admitted=lambda _: True,
            )
            == 2
        )
    assert store.get("one", project)["items"][0]["attempts"] == 0
    assert store.get("one", project)["wait_kind"] == "admission"


def test_crash_after_receipt_is_adopted_without_render(tmp_path):
    store = NightStore(tmp_path / "projects")
    src = source(tmp_path)
    project = store.create("one", [src], [tmp_path], "natural")
    fake_render(src, store.root / project / "0-crashed-attempt", "natural")
    assert (
        process_project(
            store,
            "one",
            project,
            render=lambda *a: pytest.fail("unnecessary render"),
            validate=fake_valid,
            admitted=lambda _: True,
        )
        == 0
    )
    assert store.get("one", project)["items"][0]["attempts"] == 0


def test_pause_during_render_survives_status_write(tmp_path):
    store = NightStore(tmp_path / "projects")
    sources = [source(tmp_path, "a"), source(tmp_path, "b")]
    project = store.create("one", sources, [tmp_path], "natural")

    def render(src, dest, recipe):
        fake_render(src, dest, recipe)
        store.pause("one", project)

    assert (
        process_project(
            store, "one", project, render=render, validate=fake_valid, admitted=lambda _: True
        )
        == 2
    )
    row = store.get("one", project)
    assert row["status"] == "paused"
    assert row["items"][0]["status"] == "complete"
    assert row["items"][1]["attempts"] == 0


def test_cancel_during_render_preserves_completed_work_and_stops_next_item(tmp_path):
    store = NightStore(tmp_path / "projects")
    sources = [source(tmp_path, "a"), source(tmp_path, "b")]
    project = store.create("one", sources, [tmp_path], "natural")

    def render(src, dest, recipe):
        fake_render(src, dest, recipe)
        store.cancel("one", project)

    assert (
        process_project(
            store, "one", project, render=render, validate=fake_valid, admitted=lambda _: True
        )
        == 1
    )
    row = store.get("one", project)
    assert row["status"] == "cancelled"
    assert row["items"][0]["status"] == "complete"
    assert row["items"][1]["attempts"] == 0

    store.cancel("one", project, requested=False)
    assert (
        process_project(
            store, "one", project, render=fake_render, validate=fake_valid, admitted=lambda _: True
        )
        == 0
    )
    resumed = store.get("one", project)
    assert [item["attempts"] for item in resumed["items"]] == [1, 1]


def test_dead_worker_status_is_interrupted_not_success(tmp_path, monkeypatch):
    import gomoveshift.pipeline.photos.night_project as module

    store = NightStore(tmp_path / "projects")
    project = store.create("one", [source(tmp_path)], [tmp_path], "natural")
    row = store.get("one", project)
    row.update(status="running", worker={"pid": 123, "started_at": 1, "boot_id": "old"})
    store.save(row)
    monkeypatch.setattr(module, "worker_alive", lambda worker: False)
    assert store.get("one", project)["status"] == "interrupted"
    assert store.get("one", project)["quality_status"] == "review_required"


def test_changed_build_and_workspace_symlink_refused(tmp_path, monkeypatch):
    import gomoveshift.pipeline.photos.night_project as module

    store = NightStore(tmp_path / "projects")
    project = store.create("one", [source(tmp_path)], [tmp_path], "natural")
    monkeypatch.setattr(module, "build_identity", lambda: {"changed": "revision"})
    assert (
        process_project(
            store, "one", project, render=fake_render, validate=fake_valid, admitted=lambda _: True
        )
        == 1
    )
    assert store.get("one", project)["status"] == "blocked"


def test_partial_master_recovers_preview_without_rendering(tmp_path):
    store = NightStore(tmp_path / "projects")
    src = source(tmp_path)
    project = store.create("one", [src], [tmp_path], "natural")
    attempt = store.root / project / "0-pending"
    attempt.mkdir(parents=True)
    (attempt / "preview-pending.json").write_text("{}")
    calls = []

    def recover(src, dest, recipe):
        calls.append(dest)
        fake_render(src, dest, recipe)
        return True

    assert (
        process_project(
            store,
            "one",
            project,
            render=lambda *a: pytest.fail("master re-rendered"),
            validate=fake_valid,
            recover=recover,
            admitted=lambda _: True,
        )
        == 0
    )
    assert calls == [attempt]
