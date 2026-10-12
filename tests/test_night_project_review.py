"""Independent resource and identity regressions found during project review."""

import json

from gomoveshift.pipeline.photos import night_project as module


def project(tmp_path, name="one"):
    source = tmp_path / f"{name}.linear"
    source.mkdir()
    (source / "manifest.json").write_text(
        json.dumps({"schema": 4, "payloads": {"sky_sum.npy": {"shape": [10, 20, 3]}}})
    )
    store = module.NightStore(tmp_path / f"{name}-projects")
    identifier = store.create("brand", [source], [tmp_path], "natural")
    return store, identifier


def test_receipt_decoding_waits_for_memory_admission(tmp_path):
    store, identifier = project(tmp_path)
    attempt = store.root / identifier / "0-old-attempt"
    attempt.mkdir(parents=True)
    validation_calls = []

    def validate(*args):
        validation_calls.append(args)
        return False

    assert (
        module.process_project(
            store,
            "brand",
            identifier,
            render=lambda *args: None,
            validate=validate,
            admitted=lambda _: False,
        )
        == 2
    )
    # validate_render fully decodes JPEG and TIFF, allocating hundreds of MB.
    assert validation_calls == [], "old receipt decoding bypassed the memory gate"


def test_separate_workspaces_share_heavy_worker_admission(tmp_path):
    first, first_id = project(tmp_path, "first")
    second, second_id = project(tmp_path, "second")
    nested_results = []

    def render(*args):
        nested_results.append(
            module.process_project(
                second,
                "brand",
                second_id,
                render=lambda *args: None,
                validate=lambda *args: True,
                admitted=lambda _: True,
            )
        )

    module.process_project(
        first,
        "brand",
        first_id,
        render=render,
        validate=lambda *args: True,
        admitted=lambda _: True,
    )
    assert nested_results == [2], "different project roots admitted concurrent heavy workers"
    assert second.get("brand", second_id)["items"][0]["attempts"] == 0


def test_build_change_during_last_render_cannot_finish_old_request(tmp_path, monkeypatch):
    store, identifier = project(tmp_path)

    def render(*args):
        monkeypatch.setattr(module, "build_identity", lambda: {"changed": "build"})

    result = module.process_project(
        store,
        "brand",
        identifier,
        render=render,
        validate=lambda *args: True,
        admitted=lambda _: True,
    )
    assert result == 1
    row = store.get("brand", identifier)
    assert row["status"] != "complete"
    assert row["items"][0]["output"] is None


def test_liveness_read_cannot_overwrite_concurrent_worker_completion(tmp_path, monkeypatch):
    store, identifier = project(tmp_path)
    stale = store.get("brand", identifier)
    stale.update(status="running", worker={"pid": 999999, "boot_id": "old", "started_at": "old"})
    store.save(stale)

    def worker_exited(_worker):
        # The original worker committed completion after SELECT, then exited
        # before the reader checked its liveness.
        completed = dict(stale, status="complete")
        completed["items"] = [dict(stale["items"][0], status="complete", output="verified-output")]
        store.save(completed)
        return False

    monkeypatch.setattr(module, "worker_alive", worker_exited)
    observed = store.get("brand", identifier)
    assert observed["status"] == "complete"
    assert observed["items"][0]["output"] == "verified-output"
    assert store.get("brand", identifier)["status"] == "complete"


def test_build_change_during_receipt_adoption_cannot_finish_old_request(tmp_path, monkeypatch):
    store, identifier = project(tmp_path)
    (store.root / identifier / "0-recoverable").mkdir(parents=True)

    def validate(*args):
        monkeypatch.setattr(module, "build_identity", lambda: {"changed": "build"})
        return True

    def unexpected_render(*args):
        raise AssertionError("a changed-build request must not render")

    result = module.process_project(
        store,
        "brand",
        identifier,
        render=unexpected_render,
        validate=validate,
        recover=lambda *args: False,
        admitted=lambda _: True,
    )
    assert result == 1
    row = store.get("brand", identifier)
    assert row["status"] == "failed"
    assert row["items"][0]["output"] is None
    assert "build changed" in row["items"][0]["error"]


def test_night_worker_rejects_nonregular_admission_file(tmp_path, monkeypatch):
    import os
    import pytest

    store, identifier = project(tmp_path)
    lock = tmp_path / "fifo.lock"
    os.mkfifo(lock)
    monkeypatch.setenv("GMS_NIGHT_ADMISSION_LOCK", str(lock))
    called = []
    with pytest.raises(OSError, match="regular file"):
        module.process_project(
            store,
            "brand",
            identifier,
            render=lambda *args: called.append("render"),
            validate=lambda *args: True,
            admitted=lambda _: True,
        )
    assert called == []
