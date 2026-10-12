"""Private night routes keep source roots and brand boundaries explicit."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gomoveshift.console import night


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path / "console"))
    monkeypatch.setenv("GMS_CONSOLE_BRAND", "test")
    monkeypatch.delenv("GMS_NIGHT_SOURCE_ROOTS", raising=False)
    app = FastAPI()
    app.include_router(night.router)
    return TestClient(app)


def test_page_explains_scope_and_requires_source_roots(client):
    response = client.get("/photos/night")
    assert response.status_code == 200
    assert "Existing linear stacks" in response.text
    response = client.post("/photos/night", data={"sources": "/private", "recipe": "natural"})
    assert response.status_code == 400


def test_collection_scan_is_read_only_and_scoped_to_configured_roots(tmp_path, monkeypatch):
    allowed = tmp_path / "cards"
    card = allowed / "night-one"
    outside = tmp_path / "outside"
    card.mkdir(parents=True)
    outside.mkdir()
    found = Mock(as_dict=Mock(return_value={"light_count": 12}))
    scan = Mock(return_value=found)
    monkeypatch.setattr(night, "scan_collection", scan)

    assert night.inspect_collection(str(card), [allowed]) is found
    scan.assert_called_once_with(card.resolve())
    with pytest.raises(ValueError, match="configured source roots"):
        night.inspect_collection(str(outside), [allowed])
    with pytest.raises(ValueError, match="required"):
        night.inspect_collection(" ", [Path.cwd()])


def test_collection_scan_page_shows_findings_without_launching_worker(
    client, tmp_path, monkeypatch
):
    card = tmp_path / "card"
    card.mkdir()
    monkeypatch.setenv("GMS_NIGHT_SOURCE_ROOTS", str(tmp_path))
    collection = Mock()
    collection.as_dict.return_value = {
        "root": str(card),
        "light_count": 8,
        "likely_dark_count": 2,
        "capture_span_s": 420.0,
        "unreadable": [str(card / "bad.NEF")],
        "likely_darks": [str(card / "darks" / "dark.NEF")],
        "existing_linear": [str(card / "ready.linear")],
        "invalid_linear": [],
        "refused": [],
        "sequences": [
            {
                "frame_count": 8,
                "started_at": "2026-09-14T23:58:00",
                "ended_at": "2026-09-15T00:05:00",
                "iso": 3200,
                "exposure_s": 15.0,
                "geometry": {"width": 6000, "height": 4000, "confidence": "recorded"},
                "timezone_confidence": "camera_clock_only",
                "break_before": None,
                "frames": [],
            }
        ],
        "unsequenced": [
            {
                "frames": [str(card / "DSC_0001.NEF")],
                "frame_count": 1,
                "reason": "fewer_than_minimum_compatible_frames",
                "boundary_before": {"kind": "iso_change"},
                "boundary_after": {"kind": "exposure_change"},
            }
        ],
    }
    scan = Mock(return_value=collection)
    launch = Mock()
    monkeypatch.setattr(night, "scan_collection", scan)
    monkeypatch.setattr(night.night_workers, "launch", launch)

    response = client.post("/photos/night/scan", data={"source": str(card)})

    assert response.status_code == 200
    assert ">8</strong> likely light frames" in response.text
    assert ">2</strong> likely dark frames" in response.text
    assert "Cross-midnight" in response.text
    assert "Geometry: 6000×4000" in response.text
    assert "1 valid light frame was omitted from compatible sequences" in response.text
    assert "iso change" in response.text
    assert "exposure change" in response.text
    assert str(card / "ready.linear") in response.text
    assert 'name="sources"' in response.text
    launch.assert_not_called()


def test_product_submission_rescans_and_launches_one_typed_project(tmp_path, monkeypatch):
    collection = Mock()
    store = Mock()
    store.create_product_request.return_value = "typed-project"
    inspect = Mock(return_value=collection)
    launch = Mock()
    monkeypatch.setattr(night, "_roots", lambda: [tmp_path])
    monkeypatch.setattr(night, "_store", lambda: store)
    monkeypatch.setattr(night, "inspect_collection", inspect)
    monkeypatch.setattr(night, "_launch", launch)

    project = night.create_product_project(
        str(tmp_path / "card"),
        ["best_stills", "timelapse"],
        sequence_index=1,
        recipe="accent",
        storage_budget_gib="2",
    )

    assert project == "typed-project"
    inspect.assert_called_once_with(str(tmp_path / "card"), [tmp_path])
    request = store.create_product_request.call_args.args[3]
    assert [product.value for product in request.products] == ["best_stills", "timelapse"]
    assert store.create_product_request.call_args.kwargs == {
        "sequence_index": 1,
        "storage_budget_bytes": 2 * 1024**3,
    }
    launch.assert_called_once_with(store, project)


def test_comparison_project_reuses_collection_for_best_stills_only(tmp_path, monkeypatch):
    collection = Mock()
    row = {
        "id": "original",
        "kind": "night_products",
        "schema": 2,
        "requested_products": ["best_stills", "timelapse"],
        "collection_root": str(tmp_path / "card"),
        "storage_budget_bytes": 123,
    }
    store = Mock()
    store.create_product_request.return_value = "comparison"
    monkeypatch.setattr(night, "_store", lambda: store)
    monkeypatch.setattr(night, "_get", lambda *_args: row)
    monkeypatch.setattr(night, "_roots", lambda: [tmp_path])
    monkeypatch.setattr(night, "inspect_collection", Mock(return_value=collection))
    launch = Mock()
    monkeypatch.setattr(night, "_launch", launch)

    assert night.create_comparison_project("original", "bold") == "comparison"

    request = store.create_product_request.call_args.args[3]
    assert [product.value for product in request.products] == ["best_stills"]
    assert store.create_product_request.call_args.args[4] == "bold"
    assert store.create_product_request.call_args.kwargs == {
        "storage_budget_bytes": 123,
    }
    launch.assert_called_once_with(store, "comparison")


def test_comparison_projects_match_exact_linear_inputs_and_validated_previews(monkeypatch):
    linear = [{"path": "/card/a.linear", "manifest_sha256": "same"}]

    def candidate(identifier, recipe, inputs):
        return {
            "id": identifier,
            "brand": "brand",
            "schema": 2,
            "kind": "night_products",
            "recipe": recipe,
            "status": "complete",
            "requested_products": ["best_stills"],
            "inputs": {"linear": inputs, "lights": [], "darks": []},
            "tasks": [{"product": "best_stills", "output": "/private/output"}],
        }

    natural = candidate("natural", "natural", linear)
    bold = candidate("bold", "bold", linear)
    pending = candidate("pending", "accent", linear)
    pending["status"] = "running"
    unrelated = candidate("other", "accent", [{"path": "/other"}])
    store = Mock()
    store.list.return_value = [natural, unrelated, pending, bold]
    monkeypatch.setattr(
        night,
        "product_media",
        lambda row, _store: (
            [[]] if row["id"] == "pending" else [[{"kind": "image", "url": f"/{row['id']}.jpg"}]]
        ),
    )

    comparisons = night.comparison_projects(natural, store)

    assert [item["recipe"] for item in comparisons] == ["natural", "accent", "bold"]
    assert comparisons[0]["media"][0]["url"] == "/natural.jpg"
    assert comparisons[1]["status"] == "running" and comparisons[1]["media"] == []
    assert comparisons[2]["media"][0]["url"] == "/bold.jpg"


def test_product_media_lists_only_validated_browser_outputs(tmp_path, monkeypatch):
    store = Mock(root=tmp_path)
    attempt = tmp_path / "project" / "attempt"
    attempt.mkdir(parents=True)
    (attempt / "preview.jpg").write_bytes(b"image")
    (attempt / "night.mp4").write_bytes(b"video")
    (attempt / "master.tif").write_bytes(b"master")
    row = {
        "id": "project",
        "schema": 2,
        "kind": "night_products",
        "tasks": [{"product": "timelapse", "output": str(attempt)}],
    }
    monkeypatch.setattr(
        night,
        "validate_product_receipt",
        lambda *_args: {
            "outputs": [
                {"path": "preview.jpg", "role": "preview"},
                {"path": "night.mp4", "role": "video"},
                {"path": "master.tif", "role": "master"},
            ]
        },
    )

    media = night.product_media(row, store)

    assert [item["kind"] for item in media[0]] == ["image", "video"]
    assert all("master.tif" not in item["url"] for item in media[0])


def test_prepare_product_export_records_unpublished_privacy_copy(tmp_path, monkeypatch):
    store = Mock(root=tmp_path)
    row = {
        "id": "project",
        "brand": "brand",
        "tasks": [{"product": "deep_still"}] * 2 + [{"product": "best_stills"}],
    }
    record = {"export_id": "e" * 64, "publishable": False}
    monkeypatch.setattr(night, "_store", lambda: store)
    monkeypatch.setattr(night, "_get", lambda *_args: row)
    prepare = Mock(return_value=record)
    monkeypatch.setattr(night, "prepare_private_export", prepare)

    assert night.prepare_product_export("project", 2) == record

    prepare.assert_called_once_with(row, 2, tmp_path)
    store.add_private_export.assert_called_once_with("brand", "project", record)


def test_prepare_product_export_routes_timelapse_through_video_privacy(tmp_path, monkeypatch):
    store = Mock(root=tmp_path)
    row = {
        "id": "project",
        "brand": "brand",
        "tasks": [{"product": "timelapse"}],
    }
    request = {"id": "v" * 64, "status": "queued"}
    monkeypatch.setattr(night, "_store", lambda: store)
    monkeypatch.setattr(night, "_get", lambda *_args: row)
    store.request_private_export.return_value = request
    launch = Mock()
    monkeypatch.setattr(night, "_launch", launch)

    assert night.prepare_product_export("project", 0) == request

    store.request_private_export.assert_called_once_with("brand", "project", 0)
    launch.assert_called_once_with(store, "project")
    store.add_private_export.assert_not_called()


def test_create_scopes_brand_and_launches_supported_cli(client, tmp_path, monkeypatch):
    source = tmp_path / "stack.linear"
    source.mkdir()
    (source / "manifest.json").write_text("{}")
    monkeypatch.setenv("GMS_NIGHT_SOURCE_ROOTS", str(tmp_path))
    launch = Mock(return_value=42)
    monkeypatch.setattr(night.night_workers.jobs, "launch", launch)
    response = client.post(
        "/photos/night", data={"sources": str(source), "recipe": "accent"}, follow_redirects=False
    )
    assert response.status_code == 303
    argv = launch.call_args.args[2]
    assert argv[3:5] == ["photos", "night"]
    assert argv[argv.index("--brand") + 1] == "test"
    project = response.headers["location"].split("/")[-1]
    assert client.get(response.headers["location"]).status_code == 200
    monkeypatch.setenv("GMS_CONSOLE_BRAND", "other")
    assert client.get(f"/photos/night/{project}").status_code == 404
    assert client.get(f"/photos/night/{project}/preview/0").status_code == 404


def test_outside_source_root_is_rejected(client, tmp_path, monkeypatch):
    root = tmp_path / "allowed"
    root.mkdir()
    monkeypatch.setenv("GMS_NIGHT_SOURCE_ROOTS", str(root))
    response = client.post("/photos/night", data={"sources": str(tmp_path), "recipe": "natural"})
    assert response.status_code == 400


def test_preview_requires_byte_integrity_without_full_stack_loading(client, tmp_path, monkeypatch):
    import hashlib
    import json
    from dataclasses import asdict
    from gomoveshift.pipeline.photos import night_render
    from gomoveshift.pipeline.photos.astro_recipe import recipe_for

    store = night._store()
    destination = store.root / "project" / "0-attempt"
    destination.mkdir(parents=True)
    source = tmp_path / "source.linear"
    source.mkdir()
    manifest = source / "manifest.json"
    manifest.write_text("{}")
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    image = destination / "preview.jpg"
    image.write_bytes(b"private preview")
    row = {
        "recipe": "natural",
        "items": [
            {
                "status": "complete",
                "source": str(source),
                "manifest_sha256": manifest_sha,
                "output": str(destination),
            }
        ],
    }
    monkeypatch.setattr(night, "_get", lambda store, project: row)
    full_validation = Mock(side_effect=AssertionError("GET must not load full stack"))
    monkeypatch.setattr(night_render, "validate_render", full_validation)
    receipt = {
        "schema": 1,
        "state": "complete",
        "encoding": "sRGB",
        "crop": None,
        "quality_review_required": True,
        "recipe_name": "natural",
        "recipe": asdict(recipe_for("natural")),
        "source_manifest_sha256": manifest_sha,
        "outputs": {"preview.jpg": {"sha256": hashlib.sha256(image.read_bytes()).hexdigest()}},
    }
    assert client.get("/photos/night/project/preview/0").status_code == 404
    (destination / "receipt.json").write_text(json.dumps(receipt))
    response = client.get("/photos/night/project/preview/0")
    assert response.status_code == 200 and response.content == b"private preview"
    assert response.headers["cache-control"] == "private, no-store"
    full_validation.assert_not_called()
    image.write_bytes(b"tampered")
    assert client.get("/photos/night/project/preview/0").status_code == 404
    image.write_bytes(b"private preview")
    manifest.write_text('{"changed": true}')
    assert client.get("/photos/night/project/preview/0").status_code == 404
    assert client.get("/photos/night/project/preview/-1").status_code == 404
    row["items"][0]["output"] = str(tmp_path)
    assert client.get("/photos/night/project/preview/0").status_code == 404


def test_pause_cancel_resume_persist_intent_and_use_worker(client, monkeypatch):
    store = Mock()
    store.root = night._store().root
    store.get.return_value = {"id": "project"}
    monkeypatch.setattr(night, "_store", lambda: store)
    launch = Mock()
    monkeypatch.setattr(night, "_launch", launch)
    assert client.post("/photos/night/project/pause", follow_redirects=False).status_code == 303
    store.pause.assert_called_with("test", "project")
    launch.assert_not_called()
    assert client.post("/photos/night/project/cancel", follow_redirects=False).status_code == 303
    store.cancel.assert_called_with("test", "project")
    launch.assert_not_called()
    assert client.post("/photos/night/project/resume", follow_redirects=False).status_code == 303
    store.pause.assert_called_with("test", "project", requested=False)
    store.cancel.assert_called_with("test", "project", requested=False)
    launch.assert_called_once_with(store, "project")


def test_preview_obeys_console_auth(monkeypatch):
    from gomoveshift.console.auth import ConsoleAuthMiddleware

    monkeypatch.setenv("GMS_CONSOLE_TOKEN", "private-test-token")
    app = FastAPI()
    app.add_middleware(ConsoleAuthMiddleware)
    app.include_router(night.router)
    client = TestClient(app)
    assert client.get("/photos/night/project/preview/0").status_code == 401


def test_budget_form_persists_and_displays_immutable_limit(client, tmp_path, monkeypatch):
    source = tmp_path / "budget.linear"
    source.mkdir()
    (source / "manifest.json").write_text("{}")
    monkeypatch.setenv("GMS_NIGHT_SOURCE_ROOTS", str(tmp_path))
    monkeypatch.setattr(night, "_launch", Mock())
    assert 'name="storage_budget_gib"' in client.get("/photos/night").text
    response = client.post(
        "/photos/night",
        data={
            "sources": str(source),
            "storage_budget_gib": "1.5",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    identifier = response.headers["location"].rsplit("/", 1)[1]
    row = night._store().get("test", identifier)
    assert row["storage_budget_bytes"] == 1610612736
    page = client.get(response.headers["location"]).text
    assert "1610612736 bytes" in page
    assert "new request" in page


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "garbage", "1e999999", "0.00000000001"])
def test_budget_form_rejects_invalid_values_before_launch(client, tmp_path, monkeypatch, value):
    source = tmp_path / "invalid.linear"
    source.mkdir()
    (source / "manifest.json").write_text("{}")
    monkeypatch.setenv("GMS_NIGHT_SOURCE_ROOTS", str(tmp_path))
    launch = Mock()
    monkeypatch.setattr(night, "_launch", launch)
    response = client.post(
        "/photos/night",
        data={
            "sources": str(source),
            "storage_budget_gib": value,
        },
        follow_redirects=False,
    )
    assert response.status_code == 400
    launch.assert_not_called()
    assert night._store().list("test") == []


def test_blank_budget_keeps_new_request_uncapped(client, tmp_path, monkeypatch):
    source = tmp_path / "uncapped.linear"
    source.mkdir()
    (source / "manifest.json").write_text("{}")
    monkeypatch.setenv("GMS_NIGHT_SOURCE_ROOTS", str(tmp_path))
    monkeypatch.setattr(night, "_launch", Mock())
    response = client.post(
        "/photos/night",
        data={
            "sources": str(source),
            "storage_budget_gib": " ",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    identifier = response.headers["location"].rsplit("/", 1)[1]
    assert "storage_budget_bytes" not in night._store().get("test", identifier)
    assert "Uncapped" in client.get(response.headers["location"]).text


def test_project_displays_uncertain_launch_and_disabled_ticker(client, tmp_path, monkeypatch):
    source = tmp_path / "supervised.linear"
    source.mkdir()
    (source / "manifest.json").write_text("{}")
    store = night._store()
    project = store.create("test", [source], [tmp_path], "natural")
    monkeypatch.delenv("GMS_CONSOLE_TICKER", raising=False)
    night.night_workers._save(
        store.root,
        "test",
        project,
        {
            "enabled": True,
            "launches": 0,
            "crash_retries": 0,
            "next_retry_at": 100,
            "claim": "launching",
            "jobs_floor": 0,
            "observation": "Launch outcome unknown; manual process reconciliation required.",
        },
    )
    page = client.get(f"/photos/night/{project}").text
    assert "Launch outcome unknown" in page
    assert "disabled; waiting projects require manual Resume" in page


def test_supervisor_health_distinguishes_healthy_error_and_stale(monkeypatch):
    now = datetime(2026, 9, 16, tzinfo=timezone.utc)
    ticker = SimpleNamespace(
        is_alive=lambda: True,
        started_at=now - timedelta(seconds=30),
        last_night_ok_at=now - timedelta(seconds=10),
        last_night_error=None,
    )
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(console_ticker=ticker)))
    monkeypatch.setenv("GMS_CONSOLE_TICKER", "1")
    monkeypatch.setenv("GMS_CONSOLE_TICK_SECONDS", "60")

    assert night.supervisor_health(request, now=now)["state"] == "healthy"
    ticker.last_night_error = "database busy"
    assert night.supervisor_health(request, now=now)["state"] == "degraded"
    ticker.last_night_error = None
    ticker.last_night_ok_at = now - timedelta(seconds=181)
    stale = night.supervisor_health(request, now=now)
    assert stale["state"] == "degraded" and "overdue" in stale["message"]


def test_project_template_shows_review_only_retention_proposal():
    request = SimpleNamespace(url=SimpleNamespace(path="/photos/night/abc"), query_params={})
    html = night._templates.get_template("night.html").render(
        request=request,
        nav_entries=[],
        project={
            "id": "a" * 64,
            "kind": "night_products",
            "status": "complete",
            "recipe": "natural",
            "pause_requested": False,
            "tasks": [],
            "private_exports": [],
        },
        collection=None,
        product_media=[],
        supervision=None,
        supervisor_health={"state": "healthy", "message": "running"},
        configured=True,
        storage={
            "source_linear_bytes": 0,
            "source_raw_bytes": 0,
            "generated_bytes": 5,
            "warnings": [],
            "retention_proposal": {
                "candidate_count": 1,
                "reclaimable_bytes": 5,
                "candidates": [{"path": "/private/attempt", "reason": "Review only", "bytes": 5}],
                "protected": [{"path": "/private/source", "reason": "Source"}],
            },
        },
    )

    assert "Review-only retention proposal" in html
    assert "Waypost will not delete" in html
    assert "/private/attempt" in html


def test_progress_eta_updates_and_excludes_queue_time():
    row = {
        "current_stage": "timelapse",
        "stage_started_at": 100,
        "stage_progress": {"completed": 1, "total": 4},
    }

    assert night.progress_eta(row, now=110) == {
        "remaining_seconds": 30,
        "scope": "current stage; queueing excluded",
    }
    assert night.progress_eta(row, now=120)["remaining_seconds"] == 60
    assert night.progress_eta({"status": "waiting"}, now=120) is None


def test_project_template_offers_and_displays_private_video_keeper():
    request = SimpleNamespace(url=SimpleNamespace(path="/photos/night/abc"), query_params={})
    html = night._templates.get_template("night.html").render(
        request=request,
        nav_entries=[],
        project={
            "id": "a" * 64,
            "kind": "night_products",
            "status": "complete",
            "recipe": "natural",
            "pause_requested": False,
            "requested_products": ["timelapse"],
            "tasks": [{"product": "timelapse", "status": "complete", "attempts": 1}],
            "private_exports": [
                {"product": "timelapse", "privacy_regions": 2, "media_type": "video"}
            ],
            "private_export_requests": [
                {
                    "product": "timelapse",
                    "task_index": 0,
                    "status": "complete",
                    "attempts": 1,
                }
            ],
        },
        collection=None,
        product_media=[[{"kind": "video", "url": "/private/timelapse.mp4"}]],
        comparisons=[],
        comparison_missing=[],
        supervision=None,
        supervisor_health={"state": "healthy", "message": "running"},
        configured=True,
        storage={
            "source_linear_bytes": 0,
            "source_raw_bytes": 10,
            "generated_bytes": 10,
            "warnings": [],
            "retention_proposal": None,
        },
    )

    assert "Queue privacy-checked private video" in html
    assert "Private video jobs" in html
    assert "Location metadata absent" in html
    assert "Privacy-checked private timelapse keeper" in html
