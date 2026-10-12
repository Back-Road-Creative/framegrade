"""Brand-scoped console routes for private night redevelopment projects."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from ..pipeline.photos.astro_recipe import recipe_for
from ..pipeline.photos.face_filter import FaceBlurError
from ..pipeline.photos.night_budget import budget_bytes
from ..pipeline.photos.night_collection import ProductRequest, scan_collection
from ..pipeline.photos.night_inventory import inventory_project
from ..pipeline.photos.night_private_export import (
    prepare_private_export,
    validate_private_export,
)
from ..pipeline.photos.night_private_video import (
    validate_private_video,
)
from ..pipeline.photos.night_products import validate_product_receipt
from ..pipeline.photos.night_project import NightStore
from . import night_workers, settings

router = APIRouter(prefix="/photos/night")
_templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def _navigation() -> list:
    from .app import NAV_ENTRIES

    disabled = settings.brand_disabled_tabs(settings.current_brand())
    return [entry for entry in NAV_ENTRIES if entry[0] not in disabled]


def _store() -> NightStore:
    return NightStore(settings.console_data_dir() / "night-projects")


def progress_eta(row: dict, *, now: float | None = None) -> dict | None:
    """Estimate only the active stage from persisted milestones, never queue time."""
    progress = row.get("stage_progress")
    started = row.get("stage_started_at")
    if (
        not row.get("current_stage")
        or not isinstance(progress, dict)
        or not isinstance(started, (int, float))
    ):
        return None
    completed, total = progress.get("completed"), progress.get("total")
    if type(completed) is not int or type(total) is not int or completed <= 0 or total < completed:
        return None
    elapsed = (time.time() if now is None else now) - started
    if elapsed < 0:
        return None
    remaining = math.ceil(elapsed / completed * (total - completed))
    return {
        "remaining_seconds": remaining,
        "scope": "current stage; queueing excluded",
    }


def _roots() -> list[Path]:
    return [
        Path(p) for p in os.environ.get("GMS_NIGHT_SOURCE_ROOTS", "").split(os.pathsep) if p.strip()
    ]


def _get(store: NightStore, project: str) -> dict:
    try:
        return store.get(settings.current_brand(), project)
    except KeyError as exc:
        raise HTTPException(404, "Unknown night project") from exc


def _launch(store: NightStore, project: str) -> None:
    night_workers.launch(store, project)


def supervisor_health(request: Request, *, now: datetime | None = None) -> dict[str, str]:
    """Report the actual in-process night supervisor state without starting it."""
    if os.environ.get("GMS_CONSOLE_TICKER") != "1":
        return {
            "state": "disabled",
            "message": "disabled; waiting projects require manual Resume",
        }
    ticker = getattr(request.app.state, "console_ticker", None)
    if ticker is None:
        return {
            "state": "starting",
            "message": "configured, but the supervisor has not started in this console process",
        }
    if not ticker.is_alive():
        return {"state": "degraded", "message": "stopped; automatic retry is unavailable"}
    if ticker.last_night_error:
        return {
            "state": "degraded",
            "message": f"last supervision pass failed: {ticker.last_night_error}",
        }
    now = datetime.now(timezone.utc) if now is None else now
    reference = ticker.last_night_ok_at or ticker.started_at
    threshold = 3 * float(os.environ.get("GMS_CONSOLE_TICK_SECONDS", "60"))
    age = max(0, int((now - reference).total_seconds()))
    if age > threshold:
        return {
            "state": "degraded",
            "message": f"supervision pass is overdue ({age}s since last success)",
        }
    if ticker.last_night_ok_at is None:
        return {"state": "starting", "message": "running; first supervision pass is pending"}
    return {
        "state": "healthy",
        "message": f"running; last supervision pass succeeded {age}s ago",
    }


def inspect_collection(source: str, roots: list[Path]):
    """Inspect one real source directory within the configured private roots."""
    if not source.strip():
        raise ValueError("night collection source is required")
    path = Path(source).expanduser()
    if path.is_symlink():
        raise ValueError("night collection root must not be a symlink")
    resolved = path.resolve(strict=True)
    allowed = [root.resolve(strict=True) for root in roots]
    if not any(resolved.is_relative_to(root) for root in allowed):
        raise ValueError("night collection must be inside configured source roots")
    return scan_collection(resolved)


def create_product_project(
    source: str,
    products: list[str],
    *,
    sequence_index: int = 0,
    recipe: str = "natural",
    storage_budget_gib: str = "",
) -> str:
    """Rescan current headers, persist the typed request, then launch its worker."""
    roots = _roots()
    if not roots:
        raise ValueError("Configure GMS_NIGHT_SOURCE_ROOTS before creating a product request")
    collection = inspect_collection(source, roots)
    store = _store()
    project = store.create_product_request(
        settings.current_brand(),
        collection,
        roots,
        ProductRequest.create(products),
        recipe,
        sequence_index=sequence_index,
        storage_budget_bytes=budget_bytes(storage_budget_gib),
    )
    _launch(store, project)
    return project


def create_comparison_project(project: str, recipe: str) -> str:
    """Queue a best-still treatment against the same rescanned collection."""
    if recipe not in {"natural", "accent", "bold"}:
        raise ValueError("unsupported comparison treatment")
    store = _store()
    row = _get(store, project)
    if (
        row.get("schema") != 2
        or row.get("kind") != "night_products"
        or "best_stills" not in row.get("requested_products", [])
    ):
        raise ValueError("best-still product project required for treatment comparison")
    roots = _roots()
    collection = inspect_collection(row["collection_root"], roots)
    kwargs = {}
    if "storage_budget_bytes" in row:
        kwargs["storage_budget_bytes"] = row["storage_budget_bytes"]
    comparison = store.create_product_request(
        settings.current_brand(),
        collection,
        roots,
        ProductRequest.create(["best_stills"]),
        recipe,
        **kwargs,
    )
    _launch(store, comparison)
    return comparison


def prepare_product_export(project: str, task_index: int) -> dict:
    """Create a still keeper or durably queue a privacy-checked video keeper."""
    store = _store()
    row = _get(store, project)
    tasks = row.get("tasks", [])
    if task_index < 0 or task_index >= len(tasks):
        raise ValueError("unknown night product task")
    if tasks[task_index].get("product") == "timelapse":
        request = store.request_private_export(row["brand"], project, task_index)
        _launch(store, project)
        return request
    record = prepare_private_export(row, task_index, store.root)
    store.add_private_export(row["brand"], project, record)
    return record


def product_media(row: dict, store: NightStore) -> list[list[dict]]:
    """Return receipt-validated browser media without exposing local paths."""
    tasks = row.get("tasks", [])
    media: list[list[dict]] = [[] for _ in tasks]
    if row.get("schema") != 2 or row.get("kind") != "night_products":
        return media
    try:
        project_path = store.root / row["id"]
        project_root = project_path.resolve(strict=True)
        if project_path.is_symlink() or not project_root.is_relative_to(store.root.resolve()):
            return media
    except (OSError, KeyError, TypeError):
        return media
    for task_index, task in enumerate(tasks):
        output = task.get("output")
        if not output:
            continue
        try:
            attempt = Path(output)
            resolved_attempt = attempt.resolve(strict=True)
            if (
                attempt.is_symlink()
                or not resolved_attempt.is_dir()
                or not resolved_attempt.is_relative_to(project_root)
            ):
                continue
            receipt = validate_product_receipt(row, task, resolved_attempt)
            if receipt is None:
                continue
            for output_index, spec in enumerate(receipt["outputs"]):
                suffix = Path(spec["path"]).suffix.lower()
                kind = (
                    "image"
                    if spec.get("role") == "preview" and suffix in {".jpg", ".jpeg"}
                    else "video"
                    if spec.get("role") == "video" and suffix == ".mp4"
                    else None
                )
                if kind is None:
                    continue
                path = resolved_attempt / spec["path"]
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or not path.resolve().is_relative_to(resolved_attempt)
                ):
                    continue
                media[task_index].append(
                    {
                        "kind": kind,
                        "name": Path(spec["path"]).name,
                        "url": (f"/photos/night/{row['id']}/product/{task_index}/{output_index}"),
                        "output_index": output_index,
                    }
                )
        except (OSError, ValueError, TypeError, KeyError):
            continue
    return media


def comparison_projects(row: dict, store: NightStore) -> list[dict]:
    """Find best-still projects and validated previews for the same linear inputs."""
    if (
        row.get("schema") != 2
        or row.get("kind") != "night_products"
        or "best_stills" not in row.get("requested_products", [])
    ):
        return []
    linear = row.get("inputs", {}).get("linear")
    if not isinstance(linear, list) or not linear:
        return []
    found: dict[str, dict] = {}
    for candidate in store.list(row["brand"]):
        if (
            candidate.get("schema") != 2
            or candidate.get("kind") != "night_products"
            or candidate.get("inputs", {}).get("linear") != linear
            or "best_stills" not in candidate.get("requested_products", [])
            or candidate.get("recipe") in found
        ):
            continue
        tasks = candidate.get("tasks", [])
        task_index = next(
            (index for index, task in enumerate(tasks) if task.get("product") == "best_stills"),
            None,
        )
        if task_index is None:
            continue
        media = [
            item for item in product_media(candidate, store)[task_index] if item["kind"] == "image"
        ]
        found[candidate["recipe"]] = {
            "id": candidate["id"],
            "recipe": candidate["recipe"],
            "status": candidate["status"],
            "media": media,
        }
    return [found[recipe] for recipe in ("natural", "accent", "bold") if recipe in found]


@router.get("")
def index(request: Request):
    return _templates.TemplateResponse(
        request=request,
        name="night.html",
        context={
            "projects": _store().list(settings.current_brand()),
            "project": None,
            "collection": None,
            "configured": bool(_roots()),
            "nav_entries": _navigation(),
        },
    )


@router.post("/scan")
def scan(request: Request, source: str = Form(...)):
    roots = _roots()
    if not roots:
        raise HTTPException(400, "Configure GMS_NIGHT_SOURCE_ROOTS before scanning a card")
    try:
        collection = inspect_collection(source, roots).as_dict()
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return _templates.TemplateResponse(
        request=request,
        name="night.html",
        context={
            "projects": _store().list(settings.current_brand()),
            "project": None,
            "collection": collection,
            "configured": True,
            "nav_entries": _navigation(),
        },
    )


@router.post("")
def create(
    sources: str = Form(...),
    recipe: str = Form("natural"),
    storage_budget_gib: str = Form(""),
):
    roots = _roots()
    if not roots:
        raise HTTPException(400, "Configure GMS_NIGHT_SOURCE_ROOTS before importing stacks")
    store = _store()
    try:
        project = store.create(
            settings.current_brand(),
            [Path(p.strip()) for p in sources.splitlines() if p.strip()],
            roots,
            recipe,
            storage_budget_bytes=budget_bytes(storage_budget_gib),
        )
        _launch(store, project)
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return RedirectResponse(f"/photos/night/{project}", status_code=303)


@router.post("/products")
def create_products(
    source: str = Form(...),
    products: list[str] = Form(...),
    sequence_index: int = Form(0),
    recipe: str = Form("natural"),
    storage_budget_gib: str = Form(""),
):
    try:
        project = create_product_project(
            source,
            products,
            sequence_index=sequence_index,
            recipe=recipe,
            storage_budget_gib=storage_budget_gib,
        )
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return RedirectResponse(f"/photos/night/{project}", status_code=303)


@router.get("/{project}")
def detail(request: Request, project: str):
    store = _store()
    row = _get(store, project)
    comparisons = comparison_projects(row, store)
    compared_recipes = {item["recipe"] for item in comparisons}
    return _templates.TemplateResponse(
        request=request,
        name="night.html",
        context={
            "projects": [],
            "project": row,
            "collection": None,
            "storage": inventory_project(row, store.root),
            "product_media": product_media(row, store),
            "comparisons": comparisons,
            "comparison_missing": [
                recipe for recipe in ("natural", "accent", "bold") if recipe not in compared_recipes
            ]
            if "best_stills" in row.get("requested_products", [])
            else [],
            "supervision": night_workers.observation(store.root, settings.current_brand(), project),
            "supervisor_health": supervisor_health(request),
            "progress_eta": progress_eta(row),
            "configured": bool(_roots()),
            "nav_entries": _navigation(),
        },
    )


@router.post("/{project}/compare/{recipe}")
def create_comparison(project: str, recipe: str):
    try:
        comparison = create_comparison_project(project, recipe)
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return RedirectResponse(f"/photos/night/{comparison}", status_code=303)


@router.get("/{project}/product/{task_index}/{output_index}")
def product_output(project: str, task_index: int, output_index: int):
    store = _store()
    row = _get(store, project)
    media = product_media(row, store)
    if task_index < 0 or task_index >= len(media):
        raise HTTPException(404, "Validated product output unavailable")
    found = next(
        (item for item in media[task_index] if item["output_index"] == output_index),
        None,
    )
    if found is None:
        raise HTTPException(404, "Validated product output unavailable")
    task = row["tasks"][task_index]
    attempt = Path(task["output"])
    receipt = validate_product_receipt(row, task, attempt)
    if receipt is None or output_index < 0 or output_index >= len(receipt["outputs"]):
        raise HTTPException(404, "Validated product output unavailable")
    path = attempt / receipt["outputs"][output_index]["path"]
    return FileResponse(
        path,
        media_type="image/jpeg" if found["kind"] == "image" else "video/mp4",
        headers={"Cache-Control": "private, no-store"},
    )


@router.post("/{project}/export/{task_index}")
def create_private_export(project: str, task_index: int):
    try:
        prepare_product_export(project, task_index)
    except (FaceBlurError, OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return RedirectResponse(f"/photos/night/{project}", status_code=303)


@router.get("/{project}/export/{export_index}")
def private_export(project: str, export_index: int):
    store = _store()
    row = _get(store, project)
    exports = row.get("private_exports", [])
    if export_index < 0 or export_index >= len(exports):
        raise HTTPException(404, "Validated private export unavailable")
    record = exports[export_index]
    validator = (
        validate_private_video if record.get("media_type") == "video" else validate_private_export
    )
    path = validator(row, record, store.root)
    if path is None:
        raise HTTPException(404, "Validated private export unavailable")
    return FileResponse(
        path,
        media_type="video/mp4" if record.get("media_type") == "video" else "image/jpeg",
        headers={"Cache-Control": "private, no-store"},
    )


@router.post("/{project}/pause")
def pause(project: str):
    store = _store()
    _get(store, project)
    store.pause(settings.current_brand(), project)
    return RedirectResponse(f"/photos/night/{project}", status_code=303)


@router.post("/{project}/cancel")
def cancel(project: str):
    store = _store()
    _get(store, project)
    store.cancel(settings.current_brand(), project)
    return RedirectResponse(f"/photos/night/{project}", status_code=303)


@router.post("/{project}/resume")
def resume(project: str):
    store = _store()
    _get(store, project)
    store.pause(settings.current_brand(), project, requested=False)
    store.cancel(settings.current_brand(), project, requested=False)
    _launch(store, project)
    return RedirectResponse(f"/photos/night/{project}", status_code=303)


def _bounded_bytes(path: Path, limit: int) -> bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError("Invalid or oversized preview metadata")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("Preview metadata grew beyond limit")
    return data


def _preview_integrity(item: dict, destination: Path, recipe: str) -> None:
    """Check completed derivative bytes, not full stack quality or source payloads.

    Workers validate source payloads and both exports under resource admission.
    Viewing does bounded I/O only: at most 2 MiB metadata and 64 MiB JPEG bytes.
    A changed source manifest or preview invalidates this view. Changes to source
    payloads with an unchanged manifest are detected on worker reuse, not this GET.
    """
    manifest = _bounded_bytes(Path(item["source"]) / "manifest.json", 1024 * 1024)
    receipt = json.loads(_bounded_bytes(destination / "receipt.json", 1024 * 1024))
    manifest_sha = hashlib.sha256(manifest).hexdigest()
    if (
        manifest_sha != item["manifest_sha256"]
        or receipt["source_manifest_sha256"] != manifest_sha
        or receipt["schema"] != 1
        or receipt["state"] != "complete"
        or receipt["recipe_name"] != recipe
        or receipt["recipe"] != asdict(recipe_for(recipe))
        or receipt["encoding"] != "sRGB"
        or receipt["crop"] is not None
        or receipt["quality_review_required"] is not True
    ):
        raise ValueError("Preview provenance mismatch")
    image = destination / "preview.jpg"
    limit = 64 * 1024 * 1024
    if image.stat().st_size > limit:
        raise ValueError("Preview exceeds serving limit")
    digest = hashlib.sha256()
    size = 0
    with image.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(block)
            if size > limit:
                raise ValueError("Preview grew beyond serving limit")
            digest.update(block)
    if digest.hexdigest() != receipt["outputs"]["preview.jpg"]["sha256"]:
        raise ValueError("Preview bytes do not match receipt")


@router.get("/{project}/preview/{index}")
def preview(project: str, index: int):
    store = _store()
    row = _get(store, project)
    if index < 0 or index >= len(row["items"]):
        raise HTTPException(404, "Unknown preview")
    item = row["items"][index]
    try:
        if item["status"] != "complete" or not item["output"]:
            raise ValueError("Preview is not complete")
        destination = Path(item["output"])
        expected = store.root / project
        if destination.is_symlink() or destination.resolve().parent != expected.resolve():
            raise ValueError("Invalid preview location")
        if expected.is_symlink() or not expected.resolve().is_relative_to(store.root):
            raise ValueError("Invalid project location")
        image = destination / "preview.jpg"
        if image.is_symlink() or not image.resolve().is_relative_to(expected.resolve()):
            raise ValueError("Invalid preview file")
        _preview_integrity(item, destination, row["recipe"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(404, "Validated preview unavailable") from exc
    return FileResponse(
        image, media_type="image/jpeg", headers={"Cache-Control": "private, no-store"}
    )
