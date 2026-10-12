"""Private, durable redevelopment projects. Never publishes or deletes source media.

A filesystem lock fences participating workers across console and CLI processes.
Completed attempt directories remain adoptable after a DB-commit crash. This is
local-host admission, not a reservation against unrelated media/CI processes.
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sqlite3
import stat
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from ..media_admission import media_admission
from .night_collection import NightCollection, NightProduct, ProductRequest

MAX_PRODUCT_INPUTS = 4096


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_identity() -> dict:
    """Installed interpreter, media tools and dependency versions for product resumes."""
    packages = {}
    for name in (
        "numpy",
        "rawpy",
        "opencv-python-headless",
        "tifffile",
        "Pillow",
        "redactcam",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    binaries = {}
    for name, candidate in (
        ("python", sys.executable),
        ("ffmpeg", shutil.which("ffmpeg")),
        ("ffprobe", shutil.which("ffprobe")),
    ):
        if candidate:
            path = Path(candidate).resolve(strict=True)
            binaries[name] = {"path": str(path), "sha256": digest(path)}
        else:
            binaries[name] = None
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "packages": packages,
        "binaries": binaries,
    }


def build_identity() -> dict:
    root = Path(__file__).parent
    return {
        name: digest(root / name)
        for name in (
            "night_project.py",
            "night_render.py",
            "astro_recipe.py",
            "astro_linear.py",
            "astro_stack.py",
            "develop.py",
            "night_collection.py",
            "night_products.py",
            "night_private_export.py",
            "night_private_video.py",
        )
    }


def process_start(pid: int) -> str:
    stat = Path(f"/proc/{pid}/stat").read_text()
    return stat.rsplit(")", 1)[1].split()[19]


def worker_alive(worker: dict | None) -> bool:
    if not worker:
        return False
    try:
        return (
            worker["boot_id"] == Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            and process_start(worker["pid"]) == worker["started_at"]
        )
    except (KeyError, OSError, IndexError):
        return False


def _mark_cancelled(row: dict) -> None:
    """Record a safe stage-boundary stop while preserving completed artifacts."""
    row.update(
        status="cancelled",
        reason="Cancelled after the current render; completed artifacts are preserved. Resume to continue.",
        current_stage=None,
        worker=None,
    )
    row.pop("stage_progress", None)
    row.pop("wait_kind", None)


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _within(path: Path, roots: list[Path]) -> Path:
    resolved = path.resolve(strict=True)
    if not any(
        resolved.is_relative_to(root.resolve(strict=True)) for root in roots if root.exists()
    ):
        raise ValueError("source escapes configured roots")
    return resolved


def _file_identity(path: Path, roots: list[Path]) -> dict:
    """Stable local identity for one RAW input without decoding its pixels."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("night product inputs must be regular files, not links")
    resolved = _within(path, roots)
    info = resolved.stat()
    return {
        "path": str(resolved),
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "device": info.st_dev,
        "inode": info.st_ino,
    }


def _linear_identity(path: Path, roots: list[Path]) -> dict:
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise ValueError("linear product inputs must be real directories")
    resolved = _within(path, roots)
    manifest = _within(resolved / "manifest.json", [resolved])
    data = json.loads(manifest.read_text())
    if not isinstance(data, dict) or "schema" not in data:
        raise ValueError(f"invalid linear manifest: {manifest}")
    return {
        "path": str(resolved),
        "manifest_sha256": digest(manifest),
    }


class NightStore:
    """One durable workspace, with brand-scoped, idempotent requests."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        with self.connection() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS night_projects ("
                "id TEXT PRIMARY KEY, brand TEXT NOT NULL, data TEXT NOT NULL)"
            )

    @contextmanager
    def connection(self):
        conn = sqlite3.connect(self.root / "projects.sqlite3", timeout=30)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def create(
        self,
        brand: str,
        sources: list[Path],
        roots: list[Path],
        recipe: str,
        *,
        storage_budget_bytes: int | None = None,
    ) -> str:
        if storage_budget_bytes is not None and (
            type(storage_budget_bytes) is not int or storage_budget_bytes <= 0
        ):
            raise ValueError("storage budget must be a positive integer byte count")
        if not brand or not sources or recipe not in {"natural", "accent", "bold"}:
            raise ValueError("brand, nonempty sources and a supported recipe are required")
        items = []
        for source in sorted(set(sources)):
            path = _within(Path(source), roots)
            manifest = _within(path / "manifest.json", [path])
            json.loads(manifest.read_text())
            items.append(
                {
                    "source": str(path),
                    "manifest_sha256": digest(manifest),
                    "status": "queued",
                    "attempts": 0,
                    "output": None,
                    "error": None,
                }
            )
        identity = {
            "schema": 1,
            "build": build_identity(),
            "runtime": runtime_identity(),
            "interpreter": sys.executable,
            "brand": brand,
            "recipe": recipe,
            "inputs": [(i["source"], i["manifest_sha256"]) for i in items],
        }
        if storage_budget_bytes is not None:
            identity["storage_budget_bytes"] = storage_budget_bytes
        project = hashlib.sha256(_canonical(identity).encode()).hexdigest()
        row = {
            **identity,
            "id": project,
            "items": items,
            "status": "queued",
            "pause_requested": False,
            "cancel_requested": False,
            "created_at": time.time(),
            "updated_at": time.time(),
            "source_roots": [str(Path(p).resolve()) for p in roots],
            "quality_status": "review_required",
            "worker": None,
        }
        with self.connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO night_projects VALUES (?, ?, ?)",
                (project, brand, _canonical(row)),
            )
        return project

    def create_product_request(
        self,
        brand: str,
        collection: NightCollection,
        roots: list[Path],
        request: ProductRequest,
        recipe: str = "natural",
        *,
        sequence_index: int = 0,
        storage_budget_bytes: int | None = None,
    ) -> str:
        """Persist one typed multi-product request before any image work starts."""
        if not brand or recipe not in {"natural", "accent", "bold"}:
            raise ValueError("brand and a supported recipe are required")
        if not isinstance(request, ProductRequest):
            raise ValueError("a typed product request is required")
        if storage_budget_bytes is not None and (
            type(storage_budget_bytes) is not int or storage_budget_bytes <= 0
        ):
            raise ValueError("storage budget must be a positive integer byte count")
        if collection.root.is_symlink() or not collection.root.is_dir():
            raise ValueError("night collection must be a real directory")
        try:
            root = _within(collection.root, roots)
        except (OSError, ValueError) as exc:
            raise ValueError("night collection must be inside configured roots") from exc

        products = request.products
        needs_sequence = any(
            product in {NightProduct.TIMELAPSE, NightProduct.STAR_TRAILS} for product in products
        )
        sequence = None
        if needs_sequence:
            if sequence_index < 0 or sequence_index >= len(collection.sequences):
                raise ValueError("a selected compatible sequence is required for video or trails")
            sequence = collection.sequences[sequence_index]
            if sequence.frame_count < 2:
                raise ValueError("a selected compatible sequence needs at least two frames")
        if NightProduct.BEST_STILLS in products and not collection.existing_linear:
            raise ValueError("best_stills requires at least one validated linear stack")
        if NightProduct.DEEP_STILL in products and len(collection.existing_linear) < 2:
            raise ValueError("deep_still requires at least two validated linear stacks")

        light_frames = sequence.frames if sequence is not None else ()
        if len(light_frames) + len(collection.likely_darks) > MAX_PRODUCT_INPUTS:
            raise ValueError(f"night product request exceeds {MAX_PRODUCT_INPUTS} RAW inputs")
        light_inputs = []
        for frame in light_frames:
            identity = _file_identity(frame.path, [root])
            identity.update(
                captured_at=frame.captured_at.isoformat(),
                iso=frame.iso,
                exposure_s=frame.exposure_s,
            )
            light_inputs.append(identity)
        dark_inputs = (
            [_file_identity(path, [root]) for path in collection.likely_darks]
            if needs_sequence
            else []
        )
        needs_linear = any(
            product in {NightProduct.BEST_STILLS, NightProduct.DEEP_STILL} for product in products
        )
        linear_inputs = (
            [_linear_identity(path, [root]) for path in collection.existing_linear]
            if needs_linear
            else []
        )
        inputs = {"lights": light_inputs, "darks": dark_inputs, "linear": linear_inputs}
        requested = [product.value for product in products]
        identity = {
            "schema": 2,
            "kind": "night_products",
            "build": build_identity(),
            "runtime": runtime_identity(),
            "interpreter": sys.executable,
            "brand": brand,
            "recipe": recipe,
            "requested_products": requested,
            "sequence_index": sequence_index if needs_sequence else None,
            "inputs": inputs,
        }
        if storage_budget_bytes is not None:
            identity["storage_budget_bytes"] = storage_budget_bytes
        project = hashlib.sha256(_canonical(identity).encode()).hexdigest()
        capture = None
        if sequence is not None:
            capture = {
                "started_at": sequence.started_at.isoformat(),
                "ended_at": sequence.ended_at.isoformat(),
                "frame_count": sequence.frame_count,
            }
        row = {
            **identity,
            "id": project,
            "collection_root": str(root),
            "capture": capture,
            "tasks": [
                {
                    "product": product,
                    "status": "queued",
                    "attempts": 0,
                    "output": None,
                    "receipt": None,
                    "error": None,
                }
                for product in requested
            ],
            "status": "queued",
            "pause_requested": False,
            "cancel_requested": False,
            "created_at": time.time(),
            "updated_at": time.time(),
            "source_roots": [str(Path(path).resolve()) for path in roots],
            "quality_status": "review_required",
            "worker": None,
        }
        encoded = _canonical(row)
        if len(encoded.encode()) > 1024 * 1024:
            raise ValueError("night product request metadata exceeds 1 MiB")
        with self.connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO night_projects VALUES (?, ?, ?)",
                (project, brand, encoded),
            )
        return project

    def get(self, brand: str, project: str) -> dict:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT data FROM night_projects WHERE id=? AND brand=?", (project, brand)
            ).fetchone()
        if not row:
            raise KeyError(project)
        data = json.loads(row[0])
        if data["status"] == "running" and not worker_alive(data.get("worker")):
            data["status"] = "interrupted"
            data["reason"] = "Recorded worker is no longer alive; resume validates saved receipts."
            data["updated_at"] = time.time()
            with self.connection() as conn:
                # A worker may have committed completion (or a new worker may
                # have started) since our snapshot. Only retire that exact row.
                changed = conn.execute(
                    "UPDATE night_projects SET data=? WHERE id=? AND brand=? AND data=?",
                    (_canonical(data), project, brand, row[0]),
                ).rowcount
                if not changed:
                    current = conn.execute(
                        "SELECT data FROM night_projects WHERE id=? AND brand=?", (project, brand)
                    ).fetchone()
                    if not current:
                        raise KeyError(project)
                    data = json.loads(current[0])
        return data

    def list(self, brand: str) -> list[dict]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT data FROM night_projects WHERE brand=?", (brand,)
            ).fetchall()
        return sorted(
            (self.get(brand, json.loads(row[0])["id"]) for row in rows),
            key=lambda r: r["created_at"],
            reverse=True,
        )

    def save(self, row: dict) -> None:
        row["updated_at"] = time.time()
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            prior = conn.execute(
                "SELECT data FROM night_projects WHERE id=? AND brand=?", (row["id"], row["brand"])
            ).fetchone()
            if not prior:
                raise KeyError(row["id"])
            # Never lose a concurrent control request or heartbeat while rendering a stage.
            current = json.loads(prior[0])
            row["pause_requested"] = current.get("pause_requested", False)
            row["cancel_requested"] = current.get("cancel_requested", False)
            if current.get("heartbeat_at", 0) > row.get("heartbeat_at", 0):
                row["heartbeat_at"] = current["heartbeat_at"]
            current_progress = current.get("stage_progress")
            saved_progress = row.get("stage_progress")
            if (
                current.get("current_stage") == row.get("current_stage")
                and isinstance(current_progress, dict)
                and current_progress.get("completed", -1)
                > (saved_progress.get("completed", -1) if isinstance(saved_progress, dict) else -1)
            ):
                row["stage_progress"] = current_progress
                row["last_progress_at"] = current["last_progress_at"]
            # A stale worker snapshot must not discard keepers or export requests
            # committed by another console transaction while a long stage ran.
            exports = {item.get("export_id"): item for item in row.get("private_exports", [])}
            for item in current.get("private_exports", []):
                exports.setdefault(item.get("export_id"), item)
            if exports:
                row["private_exports"] = list(exports.values())
            requests = {item.get("id"): item for item in row.get("private_export_requests", [])}
            for item in current.get("private_export_requests", []):
                requests.setdefault(item.get("id"), item)
            if requests:
                row["private_export_requests"] = list(requests.values())
            conn.execute(
                "UPDATE night_projects SET data=? WHERE id=? AND brand=?",
                (_canonical(row), row["id"], row["brand"]),
            )

    def heartbeat(
        self,
        brand: str,
        project: str,
        worker: dict,
        *,
        now: float | None = None,
    ) -> bool:
        """Refresh liveness only for the exact running worker identity."""
        now = time.time() if now is None else now
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            saved = conn.execute(
                "SELECT data FROM night_projects WHERE id=? AND brand=?", (project, brand)
            ).fetchone()
            if not saved:
                raise KeyError(project)
            row = json.loads(saved[0])
            if row.get("status") != "running" or row.get("worker") != worker:
                return False
            row["heartbeat_at"] = now
            conn.execute(
                "UPDATE night_projects SET data=? WHERE id=? AND brand=?",
                (_canonical(row), project, brand),
            )
        return True

    def progress(
        self,
        brand: str,
        project: str,
        worker: dict,
        stage: str,
        completed: int,
        total: int,
        *,
        now: float | None = None,
    ) -> bool:
        """Persist a monotonic stage milestone for the exact running worker."""
        if (
            type(completed) is not int
            or type(total) is not int
            or total <= 0
            or completed < 0
            or completed > total
        ):
            raise ValueError("invalid night stage progress")
        now = time.time() if now is None else now
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            saved = conn.execute(
                "SELECT data FROM night_projects WHERE id=? AND brand=?", (project, brand)
            ).fetchone()
            if not saved:
                raise KeyError(project)
            row = json.loads(saved[0])
            if (
                row.get("status") != "running"
                or row.get("worker") != worker
                or row.get("current_stage") != stage
            ):
                return False
            prior = row.get("stage_progress", {})
            if (
                isinstance(prior, dict)
                and prior.get("total") == total
                and isinstance(prior.get("completed"), int)
                and completed < prior["completed"]
            ):
                return False
            row.update(
                heartbeat_at=now,
                last_progress_at=now,
                stage_progress={"completed": completed, "total": total},
            )
            conn.execute(
                "UPDATE night_projects SET data=? WHERE id=? AND brand=?",
                (_canonical(row), project, brand),
            )
        return True

    def pause(self, brand: str, project: str, *, requested: bool = True) -> None:
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            record = conn.execute(
                "SELECT data FROM night_projects WHERE id=? AND brand=?", (project, brand)
            ).fetchone()
            if not record:
                raise KeyError(project)
            row = json.loads(record[0])
            row["pause_requested"] = requested
            conn.execute(
                "UPDATE night_projects SET data=? WHERE id=? AND brand=?",
                (_canonical(row), project, brand),
            )

    def cancel(self, brand: str, project: str, *, requested: bool = True) -> None:
        """Persist graceful cancellation without interrupting an active render."""
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            record = conn.execute(
                "SELECT data FROM night_projects WHERE id=? AND brand=?", (project, brand)
            ).fetchone()
            if not record:
                raise KeyError(project)
            row = json.loads(record[0])
            row["cancel_requested"] = requested
            if requested and row.get("status") != "running":
                _mark_cancelled(row)
            conn.execute(
                "UPDATE night_projects SET data=? WHERE id=? AND brand=?",
                (_canonical(row), project, brand),
            )

    def request_private_export(self, brand: str, project: str, task_index: int) -> dict:
        """Queue or explicitly retry one durable timelapse privacy export."""
        if type(task_index) is not int or task_index < 0:
            raise ValueError("unknown night product task")
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            saved = conn.execute(
                "SELECT data FROM night_projects WHERE id=? AND brand=?", (project, brand)
            ).fetchone()
            if not saved:
                raise KeyError(project)
            row = json.loads(saved[0])
            tasks = row.get("tasks", [])
            if (
                row.get("schema") != 2
                or row.get("kind") != "night_products"
                or task_index >= len(tasks)
                or tasks[task_index].get("product") != "timelapse"
                or tasks[task_index].get("status") not in {"complete", "partial"}
                or not tasks[task_index].get("output")
            ):
                raise ValueError("completed timelapse product required for private video export")
            request_id = hashlib.sha256(f"{project}:timelapse:{task_index}".encode()).hexdigest()
            requests = row.setdefault("private_export_requests", [])
            request = next((item for item in requests if item.get("id") == request_id), None)
            if request is None:
                request = {
                    "id": request_id,
                    "task_index": task_index,
                    "product": "timelapse",
                    "status": "queued",
                    "attempts": 0,
                    "created_at": time.time(),
                }
                requests.append(request)
            elif request.get("status") == "failed":
                request["status"] = "queued"
                request.pop("error", None)
            if request["status"] == "queued" and row.get("status") != "running":
                row["status"] = "queued"
            row["updated_at"] = time.time()
            conn.execute(
                "UPDATE night_projects SET data=? WHERE id=? AND brand=?",
                (_canonical(row), project, brand),
            )
        return json.loads(_canonical(request))

    def add_private_export(self, brand: str, project: str, record: dict) -> None:
        """Append one immutable, receipt-bound keeper without replacing worker state."""
        if (
            not isinstance(record, dict)
            or record.get("project_id") != project
            or not isinstance(record.get("export_id"), str)
            or len(record["export_id"]) != 64
            or any(character not in "0123456789abcdef" for character in record["export_id"])
            or record.get("publishable") is not False
        ):
            raise ValueError("invalid private night export record")
        for key in ("path", "receipt"):
            path = Path(record.get(key, ""))
            if path.is_absolute() or not path.parts or ".." in path.parts:
                raise ValueError("private export paths must be project-relative")
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            saved = conn.execute(
                "SELECT data FROM night_projects WHERE id=? AND brand=?", (project, brand)
            ).fetchone()
            if not saved:
                raise KeyError(project)
            row = json.loads(saved[0])
            if row.get("schema") != 2 or row.get("kind") != "night_products":
                raise ValueError("typed night product project required")
            task_index = record.get("task_index")
            if (
                type(task_index) is not int
                or task_index < 0
                or task_index >= len(row["tasks"])
                or row["tasks"][task_index].get("product") != record.get("product")
                or row["tasks"][task_index].get("status") not in {"complete", "partial"}
            ):
                raise ValueError("private export does not match a completed product task")
            exports = row.setdefault("private_exports", [])
            existing = next(
                (item for item in exports if item.get("export_id") == record["export_id"]), None
            )
            if existing is not None:
                if existing != record:
                    raise ValueError("private export identity already has different metadata")
                return
            exports.append(json.loads(_canonical(record)))
            row["updated_at"] = time.time()
            conn.execute(
                "UPDATE night_projects SET data=? WHERE id=? AND brand=?",
                (_canonical(row), project, brand),
            )


def admission(source: Path) -> bool:
    """Conservative current availability; does not claim an OS reservation."""

    manifest = json.loads((source / "manifest.json").read_text())
    shape = manifest["payloads"]["sky_sum.npy"]["shape"]
    pixels = int(shape[0]) * int(shape[1])
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(memory["MemAvailable"].split()[0]) * 1024
    return available >= pixels * 200 + 2 * 1024**3


class StorageWaitError(RuntimeError):
    """Structured capacity wait; immutable budget exhaustion is not transient."""

    def __init__(self, message: str, *, kind: str):
        super().__init__(message)
        self.kind = kind


def generated_bytes(target: Path, sources: list[Path]) -> int:
    """Count partial and complete outputs, excluding protected source trees."""
    protected = {source.resolve() for source in sources}
    total = 0

    def failed(error):
        raise error

    for directory, dirs, files in os.walk(target, followlinks=False, onerror=failed):
        base = Path(directory)
        if base.resolve() in protected:
            dirs[:] = []
            continue
        for name in [*dirs, *files]:
            path = base / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise ValueError("Storage accounting refuses symlink entries")
            if stat.S_ISREG(info.st_mode):
                total += info.st_size
            elif not stat.S_ISDIR(info.st_mode):
                raise ValueError("Storage accounting requires regular files")
    return total


def storage_preflight(row: dict, target: Path, source: Path, *, reserve=None) -> None:
    """Check existing bytes, then optional conservative TIFF/JPEG reservation.

    ``reserve=None`` checks usage only, allowing valid reuse on a full disk.
    This estimate is admission, not a filesystem quota against outside writers.
    """
    used = generated_bytes(target, [Path(item["source"]) for item in row["items"]])
    required = 0
    if reserve is not None:
        try:
            shape = json.loads((source / "manifest.json").read_text())["payloads"]["sky_sum.npy"][
                "shape"
            ]
            if (
                not isinstance(shape, list)
                or len(shape) != 3
                or shape[2] != 3
                or any(type(size) is not int or size <= 0 for size in shape)
            ):
                raise ValueError
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid linear stack dimensions for storage admission") from exc
        pixels = shape[0] * shape[1]
        required = pixels * (6 if reserve == "preview" else 12)
        required += (4 if reserve == "preview" else 16) * 1024**2
    budget = row.get("storage_budget_bytes")
    if budget is not None and used + required > budget:
        raise StorageWaitError(
            f"Waiting for project storage budget: {used} used + {required} reserved bytes "
            f"exceeds {budget}; preserve files or create a request with a larger budget.",
            kind="budget",
        )
    if required and shutil.disk_usage(target).free < required + 1024**3:
        raise StorageWaitError(
            f"Waiting for output free space: {required} reserved bytes plus 1 GiB headroom required.",
            kind="disk",
        )


def process_project(
    store: NightStore,
    brand: str,
    project: str,
    *,
    render=None,
    validate=None,
    recover=None,
    admitted=admission,
) -> int:
    """One resumable pass: 0 complete, 1 partial/failed, 2 waiting/paused.

    Caller may retry waiting passes with bounded backoff. A lock is held until
    every in-flight render stops; no lease expiry can admit a second writer.
    """
    row = store.get(brand, project)
    if row.get("schema") == 2 and row.get("kind") == "night_products":
        from .night_products import process_product_project

        return process_product_project(store, brand, project)
    if (
        row.get("build") != build_identity()
        or row.get("runtime") != runtime_identity()
        or row.get("interpreter") != sys.executable
    ):
        row.update(status="blocked", reason="Processing build changed; create a new request.")
        store.save(row)
        return 1
    if row.get("cancel_requested"):
        _mark_cancelled(row)
        store.save(row)
        return 1
    # All local workers for this OS user share admission, even across project roots.
    # Containers/other users must configure the same host-mounted lock explicitly.
    with media_admission() as host_admitted, (store.root / "heavy.lock").open("a") as lock:
        try:
            if not host_admitted:
                raise BlockingIOError("Another media worker holds admission")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            current = store.get(brand, project)
            if current["status"] not in {"running", "complete", "paused"}:
                current.update(
                    status="waiting",
                    wait_kind="admission",
                    reason="Another media worker holds admission.",
                )
                store.save(current)
            return 2
        if render is None or validate is None or recover is None:
            from .night_render import recover_preview, render_artifact, validate_render

            render = render or render_artifact
            validate = validate or validate_render
            recover = recover or recover_preview
        if row.get("cancel_requested"):
            _mark_cancelled(row)
            store.save(row)
            return 1
        if row.get("pause_requested"):
            row["status"] = "paused"
            store.save(row)
            return 2
        row["worker"] = {
            "pid": os.getpid(),
            "started_at": process_start(os.getpid()),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        }
        row["status"] = "running"
        row.pop("reason", None)
        row.pop("wait_kind", None)
        store.save(row)
        target = store.root / project
        if target.is_symlink():
            row.update(status="blocked", reason="Project output directory must not be a symlink.")
            store.save(row)
            return 1
        target.mkdir(exist_ok=True)
        for index, item in enumerate(row["items"]):
            controls = store.get(brand, project)
            if controls.get("cancel_requested"):
                _mark_cancelled(row)
                store.save(row)
                return 1
            if controls.get("pause_requested"):
                row["status"] = "paused"
                store.save(row)
                return 2
            source = Path(item["source"])
            try:
                if row["build"] != build_identity():
                    raise ValueError("Processing build changed during the run")
                _within(source, [Path(p) for p in row["source_roots"]])
                if digest(source / "manifest.json") != item["manifest_sha256"]:
                    raise ValueError("source manifest changed; create a new project")
                storage_preflight(row, target, source)
                if not admitted(source):
                    item["status"] = "waiting"
                    row["status"] = "waiting"
                    row["reason"] = "Waiting for memory; no input was skipped."
                    row["wait_kind"] = "memory"
                    store.save(row)
                    return 2
                # Discover receipt-bearing attempts even if the worker died before DB commit.
                reusable = None
                candidates = sorted(target.glob(f"{index}-*"), reverse=True)
                # Read-only reuse wins over recovery, regardless of UUID order.
                # A pending attempt must not demand space when a complete one exists.
                for candidate in candidates:
                    try:
                        if validate(source, candidate, row["recipe"]):
                            reusable = candidate
                            break
                    except (OSError, ValueError, KeyError):
                        pass
                for candidate in candidates if reusable is None else ():
                    try:
                        if (candidate / "preview-pending.json").exists() and not (
                            candidate / "receipt.json"
                        ).exists():
                            storage_preflight(row, target, source, reserve="preview")
                            recover(source, candidate, row["recipe"])
                            storage_preflight(row, target, source)
                            if validate(source, candidate, row["recipe"]):
                                reusable = candidate
                                break
                    except (OSError, ValueError, KeyError):
                        pass
                if reusable is not None:
                    if row["build"] != build_identity():
                        raise ValueError("Processing build changed during receipt adoption")
                    item.update(status="complete", output=str(reusable), error=None)
                    store.save(row)
                    continue
                storage_preflight(row, target, source, reserve="master")
                destination = target / f"{index}-{uuid.uuid4().hex}"
                item["attempts"] += 1
                item.update(status="running", error=None)
                store.save(row)
                render(source, destination, row["recipe"])
                storage_preflight(row, target, source)
                if not validate(source, destination, row["recipe"]):
                    raise ValueError("render returned without validated complete artifacts")
                if row["build"] != build_identity():
                    raise ValueError("Processing build changed during render")
                item.update(status="complete", output=str(destination), error=None)
            except StorageWaitError as exc:
                item["status"] = "waiting"
                row.update(status="waiting", reason=str(exc), wait_kind=exc.kind)
                store.save(row)
                return 2
            except Exception as exc:  # noqa: BLE001 - retain each failure and truthful aggregate
                item.update(status="failed", error=str(exc), output=None)
            store.save(row)
        completed = sum(item["status"] == "complete" for item in row["items"])
        row["status"] = (
            "complete" if completed == len(row["items"]) else "partial" if completed else "failed"
        )
        store.save(row)
        return 0 if row["status"] == "complete" else 1
