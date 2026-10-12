"""Supported CLI adapter for private, resumable night redevelopment."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from ...pipeline.photos.night_budget import budget_bytes
from ...pipeline.photos.night_collection import ProductRequest, scan_collection
from ...pipeline.photos.night_project import NightStore, process_project


def cmd_night(args) -> int:
    """Run a bounded worker; paused/resource waits exit 2, never false success."""
    try:
        wait = float(getattr(args, "wait_seconds", 600))
        poll = float(getattr(args, "poll_seconds", 15))
        if not math.isfinite(wait) or not math.isfinite(poll) or wait < 0 or poll <= 0:
            raise ValueError("wait must be finite and nonnegative; poll must be positive")
        controls = [args.pause, getattr(args, "cancel", False), args.resume, args.status]
        if sum(bool(value) for value in controls) > 1:
            raise ValueError("choose only one of pause, cancel, resume, status")
        collection_path = getattr(args, "collection", None)
        products = getattr(args, "products", None)
        sequence_index = getattr(args, "sequence_index", 0)
        inputs = [args.project, args.sources, getattr(args, "scan", None), collection_path]
        if sum(bool(value) for value in inputs) != 1:
            raise ValueError(
                "provide one of an existing project, source stacks, collection request, or scan"
            )
        if getattr(args, "scan", None):
            if (
                any(controls)
                or getattr(args, "storage_budget_gib", None) is not None
                or products
                or sequence_index
            ):
                raise ValueError("collection scan cannot use project controls or request options")
            print(json.dumps(scan_collection(Path(args.scan)).as_dict(), sort_keys=True))
            return 0
        budget_option = getattr(args, "storage_budget_gib", None)
        if args.project and budget_option is not None:
            raise ValueError(
                "storage budget belongs to a new request; existing projects are immutable"
            )
        budget = budget_bytes(budget_option)
        store = NightStore(Path(args.workspace))
        project = args.project
        if not project:
            if any(controls) or not args.source_root:
                raise ValueError("new projects require source roots and no control flags")
            if collection_path:
                if not products:
                    raise ValueError("collection requests require at least one --products value")
                collection = scan_collection(Path(collection_path))
                project = store.create_product_request(
                    args.brand,
                    collection,
                    [Path(p) for p in args.source_root],
                    ProductRequest.create(products),
                    args.recipe,
                    sequence_index=sequence_index,
                    storage_budget_bytes=budget,
                )
            else:
                if products or sequence_index:
                    raise ValueError("product and sequence options require --collection")
                project = store.create(
                    args.brand,
                    [Path(p) for p in args.sources],
                    [Path(p) for p in args.source_root],
                    args.recipe,
                    storage_budget_bytes=budget,
                )
        if args.pause:
            store.pause(args.brand, project)
            return 0
        if getattr(args, "cancel", False):
            store.cancel(args.brand, project)
            return 0
        if args.status:
            print(json.dumps(store.get(args.brand, project), sort_keys=True))
            return 0
        if args.resume:
            store.pause(args.brand, project, requested=False)
            store.cancel(args.brand, project, requested=False)
        deadline = time.monotonic() + wait
        print(f"Night project: {project}", flush=True)
        while True:
            code = process_project(store, args.brand, project)
            if code != 2:
                return code
            current = store.get(args.brand, project)
            if current.get("cancel_requested") or current.get("status") == "cancelled":
                return 1
            if current.get("pause_requested"):
                return 2
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                print("Project remains waiting; resume with the same project ID.", flush=True)
                return 2
            time.sleep(min(poll, remaining))
    except (OSError, ValueError, KeyError) as exc:
        print(f"Night project failed: {exc}", flush=True)
        return 1
