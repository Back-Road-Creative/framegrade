"""Read-only discovery of private night projects; never gallery registration."""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path


def read_night_projects(root: Path, brand: str) -> dict:
    """Return at most 50 project links, without media paths or pixel decoding.

    Status is recorded execution state, not fresh artifact or privacy validation.
    The private project routes remain responsible for verifying served previews.
    """
    result = {"projects": [], "unavailable": False}
    database = Path(root) / "projects.sqlite3"
    try:
        if any(path.is_symlink() for path in (database, *database.parents)):
            return {**result, "unavailable": True}
        if not database.exists():
            return result
        with closing(
            sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True, timeout=1)
        ) as db:
            rows = db.execute(
                "SELECT id, substr(data, 1, 1048577) FROM night_projects "
                "WHERE brand=? ORDER BY rowid DESC LIMIT 50",
                (brand,),
            ).fetchall()
        for identifier, raw in rows:
            if not re.fullmatch(r"[0-9a-f]{64}", identifier) or len(raw) > 1048576:
                result["unavailable"] = True
                continue
            try:
                row = json.loads(raw)
                if row["id"] != identifier or row["brand"] != brand:
                    raise ValueError("project identity mismatch")
                product_project = row.get("kind") == "night_products"
                if product_project:
                    if row.get("schema") != 2:
                        raise ValueError("invalid product project schema")
                    items = row["tasks"]
                    if not isinstance(items, list) or not all(
                        isinstance(item, dict) for item in items
                    ):
                        raise ValueError("invalid product tasks")
                    requested = row["requested_products"]
                    if (
                        not isinstance(requested, list)
                        or not all(isinstance(value, str) for value in requested)
                        or [item.get("product") for item in items] != requested
                    ):
                        raise ValueError("invalid requested products")
                else:
                    items = row["items"]
                if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
                    raise ValueError("invalid project work units")
                exports = row.get("private_exports", [])
                if not isinstance(exports, list) or not all(
                    isinstance(item, dict) for item in exports
                ):
                    raise ValueError("invalid private exports")
                result["projects"].append(
                    {
                        "id": identifier,
                        "recipe": str(row["recipe"]),
                        "status": str(row["status"]),
                        "completed": sum(item.get("status") == "complete" for item in items),
                        "total": len(items),
                        "kind": "products" if product_project else "stacks",
                        "selected": len(exports),
                    }
                )
            except (ValueError, TypeError, KeyError, RecursionError):
                result["unavailable"] = True
    except (OSError, sqlite3.Error, TypeError):
        result["unavailable"] = True
    return result
