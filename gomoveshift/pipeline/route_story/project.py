"""Durable, brand-scoped Route Story project records.

Projects persist the exact source identities and storyboard recipe before a
preview or renderer can consume them.  This keeps retries and later exports
bound to the material an operator actually reviewed.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _source_identity(path: Path, roots: Sequence[Path]) -> dict[str, object]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("Route Story sources must be regular files")
    resolved = candidate.resolve(strict=True)
    allowed = [Path(root).resolve(strict=True) for root in roots if Path(root).exists()]
    if not any(resolved.is_relative_to(root) for root in allowed):
        raise ValueError("Route Story source escapes configured roots")
    with resolved.open("rb") as source:
        sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    return {
        "path": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": sha256,
    }


class RouteStoryStore:
    """Private draft projects, partitioned by brand."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS route_story_projects ("
                "id TEXT PRIMARY KEY, brand TEXT NOT NULL, data TEXT NOT NULL)"
            )

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.root / "projects.sqlite3", timeout=30)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def create(
        self, brand: str, sources: Sequence[Path], roots: Sequence[Path], recipe: dict[str, object]
    ) -> str:
        if not brand or not sources or not isinstance(recipe, dict) or not recipe:
            raise ValueError("brand, sources, and a nonempty recipe are required")
        inputs = sorted(
            (_source_identity(Path(source), roots) for source in set(sources)),
            key=lambda item: str(item["path"]),
        )
        identity = {"schema": 1, "brand": brand, "sources": inputs, "recipe": recipe}
        project = hashlib.sha256(_canonical(identity).encode()).hexdigest()
        row = {
            **identity,
            "id": project,
            "status": "draft",
            "privacy_status": "review_required",
            "created_at": time.time(),
            "updated_at": time.time(),
            "source_roots": [str(Path(root).resolve()) for root in roots],
        }
        with self.connection() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO route_story_projects VALUES (?, ?, ?)",
                (project, brand, _canonical(row)),
            )
        return project

    def get(self, brand: str, project: str) -> dict[str, object]:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT data FROM route_story_projects WHERE id=? AND brand=?", (project, brand)
            ).fetchone()
        if not row:
            raise KeyError(project)
        return json.loads(row[0])

    def update(self, brand: str, project: str, **fields: object) -> dict[str, object]:
        """Persist reviewer decisions (e.g. media placements) so a re-render never re-asks."""
        if not fields:
            raise ValueError("update requires at least one field")
        with self.connection() as connection:
            row = connection.execute(
                "SELECT data FROM route_story_projects WHERE id=? AND brand=?", (project, brand)
            ).fetchone()
            if not row:
                raise KeyError(project)
            data = json.loads(row[0])
            data.update(fields)
            data["updated_at"] = time.time()
            connection.execute(
                "UPDATE route_story_projects SET data=? WHERE id=? AND brand=?",
                (_canonical(data), project, brand),
            )
        return data

    def list(self, brand: str) -> list[dict[str, object]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT data FROM route_story_projects WHERE brand=?", (brand,)
            ).fetchall()
        return sorted(
            (json.loads(row[0]) for row in rows), key=lambda item: item["created_at"], reverse=True
        )
