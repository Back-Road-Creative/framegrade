"""Metadata-only, bounded night retry pass for the existing console ticker."""

from __future__ import annotations

import time
from pathlib import Path

from ..pipeline.photos.night_project import NightStore
from . import night_workers, settings


def supervise(root=None, brand=None, *, now=None):
    root = Path(root) if root is not None else settings.console_data_dir() / "night-projects"
    brand = brand or settings.current_brand()
    now = time.time() if now is None else now
    if brand != settings.current_brand() or not (root / "supervision.sqlite3").exists():
        return 0
    if "photos" in settings.brand_disabled_tabs(brand):
        return 0
    with night_workers.state_db(root) as db:
        rows = db.execute(
            "SELECT id FROM retries WHERE brand=? AND enabled=1 AND next_due<=? ORDER BY next_due, rowid LIMIT 20",
            (brand, now),
        ).fetchall()
    store = NightStore(root)
    launched = 0
    for (project,) in rows:
        try:
            if night_workers.launch(store, project, automatic=True, now=now):
                launched += 1
        except (OSError, ValueError, TypeError, KeyError, RecursionError, AttributeError) as exc:
            # Preserve launch uncertainty; corrupt metadata never implies a new launch.
            # The bounded error note remains visible on the project screen.
            try:
                state = night_workers.read_state(root, brand, project)
            except (ValueError, TypeError, RecursionError):
                state = None
            state = state or {
                "enabled": False,
                "launches": 0,
                "crash_retries": 0,
                "next_retry_at": now + 3600,
                "claim": "launching",
                "jobs_floor": 0,
            }
            state.update(
                enabled=False,
                last_error=str(exc)[:500],
                observation="Supervision metadata/launch error; manual review required.",
            )
            night_workers._save(root, brand, project, state)
        if launched >= 1:
            break
    return launched
