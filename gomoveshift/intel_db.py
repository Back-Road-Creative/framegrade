"""IntelDB integration helpers (extracted from legacy ``gomoveshift.main``).

Non-fatal best-effort logging of processed videos to an IntelDB instance, if
the ``INTEL_DB_PATH`` env var points to a live database and the ``intel_db``
module is importable. ``INTEL_SCRIPTS_DIR`` (optional, no default) names a
directory to add to ``sys.path`` when the module is not installed.
"""

import logging
import os
import sys

logger = logging.getLogger(__name__)

_INTEL_SCRIPTS_DIR = os.getenv("INTEL_SCRIPTS_DIR", "")


def _get_intel_db(db_path: str):
    """Load IntelDB dynamically; returns None on any failure."""
    if _INTEL_SCRIPTS_DIR and _INTEL_SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, _INTEL_SCRIPTS_DIR)
    try:
        from intel_db import IntelDB  # noqa: PLC0415

        return IntelDB(db_path)
    except ImportError as exc:
        logger.warning("Could not import IntelDB: %s", exc)
        return None
    except Exception as exc:
        logger.warning("Could not open IntelDB at %r: %s", db_path, exc)
        return None


def _log_video_to_intel_db(video_name: str, status: str) -> None:
    """Record a processed video in IntelDB (non-fatal)."""
    db_path = os.getenv("INTEL_DB_PATH", "")
    if not db_path:
        return
    db = _get_intel_db(db_path)
    if db is None:
        return
    try:
        import datetime as _dt  # noqa: PLC0415

        run_id = db.start_scrape_run("gomoveshift-video", source_name=video_name)
        date_str = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
        summary = f"GoMoveShift video {status}: {video_name}"
        db.upsert_news_item(date=date_str, summary=summary, article_links=[], source_count=1)
        db.finish_scrape_run(
            run_id,
            items_found=1,
            items_imported=1 if status == "completed" else 0,
        )
    except Exception as exc:
        logger.warning("IntelDB write failed for %s: %s", video_name, exc)
    finally:
        db.close()
