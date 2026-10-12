"""Spool a publish run's shorts into the delivery queue with no operator step.

Phase D writes ``<project>_publish_plan.json``; the hourly
``dispatch-queue --apply`` cron drains the queue. Before this module nothing
connected the two, so shorts only reached a platform when someone ran
``gomoveshift enqueue-plan --apply`` by hand. Shorts are meant to be
automatic, so a successful publish run now enqueues its own plan.

The automated pre-spool gates in ``cmd_enqueue_plan`` (media verification,
privacy, payload limits, rolling ceiling and brand cap, batch anti-spam,
plan conflicts) still decide what is spooled. Their refusal is reported as a
run error so it is loud, never a silent skip.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


def should_auto_enqueue(state: Any) -> bool:
    """True for a run that published: Phase D done, upload opted in, not skipped."""
    phases = getattr(state, "phases_completed", None) or []
    return (
        "phase_d" in phases
        and bool(getattr(state, "youtube_upload", None))
        and not getattr(state, "skip_upload", None)
        and not getattr(state, "skip_youtube_upload", None)
    )


def auto_enqueue_shorts(
    state: Any, state_file: Path, config: Any, errors: List[str]
) -> Optional[int]:
    """Run ``enqueue-plan --apply`` for this run's state when it published.

    Returns the handler's exit code, or ``None`` when the run did not
    publish. A nonzero code or an exception is appended to ``errors``."""
    if not should_auto_enqueue(state):
        return None
    from gomoveshift.cli.handlers.enqueue_plan import cmd_enqueue_plan

    args = SimpleNamespace(
        state_path=str(state_file),
        apply=True,
        output_dir=None,  # handler resolves state.output_base, where Phase D writes
        max_shorts=None,
        shadow=False,
    )
    try:
        rc = cmd_enqueue_plan(args, config)
    except Exception as exc:  # noqa: BLE001 - a crash must fail the run loudly
        logger.exception("auto-enqueue of shorts crashed")
        errors.append(f"auto-enqueue of shorts crashed: {exc}")
        return 1
    if rc != 0:
        errors.append(
            f"auto-enqueue of shorts refused (enqueue-plan exit {rc}); see the "
            "enqueue-plan output above -- no shorts from this run are queued"
        )
    else:
        logger.info("auto-enqueue: shorts spooled for the hourly dispatch cron")
    return rc
