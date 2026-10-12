"""GoMoveShift command-line interface package.

This package replaces the monolithic ``cli.py`` module.  All public names
that were importable from ``gomoveshift.cli`` in the old layout are
re-exported here so that existing tests and call-sites continue to work
without modification.
"""

import sys

# Re-export config helper that some handlers reference via the old module path
from ..config import get_config
from .handlers.analysis import cmd_analyze, cmd_find_stops, cmd_process, cmd_scan
from .handlers.cleanup import cmd_cleanup
from .handlers.comments import cmd_comments
from .handlers.console import cmd_console
from .handlers.dispatch_queue import cmd_dispatch_queue
from .handlers.doctor import cmd_doctor
from .handlers.drive import cmd_drive
from .handlers.enqueue_asset import cmd_enqueue_asset
from .handlers.enqueue_plan import cmd_enqueue_plan
from .handlers.ingest import cmd_batch, cmd_ingest
from .handlers.library_hygiene import cmd_library_hygiene
from .handlers.metrics import cmd_metrics
from .handlers.photos import cmd_photos
from .handlers.pipeline import cmd_pipeline
from .handlers.platform_decline import cmd_platform_decline
from .handlers.post_anomaly import cmd_post_anomaly
from .handlers.pub_status import cmd_publishing_status
from .handlers.publish_journal import cmd_publish_journal
from .handlers.reconcile_playlists import cmd_reconcile_playlists
from .handlers.route_story_film import cmd_route_story_film
from .handlers.route_story_prepare import cmd_route_story_prepare
from .handlers.route_story_proof import cmd_route_story_proof
from .handlers.route_story_verify import cmd_route_story_verify
from .handlers.schedule import cmd_schedule
from .handlers.sync_site import cmd_sync_site
from .handlers.upload import cmd_upload
from .handlers.youtube_auth import cmd_youtube_auth
from .parser import _build_parser
from .tmux import is_in_tmux, launch_in_tmux, tmux_available

__all__ = [
    # Entry point
    "main",
    "_build_parser",
    # tmux helpers (tested directly)
    "is_in_tmux",
    "tmux_available",
    "launch_in_tmux",
    # Command handlers
    "cmd_analyze",
    "cmd_batch",
    "cmd_cleanup",
    "cmd_console",
    "cmd_comments",
    "cmd_drive",
    "cmd_find_stops",
    "cmd_ingest",
    "cmd_metrics",
    "cmd_photos",
    "cmd_pipeline",
    "cmd_platform_decline",
    "cmd_post_anomaly",
    "cmd_process",
    "cmd_dispatch_queue",
    "cmd_enqueue_asset",
    "cmd_enqueue_plan",
    "cmd_publishing_status",
    "cmd_publish_journal",
    "cmd_reconcile_playlists",
    "cmd_route_story_prepare",
    "cmd_route_story_proof",
    "cmd_route_story_verify",
    "cmd_route_story_film",
    "cmd_scan",
    "cmd_schedule",
    "cmd_sync_site",
    "cmd_upload",
    "cmd_library_hygiene",
    "cmd_doctor",
    "cmd_youtube_auth",
]


# ── Command dispatch table ──────────────────────────────────────────────


_COMMAND_MAP = {
    "analyze": cmd_analyze,
    "process": cmd_process,
    "server": None,  # handled inline — lightweight
    "console": cmd_console,
    "scan": cmd_scan,
    "pipeline": cmd_pipeline,
    "drive": cmd_drive,
    "upload": cmd_upload,
    "cleanup": cmd_cleanup,
    "comments": cmd_comments,
    "find-stops": cmd_find_stops,
    "ingest": cmd_ingest,
    "batch": cmd_batch,
    "photos": cmd_photos,
    "publish-journal": cmd_publish_journal,
    "dispatch-queue": cmd_dispatch_queue,
    "enqueue-plan": cmd_enqueue_plan,
    "enqueue-asset": cmd_enqueue_asset,
    "publishing-status": cmd_publishing_status,
    "metrics": cmd_metrics,
    "platform-decline": cmd_platform_decline,
    "post-anomaly": cmd_post_anomaly,
    "reconcile-playlists": cmd_reconcile_playlists,
    "route-story-prepare": cmd_route_story_prepare,
    "route-story-proof": cmd_route_story_proof,
    "route-story-verify": cmd_route_story_verify,
    "route-story-film": cmd_route_story_film,
    "schedule": cmd_schedule,
    "sync-site": cmd_sync_site,
    "library-hygiene": cmd_library_hygiene,
    "doctor": cmd_doctor,
    "youtube-auth": cmd_youtube_auth,
}


def _cmd_server(args, config):
    """Start the web UI server (kept inline because it's tiny)."""
    import logging

    import uvicorn

    from ..web import create_app
    from ..web.app import _assert_localhost_only
    from .logging_setup import configure_logging

    # Wire centralized logging so unattended
    # server runs don't drop INFO output. Without this, uvicorn-only logging
    # leaves the rest of the GMS package silent.
    configure_logging(
        verbose=getattr(args, "verbose", False),
        log_file=getattr(args, "log_file", None),
        json_log=getattr(args, "log_json", False),
    )

    # Refuse non-loopback bind unless --allow-remote was passed.
    allow_remote = getattr(args, "allow_remote", False)
    _assert_localhost_only(args.host, allow_remote=allow_remote)

    config.web_ui.host = args.host
    config.web_ui.port = args.port

    # Widen CORS only when the operator explicitly opted into a remote bind.
    app = create_app(config, allow_remote=allow_remote)

    logger = logging.getLogger(__name__)
    logger.info("Starting GoMoveShift web UI at http://%s:%s", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)

    return 0


def main():
    """Main CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    config = get_config()
    # Push operator-resolved cadence into schedule_constants so every
    # downstream lazy-import reader (state_invariants, quality_gates,
    # scheduler, longform_metadata, content_ledger) sees the ~/.config
    # override instead of the DriveConfig defaults snapshot at import
    # time. Without this, a Friday override in
    # ~/.config/gomoveshift/config.yaml loads into Config correctly but
    # state_invariants still reads the module-level Wed default and
    # crashes with StateInvariantError.
    from ..pipeline import schedule_constants as _schedule_constants

    _schedule_constants.apply_config(config)

    # Special-case server (too small for its own handler module)
    if args.command == "server":
        return _cmd_server(args, config)

    handler = _COMMAND_MAP.get(args.command)
    if handler is not None:
        return handler(args, config)

    return 0


if __name__ == "__main__":
    sys.exit(main())
