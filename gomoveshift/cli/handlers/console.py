"""``gomoveshift console`` -- start the creator console (loopback only, no auth)."""

import logging


def cmd_console(args, config) -> int:
    """Start the console web server."""
    import uvicorn

    from ...console.app import create_app
    from ..logging_setup import configure_logging

    configure_logging(verbose=getattr(args, "verbose", False), log_file=None, json_log=False)
    app = create_app()

    logger = logging.getLogger(__name__)
    logger.info("Starting GoMoveShift console at http://%s:%s", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0
