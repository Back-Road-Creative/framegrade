"""Handler for the ``pipeline`` CLI command."""

import logging
import signal
import sys

from ..logging_setup import configure_logging
from ..tmux import is_in_tmux, launch_in_tmux, require_supervision, tmux_available

logger = logging.getLogger(__name__)


def _handle_sigterm(signum, frame):
    """Handle SIGTERM by raising SystemExit so try/finally cleanup runs."""
    logger.warning("SIGTERM received, shutting down...")
    raise SystemExit(143)


def cmd_pipeline(args, config):
    """Run the full long form processing pipeline."""
    from ...config import ConfigError
    from ...pipeline import PipelineOrchestrator

    # Fail-fast SEO config validation. See ``cmd_drive`` for
    # the full rationale. Runs BEFORE tmux re-exec so a missing SEO
    # service checkout surfaces in the operator's shell, not
    # mid-Phase D after hours of encode.
    try:
        config.seo.validate_paths_exist()
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # ENFORCE: Long-running pipelines MUST run in tmux for persistence
    if not is_in_tmux():
        if args.no_tmux:
            # --no-tmux asserts "I am already supervised"; verified, not trusted.
            unsupervised = require_supervision("pipeline")
            if unsupervised is not None:
                return unsupervised
        elif tmux_available():
            # Re-exec in tmux
            return launch_in_tmux(sys.argv, session_name="gms-pipeline")
        else:
            # tmux not available and user didn't opt out - fail safe
            print(
                "\u2717 ERROR: tmux is required for long-running pipelines but is not available.",
                file=sys.stderr,
            )
            print(
                "   Install tmux or use --no-tmux to run without persistence (not recommended).",
                file=sys.stderr,
            )
            return 1

    video_path = args.video

    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}", file=sys.stderr)
        return 1

    # Set up logging to both stdout and a log file in the output directory
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "pipeline.log"
    configure_logging(log_file=log_file, json_log=getattr(args, "log_json", False))

    print("=" * 60)
    print("  GoMoveShift Pipeline \u2014 Long Form Processing")
    print("=" * 60)
    print(f"  Video:    {video_path}")
    print(f"  Location: {args.location or '(auto-detect from GPS)'}")
    print(f"  Publish:  {args.publish}")
    if args.lut:
        print(f"  LUT:      {args.lut}")
    if args.draft:
        print("  Mode:     DRAFT (same quality, _DRAFT filename stamp)")
    print("=" * 60)

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGHUP, _handle_sigterm)

    orchestrator = PipelineOrchestrator(config)

    try:
        from ...pipeline.orchestrator import RunOptions

        options = RunOptions.from_legacy(
            video_path=video_path,
            location=args.location,
            publish_day=args.publish,
            output_base=args.output_dir,
            lut_path=args.lut,
            crossfade_duration=args.crossfade,
            skip_color=args.skip_color,
            skip_stops=args.skip_stops,
            fresh=args.fresh,
            skip_thumbnails=args.skip_thumbnails,
            skip_shorts=args.skip_shorts,
            skip_schedule=args.skip_schedule,
            skip_social=args.skip_social,
            skip_upload=args.skip_upload,
            skip_csv=args.skip_csv,
            skip_d_copy=args.skip_d_copy,
            skip_chapters=args.skip_chapters,
            gps_path=args.gps,
            gps_date=getattr(args, "gps_date", None),
            gps_range=getattr(args, "gps_range", None),
            draft=args.draft,
            skip_cleanup=getattr(args, "no_cleanup", False),
        )
        result = orchestrator.run(options)
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        return 130
    except SystemExit as e:
        print(f"\n\nPipeline terminated (exit code {e.code}).")
        return e.code
    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        return 1

    if result.success:
        print("\n\u2705 Pipeline complete!")
        print(f"   Output: {result.output_dir}")
        print(f"   Final:  {result.final_video}")
        if result.metadata_file:
            print(f"   Meta:   {result.metadata_file}")
        return 0
    else:
        print("\n\u274c Pipeline failed!")
        for error in result.errors:
            print(f"   Error: {error}", file=sys.stderr)
        return 1
