"""Handlers for ``ingest`` and ``batch`` CLI commands."""

import sys

from ...paths import gms_staging_dir
from ..logging_setup import configure_logging
from ..tmux import is_in_tmux, launch_in_tmux, require_supervision, tmux_available


def cmd_ingest(args, config):
    """Scan GoPro directory, reassemble chaptered files, and plan episodes."""
    from ...pipeline.ingest import create_ingest_plan, execute_ingest

    configure_logging()

    directory = args.directory.resolve()
    if not directory.is_dir():
        print(f"Error: Not a directory: {directory}", file=sys.stderr)
        return 1

    print("=" * 60)
    print("  GoMoveShift Ingest \u2014 GoPro Multi-Clip Processing")
    print("=" * 60)
    print(f"  Source: {directory}")
    print()

    plan = create_ingest_plan(
        source_dir=directory,
        min_clip_duration=args.min_clip,
        max_episode_duration=args.max_episode * 60,
        max_gap=args.max_gap * 60,
    )

    print()
    print(plan.summary())
    print()

    if not plan.episodes:
        print("No episodes found.")
        return 1

    if not args.confirm:
        print(
            "Dry run \u2014 add --confirm to execute (concat files, extract GPS, derive locations)."
        )
        return 0

    output_dir = args.output_dir or (directory / "_episodes")
    print(f"Executing ingest \u2192 {output_dir}")
    print()

    execute_ingest(
        plan,
        output_dir=output_dir,
        derive_locations=not args.no_locations,
    )

    print()
    print("=" * 60)
    print(f"  Ingest complete: {len(plan.episodes)} episodes")
    print("=" * 60)
    for ep in plan.episodes:
        print(
            f"  Episode {ep.index}: {ep.total_duration_min:.0f}min "
            f"\u2192 {ep.concat_path.name if ep.concat_path else '?'}"
        )
        print(f"    Location: {ep.location or '(unknown)'}")
        print(f"    Date: {ep.date}")
        if ep.gpx_path:
            print(f"    GPS: {ep.gpx_path.name}")
    print()
    print("Next step: gomoveshift batch <directory> --publish mon --shorts-time 12:00")

    return 0


def cmd_batch(args, config):
    """Ingest GoPro directory and run full pipeline on each episode."""
    from datetime import time as dt_time

    from ...pipeline.ingest import (
        create_ingest_plan,
        execute_ingest,
        run_batch_pipeline,
    )
    from ...pipeline.longform_metadata import LongFormMetadataGenerator

    # ENFORCE: Long-running batch MUST run in tmux
    if not is_in_tmux():
        if args.no_tmux:
            # --no-tmux asserts "I am already supervised"; verified, not trusted.
            unsupervised = require_supervision("batch pipeline")
            if unsupervised is not None:
                return unsupervised
        elif tmux_available():
            return launch_in_tmux(sys.argv, session_name="gms-batch")
        else:
            print(
                "\u2717 ERROR: tmux required for batch pipeline. Install tmux or use --no-tmux.",
                file=sys.stderr,
            )
            return 1

    directory = args.directory.resolve()
    if not directory.is_dir():
        print(f"Error: Not a directory: {directory}", file=sys.stderr)
        return 1

    # Set up logging to both stdout and a log file in the source directory
    log_file = directory / "_episodes" / "batch.log"
    configure_logging(log_file=log_file, json_log=getattr(args, "log_json", False))

    # Parse shorts time
    try:
        h, m = args.shorts_time.split(":")
        shorts_time = dt_time(int(h), int(m), 0)
    except (ValueError, AttributeError):
        print(f"Error: Invalid --shorts-time format: {args.shorts_time}", file=sys.stderr)
        return 1

    print("=" * 60)
    print("  GoMoveShift Batch \u2014 GoPro Directory Pipeline")
    print("=" * 60)
    print(f"  Source:       {directory}")
    print(f"  Publish day:  {args.publish}")
    print(f"  Shorts time:  {args.shorts_time}")
    print(f"  Max episode:  {args.max_episode} min")
    print(f"  Output:       {args.output_dir}")
    if args.start_date:
        print(f"  Start date:   {args.start_date}")
    if args.draft:
        print("  Mode:         DRAFT")
    print("=" * 60)
    print()

    # Step 1: Ingest
    print("Phase 1: Scanning and grouping GoPro clips...")
    plan = create_ingest_plan(
        source_dir=directory,
        max_episode_duration=args.max_episode * 60,
    )

    if not plan.episodes:
        print("No episodes found.")
        return 1

    print(plan.summary())
    print()

    # Use Linux-native filesystem for intermediates when source is on a cross-mount
    # (a Windows drive mounted into WSL). Avoids a large I/O penalty on NTFS.
    if str(directory).startswith("/mnt/"):
        episodes_dir = gms_staging_dir() / "_episodes"
        print("  Cross-mount source detected \u2014 using Linux FS for intermediates")
    else:
        episodes_dir = directory / "_episodes"
    print(
        f"Processing {len(plan.episodes)} episodes (ingest \u2192 pipeline \u2192 cleanup per episode)..."
    )
    print()

    # Ingest all episodes first (concat + GPS + location)
    # GPS extraction and location derivation need all episodes done
    # before pipeline starts, so scheduling dates are stable
    print("Phase 1b: Concatenating chapters, extracting GPS...")
    execute_ingest(plan, output_dir=episodes_dir)

    print()
    print(f"Phase 2: Running pipeline on {len(plan.episodes)} episodes...")
    print()

    # Step 2: Run pipeline per episode
    results = run_batch_pipeline(
        plan,
        publish_day=args.publish,
        shorts_time=shorts_time,
        output_base=args.output_dir,
        start_date=args.start_date,
        skip_color=args.skip_color,
        skip_stops=args.skip_stops,
        draft=args.draft,
        metadata_generator=LongFormMetadataGenerator,
    )

    # Summary
    succeeded = sum(1 for r in results if r.get("success"))
    total = len(results)
    print()
    print("=" * 60)
    print(f"  Batch complete: {succeeded}/{total} episodes succeeded")
    print("=" * 60)
    summary_lines = []
    for r in results:
        status = "\u2713" if r.get("success") else "\u2717"
        ep_num = r.get("episode", "?")
        pub = r.get("publish_date", "")
        out = r.get("output_dir", r.get("error", ""))
        line = f"  {status} Episode {ep_num} ({pub}): {out}"
        print(line)
        summary_lines.append(line)

    # Write summary report file
    report_path = episodes_dir / "batch_report.txt"
    try:
        with open(report_path, "w") as f:
            f.write("GoMoveShift Batch Report\n")
            f.write(f"Source: {directory}\n")
            f.write(f"Result: {succeeded}/{total} episodes succeeded\n")
            if succeeded < total:
                f.write(f"STATUS: FAILED ({total - succeeded} episodes failed)\n")
            else:
                f.write("STATUS: SUCCESS\n")
            f.write("\n")
            for line in summary_lines:
                f.write(f"{line}\n")
        print(f"\n  Report: {report_path}")
        print(f"  Log:    {log_file}")
    except Exception:
        pass

    return 0 if succeeded == total else 1
