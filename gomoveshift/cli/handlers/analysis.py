"""Handlers for analysis-related CLI commands.

Covers: analyze, process, scan, find-stops
"""

import sys
import time
from datetime import timedelta

from ...analysis.pipeline import AnalysisPipeline
from ...assembly.assembler import VideoAssembler


def cmd_analyze(args, config):
    """Run analysis on a video."""
    video_path = args.video

    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}", file=sys.stderr)
        return 1

    pipeline = AnalysisPipeline(config)

    print(f"Analyzing: {video_path}")
    print("This may take a while...")

    def progress_callback(stage, progress):
        bar_width = 30
        filled = int(bar_width * progress)
        bar = "=" * filled + "-" * (bar_width - filled)
        print(f"\r{stage}: [{bar}] {progress * 100:.0f}%", end="", flush=True)
        if progress >= 1.0:
            print()

    try:
        analysis = pipeline.analyze(
            video_path,
            progress_callback=progress_callback,
            save_frames=not args.no_frames,
        )
    except Exception as e:
        print(f"\nError during analysis: {e}", file=sys.stderr)
        return 1

    # Save analysis
    output_path = args.output
    if output_path is None:
        output_path = video_path.with_suffix(".analysis.json")

    analysis.save(output_path)
    print(f"\nAnalysis saved to: {output_path}")

    # Print summary
    print("\nSummary:")
    print(f"  Duration: {analysis.processing_stats.get('frames_processed', 0)} frames processed")
    print(f"  Locations found: {len(analysis.locations)}")
    print(f"  Dead spots found: {len(analysis.dead_spots)}")
    print(f"  Suggested cuts: {len(analysis.suggested_cuts)}")

    if analysis.locations:
        print("\nDetected locations:")
        for loc in analysis.locations[:10]:
            print(f"  {loc.timestamp_formatted}: {loc.location_name} ({loc.location_type.value})")
        if len(analysis.locations) > 10:
            print(f"  ... and {len(analysis.locations) - 10} more")

    return 0


def cmd_process(args, config):
    """Process video using analysis file."""
    from ...models import VideoAnalysis

    video_path = args.video
    analysis_path = args.analysis

    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}", file=sys.stderr)
        return 1

    if not analysis_path.exists():
        print(f"Error: Analysis file not found: {analysis_path}", file=sys.stderr)
        return 1

    analysis = VideoAnalysis.load(analysis_path)
    assembler = VideoAssembler(config)

    output_dir = args.output_dir or video_path.parent / "output"

    print(f"Processing: {video_path}")
    print(f"Output directory: {output_dir}")

    plan = assembler.create_assembly_plan(analysis, output_dir=output_dir)

    print(f"Segments to keep: {len(plan.segments_to_keep)}")
    print(f"Cut points: {len(plan.cut_points)}")
    print(f"Output files planned: {len(plan.output_paths)}")

    def progress_callback(stage, progress, filename):
        bar_width = 30
        filled = int(bar_width * progress)
        bar = "=" * filled + "-" * (bar_width - filled)
        print(f"\r{stage} [{bar}] {progress * 100:.0f}% - {filename}", end="", flush=True)
        if progress >= 1.0:
            print()

    try:
        output_files = assembler.execute(plan, progress_callback)
    except Exception as e:
        print(f"\nError during processing: {e}", file=sys.stderr)
        return 1

    print(f"\nCreated {len(output_files)} output file(s):")
    for f in output_files:
        print(f"  {f}")

    # This VideoAssembler path runs NO stop/audio/blur validation. Mark the
    # outputs EXPLICITLY ungated so they aren't mistaken for gated drive-pipeline
    # outputs — this is the manual, single-video `process` surface.
    print(
        "\n[UNGATED] These outputs bypass the stop/audio/blur validation gates "
        "(manual `process` path). Do NOT treat them as gated for upload.",
        file=sys.stderr,
    )

    return 0


def cmd_scan(args, config):
    """Quick scan of video file."""
    video_path = args.video

    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}", file=sys.stderr)
        return 1

    pipeline = AnalysisPipeline(config)
    info = pipeline.quick_scan(video_path)

    print(f"Video: {video_path.name}")
    print(f"  Duration: {info['duration']:.2f}s ({info['duration'] / 60:.1f} minutes)")
    print(f"  Resolution: {info['width']}x{info['height']}")
    print(f"  FPS: {info['fps']:.2f}")
    print(f"  Total frames: {info['total_frames']}")

    return 0


def cmd_find_stops(args, config):
    """Scan all videos in a directory for stops."""
    import json

    from ...analysis.motion import MotionAnalyzer
    from ...resume import (
        get_checkpoint_state,
        get_videos_to_process,
        merge_results,
        print_resume_status,
    )

    input_dir = args.input_dir
    force_fresh = getattr(args, "fresh", False)

    if not input_dir.exists():
        print(f"Error: Directory not found: {input_dir}", file=sys.stderr)
        return 1

    # Find ALL videos first (for total count)
    extensions = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".mxf"}
    all_videos = []
    for ext in extensions:
        all_videos.extend(input_dir.glob(f"*{ext}"))
        all_videos.extend(input_dir.glob(f"*{ext.upper()}"))
    all_videos = sorted(all_videos, key=lambda v: v.name)
    total_count = len(all_videos)

    if not all_videos:
        print(f"No video files found in {input_dir}")
        return 0

    # Get videos to process (respecting checkpoint unless --fresh)
    videos = get_videos_to_process(input_dir, force_fresh=force_fresh)

    # Show resume status if resuming
    print_resume_status(input_dir, total_count)

    print(f"Found {total_count} videos total, {len(videos)} to process")
    print(
        f"Settings: threshold={config.dead_spots.motion_threshold}, "
        f"min_duration={config.dead_spots.min_duration_seconds}s"
    )
    print("=" * 60)

    if not videos:
        print("\nAll videos already processed!")
        return 0

    analyzer = MotionAnalyzer(config)
    new_results = {}  # Only new results from this run
    total_stops = 0

    # Get starting index for display (account for already-completed)
    state = get_checkpoint_state(input_dir)
    start_index = len(state["completed_videos"]) + 1

    for i, video in enumerate(videos, start_index):
        size_gb = video.stat().st_size / (1024**3)
        print(f"\n[{i}/{total_count}] {video.name} ({size_gb:.1f} GB)")

        start_time = time.time()
        last_progress = 0.0

        def progress_callback(current_frame, total_frames, progress):
            nonlocal last_progress
            # Only update every 5% to avoid spam
            if progress - last_progress >= 0.05 or progress >= 0.99:
                elapsed = time.time() - start_time
                eta = (elapsed / progress) - elapsed if progress > 0 else 0

                bar_width = 30
                filled = int(bar_width * progress)
                bar = "\u2588" * filled + "\u2591" * (bar_width - filled)

                elapsed_str = str(timedelta(seconds=int(elapsed)))
                eta_str = str(timedelta(seconds=int(eta)))

                print(
                    f"\r  Analyzing: [{bar}] {progress * 100:.1f}% | "
                    f"Frame {current_frame:,}/{total_frames:,} | "
                    f"Time: {elapsed_str} | ETA: {eta_str}",
                    end="",
                    flush=True,
                )

                last_progress = progress

        try:
            dead_spots = analyzer.analyze(video, progress_callback=progress_callback)
            print()  # New line after progress bar completes

            video_results = {"path": str(video), "name": video.name, "stops": []}

            if dead_spots:
                print(f"  Found {len(dead_spots)} stop(s):")
                for j, spot in enumerate(dead_spots, 1):
                    start = str(timedelta(seconds=int(spot.start_time)))
                    end = str(timedelta(seconds=int(spot.end_time)))
                    duration = spot.end_time - spot.start_time

                    print(f"    {j}. {start} - {end} ({duration:.1f}s)")

                    video_results["stops"].append(
                        {
                            "start_time": spot.start_time,
                            "end_time": spot.end_time,
                            "start_formatted": start,
                            "end_formatted": end,
                            "duration": round(duration, 2),
                            "motion_energy": round(spot.motion_energy, 6),
                        }
                    )

                total_stops += len(dead_spots)
            else:
                print("  No stops found")

            new_results[video.name] = video_results

        except Exception as e:
            print(f"  ERROR: {e}")
            new_results[video.name] = {"error": str(e)}

    # Merge new results with checkpoint
    all_results = merge_results(input_dir, new_results)

    # Summary
    print("\n" + "=" * 60)
    videos_data = all_results.get("videos", {})
    completed_count = sum(
        1 for r in videos_data.values() if isinstance(r, dict) and "stops" in r and "error" not in r
    )
    total_stops = sum(
        len(r.get("stops", []))
        for r in videos_data.values()
        if isinstance(r, dict) and "stops" in r
    )
    print(f"SUMMARY: {total_stops} stops across {completed_count}/{total_count} videos")

    # Save JSON
    output_path = args.output or (input_dir / "stops_report.json")
    output_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to: {output_path}")

    return 0
