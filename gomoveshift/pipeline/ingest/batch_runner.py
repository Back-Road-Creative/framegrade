"""Batch pipeline runner — execute the full pipeline on each episode in an ingest plan.

Assigns sequential publish dates and runs PipelineOrchestrator on each
concatenated episode. Handles disk cleanup of concat intermediates.
"""

import logging
from datetime import datetime
from datetime import time as dt_time
from pathlib import Path
from typing import Optional

from .gopro_parser import IngestPlan

logger = logging.getLogger(__name__)


def run_batch_pipeline(
    plan: IngestPlan,
    publish_day: str = "mon",
    shorts_time: dt_time = dt_time(12, 0, 0),
    output_base: Optional[Path] = None,
    start_date: Optional[str] = None,
    skip_color: bool = False,
    skip_stops: bool = False,
    draft: bool = False,
    metadata_generator=None,
) -> list[dict]:
    """Run the full pipeline on each episode in the ingest plan.

    Assigns sequential publish dates (e.g., every Monday starting from next Monday).

    Args:
        plan: Executed ingest plan (episodes must have concat_path set)
        publish_day: Day of week for long-form publish
        shorts_time: Time of day for shorts
        output_base: Base output directory (default: ``gms_output_dir()``,
            i.e. ``$GMS_OUTPUT_DIR`` or ``<data home>/output``)
        start_date: First publish date (YYYY-MM-DD), defaults to next occurrence of publish_day
        skip_color: Skip color correction
        skip_stops: Skip stop detection
        draft: Mark outputs as draft
        metadata_generator: Publish-side ``LongFormMetadataGenerator`` class the pipeline's
            metadata stages use (the engine does not import it)

    Returns:
        List of result dicts per episode
    """
    from ...config import get_config
    from ...paths import gms_output_dir
    from ..orchestrator import PipelineOrchestrator

    if output_base is None:
        output_base = gms_output_dir()

    config = get_config()
    orchestrator = PipelineOrchestrator(config, metadata_generator=metadata_generator)

    # Publish dates: the first is --start-date or the next free slot on publish_day (ledger
    # occupancy, decided by seo_manager); each later episode takes the next free slot after it.
    from ..seo_slots import default_publish_slot

    next_pub = (
        datetime.strptime(start_date, "%Y-%m-%d")
        if start_date
        else default_publish_slot(config, publish_day)
    )

    results = []

    for episode in plan.episodes:
        if not episode.concat_path or not episode.concat_path.exists():
            logger.error(f"Episode {episode.index}: no concatenated file — skipping")
            results.append({"episode": episode.index, "success": False, "error": "no concat file"})
            continue

        pub_date = next_pub.strftime("%Y-%m-%d")
        logger.info(f"\n{'=' * 60}")
        logger.info(f"  Episode {episode.index}: {episode.total_duration_min:.0f}min")
        logger.info(f"  Publish: {pub_date} ({publish_day})")
        logger.info(f"  Location: {episode.location}")
        logger.info(f"{'=' * 60}")

        try:
            from ..orchestrator import RunOptions

            options = RunOptions.from_legacy(
                video_path=episode.concat_path,
                location=episode.location or "",
                publish_day=publish_day,
                output_base=output_base,
                gps_path=episode.gpx_path,
                gps_date=episode.date,
                skip_color=skip_color,
                skip_stops=skip_stops,
                draft=draft,
            )
            result = orchestrator.run(options)

            results.append(
                {
                    "episode": episode.index,
                    "success": result.success,
                    "output_dir": str(result.output_dir),
                    "final_video": str(result.final_video) if result.final_video else None,
                    "publish_date": pub_date,
                }
            )

            if result.success:
                logger.info(f"  \u2713 Episode {episode.index} complete \u2192 {result.output_dir}")
            else:
                logger.error(
                    f"  \u2717 Episode {episode.index} failed: "
                    f"{result.errors[0] if result.errors else 'unknown'}"
                )

        except Exception as e:
            logger.error(f"  \u2717 Episode {episode.index} exception: {e}")
            results.append(
                {
                    "episode": episode.index,
                    "success": False,
                    "error": str(e),
                }
            )

        # Clean up concat intermediate to reclaim disk space
        # Skip symlinks (single-chapter episodes point to original source files)
        if (
            episode.concat_path
            and episode.concat_path.exists()
            and not episode.concat_path.is_symlink()
        ):
            concat_size_mb = episode.concat_path.stat().st_size / (1024 * 1024)
            episode.concat_path.unlink()
            logger.info(f"  Cleaned up concat file ({concat_size_mb:.0f}MB reclaimed)")

        # Advance to next publish date
        next_pub = default_publish_slot(config, publish_day, after=next_pub)

    # Summary
    succeeded = sum(1 for r in results if r.get("success"))
    logger.info(f"\nBatch complete: {succeeded}/{len(results)} episodes succeeded")

    return results
