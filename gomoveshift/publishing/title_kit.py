"""Build the :class:`~gomoveshift.pipeline.title_kit.TitleKit` the media engine runs on.

GMS-side composition root: every callable is a late-bound wrapper over the
publish-side module, so a test patching the source module (``longform_metadata``,
``title_builder``) still reaches the call.
"""

from __future__ import annotations

from ..pipeline.title_kit import TitleKit


def _media_profile():
    from ..console.settings import current_brand
    from .brand_policy import load_media_profile

    return load_media_profile(current_brand())


def _per_location_cap():
    from ..console.settings import current_brand
    from .brand_policy import load_brand_policy

    return load_brand_policy(current_brand()).per_location_cap


def _superlong_title(location, part_index, part_count):
    from ..pipeline.title_builder import build_superlong_part_title

    return build_superlong_part_title(location, part_index, part_count)


def _chapter_generator(config):
    from ..pipeline.chapters.generator import ChapterGenerator

    return ChapterGenerator(config)


def _superlong_max_chapters(duration_sec):
    from ..pipeline.chapters.generator import superlong_max_chapters

    return superlong_max_chapters(duration_sec)


def _day_master_journey(day_gps_points):
    from ..pipeline.chapters.generator import build_day_master_journey

    return build_day_master_journey(day_gps_points)


def _town_events(chapters):
    from ..pipeline.chapters.marker_merger import chapter_town_events

    return chapter_town_events(chapters)


def _lf_title(*args, **kwargs):
    from ..pipeline.title_builder import build_lf_title

    return build_lf_title(*args, **kwargs)


def _unique_titles(*args, **kwargs):
    from ..pipeline.title_builder import ensure_unique_titles

    return ensure_unique_titles(*args, **kwargs)


def _published_titles():
    from ..pipeline.longform_metadata import load_published_titles

    return load_published_titles()


def _refresh_title_ledger():
    from ..pipeline.longform_metadata import refresh_channel_title_ledger

    return refresh_channel_title_ledger()


def build_title_kit() -> TitleKit:
    """The kit for a GMS CLI run (``cmd_drive`` via ``DrivePipelineOrchestrator``)."""
    from ..pipeline.chapters.generator import YOUTUBE_MIN_CHAPTERS

    return TitleKit(
        media_profile=_media_profile,
        per_location_cap=_per_location_cap,
        superlong_title=_superlong_title,
        chapter_generator=_chapter_generator,
        superlong_max_chapters=_superlong_max_chapters,
        min_chapters=YOUTUBE_MIN_CHAPTERS,
        day_master_journey=_day_master_journey,
        town_events=_town_events,
        lf_title=_lf_title,
        unique_titles=_unique_titles,
        published_titles=_published_titles,
        refresh_title_ledger=_refresh_title_ledger,
    )
