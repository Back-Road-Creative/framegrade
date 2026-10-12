"""Unified video type classification — single source of truth.

Replaces incompatible is_short definitions, the 61-479s dead zone,
conflicting duration ranges and naming scheme mismatches.

All video classification across resolve, pipeline, and content-pipeline
should delegate to classify_video() from this module.
"""

import enum
import logging

logger = logging.getLogger(__name__)

# Duration constants — single source of truth
SHORTS_MAX_DURATION: float = 60  # seconds — shorts must be <= this
LF_MIN_DURATION: float = 480  # seconds — regular LF must be >= this (8 min)
LF_MAX_DURATION: float = 1800  # seconds — regular LF must be <= this (30 min)


class VideoType(enum.Enum):
    """Canonical video type classification.

    Shorts: vertical (height > width) AND duration <= 60s.
    Regular LF: duration 480-1800s (any orientation).
    Unclassified: everything else (dead zone 61-479s, landscape shorts, over-max).
    """

    SHORT = "short"
    REGULAR_LF = "regular_lf"
    UNCLASSIFIED = "unclassified"

    def to_youtube_type(self) -> str:
        """Map to YouTube API video type: 'SHORT' or 'VIDEO'."""
        if self == VideoType.SHORT:
            return "SHORT"
        return "VIDEO"

    def to_ledger_type(self) -> str:
        """Map to content ledger type: 'shorts' or 'longform'."""
        if self == VideoType.SHORT:
            return "shorts"
        return "longform"

    def to_event_type(self) -> str:
        """Map to content events storage type: 'short' or 'long'."""
        if self == VideoType.SHORT:
            return "short"
        return "long"


def classify_video(duration_sec: float, width: int, height: int) -> VideoType:
    """Classify a video by duration and dimensions.

    Args:
        duration_sec: Video duration in seconds.
        width: Video width in pixels.
        height: Video height in pixels.

    Returns:
        VideoType classification.
    """
    is_vertical = height > width

    # Shorts: vertical AND within shorts duration
    if is_vertical and duration_sec <= SHORTS_MAX_DURATION:
        return VideoType.SHORT

    # Regular long-form: within LF duration range (any orientation)
    if LF_MIN_DURATION <= duration_sec <= LF_MAX_DURATION:
        return VideoType.REGULAR_LF

    # Everything else is unclassified
    if is_vertical and duration_sec > SHORTS_MAX_DURATION:
        logger.warning(
            "Unclassified video: vertical %dx%d, %.1fs — in dead zone (>%ds, <%ds)",
            width,
            height,
            duration_sec,
            SHORTS_MAX_DURATION,
            LF_MIN_DURATION,
        )
    elif duration_sec > LF_MAX_DURATION:
        logger.warning(
            "Unclassified video: %dx%d, %.1fs — exceeds max LF duration %ds",
            width,
            height,
            duration_sec,
            LF_MAX_DURATION,
        )
    else:
        logger.warning(
            "Unclassified video: %dx%d, %.1fs — does not match any category",
            width,
            height,
            duration_sec,
        )

    return VideoType.UNCLASSIFIED


def suggest_action(duration_sec: float, width: int, height: int) -> str:
    """Suggest action for dead-zone (UNCLASSIFIED) videos.

    Called when classify_video() returns UNCLASSIFIED. Provides a
    recommendation so dead-zone content is not silently dropped.

    Args:
        duration_sec: Video duration in seconds.
        width: Video width in pixels.
        height: Video height in pixels.

    Returns:
        Action string: "trim_to_short", "pad_to_lf", "split",
        "extend_to_lf", or "skip".
    """
    is_vertical = height > width

    if is_vertical:
        if duration_sec <= 120:
            return "trim_to_short"
        elif duration_sec >= 420:
            return "pad_to_lf"
        else:
            return "split"

    # Horizontal dead zone: can't be a short, but may be close to LF minimum
    if duration_sec >= 420:
        return "extend_to_lf"

    return "skip"
