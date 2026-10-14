"""Publish-side hooks for ``PhotoOrchestrator``.

The engine takes the ledger booking and the ThemeRotator voice as callables; the CLI
wires them with ``PhotoOrchestrator(..., **photo_run_hooks())``.
"""

from __future__ import annotations

import logging

from ..pipeline.content_ledger import ContentEntry, record_confirmed_upload
from ..pipeline.photos.voice_styles import VoiceStyle

logger = logging.getLogger(__name__)


def book_photo_slots(rows: list[dict]) -> bool:
    """Book confirmed photo slots into the shared content ledger.

    ``rows`` are ``{"date", "time", "title", "source_file"}``. Returns ``True``
    when the booking persisted; ``record_confirmed_upload`` is best-effort and
    returns ``False`` (it never raises) on a miss.
    """
    entries = [
        ContentEntry(
            date=row["date"],
            time=row["time"],
            content_type="photo",
            title=row["title"],
            platform="all",
            status="scheduled",
            source="photo_pipeline",
            source_file=row["source_file"],
        )
        for row in rows
    ]
    return record_confirmed_upload(entries)


def next_photo_voice_style() -> VoiceStyle:
    """Next voice style from the SEO manager's ``ThemeRotator``.

    A missing ``seo_manager`` or a RUNTIME rotator failure (corrupt or locked
    rotation state, a concurrent writer) must never crash caption generation —
    the rotator is an optional A/B-tracking nicety — so both fall back to
    ``FIRST_PERSON`` and surface a warning.
    """
    try:
        from seo_manager.analytics.theme_rotator import ContentTheme, ThemeRotator

        rotator = ThemeRotator()
        theme = rotator.get_next_theme(category="photo")

        # Map ContentTheme back to VoiceStyle
        mapping = {
            ContentTheme.VOICE_FIRST_PERSON: VoiceStyle.FIRST_PERSON,
            ContentTheme.VOICE_MINIMAL_ZEN: VoiceStyle.MINIMAL_ZEN,
            ContentTheme.VOICE_ENGAGING: VoiceStyle.ENGAGING_CONVERSATIONAL,
        }
        # VOICE_INTERACTIVE may not exist in older seo_manager versions
        if hasattr(ContentTheme, "VOICE_INTERACTIVE"):
            mapping[ContentTheme.VOICE_INTERACTIVE] = VoiceStyle.INTERACTIVE
        return mapping.get(theme, VoiceStyle.FIRST_PERSON)

    except ImportError:
        logger.warning("seo_manager not available, defaulting to FIRST_PERSON")
        return VoiceStyle.FIRST_PERSON
    except Exception as e:
        logger.warning(f"ThemeRotator failed ({type(e).__name__}: {e}); defaulting to FIRST_PERSON")
        return VoiceStyle.FIRST_PERSON


def record_photo_theme_usage(voice_style: str, week_start: str) -> None:
    """Record a batch's voice style in ``ThemeRotator`` for A/B tracking.

    Non-fatal but visible: a failure warns and never raises into the publish flow.
    """
    try:
        from seo_manager.analytics.theme_rotator import ContentTheme, ThemeRotator

        rotator = ThemeRotator()
        voice_map = {
            "voice_first_person": ContentTheme.VOICE_FIRST_PERSON,
            "voice_minimal_zen": ContentTheme.VOICE_MINIMAL_ZEN,
            "voice_engaging": ContentTheme.VOICE_ENGAGING,
        }
        if hasattr(ContentTheme, "VOICE_INTERACTIVE"):
            voice_map["voice_interactive"] = ContentTheme.VOICE_INTERACTIVE
        theme = voice_map.get(voice_style)
        if theme:
            rotator.record_usage(
                content_id=f"photo_batch_{week_start}",
                video_id=week_start,
                theme=theme,
                category="photo",
            )
    except Exception as e:
        logger.warning(f"Theme usage recording failed ({type(e).__name__}): {e}")


def photo_upscale_policy() -> str:
    """The brand's still-upscale policy, or ``"off"`` (today's behaviour) if unreadable."""
    from ..console.settings import current_brand
    from .brand_policy import load_media_profile

    try:
        return load_media_profile(current_brand()).upscale
    except Exception:  # noqa: BLE001 — fail closed: no ML upscale
        logger.warning("photo run: media profile unreadable -- ML upscale off")
        return "off"


def photo_run_hooks() -> dict[str, object]:
    """Keyword arguments that wire a ``PhotoOrchestrator`` to the ledger, rotator and upscale policy."""
    return {
        "upscale": photo_upscale_policy(),
        "ledger_booker": book_photo_slots,
        "voice_source": next_photo_voice_style,
        "theme_recorder": record_photo_theme_usage,
    }
