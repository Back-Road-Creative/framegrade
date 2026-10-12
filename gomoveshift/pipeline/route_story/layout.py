"""Aspect-aware HUD anchor geometry.

``hud_layout(width, height)`` is the ONLY place that knows whether a frame is
landscape or portrait. Every draw function in ``hud.py`` reads its anchor
positions from the ``HudLayout`` it returns instead of branching on aspect
ratio itself -- a new aspect ratio only ever needs a new case here, never a
scattered ``if portrait`` in draw code.
"""

from __future__ import annotations

from dataclasses import dataclass

# Portrait (9:16 social) safe zone reserved for the platform's own UI chrome
# (caption text, share/like rail, progress bar) at the bottom of the frame.
# Nothing HUD-drawn may render below ``height - SHORT_BOTTOM_SAFE_PX``.
# Adjustable: raise it if a platform's chrome grows.
SHORT_BOTTOM_SAFE_PX = 220

# Vertical gap between the stacked portrait badges/watermark.
_PORTRAIT_STACK_GAP_PX = 10

# Fraction of frame width kept clear on both sides of centred text -- mirrors
# ``hud._TITLE_SAFE_WIDTH_FRACTION`` so both surfaces read consistently.
_TITLE_SAFE_WIDTH_FRACTION = 0.88
_WATERMARK_MARGIN_X_FRACTION = (1 - _TITLE_SAFE_WIDTH_FRACTION) / 2
_WATERMARK_MARGIN_Y_FRACTION = 0.02


@dataclass(frozen=True)
class HudLayout:
    """Pixel anchors for one frame's HUD overlays, resolved once from ``image.size``."""

    width: int
    height: int
    title_y: float
    title_safe_width: float
    dashboard_panel_height: int
    dashboard_panel_top: int
    dashboard_left: int
    dashboard_right: int
    badge_summit_top: float
    badge_arrival_top: float
    watermark_x: float
    watermark_y: float
    watermark_anchor: str
    photo_label_x: int
    photo_label_y: int
    short_bottom_safe_px: int
    profile_left: int
    profile_right: int
    profile_top: float
    profile_height: int


def _profile_height(height: int) -> int:
    """The elevation profile's on-screen height, ~7% of the frame height.

    Floored so a very short/low-res frame still gets a chart tall enough to
    read the cursor and fill against, rather than a sliver a few pixels tall.
    """
    return max(48, int(height * 0.07))


def _badge_slot_height(width: int) -> float:
    """A badge's on-screen height, mirroring ``hud._badge``'s own font-size
    formula -- used only to size the portrait stack so every slot clears the
    bottom safe zone by construction, never to draw text itself."""
    return max(15, width // 70) + 18


def hud_layout(width: int, height: int) -> HudLayout:
    """Resolve every HUD anchor for a frame of this size.

    Landscape (``width >= height``) reproduces the historical literal
    positions exactly. Portrait moves the badges and watermark to
    bottom-centre, stacked clear of ``SHORT_BOTTOM_SAFE_PX``, and scales the
    dashboard/title from ``width`` (the short side in portrait) rather than
    ``height`` (the short side in landscape).
    """
    if width >= height:
        dashboard_panel_height = max(72, int(height * 0.135))
        profile_height = _profile_height(height)
        return HudLayout(
            width=width,
            height=height,
            title_y=height * 0.13,
            title_safe_width=width * _TITLE_SAFE_WIDTH_FRACTION,
            dashboard_panel_height=dashboard_panel_height,
            dashboard_panel_top=height - dashboard_panel_height,
            dashboard_left=int(width * 0.045),
            dashboard_right=int(width * 0.955),
            badge_summit_top=height * 0.80,
            badge_arrival_top=height * 0.12,
            watermark_x=width - width * _WATERMARK_MARGIN_X_FRACTION,
            watermark_y=height - height * _WATERMARK_MARGIN_Y_FRACTION,
            watermark_anchor="rs",
            photo_label_x=int(width * 0.055),
            photo_label_y=int(height * 0.84),
            short_bottom_safe_px=0,
            profile_left=int(width * 0.045),
            profile_right=int(width * 0.955),
            profile_top=height - dashboard_panel_height - profile_height,
            profile_height=profile_height,
        )

    panel_height = max(72, int(width * 0.135))
    slot = _badge_slot_height(width)
    floor = height - SHORT_BOTTOM_SAFE_PX - panel_height
    summit_top = floor - slot
    arrival_top = summit_top - _PORTRAIT_STACK_GAP_PX - slot
    watermark_y = arrival_top - _PORTRAIT_STACK_GAP_PX
    profile_height = _profile_height(height)
    return HudLayout(
        width=width,
        height=height,
        title_y=height * 0.08,
        title_safe_width=width * _TITLE_SAFE_WIDTH_FRACTION,
        dashboard_panel_height=panel_height,
        dashboard_panel_top=floor,
        dashboard_left=int(width * 0.045),
        dashboard_right=int(width * 0.955),
        badge_summit_top=summit_top,
        badge_arrival_top=arrival_top,
        watermark_x=width / 2,
        watermark_y=watermark_y,
        watermark_anchor="ms",
        photo_label_x=int(width * 0.055),
        photo_label_y=int(arrival_top - slot * 1.5),
        short_bottom_safe_px=SHORT_BOTTOM_SAFE_PX,
        profile_left=int(width * 0.045),
        profile_right=int(width * 0.955),
        profile_top=floor - profile_height,
        profile_height=profile_height,
    )
