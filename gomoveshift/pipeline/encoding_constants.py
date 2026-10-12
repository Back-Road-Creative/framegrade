"""
Encoding constants — single source of truth for all video quality parameters.

These values are FROZEN. Do not make them configurable — config drift is the
root cause of silent quality degradation across pipeline runs. If you need
different quality for testing, use --draft (which stamps _DRAFT in the filename).

Long-form: YouTube-optimized 4K POV driving content. YouTube re-encodes all
uploads to VP9/AV1 at ~15-20 Mbps, so we target the upload sweet spot where
additional source bitrate yields diminishing returns after re-encode.

Shorts: cut from the graded 10-bit day master; quality matches the long-form
encode (p7 / CQ 18 / 10-bit, the same NVENC quality flags, and a bitrate
ceiling derived from the same floor x ratio model) so graded skies don't band
on re-encode.
"""

import math
from dataclasses import dataclass
from typing import Optional

# ── Long-form encoding (4K POV driving) ──────────────────────────────────

LONGFORM_CRF = 18  # x265 fallback quality (YouTube transparent)
LONGFORM_PRESET = "slow"
LONGFORM_NVENC_PRESET = "p7"
LONGFORM_NVENC_CQ = 18  # YouTube-optimized: ~40-50 Mbps at 4K driving content
LONGFORM_PROFILE = "main10"
LONGFORM_PIX_FMT = "yuv420p10le"
LONGFORM_COLOR_RANGE = "tv"  # Limited range (16-235) — YouTube SDR standard
LONGFORM_COLORSPACE = "bt709"
LONGFORM_COLOR_PRIMARIES = "bt709"
LONGFORM_COLOR_TRC = "bt709"

# ── Near-silent audio policy ───────────
# Policy: every published video must be practically SILENT —
# at maximum playback volume, conversation and songs must not be interpretable,
# regardless of platform loudness norms. We therefore ATTENUATE ONLY: the audio
# filter never normalizes *up* toward a loudness target (two-pass loudnorm
# boosting quiet cab audio up to -20 LUFS makes speech audible).
# Long-form keeps faint ambience attenuated to a near-silent integrated ceiling;
# shorts ship with NO audio stream (`-an`) because vertical platforms
# (TikTok/IG) may re-normalize quiet audio back UP on their side — a stripped
# stream cannot be undone. See build_attenuate_filter() and the ceiling-only
# validation gates.
AUDIO_SILENCE_CEILING_LUFS = -45.0  # integrated LUFS ceiling — attenuate-only, never boost above
AUDIO_TRUE_PEAK_CAP = -10.0  # dBTP hard cap after attenuation
# Speech-intelligibility belt: the integrated-LUFS ceiling can
# still pass a file whose brief speech spikes stay interpretable, because a few
# loud words barely move the whole-file integrated average. The attenuate-only
# volume gain pulls every momentary (400 ms) window DOWN by the same amount as
# the integrated loudness, so a correctly near-silent LF sits far below this
# momentary ceiling; an un-attenuated (or under-attenuated) LF whose loud speech
# words spike is rejected upstream of upload. Measured from ffmpeg ebur128's
# per-frame momentary (M) field. Sits comfortably above the -45 integrated
# ceiling so an honestly-attenuated file never trips it on ambience alone.
AUDIO_MAX_MOMENTARY_LUFS = -30.0

# Bump whenever the audio-filter RULE changes shape (the loudnorm=-20
# boost → attenuate-only volume gain was such a change). Render fingerprints
# fold this in alongside the ceiling/cap constants so a resumed run whose audio
# policy changed re-derives the LF instead of reusing a pre-fix (audible) render.
AUDIO_FILTER_ALGO_VERSION = 1

LONGFORM_AUDIO_BITRATE = "384k"
LONGFORM_AUDIO_LUFS = AUDIO_SILENCE_CEILING_LUFS  # LF attenuate-only target (was -20 boost target)
LONGFORM_AUDIO_TP = AUDIO_TRUE_PEAK_CAP  # true-peak cap (was -1.5 YouTube spec)
LONGFORM_AUDIO_LRA = 11.0  # retained only for the loudnorm MEASUREMENT command shape
LONGFORM_AUDIO_MAX_LUFS = -43.0  # Validation gate: reject if LOUDER than ceiling (2 dB slack)
LONGFORM_AUDIO_MAX_MOMENTARY = (
    AUDIO_MAX_MOMENTARY_LUFS  # speech-intelligibility belt (max ebur128 momentary)
)

# ── Long-form bitrate floor (absolute, resolution-based) ──────────────
# Fixed floors guarantee YouTube receives enough data regardless of what
# the GoPro source bitrate happened to be. CQ 18 adapts quality to scene
# complexity — the floor just prevents the encoder from going too low on
# simple content (long straight highway). YouTube recommends 35-53 Mbps
# for 4K HEVC uploads; we sit above the minimum, not on it.

LONGFORM_MIN_BITRATE_4K = 40_000  # 40 Mbps — comfortably above YouTube's 35 Mbps rec
LONGFORM_MIN_BITRATE_1080P = 15_000  # 15 Mbps — above YouTube's 12 Mbps rec
# maxrate = 200% of floor. A 1.3x ratio pegged the encoder at the ceiling on
# every frame, so CQ 18 wanted more bits than VBR could spend. Lifting to 2.0x
# (~80 Mbps cap) lets detail-heavy scenes (foliage, shadow texture) consume
# the bits CQ 18 is asking for; easy scenes (highway) still ride near the floor.
LONGFORM_MAXRATE_RATIO = 2.0
LONGFORM_BUFSIZE_RATIO = 3.0  # bufsize = 3x floor (allow the burst the new maxrate enables)

# ── Shorts encoding (vertical 9:16 social media) ────────────────────────

# Shorts are NVENC-only and cut from the graded 10-bit day master.
# Encode quality matches the long-form master (p7 / CQ 18 / 10-bit main10):
# a lower p5 / CQ 23 / 8-bit yuv420p setting visibly banded graded skies, and
# platform re-encoding is not a reason to degrade the upload: better input
# yields a better re-encode.
SHORTS_CRF = 18  # NVENC CQ — matches LONGFORM_NVENC_CQ (lower = better)
SHORTS_PRESET = "p7"  # NVENC preset — matches LONGFORM_NVENC_PRESET
SHORTS_PROFILE = "main10"
SHORTS_PIX_FMT = "p010le"  # NVENC 10-bit input surface (same as long-form NVENC path)
SHORTS_WIDTH = 1080
SHORTS_HEIGHT = 1920
SHORTS_AUDIO_BITRATE = "128k"
SHORTS_AUDIO_LUFS = AUDIO_SILENCE_CEILING_LUFS  # shorts ship -an; retained for gate/constant parity
# Rate control mirrors the long-form floor x ratio model rather than a flat
# cap. A short is 1080x1920 — 2.07 MP, the same pixel count as a 1920x1080
# long-form — so it takes the long-form 1080p floor. The previous flat
# 8M / 12M pair predates the CQ18 + 10-bit raise and sat at roughly HALF that
# floor, which is the VBV-starves-CQ failure long-form already fixed by
# lifting its maxrate ratio: the encoder pegs the ceiling on every frame and
# CQ 18 never gets the bits it asks for on detail-heavy scenes.
SHORTS_MIN_BITRATE = LONGFORM_MIN_BITRATE_1080P  # 15 Mbps — same 2.07MP frame as a 1080p LF
SHORTS_MAXRATE_RATIO = LONGFORM_MAXRATE_RATIO
SHORTS_BUFSIZE_RATIO = LONGFORM_BUFSIZE_RATIO
SHORTS_MAXRATE = f"{int(SHORTS_MIN_BITRATE * SHORTS_MAXRATE_RATIO)}k"  # 30 Mbps
SHORTS_BUFSIZE = f"{int(SHORTS_MIN_BITRATE * SHORTS_BUFSIZE_RATIO)}k"  # 45 Mbps

# ── Rendition matrix ─────────────────────────────────────────────────
# One row per delivery platform, derived from the constants above -- never
# from an invented platform ceiling. A field is `None` when no such ceiling
# or fixed value exists yet anywhere in this file; `encode_rendition()`
# (route_story/encode.py) only enforces a field that is actually set.


@dataclass(frozen=True)
class Rendition:
    width: Optional[int]
    height: Optional[int]
    max_seconds: Optional[float]
    max_bytes: Optional[int]
    fps: Optional[int]
    codec: str
    faststart: bool
    audio: bool
    min_bitrate_kbps: Optional[int] = None

    @property
    def aspect(self) -> Optional[tuple[int, int]]:
        """Width:height reduced to lowest terms, or ``None`` when geometry is unset."""
        if self.width is None or self.height is None:
            return None
        g = math.gcd(self.width, self.height)
        return (self.width // g, self.height // g)


# ── Publish-payload text limits ──────────────────────────────────────
# One row per platform, checked by ``publishing.models.validate_payloads``
# before a delivery is spooled (``cli/handlers/enqueue_plan.py``). Values are
# the platform API's own hard limit, not house style -- a field is `None`
# when this file has no documented limit for it. Sourced from
# docs/social-publishing-payload-mapping.md and the existing YouTube upload
# gate (drive_phases/youtube_uploader.py) where a repo source exists;
# otherwise cited inline and marked unverified.


@dataclass(frozen=True)
class PayloadLimits:
    title_max_len: Optional[int] = None
    description_max_len: Optional[int] = None
    tags_total_max_len: Optional[int] = None
    caption_max_len: Optional[int] = None


PAYLOAD_LIMITS: dict[str, PayloadLimits] = {
    # docs/pipeline/69-ost-title-thumbnail-single-source-of-truth.md (100-char
    # clamp) and docs/social-publishing-payload-mapping.md (5000-char
    # snippet.description); tags_total_max_len matches the existing upload-time
    # gate at drive_phases/youtube_uploader.py MAX_TAGS_TOTAL_LEN.
    "youtube": PayloadLimits(title_max_len=100, description_max_len=5000, tags_total_max_len=500),
    # docs/social-publishing-payload-mapping.md -- "Instagram allows up to
    # 2200 chars".
    "instagram": PayloadLimits(caption_max_len=2200),
    # docs/social-publishing-payload-mapping.md documents Facebook Reels
    # description as having "no practical limit near this length" rather than
    # a numeric cap; 2200 here mirrors Instagram/TikTok as a conservative
    # stand-in. # external, unverified -- see Meta Graph API /video_reels
    # description field docs.
    "facebook": PayloadLimits(description_max_len=2200),
    # docs/social-publishing-payload-mapping.md -- TikTok's
    # `post_info.title` (the field `TikTokPayload.caption` maps to per
    # models.py) "allows 2200 chars".
    "tiktok": PayloadLimits(caption_max_len=2200),
}


RENDITIONS: dict[str, Rendition] = {
    # Long-form: resolution follows the source (LONGFORM_MIN_BITRATE_4K and
    # _1080P both exist for a reason -- there is no single fixed w/h here).
    # codec="hevc": hevc_nvenc primary / libx265 CPU fallback
    # (``_nvenc_args``, ``_x265_args`` and ``encode_video`` in
    # color_filters.py). No max_seconds/max_bytes/fps ceiling constant exists
    # in this file, so those stay None.
    "youtube_longform": Rendition(
        width=None,
        height=None,
        max_seconds=None,
        max_bytes=None,
        fps=None,
        codec="hevc",
        faststart=True,
        audio=True,
        min_bitrate_kbps=LONGFORM_MIN_BITRATE_4K,
    ),
    # Two long-form rows: ``youtube_longform`` follows the source (no geometry,
    # so ``encode_rendition`` adds no scale filter); the drive long-form
    # product is contractually this exact 4K frame, which the LF gate checks.
    "drive_longform_4k": Rendition(
        width=3840,
        height=2160,
        max_seconds=None,
        max_bytes=None,
        fps=None,
        codec="hevc",
        faststart=True,
        audio=True,
        min_bitrate_kbps=LONGFORM_MIN_BITRATE_4K,
    ),
    # Shorts: exactly 1080x1920, hevc_nvenc only (shorts_extractor.py
    # comment at the hevc_nvenc call site), no audio stream. `ShortsExtractor
    # .max_duration=45.0` (shorts_extractor.py) is a *segment-selection*
    # parameter -- which clip gets picked -- not an encode-time output
    # ceiling constant in this file, so max_seconds stays None rather than
    # conflating the two.
    "youtube_short": Rendition(
        width=SHORTS_WIDTH,
        height=SHORTS_HEIGHT,
        max_seconds=None,
        max_bytes=None,
        fps=None,
        codec="hevc",
        faststart=True,
        audio=False,
        min_bitrate_kbps=SHORTS_MIN_BITRATE,
    ),
    # Route Story social cut: the SAME 1080x1920 frame as a drive short,
    # but a DIFFERENT audio contract. This 9:16 cut keeps its
    # attenuate-only audio -- it is not cabin
    # audio, so the "shorts ship -an" rule above does not apply to it.
    # max_seconds mirrors route_story.presets.SOCIAL.max_seconds (90.0) so
    # the two ceilings cannot silently disagree; verified equal in
    # test_encoding_renditions.py rather than imported, for the same
    # import-cycle reason presets.py mirrors film.py's literals.
    "route_story_short": Rendition(
        width=SHORTS_WIDTH,
        height=SHORTS_HEIGHT,
        max_seconds=90.0,
        max_bytes=None,
        fps=None,
        codec="hevc",
        faststart=True,
        audio=True,
        min_bitrate_kbps=SHORTS_MIN_BITRATE,
    ),
}
