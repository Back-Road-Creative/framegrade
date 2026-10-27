"""Configuration management for GoMoveShift."""

import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Literal, NamedTuple, Optional

import yaml
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from gomoveshift.paths import (
    gms_config_home,
    gms_data_home,
    gms_geocoding_cache_dir,
    gms_inputs_dir,
)
from gomoveshift.video_types import LF_MIN_DURATION, SHORTS_MAX_DURATION

logger = logging.getLogger(__name__)


def _deep_merge_dict(base: dict, override: dict) -> dict:
    """Return a new dict with ``override`` recursively merged into ``base``.

    Nested mappings merge key-by-key so a partial user file (e.g.
    only ``drive.long_form_publish_days``) doesn't erase sibling keys
    in ``base`` (e.g. ``drive.gps_quality``). Non-dict values replace
    wholesale — lists included, because list-merging has no obvious
    semantics (append? replace? dedup?) and explicit replacement is
    the safer contract for an override file.
    """
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge_dict(result[key], value)
        else:
            result[key] = value
    return result


#: ``drive:`` keys that once hand-set the shorts volume or hours. Waypost decides them now
#: (``pipeline/volume_source.py``, the cadence's ``shorts_times``); a leftover key is dropped by pydantic, so the
#: loader names it instead of letting an owner believe it still has an effect.
RETIRED_CONFIG_KEYS = (
    "shorts_per_lf_count",
    "lf_length_min_n",
    "lf_length_bucket_edges_sec",
    "shorts_publish_times_by_region",
    "shorts_slot_pool_by_region",
)


#: ``drive:`` keys whose FIELD survives as the per-run ``--lf-day`` / ``--lf-time``
#: transport, so pydantic would accept a file value; the loader drops it instead.
WAYPOST_OWNED_DRIVE_KEYS = ("long_form_publish_days", "long_form_publish_time")


def _warn_on_retired_keys(raw: dict) -> None:
    """Warn once per leftover retired key under ``drive:`` (never raises), and remove the
    Waypost-owned transport keys so a file value cannot set the cadence."""
    drive = raw.get("drive")
    if not isinstance(drive, dict):
        return
    for key in RETIRED_CONFIG_KEYS:
        if key in drive:
            logger.warning("RETIRED_CONFIG_KEY key=%s (retired: Waypost decides)", key)
    for key in WAYPOST_OWNED_DRIVE_KEYS:
        if key in drive:
            logger.warning("RETIRED_CONFIG_KEY key=%s (retired: Waypost decides)", key)
            del drive[key]


def _warn_on_stop_knob_drift(raw: dict) -> None:
    """Warn when a loaded config diverges the two motion-stop knobs.

    ``stop_detection.{motion_threshold,min_stop_duration}`` is **authoritative**
    for the pipeline's actual stop removal — ``StopProcessor`` reads it and
    bridges its values onto a throwaway config's ``dead_spots`` for the internal
    ``MotionAnalyzer``. ``dead_spots.{motion_threshold,min_duration_seconds}``
    is consumed directly only by the standalone ``analyze`` preview. The two are
    meant to stay in lockstep; if they drift, the preview misrepresents what the
    pipeline will cut. Only fires for an operator file that sets BOTH blocks
    (bare ``Config()`` defaults legitimately differ), so it never noises up
    programmatic construction. See ``docs/pipeline/53-stop-config-two-blocks-one-authoritative.md``.
    """
    ds = raw.get("dead_spots")
    sd = raw.get("stop_detection")
    if not (isinstance(ds, dict) and isinstance(sd, dict)):
        return
    drift = []
    if (
        "motion_threshold" in ds
        and "motion_threshold" in sd
        and ds["motion_threshold"] != sd["motion_threshold"]
    ):
        drift.append(
            f"motion_threshold (dead_spots={ds['motion_threshold']} vs stop_detection={sd['motion_threshold']})"
        )
    if (
        "min_duration_seconds" in ds
        and "min_stop_duration" in sd
        and ds["min_duration_seconds"] != int(sd["min_stop_duration"])
    ):
        drift.append(
            f"min duration (dead_spots.min_duration_seconds={ds['min_duration_seconds']} "
            f"vs stop_detection.min_stop_duration={sd['min_stop_duration']})"
        )
    if drift:
        logger.warning(
            "Stop-detection knobs diverge between config blocks: %s. stop_detection.* is "
            "authoritative for the pipeline's stop removal; dead_spots.* drives only the "
            "standalone `analyze` preview — keep them equal or the preview will misrepresent "
            "what the pipeline cuts.",
            "; ".join(drift),
        )


def _load_encoding_constants():
    """Lazy-load encoding constants to avoid circular import with pipeline package."""
    from gomoveshift.pipeline.encoding_constants import (
        LONGFORM_AUDIO_LRA,
        LONGFORM_AUDIO_LUFS,
        LONGFORM_AUDIO_MAX_LUFS,
        LONGFORM_AUDIO_TP,
        LONGFORM_CRF,
        LONGFORM_PIX_FMT,
        LONGFORM_PRESET,
        LONGFORM_PROFILE,
    )

    return {
        "crf": LONGFORM_CRF,
        "preset": LONGFORM_PRESET,
        "profile": LONGFORM_PROFILE,
        "pix_fmt": LONGFORM_PIX_FMT,
        "audio_target_lufs": LONGFORM_AUDIO_LUFS,
        "audio_target_tp": LONGFORM_AUDIO_TP,
        "audio_target_lra": LONGFORM_AUDIO_LRA,
        # Freeze the loudness ceiling too, so YAML can't silently raise the
        # absolute audio ceiling and defeat the near-silent gate.
        "audio_max_lufs": LONGFORM_AUDIO_MAX_LUFS,
    }


class PathsConfig(BaseModel):
    input_folder: str = ""
    output_folder: str = ""
    # ``<system temp dir>/gomoveshift``: /tmp (or $TMPDIR) on Linux, %TEMP% on Windows.
    temp_folder: str = Field(
        default_factory=lambda: os.path.join(tempfile.gettempdir(), "gomoveshift")
    )
    intermediate_folder: str = ""  # ext4 override for drvfs intermediates (empty = use temp_folder)


class ExtractionConfig(BaseModel):
    sample_divisor: int = 3
    motion_sample_fps: int = 5


class SignDetectionConfig(BaseModel):
    roi_top_percent: float = 0.1
    roi_bottom_percent: float = 0.7
    min_sign_area_percent: float = 0.005
    max_sign_area_percent: float = 0.15


class OCRConfig(BaseModel):
    languages: list[str] = Field(default_factory=lambda: ["en"])


class DeadSpotsConfig(BaseModel):
    motion_threshold: float = 0.02
    min_duration_seconds: int = 30
    trim_buffer_seconds: int = 2


class SegmentationConfig(BaseModel):
    target_duration_minutes: int = 60
    min_duration_minutes: int = 50
    max_duration_minutes: int = 70


class OutputConfig(BaseModel):
    codec: str = "libx265"
    crf: int = 18  # Frozen — canonical LONGFORM_CRF from encoding_constants.py
    preset: str = "slow"  # Frozen — must match LONGFORM_PRESET in encoding_constants.py
    profile: str = "main10"  # Frozen — must match LONGFORM_PROFILE in encoding_constants.py
    pix_fmt: str = "yuv420p10le"  # Frozen — must match LONGFORM_PIX_FMT in encoding_constants.py
    audio_codec: str = "aac"
    audio_bitrate: str = "384k"  # High quality audio
    audio_target_lufs: float = (
        -45.0
    )  # near-silent attenuate-only ceiling — canonical LONGFORM_AUDIO_LUFS (frozen)
    audio_target_tp: float = -10.0  # true-peak cap (frozen — canonical LONGFORM_AUDIO_TP)
    audio_target_lra: float = 11.0  # Loudness range (measurement-command shape only)
    audio_max_lufs: float = -43.0  # Validation gate: reject if LOUDER than ceiling (frozen)
    embed_chapters: bool = True

    def model_post_init(self, __context) -> None:
        """Force frozen encoding fields to canonical values after construction.

        Quality-critical fields (crf, preset, profile, pix_fmt) are locked to
        encoding_constants.py values. YAML overrides for these fields are
        rejected with a warning to prevent config drift.
        """
        frozen = _load_encoding_constants()
        for field_name, canonical_value in frozen.items():
            user_value = getattr(self, field_name)
            if user_value != canonical_value:
                logger.warning(
                    "Encoding field %r overridden: %r -> %r — quality-critical: "
                    "these values are frozen in encoding_constants.py to prevent "
                    "config drift that degrades YouTube encode quality",
                    field_name,
                    user_value,
                    canonical_value,
                )
                object.__setattr__(self, field_name, canonical_value)


class WebUIConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765


class HysteresisConfig(BaseModel):
    """Dual enter/exit thresholds for stop classification.

    Defaults preserve the single-threshold behavior exactly: ``enabled``
    is False, and ``exit_multiplier`` is 1.0 so even an accidental
    ``enabled: true`` with the default multiplier reproduces today's
    output (enter and exit boundaries coincide).
    """

    enabled: bool = False
    # Exit "stopped" only when energy > motion_threshold * exit_multiplier.
    # Must be >= 1.0 so the exit band sits at or above the enter band.
    exit_multiplier: float = 1.0


class GPSSpeedCorroborationConfig(BaseModel):
    """GPS-speed corroboration of optical stops.

    Two-way cross-check against the GPS trace, default-on but a strict no-op for
    GPS-absent LFs (``gps_data is None`` / empty trackpoints):

    * SUBTRACTIVE — an optical "stop" whose GPS speed over the same source-time
      span is clearly above ``stop_speed_kmh`` is a false dark/tunnel stop and is
      dropped.
    * ADDITIVE — a sustained sub-``stop_speed_kmh`` span whose median speed is at
      or below ``parked_speed_kmh`` (≈ 0, allowing for parked-GPS drift) is a
      real STOP even when optical motion stayed above threshold (parked with
      passing traffic), so it is EMITTED as a stop the optics missed.

    Default-on because a GPS-less LF ignores it entirely; only a GPS-bearing LF
    changes, and there the GPS trace is authoritative for whether the vehicle
    actually moved.
    """

    enabled: bool = True
    stop_speed_kmh: float = 10.0  # GPS speed above this means "still moving"
    # Median GPS speed (km/h) at or below which a sustained span is treated as
    # genuinely parked (≈ 0) for the ADDITIVE source — high enough to tolerate
    # parked-GPS jitter, well below walking pace so a slow crawl never qualifies.
    parked_speed_kmh: float = 2.0


class StopDetectionConfig(BaseModel):
    """Configuration for stop detection (more sensitive than dead spots)."""

    min_stop_duration: float = 5.0  # Ignore stops shorter than 5 seconds
    motion_threshold: float = 0.008  # Lower threshold for sensitivity
    crossfade_duration: float = 0.5  # Crossfade between segments (0 = hard cuts)
    hysteresis: HysteresisConfig = Field(default_factory=HysteresisConfig)
    gps_speed_corroboration: GPSSpeedCorroborationConfig = Field(
        default_factory=GPSSpeedCorroborationConfig
    )


class ConfigError(RuntimeError):
    """Raised when ``Config`` filesystem-level validation fails at startup.

    Distinct from Pydantic's ``ValidationError`` (shape/type checks): covers
    operator-repairable invariants like a missing co-located checkout. CLI
    maps this to exit 2 (vs ``PipelineValidationError`` exit 1, SIGTERM 143).
    """


class SEOConfig(BaseModel):
    """Configuration for SEO integration (cross-project paths).

    Defaults are cwd-relative — every concrete deployment must override
    via config.yaml ``seo:`` section, so code that constructs
    ``LongFormMetadataGenerator`` without an explicit ``seo_storage_path=``
    never touches another deployment's real data directory. Tests should
    ALSO rely on the autouse tempdir fixture in ``tests/conftest.py``
    (``isolate_seo_storage_from_workspace``) as belt-and-suspenders.
    """

    storage_path: str = "./data/seo"
    credentials_dir: str = "./credentials"
    manager_path: str = "."
    # The site whose ledger this console/pipeline reads and writes. The SEO
    # service is multi-tenant (brand = tenant boundary), so each deployment
    # sets its own.
    site_id: str = "gomoveshift"
    # Escape hatch for dry-run / test invocations that don't touch SEO. When
    # True, ``validate_paths_exist`` returns immediately without filesystem IO.
    skip_validation: bool = False
    # ``enqueue-plan``'s rolling 7-day per-platform ceiling: the max
    # deliveries for one platform whose scheduled_at falls in any rolling
    # week, counted against the WHOLE queue (no region/project filter --
    # that is what lets it catch a second region enqueuing into the same
    # week, the case a per-drive gate cannot see).
    #
    # Default of 21 = the default cadence's whole-account steady state,
    # NOT a per-region figure: the default publish slots schedule 3
    # long-forms/week (Ireland Sun, USA Tue, USA Fri) x 7 shorts/long-form (``DriveConfig.min_shorts_per_lf``) x 1
    # delivery per platform per short = 21. This is a ceiling against
    # ACCIDENTAL doubling (e.g. two regions' plans landing in the same
    # week), not a volume target -- lowering it is a deliberate volume
    # decision gated on measured evidence, never a knob to
    # tune casually.
    rolling_week_ceiling_per_platform: int = 21

    @property
    def events_jsonl(self) -> str:
        """Path to this site's SEO events ledger."""
        return f"{self.storage_path}/{self.site_id}/events.jsonl"

    def validate_paths_exist(self, strict: bool = True) -> None:
        """Fail fast at startup if SEO paths are missing.

        Checks: parent of ``storage_path`` (the JSONL itself may be absent
        on a fresh install) and ``manager_path``. ``credentials_dir`` is
        deliberately NOT checked — the SOPS code path in
        ``pipeline/_yt_creds.py`` only ever reads
        ``creds_dir.parent / "youtube_upload_token.json.enc"`` (never
        anything *inside* ``creds_dir``), and the BrandCredentials
        fallback has its own ``gomoveshift_creds.exists()`` guard in
        ``_yt_creds.py``. Requiring the directory to exist on disk
        was redundant with the SOPS path and interrupted a launch mid-run when the
        directory was absent on a fresh checkout.

        Raises :class:`ConfigError` listing every missing path one-per-line;
        returns ``None`` on success or when ``self.skip_validation`` is True.

        ``strict`` is reserved for future per-path optionality. Named
        ``validate_paths_exist`` rather than ``assert_paths_exist`` to sidestep ``MagicMock``'s
        ``assert_*`` attribute-safety guard, which broke every CLI handler
        test that passes ``MagicMock()`` as ``config``.
        """
        if self.skip_validation:
            return
        missing: list[str] = []
        storage = Path(self.storage_path)
        if not storage.is_absolute():
            # The content ledger resolves to ``<storage_path>/
            # .content_ledger.json``. A cwd-relative value silently selects a
            # DIFFERENT ledger per launch directory, voiding the slot-collision
            # avoidance the ledger exists to provide. Fail loudly at startup
            # rather than fall back to a plausible-looking wrong ledger.
            raise ConfigError(
                f"seo.storage_path must be an absolute path, got {self.storage_path!r}. "
                "A relative path resolves against the launch directory, so two runs "
                "started from different cwds use different content ledgers and can "
                "double-book the same publish slot. Set an absolute path in config.yaml."
            )
        storage_parent = storage.parent
        if not storage_parent.is_dir():
            missing.append(f"  storage_path parent: {storage_parent}")
        manager_dir = Path(self.manager_path)
        if not manager_dir.is_dir():
            missing.append(f"  manager_path:       {manager_dir}")
        if missing:
            raise ConfigError(
                "SEO config paths missing (set seo.skip_validation=true to bypass):\n"
                + "\n".join(missing)
            )


# Filename of the posted-photo ledger inside the photo output dir.
PHOTO_STATE_FILENAME = ".photo_selection_state.json"

#: Publishing route values a run can no longer honor. An un-migrated
#: config.yaml still naming one must not brick ``Config.load()`` (which
#: ``gomoveshift.pipeline`` calls at import): the route coerces to ``off``
#: with one WARNING, publishing nothing on that platform until the operator
#: writes ``first_party`` explicitly. The only place the value is spelled.
RETIRED_PUBLISHING_ROUTES = frozenset({"metricool"})

# ── photos: THE photo-knob registry — one knob, one row ──────────────────────
# Every look/layer table below is DERIVED from this tuple, so a new knob is a new
# ROW and two knob branches edit different lines. Not cosmetic: parallel branches
# that each append to the same few shared lines conflict, because git cannot merge
# two edits to one line. ``masks.MASK_PARAMS``
# (each mask's params read off its signature) is the same derive-don't-list move.
#
# The registry is a LITERAL and must stay one — nothing here may import
# ``gomoveshift.pipeline.*``: ``Config.load`` runs at MODULE IMPORT time
# (longform_metadata builds two path constants off it), so such an import re-enters
# a half-initialised package and takes the WHOLE CLI down with an ImportError
# naming GOLDEN_COLUMNS — one configured look, every command dead (as ``border``
# mirrors BORDER_STYLES). The sync test in tests/test_photo_looks.py keeps it
# honest, importing DevelopOptions (test-time imports are outside the cycle) and
# failing on a knob missing here; tests/test_photo_knob_registry.py pins every
# derived table and refuses a half-wired row.


class Knob(NamedTuple):
    """One photo knob and every surface it belongs to, declared once.

    ``name`` is the ``photos develop`` FLAG name, because a look is recalled by
    standing in for unset flags; ``alt`` is the ``DevelopOptions`` field name when
    argparse spells it differently. ``span`` is the inclusive scalar gate,
    mirroring that flag's own exactly, so a bad number fails at config LOAD naming
    the look instead of at render time (None = no scalar bound — a string, a list,
    or a value the ``PhotosConfig`` field validators and the handler already gate).
    ``items`` is a list knob's (accepted length(s), lo, hi) — a bare int for the one
    length ``_resolve_tint`` / ``_resolve_mono`` take, a TUPLE for a knob whose flag
    takes several, as ``_resolve_hue_band`` takes six anchors or seven with orange
    30° second. One exact length was a type that could not say "6 or 7", so every
    look door refused a band the flag itself accepted. ``curve`` marks the free-length
    curves — a FLAT list of alternating x y control points, x strictly ascending,
    every value in [0, 1], shaped by ``check_look_lists`` below. ``span`` covers
    scalars only, so before those two a list knob passed BOTH look doors unchecked
    and raised inside ``CorrectionParams`` once per photo, mid-batch, after the
    decodes were paid. ``layer`` is "colour", "tone" or "" — a spatial stage is a
    property of the whole rendered frame, so a masked one would be a lens effect
    that stops at a horizon, refused here rather than rendered — and ``look`` is
    False for a knob only a masked layer may carry. ``run`` marks the opposite
    case: a flag the RUN owns outright, refused in a look by name below.
    """

    name: str
    span: tuple[float, float] | None = None
    items: tuple[int | tuple[int, ...], float, float] | None = None
    curve: bool = False
    look: bool = True
    layer: str = ""
    alt: str = ""
    run: bool = False

    @property
    def lengths(self) -> tuple[int, ...]:
        """Every list length this knob accepts, normalized — THE one place a row's
        bare int becomes the tuple the doors test membership on, so no gate has to
        know which spelling its row used."""
        want = self.items[0] if self.items is not None else ()
        return want if isinstance(want, tuple) else (want,)


# Operator travel on the five white-balance axes ``compute_correction`` MEASURES.
# DERIVED (and mirrored as a literal — this module may not import the pipeline):
# the operator value is ADDED after ``_styled``/``_amplified`` have run, so the
# style rails never see it and cannot clamp it. ``styles.STYLE_WB_CAP`` (0.10) is
# where those rails hold a style at the shipped x1 strength, ``styles.WB_HARD_CAP``
# (0.20) the renderer authority binding at EVERY strength — so the travel between
# them is what a style at its rail can still take without outrunning the renderer.
WB_OPERATOR_SPAN = 0.10

# ``grain_size`` is 0.25..4.0 rather than 0..1 because it is a RATIO on the grain
# field's reference grid, not an amount: at 0.25 that grid is 4800 px, above every
# realistic export, so the field falls back to the per-pixel draw and the resolution
# independence the grid exists for dissolves; at 4.0 the grid is 300 px, one speck
# spanning ~19 px of a 5600 px export — mottling, not silver. Both ends stay far clear
# of the ``max(2, ...)`` floor on the drawn grid, so the resample always has a field.
PHOTO_KNOBS = (
    # ── spatial: whole-frame stages, never a masked layer ──────────────────────
    Knob("vignette", span=(0.0, 1.0)),
    Knob("dehaze", span=(0.0, 1.0)),
    Knob("clarity", span=(-1.0, 1.0)),
    Knob("grain", span=(0.0, 1.0)),
    Knob("grain_size", span=(0.25, 4.0)),
    Knob("halation", span=(0.0, 1.0)),
    Knob("halation_tint", items=(3, 0.0, 1.0)),
    Knob("chromatic_ab", span=(0.0, 1.0)),
    Knob("holo_sheen", span=(0.0, 1.0)),
    Knob("scanline", span=(0.0, 1.0)),
    Knob("glitch", span=(0.0, 1.0)),
    Knob("sharpen", span=(0.0, 2.0)),
    Knob("sky_darken", span=(0.0, 1.0)),
    Knob("sky_saturate", span=(0.0, 1.0)),  # flag/look only, as white_balance_g is
    Knob("graduated_nd", span=(0.0, 1.0)),
    # Degrees the grad ND is rotated off level in its holder. Bounded at +/-90, not
    # 180: past a quarter turn the ramp points at the GROUND, so the one output a
    # grad ND must not produce is unexpressible rather than gated after the render.
    Knob("graduated_nd_angle", span=(-90.0, 90.0)),
    Knob("light_leak", span=(0.0, 1.0)),
    Knob("glow", span=(0.0, 1.0)),
    # ── colour: a look, and (bar the two below) a masked layer ─────────────────
    # Six anchors, or seven with orange 30° second — exactly what ``_resolve_hue_band``
    # takes, so a band the flag accepts is a band a look can carry.
    Knob("hue_shift", items=((6, 7), -180.0, 180.0), layer="colour"),
    Knob("hue_sat", items=((6, 7), 0.0, 5.0), layer="colour"),
    Knob("hue_lum", items=((6, 7), 0.0, 5.0), layer="colour"),
    Knob("split_tone_shadow", items=(3, -0.5, 0.5), layer="colour", alt="splittone_shadow"),
    Knob("split_tone_midtone", items=(3, -0.5, 0.5), layer="colour", alt="splittone_midtone"),
    Knob("split_tone_highlight", items=(3, -0.5, 0.5), layer="colour", alt="splittone_highlight"),
    Knob("curve_l", curve=True, layer="colour"),
    Knob("curve_r", curve=True, layer="colour"),
    Knob("curve_g", curve=True, layer="colour"),
    Knob("curve_b", curve=True, layer="colour"),
    Knob("vibrance", span=(0.0, 2.0), layer="colour"),
    Knob("bw_mix", items=(3, -2.0, 3.0), layer="colour", alt="mono_mix"),
    Knob("duotone", items=(6, 0.0, 1.0), layer="colour"),
    Knob("duotone_strength", span=(0.0, 1.0), layer="colour"),
    Knob("duotone_mid", items=(3, 0.0, 1.0), layer="colour"),
    # White balance. Green keeps its own wide span — no producer and no style rail
    # touches that channel, so the whole range is the operator's; the five below
    # share ``styles._WB_FIELDS``' rails, so their span is the headroom those leave.
    # All six are ``layer="colour"``: "warm the ground, leave the sky" is the local
    # move an operator reaches for first, and the renderer already did it — every
    # axis is a field ``DevelopOptions`` and ``CorrectionParams`` share, so
    # ``develop._look_params`` derives it and ``_grade`` routes it through
    # ``_apply_colorbalance``. Only this flag was missing, and ``check_layers``
    # refuses a key the layer set does not carry — a built capability fenced off at
    # the door. Whole-frame the operator's value is ADDED to the measured cast
    # (``develop._wb_shifted``); inside a layer the base record is fresh, so there
    # is nothing to add to and the number applied is the number written.
    Knob("white_balance_g", span=(-0.5, 0.5), layer="colour"),  # signed: + green, - magenta
    Knob("white_balance_r", span=(-WB_OPERATOR_SPAN, WB_OPERATOR_SPAN), layer="colour"),  # + warmer
    Knob("white_balance_b", span=(-WB_OPERATOR_SPAN, WB_OPERATOR_SPAN), layer="colour"),  # + cooler
    Knob("white_balance_r_high", span=(-WB_OPERATOR_SPAN, WB_OPERATOR_SPAN), layer="colour"),
    Knob("white_balance_b_high", span=(-WB_OPERATOR_SPAN, WB_OPERATOR_SPAN), layer="colour"),
    Knob("white_balance_g_high", span=(-0.5, 0.5), layer="colour"),  # green's highlight half
    # Friendlier Kelvin/tint pair over the same r/b/g axes above — ``develop._wb_shifted``
    # converts and ADDS them on top, so a look may use either or both. Whole-frame only
    # (no ``layer=``): a masked layer's own record has nothing to add onto.
    Knob("white_balance_kelvin", span=(-2000.0, 2000.0)),
    Knob("white_balance_tint", span=(-0.5, 0.5)),
    # Exposure (stops) and highlight/shadow recovery — ``exposure_ev``/``shadow_recovery``/
    # ``highlight_recovery`` are fields ``DevelopOptions`` and ``CorrectionParams`` share, so
    # ``develop._look_params`` derives them for a ``--look-variant``; ``plan_variants`` wires
    # them explicitly for a main run. 0.0 is OFF on every axis.
    Knob("exposure_ev", span=(-3.0, 3.0)),
    Knob("shadow_recovery", span=(0.0, 20.0)),
    Knob("highlight_recovery", span=(0.0, 20.0)),
    Knob("reference", alt="reference_lab"),  # the flag names a FILE, the field its stats
    # Geometry is the RUN's (``DevelopOptions.straighten_angle``/``crop_rect``): a look
    # that reshaped the frame would make two variants of one photo incomparable, so
    # both are refused in a look by name. ``--crop`` is gated at its own flag door.
    Knob("straighten", alt="straighten_angle", span=(-45.0, 45.0), look=False, run=True),
    Knob("crop", alt="crop_rect", look=False, run=True),
    # Lens-profile correction runs before geometry, so a look that reshaped the
    # frame would be measuring a different lens/EXIF correction each variant
    # — the run's choice, like straighten/crop, refused in a look by name.
    Knob("lens_correct", look=False, run=True),
    # ── tone: a masked layer ONLY. No DevelopOptions field carries one, so
    # ``develop._layered`` sets these onto the layer's ``CorrectionParams`` record
    # directly — the colour round trip that carries the set above would drop one
    # silently and render nothing. Every bound is read off the applier that renders
    # it, so a value the door accepts is a value that MOVES the picture (the
    # appliers live in gomoveshift/pipeline/color_correction.py).
    # ``GAMMA_MAX`` is the grade chain's own ceiling; the floor is its reciprocal,
    # so darkening a region has exactly the authority brightening it has, and stays
    # above 0 because ``_apply_gamma`` raises the frame to the power 1/gamma.
    Knob("gamma", span=(0.77, 1.30), look=False, layer="tone"),
    # ``_apply_curves`` clamps to ``BLACK_POINT_MIN``/``BLACK_POINT_MAX`` — past those the value renders as
    # the edge, which is a knob that lies about what it did.
    Knob("black_point", span=(-10.0, 20.0), look=False, layer="tone"),
    # ``CorrectionParams.contrast_curve``'s documented span; the S-curve knots
    # 0.25-c / 0.75+c in ``_apply_curves`` stay inside [0, 1] well past it.
    # NEGATIVE is refused because ``_apply_curves`` gates the whole
    # curve on c > 0.001 — a de-contrast move would render nothing at all.
    Knob("contrast_curve", span=(0.0, 0.15), look=False, layer="tone"),
    # ``_apply_saturation`` scales chroma around luma, so 0 is fully drained and a NEGATIVE
    # multiplier inverts hue rather than desaturating; the ceiling mirrors the
    # sibling chroma knob ``vibrance`` above.
    Knob("saturation_mult", span=(0.0, 2.0), look=False, layer="tone"),
    # ── the LUT surface, style + export, and the look's own dial ───────────────
    # The LUT SELECTION is the run's alone. ``--lut`` APPENDS, so its dest holds a
    # LIST of specs, while ``_apply_look`` fills a dest with the look's raw value —
    # a bare ``lut: portra`` reached the resolver as ``p``, ``o``, ``r``… The three
    # below stay look-able: each is a scalar with its own ``photos.<knob>`` default,
    # so a look setting one is another config layer over whatever the run selected.
    Knob("lut", look=False, run=True),
    Knob("lut_strength", span=(0.0, 1.0)),
    Knob("lut_over"),
    Knob("lut_auto"),
    Knob("styles"),
    Knob("style_strength"),
    Knob("border"),
    Knob("format"),
    Knob("layers"),  # masked colour/tone recipes — see check_layers
    Knob("look_strength", span=(0.0, 1.0)),  # this look's own dial, over the base render
)
# ── the surfaces, every one derived from the registry above ──────────────────
LOOK_KNOBS = frozenset(k.name for k in PHOTO_KNOBS if k.look)
LOOK_ALIASES = {k.alt: k.name for k in PHOTO_KNOBS if k.alt}
LOOK_RANGES = {k.name: k.span for k in PHOTO_KNOBS if k.span is not None}
LOOK_LIST_RANGES = {k.name: (k.lengths, *k.items[1:]) for k in PHOTO_KNOBS if k.items is not None}
LOOK_CURVE_KNOBS = frozenset(k.name for k in PHOTO_KNOBS if k.curve)
# Real develop flags a look may NOT name, so the refusal can say so by name rather
# than call a shipped flag an unknown knob.
LOOK_RUN_KNOBS = frozenset(k.name for k in PHOTO_KNOBS if k.run)


def check_look_lists(label: str, knobs) -> None:
    """Refuse a malformed LIST knob — curve, hue band, split tone, B&W mix,
    duotone — naming ``label`` (the look, the layer or the flag).

    THE one home for the shape and range of a look's lists, so every door agrees:
    config load, a masked layer's own knobs (``check_layers``) and
    ``--look-variant`` all route here, and a value ``--look`` refuses can no
    longer render through ``--look-variant``. Empty is OFF, exactly as the flag
    resolvers read it. Raises ``ValueError``; returns nothing when every list is
    renderable."""
    for key, value in knobs.items():
        curve = key in LOOK_CURVE_KNOBS
        if not curve and key not in LOOK_LIST_RANGES:
            continue
        if not isinstance(value, (list, tuple)) or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) for v in value
        ):
            raise ValueError(f"{label}: {key} must be a list of numbers, got {value!r}")
        vals = list(value)
        if not vals:
            continue
        if curve:
            lo, hi = 0.0, 1.0
            if len(vals) % 2 or len(vals) < 4:
                raise ValueError(
                    f"{label}: {key} needs an even-length flat list of >= 2 (x, y) "
                    f"points, got length {len(vals)}"
                )
            xs = vals[0::2]
            if any(b <= a for a, b in zip(xs, xs[1:])):
                raise ValueError(f"{label}: {key} x control points must be strictly ascending")
        else:
            lengths, lo, hi = LOOK_LIST_RANGES[key]
            if len(vals) not in lengths:
                want = " or ".join(str(n) for n in lengths)
                raise ValueError(f"{label}: {key} needs {want} values, got {len(vals)}")
            # Not normalized — a sum away from 1 is a legitimate exposure move —
            # but a non-positive sum inverts or blacks out the frame.
            if key == "bw_mix" and sum(vals) <= 0.0:
                raise ValueError(f"{label}: bw_mix weights must sum to more than 0")
        if any(not lo <= v <= hi for v in vals):
            raise ValueError(f"{label}: {key} values must each be in [{lo:g}, {hi:g}]")


# Valid-value sets for the photo validators below, mirrored as LITERALS.
# A validator may not import ``gomoveshift.pipeline.*``: config validation runs
# at module IMPORT time (``pipeline/longform_metadata.py`` calls ``get_config()``
# at import), so the import re-enters the half-built pipeline package and every
# command dies with a circular ImportError naming CSV generation — nowhere near
# the operator's config file. Deferring the import inside the validator does not
# help; the validator is already running inside an import.
# ``tests/test_photo_config_import_cycle.py`` pins each literal to its upstream
# source of truth, so one cannot go stale when the other grows a value.
PHOTO_LUT_OVER_STYLES = (  # mirrors pipeline.photos.styles.STYLES
    "natural",
    "vivid",
    "moody",
    "golden",
    "mono",
    "soft",
    "matte_shoulder",
    "matte_knee",
    "daylight",
    "cloudy",
    "shade",
)
PHOTO_SOCIAL_ASPECTS = ("square", "4:5", "9:16")  # mirrors resizer.SOCIAL_ASPECTS
PHOTO_EXPORT_FORMATS = ("jpeg", "tiff16")  # mirrors develop.EXPORT_FORMATS exactly
# ── a look's ``layers``: one grade recipe gated by one named mask ────────────
# Mirrors ``pipeline.photos.masks.MASKS`` — each name with the shape params that
# mask accepts — as a literal for the same reason LOOK_KNOBS mirrors the flag
# names: nothing here may import the pipeline; the sync test in
# tests/test_photo_masked_layers.py keeps the two honest. A layer may carry the
# params of the mask it NAMES and no others, so ``angle`` (real, for ``linear``)
# on ``sky`` is refused here rather than carried inert into the render.
LAYER_MASK_PARAMS = {
    "sky": (),
    "ground": (),
    "shadows": (),
    "midtones": (),
    "highlights": (),
    "radial": ("cx", "cy", "size", "aspect"),
    "linear": ("angle", "span", "offset"),
    "glints": ("max_area",),
    "texture": (),
    "hue": ("center", "width", "min_chroma"),
}
LAYER_MASKS = frozenset(LAYER_MASK_PARAMS)
# A layer may name TWO masks and one of these instead of a single name, so a
# recipe can ride "the highlights INSIDE the centre" — a region no single entry
# expresses. Mirrors ``masks.MASK_OPS``, pinned by the same sync test.
LAYER_MASK_OPS = frozenset({"intersect", "union", "subtract"})
# The knobs a layer may carry, each derived from the ``layer=`` it declares in
# PHOTO_KNOBS: colour, plus the TONE knobs (the most-used local move there is —
# lift the shadows under a mask, steepen the sky), each of which must be a real
# ``CorrectionParams`` field or ``_layered``'s ``replace`` refuses it outright.
# ``mask``/``mask_op``/``feather``/``invert`` are the layer's own four.
LAYER_COLOUR_KNOBS = frozenset(k.name for k in PHOTO_KNOBS if k.layer == "colour")
LAYER_TONE_KNOBS = frozenset(k.name for k in PHOTO_KNOBS if k.layer == "tone")
LAYER_KNOBS = LAYER_COLOUR_KNOBS | LAYER_TONE_KNOBS


def check_layers(label: str, layers) -> None:
    """Refuse a malformed masked-layer list, naming ``label`` (the look or flag).

    The gate sits at config LOAD and at the CLI door — upstream of any decode —
    because a layer that only fails once a render reaches it costs a batch of
    decodes to find. Raises ``ValueError``; returns nothing when every layer is
    renderable."""
    if not isinstance(layers, list):
        raise ValueError(f"{label}: layers must be a list of {{mask: NAME, <knob>: ...}}")
    for layer in layers:
        if not isinstance(layer, dict) or "mask" not in layer:
            raise ValueError(f"{label}: every layer needs a 'mask' name, got {layer!r}")
        mask, op = layer["mask"], layer.get("mask_op")
        for one in mask if isinstance(mask, list) else [mask]:
            if not isinstance(one, str) or one not in LAYER_MASKS:
                names = ", ".join(sorted(LAYER_MASKS))
                raise ValueError(f"{label}: unknown mask '{one}' (valid: {names})")
        if isinstance(mask, list):
            if len(mask) != 2:
                raise ValueError(f"{label}: a mask_op combines exactly two mask names, got {mask}")
            if op not in LAYER_MASK_OPS:
                ops = ", ".join(sorted(LAYER_MASK_OPS))
                raise ValueError(f"{label}: masks {mask} need a 'mask_op' of: {ops}, got {op!r}")
        elif op is not None:
            raise ValueError(f"{label}: 'mask_op' combines two mask names; '{mask}' is one")
        if not any(key in LAYER_KNOBS for key in layer):
            raise ValueError(f"{label}: layer '{layer['mask']}' carries no knob to apply")
        feather = layer.get("feather", 1.0)
        if not (
            isinstance(feather, (int, float)) and not isinstance(feather, bool) and feather > 0
        ):
            raise ValueError(f"{label}: layer feather must be a number > 0, got {feather!r}")
        if not isinstance(layer.get("invert", False), bool):
            raise ValueError(f"{label}: layer invert must be true or false")
        # A pair's shape params are the UNION of both operands' — ``mask_for`` hands the
        # same ``params`` to each mask, and it accepts on the same union, so the door and
        # the render agree on what is legal.
        named = mask if isinstance(mask, list) else [mask]
        shape = {key for one in named for key in LAYER_MASK_PARAMS[one]}
        for key, value in layer.items():
            if key in shape:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"{label}: layer {key}={value!r} must be a number")
                continue
            if key not in LAYER_KNOBS and key not in ("mask", "mask_op", "feather", "invert"):
                known = ", ".join(sorted(set(LAYER_KNOBS) | set(shape)))
                raise ValueError(f"{label}: '{key}' is not a layer knob (valid: {known})")
            lo, hi = LOOK_RANGES.get(key, (None, None))
            if lo is not None and not (isinstance(value, (int, float)) and lo <= value <= hi):
                raise ValueError(f"{label}: {key}={value!r} must be in [{lo:g}, {hi:g}]")
        check_look_lists(f"{label} layer '{layer['mask']}'", layer)


class PhotosConfig(BaseModel):
    """Photo-post pipeline paths and encode knobs.

    Every path used to be a module-level absolute literal inside
    ``pipeline/photos/`` with no override at all, evaluated at import
    time. They live here instead so ``config.yaml`` can move them, and
    the two env-backed ones resolve through ``gomoveshift.paths`` (the
    single home for env-var-backed paths) at *call* time — nothing is
    baked in at import.

    Empty string means "derive": ``output_dir`` falls back to
    ``GMS_PHOTO_OUTPUT_DIR`` and ``state_path`` rides ``output_dir`` so
    the state file can never drift away from the run's output. Defaults
    reproduce the pre-config paths exactly.
    """

    # Hugo site checkout the journey photos are read from. No default: unset
    # means the photo features report "not configured" (see
    # ``resolve_site_root``).
    site_root: str = ""
    output_dir: str = ""  # "" → paths.gms_photo_output_dir()
    state_path: str = ""  # "" → <output_dir>/.photo_selection_state.json
    resized_base: str = ""  # "" → <paths.gms_photo_output_dir()>/resized
    target_width: int = 1080  # Social-media upload width (px)
    jpeg_quality: int = 85  # Social resize encode quality
    # Developed gallery masters are re-encoded by the social resizer later, so
    # exporting them at the social quality pays the generation loss twice —
    # the master encode stays near-transparent and the resize is the lossy hop.
    gallery_jpeg_quality: int = 92
    # HE/HE* (TicoRAW) NEF true-raw hook: Adobe DNG Converter executable (may
    # be a Windows path under /mnt/c — WSL2 runs .exes directly). Empty =
    # disabled: an unpackable NEF falls back to its embedded full-res preview.
    he_dng_converter_exe: str = ""
    he_dng_cache_dir: str = ""  # converted .dng output; "" → alongside the source NEF
    # Phone GPX track directory (GPSLogger's sync target, flat): the day's
    # ``*.gpx`` files become a position index (built by ``sun_track.load_track_index``) so a
    # capture time can take a (lat, lon). "" = off — no track lookup at all.
    gpx_dir: str = ""
    # ``photos develop`` (style variants into a LOCAL working dir, never the
    # public site tree): "" → --out required / auto-select per SceneProfile.
    developed_output_dir: str = ""
    # Where every ``--contact-sheet`` lands, ALWAYS — the run's ``--out`` gets the
    # masters, this gets the sheet. A sheet is a REVIEW artifact: written beside
    # the masters it follows the develop into whatever scratch dir, tmp path or
    # second checkout the run happened to use, and the operator never sees the one
    # thing they asked to look at. Point it at a review volume (for example a
    # Windows drive under WSL); the run's ``--out`` basename becomes the
    # subfolder, so packs developed side by side keep their sheets apart without
    # needing a clock. Refused at the CLI door when the volume is not mounted, so
    # an unmounted volume costs nothing rather than a whole render.
    contact_sheet_root: str = ""  # "" → <paths.gms_data_home()>/contact-sheets
    develop_styles: str = ""
    # Style lever: amplifies every style's delta over the measured base.
    # 1.0 is the operator-calibrated bake; >1 pushes the looks toward renderer
    # authority, <1 softens them. Per-run override: --style-strength.
    style_strength: float = 1.0
    # Named film-emulation LUTs for ``photos develop --lut NAME|PATH``: a look
    # name → .cube file path. ``--lut`` accepts a name mapped here or a direct
    # path. A LUT look is a fixed table — the style-strength lever never scales
    # it. The operator supplies the .cube files; none are bundled.
    luts: dict[str, str] = {}
    # LUT opacity lever for ``photos develop --lut``: blend every film-emulation
    # look toward the natural (base) grade. 1.0 = full-strength LUT (the shipped
    # behavior); 0.0 = passthrough (no LUT). One global value across every --lut
    # look. Per-run override: --lut-strength. Range 0..1.
    lut_strength: float = 1.0
    # Style graded UNDER every ``photos develop --lut`` look: a STYLES key, so a
    # film emulation can ride e.g. moody's matte instead of the neutral base.
    # "" = the base grade (the shipped, byte-identical default). The style lever
    # scales it. Per-run override: --lut-over.
    lut_over: str = ""
    # Automatic LUT picks per photo for ``photos develop``: score every candidate
    # .cube against the photo's SceneProfile and render the N best. 0 = OFF (the
    # shipped manual path, byte-identical). Candidates are the ``--lut`` specs when
    # any are given, else every key of ``luts`` above. Per-run: --lut-auto.
    lut_auto: int = Field(0, ge=0)
    # Radial-vignette depth for ``photos develop``: a smooth resolution-independent
    # corner darkening applied to every variant's graded frame. 0.0 = off (the
    # shipped, byte-identical default); 1.0 = corners fully darkened. Per-run
    # override: --vignette. Range 0..1.
    vignette: float = 0.0
    # Dehaze for ``photos develop``: real atmospheric-veil SUBTRACTION via the dark-channel
    # prior (distinct from the S-curve contrast increment ``color_correction`` calls
    # dehaze), run FIRST of the spatial stages. The veil colour is MEASURED off the frame's
    # own haziest pixels, so it cannot misfire on the sub-clipping tone a real grade
    # produces. 0.0 = off (byte-identical). Per-run: --dehaze. Range 0..1.
    dehaze: float = 0.0
    # Sky darkening for ``photos develop``: the classic landscape move, rendered through
    # a SOFT sky mask that mirrors the exposure meter's bright-upper-band rule (landscape
    # frames only — a portrait still's bright top is the subject) and is measured against
    # the frame's own luma distribution, so it cannot misfire on the sub-clipping tone a
    # real grade produces. Runs under the vignette. 0.0 = off (byte-identical). Per-run:
    # --sky-darken. Range 0..1.
    sky_darken: float = 0.0
    # Clarity (midtone local-contrast) depth for ``photos develop``: a broad,
    # resolution-independent contrast lift on the luma channel, weighted to the
    # midtones so highlights/shadows are protected, applied before sharpen. 0.0 =
    # off (the shipped, byte-identical default); 1.0 = full boost. NEGATIVE softens
    # instead -- the same pass subtracted, flattening texture (skin, portraits) --
    # so -1.0 = full soften. Per-run override: --clarity. Range -1..1.
    clarity: float = 0.0
    # Luminance film-grain amount for ``photos develop``: a deterministic (fixed-
    # seed) monochrome grain field added last, on top of the vignette, weighted to
    # the midtones. The ADD counterpart to denoise -- keeps film-emulation LUTs from
    # rendering sterile-clean. 0.0 = off (the shipped, byte-identical default); 1.0 =
    # full grain. Per-run override: --grain. Range 0..1.
    grain: float = 0.0
    # Grain PARTICLE size for ``photos develop``: how coarse each speck is, the other half
    # of a film stock -- ``grain`` is how much silver, this is how big a clump. A RATIO on
    # the field's reference grid: above 1.0 is a chunkier, pushed-Tri-X speck, below it a
    # finer 100-ISO one. 1.0 = the shipped grid (byte-identical); inert while grain is 0.
    # Per-run: --grain-size. Range 0.25..4.
    grain_size: float = 1.0
    # Halation (the film highlight bleed) for ``photos develop``: the frame's own
    # brightest tones scatter a soft, RED-BIASED glow into their surroundings, over
    # the vignette and under the grain. The seed luma is MEASURED per frame, never an
    # absolute constant, so it cannot go inert on the sub-clipping tone a real grade
    # produces. 0.0 = off (byte-identical). Per-run: --halation. Range 0..1.
    halation: float = 0.0
    # Halation glow COLOUR for ``photos develop``: three (r,g,b) weights on that bleed.
    # Empty list = OFF = the shipped ``HALATION_RED_BIAS`` film physics, so the default
    # renders byte-identically; a cyan or magenta tint buys the neon bloom red cannot
    # express. Inert while halation is 0. Per-run: --halation-tint. Range 0..1.
    halation_tint: list[float] = Field(default_factory=list)
    # Light leak for ``photos develop``: a warm fog screened out of ONE FIXED corner, on
    # the halation bleed and under the fringe -- stray light that got past the camera back
    # and exposed the emulsion. No rng: one corner, one hue, one ramp shape, sized as a
    # fraction of the frame DIAGONAL so one value is one look at every export size.
    # 0.0 = off (byte-identical). Per-run: --light-leak. Range 0..1.
    light_leak: float = 0.0
    # Lateral chromatic aberration for ``photos develop``: the lens fringe -- red thrown
    # radially OUT from the frame centre, blue pulled IN, green fixed -- laid over the
    # halation and under the grain. Zero on the optical axis and growing toward the
    # corners, sized as a fraction of the HALF-DIAGONAL so one value is one look at every
    # export size. 0.0 = off (byte-identical). Per-run: --chromatic-ab. Range 0..1.
    chromatic_ab: float = 0.0
    # Holographic sheen for ``photos develop``: an iridescent tint whose HUE rotates with
    # TONE, so shadows, midtones and highlights each take a different cast and the print
    # shimmers rather than shifting as a whole. Keyed to the frame's OWN metered span, so it
    # cannot go inert on a develop peaking near 0.66 luma; low amplitude by design (a sheen,
    # not a duotone), over the fringe and under the lines. 0.0 = off. --holo-sheen. 0..1.
    holo_sheen: float = 0.0
    # CRT scanlines for ``photos develop``: a row-periodic darkening on a SOFT sine
    # profile, laid over the halation and under the grain. The line SPACING scales
    # with the export (anchored to SCANLINE_REFERENCE_DIM), so one value is one line
    # density at any size. 0.0 = off (byte-identical). Per-run: --scanline. Range 0..1.
    scanline: float = 0.0
    # Orton glow for ``photos develop``: a wide, soft copy of the frame SCREENED back
    # over the sharp original -- the classic dreamy diffusion, laid after the dehaze and
    # before the clarity so the detail is resolved back out of the bloom. The tone it
    # starts at is MEASURED per frame, never absolute, so it cannot go inert on the
    # sub-clipping tone a real grade produces. 0.0 = off (byte-identical). Per-run:
    # --glow. Range 0..1.
    glow: float = 0.0
    # Glitch for ``photos develop``: horizontal scan bands displaced sideways, each torn band
    # splitting its channels — a fault in the SIGNAL, so it lands over the scanlines and under
    # the grain. Seeded (never random); band grid and tear are both frame-relative, so one value
    # is one look at any size. 0.0 = off (byte-identical). Per-run: --glitch. Range 0..1.
    glitch: float = 0.0
    # Operator sharpen scale for ``photos develop``: a MULTIPLIER on the measured
    # unsharp amount (the detail ramp still decides which frames sharpen at all).
    # 1.0 = the measured ramp untouched (the shipped, bit-identical default); 0.0 =
    # sharpening off; 2.0 = double, clamped at the halo ceiling. Per-run override:
    # --sharpen. Range 0..2.
    sharpen: float = 1.0
    # Vibrance for ``photos develop``: a low-saturation-weighted chroma boost — muted
    # pixels take the full push, already-rich ones barely move (protects skies/foliage,
    # finer than vivid's scene-level gate). 0.0 = off (the byte-identical default). It
    # layers onto every styled variant's grade (natural included) but never the LUT
    # looks. Per-run override: --vibrance. Range 0..2.
    vibrance: float = 0.0
    # Per-hue-band HSL controls for ``photos develop`` over fixed anchor hues
    # (red 0°, yellow 60°, green 120°, cyan 180°, blue 240°, magenta 300°): an
    # empty list is OFF (the byte-identical default), else six values — or seven,
    # which inserts ORANGE 30° in second place for skin/golden hour.
    # hue_shift is per-band rotation in degrees (0 = neutral); hue_sat/hue_lum are
    # per-band multipliers (1.0 = neutral). They layer onto every styled variant's
    # grade (natural included) but never the LUT looks. Per-run override:
    # --hue-shift / --hue-sat / --hue-lum.
    hue_shift: list[float] = Field(default_factory=list)
    hue_sat: list[float] = Field(default_factory=list)
    hue_lum: list[float] = Field(default_factory=list)
    # Split-toning for ``photos develop``: three independent signed (r,g,b) tints,
    # one per tonal range — splittone_shadow rides the shadows, splittone_midtone
    # the midtones, splittone_highlight the highlights. Empty list = OFF
    # (byte-identical default), else exactly three values. They layer onto every
    # styled variant (natural included) but never the LUT looks. Per-run override:
    # --split-tone-shadow / --split-tone-midtone / --split-tone-highlight.
    splittone_shadow: list[float] = Field(default_factory=list)
    splittone_midtone: list[float] = Field(default_factory=list)
    splittone_highlight: list[float] = Field(default_factory=list)
    # Operator tone curves for ``photos develop`` — each empty (OFF,
    # byte-identical default) or a FLAT list of alternating x,y control
    # points (even length >= 4; x strictly ascending in [0, 1]). curve_l is the
    # MASTER shape, run on R/G/B alike so it carries no cast; an ADDITIONAL
    # cross-process curve per R/G/B then rides on top of it. Per-run override:
    # --curve-l / --curve-r / --curve-g / --curve-b.
    curve_l: list[float] = Field(default_factory=list)
    curve_r: list[float] = Field(default_factory=list)
    curve_g: list[float] = Field(default_factory=list)
    curve_b: list[float] = Field(default_factory=list)
    # Monochrome controls for ``photos develop``, both empty = OFF (byte-identical
    # default). mono_mix is three custom r,g,b luma weights (a red-weighted mix is
    # the red filter that darkens a blue sky); duotone is the (shadow r,g,b then
    # highlight r,g,b) ends of a luma ramp in [0, 1] — sepia/selenium/cyanotype.
    # Both layer onto every styled variant but never the LUT looks. Per-run
    # override: --bw-mix / --duotone. duotone_strength dials that ramp back toward
    # the plain monochrome: 1.0 = the full ramp (the shipped, byte-identical
    # default), 0.0 = no tone. Per-run override: --duotone-strength. Range 0..1.
    # duotone_mid is empty (a straight two-ender) or a third r,g,b anchor at mid
    # luma — a tritone. Per-run override: --duotone-mid. Range 0..1.
    mono_mix: list[float] = Field(default_factory=list)
    duotone: list[float] = Field(default_factory=list)
    duotone_strength: float = 1.0
    duotone_mid: list[float] = Field(default_factory=list)
    # Reference-image colour match for ``photos develop``: a hero frame every
    # styled variant is graded onto (Reinhard Lab mean/std transfer), never the
    # LUT looks. "" = off (the shipped, byte-identical default). Deliberately NOT
    # existence-checked here — a config may legitimately name a frame produced
    # later, and no validator may import the decoder (see LOOK_KNOBS above); the
    # CLI door decodes and measures it, failing there and naming the path.
    # Per-run override: --reference.
    reference: str = ""
    # Creative look baked into `photos ingest` gallery JPEGs. "" = the neutral
    # corrective base grade (byte-identical to no style). Per-run: --style.
    ingest_style: str = ""
    # Batch lighting-cluster distance override; None → SPAN_SIMILARITY_THRESHOLD.
    cluster_variance_threshold: Optional[float] = None
    # Presentational border/frame for ``photos develop`` exports, applied at the
    # CLI export step only (never the grade or the contact sheet). "none" (the
    # default) leaves the exported bytes byte-identical. Per-run: --border.
    border: str = "none"
    # Master container for ``photos develop``: "jpeg" (the default) is the 8-bit
    # sRGB gallery master, byte-identical; "tiff16" renders the whole chain at 16
    # bits into a 48-bit sRGB TIFF — the archival master. Per-run: --format.
    develop_format: str = "jpeg"
    # Per-frame grade record for ``photos develop``: <stem>.grade.json beside each
    # exported variant, holding the exact params/knobs/LUT and decode provenance
    # that produced it. False (the default) writes nothing and leaves the export
    # byte-identical. Per-run: --grade-sidecar.
    grade_sidecar: bool = False
    # Exposure-bracket fusion for ``photos develop``: merge each bracket the EXIF rule in
    # ``photos/bracket.py`` finds into ONE frame ahead of the grade, so a scene wider than one
    # exposure develops from all of it. False (the default) is the shipped path — every photo
    # develops singly, byte-identical. Pydantic type-gates it at load, so it needs no validator
    # here; the semantic gate (too few photos for a bracket to exist) sits at the CLI door,
    # where importing MIN_BRACKET_FRAMES is legal and this file's import is not. Per-run: --hdr.
    develop_hdr: bool = False
    # Social target aspect for the RESIZED upload copy: "" (the default) ships the
    # source's own aspect, byte-identical to before; "square"/"4:5"/"9:16" trim the
    # largest CENTRED rectangle of that ratio — after EXIF orientation, before the
    # width scale — so the composition is ours instead of a platform auto-crop.
    # One output per photo, never a variant per platform (the publish path below
    # the resizer is one resized file per selected photo). Per-run: --social-aspect.
    social_aspect: str = ""
    # Named knob combinations for ``photos develop --look NAME``: look name → the
    # develop FLAG names it sets ({"filmish": {"styles": "moody", "vignette": 0.35}}).
    # Recall replaces retyping eight flags exactly. Precedence: an explicitly-passed
    # flag beats the look, the look beats the bare photos.<knob> default. Empty (the
    # default) is inert — the shipped path, byte-identical. One key is not a flag:
    # ``layers`` is a list of masked colour recipes ({"mask": "sky", "hue_sat": [...]}),
    # each riding only where its named mask is non-zero — see ``check_layers``.
    looks: dict[str, dict] = Field(default_factory=dict)

    @field_validator("looks")
    @classmethod
    def _look_knobs_are_known_and_in_range(cls, v: dict[str, dict]) -> dict[str, dict]:
        # Load-time and LOUD, naming the look and the key: a look that only fails
        # once a render reaches the bad knob costs a batch of decodes to find.
        # Pure-local check by design — see LOOK_KNOBS on why nothing may import
        # gomoveshift.pipeline.* from here.
        for look, knobs in v.items():
            for key, value in knobs.items():
                if key == "layers":  # a list of masked colour recipes, not a knob
                    check_layers(f"look '{look}'", value)
                    continue
                if key in LOOK_RUN_KNOBS:  # a real flag, but not one a look may set
                    flag = "--" + key.replace("_", "-")
                    raise ValueError(
                        f"look '{look}': '{key}' is the RUN's choice, not a look's — "
                        f"pass {flag} on the command line."
                        + (" A look cannot bundle a LUT." if key == "lut" else "")
                    )
                if key not in LOOK_KNOBS:
                    known = ", ".join(sorted(LOOK_KNOBS))
                    raise ValueError(f"look '{look}': unknown knob '{key}' (valid: {known})")
                lo, hi = LOOK_RANGES.get(key, (None, None))
                if lo is not None and not (isinstance(value, (int, float)) and lo <= value <= hi):
                    raise ValueError(
                        f"look '{look}': {key}={value!r} must be a number in [{lo:g}, {hi:g}]"
                    )
            # After the unknown-knob pass, so a typo reports as a typo.
            check_look_lists(f"look '{look}'", knobs)
        return v

    @field_validator("sharpen")
    @classmethod
    def _sharpen_scale_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("must be in [0, 2] (1.0 = the measured amount, unscaled)")
        return v

    @field_validator("dehaze")
    @classmethod
    def _dehaze_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("must be in [0, 1] (0.0 = off, byte-identical)")
        return v

    @field_validator("sky_darken")
    @classmethod
    def _sky_darken_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("must be in [0, 1] (0.0 = off, byte-identical)")
        return v

    @field_validator("glow")
    @classmethod
    def _glow_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("must be in [0, 1] (0.0 = off, byte-identical)")
        return v

    @field_validator("scanline")
    @classmethod
    def _scanline_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("must be in [0, 1] (0.0 = off, byte-identical)")
        return v

    @field_validator("glitch")
    @classmethod
    def _glitch_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("must be in [0, 1] (0.0 = off, byte-identical)")
        return v

    @field_validator("halation")
    @classmethod
    def _halation_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("must be in [0, 1] (0.0 = off, byte-identical)")
        return v

    @field_validator("light_leak")
    @classmethod
    def _light_leak_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("must be in [0, 1] (0.0 = off, byte-identical)")
        return v

    @field_validator("chromatic_ab")
    @classmethod
    def _chromatic_ab_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("must be in [0, 1] (0.0 = off, byte-identical)")
        return v

    @field_validator("holo_sheen")
    @classmethod
    def _holo_sheen_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("must be in [0, 1] (0.0 = off, byte-identical)")
        return v

    @field_validator("hue_shift", "hue_sat", "hue_lum")
    @classmethod
    def _hue_band_empty_or_six_or_seven(cls, v: list[float]) -> list[float]:
        if v and len(v) not in (6, 7):
            raise ValueError(
                "must be empty, length 6 (red/yellow/green/cyan/blue/magenta) "
                "or length 7 (orange 30° second)"
            )
        return v

    @field_validator(
        "splittone_shadow", "splittone_midtone", "splittone_highlight", "halation_tint"
    )
    @classmethod
    def _tint_empty_or_three(cls, v: list[float]) -> list[float]:
        if v and len(v) != 3:
            raise ValueError("must be empty or length 3 (signed r,g,b tint)")
        return v

    @field_validator("mono_mix")
    @classmethod
    def _mono_mix_empty_or_three(cls, v: list[float]) -> list[float]:
        if v and len(v) != 3:
            raise ValueError("must be empty or length 3 (r,g,b luma weights)")
        if v and sum(v) <= 0.0:
            raise ValueError("weights must sum to more than 0")
        return v

    @field_validator("duotone_mid")
    @classmethod
    def _duotone_mid_empty_or_three(cls, v: list[float]) -> list[float]:
        if v and (len(v) != 3 or any(not 0.0 <= x <= 1.0 for x in v)):
            raise ValueError("must be empty or length 3 (r,g,b in [0, 1]) — the tritone anchor")
        return v

    @field_validator("duotone")
    @classmethod
    def _duotone_empty_or_six(cls, v: list[float]) -> list[float]:
        if v and len(v) != 6:
            raise ValueError("must be empty or length 6 (shadow r,g,b then highlight r,g,b)")
        if any(not 0.0 <= x <= 1.0 for x in v):
            raise ValueError("values must each be in [0, 1]")
        return v

    @field_validator("curve_l", "curve_r", "curve_g", "curve_b")
    @classmethod
    def _curve_empty_or_flat_points(cls, v: list[float]) -> list[float]:
        if v and (len(v) % 2 or len(v) < 4):
            raise ValueError("must be an even-length flat list of >= 2 (x, y) control points")
        return v

    @field_validator("lut_over")
    @classmethod
    def _lut_over_is_a_known_style(cls, v: str) -> str:
        # Checked against the PHOTO_LUT_OVER_STYLES literal — importing
        # ``styles`` here would both drag cv2 into every config load and, worse,
        # re-enter the pipeline package mid-import (see the constant's note).
        if not v:
            return v
        if v not in PHOTO_LUT_OVER_STYLES:
            raise ValueError(f"must be one of: {', '.join(PHOTO_LUT_OVER_STYLES)}")
        return v

    @field_validator("social_aspect")
    @classmethod
    def _social_aspect_is_known(cls, v: str) -> str:
        # Checked against the PHOTO_SOCIAL_ASPECTS literal, mirroring lut_over —
        # the resizer still owns the ratios, but importing it here would run a
        # pipeline import inside config validation (see the constant's note).
        if not v:
            return v
        if v not in PHOTO_SOCIAL_ASPECTS:
            raise ValueError(
                f"must be one of: {', '.join(PHOTO_SOCIAL_ASPECTS)} "
                "(or '' for the source's own aspect)"
            )
        return v

    @field_validator("border")
    @classmethod
    def _border_is_a_known_style(cls, v: str) -> str:
        # Mirror develop.BORDER_STYLES as a literal — importing develop here would
        # drag cv2 into every config load; a test pins the two sets together.
        if v not in ("none", "white", "black", "filmstrip"):
            raise ValueError("must be one of: none, white, black, filmstrip")
        return v

    @field_validator("develop_format")
    @classmethod
    def _develop_format_is_known(cls, v: str) -> str:
        # Mirrors develop.EXPORT_FORMATS as a literal, for the reason above.
        if v not in PHOTO_EXPORT_FORMATS:
            raise ValueError(f"must be one of: {', '.join(PHOTO_EXPORT_FORMATS)}")
        return v

    def resolve_site_root(self) -> Path:
        """Hugo site checkout the journey photos are read from.

        There is no default: an unset ``photos.site_root`` raises
        :class:`ConfigError` naming the key to set.
        """
        if not self.site_root:
            raise ConfigError(
                "photos.site_root is not configured; set photos.site_root in "
                "config.yaml to the Hugo site checkout the journey photos live in"
            )
        return Path(self.site_root).expanduser()

    def resolve_output_dir(self) -> Path:
        """Explicit config value wins; else ``GMS_PHOTO_OUTPUT_DIR``."""
        from .paths import gms_photo_output_dir

        return Path(self.output_dir).expanduser() if self.output_dir else gms_photo_output_dir()

    def resolve_state_path(self) -> Path:
        """Posted-photo ledger; rides ``output_dir`` unless set outright."""
        if self.state_path:
            return Path(self.state_path).expanduser()
        return self.resolve_output_dir() / PHOTO_STATE_FILENAME

    def resolve_resized_base(self) -> Path:
        """Base dir the resized upload copies are written under."""
        if self.resized_base:
            return Path(self.resized_base).expanduser()
        return self.resolve_output_dir() / "resized"

    def resolve_contact_sheet_root(self) -> Path:
        """Where every ``--contact-sheet`` lands (never beside the masters)."""
        if self.contact_sheet_root:
            return Path(self.contact_sheet_root).expanduser()
        return gms_data_home() / "contact-sheets"


class GPSQualityConfig(BaseModel):
    """Thresholds for ``gps_quality.filter_high_quality_points``.

    Tuned for highway-style driving content where we want to reject
    noisy fixes and stationary jitter before centroid + geocoding.
    Values are defaults; override per-project via ``config.yaml``.
    """

    min_gap_sec: float = 1.0


class PlaylistRule(BaseModel):
    """Single rule for ``YouTubePlaylistsConfig.rules``.

    ``match`` is a dict of derived-metadata keys → expected lowercased
    values. ``playlist_name`` is the YouTube playlist title to assign
    when the rule matches. Match-keys are intentionally string→string
    so rules read naturally as ``{ region: "ireland" }``.
    """

    match: dict[str, str] = Field(default_factory=dict)
    playlist_name: str


class YouTubePlaylistsConfig(BaseModel):
    """Playlist auto-assignment config.

    ``rules`` is evaluated in order; the first match wins. ``default``
    makes resolution total — every uploaded video receives a playlist
    assignment (``playlistItems.insert`` is universal in the
    uploader).
    """

    rules: list[PlaylistRule] = Field(default_factory=list)
    default: str = "Unsorted (fallback)"


class PlaylistReconcilerConfig(BaseModel):
    """Daily YouTube playlist reconciler: insert-only,
    quota-capped desired-membership diff. See
    ``gomoveshift/pipeline/playlist_reconciler.py`` and
    ``docs/pipeline/76-seo-contracts.md``.
    """

    max_inserts_per_run: int = 40
    # --prune also removes private/unlisted videos' entries (drained last; a
    # scheduled upload with a future publishAt is always exempt). CLI
    # --include-nonpublic/--no-include-nonpublic overrides per run.
    prune_include_nonpublic: bool = True
    newest_first: bool = True  # insert order when the plan exceeds the cap
    # Bounds ONLY orphan-short backfill; real_lf/stream/
    # highlight routing ignores it and is always considered.
    backfill_since: str = "2026-01-01"
    # Optional path to a manual videoId -> {region, state} JSON, consulted
    # only when the title yields no region signal. "" = none configured.
    overrides_path: str = ""


class EndscreenConfig(BaseModel):
    """End-screen operator-sidecar config.

    There is NO API path for programmatic end-screen injection (Data
    API v3, Partner API, and Studio API all lack the resource). The uploader
    therefore ships a doc-only checklist line per upload pointing the
    operator at Studio Editor.

    ``reference_video_id`` is the YouTube video id whose end-screen
    layout the operator wants to copy via Studio's "Import from video"
    action. When unset, the checklist line falls back to "Apply
    template" (operator picks a built-in layout). Most users will leave
    this blank until they have authored one canonical end screen they
    can point at.
    """

    reference_video_id: Optional[str] = None


def resolve_lf_mode(drive) -> str:
    """The single resolved LF selection mode for ``drive`` (never ``None``).

    ``DriveConfig`` resolves it at construction; this also covers config
    stand-ins that carry only the fields (unset mode + both part bounds set
    -> ``full_drive``). Every reader goes through here -- no reader keeps its
    own ``"best_moments"`` fallback.
    """
    mode = getattr(drive, "lf_selection_mode", None)
    if mode:
        return mode
    both = (
        getattr(drive, "full_drive_part_min_sec", None) is not None
        and getattr(drive, "full_drive_part_max_sec", None) is not None
    )
    return "full_drive" if both else "best_moments"


def lf_policy_snapshot(drive) -> dict:
    """Mode + part bounds, the shape stamped beside a saved plan and compared on resume."""
    return {
        "mode": resolve_lf_mode(drive),
        "part_min_sec": getattr(drive, "full_drive_part_min_sec", None),
        "part_max_sec": getattr(drive, "full_drive_part_max_sec", None),
    }


def lf_policy_banner(drive) -> str:
    """One launch-banner line: resolved mode, its source and the part bounds."""
    snap = lf_policy_snapshot(drive)
    lo, hi = snap["part_min_sec"], snap["part_max_sec"]
    bounds = f"part_min={lo}s part_max={hi}s" if lo is not None else "part bounds unset"
    return (
        f"LF policy: mode={snap['mode']} "
        f"(source={getattr(drive, 'lf_mode_source', 'default')}) {bounds}"
    )


def _is_valid_hhmm(value: str) -> bool:
    """True when *value* is a well-formed ``HH:MM`` 24h time string."""
    try:
        hh, mm = value.split(":", 1)
        return 0 <= int(hh) <= 23 and 0 <= int(mm) <= 59
    except (ValueError, AttributeError):
        return False


class PublishSlot(BaseModel):
    """The designed long-form publish time for one LF day.

    ``--lf-day tue`` alone determines the long-form publish time via the
    matching slot. Shorts times are NOT a property of the LF day — they are
    decided per region by Waypost (the cadence's ``shorts_times``), never by GMS
    config. A prior ``shorts_times`` field on this model duplicated that with
    contradictory numbers and was removed.
    """

    lf_time: str = Field(description="Long-form publish time for this day (EST, HH:MM 24h).")

    @field_validator("lf_time")
    @classmethod
    def _check_lf_time(cls, v):
        if not _is_valid_hhmm(v):
            raise ValueError(f"Invalid HH:MM time: {v!r}")
        return v


def _shorts_floor() -> int:
    """Waypost's shorts-per-LF floor, imported lazily so ``import gomoveshift.config``
    does not pay for ``seo_manager.analytics`` package init."""
    from seo_manager.analytics.volume_policy import SHORTS_FLOOR

    return SHORTS_FLOOR


class DriveConfig(BaseModel):
    """Configuration for Drive Pipeline (multi-file drive processing)."""

    min_segment_duration: float = 90.0  # Minimum segment duration in regular LFs (seconds)
    max_segment_duration: float = 180.0  # Cap individual segments; 0 disables cap
    crossfade_duration: float = 2.0  # Crossfade between non-contiguous segments (seconds)
    montage_peak_count: int = 6  # Number of peaks in the montage
    min_lf_duration: float = float(LF_MIN_DURATION)  # From video_types (canonical)
    # max_lf_duration is the operational default for LF assembly; the hard
    # ceiling for validation remains LF_MAX_DURATION in video_types.py.
    # Lowered from LF_MAX_DURATION (1800) to 720 to produce more, shorter LFs.
    max_lf_duration: float = 720.0
    # Each regular LF must yield this many shorts; the floor is Waypost's policy rail.
    min_shorts_per_lf: int = Field(default_factory=_shorts_floor)
    shorts_target_count: int = (
        9  # Legacy cap — single-video assembler path only (longform_assembler)
    )
    shorts_min_gap_sec: float = 60.0  # Minimum seconds between shorts
    # Short-clip duration bounds — defaults mirror
    # short_planner.DEFAULT_{MIN,MAX,TARGET}_DURATION_SEC. shorts_max_duration_sec
    # may never exceed SHORTS_MAX_DURATION (video_types.py), the hard YouTube
    # Shorts ceiling; validated in model_post_init below.
    shorts_min_duration_sec: float = 15.0
    shorts_max_duration_sec: float = 55.0
    shorts_target_duration_sec: float = 38.0
    shorts_geo_dedup_meters: float = (
        500.0  # Reject shorts within this many meters of a selected short
    )
    score_threshold_percentile: float = 15.0  # Top 85% — adaptive search starts
    # here; a lower floor keeps more footage so a long drive yields more LFs.
    # A higher percentile keeps too little of the day in practice; p15 is the
    # sweet spot, since far below it the threshold drops so low the planner
    # merges the whole day into a few oversized blobs.
    scoring_weights: dict = Field(
        default_factory=lambda: {
            # Brightness: per-second brightness signal added — funded
            # by `weather: 0.10 → 0.07` (semantic overlap; brightness is
            # the per-second version of the same lighting phenomenon) +
            # `motion_energy: 0.20 → 0.16` (trims its dominance, which
            # otherwise rewards motion under bad light).
            #
            # The `weather` key is GONE (was 0.07) and its mass moved to
            # `brightness` (0.07 → 0.14). The classifier behind it was
            # demoted — frame brightness cannot tell rain
            # from a tunnel, or sun from snow glare — so every branch now
            # reports OVERCAST and the signal was a constant 0.6 for every
            # episode. A constant cannot rank anything; carrying it just
            # diluted the signals that do discriminate. Brightness is the
            # per-second measurement of the same phenomenon and keeps the
            # lighting mass working. A missing key scores as 0.0 in the
            # scorer, so old configs that still list `weather` keep loading
            # (their weights simply must still sum to 1.0).
            #
            # Sum = 1.00 (validator at validate_weights tolerates ±0.01).
            "motion_energy": 0.16,
            "speed_variance": 0.15,
            "color_vibrancy": 0.10,
            "scene_change": 0.10,
            "sky_ratio": 0.05,
            "golden_hour": 0.10,
            "scenic_score": 0.20,
            "brightness": 0.14,
        }
    )
    # Conservative absolute floor: an LF segment whose per-sample / average
    # motion energy is below this is truly-dead (frozen) footage and excluded.
    # Kept well below ``stop_detection.motion_threshold`` (0.008) so it only cuts
    # near-frozen frames — real stops are handled structurally by the per-day
    # stop mask on the drive path. 0 disables the gate entirely.
    min_motion_energy: float = 0.001
    adaptive_threshold_enabled: bool = True  # Per-day adaptive threshold search
    adaptive_percentile_floor: int = 5  # Lowest percentile to try (below median)
    adaptive_min_segment_floor_s: int = 15  # Shortest segment duration floor (seconds)
    # Per-day STOP mask (a short must never ship parked footage). When on, a per-second veto built from GPS speed (fallback:
    # optical motion) structurally excludes stopped seconds from the shorts and
    # LF planners. Thresholds come from ``stop_detection.*``. Default on.
    stop_mask_enabled: bool = True
    # LFs are single-day by design. The non-day-first and per_day flags
    # (and their phase_b / phase_c / phase_cd_per_day implementations) were
    # removed.
    # Operator opt-outs for the measured LF sharpen / denoise (--no-sharpen /
    # --no-denoise). The grade records them on each CorrectionParams, so the
    # color-analysis cache (and a resume) carries the choice.
    no_sharpen: bool = False
    no_denoise: bool = False
    # Opt-in picture steps on the day master (--stabilize / --deflicker); each is
    # gated on measured luma jitter (pipeline/stabilize.py).
    stabilize: bool = False
    deflicker: bool = False
    # Picture steps dropped by ``drive --fast`` this run (shown in the preflight plan).
    fast_skipped: list[str] = Field(default_factory=list)
    # Route Story trail film beside each LF day (--trail / --no-trail). None is
    # auto: on when the day has a GPX and a prepared imagery bundle. The profile
    # (--trail-profile) supplies the imagery rasters and credits.
    trail: Optional[bool] = None
    trail_profile: str = ""
    preview: bool = False  # Encode 90s preview clips from middle of each episode, then exit
    no_review: bool = False  # Skip human review gates (automated checks still run)
    skip_validation: bool = False  # Skip terminal DoD validator gate (emergency only)
    skip_audio: bool = False  # Strip all audio from output (-an); user adds music externally
    skip_stop_verify: bool = False  # Bypass the post-stop-removal "stops actually cut" gate
    best_moments_mode: bool = False  # Filter LF content to top-scoring segments only
    best_moments_percentile: float = 60.0  # Keep top 40% of segments by composite_score
    best_moments_target_min: float = 900.0  # 15 min — minimum LF duration target
    best_moments_target_max: float = 1500.0  # 25 min — maximum LF duration target
    best_moments_min_keep_ratio: float = (
        0.5  # Never discard more than this fraction of qualifying footage
    )
    # Absolute composite-score floor (opt-in). When enabled, best_moments
    # selection keeps every segment whose composite_score clears a FIXED bar
    # instead of a relative percentile, so coverage scales with footage
    # quality: a scenic drive yields many LFs, a sparse one floors to a single
    # solid LF (never padded to a duration target). Calibrated for visual-only
    # (GPS-less) scoring — when GPS is present the composite distribution shifts
    # upward (GPS signal mass is live, not redistributed), so the absolute path
    # falls back to the proven percentile bar until a GPS floor is calibrated.
    best_moments_absolute_floor_enabled: bool = False  # Opt-in: absolute floor vs percentile
    best_moments_absolute_floor: float = 0.26  # GPS-less floor; ~p75 of visual-only composites
    brand_logo_path: str = ""  # Brand logo PNG for montage overlay; "" = no logo overlay
    # The env override wins over the per-user cache default, so containers
    # don't need a config.yaml change. This is the SINGLE geocode-cache namespace —
    # ``resolved_geocode_cache_dir`` routes Phase A, Phase BC, and Phase D
    # all through it so one coordinate resolves identically across a run.
    # Default from ``gomoveshift.paths.gms_geocoding_cache_dir`` (env
    # ``GMS_GEOCODING_CACHE_DIR``, else ``<cache home>/geocoding``).
    geocoding_cache_dir: str = Field(default_factory=lambda: str(gms_geocoding_cache_dir()))
    # Optional license-plate ONNX model for redactcam.
    # Plate detection is OPT-IN via a local ONNX model (faces have their own
    # model below). Empty (default) → plates disabled with a WARN (never
    # git-vendor a multi-MB model; download/pin it locally and point this at
    # it). ``~`` is expanded by the detector via ``Path(...).expanduser()``.
    plate_model_path: str = ""
    # When ``plate_model_path`` is empty, the detector fetches this YOLOv11 ONNX
    # ONCE (SHA256-verified) into ``models/`` under the cache home and reuses it.
    # GET-only, no data egress — the sole sanctioned widening of the network
    # allowlist. Point ``plate_model_url`` at an internal mirror
    # to avoid the public host; clear it to stay faces-only.
    plate_model_url: str = (
        "https://huggingface.co/morsetechlab/yolov11-license-plate-detection"
        "/resolve/main/license-plate-finetune-v1m.onnx"
    )
    plate_model_sha256: str = "53e6a1c535141a483e2fe5b802d4f8179e820561d5ed892b0bdeb5fbd200f010"  # pragma: allowlist secret (public model checksum, not a credential)
    # Privacy-blur FACE model: a single-class YOLO face detector that
    # shares the plate model's decode/tiling/GPU path. Far more accurate than the
    # bundled CenterFace, which scored foliage ABOVE real faces —
    # missing people while blurring leaves (see docs/pipeline/62-privacy-detection-the-package.md). Same
    # fetch-once/SHA256 contract as the plate model; ``face_model_path`` (operator
    # override) wins. If it can't be resolved the detector falls back to CenterFace
    # so faces never silently ship unblurred — clear ``face_model_url`` to force that.
    face_model_path: str = ""
    face_model_url: str = (
        "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov11n-face.onnx"
    )
    face_model_sha256: str = "61d8cf127ff377939cf009565457aaddf216f4d9fdd9ac24f5aab3cb10ee67b5"  # pragma: allowlist secret (public model checksum, not a credential)
    # Privacy blur: detect faces ∪ plates on the stops-cut file,
    # composite a blur over them in the stage-9 master. The blur_gate hard-fails
    # any blur-ENABLED run whose master isn't provably blurred (see
    # ``pipeline/blur_gate.py``); ``--skip-blur`` bypasses per-run. Default ON:
    # privacy protection is the safe default — an operator opts OUT per run with
    # ``--skip-blur``.
    blur_enabled: bool = True
    blur_sample_fps: float = (
        8.0  # detection cadence; gap ≈0.125 s ≤ ~0.13 s flow horizon (full coverage)
    )
    blur_conf: float = 0.4  # PLATE detector confidence (squarish road-sign FPs need ≥0.4)
    # Face detector confidence, INDEPENDENT of the plate conf: a small clear
    # pedestrian face can score well below a plate's usable threshold, so face recall is pushed without
    # lowering the plate conf and resurfacing road-sign FPs. At this conf the face
    # model DOES fire on foliage/walls/shadows (0.5-0.7) — confidence can't separate
    # them from a real small face — so blur_face_confirm_min_detections (below) is
    # what keeps precision; never lower this conf without that confirmation on.
    blur_face_conf: float = 0.15
    # Faces get their OWN, denser cadence than plates (a small low-texture face can't
    # be flow-tracked between sparse samples — its box freezes while it walks — so
    # snap it more often). Faces are cheap relative to the plate-detector bottleneck.
    blur_face_sample_fps: float = 16.0
    # Temporal confirmation: a face track is blurred only once it has this many
    # associated detections. A real face re-detects across samples as it persists
    # (kept, retroactively); a foliage/shadow/wall FP fires once then coasts (1
    # detection → dropped). This is the precision mechanism that lets blur_face_conf
    # run at 0.15. 1 disables (blur every face track). 2 drops single-frame FP flicker.
    blur_face_confirm_min_detections: int = 2
    # Backward pre-roll: cover up to this many frames BEFORE an object's first
    # detection by extrapolating its box backward along its early velocity — a
    # small/distant oncoming plate is readable, and a pedestrian present, for a few
    # frames before the detector first scores them (the residual the forward boost
    # can't reach). Only confirmed, moving tracks pre-roll and the box shrinks going
    # back (stops when it collapses), bounding the bare-road-blur risk. 0 disables.
    blur_pre_roll_frames: int = 5
    # Adaptive plate cadence: while a plate track's per-frame flow speed is ≥ this,
    # plates re-detect EVERY frame (not just blur_sample_fps), so a moving oncoming
    # plate — which between sparse samples drifts out of its lagging flow box and
    # becomes readable — gets its box snapped to ground truth each frame. Set low so
    # ANY moving plate (not just an extreme close pass) is covered; a static/parked
    # plate (speed ≈0) doesn't leak so doesn't need the boost. 0 disables. boost_max_run
    # keeps it on a few frames past the last fast reading; cost is bounded to the
    # frames a plate is actually on screen and moving.
    blur_plate_boost_speed_px: int = 8
    blur_plate_boost_max_run: int = 6
    # Flat privacy backstop on every box; the per-box MOTION margin that covers a
    # fast plate's flow lag lives in the tracker (`track_motion_*`), scaled by the
    # box's own speed — not its size, which would balloon a large static FP.
    blur_box_dilation_px: int = 8
    # Size-proportional face margin floor (this fraction × the box's larger side),
    # on top of the tracker's speed-scaled motion margin. The speed margin collapses
    # to the bare frontal-detector box when a face's flow speed drops to ≈0 (a
    # pedestrian exiting frame loses her flow features), exposing the turned
    # cheek/profile the tight box misses; the proportional floor keeps the blur on
    # the whole head. Proportional (not flat) so a tiny distant FP face isn't
    # ballooned — precision held. 0 = speed-only margin (the pre-fix behaviour).
    blur_face_margin_frac: float = 0.35
    # Face-specific motion-margin gain (overrides the shared track motion gain for
    # faces only). A fast face's optical-flow box lags its true position between
    # samples — the blur trails to the right of a fast leftward subject and collapses
    # as it exits frame — so faces get a larger speed-scaled margin than plates to
    # keep the blur over the leading edge and the trailing exit. Capped by the
    # tracker motion_max; plates keep the lower shared gain (their margin is already
    # speed-scaled + floored, and over-padding a plate would balloon structure FPs).
    blur_face_motion_gain: float = 1.6
    blur_strength: int = 12  # boxblur luma radius at 1080p; scales with width
    # Tiled inference: 4K frames split into ~blur_tile_px tiles so small/distant
    # faces and plates detect at native scale instead of vanishing in a
    # whole-frame downscale; 1080p mostly stays single-tile. Overlap keeps a
    # seam-straddling object whole in at least one tile.
    blur_tile_px: int = 1280
    blur_tile_overlap: float = 0.15
    # Plate aspect band: the YOLO plate model fires confidently on rectangular
    # road signs (confidence can't separate them) AND on long horizontal
    # structures (a far-bank shoreline or bridge railing boxes at a very high
    # ratio), so keep only plate boxes inside a width/height band.
    # Real EU plates box ≈2.4-3:1 (≤4.7:1 head-on); signs box
    # ≈1.4-1.65:1. 2.0 / 6.0 bracket the gap on both sides.
    blur_plate_min_aspect: float = 2.0
    blur_plate_max_aspect: float = 6.0
    # Plate max-area gate: the model also fires on big stone walls / verges / a
    # bridge abutment at a plausible 2.0-2.7:1 (the band can't see them), boxing
    # a large share of the frame, AND on large roadside place-name signs at a
    # plate-like aspect. A real plate is ≤0.4%
    # of a 4K frame, so drop any plate box over this fraction — 0.6% keeps real plates
    # (≤0.4%) while dropping such signs (≥0.84%). 0 disables; faces are never area-gated.
    blur_plate_max_area_frac: float = 0.006
    # Vehicle-derived CABIN blur (occupant privacy). A vehicle's driver/passengers
    # sit behind a moving windshield where per-face detection is unreliable (the
    # face model fires on the wall/roof/glass reflections, never the face) and a
    # plate-anchored box mis-locates the cabin as the car passes (the windshield
    # leans back off the front plate). So a COCO vehicle detector boxes the whole
    # car, and the blur covers the TOP ``blur_vehicle_cabin_frac`` of that box —
    # where occupants sit at EVERY angle (windshield head-on, side windows on the
    # pass). It follows the car via the vehicle track. 0 disables (occupants
    # unblurred). ``blur_vehicle_conf`` is the vehicle detector confidence; cars are
    # large/high-confidence so this is generous.
    blur_vehicle_cabin_frac: float = 0.6
    # Vehicle detector confidence. Set HIGH (0.6) on purpose: the COCO model
    # misclassifies rectangular roadside signs/boards as cars at ~0.35-0.55, while a
    # real oncoming car scores 0.63-0.95 — so 0.6 keeps cars
    # and drops sign false positives (signs must stay readable, like the plate band).
    blur_vehicle_conf: float = 0.6
    # Only blur vehicles worth it: tall enough that occupants are discernible
    # (``blur_vehicle_min_height_frac`` × frame height) and not glued to the bottom
    # strip (box top above ``1 - blur_vehicle_ignore_bottom_frac`` of the frame — the
    # ego car's own hood, which the COCO model fires on but carries no occupants). A
    # detected worth-blurring vehicle also arms the per-frame detection boost so its
    # cabin is snapped every frame of its pass (not flow-drifted between sparse samples).
    blur_vehicle_min_height_frac: float = 0.07
    blur_vehicle_ignore_bottom_frac: float = 0.12
    # Vehicle (car/bus/truck/motorcycle) detector for the cabin blur. Same
    # fetch-once/SHA256/cache contract as the face/plate models (a COCO YOLOv11
    # ONNX); ``vehicle_model_path`` (operator override) wins. An unresolvable model
    # disables the cabin blur (WARN) — never fatal. Clear ``vehicle_model_url`` to
    # disable the occupant cabin blur entirely.
    vehicle_model_path: str = ""
    vehicle_model_url: str = (
        "https://huggingface.co/aaurelions/yolo11n.onnx/resolve/main/yolo11n.onnx"
    )
    vehicle_model_sha256: str = "5ad2aa77f7f678c06880f024444cb02f5eb53f991b4a64c819805c6c701fc9b8"  # pragma: allowlist secret (public model checksum, not a credential)
    # Occupant-coverage gate (blur_gate.check_cabin_coverage): before the irreversible
    # encode, re-detect plate-corroborated vehicles and confirm the blur MASK covers
    # each one's cabin — fail the run rather than ship a driver exposed. Sampled at
    # this cadence (cost vs thoroughness); a cabin under this mask-coverage fraction is
    # a leak (0.5 absorbs the verifier's independent-detection box offset, see
    # blur_coverage). Bypassed by --skip-blur / blur_vehicle_cabin_frac=0.
    blur_cabin_gate_sample_fps: float = 4.0
    blur_cabin_gate_min_coverage: float = 0.5
    # WHOLE-PERSON blur (a person standing OUTSIDE the vehicle must not get only a
    # face blur — their body/pose is still identifying). The COCO detector
    # (person class 0 — same yolo11n model the vehicle detector uses) boxes the FULL
    # person and the mask blurs the whole box (not just the head). Default ON. An
    # unresolvable model degrades to WARN (faces still ship blurred), never fatal.
    blur_person_enabled: bool = True
    blur_person_conf: float = 0.35
    # Drop tiny distant pedestrians (no identifiable person below this frame-height
    # fraction) so a crowd of far dots isn't blurred; 0 keeps every person box.
    blur_person_min_height_frac: float = 0.05
    person_model_path: str = ""
    person_model_url: str = (
        "https://huggingface.co/aaurelions/yolo11n.onnx/resolve/main/yolo11n.onnx"
    )
    person_model_sha256: str = "5ad2aa77f7f678c06880f024444cb02f5eb53f991b4a64c819805c6c701fc9b8"  # pragma: allowlist secret (public model checksum, not a credential)
    skip_blur: bool = (
        False  # DIAGNOSTIC: bypass blur detection (config.drive mirror, like skip_audio)
    )
    # INSERTS (--insert-photos / --insert-clip): per-invocation mirrors set by the drive
    # handler (``inserts.apply_insert_flags``); never persisted to PipelineState.
    insert_photos: str = ""
    insert_clips: list[str] = Field(default_factory=list)  # raw "PATH@MM:SS" specs
    # ROAD-SIGN GPS VERIFICATION (--sign-verify). Master switch for the whole
    # feature: the Phase A per-day sign OCR pass AND the advisory per-LF
    # comparison of what the signs say against the LF's resolved location.
    # Default OFF for its first release for two reasons: it adds a second
    # decode pass over the day's footage to a run that already takes hours,
    # and the advisory warning should be observed before the cost is imposed on
    # every run. Detection/OCR confidence knobs are NOT duplicated here; they
    # live on SignDetectionConfig / OCRConfig (their single home).
    sign_verify: bool = False
    # Escalate a corroborated state/county conflict from a label-gate warning
    # to a pre-encode HARD FAIL. Opt-in: a false positive here aborts a
    # long run before encode, so the advisory mode comes first.
    sign_verify_strict: bool = False
    # Cost governor: hard cap on native-resolution frame grabs per day. Sign
    # text is only legible at full resolution, so each sample is expensive;
    # the cap bounds the pass's wall-clock regardless of how long the day is.
    sign_verify_max_samples_per_day: int = 120
    # Only MOVING seconds are sampled (ground speed, m/s). A parked second's
    # sign is not evidence about the route — it is whatever happens to be
    # framed outside the windscreen, and it would be sampled repeatedly.
    sign_verify_min_speed_mps: float = 5.0
    # A lone noisy hit never contradicts GPS: a conflict needs at least this
    # many INDEPENDENT hits asserting the same wrong county/state before it is
    # reported (or, under strict, raised). One misread sign is an OCR error;
    # several agreeing ones are evidence.
    sign_verify_min_corroborating_hits: int = 2
    # NVDEC hardware decode for the day-master encode. When True,
    # ``single_pass._run_encode_step`` adds ``-hwaccel cuda
    # -hwaccel_output_format cuda`` before the input and prepends
    # ``hwdownload,format=<nv12|p010le>`` to the filter chain so CPU filters
    # see normal frames after the GPU decode. The download format is chosen
    # by construction from the source's probed ``pix_fmt`` bit depth
    # (``single_pass._hwdownload_format_for_pix_fmt``): 10-bit Main 10 GoPro
    # sources → ``p010le``, 8-bit → ``nv12``. (The earlier fixed
    # ``nv12|p010le`` alternation aborted every 10-bit day master —
    # ffmpeg pins ``hwdownload`` to ``nv12`` and fails the 10-bit surface.)
    # This gives a clear wall-clock speedup on
    # 4K HEVC with an encode-noise-floor SSIM delta against the CPU-decode
    # baseline and no visible artifact change. NVDEC decode only — color/scale/denoise filters
    # still run on CPU because ``scale_npp``/``scale_cuda`` are not built
    # into this ffmpeg (no ``--enable-libnpp``). Default True;
    # set False in user config.yaml to revert.
    use_nvdec_decode: bool = True
    # Filename guard for YouTube upload. Also admits ``sl_NN.mp4`` (superlong
    # upload slot) and, for the Shorts publishing adapter,
    # ``short_NN.mp4``, the canonical short filename from ``day_pipeline.py``.
    # ``YouTubeUploader.upload_video`` only lets files through whose basename
    # matches this regex. Default blocks everything except the canonical LF
    # naming produced by ``day_pipeline.py`` (``lf_NN.mp4``) and the
    # canonical superlong-part naming produced by
    # ``day_pipeline._produce_superlong_parts`` (``sl_NN.mp4``) — defense in
    # depth against legacy "episode NN FINAL" drafts and any future code
    # path that accidentally bypasses LF/superlong/short assembly. Stays a SINGLE
    # config-driven pattern — never add a second field or a parallel
    # ``startswith`` check beside this one. Dev/testing may loosen this to
    # ``.*`` to disable the guard. The pattern is read lazily via
    # ``get_config()`` at call time so overrides take effect per-invocation.
    youtube_upload_filename_pattern: str = r"^(?:lf|sl|short)_\d{2}\.mp4$"
    # Upload body completeness. YouTube Data API v3
    # ``videos.insert`` accepts these fields on every upload; sending
    # explicit values avoids YouTube's own undocumented per-field
    # defaults drifting the body over time.
    # https://developers.google.com/youtube/v3/docs/videos/insert
    youtube_default_language: str = "en"  # snippet.defaultLanguage (BCP-47)
    youtube_license: str = "youtube"  # status.license: "youtube" | "creativeCommon"
    youtube_embeddable: bool = True  # status.embeddable
    # status.containsSyntheticMedia — per-surface flag (external semantics,
    # unverified against ours). None (default) omits the field entirely so
    # existing uploads are unaffected until this is deliberately set.
    youtube_contains_synthetic_media: Optional[bool] = None
    # Proactive YouTube Data API v3 quota accounting. Google's
    # published per-call costs and default daily project quota — NOT
    # ours, and they change — so they are configurable rather than
    # hardcoded in the upload path. ``_upload_lfs_to_youtube``
    # (external_systems_coordinator.py) sums the pending regular-LF +
    # superlong-part ``videos.insert`` + ``playlistItems.insert`` pairs
    # for the run BEFORE the first upload, so the operator learns the run
    # is projected over budget up front instead of discovering the same
    # ceiling via a mid-batch HTTP 403. A test double that leaves these
    # unset (e.g. ``MagicMock()``) skips the check rather than raising.
    youtube_quota_daily_budget: int = 10000
    youtube_video_insert_quota_cost: int = 1600
    youtube_playlist_insert_quota_cost: int = 50
    # ``videos.insert`` now has its OWN daily bucket (Google project
    # default 100/day), separate from the combined budget above (which
    # still governs ``playlistItems.insert``). ``publishing.youtube_adapter``
    # meters against this and DEFERS overflow to next UTC day rather than
    # failing — a deferred short's ``publishAt`` tolerates the delay.
    youtube_video_insert_daily_uploads: int = 100
    # WARN by default, refuse only when the operator opts in. The budget and
    # per-call costs above are Google's numbers, not ours: a project with a
    # raised quota makes the default ceiling wrong, and a batch that exceeds
    # it is still perfectly uploadable up to the ceiling — the reactive
    # ``YouTubeQuotaError`` pause plus ``upload_if_needed``'s skip-on-resume
    # already recover the remainder the next day. Refusing the whole batch
    # up front would remove that, so enforcement is opt-in.
    youtube_quota_enforce: bool = False
    # Montage + OST title coupling.
    # ``montage_max_duration_sec`` is the authoritative total montage duration
    # consumed by both ``montage_builder`` and ``ost_applier``.
    # ``montage_builder.MONTAGE_DURATION_SEC`` reads this value.
    # ``ost_title_crossfade_sec`` is the linear fade-out window at the tail
    # of the montage — the LOCATION title crossfades out over this duration
    # as the main video begins.
    montage_max_duration_sec: float = 15.0
    ost_title_crossfade_sec: float = 0.5
    # GPS-quality thresholds. Filter noisy GPS points before
    # centroid / geocode. See ``gps_quality.py``.
    gps_quality: GPSQualityConfig = Field(default_factory=GPSQualityConfig)
    # Configurable publish cadence. Defaults preserve the
    # historical hardwired behavior (Wed @ 11:00 EST). Operators may
    # override via ``~/.config/gomoveshift/config.yaml``:
    #
    #   drive:
    # (A config-file value for either is dropped at load: Waypost decides the days.)
    #
    # The per-run transport (``--lf-day`` / ``--lf-time``) is read by
    # ``pipeline/schedule_constants.long_form_days`` / ``long_form_time``, which answer
    # from Waypost's cadence when it is unset, and wired through
    # ``PipelineScheduler.from_config``. The module-level ``LONG_FORM_DAYS`` /
    # ``LONG_FORM_TIME`` constants come from the packaged brand seed at import time.
    # Waypost decides the long-form days and times (seo_slots.waypost_cadence). These two
    # fields are only the per-run ``--lf-day`` / ``--lf-time`` transport: empty / unset by
    # default, never read from a config file (the loader drops one, RETIRED_CONFIG_KEY), and
    # read through ``schedule_constants.long_form_days`` / ``long_form_time``.
    long_form_publish_days: list[int] = Field(default_factory=list)
    long_form_publish_time: Optional[str] = None
    # Per-LF-day designed long-form publish times (the cadence table). Keyed by 3-letter weekday. --lf-day <day>
    # derives lf_time from the matching slot; --lf-time overrides; days without
    # a slot fall back to long_form_publish_time. Shorts times are NOT here —
    # they are Waypost's per-region ``shorts_times``.
    publish_slots: dict[str, PublishSlot] = Field(
        default_factory=lambda: {
            "sun": PublishSlot(lf_time="14:00"),
            "tue": PublishSlot(lf_time="16:00"),
            "fri": PublishSlot(lf_time="15:00"),
        }
    )
    # Shorts publish times — list of HH:MM 24h strings, EST assumed.
    # Mirrors long_form_publish_time validation. Default is the afternoon
    # audience-online window (15:00 + 18:00 EST = 3-6 PM EDT peak, which is
    # also 8-11 PM Irish evening). ``--shorts-per-day N``
    # selects the first N of these windows, so this list is the single
    # source of truth; override per-cohort via config.yaml or per-invocation
    # via the drive CLI's --shorts-times flag.
    shorts_publish_times: list[str] = Field(default_factory=lambda: ["15:00", "18:00"])
    # Region-aware shorts default times — maps a drive's region to its
    # default shorts publish window, used (in place of the flat
    # ``shorts_publish_times``) when a run's ``--region`` matches a key and
    # the operator did NOT pass ``--shorts-times``. A region absent from the
    # map falls back to ``shorts_publish_times``. These are a default-UNDER-TEST spread chosen for
    # global-audience coverage + anti-cannibalization — Japan-prime 08:00 +
    # Irish-evening 12:00 EST for Ireland, US-afternoon 14:00 + 18:00 EST for
    # USA — NOT a v/day ranking winner. Operators still override per-run via
    # ``--shorts-times`` (which takes precedence).
    # Region → active journey slug for journal auto-publish. When a drive run's
    # region matches a key here and the operator did NOT pass an explicit
    # ``--auto-publish-journal SLUG``, the post-upload hook merges the run's
    # videos into ``<site>/content/journeys/<slug>.md``. EMPTY by default: an
    # entry here (or an explicit slug) wins outright. Absent both, the hook
    # falls back to resolving the journey by the run's recording date against
    # each journey's ``trip.start_date``..``end_date`` range (see
    # ``pipeline/site_sync/journey_by_date.py``) — two or more matching trips
    # REFUSE the merge (named in the error) rather than guess, so cross-trip
    # date overlap is detected, not assumed away. Only a zero-match publish
    # run still logs the loud WARNING that the site won't update. Set e.g.
    # {"exampleland": "exampleland-2025-09"} to pin a slug outright.
    active_journey_by_region: dict[str, str] = Field(default_factory=dict)
    # YouTube playlist auto-assignment. Resolver in
    # ``pipeline/playlist_resolver.py`` reads ``rules`` in order and falls
    # back to ``default``; uploader's ``upload_video_with_playlist`` then
    # get-or-creates the playlist and ``playlistItems.insert``s the video.
    youtube_playlists: YouTubePlaylistsConfig = Field(
        default_factory=lambda: YouTubePlaylistsConfig(
            rules=[
                # Format-aware routing: must
                # precede the region-only rules below — first match wins.
                PlaylistRule(
                    match={"region": "ireland", "format": "highlights"},
                    playlist_name="Ireland Highlights",
                ),
                PlaylistRule(
                    match={"region": "usa", "format": "highlights"},
                    playlist_name="USA Highlights",
                ),
                # Shorts: ``format: "shorts"`` is an
                # EXPLICIT value the shorts adapter stamps; without these
                # rules a short matches nothing and lands in the default.
                PlaylistRule(
                    match={"region": "ireland", "format": "shorts"},
                    playlist_name="Ireland Shorts",
                ),
                PlaylistRule(
                    match={"region": "usa", "format": "shorts"}, playlist_name="USA Shorts"
                ),
                PlaylistRule(match={"region": "ireland"}, playlist_name="Ireland Drives"),
                PlaylistRule(match={"region": "usa"}, playlist_name="USA Drives"),
            ],
            default="Unsorted (fallback)",
        )
    )
    # LF description footer — playlist link / cadence line / site URL,
    # rendered by ``pipeline/longform_metadata`` between the CTA block and
    # the hashtag divider. Keyed by the SAME derived region as playlist
    # auto-assignment (``pipeline/region_resolver.derive_region`` off the
    # canonical ResolvedLocation — never re-derived from raw GPX). ALL
    # defaults are EMPTY: an unconfigured run renders no footer and the
    # description is byte-identical to the pre-footer layout. Set e.g.
    # ``playlist_url_by_region: {ireland: "https://…/playlist?list=…"}``,
    # ``cadence_line_by_region: {ireland: "New Ireland drive every Sunday"}``,
    # ``site_url: "https://gomoveshift.com"``.
    playlist_url_by_region: dict[str, str] = Field(default_factory=dict)
    cadence_line_by_region: dict[str, str] = Field(default_factory=dict)
    site_url: str = ""
    # End-screen operator sidecar. NO API
    # path exists for end screens; this config feeds a doc-only checklist
    # line emitted at upload time. See ``pipeline/endscreen_sidecar.py``.
    endscreen: EndscreenConfig = Field(default_factory=EndscreenConfig)
    # Make non-GoPro source files a configurable contract rather than a regex bypass. ``ingest_mode``
    # picks between GoPro chapter/session grouping and a generic
    # "one-file-per-episode" path that accepts any video matching
    # ``video_extensions``. ``video_extensions`` is the single source of
    # truth for accepted extensions — read by both preflight and the
    # ingest scanner. Defaults preserve historical .mp4-first behaviour
    # while expanding to formats Insta360 / drone / phone exports use.
    ingest_mode: Literal["gopro", "generic"] = "gopro"
    video_extensions: list[str] = Field(
        default_factory=lambda: [".mp4", ".mov", ".mkv", ".m4v", ".mxf"]
    )
    # LF selection mode.
    # ``"best_moments"`` keeps the legacy sliding-window planner. ``"full_drive"``
    # produces exactly one LF spanning the full post-stop-removal timeline —
    # the Insta360 use case where the operator wants the whole drive, not
    # highlights. CLI override: ``--lf-mode {best_moments,full_drive}``.
    # ``full_drive_max_duration_sec`` caps the single LF (default 2h);
    # ``plan_regular_lfs(mode="full_drive")`` raises ``ValueError`` rather
    # than silently truncate when the source exceeds the ceiling.
    # ``None`` = "operator did not say": ``_resolve_lf_policy`` turns it into
    # ``full_drive`` when BOTH part bounds below are set (the bounds only mean
    # anything under full_drive; a config with bounds and no mode used to fall
    # silently back to 8-12 min best_moments), else ``best_moments``. After
    # construction the field is never ``None`` -- every reader goes through
    # ``resolve_lf_mode`` / this resolved value.
    lf_selection_mode: Optional[Literal["best_moments", "full_drive"]] = None
    # Provenance of the resolved mode: default | yaml | cli | resumed-state.
    lf_mode_source: str = "default"
    full_drive_max_duration_sec: int = 7200
    # Balanced long-form parts for full_drive (day-first): when BOTH are set a
    # day's kept timeline is split into ceil(kept/max) near-equal parts, each
    # >= min and (barring a one-part day of 91-119 min) <= max; the 480 s floor
    # and full_drive_max_duration_sec ceiling apply per part. ``None`` = one LF
    # per day, as before. See ``regular_lf_planner.split_ranges_balanced``.
    full_drive_part_min_sec: Optional[int] = None
    full_drive_part_max_sec: Optional[int] = None

    @model_validator(mode="after")
    def _check_full_drive_part_bounds(self) -> "DriveConfig":
        lo, hi = self.full_drive_part_min_sec, self.full_drive_part_max_sec
        if (lo is None) != (hi is None):
            raise ValueError(
                "full_drive_part_min_sec and full_drive_part_max_sec must be set "
                "together or both left unset"
            )
        if lo is not None and not 0 < lo <= hi:
            raise ValueError(
                f"full_drive_part: need 0 < full_drive_part_min_sec <= "
                f"full_drive_part_max_sec (got {lo}, {hi})"
            )
        return self

    @model_validator(mode="after")
    def _resolve_lf_policy(self) -> "DriveConfig":
        if self.lf_selection_mode is None:
            both = self.full_drive_part_min_sec is not None and (
                self.full_drive_part_max_sec is not None
            )
            self.lf_selection_mode = "full_drive" if both else "best_moments"
            self.lf_mode_source = "default"
            if both:
                # Compatibility MIGRATION: configs written before the mode was
                # inferred set only the part bounds (and silently got 8-12 min
                # best_moments). Remove once every config states its mode.
                logger.warning(
                    "LF-POLICY-MIGRATION: mode inferred full_drive from part bounds "
                    "(full_drive_part_min_sec=%s full_drive_part_max_sec=%s); set "
                    "lf_selection_mode explicitly to silence this",
                    self.full_drive_part_min_sec,
                    self.full_drive_part_max_sec,
                )
        elif self.lf_mode_source == "default":
            self.lf_mode_source = "yaml"
        self.warn_lf_policy_override()
        return self

    def set_lf_mode(self, mode: str, source: str) -> None:
        """Set the resolved mode from an explicit source (cli | resumed-state)."""
        self.lf_selection_mode = mode  # type: ignore[assignment]
        self.lf_mode_source = source
        self.warn_lf_policy_override()

    def warn_lf_policy_override(self) -> None:
        """Named warning when an operator-chosen best_moments shadows part bounds.

        Only reachable through an explicit choice (yaml/cli/resumed-state): an
        unset mode never resolves to best_moments while both bounds are set.
        """
        lo, hi = self.full_drive_part_min_sec, self.full_drive_part_max_sec
        if self.lf_selection_mode == "best_moments" and lo is not None and hi is not None:
            logger.warning(
                "LF-POLICY-OVERRIDE: operator explicitly chose lf_selection_mode=best_moments (source=%s), overriding "
                "full_drive_part_min_sec=%s full_drive_part_max_sec=%s -- the part bounds "
                "are IGNORED and LFs use min/max_lf_duration (%ss-%ss)",
                self.lf_mode_source,
                lo,
                hi,
                self.min_lf_duration,
                self.max_lf_duration,
            )

    # Superlong part planner (SEPARATE from full_drive above — RegularLF
    # machinery must never see these parts). Caps each part's KEPT duration;
    # grace widens it so a day just over the ceiling isn't force-split.
    # See ``pipeline/superlong_planner.plan_superlong_parts``.
    superlong_part_max_duration_sec: int = 14400
    superlong_split_grace: float = 0.08
    # Opt-in per-day sl_NN.mp4 production. CLI: --superlong (flag_registry.py).
    superlong_enabled: bool = False
    # Recurring LOCATION lower-thirds burned during the superlong OST pass
    # (ost_applier._prepare_lower_thirds): how long each label stays on
    # screen, and the min gap enforced so a dense town run doesn't strobe.
    superlong_lower_third_duration_sec: float = 8.0
    superlong_lower_third_min_gap_sec: float = 45.0
    # Opt-in GPS event captions on REGULAR long-form videos (``--gps-captions``):
    # "off" burns none (today's path), "events" burns one "Entering <town>" card
    # per supported town event of the LF's own chapters. Own timing knobs, so the
    # superlong values above never move.
    gps_captions: str = "off"
    regular_lf_lower_third_duration_sec: float = 8.0
    regular_lf_lower_third_min_gap_sec: float = 45.0
    # Superlong per-part publish DAY. Saturday is a placeholder default chosen because it
    # falls outside every currently-designed LF/shorts cadence (Ireland Sun,
    # USA Tue/Fri). Settable per run via ``--superlong-day`` (flag_registry.py,
    # sentinel — an absent flag preserves a resumed run's saved value);
    # change this field to move the default for every run that doesn't pass
    # the flag. 3-letter weekday abbreviation (mon..sun). Resolved via
    # ``content_ledger.next_free_superlong_slot``,
    # which reuses ``next_free_slot``'s region-aware ledger occupancy so a
    # part's slot can never double-book an already-booked LF day.
    superlong_publish_day: str = "sat"
    # Region → superlong publish weekday, the location-aware override that
    # mirrors every other per-region drive setting
    # (``active_journey_by_region``). A region present here wins; a region
    # absent (or a run with no ``--region``) falls back to the flat
    # ``superlong_publish_day`` above — the same precedence
    # ``resolve_shorts_times_for_region`` uses. EMPTY by default so behaviour
    # is unchanged until an operator opts a region in; resolved via
    # ``schedule_constants.resolve_superlong_publish_day``. Ireland and USA
    # run different LF cadences (Sun vs Tue/Fri), so their parts should be
    # free to drip on different days.
    superlong_publish_day_by_region: dict[str, str] = Field(default_factory=dict)
    # Source video files read from another filesystem than staging (WSL2
    # drvfs, NFS) cost read speed only: pre-flight resolves each file's
    # symlink and logs a PERFORMANCE warning, never a refusal. True
    # lowers that report to INFO. Dangling/unreadable files refuse regardless.
    allow_cross_fs_source: bool = False
    # When WeatherDetector returns degraded measurements (fewer than
    # min_success_ratio chunks succeed), the scorer falls back to
    # DEFAULT_WEATHER_CLASS instead of raising. This defaulted to False —
    # a hard fail — back when weather carried 7% of the composite score
    # and a silent fallback meant scoring on a fabricated class. With the
    # weather weight removed (see scoring_weights above), the fallback
    # changes no ranking whatsoever, so failing a multi-hour drive over an
    # unweighted measurement is pure cost. Flip back to False alongside any
    # change that gives weather real weight again.
    allow_degraded_weather: bool = True
    # Default ffprobe timeout (seconds) for validation/duration probes. A
    # valid file probed while a heavy concurrent encode pins the CPU can take
    # longer than the 30s default; the FFprobeClient retries a timeout a few
    # times (it is transient load, not corruption) before failing, but an
    # operator on a slow machine can raise this floor so probes never time out
    # mid-run. Threaded into FFprobeClient at its pipeline construction sites.
    ffprobe_timeout_sec: int = 30
    # Whether the Metricool shorts CSV publishes to TikTok. Default True:
    # csv_generator emits TikTok=TRUE + populated TikTok columns so Metricool
    # schedules the short on TikTok. Publishing is handled entirely by
    # Metricool and needs no first-party TikTok auth. The separate analytics
    # path (``TikTokCollector``, Display API ``video/query``) is independent
    # of this knob: it needs its own first-party credentials with the
    # ``video.list`` scope and degrades to no metrics without them, and it
    # never gated *publishing*. Set False to suppress
    # TikTok scheduling (e.g. a non-TikTok account). Wired into Phase D's
    # ``_emit_consolidated_shorts_csv`` so this knob is live, not decorative.
    # It gates the CSV only, never ``publishing_route_tiktok``: TikTok's inbox
    # route runs ``first_party`` with this False so the CSV has no TikTok column.
    publish_tiktok: bool = True
    # Weather chunked-brightness measurement knobs. The
    # default ffmpeg invocation uses -hwaccel auto (CUDA where available)
    # — matches the rest of the pipeline and stops weather being the lone
    # software-decode hotspot on 4K HEVC. weather_chunk_max_sec bounds
    # each ffmpeg invocation's input span; weather_chunk_timeout_cap_sec
    # caps the per-chunk wall budget. Raise either when even CUDA
    # signalstats can't keep up (8K sources).
    weather_chunk_max_sec: float = 120.0
    weather_chunk_timeout_cap_sec: float = 180.0
    weather_hwaccel: str = "auto"
    # Per-platform publish route (YouTube Shorts, Instagram Reels, Facebook
    # Reels, TikTok adapters under gomoveshift/publishing/). "first_party"
    # opts a platform into the adapters; "off" ships nothing for that
    # platform. Metricool is retired -- it is no longer a valid value (see
    # _validate_publishing_route). The built-in default is "off": a config
    # that names no route publishes nowhere, so a fresh checkout (or the
    # test suite's blank HOME) never enters Phase D's publish-plan sidecar
    # by accident; the operator opts each platform in explicitly. Each
    # field's DEFAULT otherwise comes from its
    # GMS_PUBLISHING_ROUTE_* env var (same env-supplies-the-default,
    # explicit-config-wins precedent as geocoding_cache_dir above) — this is
    # what lets the console Settings tab flip a route per brand without
    # editing config.yaml. An explicit config.yaml/constructor value still
    # wins over the env var.
    publishing_route_youtube: str = Field(
        default_factory=lambda: os.environ.get("GMS_PUBLISHING_ROUTE_YOUTUBE", "off"),
        validate_default=True,
    )
    publishing_route_instagram: str = Field(
        default_factory=lambda: os.environ.get("GMS_PUBLISHING_ROUTE_INSTAGRAM", "off"),
        validate_default=True,
    )
    publishing_route_facebook: str = Field(
        default_factory=lambda: os.environ.get("GMS_PUBLISHING_ROUTE_FACEBOOK", "off"),
        validate_default=True,
    )
    publishing_route_tiktok: str = Field(
        default_factory=lambda: os.environ.get("GMS_PUBLISHING_ROUTE_TIKTOK", "off"),
        validate_default=True,
    )
    #: Legacy config.yaml key from the Metricool era. No longer read for
    #: anything -- accepted only so an un-migrated config.yaml doesn't
    #: crash construction; a single WARNING flags it for removal.
    rollback: Optional[str] = Field(default=None, exclude=True)
    # Attests that TikTok has APPROVED the Content Posting API audit for
    # this app. Setting this True without real TikTok approval does not
    # unlock public posting — TikTok force-privates PUBLIC_TO_EVERYONE posts
    # from an unaudited client regardless of this flag (see
    # tiktok_adapter.assert_public_routing_allowed); it just silently ships
    # posts that stay private. Default False.
    publishing_tiktok_audit_approved: bool = False
    # Audience for every TikTok short in a plan. Defaults to the PRIVATE
    # audience so the first audited run is a real canary: the operator opts
    # into a public audience deliberately, once. Note this is only the
    # audience we REQUEST — TikTok gives us no way to read back what was
    # applied (see tiktok_adapter.confirm_publish), so a wrong value here is
    # not recoverable after the fact.
    publishing_tiktok_privacy_level: str = "SELF_ONLY"

    @field_validator("publishing_tiktok_privacy_level")
    @classmethod
    def _validate_tiktok_privacy_level(cls, v: str) -> str:
        from gomoveshift.publishing.models import TIKTOK_PRIVACY_LEVELS

        if v not in TIKTOK_PRIVACY_LEVELS:
            raise ValueError(
                f"publishing_tiktok_privacy_level must be one of "
                f"{sorted(TIKTOK_PRIVACY_LEVELS)}, got {v!r}"
            )
        return v

    @field_validator(
        "publishing_route_youtube",
        "publishing_route_instagram",
        "publishing_route_facebook",
        "publishing_route_tiktok",
    )
    @classmethod
    def _validate_publishing_route(cls, v: str, info: ValidationInfo) -> str:
        allowed = {"first_party", "off"}
        if v in RETIRED_PUBLISHING_ROUTES:
            logger.warning(
                "DriveConfig: %s=%r is a retired publishing route -- treating it as 'off' "
                "(nothing publishes on that platform); set 'first_party' in config.yaml "
                "to publish there, or 'off' to silence this warning.",
                info.field_name,
                v,
            )
            return "off"
        if v not in allowed:
            raise ValueError(f"publishing route must be one of {sorted(allowed)}, got {v!r}")
        return v

    @field_validator("rollback")
    @classmethod
    def _warn_legacy_rollback(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            logger.warning(
                "DriveConfig: legacy 'rollback' config key (%r) is ignored -- Metricool is "
                "retired; remove it from config.yaml. Routes are 'first_party' or 'off'.",
                v,
            )
        return v

    @field_validator("video_extensions")
    @classmethod
    def _validate_video_extensions(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("video_extensions must contain at least one entry")
        normalized: list[str] = []
        for ext in v:
            if not isinstance(ext, str) or not ext.startswith("."):
                raise ValueError(f"video_extensions entries must start with '.' (got {ext!r})")
            normalized.append(ext.lower())
        return normalized

    @field_validator("long_form_publish_days")
    @classmethod
    def _validate_long_form_publish_days(cls, v: list[int]) -> list[int]:
        for d in v:
            if not isinstance(d, int) or d < 0 or d > 6:
                raise ValueError(
                    f"long_form_publish_days entries must be ints in [0, 6], got {d!r}"
                )
        return v

    @field_validator("long_form_publish_time")
    @classmethod
    def _validate_long_form_publish_time(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if not isinstance(v, str) or not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", v):
            raise ValueError(f"long_form_publish_time must match HH:MM (24h), got {v!r}")
        return v

    @field_validator("shorts_publish_times")
    @classmethod
    def _validate_shorts_publish_times(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("shorts_publish_times must contain at least one HH:MM entry")
        pat = re.compile(r"([01]\d|2[0-3]):[0-5]\d")
        for entry in v:
            if not isinstance(entry, str) or not pat.fullmatch(entry):
                raise ValueError(
                    f"shorts_publish_times entries must match HH:MM (24h), got {entry!r}"
                )
        return v

    def model_post_init(self, __context) -> None:
        """Validate config invariants after construction."""
        self.validate_weights()
        self.check_shorts_consistency()
        if self.min_lf_duration > self.max_lf_duration:
            raise ValueError(
                f"min_lf_duration ({self.min_lf_duration}) must be <= "
                f"max_lf_duration ({self.max_lf_duration})"
            )
        if self.shorts_max_duration_sec > SHORTS_MAX_DURATION:
            raise ValueError(
                f"shorts_max_duration_sec ({self.shorts_max_duration_sec}) must be "
                f"<= {SHORTS_MAX_DURATION} (YouTube Shorts hard ceiling)"
            )
        if not (
            self.shorts_min_duration_sec
            <= self.shorts_target_duration_sec
            <= self.shorts_max_duration_sec
        ):
            raise ValueError(
                "shorts_min_duration_sec <= shorts_target_duration_sec <= "
                "shorts_max_duration_sec is required, got "
                f"shorts_min_duration_sec={self.shorts_min_duration_sec}, "
                f"shorts_target_duration_sec={self.shorts_target_duration_sec}, "
                f"shorts_max_duration_sec={self.shorts_max_duration_sec}"
            )
        if self.min_segment_duration <= 0:
            raise ValueError(f"min_segment_duration must be > 0, got {self.min_segment_duration}")
        if self.montage_peak_count < 1:
            raise ValueError(f"montage_peak_count must be >= 1, got {self.montage_peak_count}")
        if self.montage_max_duration_sec <= 0:
            raise ValueError(
                f"montage_max_duration_sec must be > 0, got {self.montage_max_duration_sec}"
            )
        if self.superlong_part_max_duration_sec <= 0:
            raise ValueError(
                "superlong_part_max_duration_sec must be > 0, got "
                f"{self.superlong_part_max_duration_sec}"
            )
        if not (0.0 <= self.superlong_split_grace <= 0.5):
            raise ValueError(
                f"superlong_split_grace must be in [0, 0.5], got {self.superlong_split_grace}"
            )
        if self.gps_captions not in ("off", "events"):
            raise ValueError(f"gps_captions must be 'off' or 'events', got {self.gps_captions!r}")
        if self.regular_lf_lower_third_duration_sec <= 0:
            raise ValueError("regular_lf_lower_third_duration_sec must be > 0")
        if self.regular_lf_lower_third_min_gap_sec < 0:
            raise ValueError("regular_lf_lower_third_min_gap_sec must be >= 0")
        _weekdays = {"mon", "tue", "wed", "thu", "fri", "sat", "sun"}
        if self.superlong_publish_day.lower() not in _weekdays:
            raise ValueError(
                "superlong_publish_day must be a 3-letter weekday abbreviation "
                f"(mon..sun), got {self.superlong_publish_day!r}"
            )
        for _region, _day in (self.superlong_publish_day_by_region or {}).items():
            if not isinstance(_day, str) or _day.lower() not in _weekdays:
                raise ValueError(
                    f"superlong_publish_day_by_region[{_region!r}] must be a 3-letter "
                    f"weekday abbreviation (mon..sun), got {_day!r}"
                )

    def validate_weights(self) -> None:
        """Ensure scoring weights sum to 1.0."""
        total = sum(self.scoring_weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Drive scoring weights must sum to 1.0, got {total:.3f}")

    def check_shorts_consistency(self) -> None:
        """Log a warning if shorts_target_count < min_shorts_per_lf.

        This is not a hard error — the validation gate in Phase D handles
        the actual enforcement. But it indicates a likely misconfiguration.
        """
        if self.shorts_target_count < self.min_shorts_per_lf:
            logger.warning(
                "shorts_target_count (%d) < min_shorts_per_lf (%d) — "
                "the planner will cap at %d shorts but the validation "
                "gate requires %d. Raise shorts_target_count or lower "
                "min_shorts_per_lf.",
                self.shorts_target_count,
                self.min_shorts_per_lf,
                self.shorts_target_count,
                self.min_shorts_per_lf,
            )

    def resolved_geocode_cache_dir(self) -> Path:
        """The single reverse-geocode cache directory for a whole run.

        Every geocoder — Phase A GPX derivation, Phase BC per-LF centroid
        day labels, and Phase D chapters — resolves through here, so one
        coordinate caches (and therefore resolves) identically across the
        title, the day label, the ``Through:`` line, and the chapters
        within a single run.

        ``geocoding_cache_dir`` already honors ``GMS_GEOCODING_CACHE_DIR``
        (via its field default, with an explicit config value winning over
        it). ``GMS_GPS_CACHE_DIR`` is the legacy Phase-A/BC env override —
        still honored, and now it steers ALL consumers at one directory
        instead of only Phase A. ``or`` (not a default arg) so an unset OR
        empty env var falls back instead of resolving ``Path("")`` to the
        process CWD. ``~`` is expanded here at the single read site.
        """
        raw = os.environ.get("GMS_GPS_CACHE_DIR") or self.geocoding_cache_dir
        return Path(raw).expanduser()


class Config(BaseModel):
    """Main configuration container."""

    paths: PathsConfig = Field(default_factory=PathsConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    sign_detection: SignDetectionConfig = Field(default_factory=SignDetectionConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    dead_spots: DeadSpotsConfig = Field(default_factory=DeadSpotsConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    web_ui: WebUIConfig = Field(default_factory=WebUIConfig)
    stop_detection: StopDetectionConfig = Field(default_factory=StopDetectionConfig)
    seo: SEOConfig = Field(default_factory=SEOConfig)
    drive: DriveConfig = Field(default_factory=DriveConfig)
    photos: PhotosConfig = Field(default_factory=PhotosConfig)
    playlist_reconciler: PlaylistReconcilerConfig = Field(default_factory=PlaylistReconcilerConfig)

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "Config":
        """Load configuration from YAML, deep-merging user overrides.

        Precedence (lowest → highest):

        1. Pydantic defaults
        2. Project config: ``./config.yaml`` (or ``./config.yml``) in the
           launch directory, else ``config.yaml`` in ``gms_inputs_dir()`` --
           the operator's private inputs layer
        3. User override: ``config.yaml`` under the config home
           (``~/.config/gomoveshift/`` by default)

        Nested mappings are merged key-by-key so a user file that only
        sets ``drive.long_form_publish_days`` does NOT wipe the repo's
        ``drive.gps_quality`` block. Non-mapping values at any level are
        replaced wholesale (lists included — explicit override beats
        the ambiguity of list-merging).

        When ``config_path`` is provided explicitly, only that file is
        read (no user override applied) — preserves existing behaviour
        for callers that know the exact file they want. Matches how
        ``git``, ``vim``, and most Unix tools layer system → user config.

        Rationale: the earlier search-first-win behaviour
        meant the repo's tracked sample ``config.yaml`` always shadowed
        ``~/.config/gomoveshift/config.yaml`` when invoked from the
        service directory. Operators could not override cadence without
        editing the tracked file (dirty repo + ownership issues) or
        running from an unusual cwd.
        """
        if config_path is not None:
            if config_path.exists():
                with open(config_path) as f:
                    try:
                        data = yaml.safe_load(f) or {}
                    except yaml.YAMLError as e:
                        raise ConfigError(f"malformed YAML in {config_path}: {e}") from e
                _warn_on_stop_knob_drift(data)
                _warn_on_retired_keys(data)
                return cls.model_validate(data)
            return cls()

        project_candidates = [
            Path("config.yaml"),
            Path("config.yml"),
            gms_inputs_dir() / "config.yaml",
        ]
        user_path = gms_config_home() / "config.yaml"

        def _read_yaml_dict(path: Path) -> Optional[dict]:
            """Return ``path``'s YAML as a dict, or None if missing/empty/
            not a mapping. Tolerates TOCTOU and over-eager
            ``Path.exists`` monkeypatches in tests."""
            try:
                with open(path) as f:
                    loaded = yaml.safe_load(f)
            except (FileNotFoundError, IsADirectoryError, PermissionError):
                return None
            except yaml.YAMLError as e:
                raise ConfigError(f"malformed YAML in {path}: {e}") from e
            if not isinstance(loaded, dict):
                return None
            return loaded

        merged: dict = {}
        for candidate in project_candidates:
            if candidate.exists():
                data = _read_yaml_dict(candidate)
                if data is not None:
                    merged = _deep_merge_dict(merged, data)
                break

        if user_path.exists():
            user_data = _read_yaml_dict(user_path)
            if user_data is not None:
                merged = _deep_merge_dict(merged, user_data)

        if merged:
            _warn_on_stop_knob_drift(merged)
            _warn_on_retired_keys(merged)
            return cls.model_validate(merged)
        return cls()


def get_config(config_path: Optional[Path] = None) -> Config:
    """Get the application configuration."""
    return Config.load(config_path)
