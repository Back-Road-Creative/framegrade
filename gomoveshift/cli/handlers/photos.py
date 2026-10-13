"""Handler for the ``photos`` CLI command and its subcommands."""

import logging
import math
import sys
from collections import Counter
from pathlib import Path

from ..logging_setup import configure_logging

logger = logging.getLogger(__name__)


def _build_voice_map():
    """Build CLI voice-name -> VoiceStyle mapping (lazy to avoid top-level import)."""
    from ...pipeline.photos.voice_styles import VoiceStyle

    return {
        "first_person": VoiceStyle.FIRST_PERSON,
        "minimal_zen": VoiceStyle.MINIMAL_ZEN,
        "engaging": VoiceStyle.ENGAGING_CONVERSATIONAL,
        "interactive": VoiceStyle.INTERACTIVE,
    }


#: CLI voice-name -> VoiceStyle enum.  Populated on first access.
VOICE_MAP: dict | None = None


def _get_voice_map():
    """Return the voice map, building it on first call."""
    global VOICE_MAP  # noqa: PLW0603
    if VOICE_MAP is None:
        VOICE_MAP = _build_voice_map()
    return VOICE_MAP


def _cmd_photos_ingest(args) -> int:
    """Camera card -> developed gallery JPEGs -> journey frontmatter.

    ``--journey`` is a slug by default; an explicit ``*.md`` path wins.

    ``--location`` is optional: omitted, the card's EXIF GPS elects the label.
    The resolution order is the library's and is not second-guessed here — an
    operator label wins outright, EXIF elects only in its absence, and neither
    means a hard failure. This prints what was resolved so the operator can see
    which label was persisted.

    ``--style`` bakes a creative look into the gallery masters (validated here,
    the same door ``photos develop`` uses); omitted it falls to
    ``photos.ingest_style`` (default "" = the neutral base grade, byte-identical
    to no style). ``--style-strength`` is the per-run lever.
    """
    from ...config import ConfigError, get_config
    from ...pipeline.photos.develop import PREVIEW_FALLBACK_DECODER
    from ...pipeline.photos.exif_gps import NO_EXIF_GPS, NO_GEOCODE
    from ...pipeline.photos.ingest import PhotoIngestError, ingest
    from ...pipeline.photos.selector import JOURNEYS_CONTENT, default_site_root
    from ...pipeline.photos.styles import STYLES

    journey_path = Path(args.journey)
    if journey_path.suffix != ".md":
        try:
            root = Path(args.site_path) if args.site_path else default_site_root()
        except ConfigError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        journey_path = root / JOURNEYS_CONTENT / f"{args.journey}.md"

    photos = get_config().photos
    style = args.style if args.style is not None else photos.ingest_style
    strength = args.style_strength if args.style_strength is not None else photos.style_strength
    if style:
        if style not in STYLES:
            print(f"Unknown style {style!r} (available: {', '.join(STYLES)})", file=sys.stderr)
            return 1
        if not _style_strength_ok(strength):
            print(_STYLE_STRENGTH_REFUSAL.format(got=strength), file=sys.stderr)
            return 1

    try:
        report = ingest(
            args.card_dir,
            journey_path,
            day_date=args.day,
            location=args.location,
            site_root=args.site_path,
            skip_blur=getattr(args, "skip_blur", False),
            redevelop=getattr(args, "redevelop", False),
            dry_run=getattr(args, "dry_run", False),
            style=style,
            style_strength=strength,
        )
    except PhotoIngestError as e:
        print(f"Ingest failed: {e}", file=sys.stderr)
        # Only the location-resolution failures are actionable via the flag;
        # an unknown day or a bad decode must not be papered over with it.
        if not args.location and any(r in str(e) for r in (NO_EXIF_GPS, NO_GEOCODE)):
            print(
                "  This card carries no usable GPS EXIF, so nothing can elect a label. "
                "Re-run with --location 'LABEL' — a location is never defaulted or invented.",
                file=sys.stderr,
            )
        return 1

    verb = "DRY RUN — would ingest" if report.dry_run else "Ingested"
    print(f"{verb} {len(report.ingested)} photo(s) into {journey_path}")
    if report.location:
        print(f"  location: {report.location}")
    if report.decoder_counts:
        counts = ", ".join(f"{tag} x{n}" for tag, n in sorted(report.decoder_counts.items()))
        print(f"  decode provenance: {counts}")
        if report.decoder_counts.get(PREVIEW_FALLBACK_DECODER):
            print("  NOTE: preview-fallback develops are 8-bit — DNG hook + --redevelop to fix")
    if report.blur_skipped and not report.dry_run:
        print("  blur SKIPPED (--skip-blur): gallery JPEGs are UNBLURRED")
    elif not report.dry_run and (report.ingested or report.redeveloped):
        print(f"  blurred privacy regions in {report.blurred} photo(s)")
    for name in report.ingested:
        print(f"  + {name}")
    for name in report.redeveloped:
        print(f"  ~ {name} (re-developed from preview fallback)")
    for source, reason in report.skipped:
        print(f"  - skipped {source}: {reason}")
    return 0


def _resolve_luts(specs, luts_map):
    """Resolve each ``--lut`` spec to a ``(name, Cube)`` pair, once, upfront.

    A spec that matches a ``photos.luts`` key is that named look (variant name =
    the key); anything else is a filesystem path (variant name = its file stem).
    A malformed/unreadable ``.cube`` raises ``CubeError`` — a global error, so it
    fails the whole invocation here rather than once per photo in the loop.
    """
    from pycube_lut import load_cube

    resolved = []
    for spec in specs:
        if spec in luts_map:
            name, path = spec, Path(luts_map[spec]).expanduser()
        else:
            path = Path(spec).expanduser()
            name = path.stem
        resolved.append((name, load_cube(path)))
    return resolved


#: The style lever's refusal, shared by ``photos ingest`` and ``photos develop``.
_STYLE_STRENGTH_REFUSAL = "--style-strength must be a finite value > 0 (got {got})"


def _style_strength_ok(strength: float) -> bool:
    """The style lever is the ONE knob with no upper bound, so it cannot use the
    ``lo <= v <= hi`` form every sibling gate uses — and that form is what
    refuses NaN for free, since NaN loses every comparison. Spelled ``strength
    <= 0`` this gate waved NaN through: it reached ``CorrectionParams``, and the
    render silently produced a different grade than the flag or
    ``photos.style_strength`` asked for, with no error on any path. Require a
    finite positive number explicitly. Infinity is refused with it — the rails
    saturate by about x2, so an unbounded lever buys nothing but a value that
    cannot be printed, compared, or recorded in a ``.grade.json`` sidecar."""
    return math.isfinite(strength) and strength > 0


def _resolve_hue_band(supplied, cfg_default, flag: str, lo: float, hi: float) -> tuple:
    """Resolve one ``--hue-*`` band (flag beats config) and validate it before any
    decode — an upstream gate. Empty = OFF; otherwise six per-anchor-hue values
    (red/yellow/green/cyan/blue/magenta) or seven with ORANGE second, each within
    ``[lo, hi]``. The arity gate lives HERE rather than in argparse ``nargs``,
    which cannot express "6 or 7"; it still fires before any decode. Returns the
    band as a tuple; raises ``ValueError`` with an operator message on a bad
    length or an out-of-range value."""
    band = list(supplied) if supplied is not None else list(cfg_default)
    if band and len(band) not in (6, 7):
        raise ValueError(
            f"{flag} needs 6 values (red/yellow/green/cyan/blue/magenta) or 7 "
            f"(orange 30° second), got {len(band)}"
        )
    if any(not lo <= v <= hi for v in band):
        raise ValueError(f"{flag} values must each be in [{lo:g}, {hi:g}]")
    return tuple(band)


def _resolve_tint(supplied, cfg_default, flag: str, lo: float, hi: float) -> tuple:
    """Resolve one ``--split-tone-*`` tint (flag beats config) and validate it before
    any decode — an upstream gate. Empty = OFF; otherwise exactly three signed
    (r,g,b) values, each within ``[lo, hi]``. Returns the tint as a tuple; raises
    ``ValueError`` with an operator message on a bad length or an out-of-range value."""
    tint = list(supplied) if supplied is not None else list(cfg_default)
    if tint and len(tint) != 3:
        raise ValueError(f"{flag} needs exactly 3 values (signed r,g,b), got {len(tint)}")
    if any(not lo <= v <= hi for v in tint):
        raise ValueError(f"{flag} values must each be in [{lo:g}, {hi:g}]")
    return tuple(tint)


def _resolve_mono(supplied, cfg_default, flag: str, count: int, lo: float, hi: float) -> tuple:
    """Resolve one B&W control (``--bw-mix`` / ``--duotone``; flag beats config) and
    validate it before any decode — an upstream gate. Empty = OFF; otherwise exactly
    ``count`` values, each within ``[lo, hi]``. Returns a tuple; raises ``ValueError``
    with an operator message on a bad length or an out-of-range value."""
    vals = list(supplied) if supplied is not None else list(cfg_default)
    if vals and len(vals) != count:
        raise ValueError(f"{flag} needs exactly {count} values, got {len(vals)}")
    if any(not lo <= v <= hi for v in vals):
        raise ValueError(f"{flag} values must each be in [{lo:g}, {hi:g}]")
    return tuple(vals)


def _resolve_curve(supplied, cfg_default, flag: str) -> tuple:
    """Resolve one ``--curve-*`` channel curve (flag beats config) and validate it
    before any decode — an upstream gate. Empty = OFF; otherwise a FLAT list of
    alternating x y control points (even length >= 4, >= 2 points, x strictly
    ascending in [0, 1]). Returns a tuple; raises ``ValueError`` on a bad shape.
    The ``CorrectionParams`` __post_init__ is the structural gate underneath."""
    pts = list(supplied) if supplied is not None else list(cfg_default)
    if pts and (len(pts) % 2 or len(pts) < 4):
        raise ValueError(f"{flag} needs an even-length flat list of >= 2 (x, y) points")
    xs = pts[0::2]
    if any(b <= a for a, b in zip(xs, xs[1:])):
        raise ValueError(f"{flag} x control points must be strictly ascending")
    if any(not 0.0 <= x <= 1.0 for x in xs):
        raise ValueError(f"{flag} x control points must each be in [0, 1]")
    return tuple(pts)


#: What a ``.grade.json`` this build wrote always carries and a foreign JSON does
#: not — each key by NAME and by SHAPE, so a ``provenance`` that is a string is
#: refused at the door rather than raising off the ``.get`` below.
_GRADE_KEYS = {"variant": str, "params": dict, "options": dict, "provenance": dict}

# Whole-frame tone knobs that, like the white-balance axes, have a flag and no
# config default; ``_cmd_photos_develop`` gates them in the same loop.
_TONE_FLAG_KNOBS = ("exposure_ev", "shadow_recovery", "highlight_recovery")


def _load_grade_sidecar(path: Path):
    """Parse one ``.grade.json`` into ``(variant, CorrectionParams, DevelopOptions)``.

    An UPSTREAM gate: a file that is not a sidecar, cannot be parsed, or was
    written by another schema or develop algorithm is refused HERE, at the door,
    before a single photo is decoded — never as a raise mid-batch. The version
    pair is refused outright rather than warned past because the whole point of a
    replay is the pixels the operator already approved, and no other algorithm
    version can produce them; the sidecar is one develop away from being rewritten.
    Raises ``ValueError`` carrying the operator message."""
    import json

    from ...pipeline.photos.develop import (
        DEVELOP_ALGO_VERSION,
        GRADE_SIDECAR_VERSION,
        correction_params_from_sidecar,
        develop_options_from_sidecar,
    )

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ValueError(f"{path}: not readable as JSON ({e})") from None
    if not isinstance(data, dict) or any(
        not isinstance(data.get(k), t) for k, t in _GRADE_KEYS.items()
    ):
        raise ValueError(f"{path}: not a photos develop grade sidecar")
    if data.get("sidecar_version") != GRADE_SIDECAR_VERSION:
        raise ValueError(
            f"{path}: sidecar schema v{data.get('sidecar_version')}, this build writes "
            f"v{GRADE_SIDECAR_VERSION} — re-develop the frame with --grade-sidecar"
        )
    stamped = data["provenance"].get("develop_algo_version")
    if stamped != DEVELOP_ALGO_VERSION:
        raise ValueError(
            f"{path}: recorded under develop algorithm {stamped}, this build is "
            f"{DEVELOP_ALGO_VERSION} — those numbers cannot replay to the frame that "
            "was approved; re-develop it with --grade-sidecar and approve that"
        )
    try:
        return (
            str(data["variant"]),
            correction_params_from_sidecar(data["params"]),
            develop_options_from_sidecar(data["options"]),
        )
    except (AttributeError, TypeError, ValueError) as e:
        raise ValueError(f"{path}: unusable grade record ({e})") from None


def _replayed_options(recorded, options, args):
    """The recorded knob record, minus every knob this command line re-states.
    Precedence is the one ``_apply_look`` already establishes and for the same
    reason: an explicitly passed flag is non-``None``, so it BEATS the file, and
    every knob left unsaid is restored from it. The field -> flag-dest map is
    ``config.LOOK_ALIASES``, read off the one knob registry rather than kept by
    hand here, so a knob renamed or added there cannot silently stop being
    replayable (``reference_lab`` is recorded as measured numbers, and its flag
    ``--reference`` names the file they were measured from)."""
    from dataclasses import fields, replace

    from ...config import LOOK_ALIASES

    kept = {
        f.name: getattr(recorded, f.name)
        for f in fields(recorded)
        if getattr(args, LOOK_ALIASES.get(f.name, f.name), None) is None
    }
    return replace(options, **kept)


def _apply_look(args, knobs) -> None:
    """Stand a named ``photos.looks`` entry in for the develop flags it sets.

    PRECEDENCE, and why this is a FILL rather than an overwrite: an explicitly
    passed flag is already non-``None``, so it is left alone and BEATS the look;
    a knob the look does not carry stays ``None`` and falls through to the bare
    ``photos.<knob>`` default in the resolution below. Explicit flag > look >
    config default, by construction — no per-knob threading, and a knob added to
    ``photos develop`` is recallable in a look without touching this function.
    Every look key is a validated flag dest defaulting to ``None`` and holding a
    SCALAR (config's ``LOOK_KNOBS`` gates that at load; a test pins the ``None``
    defaults). The fill is raw, so an ``append`` flag's dest — ``--lut``, whose
    dest holds a LIST — could never be on that table: given a bare string it is
    resolved character by character. ``LOOK_RUN_KNOBS`` refuses those by name.
    """
    for key, value in knobs.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)


def _unknown_look(flag: str, name: str, builtin, configured) -> str:
    """The refusal both look flags print: the name, then the built-in and configured
    names as SEPARATE tagged groups ("none" for an empty one)."""
    return (
        f"Unknown {flag}: {name} (built-in: {', '.join(sorted(builtin)) or 'none'}; "
        f"configured: {', '.join(sorted(configured)) or 'none'})"
    )


def _bad_border(label: str, style: str, styles) -> str:
    """The refusal both border doors print — the run's resolved style, and a look
    variant's own — naming which one carried the unknown style."""
    return f"{label}: unknown border style {style!r} (choose from {', '.join(styles)})"


def _print_variant_plan(path: Path, plan, strength: float) -> None:
    """Dry-run surface: SceneProfile fields, chosen styles, per-style params."""
    scene = " ".join(f"{k}={getattr(plan.scene, k)}" for k in plan.scene.__dataclass_fields__)
    print(f"DRY RUN — would develop {path} (style strength x{strength:g})\n  scene: {scene}")
    print(f"  styles: {', '.join(name for name, _ in plan.styled)}")
    if plan.lut_looks:
        looks = ", ".join(name for name, _ in plan.lut_looks)
        # A composed look rides a styled grade, so the style lever DOES reach it.
        rides = (
            "over a --lut-over styled grade the style strength scales"
            if plan.lut_params is not None
            else "style-strength does not apply"
        )
        print(f"  lut looks: {looks} (opacity x{plan.lut_strength:g}; {rides})")
    if plan.look_variants:
        names = ", ".join(name for name, _ in plan.look_variants)
        print(f"  look variants: {names} (each over the base its look names, else natural)")
    if plan.lut_auto_picks:  # an unauditable automatic choice is worse than a manual one
        picks = ", ".join(f"{n} score={s:+.3f}" for n, s in plan.lut_auto_picks)
        print(f"  lut auto-picks (scored against the scene above): {picks}")
    print(f"  {'style':<8} {'sat':>5} {'contrast':>8} {'gamma':>5} {'shadow':>6} {'high':>5}")
    for name, p in plan.styled:
        print(
            f"  {name:<8} {p.saturation_mult:>5.2f} {p.contrast_curve:>8.3f} "
            f"{p.gamma:>5.2f} {p.shadow_lift:>6.1f} {p.highlight_pull:>5.1f}"
        )


def _cmd_photos_develop(args) -> int:
    """Standalone per-photo style-variant develop into a LOCAL working dir —
    per-photo by design (no clustering), never the blur-gated public site tree.

    ``--look NAME`` recalls a ``photos.looks`` combination first, before any knob
    is resolved: precedence is explicit flag > look > bare ``photos.<knob>``
    default, so every resolution below reads the look as its config layer.
    """
    from pycube_lut import CubeError

    from ...config import LOOK_RANGES, check_layers, check_look_lists, get_config
    from ...pipeline.photos.bracket import MIN_BRACKET_FRAMES, detect_brackets
    from ...pipeline.photos.develop import (
        BORDER_STYLES,
        HDR_STEM_SUFFIX,
        LOOK_VARIANT_PREFIX,
        SUPPORTED_EXTENSIONS,
        DevelopOptions,
        PhotoDecodeError,
        _add_border,
        _analysis_rgb,
        compose_contact_sheet,
        develop_variants,
        export_jpeg,
        export_tiff16,
        fuse_bracket,
        look_variant_options,
        plan_variants,
        reference_lab_from_file,
        write_grade_sidecar,
    )
    from ...pipeline.photos.exif_gps import extract_exif_capture_metadata
    from ...pipeline.photos.looks import BUILTIN_LOOKS
    from ...pipeline.photos.styles import STYLES
    from ...pipeline.photos.subject import (
        read_capture_optics,
        read_subject_correction,
        recognise_subject,
    )

    photos = get_config().photos
    # The operator's config WINS a name collision with a built-in preset, so their
    # override survives an upgrade that ships that name. One table, both flags.
    known = {**BUILTIN_LOOKS, **photos.looks}
    # The look layer lands BEFORE every resolution below, so each one keeps its
    # single `flag if not None else config` line and inherits the precedence.
    if args.look is not None:
        if args.look not in known:
            print(_unknown_look("--look", args.look, BUILTIN_LOOKS, photos.looks), file=sys.stderr)
            return 1
        _apply_look(args, known[args.look])
    raw = args.styles if args.styles is not None else photos.develop_styles
    styles = [s.strip() for s in raw.split(",") if s.strip()] if raw else None
    strength = args.style_strength if args.style_strength is not None else photos.style_strength
    if not _style_strength_ok(strength):
        print(_STYLE_STRENGTH_REFUSAL.format(got=strength), file=sys.stderr)
        return 1
    lut_strength = args.lut_strength if args.lut_strength is not None else photos.lut_strength
    if not 0.0 <= lut_strength <= 1.0:
        print(f"--lut-strength must be in [0, 1] (got {lut_strength})", file=sys.stderr)
        return 1
    # The run-level default for every --look-variant panel's dial. No photos.<knob>
    # layer: the whole look (1.0) is the neutral value, and a look that wants less
    # carries its own `look_strength:`, which `look_variant_options` lays on top.
    look_strength = args.look_strength if args.look_strength is not None else 1.0
    if not 0.0 <= look_strength <= 1.0:
        print(f"--look-strength must be in [0, 1] (got {look_strength})", file=sys.stderr)
        return 1
    # A dial under 1.0 reaches --look-variant PANELS only, and refusing it here is
    # the only honest answer for a run with none: `--look` FOLDS the look into the
    # run's own knobs above, so nothing downstream can say which knob the look set
    # (an explicit flag beats it and is now the same value in the same slot), and
    # `styles`/`style_strength`/`lut`/`border`/`format` are no DevelopOptions field
    # at all — they decide which panels exist and how they export, which no pixel
    # lerp can halve. There is no look-free base render to blend toward. Reading
    # the RESOLVED value gates a recipe's own `look_strength:` identically, so the
    # two doors cannot disagree; 1.0 asks for the whole look and still renders.
    if look_strength < 1.0 and not args.look_variant:
        print(
            f"look strength {look_strength:g} (--look-strength, or a look's own "
            "`look_strength:`) has nothing to dial: it blends a --look-variant panel "
            "toward the run's base render, and this run has none. --look bakes the "
            f"whole look — render it as --look-variant {args.look or 'NAME'} to dial it.",
            file=sys.stderr,
        )
        return 1
    vignette = args.vignette if args.vignette is not None else photos.vignette
    if not 0.0 <= vignette <= 1.0:
        print(f"--vignette must be in [0, 1] (got {vignette})", file=sys.stderr)
        return 1
    clarity = args.clarity if args.clarity is not None else photos.clarity
    if not -1.0 <= clarity <= 1.0:
        print(f"--clarity must be in [-1, 1] (got {clarity})", file=sys.stderr)
        return 1
    grain = args.grain if args.grain is not None else photos.grain
    if not 0.0 <= grain <= 1.0:
        print(f"--grain must be in [0, 1] (got {grain})", file=sys.stderr)
        return 1
    grain_size = args.grain_size if args.grain_size is not None else photos.grain_size
    if not 0.25 <= grain_size <= 4.0:
        print(f"--grain-size must be in [0.25, 4] (got {grain_size})", file=sys.stderr)
        return 1
    halation = args.halation if args.halation is not None else photos.halation
    if not 0.0 <= halation <= 1.0:
        print(f"--halation must be in [0, 1] (got {halation})", file=sys.stderr)
        return 1
    light_leak = args.light_leak if args.light_leak is not None else photos.light_leak
    if not 0.0 <= light_leak <= 1.0:
        print(f"--light-leak must be in [0, 1] (got {light_leak})", file=sys.stderr)
        return 1
    chromatic_ab = args.chromatic_ab if args.chromatic_ab is not None else photos.chromatic_ab
    if not 0.0 <= chromatic_ab <= 1.0:
        print(f"--chromatic-ab must be in [0, 1] (got {chromatic_ab})", file=sys.stderr)
        return 1
    holo_sheen = args.holo_sheen if args.holo_sheen is not None else photos.holo_sheen
    if not 0.0 <= holo_sheen <= 1.0:
        print(f"--holo-sheen must be in [0, 1] (got {holo_sheen})", file=sys.stderr)
        return 1
    scanline = args.scanline if args.scanline is not None else photos.scanline
    if not 0.0 <= scanline <= 1.0:
        print(f"--scanline must be in [0, 1] (got {scanline})", file=sys.stderr)
        return 1
    glow = args.glow if args.glow is not None else photos.glow
    if not 0.0 <= glow <= 1.0:
        print(f"--glow must be in [0, 1] (got {glow})", file=sys.stderr)
        return 1
    glitch = args.glitch if args.glitch is not None else photos.glitch
    if not 0.0 <= glitch <= 1.0:
        print(f"--glitch must be in [0, 1] (got {glitch})", file=sys.stderr)
        return 1
    dehaze = args.dehaze if args.dehaze is not None else photos.dehaze
    if not 0.0 <= dehaze <= 1.0:
        print(f"--dehaze must be in [0, 1] (got {dehaze})", file=sys.stderr)
        return 1
    sky_darken = args.sky_darken if args.sky_darken is not None else photos.sky_darken
    if not 0.0 <= sky_darken <= 1.0:
        print(f"--sky-darken must be in [0, 1] (got {sky_darken})", file=sys.stderr)
        return 1
    # The chroma half of the sky move: a flag/look knob with no config default, as
    # --white-balance-g is. Gated HERE, at the door, so a bad amount never decodes.
    sky_saturate = args.sky_saturate or 0.0
    if not 0.0 <= sky_saturate <= 1.0:
        print(f"--sky-saturate must be in [0, 1] (got {sky_saturate})", file=sys.stderr)
        return 1
    # The grad ND: a flag/look knob with no photos.<knob> layer, as --white-balance-g
    # is. Its ANGLE is bounded at a quarter turn HERE, at the door, which is what makes
    # a ground-darkening grad — the one output this filter must not produce —
    # unexpressible rather than something the render has to be inspected for.
    graduated_nd = args.graduated_nd or 0.0
    if not 0.0 <= graduated_nd <= 1.0:
        print(f"--graduated-nd must be in [0, 1] (got {graduated_nd})", file=sys.stderr)
        return 1
    graduated_nd_angle = args.graduated_nd_angle or 0.0
    if not -90.0 <= graduated_nd_angle <= 90.0:
        got = graduated_nd_angle
        print(f"--graduated-nd-angle must be in [-90, 90] (got {got})", file=sys.stderr)
        return 1
    sharpen = args.sharpen if args.sharpen is not None else photos.sharpen
    if not 0.0 <= sharpen <= 2.0:
        print(f"--sharpen must be in [0, 2] (got {sharpen})", file=sys.stderr)
        return 1
    # Border is a presentational EXPORT op — the argparse choices and the
    # PhotosConfig validator both gate the style upstream, before any decode.
    # A --look writes its knobs straight onto `args`, past BOTH of those gates, so
    # the resolved style is re-gated HERE rather than by `_add_border` mid-batch.
    border = args.border if args.border is not None else photos.border
    if border not in BORDER_STYLES:
        print(_bad_border("--border", border, BORDER_STYLES), file=sys.stderr)
        return 1
    # Master container, gated upstream of any decode exactly as --border is; it
    # also sets the RENDER depth, since upsampling an 8-bit frame at the export
    # step would buy a wider container rather than more depth.
    export_format = args.format if args.format is not None else photos.develop_format
    depth = 16 if export_format == "tiff16" else 8
    # Metadata-only opt-in, so either switch turns it on; off writes nothing.
    grade_sidecar = args.grade_sidecar or photos.grade_sidecar
    # Same either-switch opt-in; off, no bracket is detected and nothing is fused.
    hdr = args.hdr or photos.develop_hdr
    vibrance = args.vibrance if args.vibrance is not None else photos.vibrance
    if not 0.0 <= vibrance <= 2.0:
        print(f"--vibrance must be in [0, 2] (got {vibrance})", file=sys.stderr)
        return 1
    # Every white-balance axis: flag/look knobs with no config default, as --layers
    # is. Each is gated HERE, at the door, against the SAME span its registry row
    # declares, so a bad shift never reaches a decode — the one table below is
    # LOOK_RANGES itself, so the flag door and the config door cannot disagree.
    # Exposure and highlight/shadow recovery ride the same door: flag-only, no
    # config default, gated against their registry span before any decode.
    white_balance = {}
    for name, (lo, hi) in LOOK_RANGES.items():
        if not name.startswith("white_balance") and name not in _TONE_FLAG_KNOBS:
            continue
        white_balance[name] = getattr(args, name) or 0.0
        if not lo <= white_balance[name] <= hi:
            flag = "--" + name.replace("_", "-")
            print(
                f"{flag} must be in [{lo:g}, {hi:g}] (got {white_balance[name]})", file=sys.stderr
            )
            return 1
    # Geometry: flag-only and run-owned (no config default, no look may set it),
    # gated here against the registry span before any decode.
    straighten_angle = args.straighten or 0.0
    lo, hi = LOOK_RANGES["straighten"]
    if not lo <= straighten_angle <= hi:
        print(f"--straighten must be in [{lo:g}, {hi:g}] (got {straighten_angle})", file=sys.stderr)
        return 1
    crop_rect = tuple(args.crop or ())
    if crop_rect:
        if not all(0.0 <= v <= 1.0 for v in crop_rect):
            print(f"--crop values must each be in [0, 1] (got {list(crop_rect)})", file=sys.stderr)
            return 1
        left, top, right, bottom = crop_rect
        if not (left < right and top < bottom):
            print(
                f"--crop needs left < right and top < bottom (got {list(crop_rect)})",
                file=sys.stderr,
            )
            return 1
    lens_correct = bool(args.lens_correct)
    try:
        hue_shift = _resolve_hue_band(
            args.hue_shift, photos.hue_shift, "--hue-shift", -180.0, 180.0
        )
        hue_sat = _resolve_hue_band(args.hue_sat, photos.hue_sat, "--hue-sat", 0.0, 5.0)
        hue_lum = _resolve_hue_band(args.hue_lum, photos.hue_lum, "--hue-lum", 0.0, 5.0)
        splittone_shadow = _resolve_tint(
            args.split_tone_shadow, photos.splittone_shadow, "--split-tone-shadow", -0.5, 0.5
        )
        splittone_midtone = _resolve_tint(
            args.split_tone_midtone, photos.splittone_midtone, "--split-tone-midtone", -0.5, 0.5
        )
        splittone_highlight = _resolve_tint(
            args.split_tone_highlight,
            photos.splittone_highlight,
            "--split-tone-highlight",
            -0.5,
            0.5,
        )
        # Empty = the film red bias; three weights in [0, 1] colour the bleed.
        halation_tint = _resolve_tint(
            args.halation_tint, photos.halation_tint, "--halation-tint", 0.0, 1.0
        )
        curve_l = _resolve_curve(args.curve_l, photos.curve_l, "--curve-l")
        curve_r = _resolve_curve(args.curve_r, photos.curve_r, "--curve-r")
        curve_g = _resolve_curve(args.curve_g, photos.curve_g, "--curve-g")
        curve_b = _resolve_curve(args.curve_b, photos.curve_b, "--curve-b")
        mono_mix = _resolve_mono(args.bw_mix, photos.mono_mix, "--bw-mix", 3, -2.0, 3.0)
        if mono_mix and sum(mono_mix) <= 0.0:
            raise ValueError("--bw-mix weights must sum to more than 0")
        duotone = _resolve_mono(args.duotone, photos.duotone, "--duotone", 6, 0.0, 1.0)
        duotone_strength = (
            args.duotone_strength if args.duotone_strength is not None else photos.duotone_strength
        )
        if not 0.0 <= duotone_strength <= 1.0:
            raise ValueError(f"--duotone-strength must be in [0, 1] (got {duotone_strength})")
        duotone_mid = _resolve_mono(args.duotone_mid, photos.duotone_mid, "--duotone-mid", 3, 0, 1)
        if duotone_mid and not duotone:
            raise ValueError("--duotone-mid needs --duotone (the ramp's two ends)")
        # Masked colour layers, from --layers or the look's `layers:`. Re-gated
        # here because the FLAG path never passed the config validator; a look's
        # own list was already refused at load, naming the look.
        check_layers("--layers", args.layers if args.layers is not None else [])
        layers = tuple(args.layers or ())
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1
    # The hero frame every styled variant is matched to. Decoded and MEASURED
    # HERE, at the door, so an unreadable reference fails naming the path before
    # a single source photo is decoded — never once per photo, mid-render. What
    # travels onward is six Lab numbers, so no render re-reads the file.
    reference = args.reference if args.reference is not None else photos.reference
    reference_lab = ()
    if reference:
        try:
            reference_lab = reference_lab_from_file(Path(reference).expanduser())
        except (PhotoDecodeError, OSError, ValueError) as e:
            print(f"--reference unusable: {e}", file=sys.stderr)
            return 1
    if styles and (unknown := [s for s in styles if s not in STYLES]):
        avail = ", ".join(STYLES)
        print(f"Unknown style(s): {', '.join(unknown)} (available: {avail})", file=sys.stderr)
        return 1
    # The style graded UNDER every --lut look ("" = the neutral base grade).
    lut_over = args.lut_over if args.lut_over is not None else photos.lut_over
    if lut_over and lut_over not in STYLES:
        avail = ", ".join(STYLES)
        print(f"Unknown --lut-over style: {lut_over} (available: {avail})", file=sys.stderr)
        return 1

    # LUT looks ADD to the style variants — combine with e.g. --styles natural
    # for a LUT-only develop. Resolved once; a bad .cube fails the whole run.
    # --lut-auto turns that resolved list into the CANDIDATE SET scored per photo:
    # the operator's own --lut specs when any were named, else the whole library.
    lut_auto = args.lut_auto if args.lut_auto is not None else photos.lut_auto
    specs = getattr(args, "lut", None) or (sorted(photos.luts) if lut_auto else [])
    try:
        luts = _resolve_luts(specs, photos.luts)
    except CubeError as e:
        print(f"Cannot load LUT: {e}", file=sys.stderr)
        return 1
    if not 0 <= lut_auto <= len(luts):
        msg = f"--lut-auto must be in [0, {len(luts)}] — the candidate LUT count"
        print(f"{msg} (--lut specs, else photos.luts) — got {lut_auto}", file=sys.stderr)
        return 1

    # Every knob above is resolved and range-gated by now; pack them into the
    # one record the develop surface takes. Packing only — no logic lives here.
    options = DevelopOptions(
        vignette=vignette,
        dehaze=dehaze,
        sky_darken=sky_darken,
        sky_saturate=sky_saturate,
        graduated_nd=graduated_nd,
        graduated_nd_angle=graduated_nd_angle,
        glow=glow,
        clarity=clarity,
        grain=grain,
        grain_size=grain_size,
        halation=halation,
        halation_tint=halation_tint,
        light_leak=light_leak,
        chromatic_ab=chromatic_ab,
        holo_sheen=holo_sheen,
        scanline=scanline,
        glitch=glitch,
        sharpen=sharpen,
        hue_shift=hue_shift,
        hue_sat=hue_sat,
        hue_lum=hue_lum,
        splittone_shadow=splittone_shadow,
        splittone_midtone=splittone_midtone,
        splittone_highlight=splittone_highlight,
        curve_l=curve_l,
        curve_r=curve_r,
        curve_g=curve_g,
        curve_b=curve_b,
        **white_balance,
        straighten_angle=straighten_angle,
        crop_rect=crop_rect,
        lens_correct=lens_correct,
        vibrance=vibrance,
        mono_mix=mono_mix,
        duotone=duotone,
        duotone_strength=duotone_strength,
        duotone_mid=duotone_mid,
        reference_lab=reference_lab,
        layers=layers,
        look_strength=look_strength,
        lut_strength=lut_strength,
        lut_over=lut_over,
        lut_auto=lut_auto,
    )

    # --look-variant NAME (repeatable): each named look becomes its OWN panel over
    # the natural base grade, so two candidates land on one contact sheet. It never
    # touches the run's knobs — each panel rides the record resolved above with that
    # look's on top. Unknown name AND malformed knob are refused HERE, pre-decode.
    # A look's `border` is an EXPORT knob, not a develop one, so it is no
    # DevelopOptions field and cannot ride the record above — it is kept HERE
    # instead, keyed by the panel name the export loop iterates, so a look whose
    # matte IS the look (`polaroid`) renders with it. A look naming none is absent
    # from the map and falls back to the run's border, as every other variant does.
    # `look_options` is the same panel-name map over the record each panel actually
    # RENDERS through, so the sidecar below records that record rather than the run's.
    # `look_styles` is the third such map: a look's base `styles` is no DevelopOptions
    # field either, and the colour knobs are authored to ride THAT base — dropping it
    # rendered `--look-variant NAME` as a different picture from `--look NAME`.
    look_variants = []
    look_borders = {}
    look_options = {}
    look_styles = {}
    for name in args.look_variant or []:
        if name not in known:
            refusal = _unknown_look("--look-variant", name, BUILTIN_LOOKS, photos.looks)
            print(refusal, file=sys.stderr)
            return 1
        if (style := known[name].get("border", border)) not in BORDER_STYLES:
            print(_bad_border(f"--look-variant {name}", style, BORDER_STYLES), file=sys.stderr)
            return 1
        look_borders[f"{LOOK_VARIANT_PREFIX}{name}"] = style
        # ONE panel is ONE picture, so its base is ONE known style — refused HERE,
        # pre-decode, where the run's own --styles list is refused, rather than
        # inside apply_style once per photo, mid-batch.
        base = [s.strip() for s in str(known[name].get("styles", "")).split(",") if s.strip()]
        if len(base) > 1 or any(s not in STYLES for s in base):
            got = known[name]["styles"]
            avail = ", ".join(STYLES)
            msg = f"--look-variant {name}: base styles must name exactly one known style"
            print(f"{msg} (got {got!r}; available: {avail})", file=sys.stderr)
            return 1
        look_styles[name] = base[0] if base else ""
        try:
            # The variant flag skips every resolver above, so its lists are gated
            # HERE, through the one table config load uses — otherwise a band
            # --look refuses renders through --look-variant.
            check_look_lists(f"look '{name}'", known[name])
            resolved_look = look_variant_options(known[name], options)
        except ValueError as e:
            print(f"--look-variant {name} unusable: {e}", file=sys.stderr)
            return 1
        look_variants.append((name, resolved_look))
        look_options[f"{LOOK_VARIANT_PREFIX}{name}"] = resolved_look

    # --from-grade PATH.grade.json: REPLAY an approved grade instead of measuring a
    # new one. The whole recorded record is restored over the run's — colour params
    # and spatial knobs alike, since a replay that dropped them would hand back
    # different pixels than the proof — and exactly ONE variant renders, under its
    # recorded name. Read and refused HERE, at the door, before any decode.
    replay = None
    if args.from_grade:
        try:
            variant, grade_params, recorded = _load_grade_sidecar(
                Path(args.from_grade).expanduser()
            )
        except ValueError as e:
            print(f"--from-grade unusable: {e}", file=sys.stderr)
            return 1
        if luts or look_variants:
            print(
                "--from-grade renders exactly the recorded variant; drop --lut/--lut-auto/"
                "--look-variant, whose extra panels would ride the MEASURED base instead",
                file=sys.stderr,
            )
            return 1
        options = _replayed_options(recorded, options, args)
        replay = (variant, grade_params)

    paths = [Path(p) for p in args.paths]
    if bad := [str(p) for p in paths if p.suffix.lower() not in SUPPORTED_EXTENSIONS]:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        print(
            f"Unsupported extension(s): {', '.join(bad)} (supported: {supported})", file=sys.stderr
        )
        return 1

    # --hdr: each detected bracket becomes ONE fused source and its members are CONSUMED —
    # they develop once, fused, never also singly. Grouping reads EXIF only, so it and the
    # refusal both land before a single file is decoded.
    sources = list(paths)
    if hdr:
        if len(paths) < MIN_BRACKET_FRAMES:
            msg = f"--hdr needs at least {MIN_BRACKET_FRAMES} photos to find a bracket in"
            print(f"{msg} (got {len(paths)})", file=sys.stderr)
            return 1
        groups = detect_brackets(paths)
        consumed = {p for group in groups for p in group}
        sources = [p for p in paths if p not in consumed] + list(groups)
        print(f"HDR: fusing {len(groups)} bracket(s) from {len(consumed)} of {len(paths)} photo(s)")

    out_dir = None
    if not args.dry_run:
        out = args.out or photos.developed_output_dir
        if not out:
            print("--out is required (photos.developed_output_dir is unset)", file=sys.stderr)
            return 1
        out_dir = Path(out).expanduser()

    # The sheet's home is config, never ``--out``: see PhotosConfig.contact_sheet_root
    # on why a review artifact must not follow the masters into a scratch dir. The
    # location check is an UPSTREAM gate — refusing here costs nothing, refusing after
    # the render costs the whole develop and still has nowhere to put the sheet.
    # An explicitly configured root must already have its parent (an unmounted
    # review drive is refused); the default under the data home is created.
    sheet_dir = None
    if args.contact_sheet and not args.dry_run:
        root = photos.resolve_contact_sheet_root()
        if photos.contact_sheet_root and not root.parent.is_dir():
            print(
                f"--contact-sheet needs {root.parent} (photos.contact_sheet_root = {root}); "
                "that directory does not exist (is the review drive mounted?)",
                file=sys.stderr,
            )
            return 1
        sheet_dir = root / out_dir.name
        sheet_dir.mkdir(parents=True, exist_ok=True)

    failed = 0
    for item in sources:
        # A bracket arrives as a tuple and fuses inside the per-photo try below, so one
        # unfusable bracket fails like one bad file rather than failing the whole run.
        group = item if isinstance(item, tuple) else ()
        path = group[0] if group else item
        stem = f"{path.stem}{HDR_STEM_SUFFIX}" if group else path.stem
        try:
            source = fuse_bracket(group) if group else path
            # What the photo is OF, read off EXIF/grouping/the operator sidecar
            # for THIS path. Only the auto-select path (no explicit ``styles``)
            # consults it, same as ``select_styles`` itself.
            #
            # Two evidence sources ``recognise_subject`` accepts are NOT supplied
            # here, so say so rather than let the weaker reading pass for a full
            # one: ``scene`` (the measured SceneProfile) does not exist yet --
            # ``plan_variants`` computes it after the decode, below -- so the
            # night test falls back to camera-local wall clock; and
            # ``session_astro_frames`` stays 1 because this loop develops one
            # path at a time and never sees the card, so a tracked sky sequence
            # earns no CONFIDENCE_SEQUENCE_BONUS here. A reading that clears the
            # routing floor on EXIF alone still routes; one that needed either of
            # those to clear it simply does not, and the ladder stays unchanged.
            # ``recognise_photos`` is the whole-card entry point that has both.
            subject = (
                recognise_subject(
                    exif=extract_exif_capture_metadata(path),
                    optics=read_capture_optics(path),
                    in_exposure_bracket=bool(group),
                    correction=read_subject_correction(path),
                )
                if not styles
                else None
            )
            if args.dry_run:
                plan = plan_variants(
                    source,
                    styles,
                    strength,
                    luts,
                    options,
                    look_variants,
                    look_styles,
                    replay,
                    subject=subject,
                )
                _print_variant_plan(path, plan, strength)
            else:
                panels = []
                write, ext = (export_tiff16, "tif") if depth == 16 else (export_jpeg, "jpg")
                # fmt: off
                variants = develop_variants(
                    source, styles, strength, luts, options, depth, look_variants,
                    look_styles, replay, subject=subject,
                )
                # fmt: on
                for name, developed in variants:
                    bordered = _add_border(developed.rgb, look_borders.get(name, border))
                    master = write(bordered, out_dir / f"{stem}_{name}.{ext}")
                    print(f"  + {master}")
                    if grade_sidecar:
                        # The record THIS panel rendered through: a look panel's own,
                        # every other variant's the run's. Both the recorded knobs and
                        # the stamped cache key come off it, so a pair of looks
                        # separated by a spatial knob alone cannot share either.
                        write_grade_sidecar(
                            developed,
                            master,
                            name,
                            strength,
                            look_options.get(name, options),
                            luts,
                            depth=depth,
                        )
                    if args.contact_sheet:
                        # The sheet is an 8-bit review JPEG, so a 16-bit master is
                        # downconverted into it — never the other way round.
                        panels.append((name, _analysis_rgb(developed.rgb)))
                # Only reached when every variant rendered: a failed variant
                # raises above, so a partial photo never gets a sheet.
                if args.contact_sheet:
                    sheet = compose_contact_sheet(panels, cols=args.contact_cols)
                    print(f"  + {export_jpeg(sheet, sheet_dir / f'{stem}_contact.jpg')}")
        except (PhotoDecodeError, ValueError, OSError) as e:
            failed += 1
            print(f"  ! {path}: {e}", file=sys.stderr)
    if failed:
        print(f"{failed} of {len(sources)} photo(s) failed to develop", file=sys.stderr)
    return 1 if failed else 0


def _cmd_photos_reel(args) -> int:
    """Stills -> one vertical Ken Burns MP4, cut (never crossfaded) between them.

    A local working render like ``photos develop``: given files in, one video
    out, touching neither the develop/grade chain nor the publish path.
    """
    from ...pipeline.photos.reel import REEL_SECONDS_PER_STILL, ReelError, render_reel

    seconds = args.seconds if args.seconds is not None else REEL_SECONDS_PER_STILL
    if not (math.isfinite(seconds) and seconds > 0):
        print(f"--seconds must be a positive number (got {seconds})", file=sys.stderr)
        return 1
    try:
        out = render_reel(args.paths, args.out, seconds=seconds)
    except ReelError as e:
        print(f"Reel failed: {e}", file=sys.stderr)
        return 1
    print(f"Rendered {len(args.paths)} still(s) x {seconds}s -> {out}")
    return 0


def _cmd_photos_daylight_timelapse(args) -> int:
    """Daylight stills -> one MP4, ordered by capture order. The generic sibling
    of ``photos astro --timelapse``: no dark/hot-pixel calibration, no
    shared-tone develop, just decode-scale-encode over ``pipeline.timelapse``.
    """
    from ...pipeline.timelapse import TIMELAPSE_FPS, TimelapseError, render_stills_timelapse

    fps = args.fps if args.fps is not None else TIMELAPSE_FPS
    if fps <= 0:
        print(f"--fps must be positive (got {args.fps})", file=sys.stderr)
        return 1
    height = args.height
    if height is not None and height < 0:
        print(f"--height must be positive (got {args.height})", file=sys.stderr)
        return 1
    try:
        out, used, rejected = render_stills_timelapse(
            args.paths, args.out, fps=fps, scale_height=height
        )
    except TimelapseError as e:
        print(f"Timelapse failed: {e}", file=sys.stderr)
        return 1
    print(f"{used} still(s) at {fps} fps -> {out} ({len(rejected)} rejected)")
    return 0


def _astro_frame_megapixels(path) -> float:
    """One frame's sensor megapixels from LibRaw's header alone (``raw.sizes``) --
    no ``unpack()``/``postprocess()``, so this never decodes anything.

    Raises:
        AstroCalibrationError: ``path`` can't be opened as RAW at all -- e.g.
            a non-RAW file (a dual-format JPEG) that slipped past
            ``detect_sequences`` and landed first in the sorted sequence.
    """
    import rawpy

    from ...pipeline.photos.astro_calibrate import AstroCalibrationError

    try:
        with rawpy.imread(str(path)) as raw:
            sizes = raw.sizes
    except rawpy.LibRawError as exc:
        raise AstroCalibrationError(f"{path}: not a readable RAW frame ({exc})") from exc
    return (int(sizes.width) * int(sizes.height)) / 1_000_000.0


def _astro_frame_shape(path) -> tuple[int, int]:
    """One frame's decoded ``(height, width)`` from LibRaw's header alone (``raw.sizes``) --
    the exact same cheap header peek ``_astro_frame_megapixels`` uses, so building the
    operator's ``--horizon`` mask never decodes (``postprocess``es) an extra frame.

    ``raw.sizes.height``/``.width`` are the *visible sensor* size, before the camera's
    orientation is applied. ``prepare_frames``/triage measure against ``postprocess()``
    output, and ``user_flip`` is left at its default there too (see develop.py's
    ``_decode_raw``), so LibRaw applies the camera flip itself -- a quarter-turn flip
    (5/6, see develop.py's ``_ROT90_BY_FLIP``) makes the decoded array the transpose of
    the visible-sensor size. Skipping this on a portrait sequence hands triage a mask
    of the wrong shape and it crashes with an IndexError instead of stacking.

    Raises:
        AstroCalibrationError: ``path`` can't be opened as RAW at all.
    """
    import rawpy

    from ...pipeline.photos.astro_calibrate import AstroCalibrationError

    try:
        with rawpy.imread(str(path)) as raw:
            sizes = raw.sizes
    except rawpy.LibRawError as exc:
        raise AstroCalibrationError(f"{path}: not a readable RAW frame ({exc})") from exc
    hw = (int(sizes.height), int(sizes.width))
    return hw[::-1] if int(getattr(sizes, "flip", 0)) in (5, 6) else hw


def _diagnose_no_sequence(
    paths, max_frame_gap_s, max_chunk_span_s, min_sequence_frames, detect_sequences
) -> str:
    """Why ``detect_sequences`` found nothing: probes both gates maximally relaxed,
    then names every real gate still blocking, span guidance included."""
    from ...pipeline.photos.exif_gps import extract_exif_capture_timestamp

    seq_logger = logging.getLogger("gomoveshift.pipeline.photos.astro_sequence")
    previous_level = seq_logger.level
    seq_logger.setLevel(logging.ERROR)  # this probe is always "sparse" by construction
    try:
        relaxed = detect_sequences(paths, max_frame_gap_s=float("inf"), min_sequence_frames=1)
    finally:
        seq_logger.setLevel(previous_level)
    if not relaxed:
        return "no frame has a readable capture time/ISO/exposure to group"
    longest = max(relaxed, key=len)
    times = sorted(t for t in (extract_exif_capture_timestamp(p) for p in longest) if t)
    widest_gap = (
        max((b - a).total_seconds() for a, b in zip(times, times[1:])) if len(times) > 1 else 0.0
    )
    clauses = []
    if widest_gap > max_frame_gap_s:
        full_span = (times[-1] - times[0]).total_seconds() if len(times) > 1 else widest_gap
        span = f" --max-chunk-span {full_span:.1f}" if full_span > max_chunk_span_s else ""
        clauses.append(
            f"--max-frame-gap {max_frame_gap_s}s is the blocking gate -- the widest gap "
            f"between consecutive frames is {widest_gap:.1f}s; pass --max-frame-gap "
            f"{widest_gap:.1f}{span} to admit all {len(longest)} frames as one chunk"
        )
    if len(longest) < min_sequence_frames:
        clauses.append(
            f"--min-sequence-frames {min_sequence_frames} is the blocking gate -- the longest "
            f"run at this --max-frame-gap is only {len(longest)} frame(s); pass "
            f"--min-sequence-frames {len(longest)} or fewer to admit it"
        )
    return " -- and ".join(clauses)


def _astro_output_name(chunk, dark_paths, horizon, prefix="astro_stack", suffix=".tif") -> str:
    """One chunk's deterministic master filename: its own first + last source
    frame identity, plus a short fingerprint of the render-affecting knobs
    (``--horizon``, ``--darks``); ``prefix`` names the render, so an ``astro_trail``
    never overwrites a stack, ``suffix`` its container. NOT the invocation-local ordinal
    -- that ordinal says nothing about which frames are inside, so re-running a
    night (or a night whose chunk boundaries shifted under a different
    --stack-len/--max-frame-gap/--max-chunk-span) silently ``os.replace``d an
    already-stacked (~1 min/frame) master under the same name. Two runs over
    the SAME frames with the SAME render knobs collide on purpose: that's what
    lets the caller refuse an accidental re-run instead of losing the render.
    """
    import hashlib

    first, last = Path(chunk[0]).stem, Path(chunk[-1]).stem
    # The readable stems above are for humans only -- two different cards can
    # share a bare filename (a common camera default), so the identity that
    # actually decides collision/no-collision is keyed on the frames' own
    # resolved paths, never the stem alone.
    fingerprint_src = (
        f"first={Path(chunk[0]).resolve()}|last={Path(chunk[-1]).resolve()}|n={len(chunk)}|"
        f"horizon={horizon}|darks={','.join(sorted(str(Path(d).resolve()) for d in dark_paths))}"
    )
    fingerprint = hashlib.sha256(fingerprint_src.encode()).hexdigest()[:8]
    return f"{prefix}_{first}_{last}_{fingerprint}{suffix}"


def _libraw_cannot_unpack(path) -> bool:
    """True when LibRaw opens ``path`` but cannot unpack its raw stream -- the Nikon High
    Efficiency (HE/HE*) TicoRAW signature, and the only condition the gate may refuse on.
    The caller probes ONE candidate, not all, because a card comes off one body at one
    setting: this establishes "this card's frames do not unpack", never "each listed file
    does not". Opening alone proves nothing (LibRaw parses an HE header fine -- it is why
    the megapixel peek above works on one), so the probe reaches ``unpack`` and stops
    there, no postprocess. A file that will not open at all, or fails any other way, is
    NOT reported -- that open failure is the decode path's to report, not a preflight's."""
    import rawpy

    try:
        raw_ctx = rawpy.imread(str(path))
    except Exception:
        return False
    try:
        with raw_ctx as raw:
            raw.unpack()
    except rawpy.LibRawError:
        return True
    except Exception:
        return False
    return False


def _astro_dark_identity_refusals(light_paths, dark_paths) -> list[str]:
    """Every darks-vs-lights identity incompatibility across the WHOLE selected input
    set, not just one representative frame -- a settings change mid-night (ISO bumped
    after chunk 3) must not pass because chunk 1's frame matched the darks. Returns
    every distinct refusal message (stable order), or ``[]`` when every light/dark
    combination checked is usable.

    Read off the calibrate module (not imported by name) so the identity rule stays in
    one place: this preflight and the per-chunk master-dark build must never disagree
    about what a usable dark is. A file that cannot be opened is skipped -- an
    unreadable calibration frame is the decode path's failure to report, exactly as the
    HE probe above treats an unopenable light, and refusing a whole night over one
    unreadable file is worse than the bug.

    Lights are de-duplicated by their read identity (camera/shape/cfa/iso/exposure)
    before comparing against darks, so a big night costs a handful of dark checks per
    distinct settings group, not ``len(lights) * len(darks)`` full comparisons -- both
    reads are metadata-only, but there is no reason to repeat an identical check.
    """
    from ...pipeline.photos import astro_calibrate as calibrate

    distinct_lights: dict[tuple, object] = {}
    for light_path in light_paths:
        try:
            light = calibrate.read_capture_identity(light_path)
        except calibrate.AstroCalibrationError:
            continue
        key = (light.camera, light.raw_shape, light.cfa_pattern, light.iso, light.exposure_s)
        distinct_lights.setdefault(key, light)

    refusals: list[str] = []
    seen_messages: set[str] = set()
    for light in distinct_lights.values():
        for dark_path in dark_paths:
            try:
                dark = calibrate.read_capture_identity(dark_path)
                calibrate.validate_dark_identity(light, dark)
            except calibrate.AstroDarkIdentityMismatchError as exc:
                message = str(exc)
                if message not in seen_messages:
                    seen_messages.add(message)
                    refusals.append(message)
            except calibrate.AstroCalibrationError:
                continue
    return refusals


ASTRO_SECONDS_PER_FRAME_ESTIMATE = 60.0
"""Approximate single-worker decode+stack throughput
(``astro_orchestrator.py``'s own "~1 min/frame" figure, used there to warn about
run duration) -- the basis for the first-artifact deadline estimate below."""


def _astro_first_artifact_deadline_s(chunks, decode_workers: int) -> float:
    """Seconds until the FIRST chunk's master should exist, from the ~1
    min/frame single-worker throughput above, divided across ``decode_workers``.

    A supervisor watching a long run can treat materially exceeding this deadline
    with no artifact yet as a stall, instead of waiting blind for the whole night
    (a runtime estimate from typical throughput, revised when a first-output
    milestone is missed). ``0.0`` when there is
    nothing to stack.
    """
    if not chunks:
        return 0.0
    first_chunk_frames = len(chunks[0])
    return (first_chunk_frames * ASTRO_SECONDS_PER_FRAME_ESTIMATE) / max(decode_workers, 1)


def _astro_ram_fits(frame_count: int, megapixels: float, dark_count: int) -> bool:
    """Whether one chunk of ``frame_count`` frames fits the RAM free RIGHT NOW.

    Judged with the SAME ``estimate_astro_stack_ram_gb`` + ``check_ram`` pair
    ``preflight_astro_stack`` itself uses (both read off the preflight module, so a
    monkeypatch or a future swap moves them together): an answer here can never
    disagree with the refusal that prompted it."""
    from ...pipeline.photos.preflight import check_ram, estimate_astro_stack_ram_gb

    try:
        check_ram(
            min_free_gb=estimate_astro_stack_ram_gb(frame_count, megapixels, dark_count=dark_count)
        )
    except RuntimeError:
        return False
    return True


def _astro_stack_len_advice(runs, *, stack_len, max_span_s, megapixels, dark_count) -> str:
    """The operator's next command after an up-front RAM refusal: the largest
    ``--stack-len`` below the refused one whose every chunk fits the RAM free right now.

    Each candidate is RE-CHUNKED rather than taken at face value, because
    ``chunk_sequence`` merges a trailing chunk shorter than its ``MIN_SEQUENCE_FRAMES``
    floor into the predecessor -- so a candidate's real chunks can be LONGER than the
    candidate itself (on a 15-frame run every ``--stack-len`` from 11 to 15 collapses to
    one 15-frame chunk) and a value read straight off the estimator could be one that
    still does not fit."""
    from ...pipeline.photos.astro_sequence import MIN_SEQUENCE_FRAMES, chunk_sequence

    for candidate in range(stack_len - 1, MIN_SEQUENCE_FRAMES - 1, -1):
        chunks = [c for run in runs for c in chunk_sequence(run, candidate, max_span_s)]
        if chunks and all(_astro_ram_fits(len(c), megapixels, dark_count) for c in chunks):
            return (
                f"retry with --stack-len {candidate} -- the largest whose every chunk fits "
                "the RAM free right now (chunks re-derived at that value, so a merged "
                "trailing chunk is counted at its real length)"
            )
    return (
        f"no --stack-len fits: even a {MIN_SEQUENCE_FRAMES}-frame chunk (the chunker's own "
        "floor) needs more RAM than is free right now -- free memory, or stack fewer frames"
    )


#: ``astro_triage.classify_shape`` verdicts worth stopping a night for, each with the
#: action it implies. ``indeterminate`` is deliberately absent: it is the normal case,
#: and a line on every healthy run is noise the one real warning would be lost in.
_ASTRO_SHAPE_ADVICE = {
    "likely_defocus": "stars are round and fat -- REFOCUS before shooting more",
    "likely_trailing": "stars are elongated -- check tracking/mount before shooting more",
    "corner_aberration": "corners are soft against a sharp centre -- check the lens/field",
}


def _astro_shape_note(metrics):
    """One line for a chunk's dominant ``classify_shape`` verdict, or ``None`` when
    nothing stands out (no majority, or the majority is ``indeterminate``). The
    point of printing it: a card with round, fat stars (high ``fwhm_px`` at low
    ``ellipticity``) reads ``likely_defocus`` from the first frame, whereas without
    this the operator learns the cause only after the whole night failed to
    register. A REPORT, never a gate: ``classify_shape``
    touches no ``usable``/cloud/dew flag, so a deliberately defocused frame still stacks."""
    from ...pipeline.photos.astro_triage import classify_shape

    counts = Counter(classify_shape(m) for m in metrics)
    total = sum(counts.values())
    ranked = [(n, verdict) for verdict, n in counts.items() if verdict in _ASTRO_SHAPE_ADVICE]
    if not ranked:
        return None
    count, verdict = max(ranked)
    if count * 2 <= total:
        return None  # one odd frame in a run is not the night's diagnosis
    return (
        f"** {count}/{total} frame(s) read {verdict}: {_ASTRO_SHAPE_ADVICE[verdict]} "
        "(shape thresholds are PROVISIONAL; this is a report, not a gate)"
    )


def _expand_raw_paths(raw_args, label: str) -> list[str]:
    """Expand any directory argument to the RAW files it contains (sorted by
    name), leaving a file argument untouched -- a mix of files and directories
    in one invocation is fine. A directory that expands to nothing is a hard
    error naming that directory, never a silent empty/thinner run."""
    from ...pipeline.photos.develop import RAW_EXTENSIONS

    expanded: list[str] = []
    for raw in raw_args:
        path = Path(raw)
        if path.is_dir():
            found = sorted(
                f for f in path.iterdir() if f.is_file() and f.suffix.lower() in RAW_EXTENSIONS
            )
            if not found:
                exts = ", ".join(sorted(RAW_EXTENSIONS))
                raise ValueError(f"{label} directory {path} contains no RAW files ({exts})")
            expanded.extend(str(f) for f in found)
        else:
            expanded.append(raw)
    return expanded


def _contact_sheet_path(photos, key: str) -> Path | None:
    root = photos.resolve_contact_sheet_root()
    if photos.contact_sheet_root and not root.parent.is_dir():
        print(f"--contact-sheet needs {root.parent} to exist", file=sys.stderr)
        return None
    sheet_dir = root / key
    sheet_dir.mkdir(parents=True, exist_ok=True)
    return sheet_dir / f"{key}_contact.jpg"


def _cmd_photos_astro(args) -> int:
    """RAW night frames -> stacked masters: group into intervalometer runs, chunk,
    gate resources, then calibrate/register/stack per chunk. Pool-independent like
    ``develop``/``reel``; ``photos ingest`` applies the usual blur/GPS/location
    gates on the exported masters afterward."""
    from ...config import get_config
    from ...pipeline.photos import astro_timelapse as tlapse
    from ...pipeline.photos.astro_calibrate import AstroCalibrationError, AstroHighEfficiencyError
    from ...pipeline.photos.astro_contact import ContactSheetError, master_panels, stack_panels
    from ...pipeline.photos.astro_orchestrator import (
        DECODE_WORKERS_DEFAULT,
        AstroOrchestratorError,
        prepare_frames,
        stack_run,
    )
    from ...pipeline.photos.astro_sequence import (
        MAX_CHUNK_SPAN_S,
        MAX_FRAME_GAP_S,
        MIN_SEQUENCE_FRAMES,
        STACK_LEN_DEFAULT,
        chunk_sequence,
        detect_sequences,
    )
    from ...pipeline.photos.astro_stack import horizon_mask
    from ...pipeline.photos.astro_trails import preflight_trail, render_trail
    from ...pipeline.photos.astro_triage import update_run_cloud_baseline
    from ...pipeline.photos.develop import (
        HE_DNG_CONVERTER_DEFAULT_PATHS,
        compose_contact_sheet,
        export_jpeg,
        he_dng_inputs_needing_conversion,
        resolve_he_dng_converter,
    )
    from ...pipeline.photos.preflight import preflight_astro_run_disk, preflight_astro_stack

    photos = get_config().photos

    # --from-master: skip stacking (~1 min/frame) entirely.
    if args.from_master:
        if not args.contact_sheet:
            print("--from-master needs --contact-sheet", file=sys.stderr)
            return 1
        sheet_path = _contact_sheet_path(photos, Path(args.output_dir).name)
        if sheet_path is None:
            return 1
        try:
            panels = master_panels(args.from_master)
        except ContactSheetError as e:
            print(str(e), file=sys.stderr)
            return 1
        sheet = compose_contact_sheet(panels)
        print(f"contact sheet -> {export_jpeg(sheet, sheet_path)}")
        return 0
    if not args.paths:
        print("photos astro needs light frame paths, or --from-master PATH", file=sys.stderr)
        return 1
    try:
        args.paths = _expand_raw_paths(args.paths, "photos astro")
        args.darks = _expand_raw_paths(args.darks, "--darks")
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1
    if args.contact_sheet and (args.trails or args.timelapse or args.triage_only):
        mode = "--trails" if args.trails else "--timelapse" if args.timelapse else "--triage-only"
        print(
            f"--contact-sheet needs an actual stacked master; {mode} never produces "
            "one -- drop it, or use --from-master against an existing stack",
            file=sys.stderr,
        )
        return 1
    keep_linear = args.keep_linear or args.linear_only
    if keep_linear and (args.trails or args.timelapse or args.triage_only):
        mode = "--trails" if args.trails else "--timelapse" if args.timelapse else "--triage-only"
        print(
            f"--keep-linear/--linear-only persist registered stack components; {mode} "
            "does not produce that stack",
            file=sys.stderr,
        )
        return 1
    if args.contact_sheet and args.linear_only:
        print("--contact-sheet needs a developed TIFF; --linear-only skips it", file=sys.stderr)
        return 1

    # Upstream gate, before any file is even grouped into a run: an out-of-range
    # --horizon is the same bad value on every chunk, so it is refused once, here,
    # rather than once per chunk mid-loop.
    if args.horizon is not None and not 0.0 <= args.horizon <= 1.0:
        print(f"--horizon must be in [0, 1] (got {args.horizon})", file=sys.stderr)
        return 1

    # Same upstream posture as --horizon above: a bad rate is the same bad rate on every
    # chunk, so it is refused once here rather than after a chunk's ~1 min/frame decode.
    timelapse_fps = args.timelapse_fps if args.timelapse_fps is not None else tlapse.TIMELAPSE_FPS
    if timelapse_fps <= 0:
        print(f"--timelapse-fps must be positive (got {args.timelapse_fps})", file=sys.stderr)
        return 1

    # Refused here for the same reason as the rate: a bad height is bad on every chunk, and
    # this one would otherwise surface inside cv2.resize mid-render, hours in. 0 is the
    # documented "no cap" spelling, not a zero-pixel render, so it passes through as None --
    # the value the render already reads as uncapped; only a NEGATIVE height is nonsense.
    timelapse_height = args.timelapse_height
    if timelapse_height is None:
        timelapse_height = tlapse.TIMELAPSE_HEIGHT_DEFAULT
    elif timelapse_height == 0:
        timelapse_height = None
    elif timelapse_height < 0:
        print(f"--timelapse-height must be positive (got {args.timelapse_height})", file=sys.stderr)
        return 1

    # Resolved ONCE and handed to both the preflight and prepare_frames: sizing the gate
    # for one concurrency and decoding at another is how a run OOMs past a gate that passed.
    decode_workers = (
        args.decode_workers if args.decode_workers is not None else DECODE_WORKERS_DEFAULT
    )
    if decode_workers < 1:
        print(f"--decode-workers must be >= 1 (got {args.decode_workers})", file=sys.stderr)
        return 1

    stack_len = args.stack_len if args.stack_len is not None else STACK_LEN_DEFAULT
    max_frame_gap_s = args.max_frame_gap if args.max_frame_gap is not None else MAX_FRAME_GAP_S
    max_chunk_span_s = args.max_chunk_span if args.max_chunk_span is not None else MAX_CHUNK_SPAN_S
    min_sequence_frames = (
        args.min_sequence_frames if args.min_sequence_frames is not None else MIN_SEQUENCE_FRAMES
    )
    for flag, value in (
        ("--max-frame-gap", max_frame_gap_s),
        ("--max-chunk-span", max_chunk_span_s),
    ):
        if not (math.isfinite(value) and value > 0):
            print(
                f"{flag} must be a positive, finite number of seconds (got {value})",
                file=sys.stderr,
            )
            return 1
    if min_sequence_frames < 1:
        print(f"--min-sequence-frames must be >= 1 (got {min_sequence_frames})", file=sys.stderr)
        return 1
    if max_chunk_span_s < max_frame_gap_s:
        print(
            f"--max-chunk-span {max_chunk_span_s}s must be >= --max-frame-gap {max_frame_gap_s}s "
            "-- a chunk span below the frame gap can never hold two frames of a sparse-cadence run",
            file=sys.stderr,
        )
        return 1

    # Upstream gate: the converter's availability used to be discovered lazily, once per
    # frame, at decode time -- a card of HE/HE* NEFs with no converter warned per frame,
    # dropped every one, and ended in a generic "no chunk produced a stacked master"
    # with the decode work already spent. Resolve it ONCE here, before any frame decodes,
    # via develop's own rule (shared, so gate and decode path cannot disagree). Only when
    # nothing resolves is the probe's I/O paid, and the refusal rides LibRaw's verdict,
    # never a suffix. --triage-only is gated identically; the per-frame warning stays the
    # backstop for a converter that vanishes mid-run.
    needs_conversion = he_dng_inputs_needing_conversion([*args.paths, *args.darks])
    if needs_conversion:
        converter_exe = resolve_he_dng_converter()
        if converter_exe:
            print(
                f"HE->DNG converter: {converter_exe} ({len(needs_conversion)} input(s) to convert)"
            )
        elif _libraw_cannot_unpack(needs_conversion[0]):
            probed = ", ".join(HE_DNG_CONVERTER_DEFAULT_PATHS) or "(no default path known)"
            print(
                f"{len(needs_conversion)} input(s) need DNG conversion and no converter "
                "resolves -- LibRaw cannot unpack them, the signature of a Nikon High "
                "Efficiency (HE/HE*) TicoRAW NEF, so every frame would be dropped and the "
                "run would stack a thinner master or none. Set photos.he_dng_converter_exe, "
                f"or install Adobe DNG Converter at a probed default path ({probed})",
                file=sys.stderr,
            )
            return 1

    print(
        f"sequence gates: --max-frame-gap {max_frame_gap_s}s --max-chunk-span "
        f"{max_chunk_span_s}s --min-sequence-frames {min_sequence_frames}"
    )
    seq_logger = logging.getLogger("gomoveshift.pipeline.photos.astro_sequence")
    relay = logging.StreamHandler(sys.stderr)
    relay.setFormatter(logging.Formatter("%(message)s"))
    seq_logger.addHandler(relay)
    previous_propagate, seq_logger.propagate = seq_logger.propagate, False
    try:
        runs = detect_sequences(
            args.paths, max_frame_gap_s=max_frame_gap_s, min_sequence_frames=min_sequence_frames
        )
    finally:
        seq_logger.removeHandler(relay)
        seq_logger.propagate = previous_propagate
    if not runs:
        reason = _diagnose_no_sequence(
            args.paths, max_frame_gap_s, max_chunk_span_s, min_sequence_frames, detect_sequences
        )
        print(
            f"No intervalometer run found in {len(args.paths)} file(s) -- {reason}", file=sys.stderr
        )
        return 1
    whole_run = args.trails or args.timelapse
    if whole_run:
        # Frame-count-independent peak RAM in both, and splitting cuts arcs / a clip apart.
        chunks = [list(run) for run in runs]
    else:
        chunks = [
            c
            for run in runs
            for c in chunk_sequence(run, stack_len=stack_len, max_span_s=max_chunk_span_s)
        ]
    if not chunks:
        lengths = ", ".join(str(len(r)) for r in runs)
        print(
            f"Every detected run was too short/spread out to chunk -- run length(s) {lengths} "
            f"vs chunk_sequence's fixed {MIN_SEQUENCE_FRAMES}-frame floor at "
            f"--max-chunk-span {max_chunk_span_s}s",
            file=sys.stderr,
        )
        return 1

    output_dir = Path(args.output_dir)
    dark_paths = tuple(args.darks)
    try:
        megapixels = _astro_frame_megapixels(chunks[0][0])
    except (AstroCalibrationError, AstroHighEfficiencyError) as e:
        print(
            f"{e} -- drop non-RAW files (e.g. dual-format JPEGs) from the input paths",
            file=sys.stderr,
        )
        return 1

    # `chunks` is fully built above, so every target name is computable before
    # any chunk decodes -- an upstream gate, once here, instead of a per-chunk
    # check mid-loop that would let earlier chunks burn a real ~1 min/frame
    # decode+stack before a collision on a LATER chunk is ever noticed.
    # No-op for --triage-only, which never writes a master.
    mode = ("astro_timelapse", ".mp4") if args.timelapse else ("astro_stack", ".tif")
    if args.trails:
        mode = ("astro_trail", ".tif")
    targets = [output_dir / _astro_output_name(c, dark_paths, args.horizon, *mode) for c in chunks]
    linear_targets = [
        output_dir / _astro_output_name(c, dark_paths, args.horizon, "astro_stack", ".linear")
        for c in chunks
    ]
    if not args.triage_only:
        planned_targets = []
        if not args.linear_only:
            planned_targets.append(("master", targets))
        if keep_linear:
            planned_targets.append(("linear artifact", linear_targets))
        for kind, kind_targets in planned_targets:
            seen: dict[Path, int] = {}
            for i, target in enumerate(kind_targets, start=1):
                first_i = seen.setdefault(target, i)
                if first_i != i:
                    print(
                        f"chunk {i}/{len(chunks)} and chunk {first_i}/{len(chunks)} would both "
                        f"write {kind} {target} -- refusing regardless of --overwrite",
                        file=sys.stderr,
                    )
                    return 1
        existing_linear = [target for target in linear_targets if keep_linear and target.exists()]
        if existing_linear:
            for target in existing_linear:
                print(
                    f"refusing to replace existing linear artifact {target}; move or remove "
                    "the validated artifact explicitly before re-running",
                    file=sys.stderr,
                )
            return 1
        if not args.overwrite and not args.linear_only:
            existing = [t for t in targets if t.exists()]
            if existing:
                for t in existing:
                    print(
                        f"refusing to overwrite existing master {t} "
                        "-- pass --overwrite to replace it",
                        file=sys.stderr,
                    )
                return 1

    if keep_linear:
        try:
            preflight_astro_run_disk(
                output_dir,
                chunk_count=len(chunks),
                megapixels=megapixels,
                keep_linear=True,
                write_tiffs=not args.linear_only,
            )
        except RuntimeError as e:
            print(
                f"aggregate linear-artifact preflight refused before any decode -- {e}",
                file=sys.stderr,
            )
            return 1

    # Darks that cannot calibrate the lights refuse the run, so the run's only possible
    # outcome is already known -- and the per-chunk check discovers it one decoded chunk
    # at a time. On a long night that means decoding every chunk, refusing each one, and
    # writing nothing. Checked
    # against EVERY chunk's light, not just the first -- a settings change mid-night
    # (e.g. ISO bumped after chunk 3) must not slip through because chunk 1 matched.
    # Identity is metadata-only, so the whole answer costs a handful of cheap reads:
    # take it here, with the other before-any-decode gates, and list every
    # incompatibility in one refusal instead of failing hours in on a later chunk.
    if dark_paths:
        all_lights = [frame for chunk in chunks for frame in chunk]
        refusals = _astro_dark_identity_refusals(all_lights, dark_paths)
        if refusals:
            plural = "" if len(refusals) == 1 else "ies"
            singular = "y" if len(refusals) == 1 else ""
            print(
                f"darks refused before any decode -- {len(refusals)} "
                f"incompatibilit{singular}{plural}:",
                file=sys.stderr,
            )
            for message in refusals:
                print(f"  - {message}", file=sys.stderr)
            return 1

    # A supervisor watching this run needs a first-artifact deadline before it starts,
    # not only a final duration after the fact: materially exceeding this with no
    # artifact yet is a stall signal. --triage-only never writes a master, so it has no
    # first-artifact milestone to promise.
    if not args.triage_only:
        first_artifact_deadline_s = _astro_first_artifact_deadline_s(chunks, decode_workers)
        logger.info(
            "first artifact expected within ~%.0fs (chunk 1/%d, %d frame(s), %d decode worker(s))",
            first_artifact_deadline_s,
            len(chunks),
            len(chunks[0]),
            decode_workers,
        )
        print(
            f"first artifact expected within ~{first_artifact_deadline_s:.0f}s "
            f"(chunk 1/{len(chunks)}, {len(chunks[0])} frame(s), {decode_workers} "
            "decode worker(s))"
        )

    # Every chunk's size is known HERE, so the RAM the whole run needs is too -- gate it
    # once, before any frame decodes, the same upstream posture the name-collision pass
    # above and the --horizon check take. Gating per chunk instead would let early chunks
    # decode and stack, then refuse a later one for RAM and kill the run mid-loop. The LARGEST chunk answers
    # for all of them -- the estimate rises monotonically with frame count, so if it fits,
    # every shorter chunk does. --trails is exempt: its gate (preflight_trail) has no
    # frame-count term, so it already refuses on chunk 1 with nothing decoded, and it has
    # no --stack-len knob to advise on (chunk_sequence is skipped entirely).
    if not whole_run:
        largest = max(len(c) for c in chunks)
        try:
            preflight_astro_stack(
                output_dir,
                frame_count=largest,
                megapixels=megapixels,
                dark_count=len(dark_paths),
                decode_workers=decode_workers,
            )
        except RuntimeError as e:
            print(
                f"preflight refused before any decode -- largest of {len(chunks)} chunk(s) "
                f"is {largest} frame(s) -- {e}",
                file=sys.stderr,
            )
            # Only RAM has a --stack-len answer; a disk shortfall gets no such advice.
            if not _astro_ram_fits(largest, megapixels, len(dark_paths)):
                print(
                    _astro_stack_len_advice(
                        runs,
                        stack_len=stack_len,
                        max_span_s=max_chunk_span_s,
                        megapixels=megapixels,
                        dark_count=len(dark_paths),
                    ),
                    file=sys.stderr,
                )
            return 1

    written = 0
    if not args.triage_only:
        output_dir.mkdir(parents=True, exist_ok=True)
    # Carried forward across chunks (never a second decode pass -- see
    # astro_triage.update_run_cloud_baseline), so a chunk entirely past a cloud
    # transition still judges against the run's own clear sky, not just itself.
    run_cloud_baseline = None
    sheet_written = not args.contact_sheet  # one stack answers for the whole sheet
    for i, chunk in enumerate(chunks, start=1):
        label = f"chunk {i}/{len(chunks)} ({len(chunk)} frames)"
        out_path = None if args.linear_only else targets[i - 1]
        linear_path = linear_targets[i - 1] if keep_linear else None
        # Per chunk, because the RAM gate below may lower it for THIS chunk alone; the
        # next chunk asks again and gets the operator's own value back when RAM recovers.
        chunk_workers = decode_workers
        try:
            # dark_count is the run's OWN --darks, not the estimator's no-darks default:
            # prepare_frames holds every dark live while build_master_dark converts and
            # stacks them, so a gate told 0 under-counts precisely when darks are adding
            # the memory (the estimator charges the same extra memory for the darks at any
            # chunk frame count). meteor_reinsertion is
            # left at its off default deliberately: this path passes stack_run no
            # `meteors=` policy, so reinsertion cannot happen and its layer is never
            # allocated -- claiming it would over-count and false-fail a fitting run.
            # --trails gates on what the max blend holds (no frame_count term -- its peak has
            # none); --triage-only keeps the stack gate, prepare_frames holds the chunk live.
            if whole_run and not args.triage_only:
                gate = preflight_trail if args.trails else tlapse.preflight_timelapse
                gate(output_dir, megapixels=megapixels, dark_count=len(dark_paths))
            else:
                # degrade_workers: a mid-run dip costs speed, never the chunk. The
                # approved count is what prepare_frames must then decode with, so the
                # gate and the decode cannot disagree.
                chunk_workers = preflight_astro_stack(
                    output_dir,
                    frame_count=len(chunk),
                    megapixels=megapixels,
                    dark_count=len(dark_paths),
                    decode_workers=decode_workers,
                    degrade_workers=True,
                )
                if chunk_workers != decode_workers:
                    print(
                        f"{label}: free RAM is short -- decode workers "
                        f"{decode_workers} -> {chunk_workers} for this chunk "
                        "(slower, but it stacks)",
                        flush=True,
                    )
        except RuntimeError as e:
            # A BACKSTOP now, not the gate -- the run was sized above before any decode.
            # Free RAM can still fall mid-run (this box runs other work), and when it does
            # the masters already stacked must survive, so skip this chunk exactly as a
            # failed chunk does below rather than killing the run. `written == 0` still
            # carries a produced-nothing run to a non-zero exit.
            print(f"{label}: preflight refused -- {e} -- skipping this chunk", file=sys.stderr)
            continue
        try:
            sky_mask = None
            if args.horizon is not None:
                shape = _astro_frame_shape(chunk[0])
                # horizon_mask returns a feathered [0, 1] blend weight; triage
                # (measure_frame/detect_stars) takes a bool "sky" mask, so threshold
                # at the same boundary row the eventual blend uses.
                sky_mask = horizon_mask(shape, horizon_frac=args.horizon) >= 0.5
            if args.timelapse and not args.triage_only:
                # BEFORE the render, not after: a short sequence makes a very short clip (90
                # frames at 24 fps is 3.75s), learned while shooting more is still cheap.
                dur = len(chunk) / timelapse_fps  # one captured frame, one video frame
                fps = f"{len(chunk)} frames at {timelapse_fps} fps = {dur:.1f}s of video"
                print(f"{label}: {fps} -- no frame is duplicated or interpolated", flush=True)
                path, used, dropped = tlapse.render_timelapse(
                    chunk,
                    out_path,
                    dark_paths,
                    sky_mask,
                    fps=timelapse_fps,
                    scale_height=timelapse_height,
                )
                written += 1
                print(f"{label}: {used} frame(s) rendered, {len(dropped)} dropped -> {path}")
                continue
            if args.trails and not args.triage_only:
                # One long silent phase, so say what it is doing first.
                print(f"{label}: decoding + maximum-blending, streaming...", flush=True)
                path, used, dropped = render_trail(chunk, out_path, dark_paths, sky_mask)
                written += 1
                print(f"{label}: {used} frame(s) of arcs, {len(dropped)} dropped -> {path}")
                continue
            # Decode+calibrate is slow and prints nothing else -- silence is
            # indistinguishable from a hang (an OOM-killed child would go unnoticed). flush
            # because a redirected stdout block-buffers and delivers only at exit.
            print(f"{label}: decoding + calibrating...", flush=True)
            # cloud_baseline omitted entirely (rather than passed as None) before the
            # run has established one -- identical to prepare_frames' own default,
            # kept explicit only once there is a real baseline to pass.
            prepare_kwargs = {
                "dark_paths": dark_paths,
                "horizon_mask": sky_mask,
                "decode_workers": chunk_workers,
                "detect_meteors": args.detect_meteors,
            }
            if run_cloud_baseline is not None:
                prepare_kwargs["cloud_baseline"] = run_cloud_baseline
            prepared = prepare_frames(chunk, **prepare_kwargs)
            run_cloud_baseline = update_run_cloud_baseline(run_cloud_baseline, prepared.metrics)
            # Printed here, not inside the --triage-only branch: this is the earliest
            # moment the verdict exists, and on a full run it is the last one at which
            # the operator can still stop and refocus instead of stacking a lost night.
            shape_note = _astro_shape_note(prepared.metrics)
            if shape_note:
                print(f"{label}: {shape_note}", flush=True)
            if args.triage_only:
                usable = sum(1 for m in prepared.metrics if m.usable)
                print(
                    f"{label}: {usable}/{len(prepared.frames)} usable, "
                    f"{len(prepared.report.cloud_frames)} cloud, "
                    f"{len(prepared.report.dew_frames)} dew"
                )
                continue
            # Register+stack is the SECOND silent multi-minute phase.
            print(f"{label}: registering + stacking {len(prepared.frames)} frame(s)...", flush=True)
            report = stack_run(
                prepared,
                out_path,
                horizon_frac=args.horizon,
                linear_artifact_path=linear_path,
                linear_only=args.linear_only,
            )
        except (AstroOrchestratorError, AstroCalibrationError, AstroHighEfficiencyError) as e:
            print(f"{label}: {e}", file=sys.stderr)
            continue
        written += 1
        if args.linear_only:
            print(
                f"{label}: retained {report.stack.n_frames} frame(s) -> "
                f"{report.linear_artifact_path}"
            )
            continue
        # An operator watching the run must see how thin the stack got, not
        # just how many frames survived registration.
        coverage = report.stack.coverage
        coverage_note = (
            f", {coverage.below_floor_fraction:.1%} below coverage floor"
            if coverage is not None
            else ""
        )
        # The chroma blur is cosmetic and irreversible, so the run says what it did
        # rather than changing every master silently.
        chroma_note = (
            f", colour speckle {report.chroma_sigma_before:.4f} -> {report.chroma_sigma_after:.4f}"
            if report.chroma_sigma_before > 0
            else ""
        )
        print(
            f"{label}: stacked {report.stack.n_frames} frame(s){coverage_note}"
            f"{chroma_note} -> {report.image_path}"
        )
        # The .tif reads as a black rectangle in a file browser; say where the
        # openable version landed, or the operator renders one by hand again.
        if report.preview_path is not None:
            print(f"{label}: preview -> {report.preview_path}")
        if report.linear_artifact_path is not None:
            print(f"{label}: linear artifact -> {report.linear_artifact_path}")
        if not sheet_written and (sheet_path := _contact_sheet_path(photos, output_dir.name)):
            sheet = compose_contact_sheet(stack_panels(report))
            print(f"{label}: contact sheet -> {export_jpeg(sheet, sheet_path)}")
        sheet_written = True

    if args.triage_only:
        return 0
    if written == 0:
        print("No chunk produced a stacked master", file=sys.stderr)
        return 1
    if written < len(chunks):
        # An incomplete night is not a success. Every skip above (RAM backstop, decode or
        # stack failure) deliberately keeps the run alive so the masters already stacked
        # survive -- but a zero exit status would let a run that lost most of its chunks
        # report the same thing a clean night does, and an unattended caller
        # (cron, a wrapper script) could not tell them apart. Derived from the counts
        # rather than tracked per skip path, so a skip added later cannot forget to flag
        # itself. Non-zero, not a raise: this handler's whole convention is a stderr line
        # plus a return code, and the masters are already on disk -- there is nothing left
        # to unwind. 1, not a new code: `photos astro` returns only 0 or 1 today, and
        # "the run is not a success" is exactly what 1 already means here.
        print(
            f"{len(chunks) - written} of {len(chunks)} chunk(s) produced no stacked master "
            f"-- this run is INCOMPLETE; the {written} master(s) above are still valid",
            file=sys.stderr,
        )
        return 1
    return 0


def _cmd_photos_astro_inspect(args) -> int:
    """READ-ONLY: contact sheet of 1:1 corner/centre crops + star FWHM/ellipticity
    from an existing stacked master. Never modifies/re-develops the master."""
    from ...pipeline.photos.astro_contact import ContactSheetError, inspect_master
    from ...pipeline.photos.develop import compose_contact_sheet, export_jpeg

    try:
        panels, summary = inspect_master(args.master)
    except ContactSheetError as e:
        print(str(e), file=sys.stderr)
        return 1
    sheet = compose_contact_sheet(panels, cols=3)
    out_path = export_jpeg(sheet, args.out)
    print(f"contact sheet -> {out_path}")
    print(summary)
    return 0


def _cmd_photos_astro_combine(args) -> int:
    """Reusable linear artifacts -> one developed TIFF plus coverage/provenance."""
    from ...pipeline.photos.astro_combine import AstroCombineError, deep_combine
    from ...pipeline.photos.astro_combine_output import (
        MATERIAL_THIN_COVERAGE_FRACTION,
        DeepOutputError,
        publish_deep_product,
    )

    try:
        result = deep_combine(
            [Path(path) for path in args.paths],
            reference=Path(args.reference) if args.reference else None,
            scope=args.scope,
        )
        paths = publish_deep_product(result, Path(args.out))
    except (AstroCombineError, DeepOutputError, OSError) as exc:
        print(f"astro-combine failed: {exc}", file=sys.stderr)
        return 1
    coverage = result.coverage
    if coverage.below_floor_fraction >= MATERIAL_THIN_COVERAGE_FRACTION:
        print(
            f"WARNING: {coverage.below_floor_fraction:.2%} of sky pixels are below the "
            f"{coverage.floor:g}-contributor floor; inspect {paths.coverage}",
            file=sys.stderr,
        )
    print(f"deep master -> {paths.image}")
    print(f"coverage -> {paths.coverage}")
    print(f"provenance -> {paths.sidecar}")
    return 0


def cmd_photos(args, config):
    """Daily photo social media posting pipeline."""
    if args.photos_command == "night":
        from .night import cmd_night

        return cmd_night(args)

    astro_handlers = {
        "astro": _cmd_photos_astro,
        "astro-combine": _cmd_photos_astro_combine,
        "astro-inspect": _cmd_photos_astro_inspect,
    }
    if args.photos_command in astro_handlers:
        from ...pipeline.media_admission import MediaAdmissionError, media_admission

        configure_logging()
        try:
            with media_admission() as acquired:
                if not acquired:
                    print("Waiting for shared media admission; no images decoded.", file=sys.stderr)
                    return 2
                return astro_handlers[args.photos_command](args)
        except MediaAdmissionError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    from ...config import ConfigError
    from ...pipeline.photos import face_filter
    from ...pipeline.photos.orchestrator import (
        PhotoOrchestrator,
        describe_scheduler_export,
    )
    from ...publishing.photo_run_hooks import photo_run_hooks

    configure_logging()

    # Blur is ON by default; apply the per-run opt-out before any selection or
    # resize, so no published copy can be written under a stale setting.
    face_filter.set_blur_enabled(not getattr(args, "skip_blur", False))

    # Ingest runs before the orchestrator: it feeds the photo pool rather than
    # consuming it, so it must not require a populated journey/selector.
    if args.photos_command == "ingest":
        return _cmd_photos_ingest(args)
    # Develop is likewise pool-independent: local working files only.
    if args.photos_command == "develop":
        return _cmd_photos_develop(args)
    # Reel likewise: given stills in, one local MP4 out, no pool, no publish.
    if args.photos_command == "reel":
        return _cmd_photos_reel(args)
    # Daylight timelapse likewise: given stills in, one local MP4 out, no pool, no publish.
    if args.photos_command == "daylight-timelapse":
        return _cmd_photos_daylight_timelapse(args)
    # Per-run social crop, resolved at construction so an unknown label fails
    # before selection — None falls through to photos.social_aspect.
    orchestrator = PhotoOrchestrator(
        social_aspect=getattr(args, "social_aspect", None), **photo_run_hooks()
    )

    # Explicit escape for the fail-closed state load: archive the ledger aside
    # (never delete) BEFORE any selection touches it, so the accepted wipe
    # stays recoverable and the run starts from a genuinely fresh state.
    if getattr(args, "reset_selection_state", False):
        from ...pipeline.photos.selector import PhotoSelectionState

        archived = PhotoSelectionState.reset(orchestrator.state_path)
        if archived:
            print(f"Selection state archived aside: {archived}")
        else:
            print("No selection state file to reset; starting fresh.")

    if args.photos_command == "generate":
        voice = None
        if args.voice:
            voice = _get_voice_map()[args.voice]

        try:
            batch = orchestrator.generate_batch(
                week_start=args.week,
                count=args.count,
                voice_style=voice,
                filter_faces=getattr(args, "no_faces", False),
                allow_shortfall=getattr(args, "allow_photo_count", False),
                overwrite=getattr(args, "overwrite", False),
            )
            print(f"Generated {len(batch.captions)} captions ({batch.voice_style})")
            print("Review YAML, then publish:")
            # Resolved caption dir: GMS_PHOTO_OUTPUT_DIR-based, not cwd-relative.
            caption_dir = orchestrator.caption_gen.caption_dir
            caption_path = caption_dir / f"{batch.week_start}_captions.yaml"
            print(f"  gomoveshift photos publish --captions {caption_path}")
        except Exception as e:
            print(f"Generation failed: {type(e).__name__}: {e}", file=sys.stderr)
            return 1

    elif args.photos_command == "publish":
        try:
            csv_path = orchestrator.publish_batch(
                caption_yaml=args.captions,
                force=args.force,
                start_date=getattr(args, "start", None),
                end_date=getattr(args, "end_date", None),
            )
            print(describe_scheduler_export(csv_path))
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Publish failed: {type(e).__name__}: {e}", file=sys.stderr)
            return 1

    elif args.photos_command == "release":
        try:
            released = orchestrator.release_batch(args.week)
            print(f"Released {len(released)} held photo(s) from batch {args.week}")
            for key in released:
                print(f"  - {key}")
        except ConfigError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    elif args.photos_command == "status":
        try:
            status = orchestrator.status()
        except ConfigError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print("Photo Posting Status")
        print("=" * 40)
        print(f"  Journeys loaded:  {status['journeys']}")
        print(f"  Total photos:     {status['total_photos']}")
        print(f"  Already posted:   {status['posted']}")
        print(f"  Remaining:        {status['remaining']}")
        print(f"  Days of content:  {status['days_of_content']}")

    elif args.photos_command == "cleanup":
        plan = orchestrator.cleanup(
            min_age_days=args.age,
            confirm=args.confirm,
        )

        if not plan.files:
            print("Nothing to clean up.")
        else:
            total_bytes = sum(f["size"] for f in plan.files)
            prefix = "DRY RUN \u2014 " if not args.confirm else ""
            print(f"{prefix}Cleanup plan: {len(plan.files)} files, {total_bytes / 1024:.1f} KB")
            print()
            for f in plan.files:
                size_kb = f["size"] / 1024
                print(
                    f"  {'DELETE' if args.confirm else 'WOULD DELETE'}: "
                    f"{f['path']} ({size_kb:.1f} KB) \u2014 {f['reason']}"
                )

        if plan.skipped:
            print()
            print(f"Skipped {len(plan.skipped)} item(s):")
            for s in plan.skipped:
                print(f"  SKIP: {s['path']} \u2014 {s['reason']}")

        if not args.confirm and plan.files:
            print()
            print("This is a dry run. Add --confirm to actually delete.")

    elif args.photos_command == "batch":
        voice = None
        if args.voice:
            voice = _get_voice_map()[args.voice]

        try:
            csv_path = orchestrator.auto_batch(
                start_date=args.start,
                end_date=args.end,
                output_csv_dir=args.output,
                voice_style=voice,
                filter_faces=getattr(args, "no_faces", False),
            )
            print(describe_scheduler_export(csv_path))
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Batch failed: {e}", file=sys.stderr)
            return 1

    else:
        print(
            "Usage: gomoveshift photos "
            "{ingest|develop|generate|publish|release|status|cleanup|batch}"
        )
        return 1

    return 0
