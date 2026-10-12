"""Subject recognition — what a photo is OF, beside what ``scene.py`` measures.

``SceneProfile`` measures a frame's LIGHT (exposure, warmth, saturation, haze,
dynamic range, chroma headroom) and the styles ladder picks treatments from it.
Nothing said what the frame is a photograph of, so a 20-second ISO-3200 frame of
the Milky Way and a 20-second daylight frame of a waterfall were the same object
to this pipeline.

Three deliberate positions, each pinned by a test:

**Unknown is a real answer.** ``SubjectCategory.UNKNOWN`` at confidence exactly
``0.0`` is what this module returns whenever the evidence matches no category —
never the commonest category, never a default. The discipline is the one
``tracking/schema.py`` already documents for ``Optional[...] = None``: "not
reported" is not a measured zero. A named category may never carry confidence
0.0 and ``UNKNOWN`` may never carry more; the dataclass refuses both.

**The evidence is EXIF, source grouping and the operator — in that order of
availability and the reverse order of authority.** EXIF is free, reliable and
frequently stronger than the pixels: a 20 s exposure at ISO 3200 under a solar
elevation of −31° is astro far more certainly than any colour statistic could
say. Source grouping adds what one frame cannot show — a run of consecutive
matching frames from one session is a *sequence*, and the existing bracket
detector (``bracket.py``) already says which files are one HDR set. An operator
correction, persisted in a ``.subject.json`` sidecar beside the photo, beats
every inference outright and is the only reading that reaches confidence 1.0.

**Heuristic, offline, and honest about it.** There is no model here and no
network call at import or at inference — every rule is EXIF arithmetic plus the
measurements already computed. The confidences are calibrated to how much the
evidence actually proves, not to how often the guess is right: ``LANDSCAPE`` off
a wide stopped-down daylight frame is the weakest reading in the table because
the pixels were never asked. A future model-backed recogniser must degrade to
``UNKNOWN`` when unavailable, never to a guess.

**Recognition routes style selection, above a confidence floor.**
``styles.select_styles`` takes this module's ``SubjectReading`` as an optional
fourth input and, only when the reading's category has an evidenced style
preference (``styles.SUBJECT_STYLE_ROUTING``) AND its confidence clears
``styles.SUBJECT_ROUTING_CONFIDENCE_MIN``, moves that style to right after
``natural`` — never displacing it. Every other case — no reading, ``UNKNOWN``,
an unrouted category, or a reading below the floor — stays bit-identical to
selection with no subject at all, which is what ``select_styles(scene)`` (no
``subject`` argument) still does everywhere it is called today.
"""

from __future__ import annotations

import datetime
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .bracket import detect_brackets
from .exif_gps import (
    extract_exif_aperture,
    extract_exif_capture_metadata,
    extract_exif_capture_timestamp,
    extract_exif_focal_length,
)
from .scene import PhotoExif, SceneProfile

logger = logging.getLogger(__name__)


class SubjectCategory(str, Enum):
    """The taxonomy, deliberately small and grounded in what this pipeline shoots.

    Travel and nature stills off a photographer's cards: night skies, animals
    at a long lens, the occasional person, tripod water frames, wide scenics,
    and the exposure brackets the HDR merge already groups. Anything else is
    ``UNKNOWN`` — a sprawling ontology this pipeline cannot evidence would just
    be a guess with more names.
    """

    UNKNOWN = "unknown"
    ASTRO = "astro"
    WILDLIFE = "wildlife"
    PORTRAIT = "portrait"
    LONG_EXPOSURE = "long_exposure"
    EXPOSURE_BRACKET = "exposure_bracket"
    LANDSCAPE = "landscape"


# Confidences. Operator is the only 1.0 on the board; inference caps below it,
# because no heuristic off EXIF alone is ever certain.
CONFIDENCE_OPERATOR = 1.0
CONFIDENCE_INFERENCE_MAX = 0.95
CONFIDENCE_GROUPED = 0.9  # the bracket detector's own grouping rule said so
CONFIDENCE_TRACKED_NIGHT = 0.85  # long exposure + high ISO + a measured sun below the horizon
CONFIDENCE_MODERATE = 0.65  # two agreeing EXIF facts
CONFIDENCE_WEAK = 0.45  # one EXIF fact consistent with the category and nothing against it
CONFIDENCE_SEQUENCE_BONUS = 0.1  # consecutive frames from one session agreeing

# Astro: a shutter no handheld frame carries, a gain no daylight frame needs,
# and a sky that is actually dark. All three, or it is not astro.
ASTRO_EXPOSURE_S_MIN = 8.0
ASTRO_ISO_MIN = 1600.0
#: Sun at/below the end of nautical twilight — the sky the Milky Way needs.
NIGHT_SOLAR_ELEVATION_MAX = -12.0
#: Camera-LOCAL wall clock, the no-track fallback only, with the same honest
#: limitation ``styles._is_golden_hour`` documents: it does not move with
#: latitude, season or DST. Hours >= the evening start or <= the morning end.
NIGHT_HOUR_EVENING_START = 21
NIGHT_HOUR_MORNING_END = 4
#: Frames from one session that must agree before a run reads as a sequence.
ASTRO_SEQUENCE_FRAMES_MIN = 4
#: Gap that ends a session. Wider than the bracket rule's ``MAX_GAP_S`` (45 s):
#: a tracked sky sequence idles between subs, and a session survives a lens change.
SESSION_GAP_S = 600.0

# Optics. A 200 mm floor is where a travel kit's long end starts and no scenic
# lives; the portrait band is the classic 35-135 mm at a subject-separating
# aperture; landscape is the wide, stopped-down end.
WILDLIFE_FOCAL_MM_MIN = 200.0
WILDLIFE_SHUTTER_S_MAX = 1 / 250
PORTRAIT_FOCAL_MM_MIN = 35.0
PORTRAIT_FOCAL_MM_MAX = 135.0
PORTRAIT_APERTURE_MAX = 2.8
LANDSCAPE_FOCAL_MM_MAX = 35.0
LANDSCAPE_APERTURE_MIN = 8.0

# Long exposure that is NOT astro: the tripod water frame — a shutter no hand
# holds, at a gain that says the scene was bright. Named for what the EXIF
# proves (a long exposure), not for what it probably is (a waterfall): the
# pixels were never asked, and a category this module cannot evidence is a
# fabrication with a nicer name.
LONG_EXPOSURE_S_MIN = 0.25
LONG_EXPOSURE_ISO_MAX = 400.0

#: Sidecar beside the photo, the ``.grade.json`` pattern ``develop.py`` uses.
SUBJECT_SIDECAR_SUFFIX = ".subject.json"
SUBJECT_SIDECAR_SCHEMA = "gms.photo.subject/1"

SOURCE_OPERATOR = "operator"
SOURCE_INFERENCE = "inference"


@dataclass(frozen=True)
class CaptureOptics:
    """Lens metadata, caller-supplied like ``PhotoExif``. ``None`` = not reported.

    Deliberately separate from ``PhotoExif``: nothing in scene profiling meters
    a lens, and widening that record would ripple through every profile call
    site (the reason ``extract_exif_aperture`` already sits outside it).
    """

    focal_length_mm: float | None = None
    f_number: float | None = None


@dataclass(frozen=True)
class SubjectReading:
    """One photo's subject, its confidence, where it came from, and why.

    ``confidence`` is 0-1. ``UNKNOWN`` carries exactly 0.0 and every named
    category carries more than 0.0 — the invariant that stops an unknown from
    being laundered into a low-confidence guess, or a guess into an unknown.
    """

    category: SubjectCategory
    confidence: float
    source: str = SOURCE_INFERENCE
    evidence: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be within 0-1, got {self.confidence!r}")
        if self.category is SubjectCategory.UNKNOWN and self.confidence != 0.0:
            raise ValueError("unknown is an answer, not a weak guess: confidence must be 0.0")
        if self.category is not SubjectCategory.UNKNOWN and self.confidence == 0.0:
            raise ValueError(f"{self.category.value} at confidence 0.0 is unknown; say so")
        if self.source not in (SOURCE_OPERATOR, SOURCE_INFERENCE):
            raise ValueError(f"unknown source {self.source!r}")


#: What this module returns whenever nothing is proven. A module-level constant
#: so every "no answer" path is literally the same object.
UNKNOWN_SUBJECT = SubjectReading(SubjectCategory.UNKNOWN, 0.0, SOURCE_INFERENCE, ())


def _sidecar_path(photo_path) -> Path:
    return Path(f"{Path(photo_path)}{SUBJECT_SIDECAR_SUFFIX}")


def read_subject_correction(photo_path) -> SubjectCategory | None:
    """The operator's correction for this photo, or ``None`` — never a guess.

    ``None`` means "no correction recorded" and is distinct from a recorded
    ``UNKNOWN``, which is the operator saying they looked and could not tell.
    Never raises: an absent, unreadable, malformed or unrecognised sidecar all
    read ``None``, so one bad file can't abort a card.
    """
    path = _sidecar_path(photo_path)
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict):
        return None
    try:
        return SubjectCategory(record.get("category"))
    except ValueError:
        logger.debug("Unrecognised subject category %r in %s", record.get("category"), path)
        return None


def write_subject_correction(photo_path, category: SubjectCategory) -> Path:
    """Persist an operator correction beside the photo; return the sidecar path."""
    if not isinstance(category, SubjectCategory):
        raise ValueError(f"expected a SubjectCategory, got {category!r}")
    path = _sidecar_path(photo_path)
    record = {
        "schema": SUBJECT_SIDECAR_SCHEMA,
        "category": category.value,
        "source": SOURCE_OPERATOR,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return path


def read_capture_optics(photo_path) -> CaptureOptics:
    """This photo's lens metadata off EXIF. Never raises; absent tags read ``None``."""
    return CaptureOptics(
        focal_length_mm=extract_exif_focal_length(photo_path),
        f_number=extract_exif_aperture(photo_path),
    )


def _night(exif: PhotoExif, scene: SceneProfile | None) -> tuple[bool, bool, str]:
    """``(is_night, tracked, evidence)`` — was the sky dark, and how do we know?

    A measured ``solar_elevation_deg`` is authoritative in BOTH directions: with
    the sun 40° up, a long high-ISO exposure is not astro however it meters.
    Without a track, the local wall clock and a low-key frame are the fallback.
    """
    if exif.solar_elevation_deg is not None:
        below = exif.solar_elevation_deg <= NIGHT_SOLAR_ELEVATION_MAX
        return below, True, f"solar elevation {exif.solar_elevation_deg:.0f} deg"
    hour = exif.capture_hour
    if hour is not None and (hour >= NIGHT_HOUR_EVENING_START or hour <= NIGHT_HOUR_MORNING_END):
        return True, False, f"capture hour {hour:02d} (camera-local)"
    if scene is not None and scene.exposure_class == "low_key":
        return True, False, "low-key frame, no solar fix"
    return False, False, ""


def _daylight(exif: PhotoExif) -> bool:
    """True only on positive evidence of daylight — an unknown sky is not day."""
    if exif.solar_elevation_deg is not None:
        return exif.solar_elevation_deg > 0.0
    hour = exif.capture_hour
    return hour is not None and NIGHT_HOUR_MORNING_END < hour < NIGHT_HOUR_EVENING_START


def _astro_like(exif: PhotoExif, scene: SceneProfile | None = None) -> bool:
    """The shutter/gain/darkness triple, without the sequence or confidence."""
    if not exif.exposure_s or not exif.iso:
        return False
    if exif.exposure_s < ASTRO_EXPOSURE_S_MIN or exif.iso < ASTRO_ISO_MIN:
        return False
    return _night(exif, scene)[0]


def recognise_subject(
    *,
    scene: SceneProfile | None = None,
    exif: PhotoExif | Mapping | None = None,
    optics: CaptureOptics | None = None,
    in_exposure_bracket: bool = False,
    session_astro_frames: int = 1,
    correction: SubjectCategory | None = None,
) -> SubjectReading:
    """What this photo is of. Pure, offline and deterministic — no file reads.

    ``correction`` (the operator) wins outright. Otherwise the rules run in
    descending strength of evidence — source grouping, then the EXIF triples,
    then the single-fact optics reads — and the first one that MATCHES answers.
    Nothing matching is ``UNKNOWN_SUBJECT``, never a fallback category.
    """
    if correction is not None:
        confidence = 0.0 if correction is SubjectCategory.UNKNOWN else CONFIDENCE_OPERATOR
        return SubjectReading(correction, confidence, SOURCE_OPERATOR, ("operator correction",))

    if isinstance(exif, Mapping):
        exif = PhotoExif(**{k: exif.get(k) for k in PhotoExif.__dataclass_fields__})
    exif = exif or PhotoExif()
    optics = optics or CaptureOptics()

    if in_exposure_bracket:
        return SubjectReading(
            SubjectCategory.EXPOSURE_BRACKET,
            CONFIDENCE_GROUPED,
            SOURCE_INFERENCE,
            ("source grouping: one exposure bracket",),
        )

    if exif.exposure_s and exif.iso:
        night, tracked, why = _night(exif, scene)
        if night and exif.exposure_s >= ASTRO_EXPOSURE_S_MIN and exif.iso >= ASTRO_ISO_MIN:
            evidence = [f"exif: {exif.exposure_s:g} s at ISO {exif.iso:g}", why]
            confidence = CONFIDENCE_TRACKED_NIGHT if tracked else CONFIDENCE_MODERATE
            if session_astro_frames >= ASTRO_SEQUENCE_FRAMES_MIN:
                evidence.append(f"source grouping: sequence of {session_astro_frames} frames")
                confidence = min(confidence + CONFIDENCE_SEQUENCE_BONUS, CONFIDENCE_INFERENCE_MAX)
            return SubjectReading(
                SubjectCategory.ASTRO, confidence, SOURCE_INFERENCE, tuple(evidence)
            )

    focal, aperture = optics.focal_length_mm, optics.f_number
    night_now = _night(exif, scene)[0]

    if focal is not None and focal >= WILDLIFE_FOCAL_MM_MIN and not night_now:
        evidence = [f"exif: {focal:g} mm"]
        confidence = CONFIDENCE_WEAK
        if exif.exposure_s and exif.exposure_s <= WILDLIFE_SHUTTER_S_MAX:
            evidence.append(f"exif: {exif.exposure_s:g} s, action shutter")
            confidence = CONFIDENCE_MODERATE
        return SubjectReading(
            SubjectCategory.WILDLIFE, confidence, SOURCE_INFERENCE, tuple(evidence)
        )

    if (
        focal is not None
        and aperture is not None
        and PORTRAIT_FOCAL_MM_MIN <= focal <= PORTRAIT_FOCAL_MM_MAX
        and aperture <= PORTRAIT_APERTURE_MAX
    ):
        return SubjectReading(
            SubjectCategory.PORTRAIT,
            CONFIDENCE_MODERATE,
            SOURCE_INFERENCE,
            (f"exif: {focal:g} mm at f/{aperture:g}, subject-separating",),
        )

    if (
        exif.exposure_s
        and exif.exposure_s >= LONG_EXPOSURE_S_MIN
        and exif.iso is not None
        and exif.iso <= LONG_EXPOSURE_ISO_MAX
        and not night_now
    ):
        return SubjectReading(
            SubjectCategory.LONG_EXPOSURE,
            CONFIDENCE_MODERATE,
            SOURCE_INFERENCE,
            (f"exif: {exif.exposure_s:g} s at ISO {exif.iso:g}, tripod-length in bright light",),
        )

    if (
        focal is not None
        and aperture is not None
        and focal <= LANDSCAPE_FOCAL_MM_MAX
        and aperture >= LANDSCAPE_APERTURE_MIN
        and _daylight(exif)
    ):
        return SubjectReading(
            SubjectCategory.LANDSCAPE,
            CONFIDENCE_WEAK,
            SOURCE_INFERENCE,
            (f"exif: {focal:g} mm at f/{aperture:g} in daylight — optics only, pixels unread",),
        )

    return UNKNOWN_SUBJECT


def _session_astro_frames(
    paths: Sequence[Path],
    exifs: Mapping[Path, PhotoExif],
    stamps: Mapping[Path, datetime.datetime | None],
) -> dict[Path, int]:
    """How many astro-like frames each path's session holds.

    A session is a run of consecutive captures no more than ``SESSION_GAP_S``
    apart — the source grouping a single frame cannot show. A frame with no
    readable capture time is its own session of one, never folded into someone
    else's run.
    """
    timed = sorted((p for p in paths if stamps.get(p)), key=lambda p: (stamps[p], str(p)))
    counts: dict[Path, int] = {p: 1 for p in paths}
    session: list[Path] = []

    def _close() -> None:
        astro = sum(1 for p in session if _astro_like(exifs[p]))
        for p in session:
            counts[p] = astro

    for path in timed:
        if session and (stamps[path] - stamps[session[-1]]).total_seconds() > SESSION_GAP_S:
            _close()
            session = []
        session.append(path)
    _close()
    return counts


def recognise_photos(
    photo_paths, scenes: Mapping[Path, SceneProfile] | None = None
) -> dict[Path, SubjectReading]:
    """Recognise a whole card at once, reading EXIF and sidecars from disk.

    This is where the three non-pixel evidence sources meet: ``detect_brackets``
    supplies bracket membership, session runs supply sequences, and each photo's
    ``.subject.json`` supplies the operator's correction, which wins. ``scenes``
    optionally supplies already-computed ``SceneProfile``s — nothing here
    decodes a pixel, so a card reads in milliseconds and a caller that has
    profiled its frames gets the extra evidence for free.
    """
    paths = [Path(p) for p in photo_paths or []]
    bracketed = {Path(p) for group in detect_brackets(paths) for p in group}
    exifs = {p: extract_exif_capture_metadata(p) for p in paths}
    stamps = {p: extract_exif_capture_timestamp(p) for p in paths}
    sequence_frames = _session_astro_frames(paths, exifs, stamps)
    return {
        p: recognise_subject(
            scene=(scenes or {}).get(p),
            exif=exifs[p],
            optics=read_capture_optics(p),
            in_exposure_bracket=p in bracketed,
            session_astro_frames=sequence_frames[p],
            correction=read_subject_correction(p),
        )
        for p in paths
    }
