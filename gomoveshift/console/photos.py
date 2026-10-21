"""Console Photos tab helpers (minimal slice: ingest + develop only).

``build_photos_argv`` mirrors a manual ``gomoveshift photos <action> ...``
invocation (``gomoveshift/cli/parser.py::_add_photos_parser``) -- the console
never invents a second launch path. Selector, reel and astro are launched from
the command line, not from this screen; the screen says so from
``capability_status`` rather than from prose.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from urllib.parse import urlencode

_ACTIONS = ("ingest", "develop")

#: (label, engine module, the command that runs it, console action or None).
#: The engine column is a real importable module, so ``capability_status``
#: answers "does this exist?" by asking the build instead of trusting a
#: sentence. The page previously read "Selector, reel and astro are not built
#: yet" while all three engines shipped and every one had a CLI command --
#: a claim no reader could tell from a genuinely missing feature.
_CAPABILITIES: tuple[tuple[str, str, str, "str | None"], ...] = (
    ("Import photos", "gomoveshift.pipeline.photos.ingest", "gomoveshift photos ingest", "ingest"),
    ("Develop", "gomoveshift.pipeline.photos.develop", "gomoveshift photos develop", "develop"),
    ("Selector", "gomoveshift.pipeline.photos.selector", "gomoveshift photos generate", None),
    ("Reel", "gomoveshift.pipeline.photos.reel", "gomoveshift photos reel", None),
    ("Astro", "gomoveshift.pipeline.photos.astro_orchestrator", "gomoveshift photos astro", None),
)

#: The normal creator path, in order. Each stage points at the console screen
#: that already drives it -- the review, schedule and results screens are not
#: rebuilt here, they are named and linked so the journey reads as one flow.
_STAGES: tuple[tuple[str, str, str], ...] = (
    ("Import", "/photos#import", "Bring a card's photos into a journey day."),
    ("Suggested treatment", "/photos#develop", "Apply a look, then adjust if you want to."),
    ("Review", "/library", "Look at what came out before anything ships."),
    ("Schedule", "/schedule", "Pick when it publishes."),
    ("Results", "/seo", "See how it performed."),
)


def capability_status() -> list[dict]:
    """What this build can do with photos, and where each part is driven from.

    ``installed`` is measured (``importlib.util.find_spec``), never asserted:
    a capability whose engine module imports can never be rendered as missing.
    ``console`` is the narrower, honest claim -- whether *this screen* launches
    it -- and ``command`` is what to type when it does not.
    """
    rows = []
    for label, module, command, action in _CAPABILITIES:
        rows.append(
            {
                "name": label,
                "module": module,
                "command": command,
                "installed": importlib.util.find_spec(module) is not None,
                "console": action is not None and action in _ACTIONS,
                "reason": ""
                if importlib.util.find_spec(module) is not None
                else f"module {module} is not importable",
            }
        )
    rows.extend(_drive_capability_rows())
    return rows


def _drive_capability_rows() -> list[dict]:
    """The two photo-adjacent drive features, in the same row shape.

    Their ``installed`` flag means "can run here now" and ``reason`` says what is
    missing when it cannot; the checks live beside the Route Story console code
    (``console/route_story.py``). Neither is launched from the console.
    """
    from . import route_story

    drive = (
        (
            "Trail film",
            "gomoveshift.pipeline.drive_phases.trail_film",
            "gomoveshift drive --trail --trail-profile PROFILE.yaml",
            route_story.trail_film_unavailable(),
        ),
        (
            "Inserts",
            "gomoveshift.pipeline.inserts",
            "gomoveshift drive --insert-photos DIR --insert-clip FILE@MM:SS",
            route_story.inserts_unavailable(),
        ),
    )
    return [
        {
            "name": name,
            "module": module,
            "command": command,
            "installed": not reason,
            "console": False,
            "reason": reason,
        }
        for name, module, command, reason in drive
    ]


def journey_stages(journey: str = "") -> list[dict]:
    """The five creator-journey stages, each carrying the selected journey.

    Navigation preserves the selection: every stage link is threaded with
    ``?journey=<slug>`` so hopping to review or schedule and back does not
    make the operator retype which journey they are working on.
    """
    stages = []
    for name, href, hint in _STAGES:
        target = href
        if journey:
            base, _, fragment = href.partition("#")
            target = f"{base}?{urlencode({'journey': journey})}"
            if fragment:
                target = f"{target}#{fragment}"
        stages.append({"name": name, "href": target, "hint": hint})
    return stages


def resolved_looks() -> dict:
    """The same ``{built-in} merged with {operator-configured}`` table the CLI's
    ``photos develop --look`` handler resolves against
    (``cli/handlers/photos.py::_cmd_photos_develop``) -- one source of truth so
    the console never drifts from what the CLI will actually accept."""
    from ..config import get_config
    from ..pipeline.photos.looks import BUILTIN_LOOKS

    return {**BUILTIN_LOOKS, **get_config().photos.looks}


def resolved_styles() -> list[str]:
    """Every style ``photos develop --styles`` will accept, from the one catalog the
    renderer reads (``pipeline/photos/catalog.py``) -- the list the form offers is
    the list the run cannot refuse."""
    from ..pipeline.photos.catalog import style_names

    return style_names()


def validate_develop_paths(paths: list[str], out: str) -> list[str]:
    """Filesystem problems with a develop request, each named specifically.

    The develop form's ``paths`` and ``out`` fields are
    ordinary visible free text (never hidden inside a closed ``<details>`` --
    only the numeric film-finish dials are), so nothing stopped a typo from
    reaching ``jobs.launch`` and failing deep inside the job log where the
    operator's form is already gone. This is the upstream gate: called from
    the route before any job is launched, and each entry names the ONE fact
    that is wrong -- "no such file" and "is a directory" are different facts
    from a generic "invalid path", which is the same silence this exists to
    replace. ``out`` need not exist yet (``photos develop`` creates it); it
    is only rejected when something already there is not a directory.
    """
    errors = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            errors.append(f"{raw}: no such file")
        elif path.is_dir():
            errors.append(f"{raw}: is a directory, not a photo file")
        elif not os.access(path, os.R_OK):
            errors.append(f"{raw}: not readable")
    out_path = Path(out)
    if out_path.exists() and not out_path.is_dir():
        errors.append(f"{out}: is a file, not a directory")
    return errors


def validate_panorama_paths(paths: list[str]) -> list[str]:
    """Filesystem/count problems with a panorama stitch request, each named
    specifically -- same upstream-gate shape as ``validate_develop_paths``.
    ``run_panorama`` calls this before touching the pipeline, so a bad
    selection never reaches ``stitch_panorama``."""
    errors = []
    non_blank = [p for p in paths if str(p).strip()]
    if len(non_blank) < 2:
        errors.append(f"panorama needs at least 2 photos, got {len(non_blank)}")
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            errors.append(f"{raw}: no such file")
        elif path.is_dir():
            errors.append(f"{raw}: is a directory, not a photo file")
        elif not os.access(path, os.R_OK):
            errors.append(f"{raw}: not readable")
    return errors


def run_panorama(paths: list[str], out: str) -> dict:
    """Stitch ``paths`` into one panorama at ``out``, in-process (no CLI
    subprocess -- there is no ``gomoveshift photos panorama`` command yet).
    Refuses via ``validate_panorama_paths`` before calling the pipeline, and
    returns a plain dict rather than ``PanoramaResult`` so callers (templates,
    JSON responses) don't need the dataclass import."""
    errors = validate_panorama_paths(paths)
    if errors:
        raise ValueError("; ".join(errors))
    from ..pipeline.photos.panorama import stitch_panorama

    result = stitch_panorama(paths, out)
    return {
        "out_path": str(result.out_path),
        "width": result.width,
        "height": result.height,
        "valid_area_fraction": result.valid_area_fraction,
    }


def journey_photo_options(journey: str, site_root=None) -> dict:
    """The photo set selectable for ``develop`` under one journey.

    Reads the SAME directory the selector treats as the live gallery
    (``selector.IMAGES_BASE`` under ``selector.default_site_root()``) --
    never a second, re-derived glob. This is the picker's data: the
    operator chooses from real files instead of typing a path per photo.

    ``status`` names ONE of four distinct facts rather than collapsing an
    empty result to a blank list, which is the exact silence this picker exists to avoid:
    ``no_journey`` (nothing selected yet), ``no_dir`` (this journey has no
    images directory), ``empty`` (the directory exists and has nothing in
    it), ``unreadable`` (the directory exists but could not be listed), or
    ``ok`` with the photos found.
    """
    if not journey:
        return {
            "status": "no_journey",
            "message": "Select a journey to see its photos.",
            "photos": [],
        }

    from ..config import ConfigError
    from ..pipeline.photos.selector import IMAGES_BASE, default_site_root

    try:
        root = site_root if site_root is not None else default_site_root()
    except ConfigError as exc:
        return {"status": "no_dir", "message": str(exc), "photos": []}
    images_dir = root / IMAGES_BASE / journey
    if not images_dir.exists():
        return {
            "status": "no_dir",
            "message": f"{journey}: no images directory yet ({images_dir}).",
            "photos": [],
        }
    try:
        entries = sorted((p for p in images_dir.iterdir() if p.is_file()), key=lambda p: p.name)
    except OSError as exc:
        return {
            "status": "unreadable",
            "message": f"{journey}: could not read {images_dir}: {exc}",
            "photos": [],
        }
    if not entries:
        return {
            "status": "empty",
            "message": f"{journey}: images directory is empty ({images_dir}).",
            "photos": [],
        }
    return {
        "status": "ok",
        "message": "",
        "photos": [{"filename": p.name, "path": str(p)} for p in entries],
    }


def build_photos_argv(action: str, **kwargs) -> list[str]:
    """Argv for ``gomoveshift photos <action> ...``, operator order.

    Supports ``ingest`` and ``develop`` only -- ``selector``/``reel``/``astro``
    raise, same as an unknown action.
    """
    if action == "ingest":
        return _ingest_argv(**kwargs)
    if action == "develop":
        return _develop_argv(**kwargs)
    raise ValueError(f"unsupported photos action: {action!r}")


def _ingest_argv(
    *,
    card_dir: str,
    journey: str,
    day: str,
    location: "str | None" = None,
    dry_run: bool = True,
) -> list[str]:
    # The page marks these three `required`, which a browser enforces and
    # a direct POST does not. Blank, they built `--day ''` and launched a job
    # that could only fail inside the log -- by which point the operator's form
    # is gone. Same one-source-of-truth rule as the dial ranges below: never
    # offer the CLI a value it would refuse.
    for field, value in (("card_dir", card_dir), ("journey", journey), ("day", day)):
        if not str(value).strip():
            raise ValueError(f"{field}: required")
    argv = [
        sys.executable,
        "-m",
        "gomoveshift",
        "photos",
        "ingest",
        str(card_dir),
        "--journey",
        journey,
        "--day",
        day,
    ]
    if location:
        argv += ["--location", location]
    if dry_run:
        argv.append("--dry-run")
    return argv


# The film-finish dials this slice exposes: flag, (min, max) -- ranges match
# each ``add_argument`` call in ``cli/parser.py``'s develop subparser exactly,
# so the console can never offer a value the CLI would refuse.
_DIAL_RANGES: dict[str, tuple[float, float]] = {
    "vignette": (0.0, 1.0),
    "clarity": (-1.0, 1.0),
    "grain": (0.0, 1.0),
    "grain_size": (0.25, 4.0),
    "halation": (0.0, 1.0),
    # White balance group: ranges match config.PHOTO_KNOBS'
    # white_balance_* spans exactly, same one-source-of-truth discipline.
    "white_balance_g": (-0.5, 0.5),
    "white_balance_r": (-0.1, 0.1),
    "white_balance_b": (-0.1, 0.1),
    "white_balance_r_high": (-0.1, 0.1),
    "white_balance_b_high": (-0.1, 0.1),
    "white_balance_g_high": (-0.5, 0.5),
}


def _develop_argv(
    *,
    paths: list[str],
    out: str,
    styles: "str | None" = None,
    look: "str | None" = None,
    vignette: "float | None" = None,
    clarity: "float | None" = None,
    grain: "float | None" = None,
    grain_size: "float | None" = None,
    halation: "float | None" = None,
    white_balance_g: "float | None" = None,
    white_balance_r: "float | None" = None,
    white_balance_b: "float | None" = None,
    white_balance_r_high: "float | None" = None,
    white_balance_b_high: "float | None" = None,
    white_balance_g_high: "float | None" = None,
    export_format: "str | None" = None,
) -> list[str]:
    argv = [sys.executable, "-m", "gomoveshift", "photos", "develop", *[str(p) for p in paths]]
    argv += ["--out", str(out)]
    if styles:
        # Same gate `look` gets below: refuse by name here, where the operator can
        # read it, not at develop time inside the job log.
        names = [n.strip() for n in styles.split(",") if n.strip()]
        unknown = [n for n in names if n not in resolved_styles()]
        if unknown:
            raise ValueError(f"unknown style: {', '.join(repr(n) for n in unknown)}")
        argv += ["--styles", ",".join(names)]
    if look:
        if look not in resolved_looks():
            raise ValueError(f"unknown look: {look!r}")
        argv += ["--look", look]
    dials = {
        "vignette": vignette,
        "clarity": clarity,
        "grain": grain,
        "grain_size": grain_size,
        "halation": halation,
        "white_balance_g": white_balance_g,
        "white_balance_r": white_balance_r,
        "white_balance_b": white_balance_b,
        "white_balance_r_high": white_balance_r_high,
        "white_balance_b_high": white_balance_b_high,
        "white_balance_g_high": white_balance_g_high,
    }
    for name, value in dials.items():
        if value is None:
            continue
        lo, hi = _DIAL_RANGES[name]
        if not lo <= value <= hi:
            raise ValueError(f"--{name.replace('_', '-')} must be in range {lo}..{hi}, got {value}")
        argv += [f"--{name.replace('_', '-')}", str(value)]
    if export_format:
        # JPEG is the default and adds nothing, so the default argv stays
        # byte-identical. The allowed set is config's, mirroring the engine's.
        from ..config import PHOTO_EXPORT_FORMATS

        if export_format not in PHOTO_EXPORT_FORMATS:
            raise ValueError(
                f"export format must be one of: {', '.join(PHOTO_EXPORT_FORMATS)}, "
                f"got {export_format!r}"
            )
        if export_format != "jpeg":
            argv += ["--format", export_format]
    return argv
