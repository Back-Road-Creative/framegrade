"""Private console API for Route Story draft projects."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from ..paths import InputsDirError, gms_inputs_dir, gms_route_story_cache_dir
from ..pipeline.route_story.gpx import load_gpx_segments
from ..pipeline.route_story.metrics import summarize_track
from ..pipeline.route_story.opening import SatelliteOpening
from ..pipeline.route_story.project import RouteStoryStore
from ..pipeline.route_story.satellite import SatelliteRaster
from . import settings

router = APIRouter(prefix="/route-stories")
_SATELLITE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _store() -> RouteStoryStore:
    return RouteStoryStore(settings.console_data_dir() / "route-story-projects")


def _roots() -> list[Path]:
    return [
        Path(root)
        for root in os.environ.get("GMS_ROUTE_STORY_SOURCE_ROOTS", "").split(os.pathsep)
        if root
    ]


def _verified_source(item: dict[str, object]) -> Path:
    """Return a reviewed source only when its current bytes still match the draft."""
    path = Path(str(item["path"]))
    try:
        with path.open("rb") as source:
            current = hashlib.file_digest(source, "sha256").hexdigest()
    except OSError as exc:
        raise ValueError("Route Story reviewed source is unavailable") from exc
    if current != item.get("sha256"):
        raise ValueError("Route Story reviewed source changed after draft creation")
    return path


@router.get("")
def list_projects() -> list[dict[str, object]]:
    return _store().list(settings.current_brand())


@router.post("", status_code=201)
async def create_project(request: Request) -> JSONResponse:
    payload = await request.json()
    try:
        project = _store().create(
            settings.current_brand(),
            [Path(value) for value in payload.get("sources", [])],
            _roots(),
            payload.get("recipe", {}),
        )
    except (TypeError, ValueError, OSError) as exc:
        raise HTTPException(422, str(exc)) from exc
    return JSONResponse({"id": project}, status_code=201)


@router.get("/{project}/telemetry")
def project_telemetry(project: str) -> dict[str, object]:
    """Return source-clock facts for a saved draft without starting a render."""
    try:
        draft = _store().get(settings.current_brand(), project)
        gpx_sources = [
            _verified_source(item)
            for item in draft["sources"]
            if str(item["path"]).endswith(".gpx")
        ]
        if len(gpx_sources) != 1:
            raise ValueError("Route Story draft needs exactly one GPX source for telemetry")
        speed = float(draft["recipe"].get("moving_speed_mps", 0.5))
        return asdict(summarize_track(load_gpx_segments(gpx_sources[0]), moving_speed_mps=speed))
    except KeyError as exc:
        raise HTTPException(404, "Route Story draft was not found") from exc
    except (TypeError, ValueError, OSError) as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/{project}/opening")
def project_opening(project: str) -> dict[str, object]:
    """Describe an approved draft's local satellite opening without rendering it."""
    try:
        draft = _store().get(settings.current_brand(), project)
        recipe = draft["recipe"]
        opening_recipe = recipe.get("opening")
        if not isinstance(opening_recipe, dict):
            raise ValueError("Route Story draft needs an opening recipe")
        gpx_sources = [
            _verified_source(item)
            for item in draft["sources"]
            if str(item["path"]).endswith(".gpx")
        ]
        raster_sources = [
            _verified_source(item)
            for item in draft["sources"]
            if Path(str(item["path"])).suffix.lower() in _SATELLITE_SUFFIXES
        ]
        if len(gpx_sources) != 1 or len(raster_sources) != 1:
            raise ValueError("Route Story opening needs one GPX and one satellite image source")
        segments = load_gpx_segments(gpx_sources[0])
        start = segments[0][0]
        satellite_recipe = opening_recipe.get("satellite")
        if not isinstance(satellite_recipe, dict):
            raise ValueError("Route Story opening needs satellite geographic bounds and credit")
        satellite = SatelliteRaster(raster_sources[0], **satellite_recipe)
        opening = SatelliteOpening.for_route_start(
            latitude=start.latitude,
            longitude=start.longitude,
            start_frame=0,
            end_frame=int(opening_recipe.get("frames", 300)),
            final_zoom=float(opening_recipe.get("final_zoom", 12)),
        )
        midpoint = (opening.start_frame + opening.end_frame) // 2
        return {
            "camera": {
                "start": asdict(opening.pose_at_frame(opening.start_frame)),
                "middle": asdict(opening.pose_at_frame(midpoint)),
                "end": asdict(opening.pose_at_frame(opening.end_frame)),
            },
            "satellite": satellite.receipt(),
        }
    except KeyError as exc:
        raise HTTPException(404, "Route Story draft was not found") from exc
    except (TypeError, ValueError, OSError) as exc:
        raise HTTPException(422, str(exc)) from exc


#: Profile directories: ``route_story_profiles/`` in the operator's inputs dir
#: (``GMS_INPUTS_DIR``), a ``route_story_profiles/`` directory beside the package
#: (if present) and ``route-story-profiles/`` under the console data dir.
def profile_dirs() -> list[Path]:
    service_root = Path(__file__).resolve().parent.parent.parent
    return [
        gms_inputs_dir() / "route_story_profiles",
        service_root / "route_story_profiles",
        settings.console_data_dir() / "route-story-profiles",
    ]


def list_profiles() -> list[str]:
    """Distinct ``*.yaml`` basenames across every profile dir, sorted."""
    names: set[str] = set()
    for directory in profile_dirs():
        if not directory.is_dir():
            continue
        for entry in directory.glob("*.yaml"):
            if entry.is_file():
                names.add(entry.name)
    return sorted(names)


def resolve_profile(name: str) -> Path:
    """Basename only: ``/``, ``..``, or a non-``.yaml`` name is refused
    before it reaches the filesystem, against the known profile dirs."""
    if not name or "/" in name or ".." in name or not name.endswith(".yaml"):
        raise ValueError(f"not a valid profile name: {name!r}")
    for directory in profile_dirs():
        candidate = directory / name
        if candidate.is_file():
            return candidate
    raise ValueError(f"unknown Route Story profile: {name!r}")


@router.get("/profiles")
def route_story_profiles() -> list[str]:
    try:
        return list_profiles()
    except InputsDirError as exc:
        raise HTTPException(422, str(exc)) from exc


#: The modules ``drive_phases/trail_film.py::default_steps`` imports to run a film,
#: and the third-party imports ``route_story/proof.py`` and ``film.py`` make at load.
_TRAIL_MODULES = (
    *(
        f"gomoveshift.pipeline.route_story.{name}"
        for name in ("prepare", "proof", "verify", "film")
    ),
    "numpy",
    "PIL",
)
#: What ``pipeline/inserts.py`` imports at module level, beyond the module itself.
_INSERT_MODULES = ("gomoveshift.pipeline.inserts", "keycut", "PIL")


def _missing_modules(modules: tuple[str, ...]) -> list[str]:
    missing = []
    for module in modules:
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(module)
    return missing


def trail_film_unavailable() -> str:
    """Why ``gomoveshift drive --trail`` cannot render a film here, or ``""``.

    Measured, not asserted: the route-story modules the trail step imports, then the
    prepared-bundle cache the step looks in (it never fetches inside a drive). A bundle
    is keyed by a day's GPX, so a populated cache is necessary, not sufficient: a day
    with no matching bundle still records a ``trail_skips`` reason at run time.
    """
    missing = _missing_modules(_TRAIL_MODULES)
    if missing:
        return f"route-story dependency not installed: {', '.join(missing)}"
    cache = gms_route_story_cache_dir()
    try:
        # A bundle is a dir holding ``manifest.json`` (``bundle.verify_bundle``); the
        # cache also keeps ``shared/`` and ``gnis/`` and any interrupted fetch's dir.
        has_bundle = cache.is_dir() and any(
            (entry / "manifest.json").is_file() for entry in cache.iterdir() if entry.is_dir()
        )
    except OSError as exc:
        return f"prepared imagery bundle cache {cache} is unreadable: {exc}"
    if not has_bundle:
        return (
            f"no prepared imagery bundle in {cache}; "
            "run the Route Story prepare step for the trip's GPX first"
        )
    return ""


def inserts_unavailable() -> str:
    """Why ``--insert-photos`` / ``--insert-clip`` cannot run here, or ``""``."""
    missing = _missing_modules(_INSERT_MODULES)
    if missing:
        return f"insert dependency not installed: {', '.join(missing)}"
    if shutil.which("ffmpeg") is None:
        return "ffmpeg is not on PATH, so the insert splice cannot encode"
    return ""
