"""FastAPI web application for video review UI."""

import asyncio
import dataclasses
import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from ..analysis.pipeline import AnalysisPipeline
from ..assembly.assembler import VideoAssembler
from ..config import Config, get_config
from ..models import ProcessingJob, VideoAnalysis

logger = logging.getLogger(__name__)


_ALLOWED_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _assert_localhost_only(host: str, *, allow_remote: bool = False) -> None:
    """Refuse to start on a non-loopback bind unless explicitly allowed.

    The web API has no authentication — it relies on the loopback bind as
    its auth boundary. Allowing a non-loopback bind without ``allow_remote``
    would expose the path-traversal-prone ``/api/frame/{path}`` route to
    the network. The ``allow_remote`` opt-in exists so an operator can run
    the UI on a trusted internal segment with an external auth proxy.
    """
    if allow_remote:
        return
    if host not in _ALLOWED_LOOPBACK_HOSTS:
        raise RuntimeError(
            f"Refusing to bind web UI on {host!r}: the API has no auth and "
            "must run on loopback only. Pass --allow-remote to override "
            "after fronting it with an auth proxy."
        )


class ApprovalRequest(BaseModel):
    """Request to approve edits for a video."""

    video_path: str
    approved_trims: list[tuple[float, float]]  # (start, end) pairs to remove
    approved_cuts: list[float]  # timestamps to split at


class JobStatus(BaseModel):
    """Status of a processing job."""

    video_path: str
    status: str
    progress: float
    message: str


# In-memory job storage (would use database in production)
_MAX_JOBS = 50
jobs: dict[str, ProcessingJob] = {}
analyses: dict[str, VideoAnalysis] = {}


def _jobs_state_path() -> Path:
    """Path to the on-disk job persistence file.

    Lives under the staging dir alongside ``pipeline_state.json``-shaped
    artefacts. Overridable for tests via ``GMS_WEB_JOBS_PATH``.
    """
    override = os.environ.get("GMS_WEB_JOBS_PATH", "")
    if override:
        return Path(override)
    from .. import paths as _paths

    return _paths.gms_staging_dir() / "_stages" / "web_jobs.json"


def _load_jobs_from_disk() -> None:
    """Restore the in-memory jobs map from disk if a snapshot exists.

    Silent no-op if the file is missing — fresh starts shouldn't fail.
    Logs at WARNING on parse errors so a corrupt snapshot is visible
    rather than masked.
    """
    path = _jobs_state_path()
    if not path.exists():
        return
    try:
        raw = json.loads(path.read_text())
    except Exception as exc:
        logger.warning("web_jobs.json parse failed (%s) — starting clean", exc)
        return
    if not isinstance(raw, dict):
        logger.warning("web_jobs.json has unexpected shape %r — ignoring", type(raw))
        return
    for key, payload in raw.items():
        try:
            # Rehydrate Path fields from their serialized string forms.
            if isinstance(payload, dict) and "video_path" in payload:
                payload["video_path"] = Path(payload["video_path"])
            if isinstance(payload, dict) and "output_files" in payload:
                payload["output_files"] = [Path(p) for p in payload["output_files"]]
            # ``analysis`` is dropped on persist (heavy, derivable); ignore
            # any value carried in the snapshot to avoid re-hydration cost.
            if isinstance(payload, dict):
                payload.pop("analysis", None)
            jobs[key] = ProcessingJob(**payload)
        except Exception as exc:
            logger.warning("web_jobs.json: skipping bad entry %r: %s", key, exc)


def _persist_jobs_to_disk() -> None:
    """Atomic-write the in-memory jobs map.

    Writes to a sibling tmp file then ``rename`` — partial writes never
    appear on disk, so a crash mid-write doesn't corrupt the snapshot.
    """
    path = _jobs_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    # ``ProcessingJob`` is a stdlib dataclass — use ``asdict`` and drop the
    # heavy ``analysis`` field (re-derivable from the analysis pipeline).
    def _job_to_dict(job: ProcessingJob) -> dict:
        d = dataclasses.asdict(job)
        d.pop("analysis", None)
        return d

    payload = {key: _job_to_dict(job) for key, job in jobs.items()}
    fd, tmp_name = tempfile.mkstemp(prefix="web_jobs.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        os.replace(tmp_name, path)
    except Exception:
        # Best-effort cleanup of the tmp file on failure; re-raise so the
        # caller sees the persistence error rather than a silent skip.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _evict_oldest(store: dict, max_size: int = _MAX_JOBS) -> None:
    """Remove oldest entries when store exceeds max_size."""
    while len(store) > max_size:
        oldest_key = next(iter(store))
        del store[oldest_key]


def _build_allowed_roots(config: Config) -> list[Path]:
    """Build the path-containment allowlist from configured folders ONLY.

    The allowlist is derived strictly from ``config.paths.input_folder``
    and ``config.paths.output_folder``. The historical
    workspace-root seed is GONE — it made the blast radius of the
    path-traversal-prone ``/api/frame/{path}`` route every sibling project
    next to this one, not just this service's own media. An unset
    folder contributes nothing; if BOTH are unset the list is empty and
    ``_resolve_safe_path`` fails loud (server-config error) rather than
    silently permitting the whole workspace or rejecting everything opaquely.
    """
    roots: list[Path] = []
    for folder in (config.paths.input_folder, config.paths.output_folder):
        if folder:
            roots.append(Path(folder).resolve())
    return roots


# The ONLY content types the still route will serve. An extension not
# in this map is refused outright — the route reviews developed photos, it is
# not a general file server, so a readable non-image inside an allowed root
# (state JSON, credentials, logs) must never leave the box through it.
_IMAGE_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def _build_photo_roots(config: Config) -> list[Path]:
    """Extra containment roots for the still-image route ONLY.

    The photo develop gate reviews stills that live outside the video
    input/output folders: the site's journey image tree (sources) and the
    configured photo output / resized bases (developed copies). Every path
    comes from ``PhotosConfig`` so an operator override moves them together.

    These roots widen the *image* route's allowlist only — ``/api/frame``
    keeps the narrower video allowlist — and they are still checked by the
    same ``_resolve_safe_path`` containment; nothing here is a bypass.
    """
    from ..pipeline.photos.selector import IMAGES_BASE

    photos = config.photos
    roots: list[Path] = []
    # The site checkout is optional: with no ``photos.site_root`` configured
    # there is no journey image tree to serve, so it contributes nothing.
    if photos.site_root:
        roots.append((photos.resolve_site_root() / IMAGES_BASE).resolve())
    roots.append(photos.resolve_output_dir().resolve())
    roots.append(photos.resolve_resized_base().resolve())
    return roots


class _FrameEncodeError(RuntimeError):
    """The decoder reported success but handed back no JPEG — maps to a 500."""


#: A single-frame grab is decode-to-timestamp on one keyframe interval; anything
#: past this is a stuck decoder, not a slow one.
_FRAME_READ_TIMEOUT_S = 30


def _read_frame_sync(path: Path, timestamp: float) -> Optional[bytes]:
    """Timestamp-accurate single-frame grab, returned as JPEG bytes.

    Seeks in SECONDS via ffmpeg rather than converting to a frame index
    (``int(timestamp * fps)``). Index arithmetic assumes a
    constant frame rate *and* a truthful container average; on a variable-rate
    source — a GoPro capture, anything assembled from clips of differing cadence
    — it lands hundreds of milliseconds away, and the reviewer then approves a
    window against content the timestamp does not contain. Input-side ``-ss`` is
    both fast (keyframe jump) and exact (decode-and-discard to the requested
    time); it is the same grab the sign pass uses.

    Still a plain blocking call, so the async endpoint keeps dispatching it
    through ``run_in_executor`` and the event loop stays free.

    Returns the JPEG bytes, ``None`` when the frame can't be read (the caller
    maps that to its 404), and raises ``_FrameEncodeError`` (caller: 500) when
    ffmpeg reports success but the payload is not a JPEG.
    """
    try:
        proc = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error",
             "-ss", f"{float(timestamp):.6f}", "-i", str(path),
             "-frames:v", "1", "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "2", "-"],
            capture_output=True, timeout=_FRAME_READ_TIMEOUT_S,
        )  # fmt: skip
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("frame read failed at %s +%.3fs: %s", path, timestamp, exc)
        return None

    if proc.returncode != 0 or not proc.stdout:
        return None
    if proc.stdout[:2] != b"\xff\xd8":  # JPEG SOI
        raise _FrameEncodeError("Frame encoding failed")
    return proc.stdout


def create_app(config: Optional[Config] = None, *, allow_remote: bool = False) -> FastAPI:
    """Create the FastAPI application.

    ``allow_remote`` widens CORS ``allow_origins`` to ``"*"`` ONLY when
    the operator has explicitly opted into a non-loopback bind (the same
    ``--allow-remote`` flag that ``_assert_localhost_only`` gates). The
    default is loopback-only origins, matching the loopback auth boundary —
    a remote bind without this flag must front the UI with an auth proxy.
    """
    # Security: This API relies on localhost binding (127.0.0.1) as its auth boundary.
    # Path containment (``_build_allowed_roots`` + ``_resolve_safe_path``) prevents
    # filesystem traversal. ``_assert_localhost_only`` refuses to start on a
    # non-loopback bind unless ``--allow-remote`` is explicitly passed.

    if config is None:
        config = get_config()

    app = FastAPI(
        title="GoMoveShift Video Processor",
        description="Review and approve driving video edits",
        version="0.1.0",
    )

    # Restore jobs from disk so a restart doesn't lose tracked work.
    _load_jobs_from_disk()

    # Loopback-only origins by default; widen to "*" ONLY under the
    # explicit --allow-remote opt-in (which presupposes an auth proxy).
    cors_origins = ["http://127.0.0.1:8765", "http://localhost:8765"]
    if allow_remote:
        cors_origins = ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization"],
    )

    def _resolve_safe_path(video_path: str, extra_roots: Optional[list[Path]] = None) -> Path:
        """Resolve a path and verify it's within allowed directories.

        ``allowed_roots`` is built ONLY from the configured input/output
        folders (no workspace-root seed). If neither is configured the list
        is empty and we raise a 500 server-config error — failing loud is
        safer than silently rejecting (opaque) or permitting a whole
        workspace tree.

        ``extra_roots`` appends per-route roots (the still route's
        photo dirs) to the SAME containment check — every caller resolves
        symlinks/``..`` first and must land inside one of the roots.
        """
        resolved = Path(video_path).resolve()
        allowed_roots = _build_allowed_roots(config) + list(extra_roots or [])
        if not allowed_roots:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Web UI misconfigured: no allowed directories configured. "
                    "Set config.paths.input_folder and/or output_folder."
                ),
            )
        for root in allowed_roots:
            try:
                if resolved.is_relative_to(root):
                    return resolved
            except (TypeError, ValueError):
                continue
        raise HTTPException(status_code=403, detail="Path outside allowed directories")

    @app.get("/health")
    async def health():
        """Dependency-aware liveness check.

        Returns 200 when ``ffmpeg`` + ``ffprobe`` resolve on PATH and
        ``config.paths.temp_folder`` is writable. Otherwise returns 503
        with a JSON body listing the failing checks — operators get a
        machine-readable reason instead of a green light over a broken
        environment.
        """
        failed: list[str] = []
        for binary in ("ffmpeg", "ffprobe"):
            if shutil.which(binary) is None:
                failed.append(binary)

        scratch = Path(config.paths.temp_folder or tempfile.gettempdir())
        try:
            scratch.mkdir(parents=True, exist_ok=True)
            probe = scratch / ".health_probe"
            probe.write_text("ok")
            probe.unlink()
        except OSError:
            failed.append("scratch_dir")

        if failed:
            return JSONResponse(status_code=503, content={"status": "fail", "failed": failed})
        return {"status": "ok"}

    # Setup templates
    templates_dir = Path(__file__).parent / "templates"
    templates_dir.mkdir(exist_ok=True)
    templates = Jinja2Templates(directory=str(templates_dir))

    # Initialize pipeline and assembler
    pipeline = AnalysisPipeline(config)
    assembler = VideoAssembler(config)

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        """Main dashboard page."""
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "jobs": list(jobs.values()),
                "analyses": analyses,
            },
        )

    @app.get("/api/jobs")
    async def list_jobs():
        """List all processing jobs."""
        return {
            "jobs": [
                {
                    "video_path": str(job.video_path),
                    "status": job.status,
                    "error": job.error_message,
                }
                for job in jobs.values()
            ]
        }

    @app.post("/api/analyze")
    async def analyze_video(
        background_tasks: BackgroundTasks,
        video_path: str,
    ):
        """Start analysis of a video file."""
        path = _resolve_safe_path(video_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Video not found")

        # Create job
        job = ProcessingJob(video_path=path, status="analyzing")
        jobs[video_path] = job
        _evict_oldest(jobs)
        try:
            _persist_jobs_to_disk()
        except Exception as exc:
            # Persistence is best-effort — losing the snapshot does not
            # break the in-memory job. Log loudly so a broken disk path
            # doesn't go unnoticed.
            logger.warning("web jobs persist failed: %s", exc, exc_info=True)

        # Run analysis in background
        def run_analysis():
            try:
                analysis = pipeline.analyze(path)
                analyses[video_path] = analysis
                _evict_oldest(analyses)
                job.analysis = analysis
                job.status = "reviewed"
            except Exception as e:
                job.status = "error"
                job.error_message = str(e)

        background_tasks.add_task(run_analysis)

        return {"status": "started", "video_path": video_path}

    @app.get("/api/analysis/{video_path:path}")
    async def get_analysis(video_path: str):
        """Get analysis results for a video."""
        if video_path not in analyses:
            raise HTTPException(status_code=404, detail="Analysis not found")

        return analyses[video_path].to_dict()

    @app.post("/api/approve")
    async def approve_edits(
        background_tasks: BackgroundTasks,
        request: ApprovalRequest,
    ):
        """Approve edits and start video processing."""
        video_path = request.video_path

        if video_path not in analyses:
            raise HTTPException(status_code=404, detail="Analysis not found")

        job = jobs.get(video_path)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")

        job.status = "processing"
        job.approved_trims = request.approved_trims
        job.approved_cuts = request.approved_cuts

        # This VideoAssembler render path runs NO stop/audio/blur validation
        # and honors only the operator's approved_trims (an empty list keeps every
        # dead spot). Mark its outputs EXPLICITLY ungated so they are never mistaken
        # for the gated drive-pipeline outputs — this is a manual/local review UI.
        job.ungated_reason = (
            "web /api/approve VideoAssembler path — no stop/audio/blur gate; "
            "trims limited to operator-approved cuts"
        )
        logger.warning(
            "Job %s: UNGATED render (web /api/approve) — no stop/audio/blur "
            "validation; outputs are for manual review, not gated upload",
            video_path,
        )

        # Run processing in background
        def run_processing():
            try:
                analysis = analyses[video_path]
                plan = assembler.create_assembly_plan(
                    analysis,
                    approved_trims=request.approved_trims,
                    approved_cuts=request.approved_cuts,
                )
                output_files = assembler.execute(plan)
                job.output_files = output_files
                job.status = "complete"
            except Exception as e:
                job.status = "error"
                job.error_message = str(e)

        background_tasks.add_task(run_processing)

        return {"status": "processing", "video_path": video_path}

    @app.get("/api/frame/{video_path:path}")
    async def get_frame(video_path: str, timestamp: float = 0):
        """Get a frame from the video at specified timestamp."""
        # Reject NaN / inf / negative: unchecked, these would flow into
        # ``int(timestamp * fps)`` and produce indeterminate seeks or
        # OpenCV errors a tier deeper than the input.
        if not math.isfinite(timestamp) or timestamp < 0:
            raise HTTPException(
                status_code=422,
                detail="timestamp must be a non-negative finite number",
            )

        path = _resolve_safe_path(video_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Video not found")

        # The OpenCV seek+decode is synchronous and would block uvicorn's
        # event loop for the full duration, starving concurrent requests. Run
        # it in the default thread-pool executor so the loop stays free.
        loop = asyncio.get_running_loop()
        try:
            frame_bytes = await loop.run_in_executor(None, _read_frame_sync, path, timestamp)
        except _FrameEncodeError:
            raise HTTPException(status_code=500, detail="Frame encoding failed") from None

        if frame_bytes is None:
            raise HTTPException(status_code=404, detail="Frame not found")
        return Response(content=frame_bytes, media_type="image/jpeg")

    @app.get("/api/image/{image_path:path}")
    async def get_image(image_path: str):
        """Serve a still image for the photo develop review gate.

        Containment first (same ``_resolve_safe_path``, widened only by the
        photo roots), then an explicit content-type allowlist, then the file
        streams straight off disk — no base64-in-JSON round trip for a
        multi-megabyte JPEG.
        """
        path = _resolve_safe_path(image_path, _build_photo_roots(config))

        media_type = _IMAGE_MEDIA_TYPES.get(path.suffix.lower())
        if media_type is None:
            # Refuse before touching the filesystem so the response can't
            # double as an existence oracle for non-image files.
            raise HTTPException(status_code=415, detail="Unsupported image type")
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Image not found")

        return FileResponse(path, media_type=media_type)

    @app.get("/review/{video_path:path}", response_class=HTMLResponse)
    async def review_video(request: Request, video_path: str):
        """Video review page."""
        if video_path not in analyses:
            raise HTTPException(status_code=404, detail="Analysis not found")

        analysis = analyses[video_path]

        return templates.TemplateResponse(
            "review.html",
            {
                "request": request,
                "analysis": analysis.to_dict(),
                "video_path": video_path,
            },
        )

    return app
