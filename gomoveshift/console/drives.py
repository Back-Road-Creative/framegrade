"""Console Drives tab helpers (minimal slice).

``build_drive_argv`` mirrors a manual ``gomoveshift drive`` invocation --
the console never invents a second launch path. The gate helpers implement
the operator side of the answer-file contract in
``pipeline/review_gate.py``: read a pending ``<gate_id>.question.json`` and
write the ``<gate_id>.answer`` the pipeline consumes on its next poll.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from gomoveshift.pipeline.validation.content_gates import (
    check_description_redundancy,
    check_label_quality,
    check_metadata_quality,
)

#: Matches ``pipeline/review_gate.py``'s ``GATE_ANSWERS_SUBDIR``.
GATE_ANSWERS_SUBDIR = "_gates"

#: Matches the glob ``review_gate.compute_gate_cd_hash`` uses to find every
#: LF's generated metadata under an output dir (``stages/metadata.py`` /
#: ``drive_phases/phase_d_metadata.py`` write ``<project>_youtube.json``).
_METADATA_GLOB = "*_youtube.json"

#: Content gates run against a real, generated metadata dict. Calls the
#: same pipeline functions the drive run itself uses (grammar via
#: ``check_metadata_quality``, LABEL-QUALITY, description redundancy)
#: -- the preview never re-implements gate logic.
_PREVIEW_GATES = (check_metadata_quality, check_label_quality, check_description_redundancy)

#: Mirrors ``cli/parser.py``'s ``--lf-mode`` choices.
LF_MODE_CHOICES = ("best_moments", "full_drive")


def build_drive_argv(
    directory: str,
    *,
    output_dir: str,
    region: "str | None" = None,
    dry_run: bool = True,
    no_review: bool = True,
    publish: "str | None" = None,
    lf_day: "str | None" = None,
    lf_mode: "str | None" = None,
    lf_target_duration: "float | None" = None,
    superlong: bool = False,
    shorts_target: "int | None" = None,
    shorts_gap: "float | None" = None,
) -> list[str]:
    """Argv for ``gomoveshift drive <directory> ...``, operator order.

    ``--dry-run`` follows unless unchecked. ``no_review`` defaults True to
    preserve every existing caller (e.g. ``console/shorts.py``) that never
    passes it; the Drives-tab form is the one caller that threads an
    explicit operator choice through, defaulting the checkbox OFF so human
    review stays ON unless the operator opts out.

    ``publish``/``lf_day`` are optional but paired (an explicit
    schedule is required): giving one without the
    other raises rather than launching a drive with an implicit schedule.

    ``lf_mode``/``lf_target_duration``/``superlong``/``shorts_target``/
    ``shorts_gap`` mirror the matching ``cli/parser.py`` flags exactly,
    omitted when falsy/``None``; ``lf_mode`` is validated against
    :data:`LF_MODE_CHOICES`.

    ``--no-tmux`` is always appended: the console job store plus the
    container's ``restart: unless-stopped`` policy already supervise this
    process (``docker/console-compose.yaml`` sets ``GMS_SUPERVISED=1``), so
    the tmux re-exec gate would otherwise fail every console-launched drive.
    """
    if bool(publish) != bool(lf_day):
        raise ValueError("--publish and --lf-day must be given together, or not at all")
    if lf_mode is not None and lf_mode not in LF_MODE_CHOICES:
        raise ValueError(f"lf_mode must be one of {LF_MODE_CHOICES!r}, got {lf_mode!r}")
    argv = [sys.executable, "-m", "gomoveshift", "drive", str(directory)]
    if region:
        argv += ["--region", region]
    argv += ["--output-dir", str(output_dir)]
    if no_review:
        argv.append("--no-review")
    if dry_run:
        argv.append("--dry-run")
    if publish and lf_day:
        argv += ["--publish", publish, "--lf-day", lf_day]
    if lf_mode:
        argv += ["--lf-mode", lf_mode]
    if lf_target_duration is not None:
        argv += ["--lf-target-duration", str(lf_target_duration)]
    if superlong:
        argv.append("--superlong")
    if shorts_target is not None:
        argv += ["--shorts-target", str(shorts_target)]
    if shorts_gap is not None:
        argv += ["--shorts-gap", str(shorts_gap)]
    argv.append("--no-tmux")
    return argv


def _gate_question_path(output_dir: "str | Path", date_key: str) -> Path:
    return Path(output_dir) / GATE_ANSWERS_SUBDIR / f"gate_cd_{date_key}.question.json"


def _gate_answer_path(output_dir: "str | Path", date_key: str) -> Path:
    return Path(output_dir) / GATE_ANSWERS_SUBDIR / f"gate_cd_{date_key}.answer"


def read_gate_question(output_dir: "str | Path", date_key: str) -> "dict | None":
    """Return the pending gate question payload, or ``None`` if none is waiting."""
    path = _gate_question_path(output_dir, date_key)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_gate_answer(
    output_dir: "str | Path", date_key: str, decision: str, note: str = ""
) -> Path:
    """Write the ``.answer`` file the gate consumes on its next poll.

    Validates ``decision`` against the pending question's ``choices`` when
    one is on disk (raises ``ValueError`` otherwise -- callers turn that
    into an HTTP 400). Retires the question file to ``.question.answered``
    afterwards so :func:`list_pending_gates` naturally stops listing it;
    the pipeline's own answer lookup never depends on the question file
    still existing, so retiring it here is safe.
    """
    question = read_gate_question(output_dir, date_key)
    if question is not None:
        choices = question.get("choices")
        if choices and decision not in choices:
            raise ValueError(f"decision {decision!r} is not one of {choices!r}")
    path = _gate_answer_path(output_dir, date_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "decision": decision,
                "note": note,
                "answered_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    question_path = _gate_question_path(output_dir, date_key)
    if question_path.exists():
        question_path.rename(question_path.with_name(question_path.stem + ".answered"))
    return path


def _gates_dirs(output_dir: "str | Path") -> list[Path]:
    """``<output_dir>/_gates`` plus one level down -- ``_consume_or_ask``
    writes under the PROJECT dir, one level below ``--output-dir``."""
    base = Path(output_dir)
    try:
        children = [base, *sorted(base.iterdir())] if base.is_dir() else [base]
    except OSError:
        children = [base]
    return [d for c in children if (d := c / GATE_ANSWERS_SUBDIR).is_dir()]


def list_pending_gates(output_dirs: "list[str]") -> list[dict]:
    """Auto-discover pending review-gate questions across ``output_dirs``.

    Globs ``*.question.json`` under :func:`_gates_dirs` for EVERY gate id,
    not only ``gate_cd_*``. Each row's ``output_dir`` is the dir that
    CONTAINS ``_gates``; ``date_key`` stays populated only for
    ``gate_cd_<date>`` ids, for the existing answer route.
    """
    pending: list[dict] = []
    for output_dir in output_dirs:
        for gates_dir in _gates_dirs(output_dir):
            container = gates_dir.parent
            for path in sorted(gates_dir.glob("*.question.json")):
                gate_id = path.name[: -len(".question.json")]
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    payload = {}
                try:
                    mtime = path.stat().st_mtime
                except OSError:
                    mtime = 0.0
                date_key = gate_id[len("gate_cd_") :] if gate_id.startswith("gate_cd_") else None
                pending.append(
                    {
                        "output_dir": str(container),
                        "gate_id": gate_id,
                        "date_key": date_key,
                        "question": payload.get("question", ""),
                        "mtime": mtime,
                    }
                )
    return pending


#: A gate ``date_key`` is either an ISO day (``2026-08-01``) or the literal
#: ``final`` (see ``drive_orchestrator.py``'s ``_pending_gate_cd``) -- never a
#: path segment. Restricting to this shape means a traversing value (``..``,
#: ``/``) can never reach ``_gate_question_path``/``_gate_answer_path``, even
#: when ``output_dir`` itself is one this console launched.
_DATE_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def is_valid_gate_date_key(date_key: str) -> bool:
    """Reject anything but the plain token shape gate filenames actually use."""
    return bool(date_key) and _DATE_KEY_RE.fullmatch(date_key) is not None


def resolve_allowed_output_dir(
    output_dir: "str | Path", known_output_dirs: "list[str]"
) -> "Path | None":
    """Match a request-supplied ``output_dir`` against the set this console's
    own job argvs actually launched -- an ``--output-dir`` itself, or the
    project dir one level below it (``_gates_dirs`` writes there too).

    Comparison is on **resolved** paths on both sides, never raw strings: a
    ``..`` segment or a symlink must not let a foreign directory pass by
    spelling itself the same as an allowed one. Returns the resolved
    ``Path`` on a match, ``None`` otherwise -- callers must not echo the
    rejected input back (it would turn the 404 into a filesystem oracle).
    """
    try:
        requested = Path(output_dir).resolve()
    except (OSError, RuntimeError):
        return None
    candidates: list[Path] = []
    for base in known_output_dirs:
        base_path = Path(base)
        candidates.append(base_path)
        try:
            if base_path.is_dir():
                candidates.extend(sorted(p for p in base_path.iterdir() if p.is_dir()))
        except OSError:
            continue
    for candidate in candidates:
        try:
            if candidate.resolve() == requested:
                return requested
        except (OSError, RuntimeError):
            continue
    return None


#: Matches this service's list of supported formats -- the only
#: suffixes the gate media view will ever try to pull a frame from. An
#: artifact with any other suffix (metadata json, csv, srt) never reaches
#: the frame route: it is filtered out here, before it is ever opened.
#: Suffix -> content type for the playback route. Gate C/D asks "Played each
#: LF and reviewed all metadata for <date>?" and a single
#: still frame per video answered it; a reviewer cannot answer that question from a still,
#: and the approval it records is what stops a delivery reaching the queue.
GATE_VIDEO_CONTENT_TYPES = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
    ".mxf": "application/mxf",
}

#: Derived from that table, never a second hand-kept list: a suffix that
#: passed the allowlist with no content type would be a KeyError -- a 500 on a
#: legitimate video.
SUPPORTED_VIDEO_SUFFIXES = frozenset(GATE_VIDEO_CONTENT_TYPES)

_LF_INDEX_RE = re.compile(r"lf_?(\d{2})")


def _lf_index_token(name: str) -> "str | None":
    """Zero-padded LF index a filename embeds (``lf_01.mp4`` / ``..._lf01_
    youtube.json`` both match ``01``) -- used only to pair a video with its
    own metadata artifact, never as a security check."""
    match = _LF_INDEX_RE.search(name)
    return match.group(1) if match else None


def _title_for_video(video_name: str, artifacts: "list[str]") -> str:
    """Best-effort title from the paired ``*_youtube.json`` artifact; ``""``
    when no metadata artifact names the same LF index or it can't be read."""
    token = _lf_index_token(video_name)
    if token is None:
        return ""
    for raw in artifacts:
        candidate = Path(raw)
        if candidate.suffix.lower() != ".json" or _lf_index_token(candidate.name) != token:
            continue
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        return str(data.get("title", ""))
    return ""


def gate_video_paths(resolved_output_dir: "str | Path", question: "dict | None") -> list[Path]:
    """The gate's own video artifacts, contained to ``resolved_output_dir``.

    ``question["artifacts"]`` is ``review_gate.py``'s own record of what a
    gate covers -- never a request-supplied path. An entry is kept only when
    its suffix is a supported video format AND its **resolved** path sits
    inside the already-validated ``resolved_output_dir``; anything else (a
    foreign path, a `..` escape, a non-video artifact) is dropped here so no
    caller can ever index into it. The returned order is the index the
    frame route and :func:`gate_media_items` both key off.
    """
    if not question:
        return []
    base = Path(resolved_output_dir).resolve()
    paths: list[Path] = []
    for raw in question.get("artifacts") or []:
        candidate = Path(raw)
        if candidate.suffix.lower() not in SUPPORTED_VIDEO_SUFFIXES:
            continue
        try:
            resolved = candidate.resolve()
        except (OSError, RuntimeError):
            continue
        if resolved.is_relative_to(base):
            paths.append(resolved)
    return paths


def gate_frame_path(
    resolved_output_dir: "str | Path", question: "dict | None", index: int
) -> "Path | None":
    """The already-contained video path for ``index`` into
    :func:`gate_video_paths`, or ``None`` when out of range -- the ONLY
    lookup the frame route uses, so a request never supplies a filesystem
    path directly, only an index into the gate's own data."""
    paths = gate_video_paths(resolved_output_dir, question)
    if index < 0 or index >= len(paths):
        return None
    return paths[index]


def _proxy_for(path: Path) -> "Path | None":
    """The review proxy sitting alongside ``path``, or ``None``.

    Proxies are optional and named ``<stem>_proxy<suffix>`` next to the
    original artifact -- never a request-supplied path, and never required:
    an absent proxy is the normal case and falls back to the original file.
    """
    candidate = path.with_name(f"{path.stem}_proxy{path.suffix}")
    if candidate.is_file() and candidate.stat().st_size > 0:
        return candidate
    return None


def resolve_gate_video_media(
    resolved_output_dir: "str | Path", question: "dict | None", index: int
) -> "tuple[Path, str] | None":
    """``(path, content_type)`` for ``GET /drives/gates/video``, or ``None``
    for a 404.

    Goes through :func:`gate_frame_path` rather than repeating its lookup, so
    playback is contained by exactly the rule the frame route already is: the
    only request input is an ``index`` into the gate's OWN artifact list, and
    a path that resolves outside the validated output dir, or whose suffix is
    not a supported video, never appears in that list to be indexed. When a
    review proxy sits alongside the resolved video, playback prefers
    it -- same containment, smaller file, no audio.
    """
    path = gate_frame_path(resolved_output_dir, question, index)
    if path is None or not path.is_file():
        return None
    proxy = _proxy_for(path)
    served = proxy if proxy is not None else path
    return served, GATE_VIDEO_CONTENT_TYPES[served.suffix.lower()]


def gate_media_items(resolved_output_dir: "str | Path", question: "dict | None") -> list[dict]:
    """One row per video this gate covers, for the gate template:
    ``{"index", "name", "title", "readable"}``. Never raises -- a missing or
    empty file is a row with ``readable=False`` so the page can render an
    honest "could not be read" state instead of a broken image or a silent
    empty list."""
    artifacts = (question or {}).get("artifacts") or []
    items = []
    for index, path in enumerate(gate_video_paths(resolved_output_dir, question)):
        try:
            readable = path.is_file() and path.stat().st_size > 0
        except OSError:
            readable = False
        items.append(
            {
                "index": index,
                "name": path.name,
                "title": _title_for_video(path.name, artifacts) if readable else "",
                "readable": readable,
            }
        )
    return items


def resolve_allowed_run_file(
    file_path: "str | Path", known_output_dirs: "list[str]"
) -> "Path | None":
    """Match a request-supplied FILE against the runs this console launched.

    :func:`resolve_allowed_output_dir` matches a directory exactly; a run's
    artefacts sit at varying depths beneath one (``<out>/pipeline_state.json``
    and ``<out>/<project>/pipeline_state.json`` are both real), so containment
    here is "inside an allowed directory" rather than "equal to one".

    Both sides are **resolved** before comparing, so a ``..`` segment or a
    symlink cannot walk out of an allowed run and back into someone else's by
    spelling itself with an allowed prefix. Returns the resolved ``Path`` on a
    match and ``None`` otherwise -- callers must not echo the rejected input
    back, which would turn the refusal into a filesystem oracle.

    The file need not exist: a run file that is missing fails honestly in the
    job that opens it, and pre-checking here would leak which paths exist.
    """
    try:
        candidate = Path(file_path).resolve()
    except (OSError, RuntimeError):
        return None
    for base in known_output_dirs:
        try:
            base_path = Path(base).resolve()
        except (OSError, RuntimeError):
            continue
        if candidate.is_relative_to(base_path):
            return candidate
    return None


def output_dirs_from_job_argvs(argv_list: "list[list[str]]") -> list[str]:
    """Extract distinct output dirs from past job argvs -- the set of dirs
    a console-launched job has ever written to, used as the search scope
    for :func:`list_pending_gates` and the publishing enqueue containment
    check.

    A drive run's argv carries ``--output-dir <dir>``; a route-story-film
    run's carries ``--output <file>`` (film.py has no ``--output-dir``) --
    that file's PARENT directory is collected instead, so a
    console-rendered film's dir becomes a known output dir too.
    """
    dirs: list[str] = []
    for argv in argv_list:
        for i, arg in enumerate(argv):
            if arg == "--output-dir" and i + 1 < len(argv):
                candidate = argv[i + 1]
            elif arg == "--output" and i + 1 < len(argv):
                candidate = str(Path(argv[i + 1]).parent)
            else:
                continue
            if candidate not in dirs:
                dirs.append(candidate)
    return dirs


def collect_metadata_previews(output_dir: "str | Path") -> list[dict]:
    """Load each generated LF metadata file under ``output_dir`` and run the
    real content gates against it (read-only -- no mutation, no launch).

    Returns one entry per ``*_youtube.json`` file, sorted by filename:
    ``{"file", "title", "description", "gates": [{"gate", "passed",
    "message"}, ...]}``. Missing/unreadable files are skipped rather than
    raised -- the route decides what an empty result means.
    """
    previews: list[dict] = []
    for path in sorted(Path(output_dir).glob(_METADATA_GLOB)):
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        gates = [
            {"gate": result.gate, "passed": result.passed, "message": result.message}
            for result in (gate_fn(metadata) for gate_fn in _PREVIEW_GATES)
        ]
        previews.append(
            {
                "file": path.name,
                "title": metadata.get("title", ""),
                "description": metadata.get("description", ""),
                "gates": gates,
            }
        )
    return previews
