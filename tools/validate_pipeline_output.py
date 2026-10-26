"""DoD (Definition of Done) validator for the GoMoveShift video pipeline.

The pipeline's Python return value of ``Success: True`` does not guarantee that
the output on disk is actually usable.  This tool audits the run-time output
against a set of concrete deliverables and exits non-zero if any check fails.

Two operating modes (see :class:`Mode`):

``POST_UPLOAD`` (15 checks: #1-6, #8-13, #15 recording date, #19 stop/motion,
#20 ``places_verified``)
    Resolves LF paths via :func:`resolve_post_upload_lfs`: prefers
    ``state["regular_lf_paths"]``, falls back to
    ``<project_dir>/long_forms/lf_NN.mp4`` (legacy delivery tree), then to a
    glob over ``<state_dir>/day_*/regular_lfs/lf_NN.mp4`` (day-first cache).
    Assumes the live ``youtube_uploads.json`` sidecar exists.  Every check runs.

``SKIP_UPLOAD_AUDIT``
    Used when the pipeline ran with ``--skip-upload``.  No delivery-path
    LFs exist yet, so the validator inspects the cache-side layout under
    ``<stages>/day_<date>/regular_lfs/lf_NN.mp4`` + the per-LF metadata
    CSV/JSON sidecars that Phase D writes up-front.  Skips checks that
    require the post-upload layout (thumbnails under ``long_forms/``,
    ``shorts/lf_NN/`` delivery tree, ``youtube_uploads.json`` sha
    comparison).  On pass, the report's footer reads
    ``audit passed (upload not attempted)`` — a clear signal to the
    operator that the run is audit-clean but has NOT been delivered to
    YouTube yet.  This is the validator half of the
    ``--skip-upload → review → --youtube-upload`` two-step publish
    workflow.

Auto-detection:
    :func:`derive_mode_from_state` inspects ``state.skip_upload`` on a
    pipeline state dict.  The orchestrator passes ``state`` into
    :func:`run_validation` which calls this helper when ``mode=Mode.AUTO``
    so the validator always routes to the right check set without the
    caller having to duplicate the decision.

See :func:`main` for the full CLI surface.
"""

from __future__ import annotations

import argparse
import enum
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# PATH first so stock installs (/usr/bin) work; the host build is the fallback.
FFPROBE_BIN = shutil.which("ffprobe") or "/usr/local/bin/ffprobe"

# LF filename is ``lf_NN.mp4`` (two digits by convention, but we accept any
# number of digits so a future run with 100+ LFs still sorts correctly).
LF_NAME_RE = re.compile(r"^lf_(\d+)\.mp4$")
LF_DIR_RE = re.compile(r"^lf_(\d+)$")
# Ghost metadata detection only applies to per-LF YouTube sidecars.
METADATA_NAME_RE = re.compile(r"_lf(\d+)_(?:youtube\.json)$")

# check_lf_durations' superlong-part ceiling fallback; orchestrator overrides.
DEFAULT_SUPERLONG_MAX_DURATION_SEC = 15552.0


def _default_state_dir(project: str) -> str:
    """``<system temp dir>/gomoveshift/<project>/_stages`` (``/tmp`` on Linux)."""
    return str(Path(tempfile.gettempdir()) / "gomoveshift" / project / "_stages")


class Mode(enum.Enum):
    """Validator operating mode.

    * ``AUTO`` — derive from state (falls back to ``POST_UPLOAD`` when
      state is missing or ambiguous); the default for CLI auto-detection.
    * ``POST_UPLOAD`` — the 15-check delivery-path audit.
    * ``SKIP_UPLOAD_AUDIT`` — cache-side LF + metadata audit used when
      the pipeline ran with ``--skip-upload``.
    """

    AUTO = "auto"
    POST_UPLOAD = "post_upload"
    SKIP_UPLOAD_AUDIT = "skip_upload_audit"


def derive_mode_from_state(state: Any | None) -> Mode:
    """Pick the right validator mode from a pipeline ``state`` dict.

    Rules:

    * ``state.get("skip_upload")`` truthy → ``SKIP_UPLOAD_AUDIT``
    * Otherwise → ``POST_UPLOAD`` (preserves existing behaviour for
      callers that don't pass state).

    Accepts anything with a ``.get(key, default)`` method (dict,
    :class:`PipelineState`) so the orchestrator can hand the state in
    directly without conversion.
    """
    if state is None:
        return Mode.POST_UPLOAD
    try:
        skip = bool(state.get("skip_upload"))
    except AttributeError:
        return Mode.POST_UPLOAD
    if skip:
        return Mode.SKIP_UPLOAD_AUDIT
    return Mode.POST_UPLOAD


@dataclass
class CheckResult:
    """One row of the validator output table."""

    number: int
    name: str
    passed: bool
    detail: str


@dataclass
class ValidationReport:
    """Aggregated output of a single validator invocation.

    ``run_validation`` returns this so callers (orchestrator, CLI, tests)
    get a stable object with the full check list plus pre-computed
    pass/fail counts and a formatter for the human-readable table.

    ``mode`` records which validator mode produced the report so
    downstream code (and humans reading the log) can tell a skip-upload
    audit apart from a post-upload one.  The formatter annotates the
    summary line accordingly.
    """

    project: str
    run_start: datetime
    checks: list[CheckResult]
    mode: Mode = field(default=Mode.POST_UPLOAD)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.checks if r.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.checks if not r.passed)

    def format_table(self) -> str:
        """Render the same human-readable table the CLI emits.

        Kept stable so downstream log parsers don't break if the internal
        check list changes.  In ``SKIP_UPLOAD_AUDIT`` mode the summary
        footer gains an explicit "audit passed (upload not attempted)"
        note when all checks pass — operators use that line to confirm
        they still need the ``--youtube-upload`` pass before the run is
        really "done".
        """
        mode_tag = self.mode.value
        lines = [
            f"== DoD Validation: {self.project} "
            f"(run-start={self.run_start.isoformat()}, mode={mode_tag}) =="
        ]
        for result in self.checks:
            tag = "PASS" if result.passed else "FAIL"
            lines.append(f"[{tag}] {result.number:02d} {result.name} — {result.detail}")
        summary = f"Summary: {self.passed_count}/{len(self.checks)} checks passed"
        if self.mode == Mode.SKIP_UPLOAD_AUDIT and self.failed_count == 0:
            summary += " — audit passed (upload not attempted)"
        lines.append(summary)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ffprobe wrapper
# ---------------------------------------------------------------------------


def ffprobe(path: Path) -> dict:
    """Call ffprobe and return the parsed JSON.

    Raises ``RuntimeError`` if ffprobe fails or the output cannot be parsed.
    """
    result = subprocess.run(  # noqa: S603 — fixed binary, argv list, no shell
        [
            FFPROBE_BIN,
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ffprobe returned non-JSON for {path}: {exc}") from exc


def _stream_duration(stream: dict, fmt: dict) -> float | None:
    """Return duration for a stream, falling back to the container."""
    for key in ("duration",):
        value = stream.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    value = fmt.get("duration")
    if value is not None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def discover_lfs(long_forms_dir: Path) -> list[tuple[int, Path]]:
    """Return sorted ``[(index, path)]`` for every ``lf_NN.mp4`` under the dir.

    ``*.video_only.mp4`` intermediate files are excluded.
    """
    if not long_forms_dir.is_dir():
        return []
    out: list[tuple[int, Path]] = []
    for entry in long_forms_dir.iterdir():
        if not entry.is_file():
            continue
        if entry.name.endswith(".video_only.mp4"):
            continue
        match = LF_NAME_RE.match(entry.name)
        if match:
            out.append((int(match.group(1)), entry))
    out.sort(key=lambda t: t[0])
    return out


def discover_state_lfs(state: Any | None) -> list[tuple[int, Path]]:
    """Discover LFs from ``state["regular_lf_paths"]`` (day-first canonical).

    The day-first orchestrator (`phase_bc_day_first`) writes every rendered
    LF's absolute path into ``state["regular_lf_paths"]`` (see
    ``phase_bc_day_first.py``). Phase D + downstream consumers read
    from this same field (``phase_d.py``). The DoD validator
    follows that contract here so day-first runs no longer false-FAIL on
    the first two checks just because Phase D never copied LFs into the
    legacy ``<project>/long_forms/`` tree.

    Filename convention is ``lf_NN.mp4`` regardless of where it lives, so
    the LF index is parsed from the basename. Entries whose paths do not
    exist on disk OR whose basenames do not match the convention are
    skipped — `discover_lfs` (legacy) and `discover_cache_side_lfs`
    (glob fallback) handle the rest.

    Returns ``[(idx, path), ...]`` sorted by index.
    """
    if state is None:
        return []
    try:
        getter = state.get
    except AttributeError:
        return []
    raw = getter("regular_lf_paths") or []
    if not isinstance(raw, list):
        return []
    out: list[tuple[int, Path]] = []
    seen: set[int] = set()
    for entry in raw:
        if entry is None:
            continue
        path = Path(entry) if not isinstance(entry, Path) else entry
        if not path.exists():
            continue
        match = LF_NAME_RE.match(path.name)
        if not match:
            continue
        idx = int(match.group(1))
        if idx in seen:
            continue
        seen.add(idx)
        out.append((idx, path))
    out.sort(key=lambda t: t[0])
    return out


def discover_state_superlong_parts(state: Any | None) -> list[tuple[int, Path]]:
    """Discover opt-in superlong parts (``sl_NN.mp4``) from
    ``state["superlong_parts"][date] == {"records": [{"path": ...,
    "part_index": ...}]}``. Never in ``regular_lf_paths``/``LF_NAME_RE``
    — invisible to every ffprobe-based check before this. Returns
    ``[(idx, path), ...]`` sorted by ``(date, part_index)``; synthetic
    indices are negative so they can't collide with a real ``lf_NN``
    index in the shared ``probes`` dict.
    """
    if state is None:
        return []
    try:
        getter = state.get
    except AttributeError:
        return []
    raw = getter("superlong_parts") or {}
    if not isinstance(raw, dict):
        return []
    entries: list[tuple[str, int, Path]] = []
    for date, day_plan in raw.items():
        if not isinstance(day_plan, dict):
            continue
        for record in day_plan.get("records") or []:
            if not isinstance(record, dict) or not record.get("path"):
                continue
            path = Path(record["path"])
            if not path.exists():
                continue
            try:
                part_index = int(record.get("part_index", 0))
            except (TypeError, ValueError):
                part_index = 0
            entries.append((str(date), part_index, path))
    entries.sort()
    return [(-(n + 1), path) for n, (*_key, path) in enumerate(entries)]


def resolve_post_upload_lfs(
    project_dir: Path,
    state: Any | None,
    state_dir: Path,
) -> tuple[list[tuple[int, Path]], str]:
    """Resolve LF (idx, path) pairs for POST_UPLOAD checks with a clear
    fallback ladder, and return the scheme that won so failure messages can
    name it.

    Resolution order:

    1. ``state["regular_lf_paths"]`` — the canonical day-first source
       used by Phase D. Skipped when state is missing/empty.
    2. ``<project_dir>/long_forms/lf_NN.mp4`` — the legacy delivery-tree
       layout (pre-day-first, still used by smoke tests + non-day-first
       integration runs).
    3. ``<state_dir>/day_*/regular_lfs/lf_NN.mp4`` — glob fallback when
       neither state nor delivery tree carries the LFs (e.g. orchestrator
       died before writing state but day pipeline left finals on disk).

    Scheme strings (returned as the second tuple element):
    ``"state"``, ``"legacy"``, ``"day_first_glob"``, or ``"none"``.
    Callers pass the scheme into ``check_lf_count``'s detail so a failing
    audit names which path scheme was tried.
    """
    lfs = discover_state_lfs(state)
    if lfs:
        return lfs, "state"
    lfs = discover_lfs(project_dir / "long_forms")
    if lfs:
        return lfs, "legacy"
    lfs = discover_cache_side_lfs(state_dir)
    if lfs:
        return lfs, "day_first_glob"
    return [], "none"


def discover_cache_side_lfs(stages_dir: Path) -> list[tuple[int, Path]]:
    """Discover every ``lf_NN.mp4`` under the cache-side day layout.

    ``--skip-upload`` runs produce LFs at:

        <stages>/day_<YYYY-MM-DD>/regular_lfs/lf_NN.mp4

    and never copy them into ``<project>/long_forms/``.  This helper
    walks the ``day_*`` subdirs, collects every matching LF file, and
    returns them sorted by LF index.

    Duplicate indices (two days carrying the same LF number, which can
    happen if a day re-encode half-completed) are resolved by keeping
    the most recently modified file.  Missing ``stages_dir`` yields an
    empty list — caller decides how to surface that.
    """
    if not stages_dir.is_dir():
        return []
    by_index: dict[int, Path] = {}
    for day_dir in stages_dir.iterdir():
        if not day_dir.is_dir() or not day_dir.name.startswith("day_"):
            continue
        lf_dir = day_dir / "regular_lfs"
        if not lf_dir.is_dir():
            continue
        for entry in lf_dir.iterdir():
            if not entry.is_file():
                continue
            if entry.name.endswith(".video_only.mp4"):
                continue
            match = LF_NAME_RE.match(entry.name)
            if match is None:
                continue
            idx = int(match.group(1))
            existing = by_index.get(idx)
            if existing is None or entry.stat().st_mtime > existing.stat().st_mtime:
                by_index[idx] = entry
    return sorted(by_index.items(), key=lambda t: t[0])


def sha256(path: Path) -> str:
    """Stream-hash a file with SHA-256."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_iso(value: str) -> datetime:
    """Parse an ISO-8601 timestamp, treating naive strings as UTC."""
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_lf_count(
    lfs: list[tuple[int, Path]],
    expected_min: int,
    scheme: str | None = None,
) -> CheckResult:
    passed = len(lfs) >= expected_min
    detail = f"{len(lfs)} {'>=' if passed else '<'} {expected_min}"
    if scheme is not None:
        detail += f" (path scheme: {scheme})"
    return CheckResult(
        1,
        "lf_count",
        passed,
        detail,
    )


def check_lf_files_size(lfs: list[tuple[int, Path]]) -> CheckResult:
    min_bytes = 100 * 1024 * 1024
    problems: list[str] = []
    for _, path in lfs:
        if not path.exists():
            problems.append(f"{path.name} missing")
            continue
        size = path.stat().st_size
        if size <= min_bytes:
            problems.append(f"{path.name} {size} bytes <= 100 MB")
    if not lfs:
        return CheckResult(2, "lf_files_size", False, "no LFs discovered")
    if problems:
        return CheckResult(2, "lf_files_size", False, "; ".join(problems))
    return CheckResult(2, "lf_files_size", True, f"all {len(lfs)} LFs > 100 MB")


def check_lf_durations(
    probes: dict[int, dict],
    lfs: list[tuple[int, Path]],
    min_dur: float,
    max_dur: float,
    superlong_parts: list[tuple[int, Path]] | None = None,
    superlong_max_dur: float | None = None,
) -> CheckResult:
    """Duration ceiling — PER-ARTIFACT: superlong parts (optional) check
    against their own ceiling; the regular-LF ceiling is never widened."""
    problems: list[str] = []
    for idx, path in lfs:
        probe = probes.get(idx)
        if probe is None:
            problems.append(f"{path.name} unprobed")
            continue
        try:
            duration = float(probe.get("format", {}).get("duration", 0.0))
        except (TypeError, ValueError):
            duration = 0.0
        if duration < min_dur or duration > max_dur:
            problems.append(f"{path.name} duration={duration:.1f}s outside [{min_dur}, {max_dur}]")
    total = len(lfs)
    if superlong_parts:
        ceiling = max_dur if superlong_max_dur is None else superlong_max_dur
        total += len(superlong_parts)
        for idx, path in superlong_parts:
            probe = probes.get(idx)
            if probe is None:
                problems.append(f"{path.name} unprobed")
                continue
            try:
                duration = float(probe.get("format", {}).get("duration", 0.0))
            except (TypeError, ValueError):
                duration = 0.0
            if duration <= 0 or duration > ceiling:
                problems.append(f"{path.name} sl duration={duration:.1f}s outside (0,{ceiling}]")
    if not lfs and not superlong_parts:
        return CheckResult(3, "lf_duration", False, "no LFs discovered")
    if problems:
        return CheckResult(3, "lf_duration", False, "; ".join(problems))
    return CheckResult(3, "lf_duration", True, f"all {total} LFs within bounds")


def check_av_drift(
    probes: dict[int, dict],
    lfs: list[tuple[int, Path]],
    max_drift_pct: float,
) -> CheckResult:
    problems: list[str] = []
    for idx, path in lfs:
        probe = probes.get(idx)
        if probe is None:
            problems.append(f"{path.name} unprobed")
            continue
        fmt = probe.get("format", {})
        video_dur: float | None = None
        audio_dur: float | None = None
        for stream in probe.get("streams", []):
            duration = _stream_duration(stream, fmt)
            if duration is None:
                continue
            codec_type = stream.get("codec_type")
            if codec_type == "video" and video_dur is None:
                video_dur = duration
            elif codec_type == "audio" and audio_dur is None:
                audio_dur = duration
        if video_dur is None or audio_dur is None:
            problems.append(f"{path.name} missing video/audio duration")
            continue
        if video_dur <= 0:
            problems.append(f"{path.name} video duration <= 0")
            continue
        drift = abs(audio_dur - video_dur) / video_dur * 100.0
        if drift >= max_drift_pct:
            problems.append(f"{path.name} drift={drift:.1f}% exceeds {max_drift_pct}%")
    if not lfs:
        return CheckResult(4, "av_drift", False, "no LFs discovered")
    if problems:
        return CheckResult(4, "av_drift", False, "; ".join(problems))
    return CheckResult(4, "av_drift", True, f"all {len(lfs)} LFs within drift budget")


def check_audio_stream(probes: dict[int, dict], lfs: list[tuple[int, Path]]) -> CheckResult:
    problems: list[str] = []
    for idx, path in lfs:
        probe = probes.get(idx)
        if probe is None:
            problems.append(f"{path.name} unprobed")
            continue
        has_audio = any(stream.get("codec_type") == "audio" for stream in probe.get("streams", []))
        if not has_audio:
            problems.append(f"{path.name} no audio stream")
    if not lfs:
        return CheckResult(5, "audio_stream", False, "no LFs discovered")
    if problems:
        return CheckResult(5, "audio_stream", False, "; ".join(problems))
    return CheckResult(5, "audio_stream", True, f"all {len(lfs)} LFs have audio")


def check_video_hevc(probes: dict[int, dict], lfs: list[tuple[int, Path]]) -> CheckResult:
    problems: list[str] = []
    for idx, path in lfs:
        probe = probes.get(idx)
        if probe is None:
            problems.append(f"{path.name} unprobed")
            continue
        codecs = [
            stream.get("codec_name")
            for stream in probe.get("streams", [])
            if stream.get("codec_type") == "video"
        ]
        if not codecs:
            problems.append(f"{path.name} no video stream")
            continue
        if codecs[0] != "hevc":
            problems.append(f"{path.name} video codec={codecs[0]} (expected hevc)")
    if not lfs:
        return CheckResult(6, "video_hevc", False, "no LFs discovered")
    if problems:
        return CheckResult(6, "video_hevc", False, "; ".join(problems))
    return CheckResult(6, "video_hevc", True, f"all {len(lfs)} LFs are hevc")


def _check_metadata_fresh(
    number: int,
    name: str,
    project_dir: Path,
    lfs: list[tuple[int, Path]],
    project: str,
    suffix: str,
    run_start: datetime,
) -> CheckResult:
    if not lfs:
        return CheckResult(number, name, False, "no LFs discovered")
    problems: list[str] = []
    # drvfs (WSL /mnt/) uses the Windows NTFS clock which can lag behind
    # the Linux clock by 20-30 seconds.  Allow a 60 s tolerance so files
    # written at pipeline start are not falsely flagged as stale.
    drvfs_skew = 60 if str(project_dir).startswith("/mnt/") else 0
    run_start_ts = run_start.timestamp() - drvfs_skew
    for idx, _path in lfs:
        fname = f"{project}_lf{idx:02d}_{suffix}"
        target = project_dir / fname
        if not target.exists():
            problems.append(f"{fname} missing")
            continue
        mtime = target.stat().st_mtime
        if mtime < run_start_ts:
            problems.append(
                f"{fname} mtime={datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()} "
                f"< run_start={run_start.isoformat()}"
            )
    if problems:
        return CheckResult(number, name, False, "; ".join(problems))
    return CheckResult(number, name, True, f"all {len(lfs)} {suffix} files fresh")


def check_youtube_json_fresh(
    project_dir: Path,
    lfs: list[tuple[int, Path]],
    project: str,
    run_start: datetime,
) -> CheckResult:
    return _check_metadata_fresh(
        8, "youtube_json_fresh", project_dir, lfs, project, "youtube.json", run_start
    )


def check_thumbnails(project_dir: Path, lfs: list[tuple[int, Path]]) -> CheckResult:
    thumb_dir = project_dir / "thumbnails"
    if not thumb_dir.is_dir():
        return CheckResult(9, "thumbnails", False, f"{thumb_dir} missing")
    if not lfs:
        return CheckResult(9, "thumbnails", False, "no LFs discovered")
    problems: list[str] = []
    for idx, _path in lfs:
        n_lf = f"lf_{idx:02d}"
        matches = [
            entry
            for entry in thumb_dir.iterdir()
            if entry.is_file() and (entry.name.startswith(n_lf) or f"_{n_lf}" in entry.name)
        ]
        if not matches:
            problems.append(f"no thumbnail for {n_lf}")
    if problems:
        return CheckResult(9, "thumbnails", False, "; ".join(problems))
    return CheckResult(9, "thumbnails", True, f"all {len(lfs)} LFs have thumbnails")


def check_no_ghost_metadata(
    project_dir: Path, lfs: list[tuple[int, Path]], project: str
) -> CheckResult:
    if not project_dir.is_dir():
        return CheckResult(10, "no_ghost_metadata", False, f"{project_dir} missing")
    max_idx = max((idx for idx, _ in lfs), default=-1)
    prefix = f"{project}_lf"
    ghosts: list[str] = []
    for entry in project_dir.iterdir():
        if not entry.is_file():
            continue
        if not entry.name.startswith(prefix):
            continue
        match = METADATA_NAME_RE.search(entry.name)
        if not match:
            continue
        idx = int(match.group(1))
        if idx > max_idx:
            ghosts.append(entry.name)
    if ghosts:
        return CheckResult(
            10,
            "no_ghost_metadata",
            False,
            f"stale metadata (max_lf={max_idx}): " + ", ".join(sorted(ghosts)),
        )
    return CheckResult(10, "no_ghost_metadata", True, f"no metadata past lf_{max_idx:02d}")


def check_shorts_count(project_dir: Path, state: Any | None = None) -> CheckResult:
    # Day-first / full-drive: short MP4s live under
    # ``<stages>/day_*/shorts/short_NN.mp4`` and are tracked authoritatively
    # in ``state["shorts_per_lf"]``. The legacy globbing path below
    # only sees ``project_dir/shorts/lf_NN/social_posts.json`` because the
    # MP4s are uploaded directly from cache and never copied to
    # ``project_dir``. Consume state when present so day-first runs are
    # validated against the real delivery, not an empty legacy directory.
    shorts_per_lf: Any = None
    if state is not None:
        shorts_per_lf = getattr(state, "shorts_per_lf", None)
        if shorts_per_lf is None and isinstance(state, dict):
            shorts_per_lf = state.get("shorts_per_lf")
    if shorts_per_lf:
        problems: list[str] = []
        inspected = 0
        for lf_key, paths in shorts_per_lf.items():
            inspected += 1
            try:
                lf_idx = int(lf_key)
            except (TypeError, ValueError):
                problems.append(f"lf_{lf_key}: non-integer key")
                continue
            if not paths:
                problems.append(f"lf_{lf_idx:02d}: no shorts in state")
                continue
            for p in paths:
                pth = Path(p)
                if not pth.is_file():
                    problems.append(f"lf_{lf_idx:02d}: {pth.name} missing")
                elif pth.stat().st_size == 0:
                    problems.append(f"lf_{lf_idx:02d}: {pth.name} empty")
        if inspected == 0:
            return CheckResult(11, "shorts_count", False, "state.shorts_per_lf empty")
        if problems:
            return CheckResult(11, "shorts_count", False, "; ".join(problems))
        return CheckResult(
            11,
            "shorts_count",
            True,
            f"all {inspected} LFs' shorts present per state.shorts_per_lf",
        )
    shorts_root = project_dir / "shorts"
    if not shorts_root.is_dir():
        return CheckResult(11, "shorts_count", False, f"{shorts_root} missing")
    problems = []
    inspected = 0
    for entry in sorted(shorts_root.iterdir()):
        if not entry.is_dir():
            continue
        if not LF_DIR_RE.match(entry.name):
            continue
        inspected += 1
        mp4_count = sum(
            1
            for child in entry.iterdir()
            if child.is_file() and child.name.startswith("short_") and child.name.endswith(".mp4")
        )
        json_path = entry / "shorts_rendered.json"
        if not json_path.exists():
            problems.append(f"{entry.name}/shorts_rendered.json missing")
            continue
        try:
            data = json.loads(json_path.read_text())
        except json.JSONDecodeError as exc:
            problems.append(f"{entry.name}/shorts_rendered.json invalid: {exc}")
            continue
        expected = data.get("num_rendered")
        if expected is None:
            problems.append(f"{entry.name}/shorts_rendered.json missing num_rendered")
            continue
        if mp4_count != expected:
            problems.append(
                f"{entry.name}: found {mp4_count} short_*.mp4 but num_rendered={expected}"
            )
    if inspected == 0:
        return CheckResult(11, "shorts_count", False, "no shorts/lf_* subdirs")
    if problems:
        return CheckResult(11, "shorts_count", False, "; ".join(problems))
    return CheckResult(11, "shorts_count", True, f"all {inspected} shorts dirs consistent")


def check_schedule_length(
    state_dir: Path, expected_weeks: int, expected_lf_min: int = 1
) -> CheckResult:
    """Schedule is well-formed for the run's declared scope.

    LF-coverage: ``len(lf_slots with dates) >= expected_lf_min``. A 1-day
    smoke with one LF correctly schedules one LF slot — rejecting it as
    "not enough slots" is a scope mismatch, not a real failure.

    Span: the combined LF + shorts calendar-day range must cover what the
    schedule's own scope needs, never more than ``expected_weeks * 7 - 1``
    days (inclusive of both endpoints, so one week of shorts = Fri through
    Thu = 6 days span). Shorts slots count toward span because single-LF
    weeks still publish shorts across the full week.

    Scope-aware requirement (an interim ``--shorts-target 2`` run must not
    exit "failed DoD validation" after uploading):

    * shorts need ``ceil(n_shorts / per_day)`` calendar days, where
      ``per_day`` is the run-wide ``drive_schedule.identity_times`` pair the
      scheduler locks (docs/pipeline/79) — absent, one a day;
    * LFs need ``(expected_weeks - 1) * 7`` days;
    * ``required = min(expected_weeks * 7 - 1, max(lf_days, shorts_days))``.

    So 1 LF + 2 shorts on a two-times-a-day region is complete at 0d (the
    LF may share the first short's day), while 14 shorts collapsed onto one
    day still fail. Span counts calendar days, not 24-hour periods: a Tue
    16:00 LF and a Wed 14:00 short span 1 day.
    """
    state_file = state_dir / "pipeline_state.json"
    if not state_file.exists():
        return CheckResult(12, "schedule_length", False, f"{state_file} missing")
    try:
        data = json.loads(state_file.read_text())
    except json.JSONDecodeError as exc:
        return CheckResult(12, "schedule_length", False, f"pipeline_state.json invalid: {exc}")
    schedule = data.get("drive_schedule") or {}
    lf_slots = schedule.get("lf_slots")
    if not lf_slots or not isinstance(lf_slots, list):
        return CheckResult(12, "schedule_length", False, "drive_schedule.lf_slots missing or empty")

    def _collect(slots: object) -> list[datetime]:
        out: list[datetime] = []
        if not isinstance(slots, list):
            return out
        for slot in slots:
            value = slot.get("date") if isinstance(slot, dict) else None
            if value is None:
                continue
            try:
                out.append(parse_iso(str(value)))
            except ValueError:
                continue
        return out

    lf_dates = _collect(lf_slots)
    if len(lf_dates) < expected_lf_min:
        return CheckResult(
            12,
            "schedule_length",
            False,
            f"need >={expected_lf_min} LF slots with dates, got {len(lf_dates)}",
        )

    shorts_dates = _collect(schedule.get("shorts_slots"))
    combined = sorted(lf_dates + shorts_dates)
    span_days = (combined[-1].date() - combined[0].date()).days if combined else 0
    identity_times = schedule.get("identity_times")
    per_day = max(1, len(identity_times) if isinstance(identity_times, list) else 0)
    shorts_days_required = max(0, math.ceil(len(shorts_dates) / per_day) - 1)
    lf_days_required = max(0, expected_weeks - 1) * 7
    required = max(0, min(expected_weeks * 7 - 1, max(lf_days_required, shorts_days_required)))
    if span_days < required:
        return CheckResult(
            12,
            "schedule_length",
            False,
            f"combined schedule span {span_days}d < required {required}d "
            f"({expected_weeks} weeks, {len(lf_dates)} LF + {len(shorts_dates)} shorts slots, "
            f"{per_day} shorts/day)",
        )
    return CheckResult(
        12,
        "schedule_length",
        True,
        f"{len(lf_dates)} LF slot(s), combined span {span_days}d >= required {required}d",
    )


def check_youtube_uploads_unchanged(
    project_dir: Path,
    snapshot_dir: Path,
    expected_sha: str | None = None,
) -> CheckResult:
    live = project_dir / "youtube_uploads.json"
    if not live.exists():
        return CheckResult(
            13,
            "youtube_uploads_unchanged",
            False,
            f"live {live} missing (pipeline deleted it?)",
        )
    live_hash = sha256(live)

    # If an explicit SHA was provided by the pipeline, hard-assert against it.
    if expected_sha is not None:
        if live_hash != expected_sha:
            return CheckResult(
                13,
                "youtube_uploads_unchanged",
                False,
                f"sha256 mismatch: live={live_hash[:12]} expected={expected_sha[:12]}",
            )
        return CheckResult(
            13,
            "youtube_uploads_unchanged",
            True,
            f"sha256 matches expected ({live_hash[:12]})",
        )

    # Fallback: compare live vs snapshot on disk.
    snap = snapshot_dir / "youtube_uploads.json"
    if not snap.exists():
        return CheckResult(
            13,
            "youtube_uploads_unchanged",
            False,
            f"snapshot {snap} missing",
        )
    snap_hash = sha256(snap)
    if live_hash != snap_hash:
        return CheckResult(
            13,
            "youtube_uploads_unchanged",
            False,
            f"sha256 mismatch: live={live_hash[:12]} snapshot={snap_hash[:12]}",
        )
    return CheckResult(13, "youtube_uploads_unchanged", True, f"sha256 matches ({live_hash[:12]})")


# ---------------------------------------------------------------------------
# Content-fidelity checks
# ---------------------------------------------------------------------------
#
# Two classes of defect that can slip past an audit:
#
#   1. a text column truncated mid-emoji or mid-word.
#   2. one recording date applied to every LF instead of each LF's own day.
#
# Check #15 catches the second class. It runs in BOTH ``skip_upload_audit``
# (skip-upload audit) and ``post_upload`` modes so issues surface as early
# as Phase D's metadata write — before the operator commits to
# ``--youtube-upload``.

# Acceptable CSV-text terminators: punctuation, quotes/closers, whitespace,
# end-of-line.  A clean text body either ends at one of these OR ends with
# a hashtag like ``#word``.
_CLEAN_BOUNDARY_CHARS = set(".…!?\"')]}\n \t")
_HASHTAG_RE = re.compile(r"#\w+\Z")
# ``📅 Filmed: <date>`` (canonical) OR legacy ``📹 Recorded: …`` line. The
# trailing value is captured verbatim and normalized by ``_parse_filmed_date`` /
# ``_filmed_dates_match``;
# the canonical render is ``Mon YYYY`` (e.g. ``Sep 2025``, the shared
# formatter), with legacy ISO ``YYYY-MM-DD`` still tolerated so historical
# descriptions validate. New emissions use ``📅 Filmed:`` (see
# ``longform_metadata.py`` / ``social_generator.format_recording_date``).
_RECORDING_DATE_RE = re.compile(r"(?:\U0001f4c5\s*Filmed|\U0001f4f9\s*Recorded):\s*([^\n]+)")


def _parse_filmed_date(value: str) -> tuple[int, int, int | None] | None:
    """Parse a ``📅 Filmed:`` value to ``(year, month, day_or_None)``.

    Accepts the canonical ``Mon YYYY`` render (``"Sep 2025"``, the shared
    ``social_generator.format_recording_date`` shape used by BOTH long-form
    descriptions and child shorts) and ``"Month YYYY"`` — both month-precision,
    so ``day`` is ``None``. Also accepts legacy day-bearing ISO ``YYYY-MM-DD``
    and ``"Month DD, YYYY"``. Returns ``None`` when the value is unparseable.
    """
    value = value.strip()
    for fmt, has_day in (
        ("%Y-%m-%d", True),
        ("%B %d, %Y", True),
        ("%b %Y", False),
        ("%B %Y", False),
    ):
        try:
            parsed = datetime.strptime(value, fmt)
        except ValueError:
            continue
        return (parsed.year, parsed.month, parsed.day if has_day else None)
    return None


def _filmed_dates_match(actual: str, expected: str) -> bool | None:
    """Whether the rendered Filmed ``actual`` matches the ``expected`` recording
    date, compared at the *rendered value's* precision.

    Returns ``None`` when ``actual`` is unparseable (a real defect — garbage
    burned into the published surface). The canonical ``Mon YYYY`` render drops
    the day, so a month-precision ``actual`` is compared at month granularity;
    a legacy day-bearing ISO ``actual`` is still compared at full day precision.
    """
    a = _parse_filmed_date(actual)
    if a is None:
        return None
    e = _parse_filmed_date(expected)
    if e is None:
        # No parseable authority to compare against — don't manufacture a fail.
        return True
    if a[2] is not None and e[2] is not None:
        return a == e
    return a[:2] == e[:2]


def _read_youtube_description(path: Path) -> str | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    desc = data.get("description") if isinstance(data, dict) else None
    if isinstance(desc, str):
        return desc
    return None


def _ends_at_clean_boundary(text: str) -> bool:
    """Heuristic: does ``text`` end at a sentence/paragraph/hashtag boundary?

    Mid-emoji or mid-word truncation yields ``False``.  Hashtag tail (``"... #travel"``)
    yields ``True`` even though the last char is a word char.
    """
    if not text:
        return False
    if _HASHTAG_RE.search(text.rstrip()):
        return True
    return text[-1] in _CLEAN_BOUNDARY_CHARS


# DoD check #14 (shorts CSV text fidelity) is retired: per-LF shorts CSVs are
# no longer produced. LF-side text fidelity is covered by
# ``check_youtube_json_fresh`` + ``check_no_ghost_metadata``.


def _expected_recording_date(state: Any, idx: int) -> str | None:
    """Best-effort: derive the expected recording date for LF ``idx`` from state.

    Tries (in order):
      1. ``state["per_lf_recording_date"][idx]`` (authoritative source)
      2. ``state["episode_dates"][state["per_lf_first_segment_index"][idx]]``
      3. None — caller treats as "skipped".
    """
    if state is None:
        return None
    try:
        getter = state.get
    except AttributeError:
        return None
    per_lf = getter("per_lf_recording_date") or {}
    if isinstance(per_lf, dict):
        # Keys may be int or str depending on serialization round-trip.
        for key in (idx, str(idx), f"{idx:02d}"):
            if key in per_lf:
                value = per_lf[key]
                if isinstance(value, str) and value.strip():
                    return value.strip()
                if hasattr(value, "strftime"):
                    return value.strftime("%Y-%m-%d")
    episode_dates = getter("episode_dates") or []
    first_seg_map = getter("per_lf_first_segment_index") or {}
    if isinstance(first_seg_map, dict) and isinstance(episode_dates, list):
        for key in (idx, str(idx), f"{idx:02d}"):
            if key in first_seg_map:
                seg_idx = first_seg_map[key]
                try:
                    seg_idx_int = int(seg_idx)
                except (TypeError, ValueError):
                    continue
                if 0 <= seg_idx_int < len(episode_dates):
                    value = episode_dates[seg_idx_int]
                    if isinstance(value, str) and value.strip():
                        return value.strip()
                    if hasattr(value, "strftime"):
                        return value.strftime("%Y-%m-%d")
    return None


def check_recording_date_per_lf(
    project_dir: Path,
    lfs: list[tuple[int, Path]],
    project: str,
    state: Any | None,
) -> CheckResult:
    """Verify each youtube.json's ``📅 Filmed:`` (legacy ``📹 Recorded:``) line matches the LF's day.

    The expected date is sourced from ``state`` (see
    :func:`_expected_recording_date`).  When state has nothing to compare
    against (e.g. neither ``per_lf_recording_date`` nor ``episode_dates`` is populated) the
    check returns PASS with a "skipped" detail rather than fail blindly —
    you can't audit without an authority.
    """
    if not lfs:
        return CheckResult(15, "recording_date_per_lf", False, "no LFs discovered")
    expected_per_lf: dict[int, str] = {}
    for idx, _ in lfs:
        expected = _expected_recording_date(state, idx)
        if expected:
            expected_per_lf[idx] = expected
    if not expected_per_lf:
        return CheckResult(
            15,
            "recording_date_per_lf",
            True,
            "skipped (no expected recording dates in state — needs per_lf_recording_date or episode_dates)",
        )
    problems: list[str] = []
    inspected = 0
    for idx, _path in lfs:
        json_path = project_dir / f"{project}_lf{idx:02d}_youtube.json"
        if not json_path.exists():
            problems.append(f"lf{idx:02d}: youtube.json missing")
            continue
        desc = _read_youtube_description(json_path)
        if desc is None:
            problems.append(f"lf{idx:02d}: youtube.json description missing/unparseable")
            continue
        expected = expected_per_lf.get(idx)
        if expected is None:
            # No authority for this LF — skip silently (we already reported
            # "skipped" above only when nothing was authoritative; here some
            # LFs have authority and some don't, so skip the unauthoritative
            # ones individually).
            continue
        match = _RECORDING_DATE_RE.search(desc)
        if not match:
            problems.append(
                f"lf{idx:02d}: youtube.json description missing '📅 Filmed:' line "
                f"(expected {expected})"
            )
            continue
        actual = match.group(1).strip()
        inspected += 1
        # Compare at the rendered value's precision: canonical ``Mon YYYY``
        # (day dropped) → month granularity; legacy ISO → full day. A None
        # result means an unparseable Filmed value — a real defect.
        verdict = _filmed_dates_match(actual, expected)
        if verdict is None:
            problems.append(f"lf{idx:02d}: unparseable '📅 Filmed:' value {actual!r}")
        elif not verdict:
            problems.append(f"lf{idx:02d}: recording date {actual} != expected {expected}")
    if problems:
        return CheckResult(
            15,
            "recording_date_per_lf",
            False,
            "; ".join(problems),
        )
    return CheckResult(
        15,
        "recording_date_per_lf",
        True,
        f"all {inspected} LFs have correct recording date",
    )


# ---------------------------------------------------------------------------
# Stop/motion check (parked-short backstop)
# ---------------------------------------------------------------------------
#
# The per-day STOP mask (``stop_mask_<date>.npz``, True == a stopped second)
# structurally excludes parked footage from the plan, and the upstream
# parked-footage content gate rejects a majority-parked short before the burn.
# Check #19 is the late-gate backstop: it re-derives, from the persisted mask + the shipped
# ``shorts_ost_audit_<date>.json`` retained windows, whether any shipped short is
# majority stationary. No-op when neither artefact is present (GPS-less / legacy
# / stop-mask-disabled), so historical runs never false-FAIL.

_STOP_MASK_RE = re.compile(r"^stop_mask_(.+)\.npz$")
_SHORTS_AUDIT_RE = re.compile(r"^shorts_ost_audit_(.+)\.json$")


def _load_stop_mask(npz_path: Path):
    """Return the persisted per-second STOP mask array, or ``None`` on failure.

    numpy is imported lazily so the validator never hard-requires it just to run
    the delivery-path checks that don't touch the mask.
    """
    try:
        import numpy as np
    except Exception:  # pragma: no cover — numpy missing
        return None
    try:
        with np.load(npz_path) as data:
            return data["mask"]
    except Exception:
        return None


def _first_by_date(roots: list[Path | None], regex: re.Pattern) -> dict[str, Path]:
    """Map ``date -> path`` for files matching ``regex`` under any root (rglob)."""
    out: dict[str, Path] = {}
    for root in roots:
        if root is None or not Path(root).is_dir():
            continue
        for p in Path(root).rglob("*"):
            if not p.is_file():
                continue
            m = regex.match(p.name)
            if m and m.group(1) not in out:
                out[m.group(1)] = p
    return out


def check_stop_motion(
    project_dir: Path,
    state_dir: Path,
    *,
    max_stationary_fraction: float = 0.5,
) -> CheckResult:
    """DoD #19: no shipped short's retained window is majority stationary.

    Pairs each ``stop_mask_<date>.npz`` with its ``shorts_ost_audit_<date>.json``
    (each short's day-master ``start_sec``/``end_sec``) and FAILS when more than
    ``max_stationary_fraction`` of a short window's seconds are masked stopped.
    PASS (no-op) when either artefact is absent — GPS-less / legacy /
    stop-mask-disabled runs never false-FAIL.
    """
    masks = _first_by_date([state_dir], _STOP_MASK_RE)
    audits = _first_by_date([state_dir, project_dir], _SHORTS_AUDIT_RE)
    if not masks or not audits:
        return CheckResult(
            19,
            "stop_motion",
            True,
            "skipped (no persisted stop mask / shorts audit — GPS-less/legacy)",
        )
    problems: list[str] = []
    inspected = 0
    for date, audit_path in sorted(audits.items()):
        mask_path = masks.get(date)
        if mask_path is None:
            continue
        mask = _load_stop_mask(mask_path)
        if mask is None or len(mask) == 0:
            continue
        try:
            shorts = json.loads(audit_path.read_text()).get("shorts", [])
        except (OSError, json.JSONDecodeError):
            continue
        for short in shorts:
            try:
                start = int(math.floor(float(short.get("start_sec", 0.0))))
                end = int(math.ceil(float(short.get("end_sec", 0.0))))
            except (TypeError, ValueError):
                continue
            total = 0
            stopped = 0
            for t in range(start, end):
                if 0 <= t < len(mask):
                    total += 1
                    if bool(mask[t]):
                        stopped += 1
            if total == 0:
                continue
            inspected += 1
            frac = stopped / total
            if frac > max_stationary_fraction:
                problems.append(
                    f"{date} short rank {short.get('rank')}: {frac:.0%} stationary "
                    f"({stopped}/{total}s masked)"
                )
    if problems:
        return CheckResult(19, "stop_motion", False, "; ".join(problems))
    if inspected == 0:
        return CheckResult(19, "stop_motion", True, "skipped (no mask/short-window overlap)")
    return CheckResult(
        19, "stop_motion", True, f"all {inspected} shipped short windows majority moving"
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate GoMoveShift pipeline output against Definition-of-Done "
            "checks. Two operating modes: 'post_upload' runs the 15 delivery-"
            "path checks (the default after --youtube-upload has landed), "
            "'skip_upload_audit' runs the cache-side audit used between the "
            "two-step --skip-upload \u2192 review \u2192 --youtube-upload "
            "workflow. Pass --mode auto to derive from pipeline_state.json."
        ),
    )
    parser.add_argument("--project", required=True, help="Project name (e.g. myproject)")
    parser.add_argument(
        "--output-base",
        default=None,
        help=(
            "Base directory containing <project>/ subdirs "
            "(default: GMS_OUTPUT_DIR, else <data home>/output)"
        ),
    )
    parser.add_argument(
        "--state-dir",
        default=None,
        help="Pipeline state dir (default: <temp dir>/gomoveshift/<project>/_stages)",
    )
    parser.add_argument(
        "--snapshot-dir",
        required=True,
        help="Pre-run snapshot dir containing youtube_uploads.json",
    )
    parser.add_argument(
        "--run-start",
        default=None,
        help="ISO-8601 timestamp, default = mtime of --snapshot-dir",
    )
    parser.add_argument("--expected-lf-min", type=int, default=1)
    parser.add_argument("--expected-weeks", type=int, default=1)
    parser.add_argument("--min-lf-duration", type=float, default=480.0)
    parser.add_argument("--max-lf-duration", type=float, default=1800.0)
    parser.add_argument("--max-av-drift-pct", type=float, default=2.0)
    parser.add_argument(
        "--mode",
        choices=[m.value for m in Mode],
        default=Mode.AUTO.value,
        help=(
            "Which check set to run. 'auto' (default) derives from the "
            "pipeline_state.json under --state-dir (skip_upload=True \u2192 "
            "skip_upload_audit; else post_upload). Use an explicit value to "
            "force a mode \u2014 useful for one-off post-run audits where the "
            "state file was rotated or deleted."
        ),
    )
    return parser


def _resolve_output_base(raw: str | None) -> Path:
    """``--output-base`` as given, else the configured pipeline output dir."""
    if raw is not None:
        return Path(raw)
    from gomoveshift.paths import gms_output_dir

    return gms_output_dir()


def _state_field(state: Any | None, name: str) -> Any:
    """Read ``name`` off a ``PipelineState`` OR the raw ``pipeline_state.json``
    dict the AUTO CLI mode loads (:func:`_load_state_from_state_dir`)."""
    if state is None:
        return None
    value = getattr(state, name, None)
    if value is None and isinstance(state, dict):
        value = state.get(name)
    return value


def check_places_verified(project_dir: Path, state: Any | None = None) -> CheckResult:
    """DoD #20: every short in ``state.shorts_per_lf`` names only the place its
    own clip's GPS verified — the same set Phase D's ``assert_all_places_verified``
    refuses on when ``--youtube-upload``.

    Runs in BOTH modes, so the ``SKIP_UPLOAD_AUDIT`` pass catches an unverified
    place before the publish relaunch. Offenders (no sidecar label, empty clip-GPS
    regions, or ``place_source != "clip_gps"``) are named via
    :func:`unverified_place_shorts`. A state with no ``shorts_per_lf`` has
    nothing to check -> PASS marked skipped. ``titles.yaml`` is an optional
    override and its absence is NOT a failure.
    """
    name = "places_verified"
    shorts_per_lf = _state_field(state, "shorts_per_lf")
    if not shorts_per_lf:
        return CheckResult(20, name, True, "skipped (state has no shorts_per_lf)")
    try:
        from gomoveshift.pipeline.drive_phases.phase_d_scheduling import unverified_place_shorts
    except ImportError as exc:  # pragma: no cover — repo checkout always has it
        return CheckResult(20, name, False, f"cannot import phase_d_scheduling: {exc}")
    problems = unverified_place_shorts(
        {lf: list(paths or []) for lf, paths in shorts_per_lf.items()}
    )
    if problems:
        return CheckResult(
            20,
            name,
            False,
            f"{len(problems)} planned short(s) have an unverified place — the "
            "--youtube-upload pass will refuse:\n  " + "\n  ".join(problems),
        )
    total = sum(len(paths or []) for paths in shorts_per_lf.values())
    return CheckResult(20, name, True, f"{total} short(s) have a clip-GPS verified place")


def resolve_run_start(raw: str | None, snapshot_dir: Path) -> datetime:
    if raw is not None:
        return parse_iso(raw)
    mtime = snapshot_dir.stat().st_mtime
    return datetime.fromtimestamp(mtime, tz=timezone.utc)


def _load_state_from_state_dir(state_dir: Path) -> dict | None:
    """Read ``pipeline_state.json`` under ``state_dir`` for auto-mode.

    Returns the parsed dict when readable, ``None`` otherwise.  The caller
    treats ``None`` the same as "no state" — auto-mode falls back to
    POST_UPLOAD in that case (matching :func:`derive_mode_from_state`).
    """
    state_file = state_dir / "pipeline_state.json"
    if not state_file.is_file():
        return None
    try:
        return json.loads(state_file.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def run_checks(args: argparse.Namespace) -> list[CheckResult]:
    """Back-compat shim: run the DoD checks from a parsed argparse namespace.

    Kept for existing tests that drive the validator through ``build_parser``
    → ``run_checks``. New callers should use :func:`run_validation` which
    returns a :class:`ValidationReport` and accepts kwargs directly.

    Honors ``args.mode`` if present.  Legacy namespaces without a ``mode``
    attribute default to the POST_UPLOAD audit.
    """
    state_dir = Path(
        args.state_dir if args.state_dir is not None else _default_state_dir(args.project)
    )
    mode_str = getattr(args, "mode", Mode.POST_UPLOAD.value)
    mode = Mode(mode_str)
    state = _load_state_from_state_dir(state_dir) if mode == Mode.AUTO else None
    report = run_validation(
        project=args.project,
        output_base=_resolve_output_base(args.output_base),
        snapshot_dir=Path(args.snapshot_dir),
        state_dir=state_dir,
        run_start=args.run_start,
        expected_lf_min=args.expected_lf_min,
        expected_weeks=args.expected_weeks,
        min_lf_duration=args.min_lf_duration,
        max_lf_duration=args.max_lf_duration,
        max_av_drift_pct=args.max_av_drift_pct,
        mode=mode,
        state=state,
    )
    return report.checks


def run_validation(
    *,
    project: str,
    output_base: Path,
    snapshot_dir: Path,
    state_dir: Path | None = None,
    run_start: str | datetime | None = None,
    expected_lf_min: int = 1,
    expected_weeks: int = 1,
    min_lf_duration: float = 480.0,
    max_lf_duration: float = 1800.0,
    max_av_drift_pct: float = 2.0,
    superlong_max_lf_duration: float = DEFAULT_SUPERLONG_MAX_DURATION_SEC,
    youtube_uploads_sha: str | None = None,
    mode: Mode | None = None,
    state: Any | None = None,
) -> ValidationReport:
    """Run the DoD checks and return a :class:`ValidationReport`.

    Pure callable: does not call ``sys.exit`` and does not print. Callers
    (orchestrator, CLI, tests) decide what to do with the report.

    Mode routing (see module docstring for the full rationale):

    * ``mode=None`` (the historical default) preserves existing callers:
      runs the 15-check ``POST_UPLOAD`` audit.
    * ``mode=Mode.POST_UPLOAD`` is identical to the ``None`` default.
    * ``mode=Mode.SKIP_UPLOAD_AUDIT`` runs the cache-side audit used
      when the pipeline ran with ``--skip-upload``.  Checks LFs under
      ``<stages>/day_<date>/regular_lfs/`` instead of
      ``<project>/long_forms/`` and drops checks that require the
      post-upload layout (thumbnails, shorts/lf_NN/, sidecar sha).
    * ``mode=Mode.AUTO`` derives the mode from ``state`` via
      :func:`derive_mode_from_state`.  Used by the orchestrator so the
      decision lives next to the state it depends on.

    Args:
        project: Project slug (e.g. ``"myproject"``).
        output_base: Base directory containing ``<project>/`` subdir.
        snapshot_dir: Pre-run snapshot dir holding ``youtube_uploads.json``.
            Also used as fallback anchor for ``run_start`` if not provided.
        state_dir: Pipeline state dir. Defaults to
            ``<temp dir>/gomoveshift/<project>/_stages`` when ``None``.
            In ``SKIP_UPLOAD_AUDIT`` mode this is ALSO where the cache-side
            ``day_*/regular_lfs/`` layout lives.
        run_start: ISO-8601 string or ``datetime``. If ``None`` the snapshot
            dir's mtime is used.
        expected_lf_min: Check 01 minimum.
        expected_weeks: Check 12 span target in weeks.
        min_lf_duration: Check 03 lower bound (seconds).
        max_lf_duration: Check 03 upper bound (seconds).
        max_av_drift_pct: Check 04 maximum audio/video drift percentage.
        superlong_max_lf_duration: Check 03 superlong-part ceiling.
        youtube_uploads_sha: If provided, check 13 hard-asserts live file
            SHA matches this value. If ``None``, falls back to comparing
            live vs snapshot on disk. Ignored in ``SKIP_UPLOAD_AUDIT``.
        mode: See routing table above.
        state: Optional pipeline state (dict or :class:`PipelineState`).
            Only consulted when ``mode=Mode.AUTO`` to derive the
            operating mode via :func:`derive_mode_from_state`.
    """
    output_base = Path(output_base)
    snapshot_dir = Path(snapshot_dir)
    # Tolerate ``--output-dir`` passed as either the parent (output_base) OR
    # the project dir itself — mirrors drive_orchestrator._resolve_paths'
    # ``already_nested`` so the DoD layout matches where phase_d actually wrote
    # the artifacts. Without this, a run started with ``--output-dir .../<proj>``
    # writes flat to output_base but DoD looks in ``output_base/<proj>/`` and
    # hard-fails after the whole encode.
    if output_base.name.lower() == project.lower():
        project_dir = output_base
    else:
        project_dir = output_base / project
    if state_dir is None:
        state_dir = Path(_default_state_dir(project))
    else:
        state_dir = Path(state_dir)

    if isinstance(run_start, datetime):
        rs_dt = run_start if run_start.tzinfo else run_start.replace(tzinfo=timezone.utc)
    elif isinstance(run_start, str):
        rs_dt = parse_iso(run_start)
    else:
        rs_dt = resolve_run_start(None, snapshot_dir)

    effective_mode = mode if mode is not None else Mode.POST_UPLOAD
    if effective_mode == Mode.AUTO:
        effective_mode = derive_mode_from_state(state)

    if effective_mode == Mode.SKIP_UPLOAD_AUDIT:
        checks = _run_skip_upload_checks(
            project=project,
            project_dir=project_dir,
            state_dir=state_dir,
            rs_dt=rs_dt,
            expected_lf_min=expected_lf_min,
            expected_weeks=expected_weeks,
            min_lf_duration=min_lf_duration,
            max_lf_duration=max_lf_duration,
            max_av_drift_pct=max_av_drift_pct,
            superlong_max_lf_duration=superlong_max_lf_duration,
            state=state,
        )
    else:
        checks = _run_post_upload_checks(
            project=project,
            project_dir=project_dir,
            snapshot_dir=snapshot_dir,
            state_dir=state_dir,
            rs_dt=rs_dt,
            expected_lf_min=expected_lf_min,
            expected_weeks=expected_weeks,
            min_lf_duration=min_lf_duration,
            max_lf_duration=max_lf_duration,
            max_av_drift_pct=max_av_drift_pct,
            superlong_max_lf_duration=superlong_max_lf_duration,
            youtube_uploads_sha=youtube_uploads_sha,
            state=state,
        )
    return ValidationReport(project=project, run_start=rs_dt, checks=checks, mode=effective_mode)


def _probe_lfs(lfs: list[tuple[int, Path]]) -> dict[int, dict]:
    """Ffprobe every LF once; errors are recorded as ``{"_error": ...}``."""
    probes: dict[int, dict] = {}
    for idx, path in lfs:
        try:
            probes[idx] = ffprobe(path)
        except RuntimeError as exc:
            probes[idx] = {"_error": str(exc)}
    return probes


def _run_post_upload_checks(
    *,
    project: str,
    project_dir: Path,
    snapshot_dir: Path,
    state_dir: Path,
    rs_dt: datetime,
    expected_lf_min: int,
    expected_weeks: int,
    min_lf_duration: float,
    max_lf_duration: float,
    max_av_drift_pct: float,
    youtube_uploads_sha: str | None,
    superlong_max_lf_duration: float = DEFAULT_SUPERLONG_MAX_DURATION_SEC,
    state: Any | None = None,
) -> list[CheckResult]:
    """The 15-check audit against the post-upload delivery tree.

    Includes content-fidelity check #15. See module docstring.

    Superlong parts fold into checks #03-#06 alongside the regular LFs.
    """
    # Day-first runs do NOT populate ``<project>/long_forms/``.  Resolve LFs
    # via state first, then the legacy delivery tree, then a glob into the
    # day-first cache, so a day-first run is not falsely failed.
    lfs, lf_scheme = resolve_post_upload_lfs(project_dir, state, state_dir)
    superlong_parts = discover_state_superlong_parts(state)
    combined = lfs + superlong_parts
    probes = _probe_lfs(combined)
    return [
        check_lf_count(lfs, expected_lf_min, scheme=lf_scheme),
        check_lf_files_size(lfs),
        check_lf_durations(
            probes,
            lfs,
            min_lf_duration,
            max_lf_duration,
            superlong_parts=superlong_parts,
            superlong_max_dur=superlong_max_lf_duration,
        ),
        check_av_drift(probes, combined, max_av_drift_pct),
        check_audio_stream(probes, combined),
        check_video_hevc(probes, combined),
        check_youtube_json_fresh(project_dir, lfs, project, rs_dt),
        check_thumbnails(project_dir, lfs),
        check_no_ghost_metadata(project_dir, lfs, project),
        check_shorts_count(project_dir, state=state),
        check_schedule_length(state_dir, expected_weeks, expected_lf_min),
        check_youtube_uploads_unchanged(project_dir, snapshot_dir, youtube_uploads_sha),
        check_recording_date_per_lf(project_dir, lfs, project, state),
        check_stop_motion(project_dir, state_dir),
        check_places_verified(project_dir, state),
    ]


def _run_skip_upload_checks(
    *,
    project: str,
    project_dir: Path,
    state_dir: Path,
    rs_dt: datetime,
    expected_lf_min: int,
    expected_weeks: int,
    min_lf_duration: float,
    max_lf_duration: float,
    max_av_drift_pct: float,
    superlong_max_lf_duration: float = DEFAULT_SUPERLONG_MAX_DURATION_SEC,
    state: Any | None = None,
) -> list[CheckResult]:
    """Cache-side audit for ``--skip-upload`` runs.

    Drops the delivery-path-specific checks from the post-upload set
    (they evaluate the Phase-D copy-out layout which does not exist yet):

    * #09 ``thumbnails`` — thumbnails land under ``project_dir/thumbnails``
      during the upload pass; nothing to validate pre-upload.
    * #11 ``shorts_count`` — inspects ``project_dir/shorts/lf_NN/`` which
      is only populated by ``D-upload-shorts``.  The cache-side
      equivalent (``day_*/shorts/``) is verified by the day pipeline's
      own finalizer gate, not here.
    * #13 ``youtube_uploads_unchanged`` — the sidecar is only rewritten
      at upload time; nothing to compare against the snapshot.

    Everything else runs against the cache-side LFs
    (``<stages>/day_*/regular_lfs/lf_NN.mp4``) + the per-LF youtube.json
    sidecars that Phase D writes under ``project_dir`` regardless of
    skip-upload.
    """
    lfs = discover_cache_side_lfs(state_dir)
    superlong_parts = discover_state_superlong_parts(state)
    combined = lfs + superlong_parts
    probes = _probe_lfs(combined)
    base_checks = [
        check_lf_count(lfs, expected_lf_min),
        check_lf_files_size(lfs),
        check_lf_durations(
            probes,
            lfs,
            min_lf_duration,
            max_lf_duration,
            superlong_parts=superlong_parts,
            superlong_max_dur=superlong_max_lf_duration,
        ),
        check_av_drift(probes, combined, max_av_drift_pct),
        check_audio_stream(probes, combined),
        check_video_hevc(probes, combined),
        check_youtube_json_fresh(project_dir, lfs, project, rs_dt),
        check_no_ghost_metadata(project_dir, lfs, project),
        check_schedule_length(state_dir, expected_weeks, expected_lf_min),
    ]
    # Content-fidelity check — runs in both modes so ``--skip-upload``
    # audits also catch the wrong-date bug before the operator commits to
    # ``--youtube-upload``.
    base_checks.extend(
        [
            check_recording_date_per_lf(project_dir, lfs, project, state),
            check_stop_motion(project_dir, state_dir),
            check_places_verified(project_dir, state),
        ]
    )
    return base_checks


def main(argv: list[str] | None = None) -> int:
    """Thin CLI wrapper. Parses argv, runs validation, prints table, exits 0/1."""
    parser = build_parser()
    args = parser.parse_args(argv)

    state_dir = Path(
        args.state_dir if args.state_dir is not None else _default_state_dir(args.project)
    )
    mode = Mode(args.mode)
    # AUTO: read the state file off disk so the CLI can reproduce the
    # orchestrator's mode decision from outside the pipeline process.
    state = _load_state_from_state_dir(state_dir) if mode == Mode.AUTO else None
    report = run_validation(
        project=args.project,
        output_base=_resolve_output_base(args.output_base),
        snapshot_dir=Path(args.snapshot_dir),
        state_dir=state_dir,
        run_start=args.run_start,
        expected_lf_min=args.expected_lf_min,
        expected_weeks=args.expected_weeks,
        min_lf_duration=args.min_lf_duration,
        max_lf_duration=args.max_lf_duration,
        max_av_drift_pct=args.max_av_drift_pct,
        mode=mode,
        state=state,
    )
    print(report.format_table())
    return 0 if report.failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
