"""``health backlog``: launch the next ``drive_backlog.yaml`` row for a region's first uncovered
LF slot, audit only (``--skip-upload --no-youtube-upload``), through the job runner (``heal.job_runner``).

Order, each a stop: the ``gms-backlog-launch`` lease (held = noop); a running drive (noop); a
region row already ``auditing`` (noop: that audit owns the next slot, so a repeat call never
launches a second drive; a failed audit that still has a row is listed under ``failed_audits`` and skipped,
never a hold on the region, since ``health heal`` resumes it; a failed one whose backlog row was dropped is ``retire``d to ``failed``
instead, and its recipes get a ``no-auto-resume`` hold so ``health heal`` leaves it dead); the region's free slots inside ``horizon_h`` (``render_horizon_h``) minus its ready
drives (none left = covered). A drive whose output dir already holds Phase D upload records
(``_uploaded``) is published: it neither holds nor covers a slot, whatever its store status; the first manifest row not in the store, not in the content ledger,
not staged by a prior run's ``resume.json`` and not under a ``no-auto-resume`` hold; storage
(``storage_plan``); the subset dir of symlinks; an install pending (``heal.install_pending``: the
venv lacks the version the code root pins, so the deferred install gets this gap); ``run_preflight``; launch; ``register_ready``
as ``auditing``. A region drive whose newest audit succeeded is handed to ``publish`` (the
caller's ``health publish``) instead of a noop; nothing here uploads or books itself.

``supersede=(project, reason)`` replaces that row's own stale run (an earlier render of the same
project, e.g. on an older release): its stage root and output dir are moved under
``SUPERSEDED_DIR`` beside them, with a ``SUPERSEDE_RECORD`` naming the reason, and the row launches
fresh. Only the region's next eligible row, never a published or booked project, never an output
holding upload records, and nothing moves until every check before the preflight has passed."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .. import run_origin
from ..pipeline import replace_scheduled as replace_scheduled_mod
from ..pipeline.preflight import PEAK_CACHE_FACTOR, PreflightBlockedError, estimate_peak_cache_gb
from ..pipeline.validation.content_gates import (
    placeless_label_claimed_words,
    placeless_label_claims_nothing,
    placeless_label_problem,
)
from . import drives as drive_jobs
from . import resume, retry_policy
from . import stage_owners as owners_mod
from .cadence import COVERAGE_HORIZON_H, _aware
from .heal import (
    _STALL_S,
    SERVICE_DIR,
    _job_records,
    bind_to_code_root,
    default_pinned_version,
    install_pending,
    job_runner,
    refresh_recipe,
)
from .heal import wrap_name as heal_job_name
from .resume import HOLD_NAME, is_held
from .store import load_all, new_incident, safe_name, utc_iso, write_row

logger = logging.getLogger(__name__)

LEASE_NAME = "gms-backlog-launch"
#: Where ``supersede`` moves a row's stale stage root and output dir (beside the live ones, so the
#: move is a rename on one device); ``staged_sources`` never reads under it.
SUPERSEDED_DIR = "_superseded"
SUPERSEDE_RECORD = "superseded.json"
#: The two ``no-auto-resume`` texts health backlog writes when it gives a drive up (``_hold_recipes``):
#: that run is dead by record, so ``cleanup --abandoned`` reclaims it as it does a superseded one.
RETIRED_HOLD_PREFIX = "retired by health backlog: "
GAVE_UP_HOLD = "store row failed and no backlog row; health backlog gave it up"
#: The incident-dir row that records ``FailedRelaunch`` attempts (``resume_attempts``, as heal's);
#: written resolved so ``health check`` neither pushes nor lists it.
RELAUNCH_KIND = "backlog_relaunch"
#: The incident-dir row raised while a region's launch is refused for lack of staging/output
#: space (one open row per region); the next backlog tick that finds the space resolves it.
STORAGE_KIND = "backlog_storage_refused"
#: Names the lease script (``acquire NAME TTL WHAT`` / ``release NAME``; exit 3 = held).
LEASE_SCRIPT_ENV = "GMS_LEASE_SCRIPT"
DEFAULT_LEASE_SCRIPT = Path("bin") / "lease.sh"
#: ``run_preflight``'s defaults: the staging floor and the output free-space floor.
STAGING_FLOOR_GB = 30.0
OUTPUT_FLOOR_GB = 50.0
STAGING_FACTOR = PEAK_CACHE_FACTOR
STORAGE_BOUND = (
    f"staging = max(day source x {STAGING_FACTOR}, {STAGING_FLOOR_GB:g} GB) "
    f"(preflight.estimate_peak_cache_gb), output = {OUTPUT_FLOOR_GB:g} GB floor; summed per "
    "device. The factor is episode concats + the day's raw concat + its master "
    "(preflight.PEAK_CACHE_FACTOR, the per-day encode gate's own terms); retained artifacts "
    "are counted through measured free space."
)

Disk = Callable[[Path], Tuple[int, float]]

#: Where released versions are read from: ``vX.Y.Z`` tags (``git ls-remote``). Overridable.
RELEASE_REMOTE_ENV = "GMS_RELEASE_REMOTE"
DEFAULT_RELEASE_REMOTE = "https://github.com/Back-Road-Creative/framegrade"
_VERSION_TAG = re.compile(r"refs/tags/v(\d+(?:\.\d+)*)$")


def _version_key(text: str) -> Tuple[int, ...]:
    return tuple(int(p) for p in text.split("."))


def latest_release_from_tags(ls_remote_output: str) -> str:
    """Highest ``vX.Y.Z`` in ``git ls-remote --tags`` output; ``ValueError`` when there is none."""
    found = [m.group(1) for line in ls_remote_output.splitlines() if (m := _VERSION_TAG.search(line.strip()))]  # fmt: skip
    if not found:
        raise ValueError("no vX.Y.Z release tags found")
    return max(found, key=_version_key)


def default_latest_version() -> str:
    """The newest released version, from the release remote's tags. Raises when unreachable."""
    remote = os.environ.get(RELEASE_REMOTE_ENV, "").strip() or DEFAULT_RELEASE_REMOTE
    done = subprocess.run(  # noqa: S603,S607 -- fixed git argv, remote from config
        ["git", "ls-remote", "--tags", "--refs", remote],
        capture_output=True, text=True, timeout=30, check=False,
    )  # fmt: skip
    if done.returncode != 0:
        raise OSError(f"git ls-remote {remote} exited {done.returncode}: {done.stderr.strip()[-200:]}")  # fmt: skip
    return latest_release_from_tags(done.stdout)


LATEST_VERSION_CACHE_NAME = "release-latest.json"
#: The release list is read at most this often; the hourly tick reuses the cached answer in between.
LATEST_VERSION_TTL = timedelta(hours=24)
#: A network failure falls back to a cached answer no older than this; older, the failure stands.
LATEST_VERSION_MAX_STALE = timedelta(days=7)


def cached_latest_version(
    cache_path: Path,
    fetch: Callable[[], str] = default_latest_version,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> Callable[[], str]:
    """``fetch`` behind a JSON cache at ``cache_path``: inside ``LATEST_VERSION_TTL`` the cached
    version is returned with no fetch; past it, fetch and rewrite. A fetch failure uses the cache
    while it is within ``LATEST_VERSION_MAX_STALE``, else re-raises exactly what ``fetch`` raised
    (so ``stale_pin_reason`` refuses as it does with no cache)."""

    def read() -> Optional[Tuple[str, datetime]]:
        try:
            data = json.loads(Path(cache_path).read_text())
            version = data["version"]
            fetched = datetime.fromisoformat(data["fetched_at"])
            if not isinstance(version, str) or not re.fullmatch(r"\d+(\.\d+)*", version):
                return None
            return version, fetched
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def latest() -> str:
        now = clock()
        cached = read()
        if cached and timedelta(0) <= now - cached[1] < LATEST_VERSION_TTL:
            return cached[0]
        try:
            version = fetch()
        except (OSError, ValueError, subprocess.SubprocessError):
            if cached and timedelta(0) <= now - cached[1] <= LATEST_VERSION_MAX_STALE:
                return cached[0]
            raise
        try:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            Path(cache_path).write_text(
                json.dumps({"version": version, "fetched_at": now.isoformat()})
            )
        except OSError:
            pass  # a cache that cannot be written only costs the next tick a fetch
        return version

    return latest


def default_live_status() -> replace_scheduled_mod.LiveStatus:
    """``live_status(ids)`` over the real YouTube client (read-only ``videos.list``)."""

    def live(ids):
        from ..pipeline.drive_phases.youtube_uploader import YouTubeUploader

        return YouTubeUploader().video_states(ids)

    return live


def default_installed_version(python: Path) -> Optional[str]:
    """``gomoveshift.__version__`` of the interpreter that will render (its venv's pin)."""
    try:
        done = subprocess.run(  # noqa: S603 -- the service interpreter, fixed snippet
            [str(python), "-c", "import gomoveshift; print(gomoveshift.__version__)"],
            capture_output=True, text=True, timeout=60, check=False,
        )  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return None
    out = done.stdout.strip()
    return out if done.returncode == 0 and re.fullmatch(r"\d+(\.\d+)*", out) else None


def stale_pin_reason(python: Path, installed_version, latest_version) -> Optional[str]:
    """Why the render interpreter may not launch (older than the newest release, or either
    version unknown), else None. Unknown is a refusal, never a guess."""
    try:
        latest = latest_version()
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return f"cannot determine the latest framegrade release ({exc}); not launching on an unverified pin"  # fmt: skip
    installed = installed_version(python)
    if not installed:
        return f"cannot read the framegrade version installed for {python}"
    if _version_key(installed) < _version_key(latest):
        return f"installed framegrade {installed} is older than the latest release {latest}; update the pin first"  # fmt: skip
    return None


def plan_digest(plan: Dict[str, Any], out_dir: Optional[Path] = None) -> str:
    """sha256 of ``replace_plan.json``: its bytes under ``out_dir`` once written, else the bytes
    ``replace_scheduled.write_plan`` would write for ``plan`` (the dry run asks before the write)."""
    if out_dir is not None:
        try:
            return hashlib.sha256((Path(out_dir) / replace_scheduled_mod.REPLACE_PLAN_FILENAME).read_bytes()).hexdigest()  # fmt: skip
        except OSError:
            pass
    return hashlib.sha256((json.dumps(plan, indent=2, sort_keys=True) + "\n").encode()).hexdigest()


def plan_lfs(plan: Dict[str, Any]) -> Tuple[Dict[int, str], Set[int]]:
    """``(expected, replace)``: every planned LF's old video id by LF index, and the LF indices
    the plan replaces (the rest are kept)."""

    def idx(key: Any) -> int:
        return int("".join(ch for ch in str(key) if ch.isdigit()))

    expected = {idx(k): rec["video_id"] for part in ("kept", "replace") for k, rec in (plan.get(part) or {}).items()}  # fmt: skip
    return expected, {idx(k) for k in (plan.get("replace") or {})}


def plan_total_lfs(plan: Dict[str, Any]) -> int:
    """How many long-forms a replace plan covers (kept plus replaced)."""
    return len(plan_lfs(plan)[0])


def read_replace_plan(*dirs: Path) -> Dict[str, Any]:
    """The first non-empty ``replace_plan.json`` under ``dirs`` (``{}`` when none holds one)."""
    for d in dirs:
        plan = replace_scheduled_mod.load_plan(Path(d))
        if plan:
            return plan
    return {}


def plan_kept_lf1_date(plan: Dict[str, Any]) -> Optional[str]:
    """The kept LF1's ``publish_at`` as an America/New_York ``YYYY-MM-DD``; ``None`` when the
    plan keeps no readable LF1."""
    from ..pipeline.schedule_constants import EST

    for key, rec in (plan.get("kept") or {}).items():
        if str(key).strip().lower().lstrip("lf") != "1" or not isinstance(rec, dict):
            continue
        try:
            when = datetime.fromisoformat(str(rec.get("publish_at")).replace("Z", "+00:00"))
        except ValueError:
            return None
        return (when if when.tzinfo else when.replace(tzinfo=timezone.utc)).astimezone(EST).date().isoformat()  # fmt: skip
    return None


def default_replace_store(config: Any) -> Callable[[], Any]:
    """``replace_store()``: the Waypost ``ScheduleStore`` of ``schedule.sqlite3``, opened only when
    a ``complete`` row is reopened for a replace."""

    def open_it() -> Any:
        from seo_manager.scheduling.store import ScheduleStore

        from ..pipeline.seo_slots import STORE_FILENAME, waypost_cadence

        cad = waypost_cadence(config)
        return ScheduleStore(Path(config.seo.storage_path) / STORE_FILENAME, cad.tz, cad.regions)

    return open_it


def exit_code(result: Dict[str, Any]) -> int:
    """2 for a refused tick, or one that went on past a refused publish retry."""
    return 2 if result.get("action") == "refused" or result.get("publish_refusals") else 0


def staging_need_gb(sources: Iterable[str]) -> float:
    return estimate_peak_cache_gb(
        [Path(s) for s in sources], safety_factor=STAGING_FACTOR, floor_gb=STAGING_FLOOR_GB
    )


def default_disk(path: Path) -> Tuple[int, float]:
    """``(st_dev, free GB)`` of ``path``, or of its nearest existing ancestor."""
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return os.stat(probe).st_dev, shutil.disk_usage(probe).free / 1024**3


def storage_plan(
    sources: Sequence[str], staging_dir: Path, output_dir: Path, disk: Disk
) -> List[Dict[str, Any]]:
    """One row per device: what this run needs there against what is free. Staging and output on
    one device are summed, never checked against the same free space twice. Free == need passes
    (``run_preflight`` blocks only on ``free < need``)."""
    by_dev: Dict[int, Dict[str, Any]] = {}
    for role, path, need in (
        ("staging", staging_dir, staging_need_gb(sources)),
        ("output", output_dir, OUTPUT_FLOOR_GB),
    ):
        dev, free = disk(path)
        row = by_dev.setdefault(dev, {"device": dev, "free_gb": free, "need_gb": 0.0, "uses": []})
        row["need_gb"] += need
        row["uses"].append(f"{role} {path} {need:.1f} GB")
    for row in by_dev.values():
        row["need_gb"] = round(row["need_gb"], 3)
        row["ok"] = row["free_gb"] >= row["need_gb"]
    return list(by_dev.values())


def _recipe_sources(recipe: Path) -> Set[str]:
    """Real paths of the video files a prior run's recipe read (its drive source dir)."""
    try:
        argv = json.loads(recipe.read_text(encoding="utf-8"))["argv"]
        src = Path(argv[argv.index("drive") + 1])
        return {os.path.realpath(p) for p in src.iterdir() if p.is_file()}
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return set()


def staged_sources(stages_root: Path) -> List[Tuple[Path, Set[str], bool]]:
    """``(stages dir, source real paths, held)`` per prior run under ``stages_root``: one level
    (``<root>/<project>/_stages``) and two (``<root>/<run>/<project>/_stages``, the layout
    ``GMS_INTERMEDIATE_DIR=<root>/<run>`` gives)."""
    out = []
    for pattern in ("*/_stages", "*/*/_stages"):
        for stages in sorted(Path(stages_root).glob(pattern)):
            if stages.relative_to(stages_root).parts[0] == SUPERSEDED_DIR:
                continue
            recipe = stages / "resume.json"
            out.append((stages, _recipe_sources(recipe), (stages / HOLD_NAME).exists()))
    return out


def placeless_label_skip_reason(label: str) -> Optional[str]:
    """Why a backlog row's ``placeless_label`` may not launch unattended: always, for a label.

    A placeless label is the LAST resort: "we should always have a better idea than 'somewhere
    on the road' if it be gps, or the previous and following day's travel". The better idea for
    a GPS-less drive is ``bracket_gpx`` (the fixes files of the drives before and after), so the
    launcher never starts a row carrying ``placeless_label``, claim-free or not; a person runs
    ``--placeless-label`` by hand when no bracket exists. When the label also claims something
    (``content_gates.placeless_label_claims_nothing`` is false) the reason names the claim too:
    2026-11-08: nctopa-2026-03-05-rec0063 rendered "Through The Valleys And Ridges" over a drive
    that is mostly divided interstate and farmland. ``None`` only for an empty label.
    """
    if not label:
        return None
    base = (
        f"placeless_label {label!r}: a placeless label is a last resort, never launched "
        "unattended; give bracket_gpx (the fixes files of the drives before and after) instead, "
        "or run --placeless-label by hand when no bracket exists"
    )
    problem = placeless_label_problem(label)
    if problem:
        return f"{base} ({problem})"
    if placeless_label_claims_nothing(label):
        return base
    claims = placeless_label_claimed_words(label)
    return (
        f"{base} (it also claims {', '.join(claims)}, which nothing measures on a GPS-less "
        "drive; a hand run takes only a claim-free label, every word in PLACELESS_NO_CLAIM_WORDS)"
    )


def location_labels_skip_reason(row) -> Optional[str]:
    """Why a backlog row's typed labels (``per_lf_locations`` / ``location``) may not launch,
    or ``None``: the drive's own ``--per-lf-location`` / ``--location`` rules
    (``cli.parser._validate_per_lf_location_args``), applied before any render so a refused row
    is skipped at the backlog with the parser's reason instead of failing the launch job at
    argparse hours later. Wording is the operator's; this judges only what the drive would.
    """
    per_lf = list(getattr(row, "per_lf_locations", ()) or ())
    location = getattr(row, "location", "") or ""
    if not per_lf and not location:
        return None
    import argparse

    from ..cli.parser import _per_lf_location_kv, _validate_per_lf_location_args

    try:
        pairs = [_per_lf_location_kv(f"{idx}={label}") for idx, label in per_lf]
        _validate_per_lf_location_args(
            argparse.Namespace(
                per_lf_locations=pairs,
                location=location,
                placeless_label=getattr(row, "placeless_label", ""),
                day_gpx=getattr(row, "day_gpx", ""),
            )
        )
    except argparse.ArgumentTypeError as exc:
        return f"location labels: {exc}"
    return None


def bracket_gpx_skip_reason(row) -> Optional[str]:
    """Why a backlog row's ``bracket_gpx`` (the two fixes files whose last/first fix bracket
    GPS-less footage) may not launch, or ``None``: exactly two paths, both present, and none
    of the typed-label / traced-footage flags the drive refuses beside ``--bracket-gpx``.
    The region itself is computed by the drive, not judged here."""
    pair = [str(p) for p in (getattr(row, "bracket_gpx", ()) or ())]
    if not pair:
        return None
    if len(pair) != 2:
        return f"bracket_gpx needs exactly two files (before, after), got {len(pair)}"
    missing = [p for p in pair if not Path(p).is_file()]
    if missing:
        return f"missing bracket_gpx {missing[0]}"
    clash = [
        name
        for name in ("placeless_label", "day_gpx", "location", "per_lf_locations")
        if getattr(row, name, None)
    ]
    if clash:
        return f"bracket_gpx cannot be combined with {', '.join(clash)}"
    return None


def _skip_reason(row, store_status, known, staged, stages_root, own_ok=False) -> Optional[str]:
    """Why ``row`` is not launched, or ``None``. ``own_ok``: the row's own stage root is being
    superseded, so a hold or recipe under it does not skip the row (``staged`` excludes it)."""
    if row.project in store_status:
        return f"in the schedule store ({store_status[row.project]})"
    if row.project in known:
        return "in the content ledger"
    missing = [s for s in row.sources if not Path(s).is_file()]
    if missing:
        return f"missing source {missing[0]}"
    if row.day_gpx and not Path(row.day_gpx).is_file():
        return f"missing day_gpx {row.day_gpx}"
    label = getattr(row, "placeless_label", "")
    if label and (label_why := placeless_label_skip_reason(label)):
        return label_why
    if labels_why := location_labels_skip_reason(row):
        return labels_why
    if bracket_why := bracket_gpx_skip_reason(row):
        return bracket_why
    for marker in sorted(Path(stages_root).glob(f"{row.project}/**/_stages/{HOLD_NAME}")):
        if not own_ok:
            return f"operator hold: {marker}"
    mine = {os.path.realpath(s) for s in row.sources}
    hits = [(st, held) for st, srcs, held in staged if mine & srcs]
    for stages, held in hits:
        if held:
            return f"operator hold: {stages / HOLD_NAME}"
    if hits:
        return f"already staged by {hits[0][0] / 'resume.json'}"
    return None


def upload_receipts(output_dir: Optional[str]) -> Dict[int, Tuple[str, datetime]]:
    """``{lf_index: (video_id, publish_at)}`` from Phase D's receipts under ``output_dir`` (the
    same files ``_uploaded`` reads), the shape Waypost's ``retire_published`` books. An entry with
    no video id, no parseable ``publish_at`` or no digit in its key is left out."""
    found: Dict[int, Tuple[str, datetime]] = {}
    if not output_dir:
        return found
    out = Path(output_dir)
    for receipt in [out / "youtube_uploads.json", *sorted(out.glob("*/youtube_uploads.json"))]:
        try:
            held = json.loads(receipt.read_text())
        except (OSError, ValueError):
            continue
        for key, r in held.items() if isinstance(held, dict) else ():
            digits = "".join(ch for ch in str(key) if ch.isdigit())
            if not (isinstance(r, dict) and r.get("video_id") and digits):
                continue
            try:
                found.setdefault(
                    int(digits), (r["video_id"], datetime.fromisoformat(r["publish_at"]))
                )
            except (KeyError, TypeError, ValueError):
                continue
    return found


def _uploaded(output_dir: Optional[str]) -> bool:
    """True when Phase D's receipt under ``output_dir`` (``youtube_uploads.json`` there or one
    level down) names a video id: the drive is published, so it neither holds nor covers a slot
    whatever its store status says. Unreadable or empty counts as not uploaded."""
    if not output_dir:
        return False
    out = Path(output_dir)
    for receipt in [out / "youtube_uploads.json", *sorted(out.glob("*/youtube_uploads.json"))]:
        try:
            held = json.loads(receipt.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(held, dict) and any(isinstance(r, dict) and r.get("video_id") for r in held.values()):  # fmt: skip
            return True
    return False


def _subset(subset: Path, sources: Sequence[str], *, create: bool) -> Optional[str]:
    """Refusal text, or ``None`` once ``subset`` holds exactly one symlink per source.

    The day-GPX slicer's own ``<stem>.gpx`` beside a source is not a mismatch: a drive with
    ``--day-gpx`` writes it there before ingest, and rewrites it on the next run."""
    from ..pipeline.ingest.day_gpx_slicer import _is_own

    want = {Path(s).name: os.path.realpath(s) for s in sources}
    stems = {Path(name).stem for name in want}
    if subset.exists():
        have = {
            p.name: os.path.realpath(p)
            for p in subset.iterdir()
            if not (
                p.suffix.lower() == ".gpx" and p.stem in stems and p.name not in want and _is_own(p)
            )
        }
        if have != want:
            extra = sorted(set(have) - set(want)) or sorted(set(want) - set(have))
            return f"subset dir {subset} does not match the manifest row ({extra or 'targets'})"
        return None
    if create:
        subset.mkdir(parents=True)
        for name, target in want.items():
            (subset / name).symlink_to(target)
    return None


def _uploads_refusal(project, out: Path, replace: Optional[Dict[str, Any]] = None) -> str:
    """Refusal text when ``out`` (``project``'s output dir) holds, or may hold, upload records.

    ``replace`` (``--replace-scheduled``: ``{"live": live_status, "now": now}``) lets records through
    when each is public or live-confirmed scheduled; the replacement plan lands in
    ``replace["plan"]``."""
    uploads = out / "youtube_uploads.json"
    if uploads.exists():
        try:
            held = json.loads(uploads.read_text())
        except (OSError, ValueError) as exc:
            return f"cannot supersede {project}: unreadable {uploads} ({exc})"
        if held and replace is not None:
            plan, why = replace_scheduled_mod.classify_uploads(
                held, replace["now"], replace["live"]
            )
            if plan is None:
                return f"cannot supersede {project}: {uploads}: {why}"
            replace["plan"] = plan
        elif held:
            return f"cannot supersede {project}: {uploads} holds upload records"
    return ""


class FailedRelaunch:
    """Is a failed drive due a relaunch from source? Heal only resumes a drive that has a readable
    ``resume.json``; one without (calibration.json only) is never reached by anything, so
    ``health backlog`` relaunches it when ``retry_policy.refusal`` allows: its class and the
    fingerprint its failure was filed under (``failure_fingerprint``, stamped by ``health
    check``) against the one now. A row with no stamp (or no incident row) is legacy: one launch.
    Each attempt is a ``resume_attempts`` entry on the project's ``RELAUNCH_KIND`` row; a refusal
    is recorded once per fingerprint and reason. A dry run writes nothing.

    ``fingerprint_fn(project, phase)`` is the current fingerprint (``retry_policy.fingerprint``)."""

    def __init__(self, incidents_dir, stages_root, *, fingerprint_fn, now, dry_run=False):
        self.dir, self.stages_root = Path(incidents_dir), Path(stages_root)
        self.fingerprint_fn, self.now, self.dry_run = fingerprint_fn, now, dry_run
        self._fph: Dict[str, str] = {}
        self._klass: Dict[str, str] = {}

    def _rows(self, project: str):
        rows, _bad = load_all(self.dir)
        key = f"{RELAUNCH_KIND}:{safe_name(project)}"
        failed = [r for r in rows if r.get("kind") == "drive_failed" and r.get("drive") == project]
        return failed, next((r for r in rows if r.get("key") == key), None)

    def _resumable(self, project: str) -> bool:
        """A readable recipe under the project's stages: ``health heal`` resumes it, not us."""
        found = self.stages_root.glob(f"{project}/**/_stages/resume.json")
        return any(resume.read_recipe(p) is not None for p in found)

    def due(self, project: str) -> Optional[str]:
        """The reason to relaunch ``project`` from source now, else ``None`` (a refusal is
        recorded unless this is a dry run)."""
        if self._resumable(project):
            return None
        failed, ledger = self._rows(project)
        failed.sort(key=lambda r: str(r.get("failed_at") or r.get("detected_at") or ""))
        newest = failed[-1] if failed else {}
        phase = newest.get("phase")
        fph = retry_policy.fingerprint_hash(self.fingerprint_fn(project, phase))
        stamped = {(r.get("failure_fingerprint") or {}).get("hash") for r in failed} - {None}
        stamp = newest.get("failure_fingerprint") or {}
        klass = stamp.get("class") or (
            retry_policy.classify_failure(newest.get("error"), phase, newest.get("gate"))
            if newest else retry_policy.UNKNOWN
        )  # fmt: skip
        attempts = (ledger or {}).get("resume_attempts") or []
        why = retry_policy.refusal(
            klass, fph, launched=retry_policy.launched_count(attempts, fph),
            failed_under=fph in stamped,
        )  # fmt: skip
        self._fph[project], self._klass[project] = fph, klass
        if why is None:
            note = "legacy row, no failure fingerprint" if not stamped else "fingerprint changed"
            return f"{klass} failure, fingerprint {fph[:12]}: {note}; relaunching from source"
        self._record(project, "refused", why, ledger)
        return None

    def record_launch(self, project: str, reason: str) -> None:
        self._record(project, "launched", reason, self._rows(project)[1])

    def _record(self, project: str, action: str, reason: str, ledger) -> None:
        if self.dry_run:
            return
        now_iso = utc_iso(self.now.timestamp())
        ledger = ledger or new_incident(
            RELAUNCH_KIND, f"{RELAUNCH_KIND}:{safe_name(project)}",
            f"backlog-relaunch-{safe_name(project)}.json", now_iso, drive=project,
            resume_attempts=[],
        )  # fmt: skip
        ledger["pushed"], ledger["resolved_at"] = True, ledger.get("resolved_at") or now_iso
        attempts = ledger.setdefault("resume_attempts", [])
        att = {
            "at": now_iso, "action": action, "reason": reason, "fp_hash": self._fph.get(project),
            "failure_class": self._klass.get(project),
        }  # fmt: skip
        if (
            action == "refused"
            and attempts
            and all(attempts[-1].get(k) == att[k] for k in ("fp_hash", "action", "reason"))
        ):
            return
        attempts.append(att)
        write_row(self.dir, ledger)


def default_fingerprint_fn(config: Any, code_root: Path) -> Callable[[str, Any], Dict[str, Any]]:
    """The fingerprint of the code and inputs a launch would run now: the service interpreter's
    versions (read once), the project's manifest row, the drive config and the failed phase."""
    python = str(Path(code_root) / SERVICE_DIR / ".venv" / "bin" / "python")
    versions: Dict[str, Any] = {}

    def fingerprint_fn(project: str, phase: Any) -> Dict[str, Any]:
        if not versions:
            versions.update(
                gomoveshift=retry_policy.default_installed_version(python),
                **retry_policy.default_sibling_versions(python),
            )
        return retry_policy.fingerprint(
            versions=versions, manifest_sha=retry_policy.default_manifest_sha({"drive": project}),
            config_sha=retry_policy.default_config_sha(config), phase=phase,
        )  # fmt: skip

    return fingerprint_fn


def _supersede_plan(
    project, stages_root, output_root, now, siblings: Sequence[str] = (),
    replace: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, str]], str]:  # fmt: skip
    """``(moves, refusal)``: the renames that set ``project``'s stage root and output dir aside,
    then the same for each dead ``siblings`` project (an earlier run of the same footage)."""
    stamp = now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    moves = []
    for name in (project, *siblings):
        out = Path(output_root) / name
        why = _uploads_refusal(name, out, replace if name == project else None)
        if why:
            return [], why
        for src in (Path(stages_root) / name, out):
            dst = src.parent / SUPERSEDED_DIR / f"{name}-{stamp}"
            if dst.exists():
                return [], f"cannot supersede {name}: {dst} already exists"
            if src.exists():
                moves.append({"from": str(src), "to": str(dst)})
    if not moves:
        return [], f"nothing to supersede: no {project} under the stages or output root"
    return moves, ""


def _supersede(project, reason, moves, now, plan=None, out_dir=None) -> None:
    """Rename each move and write the record into every moved dir. ``plan`` (``--replace-scheduled``)
    is recorded as ``backlog-supersede/2`` and written into the fresh ``out_dir`` for Phase D."""
    roots = [Path(m["from"]) for m in moves]
    recipes = [r for root in roots for r in sorted(root.glob("_stages/resume.json")) + sorted(root.glob("*/_stages/resume.json"))]  # fmt: skip
    recipe: Dict[str, Any] = {}
    if recipes:
        try:
            recipe = json.loads(recipes[0].read_text())
        except (OSError, ValueError):
            recipe = {"unreadable": str(recipes[0])}
    record = {
        "schema": "backlog-supersede/1", "project": project, "reason": reason,
        "at": now.isoformat(), "moves": moves,
        "held": any(any(root.glob(f"**/_stages/{HOLD_NAME}")) for root in roots),
        "recipe": {k: recipe.get(k) for k in ("sha", "started_at", "phase_reached", "argv")},
    }  # fmt: skip
    if plan:
        record["schema"], record["replace"] = "backlog-supersede/2", plan
    for move in moves:
        dst = Path(move["to"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        Path(move["from"]).rename(dst)
        (dst / SUPERSEDE_RECORD).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    if plan and out_dir is not None:
        replace_scheduled_mod.write_plan(Path(out_dir), plan)


def _unsupersede(moves, out_dir, now) -> str:
    """Undo ``_supersede``'s renames after a refusal: whatever fresh dir the failed attempt left
    (the replace plan, empty dirs) is removed, or set aside under ``SUPERSEDED_DIR`` when it holds
    anything else; the old dir goes back and loses its ``SUPERSEDE_RECORD``. Returns what could
    not be restored (``""`` when everything was)."""
    stamp = now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    failed = []
    for move in reversed(moves):
        src, dst = Path(move["from"]), Path(move["to"])
        try:
            if src.exists():
                plan_file = src / replace_scheduled_mod.REPLACE_PLAN_FILENAME
                plan_file.unlink(missing_ok=True)
                try:
                    for d in sorted((p for p in src.rglob("*") if p.is_dir()), reverse=True):
                        d.rmdir()
                    src.rmdir()
                except OSError:  # the attempt left real files: keep them, beside the old run
                    aside = dst.with_name(f"{dst.name}-failed-attempt-{stamp}")
                    src.rename(aside)
            dst.rename(src)
            (src / SUPERSEDE_RECORD).unlink(missing_ok=True)
            try:
                dst.parent.rmdir()  # ``SUPERSEDED_DIR`` itself, once empty
            except OSError:
                pass
        except OSError as exc:
            failed.append(f"{src}: {exc}")
    return "; ".join(failed)


def _hold_recipes(stages_root: Path, project: str, reason: str, dry_run: bool) -> List[str]:
    """``no-auto-resume`` beside each of a given-up drive's unheld resume recipes, so ``health
    heal`` never re-resumes it on a later mirror SHA; the stage dirs held (dry run: would be)."""
    held = []
    for recipe in sorted(Path(stages_root).glob(f"{project}/**/_stages/resume.json")):
        if not is_held(recipe):
            if not dry_run:
                (recipe.parent / HOLD_NAME).write_text(reason + "\n", encoding="utf-8")
            held.append(str(recipe.parent))
    return held


#: ``heal.job_name``'s commit suffix: a heal resume is ``gms-heal-<drive job>-<sha7>``.
_HEAL_SHA = re.compile(r"-[0-9a-f]{7}$")


def _is_audit_job(name: Any, project: str) -> bool:
    """``gms-backlog-<project>`` itself, or a heal resume of it (``heal.job_name``, nested once
    per resume): the run that holds the audit is whichever of them ran last.

    A ``health heal --recipe`` resume names the drive by its stages dir, so it is
    ``gms-heal-<project>-<sha7>``: that resumes the same drive and counts too (a bare
    ``<project>`` name, never wrapped by a heal, does not)."""
    base = f"gms-backlog-{project}"
    name = str(name or "")
    healed = False
    while name != base and name.startswith("gms-heal-") and _HEAL_SHA.search(name):
        inner = name[len("gms-heal-") : -8]
        if heal_job_name(inner, name[-7:]) != name:
            return False
        name, healed = inner, True
    return name == base or (healed and name == project)


def _newest_audit(jobs_dir: Path, project: str) -> Optional[Dict[str, Any]]:
    recs = [r for r in _job_records(Path(jobs_dir)) if _is_audit_job(r.get("name"), project)]
    return max(recs, key=lambda r: float(r.get("created_at") or 0)) if recs else None


def _gate_wait(jobs_dir: Path, rec: Dict[str, Any]) -> Optional[str]:
    """The job's ``DRIVE WAITING:`` line when it stopped at a human gate (its own output.log)."""
    code = (rec.get("result") or {}).get("exit_code")
    if code != drive_jobs.EXIT_WAITING_AT_GATE:
        return None
    return drive_jobs.gate_wait(code, drive_jobs._tail(Path(jobs_dir) / str(rec.get("run_id")) / "output.log"))  # fmt: skip


def _audit_running(jobs_dir: Path, project: str) -> bool:
    rec = _newest_audit(jobs_dir, project)
    return bool(rec) and (rec.get("result") or {}).get("state") == "running"


def _failed_audit(jobs_dir: Path, project: str, hint: bool = True) -> Optional[str]:
    """Refusal text when the newest ``gms-backlog-<project>`` job (or heal resume of it) ended in anything but success
    or a stop at a human gate (``_ost_gate``).

    A failed audit still holds its slot, so without this every later call is a silent noop;
    the operator relaunches (the drive resumes from its stages) or supersedes it."""
    rec = _newest_audit(jobs_dir, project)
    if rec is None:
        return None
    result = rec.get("result") or {}
    state = result.get("state")
    if state in (None, "running", "succeeded") or _gate_wait(jobs_dir, rec):
        return None
    text = f"audit failed: {project} job {rec.get('run_id')} ended {state} (exit {result.get('exit_code')})"
    if not hint:
        return text
    return (
        f"{text}; relaunch it (stages resume), or --supersede {project} "
        "to set the run aside and launch the row as it now reads"
    )


_WAIT_VERSION_RE = re.compile(r"\[gomoveshift (\d+(?:\.\d+)*)\]\s*$")


def waiting_version(line: Optional[str]) -> Optional[str]:
    """The gomoveshift version a ``DRIVE WAITING:`` line records, or None (a legacy stop)."""
    m = _WAIT_VERSION_RE.search(line or "")
    return m.group(1) if m else None


def _version_retry(
    rec, jobs_dir, stages_root, project, code_root, installed_version
) -> Optional[str]:
    """Why a drive parked at the OST gate may be resumed once more with no person: its resume
    recipe asks for the automatic system review (so code alone can clear the gate) and the version
    that stopped it differs from the installed one, or is unrecorded (a legacy stop). None when
    it may not (same version, no flag, no recipe, an unreadable installed version)."""
    if installed_version is None or code_root is None:
        return None
    recipes = sorted(Path(stages_root).glob(f"{project}/**/_stages/resume.json"))
    recipe = resume.read_recipe(recipes[0]) if recipes else None
    if recipe is None or "--ost-system-review" not in recipe["argv"]:
        return None
    bound, _cwd = bind_to_code_root(code_root, recipe["argv"])
    if bound is None:
        return None
    now = installed_version(Path(bound[0]))
    if not now:
        return None
    then = waiting_version(_gate_wait(jobs_dir, rec))
    if then == now:
        return None
    since = f"under gomoveshift {then}" if then else "under an earlier gomoveshift"
    return f"{project}: stopped at the OST gate {since}, now {now} installed; resuming once so the gate re-checks"  # fmt: skip


def _ost_gate(jobs_dir: Path, stages_root: Path, project: str,
              code_root=None, installed_version=None) -> Optional[Tuple[str, bool, Optional[str]]]:  # fmt: skip
    """``(reason, resumable, retry)`` when the newest audit of ``project`` stopped at the OST audit gate
    (every placeless drive does: no GPS, no system review), else None.

    Confirmed: each ``ost_audit_<date>.json`` sidecar has its ``.confirmed`` marker and one was
    made after the run stopped. Markers all older than the stop did not pass the gate (the audit
    changed since), so resuming would only stop at the same gate again, every tick.

    Otherwise a drive whose recipe carries ``--ost-system-review`` is resumed once per installed
    version (``_version_retry``): a code fix can make the parked day passable by the system review.
    The resume never touches a marker; the gate inside the drive decides."""
    rec = _newest_audit(jobs_dir, project)
    if rec is None or not _gate_wait(jobs_dir, rec):
        return None
    ended = float((rec.get("result") or {}).get("ended_at") or 0)
    sidecars = sorted(Path(stages_root).glob(f"{project}/**/_stages/ost_audit_*.json"))
    markers = [s.with_suffix(".confirmed") for s in sidecars]
    pending = [m for m in markers if not m.exists()]
    fresh = any(m.exists() and m.stat().st_mtime > ended for m in markers)
    if sidecars and not pending and fresh:
        return (
            f"{project}: OST audit confirmed since job {rec.get('run_id')} stopped at it",
            True,
            None,
        )
    if why := _version_retry(rec, jobs_dir, stages_root, project, code_root, installed_version):
        return why, True, why
    head = f"audit waiting at the OST audit gate: {project} job {rec.get('run_id')} stopped for a person's review; "  # fmt: skip
    if not sidecars:
        return (
            head + f"no ost_audit_<date>.json sidecar under {Path(stages_root) / project}",
            False,
            None,
        )
    todo = pending or markers  # every marker predates the stop: the audit changed, re-review it
    again = "" if pending else " (replace the existing marker: the audit changed after it)"
    return (
        head + "review " + ", ".join(str(m.with_suffix(".json")) for m in todo)
        + " and create " + ", ".join(str(m) for m in todo) + again
        + "; the next backlog tick resumes it from its stages",
        False,
        None,
    )  # fmt: skip


# ``publish.JOB_PREFIX`` (publish imports this module, so the name is repeated here; a test pins it)
PUBLISH_JOB_PREFIX = "gms-publish-"


def _failed_publish(jobs_dir: Path, project: str) -> bool:
    """The newest ``gms-publish-<project>`` job (Phase D after a passed audit) ended in anything
    but success: failed, or cancelled, stalled or killed with no exit code."""
    name = PUBLISH_JOB_PREFIX + project
    recs = [r for r in _job_records(Path(jobs_dir)) if r.get("name") == name]
    if not recs:
        return False
    result = max(recs, key=lambda r: float(r.get("created_at") or 0)).get("result") or {}
    state = result.get("state")
    # any terminal state but success is a dead publish: a job the runner cancelled, stalled or
    # killed carries no exit code, so keying on the code alone left Kerry-to-Cork unpublished
    return state not in (None, "running", "succeeded")


def _project_job_running(jobs_dir: Path, project: str) -> bool:
    """A job of this drive itself is still running: its audit (or a heal of it) or its Phase D
    publish. The one drive a relaunch, which moves its stage and output dirs, must wait for."""
    pub = [r for r in _job_records(Path(jobs_dir)) if r.get("name") == PUBLISH_JOB_PREFIX + project]
    newest = max(pub, key=lambda r: float(r.get("created_at") or 0)) if pub else {}
    return (
        _audit_running(jobs_dir, project) or (newest.get("result") or {}).get("state") == "running"
    )


def _audit_passed(jobs_dir: Path, project: str) -> bool:
    rec = _newest_audit(jobs_dir, project)
    return rec is not None and (rec.get("result") or {}).get("state") == "succeeded"


def _publish_audited(res, project: str, publish, dry_run: bool) -> Dict[str, Any]:
    """``health publish`` for the region's audited drive (its newest audit succeeded). A claim
    miss (``no claim:``, e.g. ``outside_lead_window``) leaves it ``auditing`` and waits for a
    later tick; any other refusal is this call's refusal."""
    out = publish(project, dry_run)
    res.update(project=project, publish=out)
    why = str(out.get("reason") or "")
    if out.get("action") == "launched":
        return {**res, "action": "launched", "reason": f"published: {why}"}
    if out.get("action") != "refused":
        return {**res, "action": out.get("action") or "noop", "reason": why}
    if why.startswith("no claim:"):
        return {**res, "reason": f"audit passed: {project} waits for its Waypost slot: {why}"}
    return {**res, "action": "refused", "reason": f"publish refused: {why}"}


def _resume_confirmed(res, project, stages_root, output_root, code_root, jobs_dir, dry_run,
                      launcher, installed_version, latest_version, why=None,
                      manifest_row=None, pinned_version=default_pinned_version):  # fmt: skip
    """Resume a drive whose OST audit was confirmed (or, with ``why``, parked at the gate under
    another version), from its own recipe, under its backlog job name; the store row is already
    ``auditing``, so nothing is registered. A recipe GPX that no longer exists takes the backlog
    row's, and a gone input with no replacement refuses (``heal.refresh_recipe``), as heal's own
    resume does."""
    lead = why or f"{project}: OST audit confirmed"
    recipes = sorted(Path(stages_root).glob(f"{project}/**/_stages/resume.json"))
    if any(is_held(r) for r in recipes):
        return {**res, "reason": f"{lead}, but an operator hold ({HOLD_NAME}) keeps it where it is"}  # fmt: skip
    recipe = resume.read_recipe(recipes[0]) if recipes else None
    if recipe is None:
        return {**res, "action": "refused", "reason": f"{lead}, but no readable resume.json under {Path(stages_root) / project}"}  # fmt: skip
    fresh, refreshed, gone = refresh_recipe(recipes[0], recipe, manifest_row,
                                            cwd=Path(code_root) / SERVICE_DIR, dry_run=dry_run)  # fmt: skip
    if refreshed:
        res["gpx_refreshed"] = refreshed
    bound, cwd = bind_to_code_root(code_root, fresh) if fresh else (None, gone)
    if bound is None:
        return {**res, "action": "refused", "reason": f"{project}: {cwd}"}
    bound += [f for f in ("--skip-upload", "--no-youtube-upload") if f not in bound]
    flag = "--output-dir"
    out_dir = Path(bound[bound.index(flag) + 1]) if flag in bound[:-1] else Path(output_root) / project  # fmt: skip
    argv = _job_argv(code_root, jobs_dir, project, out_dir, bound)
    env = {**resume.recipe_env(recipe), run_origin.LAUNCHED_BY_ENV: run_origin.BACKLOG_LAUNCHER}
    res.update(argv=argv, env=env, resume_recipe=str(recipes[0]))
    if pending := install_pending(installed_version(Path(bound[0])), pinned_version(Path(code_root))):  # fmt: skip
        return {**res, "action": "refused", "reason": f"{project}: {pending}"}
    if dry_run:
        return {**res, "action": "would_launch", "reason": f"{project}: would resume it from its stages"}  # fmt: skip
    stale = stale_pin_reason(Path(bound[0]), installed_version, latest_version)
    if stale:
        return {**res, "action": "refused", "reason": stale}
    ok, detail = launcher(argv, cwd, env)
    if not ok:
        return {**res, "action": "refused", "reason": f"launch failed: {detail}"}
    return {**res, "action": "launched", "reason": f"{lead}; resuming it from its stages" if not why else why}  # fmt: skip


def _job_argv(code_root, jobs_dir, project: str, out_dir: Path, drive_argv: List[str]) -> List[str]:
    return [
        "python3", str(job_runner(code_root)),
        "--name", f"gms-backlog-{project}", "--heavy", "--detach", "--stall", _STALL_S,
        "--expect", str(out_dir / "**" / "shorts" / "**" / "*.mp4"),
        "--log-root", str(jobs_dir), "--", *drive_argv,
    ]  # fmt: skip


def _slot_target(
    slots, ready: int, now: datetime, horizon_h: int = COVERAGE_HORIZON_H
) -> Optional[Tuple[datetime, str]]:
    horizon = now + timedelta(hours=horizon_h)
    free = sorted(s for s in slots if now <= s[0] <= horizon)
    return free[ready] if len(free) > ready else None


def report_slots(report: Dict[str, Any], region: str) -> List[Tuple[datetime, str]]:
    """``(slot, day)`` free LF slots for ``region`` from the schedule report; the day name
    is Waypost's own, never re-derived here."""
    out = []
    for r in report.get("next_available", []):
        if r.get("region") == region and (s := _aware(r.get("slot"))) and r.get("day"):
            out.append((s, r["day"]))
    return out


def config_render_gate(config) -> Callable[[Path, Any], Optional[str]]:
    """The drive's raw-duration refusal for a backlog row, under the row's own modes.

    ``config`` is left untouched: each call scans with a copy carrying the row's
    ``ingest_mode`` and ``lf_mode``. A dry run has no subset dir yet, so the scan
    reads a throwaway dir of links to the row's sources.
    """
    from ..pipeline.orchestrator_preflight import day_under_part_min_message

    def gate(subset: Path, row) -> Optional[str]:
        cfg = copy.copy(config)
        cfg.drive = copy.copy(config.drive)
        cfg.drive.ingest_mode = row.ingest_mode
        cfg.drive.set_lf_mode(row.lf_mode, "cli")  # the drive gets it as --lf-mode
        if Path(subset).is_dir():
            return day_under_part_min_message(cfg, Path(subset))
        with tempfile.TemporaryDirectory(prefix="gms-render-gate-") as tmp:
            for src in row.sources:
                (Path(tmp) / Path(src).name).symlink_to(os.path.realpath(src))
            return day_under_part_min_message(cfg, Path(tmp))

    return gate


def run_backlog(
    *,
    region: str,
    rows: Sequence[Any],
    drives: Sequence[Any],
    free_slots: Sequence[Tuple[datetime, str]],
    known_projects: Set[str],
    stages_root: Path,
    output_root: Path,
    sources_root: Path,
    code_root: Path,
    jobs_dir: Path,
    mirror_sha: Optional[str],
    now: datetime,
    dry_run: bool,
    lease_acquire: Callable[[], bool],
    lease_release: Callable[[], None],
    preflight: Callable[..., Any],
    render_gate: Callable[[Path, Any], Optional[str]],
    launcher: Callable[[List[str], str, Dict[str, str]], Tuple[bool, str]],
    register: Callable[..., None],
    retire: Callable[[str], None],
    disk: Disk = default_disk,
    installed_version: Callable[[Path], Optional[str]] = default_installed_version,
    latest_version: Callable[[], str] = default_latest_version,
    pinned_version: Callable[[Path], Optional[str]] = default_pinned_version,
    supersede: Optional[Tuple[str, str]] = None,
    publish: Optional[Callable[[str, bool], Dict[str, Any]]] = None,
    horizon_h: int = COVERAGE_HORIZON_H,
    failed_relaunch: Optional[FailedRelaunch] = None,
    incidents_dir: Optional[Path] = None,
    stage_owners: Optional[Sequence[owners_mod.Owner]] = None,
    release_list: Optional[Path] = None,
    replace_scheduled: bool = False,
    live_status: Optional[replace_scheduled_mod.LiveStatus] = None,
    replace_store: Optional[Callable[[], Any]] = None,
) -> Dict[str, Any]:
    res: Dict[str, Any] = {"region": region, "action": "noop", "skipped": []}
    if not dry_run and not lease_acquire():
        return {**res, "reason": f"lease {LEASE_NAME} is held by another launch"}
    try:
        out = _run(res, region, rows, drives, free_slots, known_projects, stages_root,
                    output_root, sources_root, code_root, jobs_dir, mirror_sha, now, dry_run,
                    preflight, render_gate, launcher, register, retire, disk,
                    installed_version, latest_version, supersede, publish, horizon_h,
                    failed_relaunch,
                    {"live": live_status, "now": now, "plan": None} if replace_scheduled else None,
                    replace_store, pinned_version)  # fmt: skip
        if stage_owners is not None:
            out["stage_roots"] = _stage_roots(
                stages_root, rows, drives, stage_owners, release_list, now, dry_run
            )
        if incidents_dir is not None:
            _storage_incident(Path(incidents_dir), region, out, now, dry_run)
        return out
    finally:
        if not dry_run:
            lease_release()


def _stage_roots(stages_root, rows, drives, owners, release_list, now, dry_run) -> Dict[str, Any]:
    """Owner-record upkeep for stage roots no manifest row owns: the expired records' roots are
    appended to the host-cleanup release list (a dry run only names them), and every root with
    neither a row, a store drive nor an owner record is reported."""
    today = now.astimezone(timezone.utc).date()
    known = {r.project for r in rows} | {d.project for d in drives}
    due = [o for o in owners_mod.expired(owners, today) if (Path(stages_root) / o.name).is_dir()]
    listed = [o.name for o in due]
    if release_list is not None and not dry_run:
        listed = owners_mod.list_for_release(Path(release_list), due, today)
    return {
        "listed": listed,
        "unowned": owners_mod.unowned_roots(Path(stages_root), known, owners),
    }


def _storage_incident(incidents_dir, region, res, now, dry_run) -> None:
    """The region's ``STORAGE_KIND`` row: opened (once) by a launch refused for space, resolved by
    a tick that launches or measures every device as having room. A refusal for another reason,
    or a noop that never measured, leaves it as it is. A dry run writes nothing."""
    if dry_run:
        return
    key = f"{STORAGE_KIND}:{region}"
    rows, _bad = load_all(incidents_dir)
    open_row = next((r for r in rows if r.get("key") == key and not r.get("resolved_at")), None)
    now_iso = utc_iso(now.timestamp())
    if str(res.get("reason", "")).startswith("storage:") and res.get("action") == "refused":
        if open_row is None:
            write_row(incidents_dir, new_incident(
                STORAGE_KIND, key, f"backlog_storage_refused_{safe_name(region)}.json", now_iso,
                region=region, project=res.get("project"), reason=res["reason"],
            ))  # fmt: skip
        return
    storage = res.get("storage")
    roomy = res.get("action") == "launched" or (storage and all(s["ok"] for s in storage))
    if open_row is not None and roomy:
        open_row["resolved_at"] = now_iso
        write_row(incidents_dir, open_row)


def _run(res, region, rows, drives, free_slots, known, stages_root, output_root, sources_root,
         code_root, jobs_dir, mirror_sha, now, dry_run, preflight, render_gate,
         launcher, register, retire, disk, installed_version, latest_version, supersede, publish,
         horizon_h, failed_relaunch=None, replace=None, replace_store=None,
         pinned_version=default_pinned_version):  # fmt: skip
    published = [d.project for d in drives if d.region == region and _uploaded(d.output_dir)]
    if published:
        res["published"] = published
    mine = [d for d in drives if d.region == region and d.project not in published]
    auditing = [d.project for d in mine if d.status == "auditing"]
    target = supersede[0] if supersede else None
    listed = {r.project for r in rows if r.region == region}
    for project in [p for p in auditing if p not in listed and p != target]:
        # a failed audit whose backlog row was dropped can never be relaunched or superseded;
        # its store row is retired (``failed``) so it stops holding the region's next slot
        failed = _failed_audit(jobs_dir, project)
        if failed:
            if not dry_run:
                retire(project)
            res.setdefault("retired", []).append({"project": project, "reason": failed})
            gone = RETIRED_HOLD_PREFIX + str(failed)
            held = _hold_recipes(stages_root, project, gone, dry_run)
            if held:
                res.setdefault("held", []).extend(held)
            auditing.remove(project)
            mine = [d for d in mine if d.project != project]
    for project in sorted({d.project for d in mine if d.status == "failed"} - listed):
        # a drive retired before holds existed, or set ``failed`` by hand, is never retired again
        held = _hold_recipes(stages_root, project, GAVE_UP_HOLD, dry_run)
        if held:
            res.setdefault("held", []).extend(held)
    # No "is any drive running" stop (renders run back to back for up to 45 h): a publish needs
    # no GPU, a new render queues on the shared GPU lock, and a run on the same project dirs is
    # refused by its own run lock. A running audit still holds its region (below).
    # a passed audit whose publish failed (``health check`` closed the claim ``failed``) is
    # published again, never abandoned: ``health publish`` reopens it (2026-11-19, Kerry-to-Cork).
    # A refused retry is recorded and the tick goes on, so one drive's standing refusal never
    # starves the region's next passed audit (2026-11-25, Kerry-to-Cork held Old Head); the tick
    # still exits 2 while the refusal stands (``exit_code``).
    for proj in sorted(d.project for d in mine if d.status == "failed" and d.project in listed):
        if publish and _failed_publish(jobs_dir, proj) and _audit_passed(jobs_dir, proj):
            done = _publish_audited(dict(res), proj, publish, dry_run)
            if done.get("action") != "refused":
                return done
            refused = {"project": proj, "reason": done["reason"]}
            res.setdefault("publish_refusals", []).append(refused)
    relaunch, auto = None, False
    held = None  # the first passed audit whose publish was refused (no --supersede)
    for proj in auditing:
        if held is not None:
            # a passed audit behind a refused one still gets its publish (2026-11-27, Old Head's
            # mask refusal held Galley Head's passed audit); nothing else runs past the hold
            if not _failed_audit(jobs_dir, proj) and _audit_passed(jobs_dir, proj):
                done = _publish_audited(dict(res), proj, publish, dry_run)
                bad = {"project": held["project"], "reason": held["reason"]}
                if done.get("action") != "refused":
                    return {**done, "publish_refusals": [*res.get("publish_refusals", []), bad]}
                res.setdefault("publish_refusals", []).append(
                    {"project": proj, "reason": done["reason"]}
                )
            continue
        failed = _failed_audit(jobs_dir, proj)
        gate = (
            None if failed else _ost_gate(jobs_dir, stages_root, proj, code_root, installed_version)
        )
        if gate and proj == target:
            failed = gate[0]  # superseding an audit waiting at its gate sets it aside the same way
        elif gate:
            res["project"] = proj
            if not gate[1]:
                return {**res, "reason": gate[0]}
            row = next((r for r in rows if r.project == proj), None)
            return _resume_confirmed(res, proj, stages_root, output_root, code_root,
                                     jobs_dir, dry_run, launcher, installed_version,
                                     latest_version, why=gate[2],
                                     manifest_row=row, pinned_version=pinned_version)  # fmt: skip
        if not failed and publish and proj != target and _audit_passed(jobs_dir, proj):
            done = _publish_audited(dict(res), proj, publish, dry_run)
            if done.get("action") != "refused":
                return done
            if not supersede:
                held = {**done, "project": proj}
                continue
            # an operator's supersede of another row is never held by this drive's standing
            # publish refusal (2026-11-26, Old Head held Kerry-to-Cork's re-render): recorded,
            # and the supersede goes on; the tick still exits 2 while the refusal stands
            res.setdefault("publish_refusals", []).append(
                {"project": proj, "reason": done["reason"]}
            )
            continue
        if (
            failed_relaunch is not None and not supersede and not _audit_running(jobs_dir, proj)
            and (why := failed_relaunch.due(proj))
        ):  # fmt: skip
            # the failure's inputs changed (or it was never fingerprinted): relaunch from source
            supersede, target, auto = (proj, why), proj, True
            relaunch, res["relaunch"] = proj, why
            mine = [d for d in mine if d.project != relaunch]
            break
        if not failed:
            return {**res, "reason": f"audit in progress: {proj} holds the next slot"}
        if proj == target:
            # superseding the failed audit itself: its run is set aside and the row relaunches
            # as it now reads, re-registering the same store row
            relaunch, res["relaunch"] = target, failed
            mine = [d for d in mine if d.project != relaunch]
            break
        if supersede:  # an explicit supersede of another row still stops at a failed audit
            return {**res, "action": "refused", "project": proj, "reason": failed}
        # a failed audit never holds its region: it stays ``auditing`` for ``health heal`` to
        # resume from its stages (a later passed audit publishes), and the next row launches
        res.setdefault("failed_audits", []).append({
            "project": proj,
            "reason": _failed_audit(jobs_dir, proj, hint=False) + "; health heal keeps retrying it from its stages",
        })  # fmt: skip
    if held is not None:  # no audit behind it published: the refusal still holds the region
        later = res.get("publish_refusals")
        return {**held, **({"publish_refusals": later} if later else {})}
    if relaunch is None and failed_relaunch is not None and not supersede:
        # a failed store row heal cannot resume (no readable resume.json) is set aside and
        # launched from source once ``FailedRelaunch`` says its failure fingerprint changed
        for proj in (r.project for r in rows if r.region == region):
            if (
                any(d.project == proj and d.status == "failed" for d in mine)
                and not _project_job_running(jobs_dir, proj)
                and (why := failed_relaunch.due(proj))
            ):
                supersede, target, auto = (proj, why), proj, True
                relaunch, res["relaunch"] = proj, why
                mine = [d for d in mine if d.project != relaunch]
                break
    if auto:
        rows = [r for r in rows if r.project == target]
    slot = _slot_target(free_slots, sum(d.status == "ready" for d in mine), now, horizon_h)
    if slot is None:
        return {**res, "reason": f"covered: every free slot inside {horizon_h} h has a ready drive"}  # fmt: skip
    res.update(slot=slot[0].isoformat(), lf_day=slot[1])
    status = {d.project: d.status for d in drives if d.project != relaunch}
    staged = staged_sources(stages_root)
    if supersede and not supersede[1].strip():
        return {**res, "action": "refused", "reason": f"--supersede {target} needs a --reason"}
    own_root = Path(stages_root) / str(target)
    row = None
    siblings: List[str] = []
    gated = None  # the first row the render gate refused, when no row behind it is eligible
    reopen = False  # the own row is ``complete``: Waypost reopens it for the replace
    for cand in (r for r in rows if r.region == region):
        own = cand.project == target
        mine_staged = staged
        if own:
            footage = {os.path.realpath(p) for p in cand.sources}
            # a dead sibling (same footage, another project name) is set aside with the row's own
            # run; a held one stays in ``mine_staged`` so it still refuses
            siblings = sorted({
                st.relative_to(stages_root).parts[0] for st, srcs, held in staged
                if not held and own_root not in st.parents and footage & srcs
            })  # fmt: skip
            gone = {Path(stages_root) / n for n in siblings} | {own_root}
            mine_staged = [s for s in staged if not gone & set(s[0].parents)]
        cand_status, cand_known = status, known
        if own and replace is not None:  # the drive's own upload bookings are what is replaced
            held = status.get(cand.project)
            if (
                held != "complete"
            ):  # only Waypost's reopen can replace booked slots; never ``register``
                why = f"the schedule store row is {held or 'absent'}, not complete; --replace-scheduled replaces a complete drive's bookings, and only Waypost can reopen it"  # fmt: skip
                return {**res, "action": "refused", "reason": f"cannot supersede {target}: {why}"}
            reopen = True  # needs a replace plan: checked once the plan is known
            cand_status = {k: v for k, v in status.items() if k != cand.project}
            cand_known = known - {cand.project, cand.project.strip().lower()}
        why = _skip_reason(cand, cand_status, cand_known, mine_staged, stages_root, own_ok=own)
        if why is None:
            # the drive's own pre-ingest refusal, asked before anything is moved, launched or
            # booked; a refused row is skipped so it never stalls the rows behind it
            gate = render_gate(Path(sources_root) / cand.project, cand)
            if not gate:
                row = cand
                break
            why = f"render gate: {gate}"
            if own:
                return {**res, "action": "refused", "project": cand.project, "reason": why}
            gated = gated or (cand.project, why)
        res["skipped"].append({"project": cand.project, "reason": why})
        if own:
            return {**res, "action": "refused", "reason": f"cannot supersede {target}: {why}"}
    if target and (row is None or row.project != target):
        nxt = f"the next eligible {region} row is {row.project}" if row else f"no {region} row {target} in the backlog"  # fmt: skip
        return {**res, "action": "refused", "reason": f"cannot supersede {target}: {nxt}"}
    if row is None and gated:
        return {**res, "action": "refused", "project": gated[0], "reason": gated[1]}
    if row is None:
        return {**res, "reason": f"no eligible backlog row for {region}"}
    if supersede:
        blocked = [n for n in siblings if n in status or n in known]
        if blocked:
            where = "the schedule store" if blocked[0] in status else "the content ledger"
            why = f"cannot supersede {target}: sibling run {blocked[0]} is in {where}"
            return {**res, "action": "refused", "project": row.project, "reason": why}
        moves, why = _supersede_plan(row.project, stages_root, output_root, now, siblings, replace)
        if why:
            return {**res, "action": "refused", "project": row.project, "reason": why}
        if reopen and not (replace or {}).get("plan"):
            why = "the schedule store row is complete and no upload record gives a replace plan; only Waypost can reopen it"  # fmt: skip
            return {**res, "action": "refused", "project": row.project, "reason": f"cannot supersede {target}: {why}"}  # fmt: skip
        res["supersede"] = {"reason": supersede[1], "moves": moves}
    res["project"] = row.project
    root = Path(stages_root) / row.project
    stages_dir, out_dir = root / row.project / "_stages", Path(output_root) / row.project
    subset = Path(sources_root) / row.project
    res["storage"] = storage_plan(row.sources, root, Path(output_root), disk)
    res["storage_bound"] = STORAGE_BOUND
    short = [s for s in res["storage"] if not s["ok"]]
    if short:
        text = "; ".join(f"{s['free_gb']:.1f} GB free < {s['need_gb']:.1f} GB ({', '.join(s['uses'])})" for s in short)  # fmt: skip
        return {**res, "action": "refused", "reason": f"storage: {text}"}
    python = Path(code_root) / SERVICE_DIR / ".venv" / "bin" / "python"
    if not python.is_file():
        return {**res, "action": "refused", "reason": f"no service interpreter at {python}"}
    bad = _subset(subset, row.sources, create=not dry_run)
    if bad:
        return {**res, "action": "refused", "reason": bad}
    day = res["lf_day"]
    drive_argv = [
        str(python), "-m", "gomoveshift", "drive", str(subset), "--region", region,
        "--publish", day, "--lf-day", day, "--ingest-mode", row.ingest_mode,
        "--lf-mode", row.lf_mode, "--output-dir", str(out_dir), "--skip-upload",
        "--no-youtube-upload", "--ost-system-review", "--no-review", "--force", "--no-tmux",
    ]  # fmt: skip
    if row.day_gpx:
        drive_argv += ["--day-gpx", row.day_gpx]
        if row.camera_utc_offset_hours is not None:  # unset: the slicer measures it
            drive_argv += ["--camera-utc-offset-hours", str(row.camera_utc_offset_hours)]
    if getattr(row, "placeless_label", ""):  # claim-free only: _skip_reason refused the rest
        drive_argv += ["--placeless-label", row.placeless_label]
    if getattr(row, "bracket_gpx", ()):  # both files present: _skip_reason refused the rest
        drive_argv += ["--bracket-gpx", *[str(p) for p in row.bracket_gpx]]
    for idx, label in getattr(row, "per_lf_locations", ()) or ():  # _skip_reason validated
        drive_argv += ["--per-lf-location", f"{idx}={label}"]
    if getattr(row, "location", ""):
        drive_argv += ["--location", row.location]
    argv = _job_argv(code_root, jobs_dir, row.project, out_dir, drive_argv)
    env = {
        "GMS_INTERMEDIATE_DIR": str(root),
        run_origin.LAUNCHED_BY_ENV: run_origin.BACKLOG_LAUNCHER,
    }
    res.update(argv=argv, env=env, stages_dir=str(stages_dir))
    store, reopen_kw = None, {}
    if (
        reopen
    ):  # before any rename (and in a dry run too): ask Waypost whether it would reopen the row
        plan = replace["plan"]
        expected, replaced = plan_lfs(plan)
        reopen_kw = dict(expected=expected, replace=replaced, reason=supersede[1],
                         output_dir=str(out_dir), now=now)  # fmt: skip
        store = replace_store() if replace_store else None
        if store is None:
            return {**res, "action": "refused", "reason": f"cannot supersede {target}: no schedule store to reopen the complete row"}  # fmt: skip
        if not hasattr(store, "reopen_for_replace"):
            return {**res, "action": "refused", "reason": f"cannot supersede {target}: installed seo_manager lacks reopen_for_replace; repin"}  # fmt: skip
        why = store.reopen_for_replace(row.project, plan_digest=plan_digest(plan), dry_run=True, **reopen_kw)  # fmt: skip
        if why:
            return {**res, "action": "refused", "reason": f"cannot supersede {target}: Waypost will not reopen it: {why}"}  # fmt: skip
    if pending := install_pending(installed_version(python), pinned_version(Path(code_root))):
        return {**res, "action": "refused", "reason": pending}  # before any rename: nothing moves
    if dry_run:
        return {**res, "action": "would_launch", "reason": f"{row.project} for {res['slot']}"}
    stale = stale_pin_reason(python, installed_version, latest_version)
    if stale:
        return {**res, "action": "refused", "reason": stale}
    planned = (replace or {}).get("plan")
    if supersede:
        _supersede(row.project, supersede[1], res["supersede"]["moves"], now, planned, out_dir)

    def refuse(reason):
        """A refusal after the renames: put the drive back so a retry starts from the same state
        (a replace retry needs the old ``youtube_uploads.json`` to build its plan again)."""
        if planned:
            failed = _unsupersede(res["supersede"]["moves"], out_dir, now)
            if failed:
                replace_scheduled_mod.record_incident(row.project, "restore_failed", f"{reason}; the set-aside run could not be put back: {failed}")  # fmt: skip
                reason = f"{reason}; could not restore the set-aside run (see the replace_scheduled_partial incident): {failed}"  # fmt: skip
        return {**res, "action": "refused", "reason": reason}

    try:
        preflight(
            subset, out_dir, staging_dir=stages_dir, ingest_mode=row.ingest_mode, python=python
        )
    except PreflightBlockedError as exc:
        return refuse(str(exc))
    ok, detail = launcher(argv, str(Path(code_root) / SERVICE_DIR), env)
    if not ok:
        return refuse(f"launch failed: {detail}")
    if reopen:  # register_ready raises on a complete row: Waypost's own reopen replaces it
        reopen_kw["now"] = datetime.now(timezone.utc)  # not the tick's start: the launch took time
        why = store.reopen_for_replace(
            row.project, plan_digest=plan_digest(replace["plan"], out_dir), dry_run=False, **reopen_kw
        )  # fmt: skip
        if why:  # the audit is already running on the fresh dirs: they stay, a human reopens it
            logger.error("REPLACE-REOPEN-REFUSED: %s: %s (stages already set aside, audit launched)", row.project, why)  # fmt: skip
            msg = f"launched the audit but Waypost refused to reopen {row.project}: {why}"
            replace_scheduled_mod.record_incident(row.project, "reopen_refused", f"{msg}; the store row is still complete and the replace plan is in {out_dir}")  # fmt: skip
            return {**res, "action": "refused", "reason": msg}
    else:
        register(
            project=row.project, region=region, lf_count=1, status="auditing",
            output_dir=str(out_dir), mirror_sha=mirror_sha, lf_mode=row.lf_mode,
            ingest_mode=row.ingest_mode,
        )  # fmt: skip
    if auto:
        failed_relaunch.record_launch(row.project, res["relaunch"])
    return {**res, "action": "launched", "reason": f"{row.project} for {res['slot']}"}


def lease_script(code_root: Path) -> Path:
    """The lease script: ``$GMS_LEASE_SCRIPT`` (``~`` expanded), else ``bin/lease.sh`` in
    ``code_root``."""
    raw = os.environ.get(LEASE_SCRIPT_ENV, "").strip()
    return Path(raw).expanduser() if raw else Path(code_root) / DEFAULT_LEASE_SCRIPT


def lease_fns(code_root: Path, ttl_s: int = 600) -> Tuple[Callable[[], bool], Callable[[], None]]:
    """``lease_script`` acquire/release for ``LEASE_NAME`` (exit 0 = acquired, 3 = held)."""
    tool = str(lease_script(code_root))

    def acquire() -> bool:
        cmd = ["bash", tool, "acquire", LEASE_NAME, str(ttl_s), "health backlog"]
        done = subprocess.run(cmd, capture_output=True, timeout=30, check=False)  # noqa: S603,S607
        if done.returncode not in (0, 3):
            raise RuntimeError(f"lease acquire exited {done.returncode}: {done.stderr!r}")
        return done.returncode == 0

    def release() -> None:
        cmd = ["bash", tool, "release", LEASE_NAME]
        subprocess.run(cmd, capture_output=True, timeout=30, check=False)  # noqa: S603,S607

    return acquire, release


def default_launcher(argv: List[str], cwd: str, env: Dict[str, str]) -> Tuple[bool, str]:
    try:
        done = subprocess.run(  # noqa: S603 -- argv built from the manifest row
            argv, cwd=cwd, env={**os.environ, **env}, capture_output=True, text=True,
            timeout=120, check=False,
        )  # fmt: skip
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return done.returncode == 0, (done.stderr or done.stdout).strip()[-200:]


_PREFLIGHT_CHILD = """
import json, sys
from pathlib import Path
from gomoveshift.pipeline.preflight import PreflightBlockedError, run_preflight
src, out, staging, floor, staging_floor, mode = sys.argv[1:7]
try:
    run_preflight(Path(src), Path(out), min_free_gb=int(floor), staging_dir=Path(staging),
                  min_staging_gb=float(staging_floor), ingest_mode=mode)
except PreflightBlockedError as exc:
    print(json.dumps(exc.blockers))
    sys.exit(3)
"""


def default_preflight(source_dir, output_dir, *, staging_dir, ingest_mode, python) -> None:
    """The drive's own storage/tool preflight on this row's per-day inputs, before launch.

    Runs under the drive's interpreter (``python``), not this one: tool checks such as
    gopro-overlay belong to the venv that will render. Exit 3 carries the blockers as JSON;
    any other failure is a crash, never a pass."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    argv = [
        str(python), "-c", _PREFLIGHT_CHILD, str(source_dir), str(output_dir), str(staging_dir),
        str(int(OUTPUT_FLOOR_GB)), str(STAGING_FLOOR_GB), ingest_mode,
    ]  # fmt: skip
    done = subprocess.run(  # noqa: S603 -- fixed child script, paths from the manifest row
        argv, cwd=str(Path(python).parents[2]), capture_output=True, text=True, timeout=600,
        check=False,
    )  # fmt: skip
    if done.returncode == 3:
        raise PreflightBlockedError([tuple(b) for b in json.loads(done.stdout)])
    if done.returncode != 0:
        raise RuntimeError(f"preflight exited {done.returncode}: {done.stderr.strip()[-500:]}")


def load_inputs(config: Any, region: str, ledger_path: Optional[Path], now: datetime):
    """``(region, rows, drives, free_slots, known_projects, register)`` from the live stores:
    the manifest, ``schedule.sqlite3``, ``build_schedule_report`` and the content ledger."""
    from seo_manager.core.profile import canon_region
    from seo_manager.scheduling import backlog
    from seo_manager.scheduling.store import ScheduleStore

    from ..paths import drive_backlog_path
    from ..pipeline.content_ledger import ContentLedger
    from ..pipeline.schedule_report import build_schedule_report
    from ..pipeline.seo_slots import STORE_FILENAME, waypost_cadence

    region = canon_region(region)
    cad = waypost_cadence(config)
    drives, register = open_store(
        Path(config.seo.storage_path) / STORE_FILENAME,
        lambda path: ScheduleStore(path, cad.tz, cad.regions),
    )
    ledger = ContentLedger.load(ledger_path)
    report = build_schedule_report(ledger, config, after=now)
    slots = report_slots(report, region)
    known = {(e.project_name or "").strip().lower() for e in ledger.entries} - {""}
    rows = backlog.load(drive_backlog_path())
    return region, rows, drives, slots, known, register


def render_horizon_h(config: Any) -> int:
    """How far ahead ``health backlog`` renders: the wider of ``COVERAGE_HORIZON_H`` and the
    cadence's ``lead_hours``. An audited drive publishes only into a slot it can claim, and a
    claim is refused past ``lead_hours``, so rendering further ahead than that would leave a
    finished drive holding its scratch. Rendering as far as it reaches lets the backlog run
    drive after drive instead of one a week (JP 2026-11-14)."""
    from ..pipeline.seo_slots import waypost_cadence

    return max(COVERAGE_HORIZON_H, int(waypost_cadence(config).lead_hours))


def store_retire(config: Any) -> Callable[[str], None]:
    """``retire(project)``: ``set_status failed`` in ``schedule.sqlite3``, opened only when a
    retirement happens."""

    def retire(project: str) -> None:
        from seo_manager.scheduling.store import ScheduleStore

        from ..pipeline.seo_slots import STORE_FILENAME, waypost_cadence

        cad = waypost_cadence(config)
        path = Path(config.seo.storage_path) / STORE_FILENAME
        ScheduleStore(path, cad.tz, cad.regions).set_status(project, "failed")

    return retire


def open_store(store_path: Path, make: Callable[[Path], Any]):
    """``(drives, register)`` for ``schedule.sqlite3``. An absent store has no drives (as in
    ``health check``) and is not created by reading it; only a launch's register creates it."""
    if not store_path.is_file():
        return [], lambda **kw: make(store_path).register_ready(**kw)
    store = make(store_path)
    return store.all_drives(), store.register_ready
