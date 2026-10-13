"""Re-derive OST text for an existing pipeline state without rerunning ingest.

Use case: the OST derivation code has been improved (e.g. importance
ranking, state append, drive-level span) and an already-ingested drive
needs to pick up the new text without re-running Phase A. Loads the saved
pipeline state, re-derives per-episode and drive-level OST from the
referenced GPX files, and writes the updated state back with a timestamped
backup.

CLI entry point:
    python -m gomoveshift.pipeline.redrive_ost [--dry-run] <state.json>

Also exposes `redrive_state(state_path, dry_run)` for testing and
programmatic use.
"""

import argparse
import logging
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from ._drive_helpers import (
    _bearing_to_cardinal,  # noqa: F401 — re-export for tests if needed
    differentiate_duplicate_ost,
)
from ._drive_state import load_drive_state, save_drive_state
from .ingest import (
    _get_route_analyzer,
    _get_shared_geocoder,
    derive_drive_ost_text,
    derive_episode_ost_text,
)
from .ost_overrides import apply_overrides, load_overrides
from .ost_validator import validate_ost

logger = logging.getLogger(__name__)


@dataclass
class RedriveResult:
    state_path: Path
    before_episode_locations: list[str] = field(default_factory=list)
    after_episode_locations: list[str] = field(default_factory=list)
    before_drive: str = ""
    after_drive: str = ""
    backup_path: Optional[Path] = None

    def pretty_diff(self) -> str:
        """Human-readable before/after summary."""
        lines = []
        lines.append("Drive-level OST:")
        lines.append(f"  BEFORE: {self.before_drive!r}")
        lines.append(f"  AFTER:  {self.after_drive!r}")
        lines.append("Per-episode OST:")
        for i, (before, after) in enumerate(
            zip(self.before_episode_locations, self.after_episode_locations)
        ):
            marker = " " if before == after else "*"
            lines.append(f"  {marker}Ep {i + 1}: {before!r} -> {after!r}")
        return "\n".join(lines)


def _extract_episode_meta_from_gpx(gpx_path) -> dict:
    """Local copy of the orchestrator helper — avoids the import cycle that
    would come from importing orchestrator (which imports heavy modules)."""
    if not gpx_path:
        return {"start": (0.0, 0.0), "end": (0.0, 0.0), "date": None}
    try:
        from gps_journey.core.gpx_parser import GPXParser

        parser = GPXParser()
        journey = parser.parse(Path(gpx_path))
        wps = list(journey.waypoints or [])
        if not wps:
            return {"start": (0.0, 0.0), "end": (0.0, 0.0), "date": None}
        return {
            "start": (wps[0].latitude, wps[0].longitude),
            "end": (wps[-1].latitude, wps[-1].longitude),
            "date": getattr(wps[0], "timestamp", None),
        }
    except Exception:
        return {"start": (0.0, 0.0), "end": (0.0, 0.0), "date": None}


def redrive_state(state_path: Path, dry_run: bool = False) -> RedriveResult:
    """Re-derive OST text for the pipeline state at state_path.

    Args:
        state_path: Path to the pipeline_state.json file.
        dry_run: If True, computes the new OST but does not write the state
            file or create a backup.
    """
    state_path = Path(state_path)
    if not state_path.exists():
        raise FileNotFoundError(f"State file not found: {state_path}")

    # Route through the same load/save pair every other phase uses
    # (`load_drive_state`/`save_drive_state`) instead of bare JSON I/O, so
    # redrive gets the Layer-1 invariant check and the atomic write for
    # free rather than re-implementing (and under-implementing) either.
    state = load_drive_state(state_path)

    before_episode = list(state.get("episode_locations") or [])
    before_drive = state.get("location") or ""

    gpx_files = state.get("gpx_files") or []

    # Build fresh geocoder + analyzer (same pattern as the orchestrator).
    try:
        geocoder = _get_shared_geocoder()
        route_analyzer = _get_route_analyzer(geocoder)
    except Exception as e:
        logger.warning("Route analyzer init failed: %s — OST will be empty", e)
        geocoder = None
        route_analyzer = None

    after_episode: list[str] = []
    episode_meta: list[dict] = []
    for gpx_path in gpx_files:
        ep_ost = derive_episode_ost_text(
            gpx_path,
            route_analyzer=route_analyzer,
            geocoder=geocoder,
        )
        if ep_ost:
            ok, reason = validate_ost(ep_ost, "lf")
            if not ok:
                logger.warning("Validator dropped %r: %s", ep_ost, reason)
                ep_ost = ""
        after_episode.append(ep_ost)
        episode_meta.append(_extract_episode_meta_from_gpx(gpx_path))

    # Differentiate duplicates — mirrors orchestrator pipeline
    after_episode = differentiate_duplicate_ost(after_episode, episode_meta)

    valid_gpx = [p for p in gpx_files if p]
    after_drive = ""
    if valid_gpx:
        after_drive = derive_drive_ost_text(
            valid_gpx,
            route_analyzer=route_analyzer,
            geocoder=geocoder,
            episode_locations=after_episode,
        )
        if after_drive:
            ok, reason = validate_ost(after_drive, "lf")
            if not ok:
                logger.warning("Validator dropped drive OST %r: %s", after_drive, reason)
                after_drive = ""

    # Apply user-final-say overrides from ost_overrides.json next to the
    # state file. Runs AFTER validation so the operator can pin names the
    # algorithm can't reach.
    override_dir = state_path.parent
    overrides = load_overrides(override_dir)
    if overrides:
        after_episode, after_drive = apply_overrides(
            after_episode,
            after_drive,
            overrides,
        )

    result = RedriveResult(
        state_path=state_path,
        before_episode_locations=before_episode,
        after_episode_locations=after_episode,
        before_drive=before_drive,
        after_drive=after_drive,
    )

    if dry_run:
        return result

    # Write a timestamped backup, then overwrite state via the atomic
    # writer. The backup is still worth keeping even though the write below
    # is now crash-safe: atomicity only protects against a torn file on a
    # mid-write crash, it does nothing for an operator who wants the OLD
    # (pre-redrive) OST text back because the new derivation is legitimate
    # JSON but simply wrong — that's a human rollback path, not a
    # corruption guard, so it stays.
    # Use shutil.copyfile — the state file commonly lives on an NTFS mount
    # under a /mnt/<letter> mount, where BOTH copystat() (mtime/atime) and copymode()
    # (chmod) raise PermissionError. copy2 calls copystat; copy calls
    # copymode; only copyfile skips metadata preservation entirely.
    # A recovery backup doesn't need preserved timestamps or mode bits.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = state_path.with_suffix(f".bak.{timestamp}.json")
    shutil.copyfile(state_path, backup_path)
    result.backup_path = backup_path

    state["episode_locations"] = after_episode
    if after_drive:
        state["location"] = after_drive
    # save_drive_state runs the Layer-1 invariant check and writes via
    # atomic_write_json (tempfile + os.replace) — a crash/disk-full
    # partway through leaves the pre-existing state file on disk intact
    # instead of truncated.
    save_drive_state(state_path, state)

    return result


def _configure_logging() -> None:
    """Errors render with the level/logger-name format the CLI uses.

    Same stdout handler and format as ``cli.logging_setup.configure_logging``
    (INFO, no file); inlined because the engine never imports ``cli``
    (``tests/test_media_boundary.py``).
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s %(message)s", datefmt="%H:%M:%S")
    )
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m gomoveshift.pipeline.redrive_ost",
        description="Re-derive OST text for an existing pipeline state.",
    )
    parser.add_argument("state_file", type=Path, help="Path to pipeline_state.json")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute new OST but do not write the state file.",
    )
    args = parser.parse_args(argv)

    _configure_logging()

    try:
        result = redrive_state(args.state_file, dry_run=args.dry_run)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    print(result.pretty_diff())
    if args.dry_run:
        print("\n(dry-run — state file not modified)")
    else:
        print(f"\nBackup written to: {result.backup_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
