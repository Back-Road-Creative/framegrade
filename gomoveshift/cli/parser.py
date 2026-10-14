"""Argument parser construction for the GoMoveShift CLI.

Extracted from the monolithic ``_build_parser()`` in the old ``cli.py``.
Each command's arguments are defined in a dedicated helper so that the
top-level parser definition stays readable.
"""

import argparse
import importlib.metadata
import json
import re
from datetime import date, datetime
from pathlib import Path

from ..config import LAYER_MASK_PARAMS, WB_OPERATOR_SPAN
from ..paths import gms_output_dir

#: The mask names ``--layers`` may name, READ off the same mirror of the registry the
#: config door validates against (``config`` cannot import the pipeline, and importing it
#: here would drag the whole photos package into every ``--help``). Hand-listed until now,
#: and it had gone stale: ``glints`` and ``texture`` shipped and no operator could find
#: them in the help text. Derived, a mask cannot ship undiscoverable again.
_LAYER_MASK_NAMES = ", ".join(LAYER_MASK_PARAMS)

#: Mirrors astro_combine.CLI_OUTPUT_SCOPES / DEFAULT_OUTPUT_SCOPE -- hand-listed, same reason
#: as `_LAYER_MASK_NAMES`: importing the photos pipeline here would drag it into every --help.
_ASTRO_COMBINE_SCOPES = ("auto", "full", "scoped")
_ASTRO_COMBINE_DEFAULT_SCOPE = "auto"


def _per_lf_location_kv(raw: str) -> tuple[int, str]:
    """argparse ``type=`` for ``--per-lf-location KEY=VALUE``.

    KEY is the integer LF index (1-based, matching ``RegularLFPlan.index``).
    VALUE is the override label (non-empty). Whitespace around both sides
    is trimmed. Cross-LF coverage and equal-to-``--location`` rejection
    happen later — argparse-time only validates SHAPE here so repeated
    ``--per-lf-location`` flags accumulate into a list of (idx, label)
    tuples that downstream code reduces to a dict.
    """
    if not isinstance(raw, str) or "=" not in raw:
        raise argparse.ArgumentTypeError(f"--per-lf-location expects 'INDEX=LABEL' (got {raw!r})")
    key, _, value = raw.partition("=")
    key = key.strip()
    value = value.strip()
    try:
        idx = int(key)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--per-lf-location INDEX must be an integer (got {key!r})"
        ) from exc
    if idx < 1:
        raise argparse.ArgumentTypeError(f"--per-lf-location INDEX must be >= 1 (got {idx})")
    if not value:
        raise argparse.ArgumentTypeError(f"--per-lf-location LABEL must be non-empty (got {raw!r})")
    return (idx, value)


def _insert_clip_arg(raw: str) -> str:
    """argparse ``type=`` for ``--insert-clip``: validate ``PATH@MM:SS``, keep the raw spec."""
    from ..pipeline.inserts import InsertError, parse_insert_clip

    try:
        parse_insert_clip(raw)
    except InsertError:
        raise argparse.ArgumentTypeError(f"{raw!r}: expected PATH@MM:SS (e.g. clip.mp4@01:30)")
    return raw


def _comma_components(s: str) -> list[str]:
    """Split a label on commas, strip whitespace, drop empties."""
    return [c.strip() for c in s.split(",") if c.strip()]


def _is_comma_suffix(candidate: str, label: str) -> bool:
    """True iff ``candidate`` matches a trailing comma-delimited slice of
    ``label`` at the comma boundary (not substring).

    "Example County, Exampleland" IS a comma-suffix of
    "Alpha to Beta Pass, Example County, Exampleland".
    "Exampleland" IS a comma-suffix.
    "Beta Pass, Example County, Exampleland" is NOT — its first component
    would have to be a whole comma-delimited segment of the label, but the
    label puts "Beta Pass" inside the first segment "Alpha to Beta Pass".
    """
    cand = _comma_components(candidate)
    lbl = _comma_components(label)
    if not cand or len(cand) > len(lbl):
        return False
    return lbl[-len(cand) :] == cand


def _longest_common_comma_suffix(component_lists: list[list[str]]) -> list[str]:
    """Longest common trailing slice across N comma-component lists."""
    if not component_lists:
        return []
    common = list(component_lists[0])
    for other in component_lists[1:]:
        i = 1
        while i <= len(common) and i <= len(other) and common[-i] == other[-i]:
            i += 1
        common = common[-(i - 1) :] if i > 1 else []
        if not common:
            break
    return common


def _suggest_drive_wide_from_per_lf(labels: list[str]) -> str:
    """Derive a candidate ``--location`` from per-LF labels.

    Strategy: drop the first comma-component of each label (the
    route/destination/highlight that varies LF-to-LF), then take the
    longest common comma-suffix of the remainders. For a single label
    with N components, this returns the trailing N-1 components.
    """
    if not labels:
        return ""
    tails: list[list[str]] = []
    for lbl in labels:
        comps = _comma_components(lbl)
        tails.append(comps[1:] if len(comps) >= 2 else comps)
    common = _longest_common_comma_suffix(tails)
    return ", ".join(common)


def _validate_per_lf_location_args(args: argparse.Namespace) -> None:
    """Post-parse validation for ``--per-lf-location``.

    Argparse-time shape validation already happened via
    ``_per_lf_location_kv``. Here we enforce three cross-flag rules:

      1. Reject when EVERY supplied override label equals
         ``--location`` verbatim — per-LF labels must be more specific
         than the drive-wide label, otherwise the operator's intent is
         ambiguous and we'd silently re-introduce the drive-wide
         fallback we're trying to delete.
      2. When ``--per-lf-location`` is set and ``--location``
         is set, ``--location`` must have >= 2 comma-separated
         components. Single-component values like ``"Exampleland"`` discard
         the operator's geographic intent and end up as the visible
         drive-wide fallback in Gate A / Super-LF / description-Filmed
         surfaces. Region + country (e.g. ``"Example County, Exampleland"``) is the
         floor.
      3. When ``--per-lf-location`` is set and ``--location``
         is set, ``--location`` must be a comma-suffix of every
         per-LF label (matched at the comma boundary, not as a
         substring). Catches mismatched specificity where the operator
         puts a highlight in drive-wide that isn't a clean tail of
         per-LF — e.g. ``--location "Beta Pass, Example County, Exampleland"`` with
         ``per-LF "Alpha to Beta Pass, Example County, Exampleland"``.

    Cross-LF duplicate labels are *allowed*. Mixing some overrides equal
    to ``--location`` with others that aren't is *allowed* (gate 1 only
    fires on ALL == location). The error messages name the suggested
    ``--location`` derived from the per-LF labels so the operator sees
    the right answer in the rejection.

    Coverage (every GPS-less LF must have an override) is checked
    post-discovery in PhaseBC, not here, because LF indices aren't known
    until clip discovery runs.
    """
    overrides = getattr(args, "per_lf_locations", None) or []
    if not overrides:
        return
    # Rule 0 — a label opening with a bare preposition ("To Goleen, …") burns,
    # then becomes the YouTube title, which the pre-upload metadata gate
    # refuses hours later. Same pattern, refused pre-encode.
    from ..pipeline.validation.content_gates import _GRAMMAR_LEADING_PREP_RE

    for idx, label in overrides:
        if _GRAMMAR_LEADING_PREP_RE.match(label.strip()):
            raise argparse.ArgumentTypeError(
                f"--per-lf-location {idx}={label!r} starts with a bare preposition; "
                "the YouTube title derives from this label and the metadata gate "
                "refuses it before upload. Pass the place name (e.g. 'Goleen, County Cork')."
            )
    location = (getattr(args, "location", "") or "").strip()
    if not location:
        return
    labels = [label.strip() for _idx, label in overrides]

    # Rule 1 — all-equal
    if labels and all(lbl == location for lbl in labels):
        raise argparse.ArgumentTypeError(
            "every --per-lf-location label equals --location "
            f"({location!r}) — per-LF labels must be more specific than "
            "the drive-wide label (required input when GPS "
            "is absent). Provide route-specific labels (e.g. start to "
            "end via road) or omit the flag."
        )

    # Rule 2 — drive-wide specificity floor
    location_comps = _comma_components(location)
    if len(location_comps) < 2:
        suggested = _suggest_drive_wide_from_per_lf(labels)
        suggestion = f" Suggested: --location {suggested!r}." if suggested else ""
        raise argparse.ArgumentTypeError(
            f"--location {location!r} has {len(location_comps)} comma-separated "
            "component(s) — when --per-lf-location is set, --location must "
            "have >= 2 components (region + country, e.g. 'Example County, Exampleland'). "
            "Single-token country values discard the geographic intent and "
            "surface as the drive-wide fallback in Gate A and any non-LF "
            "metadata." + suggestion
        )

    # Rule 3 — comma-suffix consistency
    for idx, label in overrides:
        if not _is_comma_suffix(location, label.strip()):
            suggested = _suggest_drive_wide_from_per_lf(labels)
            suggestion = f" Suggested: --location {suggested!r}." if suggested else ""
            raise argparse.ArgumentTypeError(
                f"--location {location!r} is not a comma-suffix of "
                f"--per-lf-location {idx}={label!r}. Drive-wide must match "
                "the trailing comma-delimited components of every per-LF "
                "label so the geographic hierarchy is consistent." + suggestion
            )


# ---------------------------------------------------------------------------
# Per-command argument builders
# ---------------------------------------------------------------------------


def _add_analyze_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("analyze", help="Analyze a video file")
    p.add_argument("video", type=Path, help="Path to video file")
    p.add_argument("-o", "--output", type=Path, help="Output JSON file for analysis")
    p.add_argument("--no-frames", action="store_true", help="Don't save detected frames")


def _add_process_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("process", help="Process video with existing analysis")
    p.add_argument("video", type=Path, help="Path to video file")
    p.add_argument("analysis", type=Path, help="Path to analysis JSON file")
    p.add_argument("-o", "--output-dir", type=Path, help="Output directory")


def _add_server_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("server", help="Start web UI server")
    p.add_argument("-p", "--port", type=int, default=8765, help="Port to run on")
    p.add_argument("-H", "--host", default="127.0.0.1", help="Host to bind to")
    p.add_argument(
        "--allow-remote",
        action="store_true",
        help=(
            "Allow binding to a non-loopback host. Required only when fronting "
            "the UI with an auth proxy — the API itself has no authentication."
        ),
    )
    p.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Write logs to this file as well as stderr",
    )
    p.add_argument(
        "--log-json",
        action="store_true",
        default=False,
        help="Emit logs as one-per-line JSON for ingestion by structured-log collectors",
    )


def _add_console_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "console", help="Start the creator console (FastAPI + Jinja, fronts the CLIs)"
    )
    p.add_argument("-p", "--port", type=int, default=8787, help="Port to run on")
    p.add_argument("-H", "--host", default="127.0.0.1", help="Host to bind to")


def _add_scan_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("scan", help="Quick scan video for info")
    p.add_argument("video", type=Path, help="Path to video file")


def _add_pipeline_parser(subparsers: argparse._SubParsersAction) -> None:
    _default_output = gms_output_dir()
    p = subparsers.add_parser(
        "pipeline",
        help="Run full long form processing pipeline (stop removal, color, OST, audio, metadata)",
    )
    p.add_argument("video", type=Path, help="Path to arranged video export")
    p.add_argument(
        "--location",
        "-l",
        default=None,
        help="Location name (e.g., 'Example County, Exampleland'). "
        "Optional when --gps is provided; auto-derived from GPS data",
    )
    p.add_argument(
        "--publish", "-p", required=True, help="Publish day (mon/tue/wed/thu/fri/sat/sun)"
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output,
        help=f"Base output directory (default: {_default_output})",
    )
    p.add_argument(
        "--lut", type=Path, default=None, help="Path to .cube LUT file for color correction"
    )
    p.add_argument(
        "--crossfade",
        type=float,
        default=0.5,
        help="Crossfade duration at stop cuts in seconds (default: 0.5)",
    )
    p.add_argument("--skip-color", action="store_true", help="Skip color correction stage")
    p.add_argument("--skip-stops", action="store_true", help="Skip stop detection/removal stage")
    p.add_argument(
        "--fresh", action="store_true", help="Force restart from scratch (ignore saved state)"
    )
    p.add_argument(
        "--no-tmux", action="store_true", help="Skip tmux launcher (run directly in current shell)"
    )
    p.add_argument("--skip-thumbnails", action="store_true", help="Skip thumbnail generation stage")
    p.add_argument("--skip-shorts", action="store_true", help="Skip shorts analysis stage")
    p.add_argument("--skip-schedule", action="store_true", help="Skip schedule generation stage")
    p.add_argument("--skip-social", action="store_true", help="Skip social post generation stage")
    p.add_argument("--skip-upload", action="store_true", help="Skip Google Drive upload stage")
    p.add_argument("--skip-csv", action="store_true", help="Skip Metricool CSV generation stage")
    p.add_argument(
        "--skip-chapters", action="store_true", help="Skip YouTube chapter generation stage"
    )
    p.add_argument(
        "--gps",
        type=Path,
        default=None,
        help="Path to GPX file (default: same dir as video, matching name or first .gpx found)",
    )
    p.add_argument(
        "--gps-date",
        type=str,
        default=None,
        help="GPS recording date (YYYY-MM-DD) to select track from multi-track GPX file",
    )
    p.add_argument(
        "--gps-range",
        type=str,
        default=None,
        help="Time range within GPS track in minutes (START-END, e.g. '0-60' or '60-191')",
    )
    p.add_argument(
        "--draft",
        action="store_true",
        help="Mark output as draft (_DRAFT filename stamp) -- same encoding quality",
    )
    p.add_argument(
        "--skip-d-copy", action="store_true", help="Skip copying deliverables to D: drive"
    )
    p.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Skip post-pipeline cleanup of staging and intermediate files",
    )


def _add_drive_parser(subparsers: argparse._SubParsersAction) -> None:
    _default_output = gms_output_dir()
    p = subparsers.add_parser(
        "drive",
        help="Process a full drive (GoPro files) into Super LF + Regular LFs + Shorts",
    )
    p.add_argument("directory", type=Path, help="Directory with GoPro MP4 files")
    p.add_argument(
        "--region",
        default=None,
        choices=["ireland", "usa"],
        help="Content region. Default: preserve the saved region on resume, "
        "else 'ireland'. An absent --region does not overwrite a resumed "
        "run's region. The publish day comes from --lf-day / --publish or the "
        "configured long-form days, not from the region.",
    )
    p.add_argument(
        "--publish",
        "-p",
        default=None,
        help="Override publish day (default: the earliest configured long-form day). Implies "
        "--youtube-upload; pass --no-youtube-upload to opt out.",
    )
    p.add_argument("--location", "-l", default="", help="Location name for metadata/OST")
    # Explicit per-LF labels for GPS-less / legacy drives. Repeatable
    # ``KEY=VALUE`` (KEY = LF index, VALUE = label). Format-validated at
    # argparse time via ``_per_lf_location_kv``. The post-discovery
    # validation gate in PhaseBC checks coverage against the actual LF
    # set; argparse only enforces shape here.
    p.add_argument(
        "--per-lf-location",
        action="append",
        type=_per_lf_location_kv,
        dest="per_lf_locations",
        default=None,
        help="Repeatable per-LF location override (format: 'INDEX=LABEL', "
        "e.g. --per-lf-location '1=Alpha to Beta via Route 9, Example County'). "
        "Required for GPS-less drives where centroid-derivation can't "
        "produce a route-specific label. Use --list-lfs to learn LF "
        "indices first.",
    )
    # Discovery-only mode. Runs Phase A + builds the drive plan,
    # prints the LF index → suggested-label table, exits 0 without
    # encoding. Operators use the table to author --per-lf-location
    # values for a follow-up invocation.
    p.add_argument(
        "--list-lfs",
        action="store_true",
        default=False,
        help="Run clip discovery only, print the LF index → suggested-"
        "label table, then exit 0. Use to learn LF indices before "
        "passing --per-lf-location overrides on a re-invocation.",
    )
    # Verification harness: like --list-lfs but resolves each LF's real
    # label through the dominant-county reducer + live geocoder, printing
    # the corrected label + county vote breakdown. Confirms labels are
    # right before an expensive re-encode. Phase A only; exits 0.
    p.add_argument(
        "--label-dry-run",
        action="store_true",
        default=False,
        help="Run clip discovery only, print each LF's corrected "
        "dominant-county label + vote breakdown, then exit 0. No encode. "
        "Use to verify labels (e.g. after a mislabel report) before rerun.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output,
        help=f"Base output directory (default: {_default_output})",
    )
    # Explicit ingest-mode flag so the operator commits to one source layout per run. Default is
    # ``None`` (sentinel) so the orchestrator can distinguish "operator
    # didn't say" from "operator chose gopro"; downstream code falls
    # back to ``config.drive.ingest_mode``.
    p.add_argument(
        "--ingest-mode",
        choices=["gopro", "generic"],
        default=None,
        help="Source ingest layout. 'gopro' = chaptered GH/GX files "
        "(default). 'generic' = one-file-per-episode (Insta360, drone, "
        "phone exports). Overrides config.drive.ingest_mode.",
    )
    # LF selection mode. ``best_moments``
    # is the legacy sliding-window planner that picks contiguous best
    # ranges. ``full_drive`` returns one LF per day spanning the full
    # post-stop-removal timeline (K balanced parts when
    # config.drive.full_drive_part_min_sec/max_sec are set) — the Insta360 use case. Default
    # ``None`` (sentinel) so the drive handler can distinguish
    # "operator didn't say" from "operator chose best_moments";
    # downstream falls back to ``config.drive.lf_selection_mode``.
    p.add_argument(
        "--lf-mode",
        choices=["best_moments", "full_drive"],
        default=None,
        help="LF selection mode. 'best_moments' (default) = sliding-window "
        "best contiguous ranges. 'full_drive' = one LF per day spanning the entire "
        "post-stop-removal timeline, or K balanced parts per day when "
        "config.drive.full_drive_part_min_sec/max_sec are set (Insta360 use case). Overrides "
        "config.drive.lf_selection_mode.",
    )
    # First-class operator knob for "I want a single highlight LF of EXACTLY N
    # minutes" — fills the gap between best_moments (variable duration, bins
    # into [min_lf_duration, max_lf_duration]) and full_drive (whole
    # timeline). Wires through cmd_drive to clamp best_moments_target_min ==
    # best_moments_target_max and widen the packer's max_lf_duration so the
    # filter's output fits in a single LF. Only valid with --lf-mode
    # best_moments (default); rejected with full_drive.
    p.add_argument(
        "--lf-target-duration",
        type=float,
        default=None,
        help="LF target duration in MINUTES (best_moments mode only). "
        "Enables the dynamic best_moments filter and clamps target_min == "
        "target_max to the requested duration so the LF lands at exactly "
        "that length. Also widens config.drive.max_lf_duration if needed so "
        "the packer can hold the target in one LF. Example: "
        "--lf-target-duration 15 → one ~15-minute highlight LF. "
        "Incompatible with --lf-mode full_drive.",
    )
    # dest matches the PipelineState field so flag_registry's completeness
    # test catches a missing row. Sentinel: default None survives resume.
    p.add_argument(
        "--superlong",
        dest="superlong_enabled",
        action="store_const",
        const=True,
        default=None,
        help="Additionally produce whole-day superlong parts (sl_NN.mp4) "
        "capped at config.drive.superlong_part_max_duration_sec each.",
    )
    # dest matches the PipelineState field so flag_registry's completeness
    # test catches a missing row. Sentinel: default None survives resume.
    p.add_argument(
        "--superlong-day",
        dest="superlong_publish_day",
        choices=["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
        default=None,
        help="Per-run override for the superlong part publish weekday "
        "(overrides config.drive.superlong_publish_day / "
        "superlong_publish_day_by_region). Same vocabulary as --lf-day.",
    )
    p.add_argument("--fresh", action="store_true", help="Force restart from scratch")
    p.add_argument("--no-tmux", action="store_true", help="Skip tmux launcher")
    # The four upload flags use the
    # argparse-sentinel pattern — default=None, action="store_const",
    # const=True. The orchestrator then knows whether the flag was
    # EXPLICITLY supplied (not None) versus defaulted (None), so its
    # sidecar auto-skip safety net can step aside whenever the CLI was
    # explicit on upload intent.
    p.add_argument(
        "--skip-upload",
        action="store_const",
        const=True,
        default=None,
        help="Skip BOTH Drive shorts upload AND YouTube LF upload "
        "(back-compat alias for --skip-drive-upload + --skip-youtube-upload)",
    )
    p.add_argument(
        "--skip-drive-upload",
        action="store_const",
        const=True,
        default=None,
        help="Skip Google Drive shorts upload only. YouTube upload still "
        "controlled by --youtube-upload / --skip-youtube-upload.",
    )
    p.add_argument(
        "--skip-youtube-upload",
        action="store_const",
        const=True,
        default=None,
        help="Skip YouTube LF upload only. Drive shorts upload still runs "
        "(unless --skip-drive-upload is also passed). Useful for regenerating "
        "Metricool CSVs with fresh Drive links without re-publishing existing "
        "YouTube videos.",
    )
    p.add_argument("--skip-csv", action="store_true", help="Skip Metricool CSV generation")
    p.add_argument(
        "--youtube-upload",
        action="store_const",
        const=True,
        default=None,
        help="Enable YouTube upload. Also implied by --publish (which enables "
        "upload by default); pass --no-youtube-upload to opt out.",
    )
    p.add_argument(
        "--phase",
        choices=["A", "BC", "D"],
        help="Run a single phase (requires prior phases completed). "
        "BC = phase_bc_day_first (encode + assemble per day).",
    )
    p.add_argument(
        "--preview",
        action="store_true",
        default=False,
        help="Encode 90s preview clip from middle of each episode then exit (no full encode)",
    )
    p.add_argument(
        "--no-review",
        action="store_true",
        help="Skip human review gates (automated checks still run)",
    )
    p.add_argument(
        "--skip-validation",
        action="store_true",
        default=False,
        help="EMERGENCY ONLY: bypass the terminal DoD validator gate. "
        "Every use logs a WARNING; never set via env variable.",
    )
    p.add_argument(
        "--no-audio",
        action="store_true",
        default=None,
        help="Strip audio from long-form outputs: the day master is encoded "
        "-an so every LF ships silent (shorts are always muted). User adds "
        "music externally. Argparse-sentinel (default None) so a resume "
        "without the flag preserves a saved --no-audio intent.",
    )
    p.add_argument(
        "--start-date",
        default=None,
        help="First long-form publish date (YYYY-MM-DD), overrides auto-scheduling",
    )
    p.add_argument(
        "--lf-publish-date",
        default=None,
        help="Rebase-on-stale-resume backstop: explicit first-LF "
        "publish date YYYY-MM-DD. Skips both the auto-rebase guard logic "
        "and the original ``start_date`` resolution — operator-authoritative "
        "anchor for resumed runs. "
        "The date must fall on the run's LF day (--lf-day/--publish). The "
        "launch is refused if it is off that day, in the past, or booked by "
        "another project; the message names the next free date.",
    )
    p.add_argument(
        "--day-gpx",
        default=None,
        metavar="PATH",
        help="Whole-day GPX to slice into one <stem>.gpx sidecar per VID_*.mp4 "
        "in the source dir before ingest (generic mode, no-GPMF cameras). "
        "Requires --camera-utc-offset-hours. Strict filename contract: only "
        "VID_YYYYMMDD_HHMMSS_NN_NNN.mp4 names are sliced (the clock comes "
        "from the name); LRV_* proxies are skipped, any other .mp4 is "
        "refused. Not written under --dry-run. Sidecars it wrote are "
        "rewritten; an operator's own <stem>.gpx is never overwritten.",
    )
    p.add_argument(
        "--camera-utc-offset-hours",
        type=float,
        default=None,
        metavar="N",
        help="Camera clock UTC offset (local = UTC + N; US Eastern DST is -4). "
        "REQUIRED with --day-gpx and never guessed.",
    )
    p.add_argument(
        "--force-no-rebase",
        action="store_const",
        const=True,
        default=None,
        help="Third backstop: disable the schedule-rebase logic "
        "entirely. Use if the three-guard heuristic ever misfires. "
        "Mutually orthogonal to ``--lf-publish-date`` (explicit wins).",
    )
    # Per-invocation cadence overrides. They beat config-file values. Resolved
    # in cmd_drive BEFORE schedule_constants.apply_config so every lazy
    # downstream reader observes the per-run cadence.
    p.add_argument(
        "--lf-day",
        choices=["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
        default=None,
        help="Long-form publish weekday for this run, narrowing Waypost's days "
        "(see `gomoveshift schedule`). Single weekday only; "
        "drives the scheduler's window math for the next-LF boundary.",
    )
    p.add_argument(
        "--lf-time",
        default=None,
        help="Long-form publish time HH:MM 24h EST for this run "
        "(overrides Waypost's time for the day).",
    )
    p.add_argument(
        "--shorts-times",
        default=None,
        help="Comma-separated shorts publish times HH:MM 24h EST for this "
        "run, e.g. '14:00,18:00' (overrides "
        "config.drive.shorts_publish_times).",
    )
    p.add_argument(
        "--shorts-per-day",
        type=int,
        default=None,
        help="Number of shorts per day (default: all of the region's configured "
        "shorts windows). Truncates to the first N windows — cannot add slots "
        "beyond what's configured for the region/day.",
    )
    p.add_argument(
        "--shorts-target",
        type=int,
        default=None,
        help="Cap on shorts extracted per long form. The count is Waypost's "
        "decision (read via volume_source); this flag can only lower it, never "
        "raise it. Multiplied by the number of LFs in each day to produce the "
        "day-level plan_shorts target.",
    )
    p.add_argument(
        "--shorts-gap",
        type=float,
        default=None,
        help="Minimum seconds between shorts (default: 60). Lower values "
        "allow more shorts from shorter LFs.",
    )
    p.add_argument(
        "--shorts-min-duration",
        type=float,
        default=None,
        help="Minimum short duration in seconds (default: 30, YouTube "
        "Shorts compliance minimum). Overrides "
        "config.drive.shorts_min_duration_sec.",
    )
    p.add_argument(
        "--shorts-max-duration",
        type=float,
        default=None,
        help="Maximum short duration in seconds (default: 45; hard ceiling "
        "60). Overrides config.drive.shorts_max_duration_sec.",
    )
    p.add_argument(
        "--shorts-target-duration",
        type=float,
        default=None,
        help="Candidate short window width in seconds, clamped to "
        "[min, max] (default: 38). Overrides "
        "config.drive.shorts_target_duration_sec.",
    )
    p.add_argument(
        "--allow-cross-fs-source",
        action="store_true",
        default=False,
        help="Allow source dir on a different filesystem than the staging "
        "dir. Default refuses cross-filesystem source (WSL2 /mnt/d, NFS, "
        "etc.) because slow IO collapses weather measurement and triples "
        "encode wall-clock.",
    )
    p.add_argument(
        "--allow-degraded-weather",
        action="store_true",
        default=False,
        help="Permit Phase A to proceed when WeatherDetector returns "
        "degraded measurements (below min_success_ratio of chunks "
        "succeeded). Default raises WeatherMeasurementError instead of "
        "silently baking DEFAULT_WEATHER_CLASS into drive_scores.npz.",
    )
    p.add_argument(
        "--keep-intermediates",
        action="store_const",
        const=True,
        default=None,
        help="Retain per-day LF/short work files (segments, with_montage, "
        ".lf_XX_work, short_XX_raw). Default is to purge each long-form's "
        "work files once its final is burned, and the rest after each day's "
        "finals are emitted — set this for debug / re-render scenarios. "
        "Argparse-sentinel (default None): a saved opt-out survives a resume "
        "that doesn't re-pass the flag.",
    )
    p.add_argument(
        "--keep-masters",
        action="store_const",
        const=True,
        default=None,
        help="Retain `day_*_master.mp4` files (~59 GB each). Default is to "
        "drop them after each day's finals are emitted and recorded — Phase "
        "D and resume never need the master. Set this only for debug / "
        "forensic re-extraction of non-emitted shorts.",
    )
    p.add_argument(
        "--allow-short-count",
        action="store_const",
        const=True,
        default=None,
        help="Accept a shorts shortfall: suppress the SHORTS-CONTRACT "
        "hard-fail when a day plans/renders fewer shorts than the required "
        "target, and let the master-deletion gate count-match against the "
        "actual (reduced) count. Argparse-sentinel (default None): a saved "
        "opt-in survives a resume that doesn't re-pass the flag.",
    )
    p.add_argument(
        "--no-sharpen",
        action="store_true",
        help="Do not sharpen the LF day master (the measured unsharp amount is forced to 0).",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Fast run: force --no-sharpen and drop the opt-in --stabilize/--deflicker "
        "steps (a saved opt-in included); the preflight plan names each skipped step. "
        "Not persisted as an opt-in. Upscale honesty and every gate are unchanged.",
    )
    p.add_argument(
        "--stabilize",
        action="store_const",
        const=True,
        default=None,
        help="Stabilise the LF day master (vidstab two-pass, gated on measured luma "
        "jitter). Refused at start-up when this ffmpeg lacks libvidstab. Opt-in; "
        "argparse-sentinel (default None) so a resume that omits the flag keeps it.",
    )
    p.add_argument(
        "--deflicker",
        action="store_const",
        const=True,
        default=None,
        help="Deflicker the LF day master (gated on measured luma jitter). Opt-in; "
        "argparse-sentinel (default None) so a resume that omits the flag keeps it.",
    )
    p.add_argument(
        "--no-denoise",
        action="store_true",
        help="Do not denoise the LF day master (hqdn3d strength is forced to 0).",
    )
    p.add_argument(
        "--trail",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Render a Route Story trail film beside each LF day (the day's GPX, "
        "photos and first LF as the placed clip). Default (neither flag): on when "
        "the day has a GPX and a prepared imagery bundle, else a logged skip. "
        "--no-trail turns it off.",
    )
    p.add_argument(
        "--trail-profile",
        default="",
        metavar="YAML",
        help="Route Story launch profile (route_story_profiles/<trip>.yaml) "
        "supplying the trail film's imagery rasters and credits; the day's GPX, "
        "date, LF clip and output override its values.",
    )
    p.add_argument(
        "--allow-bad-shorts",
        action="store_const",
        const=True,
        default=None,
        dest="skip_shorts_quality_gate",
        help="SHORTS QUALITY-GATE ESCAPE HATCH: skip Phase D's 1080x1920 / "
        "9:16 shorts dimension validation (and the empty-shorts_per_lf hard "
        "fail). Diagnostic re-runs only — shipping 16:9 shorts is the "
        "regression that gate exists to catch. Sets state's "
        "skip_shorts_quality_gate. Argparse-sentinel (default None): a saved "
        "opt-in survives a resume that doesn't re-pass the flag.",
    )
    p.add_argument(
        "--no-ledger-reconcile",
        action="store_const",
        const=True,
        default=None,
        help="Skip the startup ledger↔YouTube reconciliation (drops "
        "phantom scheduled entries before slot decisions). Escape hatch for "
        "offline/quota-constrained runs. Argparse-sentinel.",
    )
    # Argparse sentinel pattern
    # for boolean CLI flags that need the "not supplied → state wins on
    # resume" semantics. ``action="store_const"`` with ``const=True``
    # and ``default=None`` gives us:
    #   flag absent on CLI  → args.flag is None  (state wins)
    #   flag present on CLI → args.flag is True  (CLI wins)
    # Downstream (``drive_orchestrator.run`` → state merge) treats None
    # as "not supplied" and True/False as "explicitly supplied", so a
    # saved True from a prior run survives the next invocation that
    # didn't pass the flag.
    p.add_argument(
        "--ost-audit-confirmed",
        action="store_const",
        const=True,
        default=None,
        help="OST AUDIT GATE: bypass the per-day OST audit gate. "
        "Use when running non-interactively (tmux, agent-launched) after "
        "reviewing the previous run's <output_base>/ost_audit_<date>.json "
        "out-of-band. Persists to state so a resume doesn't re-prompt.",
    )
    # Mirror of --ost-audit-confirmed
    # for the Gate C/D content review. Argparse-sentinel.
    p.add_argument(
        "--gate-cd-confirmed",
        action="store_const",
        const=True,
        default=None,
        help="GATE C/D: bypass the per-day content review prompt. "
        "Persists state with a content hash; resumes auto-pass only when "
        "the hash still matches. Mismatch (operator edited metadata) re-prompts.",
    )
    # Explicit opt-out for the
    # ``--publish`` implies ``--youtube-upload`` default.
    p.add_argument(
        "--no-youtube-upload",
        action="store_const",
        const=True,
        default=None,
        help="Opt out of YouTube upload even when --publish is set. "
        "Computes schedule + Metricool CSV without firing uploads.",
    )
    p.add_argument(
        "--force-sidecar-reset",
        action="store_const",
        const=True,
        default=None,
        help="OPERATOR RECONCILIATION: wipe "
        "<output_base>/youtube_uploads.json before the Phase D integrity "
        "check. Use after manually verifying via YouTube Studio that the "
        "referenced videos are gone (aborted run, deleted drafts). The "
        "pipeline never auto-wipes — this flag is the explicit opt-in. "
        "Uses the argparse-sentinel pattern: "
        "not supplying the flag means state wins on resume.",
    )
    p.add_argument(
        "--force-unlock",
        action="store_true",
        default=False,
        help="LOCK ESCAPE HATCH: remove an existing run lock file before "
        "acquire. Use only after verifying via 'pgrep -af \"gomoveshift "
        "drive\"' that no other run is active. The orchestrator already "
        "auto-reclaims locks held by dead PIDs; this flag is for the "
        "rare case where PID-probe lies (exotic container PID-namespace "
        "edge case). Operator-only; never set programmatically.",
    )
    p.add_argument(
        "--allow-seo-degraded",
        action="store_const",
        const=True,
        default=None,
        help="QUALITY-GATE ESCAPE HATCH: permit Phase D to ship LF metadata "
        "with seo_degraded=True (SEO manager unavailable / analytics-driven "
        "path failed). Default is to HARD FAIL at the end-of-Phase-D quality "
        "gate because a degraded title is a content-quality signal. Set this "
        "flag when the SEO service is known-down and the operator has "
        "verified titles are still acceptable. See "
        "gomoveshift/pipeline/quality_gates.py (Rule 2). Uses the "
        "argparse-sentinel pattern: not "
        "supplying the flag means state wins on resume.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        dest="dry_run",
        help="PREFLIGHT ONLY: render the execution plan (per-phase + "
        "per-day RUN/SKIP + wall-clock estimate) and exit 0 without "
        "running any phase or touching the filesystem. Use to sanity-"
        "check a resumed run's intent before committing hours of encoding. "
        "See gomoveshift/pipeline/preflight_plan.py.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="PREFLIGHT: skip the interactive 'Proceed? [y/N]' prompt that "
        "follows the execution plan. Required for tmux / non-interactive "
        "runs on a tty; unnecessary when stdin isn't a tty (the prompt is "
        "auto-skipped).",
    )
    # Journal automation hook — after Phase D upload success, merges the run's
    # YouTube uploads into the site journey. Default-ON: it runs whenever a
    # journey slug resolves (explicit --auto-publish-journal SLUG, else
    # config.drive.active_journey_by_region keyed by the run's region) AND a
    # site path resolves (--site-path or GMS_SITE_PATH). Use
    # --no-auto-publish-journal to force-disable. Failure is logged but does
    # NOT fail the drive batch (uploads are irreversible; the journal write is
    # recoverable via the standalone command / replay cron).
    p.add_argument(
        "--auto-publish-journal",
        default=None,
        help="Explicit journey SLUG to merge this run's uploads into "
        "<site>/content/journeys/<SLUG>.md. Overrides the per-region default "
        "(config.drive.active_journey_by_region). Omit to use that default.",
    )
    p.add_argument(
        "--no-auto-publish-journal",
        action="store_true",
        default=False,
        help="Force-disable the journal auto-publish for this run, even when a "
        "region default or --auto-publish-journal SLUG would otherwise apply.",
    )
    p.add_argument(
        "--site-path",
        type=Path,
        default=None,
        help="Path to the Hugo site repo root for journal auto-publish. "
        "Falls back to the GMS_SITE_PATH env var. Example: "
        "/path/to/site.",
    )
    p.add_argument(
        "--skip-stop-verify",
        action="store_true",
        default=False,
        dest="skip_stop_verify",
        help="DIAGNOSTIC ESCAPE: bypass the post-stop-removal verification "
        "gate (re-scans the cut output, hard-fails on a residual stop or a "
        "duration that doesn't reconcile with the kept segments). Runs "
        "UPSTREAM of the irreversible encode; never use in normal operation.",
    )
    p.add_argument(
        "--skip-blur",
        action="store_true",
        default=False,
        dest="skip_blur",
        help="DIAGNOSTIC ESCAPE: bypass the face/license-plate blur-detection "
        "stage (runs on the stops-cut file before the irreversible encode). "
        "When blur is enabled this lets identifiable faces/plates ship "
        "unredacted — never use for published footage.",
    )
    p.add_argument(
        "--insert-photos",
        metavar="DIR",
        default=None,
        help="INSERTS: show every photo in DIR as a still card inside the long-form it "
        "was taken during. Each photo is placed by its EXIF capture clock against the "
        "day's GPX, must pass the county gate (taken inside the LF's labelled counties) "
        "and the blur + EXIF/GPS-strip gates, and is listed in the preflight print and "
        "the OST audit sidecar. A photo the track cannot place is refused. "
        "See docs/pipeline/153-insert-engine.md.",
    )
    p.add_argument(
        "--insert-clip",
        metavar="PATH@MM:SS",
        action="append",
        default=None,
        type=_insert_clip_arg,
        help="INSERTS: cut the clip into the first long-form of the drive's first day "
        "at MM:SS of that long-form (repeatable). The clip must carry the .blurred "
        "marker when the run requires blur; it is fitted to the LF frame through the "
        "upscale policy and listed in the preflight print and the OST audit sidecar.",
    )
    # --sign-verify / --no-sign-verify / --sign-verify-strict share the
    # argparse-sentinel contract: ``default=None`` on ALL THREE is
    # load-bearing. None means "the operator did not decide this run", so a
    # resume that omits the flag keeps whatever the checkpoint saved; a
    # ``False`` default would silently re-stomp a saved --sign-verify on the
    # next resume. The flag registry owns the state write and the
    # config.drive mirror (see flag_registry.FLAG_REGISTRY).
    p.add_argument(
        "--sign-verify",
        action="store_true",
        default=None,
        dest="sign_verify",
        help="Enable road-sign verification: read the road signs in the raw "
        "footage and vote what they say against each LF's resolved location "
        "as INDEPENDENT evidence (signs are not derived from the GPS track). "
        "Advisory by default — a corroborated disagreement warns at the label "
        "gate and never changes a location. Costs an extra decode pass over "
        "the day's footage.",
    )
    p.add_argument(
        "--no-sign-verify",
        action="store_false",
        default=None,
        dest="sign_verify",
        help="Explicitly disable road-sign verification for this run, "
        "including on a resume whose checkpoint saved --sign-verify.",
    )
    p.add_argument(
        "--sign-verify-strict",
        action="store_true",
        default=None,
        dest="sign_verify_strict",
        help="Escalate a corroborated sign-vs-GPS state/county conflict to a "
        "pre-encode HARD FAIL instead of a warning. Opt-in: use once you "
        "trust the advisory output on your own footage.",
    )
    p.add_argument(
        "--skip-invariants",
        action="store_true",
        default=False,
        help="BREAK-GLASS ONLY: bypass the Layer-1 state-invariants gate "
        "in save_drive_state. A latent invariant bug can otherwise make "
        "every state save raise mid-run, killing an 8-hour encode with "
        "no recovery short of --fresh. Emits WARNING to the pipeline "
        "log on every save. Never use in normal operation.",
    )


def _add_upload_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "upload",
        help="Re-run Google Drive upload (and optionally CSV) for a completed project",
    )
    p.add_argument("project_dir", type=Path, help="Path to project output directory")
    p.add_argument(
        "--credentials",
        type=Path,
        default=None,
        help="Path to plaintext Drive OAuth credentials JSON (bypasses SOPS)",
    )
    p.add_argument(
        "--token",
        type=Path,
        default=None,
        help="Path to plaintext Drive OAuth token JSON (bypasses SOPS)",
    )
    p.add_argument(
        "--init-oauth",
        action="store_true",
        help="Run initial OAuth browser flow and encrypt token (one-time setup)",
    )
    p.add_argument("--skip-csv", action="store_true", help="Upload only -- skip CSV regeneration")


def _add_cleanup_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "cleanup",
        help="Clean up intermediate pipeline files (dry-run by default)",
    )
    p.add_argument(
        "project",
        nargs="?",
        default=None,
        help="Specific project name to clean (default: all projects)",
    )
    p.add_argument(
        "--confirm", action="store_true", help="Actually delete files (default is dry-run)"
    )
    p.add_argument(
        "--min-age",
        type=int,
        default=7,
        help="Only clean projects older than N days (default: 7)",
    )
    p.add_argument(
        "--archive-base",
        type=Path,
        default=None,
        help=(
            "Distinct archive directory holding a verified copy of each "
            "project's deliverable (e.g. /path/to/archive). When set, the "
            "WSL output deliverable becomes eligible for deletion once a "
            "size-matched copy is confirmed there. Unset by default -- the "
            "deliverable is never deleted."
        ),
    )


def _add_ingest_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "ingest",
        help="Scan GoPro directory, reassemble chaptered files, plan episodes",
    )
    p.add_argument("directory", type=Path, help="Directory with GoPro MP4 files")
    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for concatenated episodes (default: <directory>/_episodes)",
    )
    p.add_argument(
        "--confirm",
        action="store_true",
        help="Execute the plan (concat files, extract GPS). Default is dry-run preview.",
    )
    p.add_argument(
        "--min-clip",
        type=float,
        default=10,
        help="Minimum clip duration in seconds to include (default: 10)",
    )
    p.add_argument(
        "--max-episode",
        type=float,
        default=75,
        help="Maximum episode duration in minutes (default: 75)",
    )
    p.add_argument(
        "--max-gap",
        type=float,
        default=120,
        help="Maximum gap between recordings in minutes to combine (default: 120)",
    )
    p.add_argument(
        "--no-locations", action="store_true", help="Skip auto-deriving locations from GPS"
    )


def _add_batch_parser(subparsers: argparse._SubParsersAction) -> None:
    _default_output = gms_output_dir()
    p = subparsers.add_parser(
        "batch",
        help="Ingest GoPro directory and run full pipeline on each episode",
    )
    p.add_argument("directory", type=Path, help="Directory with GoPro MP4 files")
    p.add_argument("--publish", "-p", default="mon", help="Long-form publish day (default: mon)")
    p.add_argument(
        "--shorts-time",
        default="12:00",
        help="Shorts publish time HH:MM (default: 12:00 = noon)",
    )
    p.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="First publish date YYYY-MM-DD (default: next occurrence of --publish day)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output,
        help=f"Base output directory (default: {_default_output})",
    )
    p.add_argument(
        "--max-episode",
        type=float,
        default=75,
        help="Maximum episode duration in minutes (default: 75)",
    )
    p.add_argument("--skip-color", action="store_true", help="Skip color correction")
    p.add_argument("--skip-stops", action="store_true", help="Skip stop detection/removal")
    p.add_argument("--draft", action="store_true", help="Mark outputs as draft")
    p.add_argument(
        "--no-tmux",
        action="store_true",
        help="Skip tmux launcher (run directly in current shell)",
    )


def _add_find_stops_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("find-stops", help="Scan all videos in a directory for stops")
    p.add_argument("input_dir", type=Path, help="Directory containing videos")
    p.add_argument(
        "-o", "--output", type=Path, help="Output JSON file (default: stops_report.json)"
    )
    p.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore checkpoint and re-process all videos from scratch",
    )


def _add_enqueue_plan_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "enqueue-plan",
        help="Spool a reviewed publish plan into the delivery queue (dry-run by default)",
    )
    p.add_argument(
        "state_path",
        type=Path,
        help="Path to the completed drive run's pipeline_state.json. Gate C/D's "
        "confirmation and content hash are read from it; an unreviewed or "
        "changed-since-review plan is refused.",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Actually write deliveries to the queue. Without this, reports "
        "what would be enqueued and writes nothing.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory holding the run's artefacts and *_publish_plan.json "
        "sidecars (default: the state file's directory).",
    )
    p.add_argument(
        "--max-shorts",
        type=int,
        default=None,
        metavar="N",
        help="Spool only the N lowest-ranked shorts per sidecar (a "
        "scale-back). Applied in memory over the approved plan -- the "
        "sidecar is never rewritten, so Gate C/D's content hash keeps "
        "describing what a human reviewed. Editing the plan file to drop "
        "shorts instead flips that hash and is refused.",
    )
    p.add_argument(
        "--shadow",
        action="store_true",
        help="Spool the deliveries as a REHEARSAL that can never be "
        "published: no drain path, including the cron, can fire a shadow "
        "row. Use this to prove the chain end to end without leaving live "
        "ammunition in the queue.",
    )


def _add_enqueue_asset_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "enqueue-asset",
        help="Spool one reviewed standalone asset (route-story film, photo export, ...) "
        "into the delivery queue (dry-run by default)",
    )
    p.add_argument("video", type=Path, help="Path to the rendered media file to enqueue.")
    p.add_argument(
        "--platform",
        action="append",
        required=True,
        metavar="PLATFORM",
        help="A platform to publish to (repeatable): youtube, instagram, facebook, tiktok.",
    )
    p.add_argument(
        "--at", required=True, metavar="ISO_UTC", help="Scheduled publish time, UTC ISO-8601."
    )
    p.add_argument("--display-timezone", default="UTC", help="Timezone the schedule is shown in.")
    p.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="JSON file with title, description, tags[], recording_date "
        "(alternative to --title/--description/--tag).",
    )
    p.add_argument("--title", default=None, help="Post title (required unless --metadata).")
    p.add_argument("--description", default=None, help="Post description/caption body.")
    p.add_argument("--tag", action="append", default=None, help="A tag/keyword (repeatable).")
    p.add_argument(
        "--receipt",
        type=Path,
        default=None,
        help="A route-story film receipt JSON ('privacy' block used as this asset's "
        "attestation). Without it, every required class is recorded inconclusive.",
    )
    p.add_argument(
        "--bundle-dir",
        type=Path,
        default=None,
        help="Directory to write the sidecar + attestation into (default: next to the video).",
    )
    p.add_argument(
        "--confirm",
        default=None,
        metavar="HASH",
        help="The review hash printed by the dry run. Required with --apply; a mismatch refuses.",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Actually write the bundle and spool it. Without this, writes nothing.",
    )
    p.add_argument(
        "--shadow",
        action="store_true",
        help="Spool as a REHEARSAL that can never be published (same as enqueue-plan --shadow).",
    )


def _add_dispatch_queue_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "dispatch-queue",
        help="Drain the social publishing delivery queue (dry-run by default)",
    )
    # Dry-run by default, same convention as publish-journal: this command
    # calls live publishing APIs, so publishing must be opted into.
    p.add_argument(
        "--apply",
        action="store_true",
        help="Actually drain the queue and publish. Without this, reports "
        "what would be drained and calls no platform API.",
    )
    p.add_argument(
        "--sidecar-dir",
        type=Path,
        default=None,
        help="Directory holding <project>_publish_plan.json sidecars "
        "(default: seo.storage_path). The queue row carries only "
        "(project, short_index, platform), so the full record is read back "
        "from the sidecar that produced it.",
    )


def _add_publishing_status_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "publishing-status",
        help="Ground-truth status of the first-party social publishing stack: "
        "credential file presence (never contents) and delivery queue row "
        "counts by state",
    )
    p.add_argument(
        "--validate",
        action="store_true",
        help="Also cross-check what every non-terminal delivery DEPENDS on -- its "
        "publish-plan sidecar, its platform credential file, and any lease left "
        "behind by a worker that no longer exists. Read-only, dispatches nothing; "
        "exits 1 when anything is wrong. Run it on a restored backup before "
        "letting a drain near it",
    )


def _add_publish_journal_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "publish-journal",
        help="Merge a completed drive run's YouTube uploads into "
        "<site>/content/journeys/<slug>.md (dry-run by default)",
    )
    # state_path is positional but optional so --replay-pending can omit it.
    # The handler validates presence in non-replay mode.
    p.add_argument(
        "state_path",
        type=Path,
        nargs="?",
        default=None,
        help="Path to the completed drive run's pipeline_state.json "
        "(required unless --replay-pending)",
    )
    p.add_argument(
        "--journey",
        required=False,
        default=None,
        help="Journey slug (resolves to <site-path>/content/journeys/<slug>.md). "
        "Example: exampleland-2025-09. Required unless --replay-pending.",
    )
    p.add_argument(
        "--site-path",
        required=False,
        default=None,
        type=Path,
        help="Path to the Hugo site repo root (e.g. "
        "/path/to/site). Cross-project boundary; "
        "no env-var fallback so the invocation visibly names the target. "
        "Required unless --replay-pending (sidecar entries carry their own "
        "journey path).",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="Write changes to the journey file (and pending sidecar). "
        "Without this, prints the planned merge and exits 0 without "
        "touching anything.",
    )
    p.add_argument(
        "--allow-day-create",
        action="store_true",
        default=False,
        help="Stub a new days[] entry when a recording_date doesn't match any "
        "existing day. Default OFF — operator should author the day's narrative / "
        "heading / locations first so the journey isn't auto-stubbed with empty fields.",
    )
    # Replay-pending mode: flush any deferred entries whose publish_at has passed.
    p.add_argument(
        "--replay-pending",
        action="store_true",
        default=False,
        help="Replay mode: scan pending_journal.json sidecars and flush "
        "any entries whose publish_at has now passed. Use with --pending-file "
        "or --scan-root. Intended for cron.",
    )
    p.add_argument(
        "--pending-file",
        default=None,
        type=Path,
        help="Replay mode: explicit path to a single pending_journal.json sidecar.",
    )
    p.add_argument(
        "--scan-root",
        default=None,
        type=Path,
        help="Replay mode: recursively scan this directory for "
        "pending_journal.json files. Combine with --pending-file to mix sources.",
    )
    p.add_argument(
        "--auto-commit",
        action="store_true",
        default=False,
        help="Replay mode: after writing journey files, git add + commit them "
        "(one commit per repo). Refuses if the target repo has uncommitted "
        "changes to tracked files (skip-and-log, never auto-stash). Designed "
        "for cron.",
    )
    p.add_argument(
        "--auto-push",
        action="store_true",
        default=False,
        help="Replay mode: after --auto-commit, push to `origin`. Push failure "
        "exits non-zero (no retry, no rebase) so cron alerting fires. Requires "
        "--auto-commit.",
    )
    p.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Write logs to this file as well as stderr",
    )


def _add_schedule_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "schedule",
        help="Show what is scheduled, when the queue ends, and the next free "
        "long-form slot per region with its associated shorts (read-only)",
    )
    p.add_argument(
        "--ledger-path",
        type=Path,
        default=None,
        help="Content ledger JSON to read. Defaults to the shared ledger.",
    )
    p.add_argument("--json", action="store_true", help="Emit the report as JSON.")


def _add_sync_site_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "sync-site",
        help="Mirror the channel's real long-forms into the Hugo site's "
        "data/videos.yaml + data/channel_manifest.json (dry-run by default)",
    )
    p.add_argument(
        "--site-path",
        type=Path,
        required=True,
        help="Hugo site repo root (e.g. "
        "/path/to/site). Writes data/videos.yaml "
        "+ data/channel_manifest.json under it.",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="Write the files. Without it, prints what would change and exits 0.",
    )
    p.add_argument(
        "--journeys",
        action="store_true",
        default=False,
        help="Also reconcile routed public long-forms into content/journeys/*.md "
        "via data/video_routing.yaml (embeds any missing, deduped; held ids defer "
        "to the next sync). Honors --apply.",
    )
    p.add_argument(
        "--credentials-dir",
        type=Path,
        default=None,
        dest="credentials_dir",
        help="Override the brand-credentials directory used to build the "
        "read-only YouTube client (default: config seo.credentials_dir).",
    )


def _add_route_story_prepare_parser(subparsers: argparse._SubParsersAction) -> None:
    from ..pipeline.route_story.prepare import (
        DEFAULT_EXAGGERATION,
        DEFAULT_MARGIN_M,
        DEFAULT_OVERPASS_ENDPOINT,
        DEFAULT_TRIP_DATE_IMAGERY,
        EXAGGERATION_MAX,
        EXAGGERATION_MIN,
    )

    p = subparsers.add_parser(
        "route-story-prepare",
        help="Build and verify an offline Route Story map bundle from a GPX trip",
    )
    p.add_argument("gpx", type=Path, help="Path to the trip's GPX file")
    p.add_argument(
        "--recorded-on",
        type=date.fromisoformat,
        required=True,
        help="Calendar day (YYYY-MM-DD) to select from the GPX; other-date and "
        "duplicate tracks are excluded from the bundle",
    )
    p.add_argument(
        "--margin-m",
        type=float,
        default=DEFAULT_MARGIN_M,
        help="Metric margin added around the route's bounding box",
    )
    p.add_argument(
        "--overpass-endpoint",
        default=DEFAULT_OVERPASS_ENDPOINT,
        help="Overpass API endpoint for named-place labels (default: try the "
        "service's own mirror list in order)",
    )
    p.add_argument(
        "--exaggeration",
        type=float,
        default=DEFAULT_EXAGGERATION,
        help=f"Terrain exaggeration, clamped to [{EXAGGERATION_MIN}, {EXAGGERATION_MAX}]",
    )
    p.add_argument(
        "--trip-date-imagery",
        action="store_true",
        default=DEFAULT_TRIP_DATE_IMAGERY,
        help="Opt into the satellite picture from the trip date for the opening "
        "national-zoom shot, falling back to the cloud-free backdrop when it is "
        "cloudy. Default is the cloud-free backdrop.",
    )


def _add_route_story_proof_parser(subparsers: argparse._SubParsersAction) -> None:
    from ..pipeline.route_story.prepare import (
        DEFAULT_EXAGGERATION,
        DEFAULT_REGIONAL_MARGIN_KM,
        DEFAULT_TRIP_DATE_IMAGERY,
        EXAGGERATION_MAX,
        EXAGGERATION_MIN,
    )

    p = subparsers.add_parser(
        "route-story-proof",
        help="Render a terrain proof clip for a GPX trip from a prepared bundle",
    )
    p.add_argument("gpx", type=Path, help="Path to the trip's GPX file")
    p.add_argument("output", type=Path, help="Output .mp4 path")
    p.add_argument(
        "--recorded-on",
        type=date.fromisoformat,
        required=True,
        help="Calendar day (YYYY-MM-DD) the bundle was prepared for -- must match "
        "the route-story-prepare call that wrote it",
    )
    p.add_argument(
        "--margin-m",
        type=float,
        default=None,
        help="Metric margin used when the bundle was prepared (defaults to prepare's own default)",
    )
    p.add_argument(
        "--exaggeration",
        type=float,
        default=DEFAULT_EXAGGERATION,
        help=f"Terrain exaggeration used when the bundle was prepared, clamped to "
        f"[{EXAGGERATION_MIN}, {EXAGGERATION_MAX}]",
    )
    p.add_argument(
        "--regional-margin-km",
        type=float,
        default=DEFAULT_REGIONAL_MARGIN_KM,
        help="Regional band margin (km) used when the bundle was prepared",
    )
    p.add_argument("--fps", type=float, default=30.0, help="Output frame rate")
    p.add_argument("--width", type=int, default=1920, help="Output frame width")
    p.add_argument("--height", type=int, default=1080, help="Output frame height")
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel render workers (default: half the box's cores, capped at 6)",
    )
    p.add_argument(
        "--trip-date-imagery",
        action="store_true",
        default=DEFAULT_TRIP_DATE_IMAGERY,
        help="Must match the route-story-prepare call that wrote the bundle -- "
        "opts into the trip-date satellite scene for the opening national-zoom "
        "shot, falling back to the cloud-free backdrop when it is cloudy. "
        "Default is the cloud-free backdrop.",
    )


def _add_route_story_verify_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "route-story-verify",
        help="Re-check a Route Story receipt (proof or bundle manifest) offline",
    )
    p.add_argument("receipt", type=Path, help="Path to the receipt JSON to re-check")


def _add_route_story_film_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "route-story-film",
        help="Render a Route Story film from a tracked launch profile "
        "(route_story_profiles/<trip>.yaml) -- the console's in-process job door",
    )
    p.add_argument(
        "--profile",
        type=Path,
        required=True,
        help="Path to a route_story_profiles/*.yaml launch profile",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print the constructed film.py argv and exit 0 without rendering",
    )


def _add_reconcile_playlists_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "reconcile-playlists",
        help="YouTube playlist reconciler: desired membership from channel "
        "classification, inserts capped per run via config "
        "playlist_reconciler.max_inserts_per_run (dry-run by default; insert-only "
        "unless --prune)",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="Execute the planned playlistItems.insert calls (and, with --prune, "
        "the deletes). Without it, prints the plan summary and exits 0.",
    )
    p.add_argument(
        "--credentials-dir",
        type=Path,
        default=None,
        dest="credentials_dir",
        help="Override the brand-credentials directory used to build the "
        "YouTube client (default: config seo.credentials_dir).",
    )
    # Exclusive: --undo inserts, so it must never combine with the read-only --audit.
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--prune",
        action="store_true",
        help="Also plan removals from the managed playlists (Unsorted entries, deleted "
        "videos, playlists a routed video no longer belongs in); ledgered, dry-run "
        "unless --apply.",
    )
    p.add_argument("--max-deletes", type=int, default=150, metavar="N", help="Delete cap per run.")
    p.add_argument(
        "--include-nonpublic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="With --prune/--audit: remove (or keep) private/unlisted videos' entries; "
        "default: config playlist_reconciler.prune_include_nonpublic. Scheduled uploads "
        "(future publishAt) are never removed.",
    )
    p.add_argument(
        "--ledger",
        type=Path,
        default=None,
        metavar="PATH",
        help="With --prune --apply: ledger path (default: under the GMS output dir).",
    )
    mode.add_argument(
        "--undo",
        type=Path,
        metavar="LEDGER",
        help="Re-insert every entry a prune ledger recorded, at its position. Runs at once.",
    )
    mode.add_argument(
        "--audit",
        action="store_true",
        help="Read-only: print one 'RECONCILE_AUDIT <key>=<count>' line per check.",
    )


def _photo_count(raw: str) -> int:
    """argparse ``type=`` for ``photos generate --count``.

    A non-positive count used to reach the selector, whose empty selection
    was misread as pool exhaustion and wiped the posted-photos ledger —
    reject it before it can touch any state.
    """
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--count must be an integer (got {raw!r})") from exc
    if value < 1:
        raise argparse.ArgumentTypeError(f"--count must be >= 1 (got {value})")
    return value


def _photo_week(raw: str) -> str:
    """argparse ``type=`` for ``photos generate --week``.

    The value is used verbatim as a path component (captions YAML, resized
    dir, Metricool CSV name), so anything but a strict zero-padded
    YYYY-MM-DD — e.g. ``../../x`` — would write artifacts outside the photo
    output tree. Empty (the default) means "today", resolved downstream.
    """
    if raw == "":
        return raw
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        raise argparse.ArgumentTypeError(f"--week must be YYYY-MM-DD (got {raw!r})")
    try:
        datetime.strptime(raw, "%Y-%m-%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--week must be YYYY-MM-DD ({raw!r} is not a real calendar date)"
        ) from exc
    return raw


def _add_social_aspect(sub: argparse.ArgumentParser) -> None:
    """Shared by the two commands that resize: ``photos publish`` and ``photos batch``."""
    sub.add_argument(
        "--social-aspect",
        dest="social_aspect",
        choices=["", "square", "4:5", "9:16"],
        default=None,
        help="Centre-crop each resized upload copy to this aspect before the width "
        "scale (default: photos.social_aspect, '' = the source's own aspect, "
        "byte-identical). One output per photo, not one per platform.",
    )


#: ``photos develop`` white-balance flags beyond ``--white-balance-g``: suffix,
#: what it moves, span. Green's highlight half keeps green's own wide range.
_WB_AXIS_FLAGS = (
    ("r", "RED, shadows and midtones — the WARM end of the temperature axis", WB_OPERATOR_SPAN),
    ("b", "BLUE, shadows and midtones — its cool end; the pair IS the slider", WB_OPERATOR_SPAN),
    ("r-high", "RED, highlights only (the range above 0.75)", WB_OPERATOR_SPAN),
    ("b-high", "BLUE, highlights only (the range above 0.75)", WB_OPERATOR_SPAN),
    ("g-high", "GREEN, highlights only — the highlight half of --white-balance-g", 0.5),
)


def _add_photos_parser(subparsers: argparse._SubParsersAction) -> None:
    def _add_skip_blur(sub: argparse.ArgumentParser) -> None:
        """Operator opt-out from the default-on face blur (mirrors the drive flag)."""
        sub.add_argument(
            "--skip-blur",
            action="store_true",
            default=False,
            dest="skip_blur",
            help="DIAGNOSTIC ESCAPE: publish photos without blurring detected faces. "
            "Blur is ON by default and is applied to the resized copy; this ships "
            "identifiable faces — never use for public posts.",
        )

    def _add_no_faces(sub: argparse.ArgumentParser) -> None:
        """Arm the selection-time withhold gate (registered once — two copies of
        this help text is how it went stale after blur moved to the resizer)."""
        sub.add_argument(
            "--no-faces",
            action="store_true",
            dest="no_faces",
            help="Withhold photos the published copy can't cover: unreadable files "
            "always, plus photos with detected faces ONLY under --skip-blur. With "
            "blur on (the default) faces are blurred at resize, never skipped.",
        )

    def _add_reset_selection_state(sub: argparse.ArgumentParser) -> None:
        """Escape hatch for the fail-closed selection-state load."""
        sub.add_argument(
            "--reset-selection-state",
            action="store_true",
            dest="reset_selection_state",
            help="EXPLICIT ESCAPE: archive the selection-state ledger aside "
            "(never deleted) and start fresh. Wipes posted-photo dedup history, "
            "so the whole shipped back catalog becomes selectable again — only "
            "for a corrupt/incompatible state file the fail-closed load rejects.",
        )

    p = subparsers.add_parser("photos", help="Daily photo social media posting pipeline")
    photos_sub = p.add_subparsers(dest="photos_command", help="Photo commands")

    # photos generate
    gen = photos_sub.add_parser("generate", help="Select photos and generate captions (Phase 1)")
    gen.add_argument(
        "--week",
        type=_photo_week,
        default="",
        help="Week start date (strict YYYY-MM-DD, default: today)",
    )
    gen.add_argument(
        "--count",
        type=_photo_count,
        default=7,
        help="Number of photos to select (default: 7, must be >= 1)",
    )
    gen.add_argument(
        "--voice",
        type=str,
        default=None,
        choices=["first_person", "minimal_zen", "engaging", "interactive"],
        help="Override voice style (default: from ThemeRotator)",
    )
    _add_no_faces(gen)
    gen.add_argument(
        "--allow-photo-count",
        action="store_true",
        dest="allow_photo_count",
        help="Accept fewer photos than --count instead of hard-failing (exhausted-pool escape)",
    )
    gen.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate even if a captions YAML exists (the old file is renamed aside)",
    )
    _add_skip_blur(gen)
    _add_reset_selection_state(gen)

    # photos publish
    pub = photos_sub.add_parser("publish", help="Upload and schedule reviewed captions (Phase 2)")
    pub.add_argument("--captions", type=str, required=True, help="Path to reviewed captions YAML")
    pub.add_argument("--force", action="store_true", help="Skip review check")
    pub.add_argument(
        "--start",
        type=str,
        default=None,
        help="First post date (YYYY-MM-DD). Default: batch week start, clamped forward if past",
    )
    pub.add_argument(
        "--end-date",
        type=str,
        default=None,
        dest="end_date",
        help="Last post date (YYYY-MM-DD, inclusive) — slots never extend past it",
    )
    _add_skip_blur(pub)  # publish is where the resize (and so the blur) happens
    _add_social_aspect(pub)

    # photos ingest — camera card -> gallery JPEGs -> journey frontmatter. Not
    # the top-level ``ingest`` command, which imports GoPro video.
    ing = photos_sub.add_parser(
        "ingest", help="Import and develop photos from a camera card into a journey"
    )
    ing.add_argument("card_dir", help="Card / folder to import from (searched recursively)")
    ing.add_argument("--journey", required=True, help="Journey slug (or path to its *.md)")
    ing.add_argument("--day", required=True, help="Journey day (YYYY-MM-DD) to register under")
    # Optional so the card's own EXIF GPS can elect the label (voted over the
    # dominant county, like a drive). An operator label stays authoritative and
    # is never geocoded over; a card with neither fails loudly rather than
    # defaulting. Scoped to this subparser — ``pipeline``/``drive`` declare
    # their own ``--location`` and keep their existing requirements.
    ing.add_argument(
        "--location",
        default=None,
        help="Location label for the photo group. Optional: derived from the card's EXIF GPS "
        "when omitted. Required for a card with no GPS EXIF — never defaulted or invented",
    )
    ing.add_argument("--site-path", dest="site_path", help="Hugo site root (default: config)")
    # Not the shared _add_skip_blur: ingest blurs the gallery EXPORT (the file
    # the public site renders), not a resized upload copy, so the warning differs.
    ing.add_argument(
        "--skip-blur",
        action="store_true",
        default=False,
        dest="skip_blur",
        help="DIAGNOSTIC ESCAPE: export gallery JPEGs without blurring faces, plates "
        "and vehicle occupants. Blur is ON by default and runs before the export; "
        "the website renders these files — never use for public journeys.",
    )
    ing.add_argument(
        "--redevelop",
        action="store_true",
        default=False,
        help="Re-develop gallery JPEGs whose recorded provenance is the preview fallback",
    )
    ing.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        dest="dry_run",
        help="Print the ingest plan (skips, provenance, location); decode and write nothing",
    )
    # Creative look baked into the gallery JPEGs. Omitted → photos.ingest_style
    # (default "" = the neutral corrective base grade, byte-identical to before).
    ing.add_argument(
        "--style",
        default=None,
        help="Bake one style into the gallery masters (default: photos.ingest_style, "
        "'' = neutral base grade). Choices: natural, vivid, moody, golden, mono, soft",
    )
    ing.add_argument(
        "--style-strength",
        type=float,
        default=None,
        dest="style_strength",
        help="Style lever, only with --style: amplify the look's delta over the measured "
        "base (default: photos.style_strength, 1.0 = the calibrated bake; 2 = extreme)",
    )

    # photos develop — per-photo style variants into a LOCAL working dir (no
    # journey/day, no clustering, no blur — never the public site tree).
    dev = photos_sub.add_parser(
        "develop", help="Develop style variants of photos into a local working directory"
    )
    dev.add_argument("paths", nargs="+", help="Photo file(s) to develop")
    dev.add_argument(
        "--out", help="Output directory (default: photos.developed_output_dir; required when unset)"
    )
    dev.add_argument(
        "--styles",
        help="Comma-separated styles (default: photos.develop_styles, else auto-select per photo)",
    )
    dev.add_argument(
        "--look",
        default=None,
        metavar="NAME",
        help="Recall a named photos.looks entry: its knobs stand in as this run's "
        "defaults instead of retyping the flags. Precedence is explicit flag > look "
        "> photos.<knob> default, so one knob can be overridden on the command line "
        "without respelling the rest. An unknown name is refused, listing the "
        "configured looks.",
    )
    dev.add_argument(
        "--look-variant",
        action="append",
        metavar="NAME",
        help="Render this named look as its OWN variant panel, over the natural base "
        "grade (repeatable, like --lut). Unlike --look — a defaults layer, so exactly "
        "one look can apply per run — this is how two candidate looks are compared "
        "side by side on one contact sheet; the panel is named look_<NAME>. Both flags "
        "may be passed in one run. An unknown name is refused before any decode.",
    )
    dev.add_argument(
        "--look-strength",
        type=float,
        default=None,
        dest="look_strength",
        help="Look dial: blend every --look-variant panel toward the run's base "
        "render (1.0 = the whole look, the default; 0.5 = half of it; 0.0 = the base "
        "grade). Turns one recipe into a family instead of an all-or-nothing bake. "
        "A look may carry its own `look_strength:` and that wins. Range 0..1.",
    )
    dev.add_argument(
        "--style-strength",
        type=float,
        default=None,
        dest="style_strength",
        help="Style lever: amplify each style's delta over the measured base "
        "(default: photos.style_strength, 1.0 = the calibrated bake; 2 = extreme)",
    )
    dev.add_argument(
        "--lut",
        action="append",
        metavar="NAME|FILE.cube",
        help="Film-emulation .cube 3D LUT to render as an extra variant (repeatable). "
        "A photos.luts name or a direct path; the variant is named for the config key "
        "or the file stem. ADDS to the style variants (use --styles natural for "
        "LUT-only). A LUT is a fixed table — --style-strength does not apply to it.",
    )
    dev.add_argument(
        "--lut-strength",
        type=float,
        default=None,
        dest="lut_strength",
        help="LUT opacity: blend every --lut look toward the natural grade "
        "(default: photos.lut_strength, 1.0 = full-strength LUT; 0.0 = no LUT). "
        "One global value applied to every --lut look. Range 0..1.",
    )
    dev.add_argument(
        "--lut-over",
        default=None,
        dest="lut_over",
        metavar="STYLE",
        help="Grade with this STYLE before applying every --lut look, instead of "
        "the neutral base (default: photos.lut_over, '' = the base grade). Composes "
        "a film look over e.g. moody's matte; --style-strength scales the style "
        "under it, and the variant is named <lut>_over_<style>.",
    )
    dev.add_argument(
        "--lut-auto",
        type=int,
        default=None,
        dest="lut_auto",
        metavar="N",
        help="Pick the N best-suited LUTs per photo automatically: each candidate cube is "
        "characterised (warm shift, contrast gain, saturation gain) and scored against that "
        "photo's scene (default: photos.lut_auto, 0 = off = the manual --lut path). Candidates "
        "are the --lut specs when any are given, else every photos.luts entry. Deterministic.",
    )
    dev.add_argument(
        "--vignette",
        type=float,
        default=None,
        help="Radial-vignette depth: smoothly darken every variant's corners "
        "(default: photos.vignette, 0.0 = off; 1.0 = corners fully darkened). "
        "Resolution-independent — the same look at any frame size. Range 0..1.",
    )
    dev.add_argument(
        "--graduated-nd",
        type=float,
        default=None,
        help="Graduated ND: a straight darkening ramp across the frame — the smoked "
        "glass a landscape photographer slides into a holder to hold a bright sky down "
        "while the foreground keeps its exposure (0.0 = off, bit-identical; 1.0 = one "
        "stop at the smoked edge). Unlike --sky-darken it finds nothing: the ramp is "
        "pure geometry, so one value is one filter on every frame. Rides with the "
        "optics, under the vignette. Range 0..1.",
    )
    dev.add_argument(
        "--graduated-nd-angle",
        type=float,
        default=None,
        help="Rotate that grad in its holder, degrees off level (0.0 = level, so the "
        "smoked half covers the sky; +90 = upright, holding the RIGHT side; -90 = the "
        "left). Bounded at a quarter turn each way, so the ramp can never be turned on "
        "the ground. Inert while --graduated-nd is 0. Range -90..90.",
    )
    dev.add_argument(
        "--clarity",
        type=float,
        default=None,
        help="Clarity (midtone local contrast): a broad luma-only contrast move "
        "applied before sharpen (default: photos.clarity, 0.0 = off; 1.0 = full "
        "boost). NEGATIVE softens — the same pass subtracted, flattening texture "
        "(skin, portraits) without the added light a --glow bloom brings; -1.0 = "
        "full soften. Highlights/shadows are protected either way. Range -1..1.",
    )
    dev.add_argument(
        "--grain",
        type=float,
        default=None,
        help="Film grain: add deterministic monochrome (luminance) grain last, on "
        "top of the vignette (default: photos.grain, 0.0 = off; 1.0 = full grain). "
        "Midtone-weighted, seeded so two develops match byte-for-byte. The particle "
        "size scales with the frame, so one value is one look at any export size. "
        "Range 0..1.",
    )
    dev.add_argument(
        "--grain-size",
        type=float,
        default=None,
        help="Grain PARTICLE size: how COARSE each speck is, where --grain says how "
        "much of it (default: photos.grain_size, 1.0 = the shipped speck). Above 1.0 "
        "is a chunkier pushed-Tri-X clump, below it a finer 100-ISO one; the look "
        "still holds at any export size. Inert unless --grain is above 0. Range 0.25..4.",
    )
    dev.add_argument(
        "--halation",
        type=float,
        default=None,
        help="Halation: bleed a soft, RED-BIASED glow out of the frame's brightest "
        "tones — the film highlight bleed, screened over the vignette and under the "
        "grain (default: photos.halation, 0.0 = off; 1.0 = full glow). The seed luma "
        "is measured per frame and the radius scales with frame width. Range 0..1.",
    )
    dev.add_argument(
        "--halation-tint",
        nargs=3,
        type=float,
        metavar=("R", "G", "B"),
        default=None,
        help="Halation glow COLOUR: three (r,g,b) weights on that bleed. Unset keeps "
        "film's own red bias (1.0 0.55 0.3, bit-identical); a cyan or magenta tint "
        "buys the neon bloom red cannot express (default: photos.halation_tint). "
        "Inert unless --halation is above 0. Range 0..1.",
    )
    dev.add_argument(
        "--light-leak",
        type=float,
        default=None,
        help="Light leak: fog ONE FIXED corner with a warm glow — stray light that got "
        "past the camera back and exposed the emulsion, screened directly onto the "
        "halation bleed and under the fringe (default: photos.light_leak, 0.0 = off; "
        "1.0 = a full leak). Deterministic: one corner, one hue, one ramp shape, no "
        "randomness, and the ramp is sized as a fraction of the frame diagonal, so one "
        "value is one look at any export size. Range 0..1.",
    )
    dev.add_argument(
        "--chromatic-ab",
        type=float,
        default=None,
        help="Lateral chromatic aberration: fringe the corners the way a real lens "
        "does — red thrown radially OUT from the frame centre, blue pulled IN, green "
        "fixed (default: photos.chromatic_ab, 0.0 = off; 1.0 = a full toy-lens fringe). "
        "Zero on the optical axis, growing toward the corners, and sized as a fraction "
        "of the half-diagonal, so one value is one look at any export size. Rides over "
        "the halation and under the grain. Range 0..1.",
    )
    dev.add_argument(
        "--holo-sheen",
        type=float,
        default=None,
        help="Holographic sheen: an iridescent tint whose HUE rotates with TONE, so "
        "shadows, midtones and highlights each take a different cast and the print "
        "shimmers like foil (default: photos.holo_sheen, 0.0 = off; 1.0 = the full "
        "oil-slick). Keyed to the frame's OWN metered span, so it bites on every "
        "photograph and not only a full-range one, and deliberately low in amplitude — "
        "--duotone/--bw-mix recolour wholesale. Over the fringe, under the lines. 0..1.",
    )
    dev.add_argument(
        "--scanline",
        type=float,
        default=None,
        help="Scanlines: lay a CRT line structure over the frame — a row-periodic "
        "darkening on a SOFT sine profile, over the halation and under the grain "
        "(default: photos.scanline, 0.0 = off; 1.0 = deepest lines). Soft, not a hard "
        "on/off comb, which would alias into moire on any resize. The line SPACING "
        "scales with the export, so one value is one line density at any output size. "
        "Range 0..1.",
    )
    dev.add_argument(
        "--glow",
        type=float,
        default=None,
        help="Orton glow: screen a wide, soft copy of the frame back over the sharp "
        "original — the classic dreamy diffusion (default: photos.glow, 0.0 = off; "
        "1.0 = full bloom). Runs after the dehaze and BEFORE clarity/sharpen, so the "
        "detail is resolved back out of the bloom instead of smeared. The tone it "
        "starts at is measured per frame and the radius scales with frame width, so "
        "one value is one look at any export size. Range 0..1.",
    )
    dev.add_argument(
        "--glitch",
        type=float,
        default=None,
        help="Glitch: tear the frame into horizontal scan bands displaced sideways, each torn "
        "band splitting its channels — the digital-corruption look, over the scanlines and "
        "under the grain (default: photos.glitch, 0.0 = off; 1.0 = the widest tear). ONE knob: "
        "it drives how MANY bands tear and how far. Seeded, never random; band grid and tear "
        "are both frame-relative, so one value is one look at any export size. Range 0..1.",
    )
    dev.add_argument(
        "--dehaze",
        type=float,
        default=None,
        help="Dehaze: SUBTRACT the atmospheric veil via the dark-channel prior — the veil "
        "colour comes from the frame's own haziest pixels, then I = J*t + A*(1-t) is "
        "inverted (default: photos.dehaze, 0.0 = off; 1.0 = strong but not total). Runs "
        "first of the spatial stages, before clarity/sharpen/halation. Range 0..1.",
    )
    dev.add_argument(
        "--sky-darken",
        type=float,
        default=None,
        help="Sky darken: pull down the sky alone through a SOFT, feathered mask — the "
        "classic landscape move (default: photos.sky_darken, 0.0 = off; 1.0 = deep but "
        "never black). The mask mirrors the exposure meter's bright-upper-band rule, is "
        "measured against the frame's own luma, and is skipped on portrait frames (a "
        "bright top there is the subject). Rides under the vignette. Range 0..1.",
    )
    dev.add_argument(
        "--sky-saturate",
        type=float,
        default=None,
        help="Sky saturate: push the CHROMA of the sky alone, through the same soft mask "
        "--sky-darken rides — the polariser half of that move, where the darkener is the "
        "grad-filter half (flag/look only, no config default; 0.0 = off, 1.0 = rich but "
        "never a poster). A lerp away from each pixel's own luma, so the sky's TONE is "
        "left where the darkener put it. Runs right after it, under the vignette. 0..1.",
    )
    dev.add_argument(
        "--sharpen",
        type=float,
        default=None,
        help="Sharpen SCALE: multiply the automatically measured unsharp amount "
        "(default: photos.sharpen, 1.0 = the measured ramp untouched and "
        "bit-identical; 0.0 = off; 2.0 = double, clamped at the halo ceiling). "
        "The detail ramp still decides which frames sharpen at all. Range 0..2.",
    )
    dev.add_argument(
        "--vibrance",
        type=float,
        default=None,
        help="Vibrance: a low-saturation-weighted chroma boost — muted pixels take "
        "the full push, already-rich ones barely move (protects skies/foliage, finer "
        "than vivid's scene-level gate). Layers onto every styled variant (natural "
        "included), never the LUT looks (default: photos.vibrance, 0.0 = off, "
        "bit-identical). Range 0..2.",
    )
    dev.add_argument(
        "--white-balance-g",
        type=float,
        default=None,
        help="Green-magenta white balance: the OTHER half of white balance, the axis "
        "a fluorescent or forest-canopy cast lives on and that the amber-blue "
        "correction cannot reach (positive = greener, negative = magenta). A signed "
        "shift on the green channel alone, weighted by tone exactly as the measured "
        "r/b balance is. Layers onto every styled variant, never the LUT looks "
        "(0.0 = off, bit-identical). Range -0.5..0.5.",
    )
    # The five axes ``_apply_colorbalance`` renders that no flag reached: the
    # amber-blue TEMPERATURE pair in each tonal range, and green's highlight half.
    # One table, since they differ only in which channel and which range.
    for axis, what, span in _WB_AXIS_FLAGS:
        dev.add_argument(
            f"--white-balance-{axis}",
            type=float,
            default=None,
            help=f"White balance, {what}. Signed shift ADDED to the cast correction "
            "the grade measured, so the auto white balance survives an operator "
            "asking for warmth. Layers onto every styled variant, never the LUT "
            f"looks (0.0 = off, bit-identical). Range {-span:g}..{span:g}.",
        )
    dev.add_argument(
        "--white-balance-kelvin",
        type=float,
        default=None,
        help="White balance in signed degrees Kelvin (+ warmer), a friendlier handle "
        "on the amber-blue --white-balance-r/-b pair it converts onto. Added to the "
        "measured cast. Whole frame only (0.0 = off, bit-identical). Range -2000..2000.",
    )
    dev.add_argument(
        "--white-balance-tint",
        type=float,
        default=None,
        help="White-balance tint (+ green, - magenta), riding straight onto the "
        "--white-balance-g axis. Whole frame only (0.0 = off, bit-identical). "
        "Range -0.5..0.5.",
    )
    dev.add_argument(
        "--exposure-ev",
        type=float,
        default=None,
        help="Exposure in stops, the first stage of the grade (+1 doubles the light). "
        "Whole frame only (0.0 = off, bit-identical). Range -3..3.",
    )
    dev.add_argument(
        "--shadow-recovery",
        type=float,
        default=None,
        help="Shadow recovery: lifts shadows on top of the measured shadow lift. "
        "Whole frame only (0.0 = off, bit-identical). Range 0..20.",
    )
    dev.add_argument(
        "--highlight-recovery",
        type=float,
        default=None,
        help="Highlight recovery: pulls highlights on top of the measured highlight "
        "pull. Whole frame only (0.0 = off, bit-identical). Range 0..20.",
    )
    dev.add_argument(
        "--straighten",
        type=float,
        default=None,
        help="Rotate by this many degrees (+ counter-clockwise) to level a tilted "
        "horizon, before any crop. Run-level: every variant shares it, no look may "
        "set it (0.0 = off, bit-identical). Range -45..45.",
    )
    dev.add_argument(
        "--crop",
        nargs=4,
        type=float,
        default=None,
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        help="Crop to these fractions of the (straightened) frame, each in [0, 1] "
        "with left < right and top < bottom. Run-level, like --straighten "
        "(unset = full frame, bit-identical).",
    )
    dev.add_argument(
        "--lens-correct",
        action="store_true",
        default=None,
        dest="lens_correct",
        help="Correct distortion/vignetting/transverse-CA against the photo's EXIF "
        "camera+lens lensfun profile, before straighten/crop. Run-level, like "
        "--straighten (off by default; a clear no-op, never a guess, with the "
        "optional 'lens' extra absent, no camera EXIF, or no matching profile). "
        "Requires `pip install -e '.[lens]'`.",
    )
    dev.add_argument(
        "--bw-mix",
        nargs=3,
        type=float,
        default=None,
        metavar=("R", "G", "B"),
        help="B&W channel mixer: custom r,g,b luma weights — a red-weighted mix is "
        "the classic red filter (dark skies, bright red rock). NOT normalized (a sum "
        "away from 1 is an exposure move) but must sum to more than 0. Layers onto "
        "every styled variant, never the LUT looks (default: photos.mono_mix; unset "
        "= off, bit-identical). Range -2..3.",
    )
    dev.add_argument(
        "--duotone",
        nargs=6,
        type=float,
        default=None,
        metavar=("SR", "SG", "SB", "HR", "HG", "HB"),
        help="Duotone toning: the shadow (SR,SG,SB) and highlight (HR,HG,HB) ends of "
        "a luma ramp — sepia/selenium/cyanotype. Rides the --bw-mix luma when one is "
        "given, else BT.709 luma. Layers onto every styled variant, never the LUT "
        "looks (default: photos.duotone; unset = off, bit-identical). Range 0..1.",
    )
    dev.add_argument(
        "--duotone-strength",
        type=float,
        default=None,
        metavar="S",
        help="How far the --duotone ramp pulls the frame off plain monochrome: 1.0 "
        "the full two-colour ramp (the default — it REPLACES every pixel's colour), "
        "0.0 no tone at all, 0.5 halfway. Ignored without --duotone (default: "
        "photos.duotone_strength). Range 0..1.",
    )
    dev.add_argument(
        "--duotone-mid",
        nargs=3,
        type=float,
        default=None,
        metavar=("R", "G", "B"),
        help="Tritone: a third r,g,b anchor the --duotone ramp passes through at mid "
        "luma, so the midtones carry their own colour instead of the straight line "
        "between the two ends. Needs --duotone (default: photos.duotone_mid; unset = "
        "a straight two-ender, bit-identical). Range 0..1.",
    )
    dev.add_argument(
        "--layers",
        type=json.loads,
        default=None,
        metavar="JSON",
        help='Masked colour layers as JSON: \'[{"mask": "sky", "hue_sat": [1,1,1,1.3,1.3,1]}]\'. '
        "Each layer's colour chain rides ONLY where its named mask "
        f"({_LAYER_MASK_NAMES} — optional feather/invert) is non-zero, "
        "composing over the whole-frame grade in list order. Normally authored once as a "
        "photos.looks entry's `layers:` and recalled with --look; unset = off, bit-identical.",
    )
    dev.add_argument(
        "--reference",
        default=None,
        metavar="PATH",
        help="Reference-image colour match: grade every styled variant to THIS "
        "frame's colour — a Reinhard transfer in CIE Lab that shifts and scales "
        "each L/a/b channel until the variant's mean and spread equal the "
        "reference's. Unlike a saved look it recalls a FRAME, not parameters. The "
        "file is decoded and measured once, at this door, so a bad path fails "
        "before any photo is developed. Never the LUT looks (default: "
        "photos.reference; unset = off, bit-identical).",
    )
    dev.add_argument(
        "--hue-shift",
        nargs="+",
        type=float,
        default=None,
        metavar="BAND",
        help="Per-hue hue rotation in DEGREES over the anchor bands in order "
        "red/yellow/green/cyan/blue/magenta (0/60/120/180/240/300°), or SEVEN values "
        "to put orange (30°) second — skin and golden hour; 0 = neutral. "
        "Layers onto every styled variant (natural included), never the LUT looks "
        "(default: photos.hue_shift; unset = off, bit-identical). Range -180..180.",
    )
    dev.add_argument(
        "--hue-sat",
        nargs="+",
        type=float,
        default=None,
        metavar="BAND",
        help="Per-hue saturation MULTIPLIER over the anchor bands in order "
        "red/yellow/green/cyan/blue/magenta (0/60/120/180/240/300°), or SEVEN values "
        "to put orange (30°) second — skin and golden hour; 1.0 = neutral. "
        "Layers onto every styled variant (natural included), never the LUT looks "
        "(default: photos.hue_sat; unset = off, bit-identical). Range 0..5.",
    )
    dev.add_argument(
        "--hue-lum",
        nargs="+",
        type=float,
        default=None,
        metavar="BAND",
        help="Per-hue luminance MULTIPLIER over the anchor bands in order "
        "red/yellow/green/cyan/blue/magenta (0/60/120/180/240/300°), or SEVEN values "
        "to put orange (30°) second — skin and golden hour; 1.0 = neutral. "
        "Layers onto every styled variant (natural included), never the LUT looks "
        "(default: photos.hue_lum; unset = off, bit-identical). Range 0..5.",
    )
    dev.add_argument(
        "--curve-l",
        nargs="+",
        type=float,
        default=None,
        metavar="X Y",
        help="MASTER tone curve: a FLAT list of alternating x y control points "
        "(>= 2 points, x ascending in 0..1), applied to R, G and B alike — tone "
        "shaping with no colour cast, by construction — UNDER the cross-process "
        "channel curves. Layers onto every styled variant (natural included), never "
        "the LUT looks (default: photos.curve_l; unset = off, bit-identical).",
    )
    for _ch, _name in (("r", "RED"), ("g", "GREEN"), ("b", "BLUE")):
        dev.add_argument(
            f"--curve-{_ch}",
            nargs="+",
            type=float,
            default=None,
            metavar="X Y",
            help=f"Cross-process {_name} tone curve: a FLAT list of alternating x y "
            "control points (>= 2 points, x ascending in 0..1), applied as an ADDITIONAL "
            f"curve on the {_name.lower()} channel AFTER the master curve. Layers onto every "
            "styled variant (natural included), never the LUT looks "
            f"(default: photos.curve_{_ch}; unset = off, bit-identical).",
        )
    dev.add_argument(
        "--split-tone-shadow",
        nargs=3,
        type=float,
        default=None,
        metavar=("R", "G", "B"),
        help="Split-tone shadows: signed (r,g,b) tint added to the shadows, fading "
        "out by the midtones (default: photos.splittone_shadow; unset = off, "
        "bit-identical). Layers onto every styled variant, never the LUT looks. Range -0.5..0.5.",
    )
    dev.add_argument(
        "--split-tone-midtone",
        nargs=3,
        type=float,
        default=None,
        metavar=("R", "G", "B"),
        help="Split-tone midtones: signed (r,g,b) tint added to the midtones, fading "
        "out at black and at white — the tonal range this chain's photos actually sit "
        "in (default: photos.splittone_midtone; unset = off, bit-identical). Layers "
        "onto every styled variant, never the LUT looks. Range -0.5..0.5.",
    )
    dev.add_argument(
        "--split-tone-highlight",
        nargs=3,
        type=float,
        default=None,
        metavar=("R", "G", "B"),
        help="Split-tone highlights: signed (r,g,b) tint added to the highlights, "
        "fading out by the midtones (default: photos.splittone_highlight; unset = "
        "off, bit-identical). Layers onto every styled variant, never the LUT looks. Range -0.5..0.5.",
    )
    dev.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Print SceneProfile, chosen styles, per-style params; decode to meter, write nothing",
    )
    dev.add_argument(
        "--contact-sheet",
        action="store_true",
        dest="contact_sheet",
        help="Also write <stem>_contact.jpg: one row of labeled panels, one per variant, "
        "for side-by-side style tuning",
    )
    dev.add_argument(
        "--contact-cols",
        type=int,
        default=None,
        dest="contact_cols",
        help="Wrap the contact sheet into rows of N panels (default: one unbroken row)",
    )
    dev.add_argument(
        "--border",
        choices=["none", "white", "black", "filmstrip"],
        default=None,
        help="Presentational border/frame drawn around each exported variant at the "
        "export step (default: photos.border, 'none' = off, byte-identical). "
        "'white'/'black' = even matte; 'filmstrip' = black frame with sprocket holes. "
        "Never applied to the contact-sheet panels.",
    )
    dev.add_argument(
        "--format",
        choices=["jpeg", "tiff16"],
        default=None,
        help="Master container per variant (default: photos.develop_format; 'jpeg' = the "
        "8-bit sRGB master, byte-identical). 'tiff16' renders the WHOLE chain at 16 bits into "
        "<stem>_<variant>.tif: a 48-bit lossless sRGB-tagged archival master that re-edits in "
        "Lightroom/Capture One without generation loss.",
    )
    dev.add_argument(
        "--grade-sidecar",
        action="store_true",
        dest="grade_sidecar",
        help="Also write <stem>_<variant>.grade.json beside each exported variant: the exact "
        "CorrectionParams, operator knobs, LUT surface and decode provenance that produced it, "
        "so the frame can be reproduced (default: photos.grade_sidecar; off writes nothing and "
        "leaves the export byte-identical)",
    )
    dev.add_argument(
        "--from-grade",
        default=None,
        dest="from_grade",
        metavar="PATH",
        help="Replay the grade recorded in a <stem>_<variant>.grade.json instead of measuring a "
        "new one. The WHOLE record is restored — the exact CorrectionParams AND the spatial and "
        "colour knobs the frame was approved with — so the export reproduces that frame; exactly "
        "ONE variant renders, under its recorded name. Any knob flag passed alongside BEATS the "
        "file (as does a --look recipe, which fills the same flags), so an approved JPEG proof "
        "re-renders as a 16-bit master with --format tiff16, and naming this photo's siblings "
        "applies its grade to them. A file that is not a sidecar this build's develop algorithm "
        "wrote is refused before any decode — a replay must not hand back different pixels "
        "than the proof that was approved.",
    )
    dev.add_argument(
        "--hdr",
        action="store_true",
        help="Fuse each exposure bracket among the given photos into ONE frame before the "
        "grade, recovering range no single frame holds (default: photos.develop_hdr; off "
        "develops every photo singly, byte-identical). Brackets are grouped from EXIF (ISO "
        "held, shutter stepping one way, close in time) and their members are CONSUMED: "
        "they export once as <first-stem>_hdr_<variant>, never also one by one.",
    )

    # photos reel — stills -> one vertical Ken Burns MP4. Its own command, not a
    # develop flag: it renders video, so no existing rendered pixel can change.
    reel = photos_sub.add_parser(
        "reel", help="Render stills as one vertical Ken Burns pan-zoom MP4 (1080x1920)"
    )
    reel.add_argument("paths", nargs="+", help="Stills, in the order they appear in the reel")
    reel.add_argument("--out", required=True, help="Output .mp4 path (its directory must exist)")
    reel.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Seconds each still holds the screen (default: 3.0). One pan-zoom "
        "spans exactly this; stills are cut together, never crossfaded",
    )

    # photos daylight-timelapse — the daytime sibling of `astro --timelapse`:
    # stills in capture order -> one MP4, no dark/hot-pixel calibration, no
    # shared-tone develop.
    dtl = photos_sub.add_parser(
        "daylight-timelapse", help="Daylight stills -> one MP4, one video frame per still"
    )
    dtl.add_argument("paths", nargs="+", help="Stills, ordered by capture order")
    dtl.add_argument("--out", required=True, help="Output .mp4 path (its directory must exist)")
    dtl.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Video frame rate (default: pipeline.timelapse.TIMELAPSE_FPS, 24)",
    )
    dtl.add_argument(
        "--height",
        type=int,
        default=None,
        help="Cap the output height in pixels, aspect preserved (default: uncapped)",
    )

    # photos astro — night RAW frames -> stacked masters, local files in/out, no
    # publish; `ingest` applies the usual blur/GPS/location gates afterward.
    astro = photos_sub.add_parser(
        "astro", help="Stack a night's RAW frames (nightscape/Milky Way) into masters"
    )
    astro.add_argument(
        "paths",
        nargs="*",
        help="Light frame RAW file(s), one night's shoot. Not needed with --from-master",
    )
    astro.add_argument(
        "--darks",
        nargs="+",
        default=(),
        help="Dark (cap-on) calibration RAW file(s), same body/settings as the lights. "
        "Optional, but omitting darks means NO hot-pixel correction happens: there is "
        "no in-frame cosmetic pass to fall back to.",
    )
    astro.add_argument(
        "--output-dir", dest="output_dir", required=True, help="Directory for the stacked masters"
    )
    astro.add_argument(
        "--stack-len",
        type=int,
        default=None,
        dest="stack_len",
        help="Frames per stack chunk (default: astro_sequence.STACK_LEN_DEFAULT, 20). NOT one "
        "size for every camera body -- RAM scales with stack-len x sensor megapixels, so a "
        "high-megapixel body needs a smaller --stack-len to clear the RAM gate at "
        "--output-dir.",
    )
    astro.add_argument(
        "--keep-linear",
        action="store_true",
        dest="keep_linear",
        help="Keep each chunk's reusable scene-linear artifact beside the master. At "
        "5600x3728 this is about 543 MB/chunk (sky numerator + contributor count + "
        "unwarped ground); the run preflight reserves all artifacts plus one atomic-write "
        "temporary before decoding.",
    )
    astro.add_argument(
        "--linear-only",
        action="store_true",
        dest="linear_only",
        help="Write complete validated linear artifacts but skip per-chunk develop/TIFF work. "
        "Implies --keep-linear; intended for one-time RAW backfills and later recombination.",
    )
    astro.add_argument(
        "--horizon",
        type=float,
        default=None,
        help="Sky/ground boundary as a fraction of frame height from the top; applies to both "
        "triage (background/star measurement) and the final blend (default: auto, from the "
        "reference frame's own detected star-density map)",
    )
    astro.add_argument(
        "--triage-only",
        action="store_true",
        dest="triage_only",
        help="Measure and report frame quality (usable/cloud/dew counts) per chunk; stack nothing",
    )
    astro.add_argument(
        "--max-frame-gap",
        type=float,
        default=None,
        dest="max_frame_gap",
        help="Seconds between neighbouring frames still read as one run (default: "
        "astro_sequence.MAX_FRAME_GAP_S, 60) -- raise for a sparse-cadence night",
    )
    astro.add_argument(
        "--max-chunk-span",
        type=float,
        default=None,
        dest="max_chunk_span",
        help="Seconds a stack chunk's frames may span end to end (default: "
        "astro_sequence.MAX_CHUNK_SPAN_S, 330) -- must be >= --max-frame-gap",
    )
    astro.add_argument(
        "--min-sequence-frames",
        type=int,
        default=None,
        dest="min_sequence_frames",
        help="Fewest frames detect_sequences needs to keep a run (default: "
        "astro_sequence.MIN_SEQUENCE_FRAMES, 5) -- chunk_sequence's own trailing-chunk "
        "floor stays fixed at 5 regardless",
    )
    # One night, three renders -- a stack (default), arcs, motion -- never two at once.
    render_mode = astro.add_mutually_exclusive_group()
    render_mode.add_argument(
        "--trails",
        action="store_true",
        help="Star-trail mode: the sky's rotation as ARCS -- frames are never registered "
        "(registration removes trailing) or averaged, they fold into a running per-pixel "
        "MAXIMUM, so --stack-len is ignored and one run renders as one master. Arcs are "
        "DASHED by the capture's own gap (interval - exposure: the operator's measured 15 s "
        "exposure on a realised 32 s interval leaves 17 s of each cycle unrecorded), and "
        "nothing interpolates that away. --darks matters most here: a maximum makes every "
        "uncorrected hot pixel permanent",
    )
    render_mode.add_argument(
        "--timelapse",
        action="store_true",
        help="Timelapse mode: the sky MOVING, one video frame per captured frame, as an MP4. "
        "Never registered or averaged, so --stack-len is ignored and one run is one clip. The "
        "sequence's MIDDLE frame sets the tone for all of it (per-frame auto-stretch is what "
        "makes a night timelapse flicker), and 90 frames at 24 fps is 3.75 s of video -- "
        "reported before the render, never padded out",
    )
    astro.add_argument(
        "--timelapse-fps",
        type=int,
        default=None,
        dest="timelapse_fps",
        metavar="N",
        help="Video frame rate for --timelapse (default: the module's 24). The capture fixes "
        "how many frames exist, so this is the only lever over how long the clip runs: a "
        "15-frame sequence is 0.6s at 24 and 1.9s at 8. Lower trades smoothness for a clip "
        "long enough to watch; no frame is ever duplicated or interpolated to pad it out. "
        "Shooting more frames is the real fix",
    )
    astro.add_argument(
        "--timelapse-height",
        type=int,
        default=None,
        dest="timelapse_height",
        metavar="N",
        help="Cap the --timelapse output height in pixels, aspect preserved (default: 2160, "
        "4K). This is a rendering COST lever, not just a file-size one: each frame's develop "
        "fits a gradient polynomial over every pixel, so a full ~20 MP frame costs several "
        "times what a 2160 frame does -- hours instead of minutes for one night, for a "
        "video no display can show. 1080 is the other useful value. Pass 0 for "
        "the full sensor, uncapped. A request taller than the frame is ignored "
        "(quality.py HONEST_RESOLUTION_RULE: never claim a native resolution the source lacked)",
    )
    astro.add_argument(
        "--decode-workers",
        type=int,
        default=None,
        dest="decode_workers",
        metavar="N",
        help="Decode and triage N frames at once (default: the module's 4; 1 is serial). "
        "rawpy's unpack/postprocess and the HE-NEF DNG converter both release the GIL, so "
        "these are threads sharing one address space -- no frame is copied between "
        "processes, so a few workers decode several times faster than serial. "
        "Each worker past the first holds another decoded frame plus its demosaic buffers "
        "live (a few tenths of a GB at ~20 MP), and the preflight is told the count, so a "
        "high N is refused up front rather than met as an OOM mid-run",
    )
    astro.add_argument(
        "--detect-meteors",
        action="store_true",
        help=(
            "Scan every frame for meteor/satellite streaks. OFF by default: the scan is "
            "~92%% of triage cost (3.52s of 3.83s on a 2943-star frame) and this command "
            "passes no meteor-reinsertion policy, so its only consumer is the sidecar. "
            "With it off, sidecar streaks read null (not []) -- not searched, not "
            "searched-and-empty."
        ),
    )
    astro.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow a chunk's stacked master to replace an existing file at its output "
        "name (first + last source frame identity + a --horizon/--darks fingerprint). "
        "Refused by default -- an accidental re-run over the same frames would otherwise "
        "silently replace an already-stacked (~1 min/frame) master.",
    )
    astro.add_argument(
        "--contact-sheet",
        action="store_true",
        dest="contact_sheet",
        help="This run's ONE stack through several develop knobs (background WB, "
        "stretch target) as one labeled grid, never re-stacking per panel -- refuses "
        "alongside --trails/--timelapse/--triage-only. Under photos.contact_sheet_root.",
    )
    astro.add_argument(
        "--from-master",
        default=None,
        dest="from_master",
        metavar="PATH",
        help="Render --contact-sheet from an existing <name>.tif + <name>.stack.json "
        "pair, seconds not ~1 min/frame -- no light frame paths needed.",
    )

    night = photos_sub.add_parser(
        "night", help="Private resumable previews from saved linear stacks"
    )
    night.add_argument("--workspace", required=True)
    night.add_argument("--brand", required=True)
    night.add_argument(
        "--storage-budget-gib",
        help="Optional positive project output budget in GiB (new requests only; omitted is uncapped)",
    )
    source = night.add_mutually_exclusive_group(required=True)
    source.add_argument("--project")
    source.add_argument("--sources", nargs="+")
    source.add_argument(
        "--scan",
        metavar="SOURCE_DIRECTORY",
        help="Read-only RAW/header and existing-linear inventory; creates no project",
    )
    source.add_argument(
        "--collection",
        metavar="SOURCE_DIRECTORY",
        help="Rescan a collection and create a typed product request",
    )
    night.add_argument(
        "--products",
        nargs="+",
        choices=["best_stills", "timelapse", "star_trails", "deep_still"],
        help="Products required from --collection; every requested type must complete",
    )
    night.add_argument(
        "--sequence-index",
        type=int,
        default=0,
        help="Zero-based compatible sequence used by timelapse/trails (default: 0)",
    )
    night.add_argument("--source-root", action="append", default=[])
    night.add_argument("--recipe", choices=["natural", "accent", "bold"], default="natural")
    control = night.add_mutually_exclusive_group()
    control.add_argument("--pause", action="store_true")
    control.add_argument("--cancel", action="store_true")
    control.add_argument("--resume", action="store_true")
    control.add_argument("--status", action="store_true")
    night.add_argument("--wait-seconds", type=float, default=600)
    night.add_argument("--poll-seconds", type=float, default=15)

    # photos astro-inspect — READ-ONLY pixel-peep of an existing stacked master:
    # 1:1 corner/centre crops + a per-region star FWHM/ellipticity summary, so
    # judging corner star roundness doesn't mean opening the TIFF by hand. Its
    # own parser block, never touching `astro`'s options above.
    astro_inspect = photos_sub.add_parser(
        "astro-inspect",
        help="Contact sheet of 1:1 corner/centre crops + star FWHM/ellipticity "
        "from a stacked master (read-only, never re-develops it)",
    )
    astro_inspect.add_argument("master", help="Stacked master .tif to inspect")
    astro_inspect.add_argument("--out", required=True, help="Output contact-sheet .jpg path")

    astro_combine = photos_sub.add_parser(
        "astro-combine",
        help="Combine reusable scene-linear astro artifacts into one developed master",
    )
    astro_combine.add_argument(
        "paths", nargs="+", help="Ordered .linear artifact directories (at least two)"
    )
    astro_combine.add_argument("--out", required=True, help="New output .tif/.tiff path")
    astro_combine.add_argument(
        "--reference",
        default=None,
        help="Artifact to use as the global reference (default: quality-qualified midpoint)",
    )
    astro_combine.add_argument(
        "--scope",
        choices=_ASTRO_COMBINE_SCOPES,
        default=_ASTRO_COMBINE_DEFAULT_SCOPE,
        help="Shipped region of the deep product: 'auto' (default) measures corner star "
        "quality on the blended full frame before any crop and ships the full sensor when it "
        "measurably passes, else the scoped fallback; 'full' always ships the whole sensor; "
        "'scoped' always crops to the central-two-thirds fallback rectangle",
    )

    # photos release — operator recovery: return an unpublished batch's photo
    # holds to the selectable pool. A published batch's photos are spent, not
    # held, and stay refused.
    rel = photos_sub.add_parser(
        "release", help="Release an unpublished batch's photo holds back to the pool"
    )
    rel.add_argument("week", type=str, help="Week start date of the batch (YYYY-MM-DD)")

    # photos status
    photos_sub.add_parser("status", help="Show photo posting status")

    # photos cleanup
    cu = photos_sub.add_parser(
        "cleanup", help="Clean up photo pipeline artifacts (dry-run by default)"
    )
    cu.add_argument(
        "--age",
        type=int,
        default=14,
        help="Only clean artifacts older than N days (default: 14)",
    )
    cu.add_argument(
        "--confirm", action="store_true", help="Actually delete files (default is dry-run)"
    )

    # photos batch
    ba = photos_sub.add_parser(
        "batch", help="One-shot: generate + publish photo posts (no review step)"
    )
    ba.add_argument("--start", type=str, required=True, help="First post date (YYYY-MM-DD)")
    ba.add_argument("--end", type=str, required=True, help="Last post date (YYYY-MM-DD, inclusive)")
    ba.add_argument(
        "--output",
        type=str,
        default=None,
        help="Copy final CSV to this directory (e.g., /path/to/export/)",
    )
    ba.add_argument(
        "--voice",
        type=str,
        default=None,
        choices=["first_person", "minimal_zen", "engaging", "interactive"],
        help="Override voice style (default: from ThemeRotator)",
    )
    _add_no_faces(ba)
    _add_skip_blur(ba)
    _add_social_aspect(ba)
    _add_reset_selection_state(ba)


def _add_comments_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register ``comments`` + its ``pull`` subcommand."""
    p = subparsers.add_parser(
        "comments",
        help="Pull viewer comments into the shared comments.jsonl ledger",
    )
    comments_sub = p.add_subparsers(dest="comments_command", help="Comment commands")

    pull = comments_sub.add_parser(
        "pull", help="Pull YouTube comments via commentThreads.list (read-only)"
    )
    pull.add_argument(
        "--channel-id",
        default=None,
        dest="channel_id",
        help="YouTube channel ID (allThreadsRelatedToChannelId). No config default.",
    )
    pull.add_argument(
        "--since",
        default="90d",
        help="Only pull comments newer than this (e.g. '90d'). Default: 90d.",
    )
    pull.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Override the output .jsonl path (default: GMS_COMMENTS_PATH, else comments.jsonl in the data home).",
    )
    pull.add_argument(
        "--credentials-dir",
        type=Path,
        default=None,
        dest="credentials_dir",
        help="Override the brand-credentials directory used to build the YouTube client.",
    )

    triage = comments_sub.add_parser(
        "triage", help="Build a markdown triage report from the comments.jsonl ledger"
    )
    triage.add_argument(
        "--in",
        type=Path,
        default=None,
        dest="in_path",
        help="Override the input .jsonl path (default: GMS_COMMENTS_PATH, else comments.jsonl in the data home).",
    )
    triage.add_argument(
        "--out",
        type=Path,
        default=None,
        dest="out_path",
        help="Override the output .md report path (default: comments-triage.md in the data home).",
    )

    import_ = comments_sub.add_parser(
        "import",
        help="Import comments from a CSV/txt export (no API — e.g. TikTok)",
    )
    import_.add_argument("file", type=Path, help="Path to the CSV or .txt export")
    import_.add_argument(
        "--platform",
        required=True,
        choices=["tiktok"],
        help="Source platform (only 'tiktok' is supported).",
    )
    import_.add_argument(
        "--video-id",
        dest="video_id",
        default=None,
        help="Video id (required for a .txt export; a CSV's own video_id/url column wins).",
    )
    import_.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Override the output .jsonl path (default: GMS_COMMENTS_PATH, else comments.jsonl in the data home).",
    )


def _add_metrics_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register ``metrics`` + its ``collect`` subcommand (idiom mirrors ``photos``)."""
    p = subparsers.add_parser(
        "metrics",
        help="Collect post-publish social performance metrics into the tracking store",
    )
    metrics_sub = p.add_subparsers(dest="metrics_command", help="Metrics commands")

    collect = metrics_sub.add_parser(
        "collect",
        help="Fetch + persist metrics for every registered short in a project",
    )
    collect.add_argument("project", help="Project name (matches the tracking record dir)")
    collect.add_argument(
        "--response-mode",
        choices=("shadow", "live"),
        default="shadow",
        dest="response_mode",
        help=(
            "Bounded-response mode for this run's suppression classification: "
            "'shadow' (default) only logs/records what a cap would limit; "
            "'live' enforces the cap and can auto-pause a platform"
        ),
    )

    metrics_sub.add_parser(
        "backfill-ids",
        help="One-shot: register tracking ids for every already-published queue row",
    )

    metrics_sub.add_parser(
        "analytics-probe",
        help="Probe YouTube Analytics access: ANALYTICS_OK (exit 0) or a reason line (exit 2)",
    )

    metrics_sub.add_parser(
        "enrol-channel",
        help="Register every channel upload the tracking store lacks; prints EVIDENCE_COVERAGE",
    )

    metrics_sub.add_parser(
        "export-evidence",
        help="Print the settled day-7 EvidenceBundle JSON (no thresholds, no recommendation)",
    )

    best_times = metrics_sub.add_parser(
        "best-times",
        help="Print the region x platform x top-3 best-time table. Report-only.",
    )
    best_times.add_argument(
        "--min-n",
        type=int,
        default=5,
        dest="min_n",
        help="Minimum sample count for a ranked slot to qualify (default: 5)",
    )
    best_times.add_argument(
        "--trend",
        action="store_true",
        help=("Also print each top slot's last 4 fold snapshots from slot_performance_history"),
    )

    feed_share = metrics_sub.add_parser(
        "feed-share",
        help=(
            "Print the interim-cadence exit rule: Shorts-feed share of channel "
            "views over the last N measured days. Report-only."
        ),
    )
    feed_share.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of most-recent measured days to evaluate (default: 7)",
    )
    feed_share.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Minimum Shorts-feed share a day must meet (default: 0.70)",
    )

    report = metrics_sub.add_parser(
        "report",
        help=(
            "Print (or save) a PerformanceReporter markdown report, "
            "one project or --all. Report-only."
        ),
    )
    report_target = report.add_mutually_exclusive_group(required=True)
    report_target.add_argument("--project", help="Project name (matches the tracking record dir)")
    report_target.add_argument(
        "--all", action="store_true", help="Cross-project report spanning every tracked project"
    )
    report.add_argument(
        "--output",
        help="Write the report to this path instead of printing it to stdout",
    )


def _add_platform_decline_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register ``platform-decline``. Top-level rather than a ``metrics``
    subcommand: it reads already-collected metrics and never fetches."""
    p = subparsers.add_parser(
        "platform-decline", help="Report platforms whose day-7 views have dropped"
    )
    p.add_argument("--window", type=int, default=14, help="Shorts per half (default: 14)")
    p.add_argument("--min-drop-pct", type=float, default=20.0, help="Min drop %% (default: 20)")


def _add_post_anomaly_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register ``post-anomaly`` — shadow-only per-post dip/spike report; no operating
    thresholds are exposed as flags (fixed policy, not a runtime knob)."""
    subparsers.add_parser(
        "post-anomaly",
        help="Report per-post day-7 view dips/spikes the references can back (shadow only)",
    )


def _add_doctor_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register ``doctor`` — one-shot install-health report: deps, GPU,
    ffmpeg filters, credentials, storage, tenant stamp. Read-only, no network."""
    p = subparsers.add_parser(
        "doctor", help="Check the install: deps, ffmpeg, GPU, credentials, storage, tenant stamp"
    )
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of text")
    p.add_argument(
        "--calibrate",
        type=Path,
        default=None,
        metavar="CLIP",
        help="Measure the day-master chain on a 15 s slice of CLIP and record the rate "
        "preflight's ETA uses",
    )


def _add_youtube_auth_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register ``youtube-auth``: mints the upload token with the
    ``youtube.upload`` + ``youtube.force-ssl`` scopes via a browser consent
    flow. ``--brand`` mints a second brand channel's own token."""
    p = subparsers.add_parser(
        "youtube-auth",
        help="Consent flow that mints the YouTube upload+comment token",
    )
    p.add_argument(
        "--brand",
        default=None,
        help="Brand slug (routes the token through the brand-scoped credentials dir)",
    )
    p.add_argument(
        "--credentials-dir",
        type=Path,
        default=None,
        dest="credentials_dir",
        help="Override the credentials directory (default: config seo.credentials_dir)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Print the target path and scopes without launching a browser",
    )


def _add_library_hygiene_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register ``library-hygiene`` (dry-run unless ``--apply``)."""
    p = subparsers.add_parser(
        "library-hygiene",
        help="List (and with --apply set private) the public low-view templated shorts",
    )
    p.add_argument("--apply", action="store_true", help="Set them private (default: list only)")
    p.add_argument(
        "--max", type=int, default=50, help="Most videos one run may change (default: 50)"
    )
    p.add_argument("--ids", nargs="+", metavar="VIDEO_ID", help="Private exactly these ids instead")
    p.add_argument(
        "--undo", metavar="LEDGER", help="Restore every video in this ledger to its prior status"
    )
    p.add_argument(
        "--ledger",
        metavar="PATH",
        help="Ledger file (default: <output>/library-hygiene/<date>-ledger.jsonl)",
    )


# ---------------------------------------------------------------------------
# Top-level parser builder
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Construct the full top-level argument parser.

    Extracted from ``main()`` so tests can exercise argument parsing in
    isolation without invoking the full CLI dispatch path.
    """
    _version = importlib.metadata.version("gomoveshift-video")
    parser = argparse.ArgumentParser(description="GoMoveShift - Driving Video Processor")
    parser.add_argument("--version", action="version", version=f"gomoveshift {_version}")
    # Emit the file handler as one JSON object per line for log
    # shippers / grep-by-field tooling. Stream handler stays human-
    # readable so ``tail -f`` is unaffected. Default off — feature flag.
    parser.add_argument(
        "--log-json",
        action="store_true",
        default=False,
        dest="log_json",
        help="Emit the on-disk log file as JSON lines "
        "(keys: timestamp, level, logger, message, exception). "
        "Stream/stdout output stays human-readable.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    _add_analyze_parser(subparsers)
    _add_process_parser(subparsers)
    _add_server_parser(subparsers)
    _add_console_parser(subparsers)
    _add_scan_parser(subparsers)
    _add_pipeline_parser(subparsers)
    _add_drive_parser(subparsers)
    _add_upload_parser(subparsers)
    _add_cleanup_parser(subparsers)
    _add_ingest_parser(subparsers)
    _add_batch_parser(subparsers)
    _add_find_stops_parser(subparsers)
    _add_photos_parser(subparsers)
    _add_publish_journal_parser(subparsers)
    _add_enqueue_plan_parser(subparsers)
    _add_enqueue_asset_parser(subparsers)
    _add_dispatch_queue_parser(subparsers)
    _add_publishing_status_parser(subparsers)
    _add_metrics_parser(subparsers)
    _add_platform_decline_parser(subparsers)
    _add_post_anomaly_parser(subparsers)
    _add_library_hygiene_parser(subparsers)
    _add_doctor_parser(subparsers)
    _add_comments_parser(subparsers)
    _add_youtube_auth_parser(subparsers)
    _add_schedule_parser(subparsers)
    _add_sync_site_parser(subparsers)
    _add_reconcile_playlists_parser(subparsers)
    _add_route_story_prepare_parser(subparsers)
    _add_route_story_proof_parser(subparsers)
    _add_route_story_verify_parser(subparsers)
    _add_route_story_film_parser(subparsers)

    return parser
