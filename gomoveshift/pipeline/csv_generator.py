"""
Pipeline CSV Generator — Metricool CSV with golden reference validation.

Generates a validated Metricool CSV combining:
  - Social post text from SocialGenerator
  - Schedule dates/times from release_schedule.yaml
  - Google Drive links from upload manifest
  - Platform settings matching the golden reference format

Usage:
    from gomoveshift.pipeline.csv_generator import CSVGenerator

    generator = CSVGenerator()
    csv_path = generator.generate(
        project_name="my-project",
        social_posts=posts,        # from SocialGenerator
        schedule=schedule,          # from PipelineScheduler
        manifest=manifest,          # from PipelineUploader
        output_dir="/path/to/output",
    )
"""

import csv
import io
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from gomoveshift.pipeline.text_format import (
    NON_TWITTER_CHAR_LIMIT,  # noqa: F401 — re-exported, single-source in text_format
    TWITTER_CHAR_LIMIT,
    close_sentence,
    truncate_for_platforms,
)
from gomoveshift.pipeline.text_format import (
    format_recording_date_pretty as _format_recording_date_pretty,
)
from gomoveshift.publishing.photo_image_url import (
    drive_direct_download_link as _drive_link_for_metricool,
)
from gomoveshift.publishing.plan_gates import (
    assert_shorts_metadata_in_region as _assert_shorts_metadata_in_region,
)
from gomoveshift.publishing.plan_gates import (
    assert_shorts_not_scheduled_before_publish as _assert_shorts_not_scheduled_before_publish,
)

from ..video_types import VideoType
from .longform_metadata import HIGHLIGHTS_TRIM_THRESHOLD
from .scheduler import ReleaseSchedule, ScheduledSlot
from .social_generator import SocialPost
from .uploader import UploadManifest

logger = logging.getLogger(__name__)


# Leading characters that make a spreadsheet treat a cell as a formula
# (OWASP CSV-injection). A cell starting with any of these executes when the
# CSV is opened in Sheets/Excel.
_CSV_INJECTION_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _escape_csv_cell(value: str) -> str:
    """Neutralize formula injection in a user/geocoder/LLM-derived text cell.

    Per OWASP, prefix a single leading apostrophe when the value begins with a
    formula-trigger character (``= + - @`` or tab/CR). The apostrophe forces
    spreadsheets to treat the cell as literal text. Empty / non-string /
    safe-leading values pass through unchanged so numeric and known-safe
    columns are never mangled.
    """
    if not isinstance(value, str) or not value:
        return value
    if value.startswith(_CSV_INJECTION_PREFIXES):
        return "'" + value
    return value


def _assert_metricool_rows_have_empty_youtube_category(rows: "List[Dict[str, str]]") -> None:
    """Runtime gate: every Metricool CSV row's ``YouTube Video Category`` must be ``""``.

    Per the Metricool CSV contract: populated numeric category IDs silently route Metricool's import to
    drafts. The two row-construction sites already force ``""``; this
    helper is the upstream-of-irreversible-artifact gate (operator handoff
    to Metricool is the irreversible step — once posts land in drafts,
    nobody notices until the schedule misfires).

    Robust-by-construction would be a structural template that doesn't
    expose the column at all, but ``GOLDEN_COLUMNS`` is fixed by Metricool's
    importer schema, so a runtime assertion is the next-best layer.
    """
    offenders = [
        (i, row.get("YouTube Video Category"))
        for i, row in enumerate(rows)
        if row.get("YouTube Video Category") != ""
    ]
    if offenders:
        raise AssertionError(
            f"Metricool CSV has {len(offenders)} row(s) with non-empty "
            f"YouTube Video Category — populated values silently route posts to "
            f"drafts. First offender: row {offenders[0][0]} = {offenders[0][1]!r}"
        )


def _assert_metricool_rows_have_media(rows: "List[Dict[str, str]]") -> None:
    """Runtime gate: every shorts Metricool CSV row must carry a media URL.

    ``Picture Url 1`` is the Google-Drive shareable link Metricool fetches to
    attach the short's video to the post. An EMPTY value is not a benign
    omission — a short is a video by definition, so a row with no media is a
    post Metricool silently imports with nothing to publish (a ``--skip-upload`` run shipped a 121-row CSV with every
    ``Picture Url 1`` blank, so Metricool showed no videos).

    ``validate_csv`` only flags a *non-empty non-URL* value, so a fully-blank
    column passed every prior check. This is the upstream-of-irreversible-
    artifact gate (the irreversible step is the operator importing the CSV
    into Metricool). Raises naming the blank rows so the caller never writes a
    dead deliverable. The caller defers (does not write) on a deliberate
    skip-upload audit, so this only fires when a CSV is actually produced
    without the media that makes it importable.
    """
    blank = [i for i, row in enumerate(rows) if not (row.get("Picture Url 1") or "").strip()]
    if blank:
        raise ValueError(
            f"METRICOOL-CSV-NO-MEDIA: {len(blank)}/{len(rows)} shorts row(s) have an "
            f"empty 'Picture Url 1' — Metricool has no video to post and silently "
            f"drops them. Shorts must be Drive-uploaded before the CSV is written. "
            f"Blank rows (0-based): {blank[:10]}{'…' if len(blank) > 10 else ''}"
        )


def _assert_metricool_rows_have_schedule(rows: "List[Dict[str, str]]") -> None:
    """Runtime gate: every shorts Metricool CSV row must carry a Date AND Time.

    A short with no schedule slot yields blank ``Date``/``Time`` cells. Metricool
    imports a dateless/timeless row as an UNSCHEDULED (immediate) post — it fires
    the instant the CSV lands instead of on its planned drop. ``validate_csv``
    only flags a *malformed* date/time, so a fully blank cell passed every prior
    check and the builder merely logged "skipping slot" while shipping the blank
    row.

    Robust-by-construction upstream, ``required_shorts_per_lf`` derives the
    planner target from the same effective window count the scheduler fills
    (``--shorts-per-day N`` caps both) AND ``cap_shorts_to_per_lf_capacity``
    trims each LF to that same ceiling before any render — the day-wide total
    matching is not enough on its own, because attribution to LFs is emergent
    and can leave one LF over-full while the total looks right (e.g. 56 shorts
    distributed 11/21/10/14 over four 14-slot windows).
    This is the upstream-of-irreversible-artifact backstop (the operator's
    Metricool import), mirroring the media gate. Raises naming the blank rows so
    a slotless deliverable is never written.
    """
    blank = [
        i
        for i, row in enumerate(rows)
        if not (row.get("Date") or "").strip() or not (row.get("Time") or "").strip()
    ]
    if blank:
        raise ValueError(
            f"METRICOOL-CSV-NO-SCHEDULE: {len(blank)}/{len(rows)} shorts row(s) "
            f"have an empty Date/Time — Metricool imports a slotless row as an "
            f"immediate post, not on its planned drop. Every short must have a "
            f"schedule slot before the CSV is written. Blank rows (0-based): "
            f"{blank[:10]}{'…' if len(blank) > 10 else ''}"
        )


_GRAMMAR_GATE_COLUMNS = ("YouTube Video Title", "TikTok Title", "Text", "YouTube Video Tags")


def _assert_metricool_rows_have_no_grammar_issues(rows: "List[Dict[str, str]]") -> None:
    """Runtime gate: no shorts row may ship a preposition-collision splice.

    Reuses ``content_gates.find_grammar_issues`` (string-based, independent of
    any structured location label) against the title, caption/hashtag ``Text``,
    TikTok title, and tags columns -- the surfaces that publish verbatim.
    Upstream-of-irreversible-artifact gate: raises naming the offending row
    BEFORE the CSV is written, so a spliced string like ``"drive through To
    Clonakilty"`` or ``"#ToClonakilty"`` can never reach the operator's
    Metricool import.
    """
    # Local import: content_gates -> pipeline_validator import chain, kept lazy
    # to avoid a module-load cycle with csv_generator.
    from .validation.content_gates import find_grammar_issues

    failures: "List[str]" = []
    for i, row in enumerate(rows):
        for col in _GRAMMAR_GATE_COLUMNS:
            for msg in find_grammar_issues(row.get(col, "")):
                failures.append(f"row {i} {col}={row.get(col)!r}: {msg}")
    if failures:
        raise ValueError(
            f"METRICOOL-CSV-GRAMMAR: {len(failures)} grammar issue(s) in the "
            f"shorts CSV. First: {failures[0]}"
        )


def _first_comment_lead(trim_ratio: Optional[float]) -> str:
    """Return the first-comment lead phrase based on the parent LF's trim band.

    When the parent LF is itself a heavy-trim cut,
    "Full drive" is a false framing — viewers expect a full drive but
    click through to a highlights reel. Branding the short's first
    comment from the same ``HIGHLIGHTS_TRIM_THRESHOLD`` used by the
    title verb and description hook keeps every surface aligned with
    the actual deliverable.

    Shorts→LF funnel: the compact ``▶`` marker + ``youtu.be`` short-link
    (built by the caller) is the funnel line — not clickable on Shorts
    (platform policy) but visible/copyable. It lives ONLY here because
    shorts rows multi-post (see :data:`DEFAULT_PLATFORMS`): a YouTube URL
    in the shared ``Text`` cell would ship inside TikTok/IG/FB captions.

    ``None`` → "▶ Full drive:" (no regression for callers that haven't
    threaded ``trim_ratio`` yet).
    """
    if trim_ratio is not None and trim_ratio < HIGHLIGHTS_TRIM_THRESHOLD:
        return "▶ Highlights reel:"
    return "▶ Full drive:"


# Golden reference path. Bundled with the wheel via hatch force-include
# (see pyproject.toml [tool.hatch.build.targets.wheel.force-include]).
# Source layout has the file at repo root; installed wheel has it inside
# the gomoveshift/ package dir. Try both, package-dir first.
_PKG_DIR = Path(__file__).resolve().parent.parent  # .../gomoveshift/
_REPO_ROOT = _PKG_DIR.parent  # repo root (source) or site-packages/ (installed)
GOLDEN_REFERENCE = _PKG_DIR / "metricool_reference_header.csv"
if not GOLDEN_REFERENCE.exists():
    GOLDEN_REFERENCE = _REPO_ROOT / "metricool_reference_header.csv"

# The exact 31-column header from the golden reference.
# MUST match exactly — any deviation causes Metricool import failure.
GOLDEN_COLUMNS = [
    "Text",
    "Date",
    "Time",
    "Draft",
    "Facebook",
    "Twitter",
    "LinkedIn",
    "GBP",
    "Instagram",
    "Pinterest",
    "TikTok",
    "YouTube",
    "Threads",
    "Picture Url 1",
    "Shortener",
    "Instagram Post Type",
    "Facebook Post Type",
    "YouTube Video Title",
    "YouTube Video Type",
    "YouTube Video Privacy",
    "YouTube video for kids",
    "YouTube Video Category",
    "YouTube Video Tags",
    "TikTok Title",
    "TikTok disable comments",
    "TikTok disable duet",
    "TikTok disable stitch",
    "TikTok Post Privacy",
    "TikTok Auto Add Music",
    "Pinterest Board",
    "First Comment Text",
]

# Public alias — spec name referenced by consumers (Phase D, tests).
CSV_HEADERS = GOLDEN_COLUMNS


def build_consolidated_shorts_csv(
    output_path: "Path",
    shorts_by_lf: "Dict[int, List]",
    slots_by_short_index: "Dict[int, object]",
    drive_links_by_short_index: "Dict[int, str]",
    parent_lf_video_id_by_short_index: "Dict[int, str]",
    parent_recording_date_by_short_index: "Dict[int, str]",
    platforms: "Optional[Dict[str, str]]" = None,
    excel_compat: bool = False,
    parent_trim_ratio_by_short_index: "Optional[Dict[int, float]]" = None,
    parent_publish_at_by_short_index: "Optional[Dict[int, str]]" = None,
    publish_tiktok: bool = True,
    require_media: bool = True,
) -> "Path":
    """Write ONE consolidated Metricool CSV covering shorts across all LFs.

    The "First Comment Text" column is set to
    ``f"▶ Full drive: https://youtu.be/{parent_video_id}"`` so every short
    links back to its parent LF's YouTube upload.

    Column order matches :data:`GOLDEN_COLUMNS` / :data:`CSV_HEADERS` exactly.

    Encoding:

    * ``excel_compat=False`` (default) → write plain ``utf-8`` with no BOM.
      Metricool is the actual consumer of this CSV and the BOM may be
      breaking its column-mapping parser (the first header column ends up
      as ``\\ufeffText`` instead of ``Text``).
    * ``excel_compat=True`` → write ``utf-8-sig`` (BOM-prefixed). Preserves
      the legacy behavior for operators who load the CSV directly into
      older Excel versions that need the BOM to detect UTF-8.

    TikTok toggle: ``publish_tiktok=True`` (default, driven by
    ``config.drive.publish_tiktok``) sets the ``TikTok`` column to ``TRUE`` and
    populates the TikTok-specific columns, so Metricool schedules the short on
    TikTok. Metricool handles publishing — no first-party TikTok auth needed.
    Pass ``publish_tiktok=False`` to force the ``TikTok`` toggle to ``FALSE``
    and leave every TikTok-specific column empty (suppresses TikTok scheduling;
    the analytics collector ``tracking.collector.TikTokCollector`` is a
    separate first-party path that never gated publishing). The 31-column GOLDEN header is unchanged either
    way — only the column *values* differ. The TikTok-specific columns are also
    left empty whenever the ``platforms`` dict itself carries ``"TikTok":
    "FALSE"`` (see below), not only via ``publish_tiktok`` — the two gates
    can't disagree.

    First-party route suppression: ``platforms`` defaults to
    :data:`DEFAULT_PLATFORMS`, every platform's toggle ``"TRUE"``/``"FALSE"``.
    A caller that has flipped a ``DriveConfig.publishing_route_<platform>``
    to ``"first_party"`` must build ``platforms`` via
    :func:`platforms_for_publishing_routes` first — Phase D's
    ``_emit_consolidated_shorts_csv`` does this — so that platform's toggle
    is forced ``"FALSE"`` here and Metricool never re-posts a short that the
    publish-plan sidecar already routes directly.

    Parent-video-ID gate (greppable ``METRICOOL-CSV-NO-PARENT-ID``): raises
    when ``parent_lf_video_id_by_short_index`` is non-empty (the caller is
    participating) but is missing an entry for one or more of this batch's
    shorts — a silent gap there drops the "First Comment Text" funnel link
    back to the parent LF for just that short with nothing failing. A caller
    that never threads the map at all (empty dict, the legacy no-participation
    case) is unaffected — that fallback to the post's own ``first_comment``
    stays intact.
    """
    from ..video_types import VideoType  # local import (avoids cycles during module import)

    plat = dict(DEFAULT_PLATFORMS if platforms is None else platforms)
    if not publish_tiktok:
        # Robust-by-construction: never let the platform toggle enable TikTok
        # when scheduling is gated off, regardless of what the caller's
        # ``platforms`` dict / DEFAULT_PLATFORMS carries.
        plat["TikTok"] = "FALSE"

    rows = []
    # Missing-mapping gate (parent LF video ID only — see below): a caller
    # that populates this map for SOME shorts but drops one is a bug, not
    # the legacy no-participation case (empty map), which is a deliberate
    # fallback to the short's own ``first_comment`` (test_shorts_fulldrive_
    # link.py::test_missing_parent_id_falls_back_unchanged). Slot/drive-link
    # gaps are already hard-gated unconditionally downstream (schedule/media
    # asserts below), so only this partial-participation case needs its own
    # check here.
    missing_parent_ids = []
    for lf_idx in sorted(shorts_by_lf.keys()):
        for post in shorts_by_lf[lf_idx]:
            short_idx = post.short_index
            if short_idx not in slots_by_short_index:
                logger.warning(
                    "Short index %d missing from slots_by_short_index — skipping slot", short_idx
                )
            if short_idx not in drive_links_by_short_index:
                logger.warning("Short index %d missing from drive_links_by_short_index", short_idx)
            if (
                parent_lf_video_id_by_short_index
                and short_idx not in parent_lf_video_id_by_short_index
            ):
                missing_parent_ids.append(short_idx)
            slot = slots_by_short_index.get(short_idx)
            drive_link = drive_links_by_short_index.get(short_idx, "")
            parent_id = parent_lf_video_id_by_short_index.get(short_idx, "")
            parent_trim = (
                parent_trim_ratio_by_short_index.get(short_idx)
                if parent_trim_ratio_by_short_index
                else None
            )

            # Recording-date enforcement. Required — Phase BC's
            # ``_assert_per_lf_recording_date_complete`` is the upstream
            # guarantee; this raise is defense-in-depth so any future
            # construction regression fails at CSV-write time rather than
            # shipping a Metricool CSV without the "Filmed ..." sentence.
            recording_date = parent_recording_date_by_short_index.get(short_idx)
            if not isinstance(recording_date, str) or not recording_date.strip():
                raise ValueError(
                    f"recording date missing for short_index {short_idx} "
                    f"(parent LF recording-date map has "
                    f"{len(parent_recording_date_by_short_index)} entries)"
                )
            filmed_suffix = f"Filmed {_format_recording_date_pretty(recording_date)}."

            row = {col: "" for col in GOLDEN_COLUMNS}
            caption = getattr(post, "caption", "") or ""
            caption_stripped = caption.rstrip()
            # Twitter=TRUE in DEFAULT_PLATFORMS, so the " Filmed <Month YYYY>."
            # sentence appended AFTER _build_caption's own 280-cap can push the
            # shared ``Text`` past Twitter's limit. Trim
            # the CAPTION (word/grapheme-safe) to leave room so the required
            # Filmed sentence always survives, rather than truncating the whole.
            _twitter_enabled = str(plat.get("Twitter", "")).upper() == "TRUE"
            if caption_stripped:
                _combined = f"{close_sentence(caption_stripped)} {filmed_suffix}"
                if _twitter_enabled and len(_combined) > TWITTER_CHAR_LIMIT:
                    # 2 = the join space + the full stop close_sentence may add
                    _room = TWITTER_CHAR_LIMIT - len(filmed_suffix) - 2
                    _kept = truncate_for_platforms(
                        caption_stripped, twitter_enabled=True, limit=_room
                    )
                    _combined = f"{close_sentence(_kept)} {filmed_suffix}"
                row["Text"] = _escape_csv_cell(_combined)
            else:
                row["Text"] = _escape_csv_cell(filmed_suffix)
            row["Date"] = getattr(slot, "date", "") if slot else ""
            row["Time"] = getattr(slot, "time", "") if slot else ""
            row["Draft"] = "FALSE"
            for platform, enabled in plat.items():
                if platform in row:
                    row[platform] = enabled
            # uc?export=download form so TikTok's media pull gets video bytes,
            # not Drive's HTML viewer page (Meta works with it too).
            row["Picture Url 1"] = _drive_link_for_metricool(drive_link)
            row["Shortener"] = "FALSE"
            row["Instagram Post Type"] = "REEL"
            row["Facebook Post Type"] = "REEL"
            row["YouTube Video Title"] = _escape_csv_cell(getattr(post, "youtube_title", ""))
            row["YouTube Video Type"] = VideoType.SHORT.to_youtube_type()
            row["YouTube Video Privacy"] = "public"
            row["YouTube video for kids"] = "no"
            # Metricool rejects populated YouTube category
            # IDs on shorts rows and silently routes the import to draft. Legacy
            # known-good CSVs leave this column empty in every row. Force "" here
            # regardless of what's on the SocialPost.
            row["YouTube Video Category"] = ""
            row["YouTube Video Tags"] = _escape_csv_cell(getattr(post, "youtube_tags", ""))
            # TikTok columns only populated when publish_tiktok is opted in.
            # Default leaves them empty so Metricool schedules no
            # TikTok post — the row dict initialised them to "" already.
            if publish_tiktok and str(plat.get("TikTok", "")).upper() == "TRUE":
                row["TikTok Title"] = getattr(post, "tiktok_title", "")
                row["TikTok disable comments"] = "no"
                row["TikTok disable duet"] = "no"
                row["TikTok disable stitch"] = "no"
                row["TikTok Post Privacy"] = "PUBLIC_TO_EVERYONE"
                row["TikTok Auto Add Music"] = "no"
            row["Pinterest Board"] = getattr(post, "pinterest_board", "")
            if parent_id:
                lead = _first_comment_lead(parent_trim)
                row["First Comment Text"] = f"{lead} https://youtu.be/{parent_id}"
            else:
                row["First Comment Text"] = _escape_csv_cell(getattr(post, "first_comment", ""))
            rows.append(row)

    # Parent-video-ID gate: raise upstream of the write when the caller
    # threads this map for some shorts but not all — a silent gap here
    # ships a row whose "First Comment Text" drops the funnel link to the
    # parent LF with no warning anyone reads.
    if missing_parent_ids:
        raise ValueError(
            f"METRICOOL-CSV-NO-PARENT-ID: {len(missing_parent_ids)} short(s) missing from "
            f"parent_lf_video_id_by_short_index while other shorts have entries — the "
            f"funnel link back to the parent LF would silently drop for these rows. "
            f"Missing short_index values: {missing_parent_ids[:10]}"
            f"{'…' if len(missing_parent_ids) > 10 else ''}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Upstream-of-irreversible-artifact gate. Construction above forces row["YouTube Video Category"] = "" already; this guards
    # against any future regression that bypasses that assignment.
    _assert_metricool_rows_have_empty_youtube_category(rows)
    # per-short region gate on the shipped title/caption/tags — hard-fail
    # upstream of the CSV write when a surface names a county the short's own
    # clip GPS never visited (the parent-LF county leaking into a far-corner
    # short's tags). No-op for GPS-less / legacy shorts.
    _assert_shorts_metadata_in_region(shorts_by_lf)
    # Grammar gate: hard-fail a preposition-collision splice ("drive
    # through To Clonakilty", "#ToClonakilty") on any shipped title/caption/
    # hashtag/tag surface, upstream of the CSV write.
    _assert_metricool_rows_have_no_grammar_issues(rows)
    # Schedule gate: a short with no slot ships blank Date/Time, which
    # Metricool imports as an immediate post rather than on its planned drop.
    # Unconditional — a slotless row is always a bug (the schedule is computed
    # regardless of upload), so this fires even on a skip-upload audit.
    _assert_metricool_rows_have_schedule(rows)
    # Media gate: a shorts CSV with blank ``Picture Url 1`` cells
    # is a dead Metricool deliverable. Default-on; the Phase D caller passes
    # ``require_media=False`` only on a deliberate skip-upload audit (and itself
    # defers the write when no media exists), so this raises only when a CSV is
    # actually produced without the Drive links that make it importable.
    if require_media:
        _assert_metricool_rows_have_media(rows)
    # Publish-timing gate: a short whose first comment links to its
    # parent LF must not post before that LF goes public, or the link 404s.
    # Only enforced when the caller threads per-short publish_at (Phase D does);
    # skip-upload audits / legacy callers omit it and are unaffected.
    if parent_publish_at_by_short_index:
        _assert_shorts_not_scheduled_before_publish(
            shorts_by_lf, slots_by_short_index, parent_publish_at_by_short_index
        )
    encoding = "utf-8-sig" if excel_compat else "utf-8"
    with open(output_path, "w", newline="", encoding=encoding) as f:
        writer = csv.DictWriter(f, fieldnames=GOLDEN_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Consolidated shorts CSV: %s (%d rows)", output_path, len(rows))
    return output_path


# Default platform settings for GoMoveShift shorts
DEFAULT_PLATFORMS = {
    "Facebook": "TRUE",
    "Twitter": "TRUE",
    "LinkedIn": "FALSE",
    "GBP": "FALSE",
    "Instagram": "TRUE",
    "Pinterest": "TRUE",
    "TikTok": "TRUE",
    "YouTube": "TRUE",
    "Threads": "TRUE",
}

# DriveConfig.publishing_route_<platform> field name -> the Metricool CSV
# toggle column it must suppress once flipped to "first_party" (Phase D's
# publish-plan sidecar, phase_d.py, then owns that platform's posting).
_PUBLISHING_ROUTE_FIELD_TO_CSV_COLUMN = {
    "publishing_route_youtube": "YouTube",
    "publishing_route_instagram": "Instagram",
    "publishing_route_facebook": "Facebook",
    "publishing_route_tiktok": "TikTok",
}


def platforms_for_publishing_routes(drive_cfg, base: "Optional[Dict[str, str]]" = None):
    """Return a CSV platform-toggle dict with every ``first_party``-routed
    platform forced to ``"FALSE"``.

    Robust-by-construction fix for the Metricool double-post: when an
    operator flips ``DriveConfig.publishing_route_<platform>`` to
    ``"first_party"``, Phase D's publish-plan sidecar starts posting that
    platform directly (phase_d.py `_emit_publish_plan_sidecar`). Without
    this, the Metricool CSV's toggle for that same platform stays
    ``"TRUE"`` and importing it posts the short a second time -- for
    Instagram, the duplicate also burns the account's shared 24h Content
    Publishing quota with Metricool.

    ``drive_cfg`` is a ``DriveConfig`` (or any object exposing the same
    ``publishing_route_*`` attributes); ``None`` and an all-``"off"``
    config (the default) both return ``base`` (default
    :data:`DEFAULT_PLATFORMS`) unchanged.
    """
    plat = dict(DEFAULT_PLATFORMS if base is None else base)
    if drive_cfg is None:
        return plat
    for field, column in _PUBLISHING_ROUTE_FIELD_TO_CSV_COLUMN.items():
        if getattr(drive_cfg, field, "off") == "first_party":
            plat[column] = "FALSE"
    return plat


# Validation patterns
TIME_PATTERN = re.compile(r"^\d{2}:\d{2}:\d{2}$")
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
VALID_BOOLEANS = {"TRUE", "FALSE"}
VALID_YES_NO = {"yes", "no"}
VALID_TIKTOK_PRIVACY = {"PUBLIC_TO_EVERYONE", "MUTUAL_FOLLOW_FRIENDS", "SELF_ONLY"}
VALID_YOUTUBE_PRIVACY = {"public", "private", "unlisted"}
VALID_POST_TYPES = {"REEL", "POST", "STORY"}
VALID_YT_VIDEO_TYPES = {"SHORT", "VIDEO"}


@dataclass
class CSVValidationResult:
    """Result of CSV validation against golden reference."""

    valid: bool
    errors: List[str]
    row_count: int
    column_count: int


def _load_golden_header() -> List[str]:
    """Load the golden reference header columns."""
    if not GOLDEN_REFERENCE.exists():
        raise FileNotFoundError(
            f"Golden reference not found: {GOLDEN_REFERENCE}\n"
            f"This file is the anchor for all Metricool CSV validation."
        )

    with open(GOLDEN_REFERENCE, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        return next(reader)


def _validate_csv_text(text: str) -> CSVValidationResult:
    """
    Validate CSV *text* against the golden reference.

    The single implementation of the rules, shared by :func:`validate_rows`
    (before the write) and :func:`validate_csv` (after it):

    Checks:
      - Header matches golden reference exactly
      - All boolean fields are TRUE/FALSE
      - All yes/no fields are yes/no (lowercase)
      - Date format is YYYY-MM-DD
      - Time format is HH:MM:SS
      - URLs start with http
      - Enum fields have valid values

    Returns:
        CSVValidationResult
    """
    errors = []
    reference_header = _load_golden_header()

    with io.StringIO(text) as f:
        reader = csv.reader(f)

        try:
            header = next(reader)
        except StopIteration:
            return CSVValidationResult(
                valid=False,
                errors=["File is empty"],
                row_count=0,
                column_count=0,
            )

        # Check header matches
        if header != reference_header:
            errors.append("Header mismatch with golden reference")
            if len(header) != len(reference_header):
                errors.append(f"  Column count: {len(header)} (expected {len(reference_header)})")
            for i, (got, expected) in enumerate(zip(header, reference_header)):
                if got != expected:
                    errors.append(f"  Column {i}: '{got}' (expected '{expected}')")

        col_idx = {name: i for i, name in enumerate(header)}
        row_count = 0

        for row_num, row in enumerate(reader, start=2):
            row_count += 1

            if len(row) != len(header):
                errors.append(f"Row {row_num}: Column count {len(row)} != header {len(header)}")
                continue

            # Date: YYYY-MM-DD
            if "Date" in col_idx:
                val = row[col_idx["Date"]]
                if val and not DATE_PATTERN.match(val):
                    errors.append(f"Row {row_num}: Invalid date '{val}' (expected YYYY-MM-DD)")

            # Time: HH:MM:SS
            if "Time" in col_idx:
                val = row[col_idx["Time"]]
                if val and not TIME_PATTERN.match(val):
                    errors.append(f"Row {row_num}: Invalid time '{val}' (expected HH:MM:SS)")

            # Boolean fields: TRUE/FALSE
            boolean_fields = [
                "Draft",
                "Facebook",
                "Twitter",
                "LinkedIn",
                "GBP",
                "Instagram",
                "Pinterest",
                "TikTok",
                "YouTube",
                "Threads",
                "Shortener",
            ]
            for field_name in boolean_fields:
                if field_name in col_idx:
                    val = row[col_idx[field_name]]
                    if val and val not in VALID_BOOLEANS:
                        errors.append(f"Row {row_num}: {field_name}='{val}' (expected TRUE/FALSE)")

            # yes/no fields (lowercase)
            yesno_fields = [
                "YouTube video for kids",
                "TikTok disable comments",
                "TikTok disable duet",
                "TikTok disable stitch",
                "TikTok Auto Add Music",
            ]
            for field_name in yesno_fields:
                if field_name in col_idx:
                    val = row[col_idx[field_name]]
                    if val and val not in VALID_YES_NO:
                        errors.append(f"Row {row_num}: {field_name}='{val}' (expected yes/no)")

            # TikTok Post Privacy
            if "TikTok Post Privacy" in col_idx:
                val = row[col_idx["TikTok Post Privacy"]]
                if val and val not in VALID_TIKTOK_PRIVACY:
                    errors.append(f"Row {row_num}: TikTok Post Privacy='{val}'")

            # YouTube Video Privacy
            if "YouTube Video Privacy" in col_idx:
                val = row[col_idx["YouTube Video Privacy"]]
                if val and val not in VALID_YOUTUBE_PRIVACY:
                    errors.append(f"Row {row_num}: YouTube Video Privacy='{val}'")

            # Post types
            for pt_field in ["Instagram Post Type", "Facebook Post Type"]:
                if pt_field in col_idx:
                    val = row[col_idx[pt_field]]
                    if val and val not in VALID_POST_TYPES:
                        errors.append(f"Row {row_num}: {pt_field}='{val}'")

            # YouTube Video Type
            if "YouTube Video Type" in col_idx:
                val = row[col_idx["YouTube Video Type"]]
                if val and val not in VALID_YT_VIDEO_TYPES:
                    errors.append(f"Row {row_num}: YouTube Video Type='{val}'")

            # Picture URL
            if "Picture Url 1" in col_idx:
                val = row[col_idx["Picture Url 1"]]
                if val and not val.startswith("http"):
                    errors.append(f"Row {row_num}: Picture Url 1 is not a URL")

    return CSVValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        row_count=row_count,
        column_count=len(header) if header else 0,
    )


def validate_rows(
    rows: List[Dict[str, str]], header: Optional[List[str]] = None
) -> CSVValidationResult:
    """
    Validate in-memory CSV rows against the golden reference.

    Runs :func:`_validate_csv_text` — the same rules, the same code — over the
    exact text ``csv.DictWriter`` would put in the file, so an invalid CSV can
    be rejected *before* it is created rather than left on disk by a failed
    post-write check. Nothing is written; the rendering is in memory.

    Args:
        rows: Column-name -> value dicts, as handed to ``csv.DictWriter``
        header: Column order for the write (defaults to ``GOLDEN_COLUMNS``)

    Returns:
        CSVValidationResult
    """
    columns = list(header) if header is not None else list(GOLDEN_COLUMNS)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)

    return _validate_csv_text(buffer.getvalue())


def validate_csv(csv_path: Path) -> CSVValidationResult:
    """
    Validate a written CSV against the golden reference.

    Prefer :func:`validate_rows` when the rows are still in memory — checking
    after the write leaves a malformed file on disk.

    Returns:
        CSVValidationResult
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        return CSVValidationResult(
            valid=False,
            errors=[f"File not found: {csv_path}"],
            row_count=0,
            column_count=0,
        )

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        return _validate_csv_text(f.read())


class CSVGenerator:
    """
    Generates a Metricool-compatible CSV matching the golden reference.

    Combines social posts, schedule, and Drive links into a validated CSV
    ready for Metricool import.
    """

    def __init__(
        self,
        platforms: Optional[Dict[str, str]] = None,
        publish_tiktok: bool = True,
    ):
        """
        Args:
            platforms: Override platform toggles (default from DEFAULT_PLATFORMS)
            publish_tiktok: Whether to publish to TikTok (default True, driven by
                ``config.drive.publish_tiktok``). True sets the TikTok toggle to
                TRUE and populates the TikTok-specific columns so Metricool
                schedules the short on TikTok. Pass False to force the toggle to
                FALSE and leave the TikTok columns empty (suppresses TikTok).
        """
        self.platforms = platforms or DEFAULT_PLATFORMS.copy()
        self.publish_tiktok = publish_tiktok

    def _build_row(
        self,
        post: SocialPost,
        slot: Optional[ScheduledSlot],
        drive_link: str,
        video_type: VideoType = VideoType.SHORT,
    ) -> Dict[str, str]:
        """
        Build a single CSV row matching the golden reference columns.

        Args:
            post: Social post content
            slot: Schedule slot (date/time)
            drive_link: Google Drive download URL
            video_type: VideoType for YouTube type branching

        Returns:
            Dict mapping column name to value
        """
        row = {col: "" for col in GOLDEN_COLUMNS}

        # Core content
        row["Text"] = _escape_csv_cell(post.caption)
        row["Date"] = slot.date if slot else ""
        row["Time"] = slot.time if slot else ""
        row["Draft"] = "FALSE"

        # Platform toggles. Force TikTok off when scheduling is gated,
        # regardless of what self.platforms / DEFAULT_PLATFORMS carries.
        for platform, enabled in self.platforms.items():
            if platform in row:
                row[platform] = enabled
        if not self.publish_tiktok:
            row["TikTok"] = "FALSE"

        # Media link. Shorts use the uc?export=download form so
        # TikTok's media pull gets video bytes, not Drive's HTML viewer page.
        row["Picture Url 1"] = _drive_link_for_metricool(drive_link)

        # URL shortener
        row["Shortener"] = "FALSE"

        # Instagram
        row["Instagram Post Type"] = "REEL"

        # Facebook
        row["Facebook Post Type"] = "REEL"

        # YouTube — branch on video type (was hardcoded "SHORT")
        # NOTE: youtube_description is NOT in the Metricool CSV golden reference.
        # It lives on SocialPost for direct YouTube API upload only.
        row["YouTube Video Title"] = _escape_csv_cell(post.youtube_title)
        row["YouTube Video Type"] = video_type.to_youtube_type()
        row["YouTube Video Privacy"] = "public"
        row["YouTube video for kids"] = "no"
        # Shorts rows MUST emit empty YouTube category;
        # Metricool rejects the numeric ID and silently drafts. Legacy known-good
        # CSVs are empty in this column across all rows. Force "" regardless of
        # what's on the SocialPost.
        row["YouTube Video Category"] = ""
        row["YouTube Video Tags"] = _escape_csv_cell(post.youtube_tags)

        # TikTok columns only populated when publish_tiktok is opted in;
        # default leaves them empty (row initialised to "") so Metricool
        # schedules no TikTok post.
        if self.publish_tiktok:
            row["TikTok Title"] = post.tiktok_title
            row["TikTok disable comments"] = "no"
            row["TikTok disable duet"] = "no"
            row["TikTok disable stitch"] = "no"
            row["TikTok Post Privacy"] = "PUBLIC_TO_EVERYONE"
            row["TikTok Auto Add Music"] = "no"

        # Pinterest
        row["Pinterest Board"] = post.pinterest_board

        # First comment (Instagram hashtag overflow)
        row["First Comment Text"] = _escape_csv_cell(post.first_comment)

        return row

    def _match_short_to_file(
        self,
        short_index: int,
        manifest: UploadManifest,
    ) -> str:
        """
        Find the Drive link for a short by index.

        Matches filenames like: short_01_rank3.mp4 → index 1
        """
        for filename, link in manifest.files.items():
            # Parse index from filename: short_XX_rankY.mp4
            # or short_XX_final.mp4 or short_XX.mp4
            parts = filename.replace(".mp4", "").split("_")
            for part in parts:
                try:
                    idx = int(part)
                    if idx == short_index:
                        return link
                except ValueError:
                    continue

        logger.warning(
            f"No Drive link found for short #{short_index} "
            f"in manifest ({len(manifest.files)} files)"
        )
        return ""

    def generate(
        self,
        project_name: str,
        social_posts: List[SocialPost],
        schedule: ReleaseSchedule,
        manifest: UploadManifest,
        output_dir: str,
        validate: bool = True,
        excel_compat: bool = False,
    ) -> Path:
        """
        Generate the Metricool CSV file.

        Args:
            project_name: Project name for filename
            social_posts: Social posts from SocialGenerator
            schedule: Release schedule from PipelineScheduler
            manifest: Upload manifest from PipelineUploader
            output_dir: Directory for output CSV
            validate: Run golden reference validation on the rows
                before writing; an invalid CSV is never created

        Returns:
            Path to generated CSV

        Raises:
            ValueError: If validation fails
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / f"{project_name}_metricool.csv"

        # Build slot lookup by short index
        slot_lookup: Dict[int, ScheduledSlot] = {}
        for slot in schedule.slots:
            slot_lookup[slot.short_index] = slot

        # Build rows — every short must have a schedule slot
        rows = []
        for post in social_posts:
            slot = slot_lookup.get(post.short_index)
            if not slot:
                raise ValueError(
                    f"No schedule slot for short #{post.short_index} "
                    f"(schedule has {len(slot_lookup)} slot(s) for {len(social_posts)} short(s))"
                )

            drive_link = self._match_short_to_file(post.short_index, manifest)
            if not drive_link:
                logger.warning(f"No Drive link for short #{post.short_index}")

            row = self._build_row(post, slot, drive_link)
            rows.append(row)

        # Verify Drive links are reachable before writing CSV
        if manifest.files:
            broken_links = []
            for filename, link in manifest.files.items():
                if not link or not link.startswith("http"):
                    broken_links.append(f"{filename}: invalid URL '{link}'")
            if broken_links:
                for bl in broken_links:
                    logger.error(f"  BROKEN DRIVE LINK: {bl}")
                raise ValueError(
                    f"{len(broken_links)} broken Drive link(s) in manifest. "
                    f"First: {broken_links[0]}"
                )

            # Check all scheduled shorts have Drive links
            missing_links = []
            for post in social_posts:
                slot = slot_lookup.get(post.short_index)
                if slot and not self._match_short_to_file(post.short_index, manifest):
                    missing_links.append(f"short #{post.short_index}")
            if missing_links:
                for ml in missing_links:
                    logger.error(f"  MISSING DRIVE LINK: {ml}")
                raise ValueError(
                    f"{len(missing_links)} scheduled short(s) have no Drive link. "
                    f"Missing: {', '.join(missing_links)}"
                )

        # Write CSV. Default encoding is plain UTF-8 (no BOM) — Metricool is
        # the actual consumer and the BOM may break its column-mapping parser.
        # Pass excel_compat=True to restore the legacy utf-8-sig BOM behavior
        # for Excel users.
        # Upstream-of-irreversible-artifact gate (populated YouTube
        # Video Category silently drafts the Metricool import).
        _assert_metricool_rows_have_empty_youtube_category(rows)
        # Wire the same media + schedule gates the drive-path CSV writer
        # uses. The schedule gate is unconditional (a slotless row imports as an
        # immediate post); it always passes here because ``generate`` already
        # rejects a post with no slot above, so it is pure defense. When the
        # manifest has NO Drive links (skip-upload / pre-upload run) every
        # ``Picture Url 1`` is blank — a dead Metricool deliverable — so DEFER the
        # write rather than ship it AND silently disable golden validation
        # (``validate=bool(manifest.files)``). With links present the media gate
        # hard-fails on any blank row before the write.
        _assert_metricool_rows_have_schedule(rows)
        if not manifest.files:
            logger.warning(
                "Deferring Metricool CSV %s: the manifest has no Drive links yet "
                "(skip-upload / pre-upload). Re-run after upload so every short "
                "carries media — refusing to write a link-less dead CSV.",
                csv_path.name,
            )
            return csv_path
        _assert_metricool_rows_have_media(rows)

        # Validate against the golden reference BEFORE the write, so a
        # CSV that fails validation is never created. Validating the written
        # file was a late gate: the raise left the malformed CSV at csv_path for
        # anything that tests existence rather than validity (stages/csv_gen.py
        # reports the failure but does not remove the file).
        if validate:
            validation = validate_rows(rows)
            if not validation.valid:
                error_summary = "\n".join(validation.errors[:10])
                logger.error(
                    f"CSV VALIDATION FAILED ({len(validation.errors)} errors):\n{error_summary}"
                )
                raise ValueError(
                    f"Generated CSV failed golden reference validation with "
                    f"{len(validation.errors)} errors. First error: "
                    f"{validation.errors[0]}"
                )

            logger.info(
                f"✅ CSV validated: {validation.row_count} rows, "
                f"{validation.column_count} columns — matches golden reference"
            )

        encoding = "utf-8-sig" if excel_compat else "utf-8"
        with open(csv_path, "w", newline="", encoding=encoding) as f:
            writer = csv.DictWriter(f, fieldnames=GOLDEN_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        logger.info(f"Generated Metricool CSV: {csv_path} ({len(rows)} rows)")

        return csv_path

    def generate_lf_row(
        self,
        title: str,
        tags: str,
        category: str,
        drive_link: str,
        schedule_date: str,
        schedule_time: str,
        output_path: Path,
        description: str = "",
    ) -> Path:
        """Generate a single-row Metricool CSV for a long-form video.

        Used for regular LF and super LF uploads. YouTube-only (no Instagram,
        TikTok, or Pinterest for long-form content).

        Args:
            title: YouTube video title.
            tags: Comma-separated YouTube tags.
            category: YouTube category ID (e.g. "19").
            drive_link: Google Drive download URL.
            schedule_date: Publish date (YYYY-MM-DD).
            schedule_time: Publish time (HH:MM:SS).
            output_path: Path for output CSV file.
            description: YouTube description text.

        Returns:
            Path to generated CSV.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        row = {col: "" for col in GOLDEN_COLUMNS}

        # YouTube only for long-form. Twitter/X is NOT enabled for LF rows,
        # so we must NOT truncate the shared ``Text`` to Twitter's 280-char
        # limit — that would clip the trailing emoji + hashtag block off the
        # YouTube description for no reason.
        row["YouTube"] = "TRUE"
        row["Instagram"] = "FALSE"
        row["TikTok"] = "FALSE"
        row["Pinterest"] = "FALSE"
        row["Facebook"] = "FALSE"
        # Twitter column stays "" (disabled) — see GOLDEN_COLUMNS default.

        twitter_enabled = (row.get("Twitter") or "").upper() == "TRUE"
        source_text = description if description else title
        row["Text"] = _escape_csv_cell(
            truncate_for_platforms(source_text, twitter_enabled=twitter_enabled)
        )
        row["Date"] = schedule_date
        row["Time"] = schedule_time
        row["Draft"] = "FALSE"

        row["Picture Url 1"] = drive_link
        row["Shortener"] = "FALSE"

        row["YouTube Video Title"] = _escape_csv_cell(title)
        row["YouTube Video Type"] = "VIDEO"
        row["YouTube Video Privacy"] = "public"
        row["YouTube video for kids"] = "no"
        row["YouTube Video Category"] = category
        row["YouTube Video Tags"] = _escape_csv_cell(tags)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=GOLDEN_COLUMNS)
            writer.writeheader()
            writer.writerow(row)

        logger.info("Generated LF CSV: %s", output_path)
        return output_path

    def generate_from_schedule_file(
        self,
        project_name: str,
        social_posts: List[SocialPost],
        schedule_path: str,
        manifest_path: str,
        output_dir: str,
        validate: bool = True,
    ) -> Path:
        """
        Generate CSV from saved schedule and manifest files.

        Convenience method for when user has reviewed/edited the schedule.

        Args:
            project_name: Project name
            social_posts: Social posts
            schedule_path: Path to release_schedule.yaml
            manifest_path: Path to gdrive_manifest.json
            output_dir: Output directory

        Returns:
            Path to generated CSV
        """
        schedule = ReleaseSchedule.load(Path(schedule_path))
        manifest = UploadManifest.load(Path(manifest_path))

        return self.generate(
            project_name=project_name,
            social_posts=social_posts,
            schedule=schedule,
            manifest=manifest,
            output_dir=output_dir,
            validate=validate,
        )
