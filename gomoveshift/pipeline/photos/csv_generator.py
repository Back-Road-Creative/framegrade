"""Metricool CSV generator for photo posts.

Same 31-column golden reference as video shorts, but with key differences:
  - YouTube = FALSE, all YouTube fields empty
  - Instagram Post Type = POST (not REEL)
  - Facebook Post Type = POST (not REEL)
  - Picture Url 1 = Drive link to photo
  - TikTok Title filled (max 90 chars)
"""

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from ..csv_generator import (
    GOLDEN_COLUMNS,
    _assert_metricool_rows_have_schedule,
    _drive_link_for_metricool,
    _escape_csv_cell,
    truncate_for_platforms,
    validate_csv,
)
from ..uploader import UploadManifest
from .scheduler import PhotoSlot

logger = logging.getLogger(__name__)


@dataclass
class PhotoPost:
    """Content for a single photo post."""

    photo_index: int
    caption: str  # Main caption (all platforms, 250 chars max)
    tiktok_caption: str  # TikTok title (90 chars max)
    hashtags: str  # Space-separated hashtag string
    first_comment: str  # Instagram first comment (hook + hashtags)
    photo_filename: str  # For matching to Drive links
    first_comment_hook: str = ""  # Engagement question prepended to first comment

    def to_dict(self) -> dict:
        return {
            "photo_index": self.photo_index,
            "caption": self.caption,
            "tiktok_caption": self.tiktok_caption,
            "hashtags": self.hashtags,
            "first_comment": self.first_comment,
            "photo_filename": self.photo_filename,
            "first_comment_hook": self.first_comment_hook,
        }

    def build_first_comment(self) -> str:
        """Build first comment text: engagement hook + hashtags."""
        if self.first_comment_hook:
            return f"{self.first_comment_hook}\n\n{self.first_comment}"
        return self.first_comment


# Platform settings for photo posts (no YouTube)
PHOTO_PLATFORMS = {
    "Facebook": "TRUE",
    "Twitter": "TRUE",
    "LinkedIn": "FALSE",
    "GBP": "FALSE",
    "Instagram": "TRUE",
    "Pinterest": "TRUE",
    "TikTok": "TRUE",
    "YouTube": "FALSE",
    "Threads": "TRUE",
}


class PhotoCSVGenerator:
    """Generates Metricool CSV for photo posts."""

    def __init__(self, platforms: Optional[Dict[str, str]] = None):
        self.platforms = platforms or PHOTO_PLATFORMS.copy()

    def _build_row(
        self,
        post: PhotoPost,
        slot: Optional[PhotoSlot],
        drive_link: str,
    ) -> Dict[str, str]:
        """Build a single CSV row for a photo post."""
        row = {col: "" for col in GOLDEN_COLUMNS}

        # Core content. Metricool applies "First Comment Text" to Instagram
        # only, so hashtags parked there never reach TikTok/Twitter/Threads/
        # Pinterest — append them to the shared Text column, which every
        # platform reads. Twitter's 280-char cap governs when Twitter is on.
        text = post.caption or ""
        tags = (post.hashtags or "").strip()
        if tags and tags not in text:
            text = f"{text}\n\n{tags}".strip()
        text = truncate_for_platforms(
            text, twitter_enabled=str(self.platforms.get("Twitter", "")).upper() == "TRUE"
        )
        # LLM-derived text: neutralize spreadsheet formula injection (OWASP).
        row["Text"] = _escape_csv_cell(text)
        row["Date"] = slot.date if slot else ""
        row["Time"] = slot.time if slot else ""
        row["Draft"] = "FALSE"

        # Platform toggles
        for platform, enabled in self.platforms.items():
            if platform in row:
                row[platform] = enabled

        # Media link — Drive photo URL in the uc?export=download form. The
        # manifest holds the /view?usp=sharing share link, which returns
        # Drive's HTML viewer page to an unauthenticated pull, so TikTok/Meta
        # attach no image. Photos are far under the ~100 MB virus-scan
        # interstitial bound that keeps /view the global default.
        row["Picture Url 1"] = _drive_link_for_metricool(drive_link)

        # URL shortener
        row["Shortener"] = "FALSE"

        # Instagram: POST not REEL for photos
        row["Instagram Post Type"] = "POST"

        # Facebook: POST not REEL for photos
        row["Facebook Post Type"] = "POST"

        # YouTube: all empty (YouTube = FALSE)
        row["YouTube Video Title"] = ""
        row["YouTube Video Type"] = ""
        row["YouTube Video Privacy"] = ""
        row["YouTube video for kids"] = ""
        row["YouTube Video Category"] = ""
        row["YouTube Video Tags"] = ""

        # TikTok
        row["TikTok Title"] = _escape_csv_cell(post.tiktok_caption)[:90]
        row["TikTok disable comments"] = "no"
        row["TikTok disable duet"] = "no"
        row["TikTok disable stitch"] = "no"
        row["TikTok Post Privacy"] = "PUBLIC_TO_EVERYONE"
        row["TikTok Auto Add Music"] = "no"

        # Pinterest
        row["Pinterest Board"] = "Travel Photography"

        # First comment (engagement hook + hashtags)
        row["First Comment Text"] = _escape_csv_cell(post.build_first_comment())

        return row

    def generate(
        self,
        photo_posts: List[PhotoPost],
        slots: List[PhotoSlot],
        manifest: UploadManifest,
        output_dir: str,
        week_label: str = "",
        validate: bool = True,
        excel_compat: bool = False,
    ) -> Path:
        """Generate Metricool CSV for photo posts.

        Args:
            photo_posts: Photo post content
            slots: Scheduled time slots
            manifest: Upload manifest with Drive links
            output_dir: Output directory
            week_label: Label for filename (e.g., "2000-01-03")
            validate: Run golden reference validation
            excel_compat: Write a UTF-8 BOM for older Excel. Default False —
                Metricool is the real consumer and the BOM can break its
                column mapping (first header reads ``\\ufeffText``). Matches
                the shorts CSV default.

        Returns:
            Path to generated CSV
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"photo_metricool_{week_label}.csv" if week_label else "photo_metricool.csv"
        csv_path = output_dir / filename

        rows = []
        for i, post in enumerate(photo_posts):
            slot = slots[i] if i < len(slots) else None
            drive_link = manifest.files.get(post.photo_filename, "")

            if not drive_link:
                raise ValueError(
                    f"No Drive link for {post.photo_filename} — cannot generate valid Metricool CSV"
                )

            row = self._build_row(post, slot, drive_link)
            rows.append(row)

        # Schedule gate — upstream of the write. A post with no slot ships a
        # blank Date/Time, which Metricool imports as an immediate post rather
        # than on its planned drop (validate_csv only flags MALFORMED dates, so
        # a blank cell passed every prior check). Same gate the shorts CSV uses.
        _assert_metricool_rows_have_schedule(rows)

        encoding = "utf-8-sig" if excel_compat else "utf-8"
        with open(csv_path, "w", newline="", encoding=encoding) as f:
            writer = csv.DictWriter(f, fieldnames=GOLDEN_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        logger.info(f"Generated photo Metricool CSV: {csv_path} ({len(rows)} rows)")

        if validate and manifest.files:
            validation = validate_csv(csv_path)
            if not validation.valid:
                error_summary = "\n".join(validation.errors[:10])
                logger.error(f"CSV VALIDATION FAILED:\n{error_summary}")
                raise ValueError(
                    f"Photo CSV failed validation with {len(validation.errors)} errors. "
                    f"First: {validation.errors[0]}"
                )
            logger.info(f"CSV validated: {validation.row_count} rows match golden reference")

        return csv_path
