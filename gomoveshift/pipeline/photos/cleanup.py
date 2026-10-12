"""Photo artifact cleanup — safe removal of photo pipeline intermediates.

Mirrors the three-phase discover → plan → execute pattern from
pipeline/cleanup.py with photo-specific artifact types and safety gates.

Usage:
    from gomoveshift.pipeline.photos.cleanup import PhotoCleanup

    out = Path("/path/to/photo-output")  # the directory the photo pipeline writes to
    cleanup = PhotoCleanup(
        caption_dir=out / "photo_captions",
        metricool_dir=out / "photo_metricool",
        resized_dir=out / "photo_resized",
    )
    batches = cleanup.discover_batches()
    plan = cleanup.plan_cleanup(batches, min_age_days=14)
    # Inspect plan.files and plan.skipped before executing
    cleanup.execute_plan(plan)
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# Default paths: the three artifact directories under the general output dir.
# ``PhotoOrchestrator`` passes its own photo output dir explicitly instead.
#
# Kept as ``None`` sentinels at import time and resolved via ``gms_output_dir()``
# at constructor call time, so the environment is read when a cleanup is built,
# not when the module is imported.
DEFAULT_CAPTION_DIR: Optional[Path] = None
DEFAULT_METRICOOL_DIR: Optional[Path] = None
DEFAULT_RESIZED_DIR: Optional[Path] = None


@dataclass
class PhotoCleanupPlan:
    """Plan describing photo artifacts to delete and items skipped with reasons."""

    files: List[Dict] = field(default_factory=list)  # [{"path": str, "size": int, "reason": str}]
    skipped: List[Dict] = field(default_factory=list)  # [{"path": str, "reason": str}]


class PhotoCleanup:
    """Safe cleanup of photo pipeline artifacts.

    Safety rules:
    - Dry-run by default (--confirm required)
    - Age-based retention (default 14 days)
    - Caption YAML only deleted if reviewed: true AND a CSV exists (published)
    - Resized intermediates only deleted if CSV exists for that week
    - Drive manifests only deleted if completed: true — on every path that can
      reach one, including inside a resized week dir (upload resume record)
    - NEVER touches: source photos in Hugo site, Drive files, content ledger
    """

    def __init__(
        self,
        caption_dir: Optional[Path] = None,
        metricool_dir: Optional[Path] = None,
        resized_dir: Optional[Path] = None,
    ):
        # Bare defaults sit under the general output dir (``gms_output_dir()``,
        # overridable with GMS_OUTPUT_DIR), resolved at call time rather than baked
        # into import-time defaults. They differ from the photo output dir
        # ``PhotoOrchestrator`` writes to, so a caller cleaning that output passes
        # its directories explicitly.
        from ...paths import gms_output_dir

        output_base = gms_output_dir()
        self.caption_dir = caption_dir or (output_base / "photo_captions")
        self.metricool_dir = metricool_dir or (output_base / "photo_metricool")
        self.resized_dir = resized_dir or (output_base / "photo_resized")

    def discover_batches(self) -> List[str]:
        """Find week labels with photo pipeline artifacts.

        Scans caption YAML files, Metricool CSVs, and resized directories
        to build a union of all week labels that have artifacts.

        Returns:
            Sorted list of week labels (YYYY-MM-DD strings)
        """
        weeks = set()

        # Caption YAMLs: {week}_captions.yaml
        if self.caption_dir.exists():
            for f in self.caption_dir.glob("*_captions.yaml"):
                week = f.name.replace("_captions.yaml", "")
                weeks.add(week)

        # Metricool CSVs: photo_metricool_{week}.csv
        if self.metricool_dir.exists():
            for f in self.metricool_dir.glob("photo_metricool_*.csv"):
                week = f.name.replace("photo_metricool_", "").replace(".csv", "")
                weeks.add(week)

        # Resized directories: {week}/
        if self.resized_dir.exists():
            for d in self.resized_dir.iterdir():
                if d.is_dir():
                    weeks.add(d.name)

        return sorted(weeks)

    def _file_age_days(self, path: Path) -> float:
        """Get age of a file/directory in days."""
        return (time.time() - path.stat().st_mtime) / 86400

    def _caption_is_reviewed(self, caption_path: Path) -> bool:
        """Check if a caption YAML has reviewed: true."""
        try:
            data = yaml.safe_load(caption_path.read_text())
            return data.get("reviewed", False) is True
        except Exception:
            return False

    def _manifest_is_completed(self, manifest_path: Path) -> bool:
        """Check if a Drive manifest has completed: true."""
        try:
            data = json.loads(manifest_path.read_text())
            return data.get("completed", False) is True
        except Exception:
            return False

    def _csv_exists_for_week(self, week: str) -> bool:
        """Check if a Metricool CSV exists for this week (publish completed)."""
        csv_path = self.metricool_dir / f"photo_metricool_{week}.csv"
        return csv_path.exists()

    def plan_cleanup(
        self,
        batches: List[str],
        min_age_days: int = 14,
    ) -> PhotoCleanupPlan:
        """Build a cleanup plan listing files to delete with reasons.

        Args:
            batches: Week labels to consider for cleanup
            min_age_days: Minimum age in days before cleanup (default: 14)

        Returns:
            PhotoCleanupPlan with files (to delete) and skipped (with reasons)
        """
        plan = PhotoCleanupPlan()

        for week in batches:
            # 1. Caption YAML caches
            caption_path = self.caption_dir / f"{week}_captions.yaml"
            if caption_path.exists():
                age = self._file_age_days(caption_path)
                if age < min_age_days:
                    plan.skipped.append(
                        {
                            "path": str(caption_path),
                            "reason": f"Too recent ({age:.1f} days < {min_age_days} day minimum)",
                        }
                    )
                elif not self._caption_is_reviewed(caption_path):
                    plan.skipped.append(
                        {
                            "path": str(caption_path),
                            "reason": "Caption YAML not reviewed (reviewed: false)",
                        }
                    )
                elif not self._csv_exists_for_week(week):
                    # Reviewed is not published. Photos are marked posted at
                    # selection, so dropping a never-published batch loses it
                    # unrecoverably — require CSV evidence that it shipped.
                    plan.skipped.append(
                        {
                            "path": str(caption_path),
                            "reason": "No CSV for this week (publish not complete)",
                        }
                    )
                else:
                    plan.files.append(
                        {
                            "path": str(caption_path),
                            "size": caption_path.stat().st_size,
                            "reason": "Caption YAML cache (reviewed, publish complete)",
                        }
                    )

            # 2. Metricool CSVs
            csv_path = self.metricool_dir / f"photo_metricool_{week}.csv"
            if csv_path.exists():
                age = self._file_age_days(csv_path)
                if age < min_age_days:
                    plan.skipped.append(
                        {
                            "path": str(csv_path),
                            "reason": f"Too recent ({age:.1f} days < {min_age_days} day minimum)",
                        }
                    )
                else:
                    plan.files.append(
                        {
                            "path": str(csv_path),
                            "size": csv_path.stat().st_size,
                            "reason": "Metricool CSV (publish complete)",
                        }
                    )

            # 3. Resized intermediates — gate: CSV must exist for this week
            resized_dir = self.resized_dir / week
            if resized_dir.exists() and resized_dir.is_dir():
                age = self._file_age_days(resized_dir)
                if age < min_age_days:
                    plan.skipped.append(
                        {
                            "path": str(resized_dir),
                            "reason": f"Too recent ({age:.1f} days < {min_age_days} day minimum)",
                        }
                    )
                elif not self._csv_exists_for_week(week):
                    plan.skipped.append(
                        {
                            "path": str(resized_dir),
                            "reason": "No CSV for this week (publish not complete)",
                        }
                    )
                else:
                    for f in sorted(resized_dir.iterdir()):
                        if not f.is_file():
                            continue
                        if f.name == "photo_gdrive_manifest.json":
                            # The Drive manifest is the upload resume record.
                            # It is planned (or kept) solely by the manifest
                            # scan below, which honors the completed gate —
                            # the CSV gate alone must never delete it.
                            continue
                        plan.files.append(
                            {
                                "path": str(f),
                                "size": f.stat().st_size,
                                "reason": "Resized intermediate (CSV verified)",
                            }
                        )

        # 4. Drive manifests — scan source dirs for photo_gdrive_manifest.json
        #    These live alongside source photos, so we find them by glob
        for manifest_dir in [self.caption_dir.parent]:
            if not manifest_dir.exists():
                continue
            for manifest_path in manifest_dir.rglob("photo_gdrive_manifest.json"):
                age = self._file_age_days(manifest_path)
                if age < min_age_days:
                    plan.skipped.append(
                        {
                            "path": str(manifest_path),
                            "reason": f"Too recent ({age:.1f} days < {min_age_days} day minimum)",
                        }
                    )
                elif not self._manifest_is_completed(manifest_path):
                    plan.skipped.append(
                        {
                            "path": str(manifest_path),
                            "reason": "Manifest not completed (upload may be in progress)",
                        }
                    )
                else:
                    plan.files.append(
                        {
                            "path": str(manifest_path),
                            "size": manifest_path.stat().st_size,
                            "reason": "Drive manifest (completed: true)",
                        }
                    )

        return plan

    def execute_plan(self, plan: PhotoCleanupPlan) -> int:
        """Execute a cleanup plan by deleting the listed files.

        Returns the number of files deleted.
        """
        deleted = 0
        for f in plan.files:
            path = Path(f["path"])
            try:
                if path.exists():
                    size_kb = f["size"] / 1024
                    logger.info(f"  Deleting: {path} ({size_kb:.1f} KB)")
                    path.unlink()
                    deleted += 1
            except Exception as e:
                logger.warning(f"  Failed to delete {path}: {e}")

        # Clean up empty resized week directories
        if self.resized_dir.exists():
            for d in self.resized_dir.iterdir():
                if d.is_dir() and not any(d.iterdir()):
                    try:
                        d.rmdir()
                        logger.info(f"  Removed empty directory: {d}")
                    except Exception:
                        pass

        return deleted
