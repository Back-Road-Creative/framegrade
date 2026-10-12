"""Read-only private project storage accounting; never a deletion policy."""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path

LINEAR_FILES = ("manifest.json", "sky_sum.npy", "sky_count.npy", "ground.npy")


def _safe(path: Path, warnings: list[str]) -> bool:
    if any(part.is_symlink() for part in (path, *path.parents)):
        warnings.append(f"Skipped symlink path: {path}")
        return False
    return True


def _bytes(path: Path, warnings: list[str]) -> int:
    try:
        if not _safe(path, warnings):
            return 0
        info = path.stat(follow_symlinks=False)
        return info.st_size if stat.S_ISREG(info.st_mode) else 0
    except OSError as exc:
        warnings.append(f"Cannot inspect {path}: {exc}")
        return 0


def _tree_bytes(root: Path, warnings: list[str], sources: set[Path]) -> int:
    total = 0

    def failed(exc):
        warnings.append(f"Cannot inspect attempt: {exc}")

    for directory, dirs, files in os.walk(root, followlinks=False, onerror=failed):
        base = Path(directory)
        if base.resolve() in sources:
            dirs[:] = []
            continue
        dirs[:] = [name for name in dirs if _safe(base / name, warnings)]
        total += sum(_bytes(base / name, warnings) for name in files)
    return total


def inventory_project(row: dict, workspace: Path) -> dict:
    """Report owned bytes and conservative review candidates without deleting anything.

    Caller must authenticate row ownership. All receipt-bearing attempts remain
    protected even if their receipt is malformed: recovery/validation owns that decision.
    """
    identifier = row.get("id")
    if not isinstance(identifier, str) or not re.fullmatch(r"[0-9a-f]{64}", identifier):
        raise ValueError("invalid project identifier")
    warnings: list[str] = []
    protected: dict[str, str] = {}

    def protect(path: Path | str, reason: str) -> None:
        protected.setdefault(str(path), reason)

    sources: set[Path] = set()
    source_bytes = 0
    source_raw_bytes = 0 if row.get("kind") == "night_products" else None
    for item in row.get("items", []):
        source = Path(item["source"])
        protect(source, "Referenced linear source; source media is never proposed.")
        if item.get("output"):
            protect(Path(item["output"]), "Current project output and its lineage.")
        if not _safe(source, warnings):
            continue
        source = source.resolve()
        if source in sources:
            continue
        sources.add(source)
        protect(source, "Referenced linear source; source media is never proposed.")
        source_bytes += sum(_bytes(source / name, warnings) for name in LINEAR_FILES)
    if row.get("kind") == "night_products":
        for item in row.get("inputs", {}).get("linear", []):
            source = Path(item["path"])
            protect(source, "Referenced linear source; source media is never proposed.")
            if not _safe(source, warnings):
                continue
            source = source.resolve()
            if source in sources:
                continue
            sources.add(source)
            source_bytes += sum(_bytes(source / name, warnings) for name in LINEAR_FILES)
        for kind in ("lights", "darks"):
            for item in row.get("inputs", {}).get(kind, []):
                source = Path(item["path"])
                protect(source, "Referenced RAW source; source media is never proposed.")
                if not _safe(source, warnings):
                    continue
                source = source.resolve()
                if source in sources:
                    continue
                sources.add(source)
                source_raw_bytes += _bytes(source, warnings)
        for task in row.get("tasks", []):
            if task.get("output"):
                protect(Path(task["output"]), "Current project output and its lineage.")
    result = {
        "source_linear_bytes": source_bytes,
        "source_raw_bytes": source_raw_bytes,
        "generated_bytes": 0,
        "protected_outputs": [],
        "cleanup_candidates": [],
        "warnings": warnings,
        "deletion_performed": False,
    }
    workspace = Path(workspace)
    project = workspace / identifier
    if not _safe(project, warnings):
        result["protected_outputs"] = sorted(protected)
        result["retention_proposal"] = _retention_proposal([], protected)
        return result
    resolved_project = project.resolve()
    protected = {str(Path(path).resolve()): reason for path, reason in protected.items()}
    if row.get("kind") == "night_products":
        for record in row.get("private_exports", []):
            for key in ("path", "receipt"):
                relative = Path(record.get(key, ""))
                candidate = project / relative
                if (
                    relative.is_absolute()
                    or not relative.parts
                    or ".." in relative.parts
                    or not _safe(candidate, warnings)
                    or not candidate.resolve().is_relative_to(resolved_project)
                ):
                    warnings.append(f"Invalid private export path excluded: {relative}")
                    continue
                protect(candidate.resolve(), "Selected private keeper and its receipt.")
    if not project.exists():
        warnings.append(f"Project output directory missing: {project}")
    else:
        for attempt in sorted(project.iterdir()):
            if not _safe(attempt, warnings):
                continue
            legacy = re.fullmatch(r"\d+-[0-9a-f]{32}", attempt.name)
            product = re.fullmatch(
                r"\d+-(?:best_stills|timelapse|star_trails|deep_still)-[0-9a-f]{32}",
                attempt.name,
            )
            export_dir = attempt.name == "exports" and row.get("kind") == "night_products"
            if not attempt.is_dir() or not (legacy or product or export_dir):
                warnings.append(f"Unrecognized project entry excluded: {attempt}")
                continue
            resolved = attempt.resolve()
            if not resolved.is_relative_to(resolved_project):
                warnings.append(f"Attempt escapes project: {attempt}")
                continue
            if export_dir:
                result["generated_bytes"] += _tree_bytes(attempt, warnings, sources)
                if any(Path(path).is_relative_to(resolved) for path in protected):
                    protect(resolved, "Contains a selected private keeper.")
                continue
            count = _tree_bytes(attempt, warnings, sources)
            result["generated_bytes"] += count
            # Treat even an unreadable/dangling receipt as potentially recoverable.
            recoverable = any(
                os.path.lexists(attempt / name)
                for name in ("receipt.json", "preview-pending.json", "product-receipt.json")
            )
            if recoverable or row.get("status") not in {
                "complete",
                "failed",
                "partial",
                "interrupted",
                "paused",
                "blocked",
            }:
                reason = (
                    "Receipt-bearing attempt retained for validation or recovery."
                    if recoverable
                    else "Active project attempt; terminal state is required before review."
                )
                protect(resolved, reason)
            if any(Path(path).is_relative_to(resolved) for path in protected):
                protect(resolved, "Contains a protected source or current output.")
            if str(resolved) not in protected:
                result["cleanup_candidates"].append(
                    {
                        "path": str(resolved),
                        "bytes": count,
                        "reason": "Unreferenced incomplete attempt; review only.",
                    }
                )
    result["protected_outputs"] = sorted(protected)
    result["retention_proposal"] = _retention_proposal(result["cleanup_candidates"], protected)
    return result


def _retention_proposal(candidates: list[dict], protected: dict[str, str]) -> dict:
    """Summarize dependencies for review; this module has no deletion operation."""
    return {
        "mode": "review_only",
        "deletion_authorized": False,
        "candidate_count": len(candidates),
        "reclaimable_bytes": sum(item["bytes"] for item in candidates),
        "candidates": list(candidates),
        "protected": [
            {"path": path, "reason": reason} for path, reason in sorted(protected.items())
        ],
    }
