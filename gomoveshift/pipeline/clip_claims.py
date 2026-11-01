"""Footage-reuse registry: which camera files a published long-form was cut from.

``<seo storage>/gomoveshift/clip_claims.jsonl`` holds one line per (project, LF,
clip) the moment that LF has a YouTube ``video_id``. Phase A reads it before the
irreversible concat/encode and refuses a drive whose camera files another
project already published (``GMS_ALLOW_CLIP_REUSE=1`` overrides).

The writer never raises (it runs inside a state save). The reader raises
:class:`ClipClaimsUnreadable` on an existing-but-unreadable registry: a gate must
refuse rather than pass on a registry it could not verify.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from ..config import get_config
from .lf_provenance_sidecar import LF_PROVENANCE_FILENAME

logger = logging.getLogger(__name__)

ALLOW_CLIP_REUSE_ENV = "GMS_ALLOW_CLIP_REUSE"


class ClipClaimsUnreadable(RuntimeError):  # noqa: N818 -- name fixed by spec
    """The registry exists but could not be read; the gate must not pass on it."""


def claims_path() -> Path:
    return Path(get_config().seo.storage_path) / "gomoveshift" / "clip_claims.jsonl"


def _attr(obj, name):
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def load_claims() -> dict[str, list[dict]]:
    """Registry rows keyed by ``clip_name``; ``{}`` when no registry exists yet."""
    path = claims_path()
    if not path.exists():
        return {}
    claims: dict[str, list[dict]] = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    logger.warning("clip_claims: skipping malformed line in %s", path)
                    continue
                if isinstance(row, dict) and row.get("clip_name"):
                    claims.setdefault(row["clip_name"], []).append(row)
    except (OSError, UnicodeDecodeError) as exc:
        raise ClipClaimsUnreadable(f"clip claims registry unreadable at {path}: {exc}") from exc
    return claims


def record_claims(output_base: Path, uploads: dict) -> int:
    """Append a claim per source clip of every LF that has a ``video_id``.

    Idempotent on (project, lf_index, clip_name). Returns the rows written;
    never raises.
    """
    try:
        return _record_claims(Path(output_base), uploads)
    except Exception as exc:  # noqa: BLE001 -- runs inside a state save; must not fail it
        logger.warning("clip_claims: could not record claims for %s: %s", output_base, exc)
        return 0


def _record_claims(output_base: Path, uploads: dict) -> int:
    sidecar = output_base / LF_PROVENANCE_FILENAME
    if not sidecar.exists():
        return 0
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    by_index = {}
    for key, rec in (uploads or {}).items():
        try:
            by_index[int(key)] = rec
        except (TypeError, ValueError):
            continue
    project = output_base.name
    wanted = []
    for lf in payload.get("lfs") or ():
        idx = lf.get("lf_index")
        rec = by_index.get(idx)
        video_id = rec.get("video_id") if isinstance(rec, dict) else None
        if not video_id:
            continue
        for clip in (lf.get("source") or {}).get("clips") or ():
            if clip.get("name"):
                wanted.append((idx, video_id, clip))
    if not wanted:
        return 0
    existing = load_claims()
    seen = {
        (r.get("project"), r.get("lf_index"), name) for name, rows in existing.items() for r in rows
    }
    now = datetime.now(tz=timezone.utc).isoformat()
    lines = []
    for idx, video_id, clip in wanted:
        key = (project, idx, clip["name"])
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            json.dumps(
                {
                    "claimed_at": now,
                    "project": project,
                    "lf_index": idx,
                    "video_id": video_id,
                    "clip_name": clip["name"],
                    "clip_size_bytes": clip.get("size_bytes"),
                    "clip_path": clip.get("path"),
                },
                sort_keys=True,
            )
        )
    if not lines:
        return 0
    path = claims_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return len(lines)


def find_reused_clips(plan, project: str) -> list[dict]:
    """Chapters of ``plan`` another project already published.

    Same name and equal size (or a size unknown on either side) is a conflict;
    same name with a different size is a different file. ``filtered_clips`` are
    not in the drive and are ignored.
    """
    claims = load_claims()
    if not claims:
        return []
    found: dict[str, dict] = {}
    for ep in _attr(plan, "episodes") or ():
        for rec in _attr(ep, "recordings") or ():
            for ch in _attr(rec, "chapters") or ():
                path = _attr(ch, "path")
                if not path:
                    continue
                name = os.path.basename(str(path))
                rows = claims.get(name)
                if not rows:
                    continue
                size = _size(path)
                for row in rows:
                    if row.get("project") == project:
                        continue
                    claimed = row.get("clip_size_bytes")
                    if size is None or claimed is None or size == claimed:
                        found.setdefault(
                            name,
                            {
                                "clip_name": name,
                                "claimed_by": row.get("project"),
                                "lf_index": row.get("lf_index"),
                                "video_id": row.get("video_id"),
                            },
                        )
                        break
    return [found[k] for k in sorted(found)]


def _size(path) -> int | None:
    try:
        return os.stat(path).st_size
    except OSError:
        return None
