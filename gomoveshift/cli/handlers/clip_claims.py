"""Handler for ``gomoveshift clip-claims backfill``.

Rebuilds the footage-reuse registry (``pipeline/clip_claims.py``) from past
outputs, which predate it. Each project folder directly under a scanned root (or
the root itself) is one project. Its source clips come from ``lf_provenance.json``
and the "published" rule is the live one: an LF counts only when
``youtube_uploads.json`` (``youtube_sidecar.load_sidecar``) gives it a
``video_id``. Outputs from before ``source.clips`` existed (no provenance file, or
LFs without clips) fall back to the camera files their ``drive.log`` names: no log
line ties a clip to one LF, so every uploaded LF claims the whole project's clips,
tagged ``basis: drive_log_project`` (coarse). A folder with neither is reported
skipped, never guessed. Dry-run by default; ``--apply`` appends via ``record_claims``, which is
idempotent, so a re-run adds nothing.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from ...pipeline.clip_claims import (
    ClipClaimsUnreadable,
    claims_path,
    pending_claims,
    record_claims,
)
from ...pipeline.lf_provenance_sidecar import LF_PROVENANCE_FILENAME
from ...pipeline.youtube_sidecar import load_sidecar

#: Camera files a drive.log names: Insta360 ``VID_<date>_<time>_<n>_<n>`` and GoPro
#: ``GX``/``GH`` chapters (the log shows stems, e.g. ``GX010064_gopro.gpx``).
_CLIP_RE = re.compile(r"(?<![A-Za-z0-9])(VID_\d{8}_\d{6}_\d+_\d+|G[XH]\d{6})(?![0-9])")


def _drive_log_clips(base: Path) -> tuple[str, ...]:
    try:
        text = (base / "drive.log").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ()
    return tuple(sorted(set(_CLIP_RE.findall(text))))


def _project_dirs(root: Path) -> list[Path]:
    if (root / LF_PROVENANCE_FILENAME).exists():
        return [root]
    return sorted(p for p in root.iterdir() if p.is_dir())


def run_clip_claims_backfill(*, roots, apply: bool, out=None, err=None) -> int:
    """Returns the CLI exit code (2 for an unusable root or registry, else 0)."""
    out = out if out is not None else sys.stdout
    err = err if err is not None else sys.stderr
    roots = [Path(r) for r in roots]
    bad = [r for r in roots if not r.is_dir()]
    if bad:
        for r in bad:
            print(f"error: not a directory: {r}", file=err)
        return 2
    added = 0
    skipped = 0
    try:
        for root in roots:
            for base in _project_dirs(root):
                has_prov = (base / LF_PROVENANCE_FILENAME).exists()
                log_clips = _drive_log_clips(base)
                if not has_prov and not log_clips:
                    print(
                        f"skip {base}: no {LF_PROVENANCE_FILENAME} and no camera clips in drive.log",
                        file=out,
                    )
                    skipped += 1
                    continue
                uploads = load_sidecar(base)
                try:
                    rows = pending_claims(base, uploads, log_clips)
                except (OSError, ValueError) as exc:
                    print(f"skip {base}: unreadable {LF_PROVENANCE_FILENAME} ({exc})", file=out)
                    skipped += 1
                    continue
                if not rows:
                    print(f"skip {base}: nothing new (unpublished or already claimed)", file=out)
                    skipped += 1
                    continue
                for r in rows:
                    print(
                        f"{'add' if apply else 'would add'} {base.name} lf{r['lf_index']} "
                        f"{r['clip_name']} {r['video_id']} ({r['basis']})",
                        file=out,
                    )
                added += len(rows) if not apply else record_claims(base, uploads, log_clips)
    except ClipClaimsUnreadable as exc:
        print(f"error: {exc}", file=err)
        return 2
    verb = "added" if apply else "would add"
    print(
        f"clip-claims backfill: {verb} {added}, skipped {skipped} folder(s) -> {claims_path()}",
        file=out,
    )
    if not apply:
        print("(dry-run; pass --apply to write the registry)", file=out)
    return 0


def cmd_clip_claims(args, config):
    """Argparse entry point for ``gomoveshift clip-claims``."""
    return run_clip_claims_backfill(
        roots=args.roots,
        apply=bool(getattr(args, "apply", False)),
    )
