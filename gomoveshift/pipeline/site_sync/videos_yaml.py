"""Generate the Hugo site's ``data/videos.yaml`` library from a channel manifest.

The library is a *generated mirror* of the channel's real long-forms: every
``real_lf`` video in the manifest must appear, so no published long-form can be
missing. The UPSERT preserves human-authored ``title``/``summary`` and never
drops an existing entry (that would break an embedded link), so the file stays
safe to regenerate on every sync.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .._atomic_write import atomic_write_text
from .channel_manifest import classify

VIDEOS_FILENAME = "videos.yaml"


def _manifest_entry(video: dict) -> dict:
    """Build a fresh ``videos.yaml`` row from a manifest video."""
    return {
        "id": video["id"],
        "title": video.get("title", ""),
        "date": (video.get("published_at") or "")[:10],
        "duration_min": round((video.get("duration_seconds") or 0) / 60),
        "summary": "",
    }


def upsert_videos_yaml(manifest: dict, existing: dict | None) -> dict:
    """Mirror the manifest's ``real_lf`` videos into a ``{"videos": [...]}`` dict.

    UPSERT rules:
      * Existing entries are keyed by ``id`` (first wins; later dupes dropped).
      * A manifest id already present KEEPS its human ``title``/``summary``
        verbatim; ``date``/``duration_min`` are filled only when missing/empty.
      * A new manifest id is appended with ``summary: ""``.
      * Existing entries absent from the manifest are KEPT (never drop a video).

    The result is deduped by id and ordered by ``date`` descending (stable on
    ties).
    """
    result: dict[str, dict] = {}
    order: list[str] = []
    for video in (existing or {}).get("videos") or []:
        vid = video.get("id")
        if not vid or vid in result:
            continue
        result[vid] = dict(video)
        order.append(vid)

    real_lf: dict[str, dict] = {}
    for video in manifest.get("videos") or []:
        if video.get("category") != "real_lf":
            continue
        vid = video.get("id")
        if not vid or vid in real_lf:
            continue
        real_lf[vid] = video

    for vid, video in real_lf.items():
        generated = _manifest_entry(video)
        if vid in result:
            entry = result[vid]
            if not entry.get("date"):
                entry["date"] = generated["date"]
            if not entry.get("duration_min"):
                entry["duration_min"] = generated["duration_min"]
        else:
            result[vid] = generated
            order.append(vid)

    entries = [result[vid] for vid in order]
    entries.sort(key=lambda e: e.get("date") or "", reverse=True)
    return {"videos": entries}


def _duration_seconds(formatted: str) -> int:
    """Seconds from a journey ``duration`` string (``MM:SS`` / ``H:MM:SS``).

    Unparseable input returns 0, which :func:`classify` calls a Short — the
    safe direction, since a Short stays OUT of the long-form library.
    """
    seconds = 0
    for part in str(formatted).split(":"):
        if not part.strip().isdigit():
            return 0
        seconds = seconds * 60 + int(part.strip())
    return seconds


@dataclass
class VideoLibrary:
    """The site's ``data/videos.yaml``, held open across a journey merge.

    :func:`journal_publisher.apply_merge` takes one as a REQUIRED argument, so
    the embed and the library row are produced by the same call: a published
    long-form cannot be embedded into a journey the library never learned about
    (the site's journey-embed deploy check only warns about that). Membership is
    decided by the one predicate that owns it — :func:`classify` plus
    :func:`upsert_videos_yaml`'s ``real_lf`` filter — so the Shorts / streams /
    held clips a journey legitimately embeds still never reach the long-form
    mirror, and there is no second, unfiltered write path.

    ``writable=False`` is the dry-run form: the upsert still runs in memory but
    nothing is written, matching a caller that also leaves the journey unwritten.
    """

    path: Path
    writable: bool
    data: dict
    dirty: bool = False

    @classmethod
    def for_site(cls, site_path: Path, *, writable: bool) -> VideoLibrary:
        """Open ``<site_path>/data/videos.yaml`` (absent file → empty library)."""
        path = Path(site_path) / "data" / VIDEOS_FILENAME
        return cls(path=path, writable=writable, data=load_videos_yaml(path) or {"videos": []})

    @classmethod
    def for_journey(cls, journey_path: Path, *, writable: bool) -> VideoLibrary:
        """Open the library of the site that owns ``journey_path``.

        The replay/cron flush knows only the journey file, so the site root is
        derived from the layout every journey writer uses,
        ``<site>/content/journeys/<slug>.md``. A journey outside that layout
        falls back to its own directory rather than guessing at an unrelated
        tree.
        """
        p = Path(journey_path)
        canonical = p.parent.name == "journeys" and len(p.parents) > 2
        return cls.for_site(p.parents[2] if canonical else p.parent, writable=writable)

    def record_embed(self, *, video_id: str, title: str, date: str, duration: str) -> None:
        """UPSERT one embedded video — a no-op unless it is a real long-form.

        Routes a manifest-shaped row through :func:`upsert_videos_yaml`, so an
        existing row keeps its human-authored ``title``/``summary`` and a
        Short/stream is filtered out exactly as it is during ``sync-site``.
        """
        seconds = _duration_seconds(duration)
        # A journey only embeds a video once it is live (a still-scheduled upload
        # is withheld as a deferral), so privacy is public here and duration or
        # title decides the category.
        video = {
            "id": video_id,
            "title": title,
            "published_at": date,
            "duration_seconds": seconds,
            "privacy_status": "public",
            "category": classify(seconds, "public", title),
        }
        merged = upsert_videos_yaml({"videos": [video]}, self.data)
        if merged != self.data:
            self.data = merged
            self.dirty = True

    def flush(self) -> None:
        """Write the library iff it changed and this is not a dry run."""
        if self.dirty and self.writable:
            dump_videos_yaml(self.data, self.path)
            self.dirty = False


def load_videos_yaml(path: Path) -> dict | None:
    """Load an existing ``videos.yaml`` (``None`` when the file is absent)."""
    p = Path(path)
    if not p.is_file():
        return None
    return yaml.safe_load(p.read_text()) or None


def dump_videos_yaml(data: dict, path: Path) -> None:
    """Write the library back in the file's block-list style under ``videos:``."""
    atomic_write_text(
        Path(path),
        yaml.safe_dump(
            data,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        ),
    )
