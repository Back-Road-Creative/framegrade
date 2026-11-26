"""A licensed music bed on Facebook and Instagram Shorts, as a 50/50 test.

GoMoveShift Shorts are rendered muted (``shorts_extractor``: ``-an``) so that no
cab conversation can ever be heard; that stays true. This module never touches
a Short's own audio: it only ever adds a licensed track to a Short that has NO
audio stream, and refuses any Short that has one. Long-form drives, Route Story
films (``pipeline.audio_profile.GmsAudioRefused``) and YouTube and TikTok are
untouched: the bed is added at dispatch, to the Facebook and Instagram upload
only, so the master on disk stays muted.

The test is declared in ``brand.yaml``::

    policy:
      shorts_music:
        start: 2026-11-27
        stop_after_days: 45
        platforms: [facebook, instagram]

While it runs, half of the Shorts (a stable hash of project and short index, so
both platforms agree and a retry agrees) get a track. Tracks live outside the
public tree, in ``<GMS_MUSIC_DIR>`` or, by default, ``music/<slug>/`` beside the
host brands dir (``GMS_BRANDS_DIR``). A track is used only when its exact file
has a complete entry in that folder's ``ledger.json`` -- the same ledger shape
as ``pipeline.audio_profile.LicenceLedger``: ``{file, title, source, licence,
url, added}``. With no licensed track, nothing changes and no Short is counted
in the test.

Every Short that actually got a track is written to ``music_receipts.jsonl`` in
the SEO storage dir, so the weekly growth review (``tracking/growth_review.py``)
compares only Shorts that really carried music with Shorts that were meant to
stay silent. Building the music copy never blocks a post: any failure posts the
muted master and writes no receipt.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess  # noqa: S404 -- fixed argv, no shell
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: The only platforms a bed may ever be added for.
MUSIC_PLATFORMS = frozenset({"facebook", "instagram"})
#: Ledger fields a track needs before it may be used.
LEDGER_FIELDS = ("file", "title", "source", "licence", "url", "added")
RECEIPTS_FILE = "music_receipts.jsonl"
#: Integrated loudness of the bed (the only audio in the upload).
TARGET_LUFS = -16.0
FADE_IN_S = 0.5
FADE_OUT_S = 1.5
_KEYS = frozenset({"start", "stop_after_days", "platforms"})


class MusicRefused(Exception):  # noqa: N818 -- a refusal, not an error
    """The Short may not get a bed (it has its own audio, or no track fits)."""


@dataclass(frozen=True)
class ShortsMusicTest:
    start: date
    stop_after_days: int
    platforms: Tuple[str, ...]

    def is_active(self, today: date) -> bool:
        return self.start <= today < self.start + timedelta(days=self.stop_after_days)


def parse_shorts_music(block: Any) -> Optional[ShortsMusicTest]:
    """Validate ``policy.shorts_music``; ``None`` when absent."""
    if block is None:
        return None
    if not isinstance(block, dict):
        raise ValueError(f"policy.shorts_music must be a mapping, got {block!r}")
    unknown = set(block) - _KEYS
    if unknown:
        raise ValueError(f"policy.shorts_music.{sorted(unknown)[0]} is not a recognized key")
    start = block.get("start")
    if isinstance(start, str):
        try:
            start = date.fromisoformat(start)
        except ValueError:
            pass
    if not isinstance(start, date):
        raise ValueError(
            f"policy.shorts_music.start must be an ISO date, got {block.get('start')!r}"
        )
    days = block.get("stop_after_days")
    if isinstance(days, bool) or not isinstance(days, int) or days < 1:
        raise ValueError(f"policy.shorts_music.stop_after_days must be an int >= 1, got {days!r}")
    platforms = block.get("platforms")
    if (
        not isinstance(platforms, (list, tuple))
        or not platforms
        or not set(platforms) <= MUSIC_PLATFORMS
    ):
        raise ValueError(
            f"policy.shorts_music.platforms must be a non-empty subset of {sorted(MUSIC_PLATFORMS)}"
        )
    return ShortsMusicTest(start, days, tuple(platforms))


def music_arm(project: str, short_index: int) -> bool:
    """True when this Short is in the music half -- stable, platform-independent."""
    digest = hashlib.sha256(f"shorts_music:{project}:{short_index}".encode()).digest()
    return digest[0] % 2 == 1


def music_dir(slug: str) -> Optional[Path]:
    """Where ``slug``'s licensed tracks and ``ledger.json`` live, or ``None``."""
    raw = os.environ.get("GMS_MUSIC_DIR")
    if raw and raw.strip():
        return Path(raw.strip())
    brands = os.environ.get("GMS_BRANDS_DIR")
    if brands and brands.strip():
        return Path(brands.strip()).parent / "music" / slug
    return None


@dataclass(frozen=True)
class Track:
    path: Path
    entry: Mapping[str, Any]


def licensed_tracks(folder: Optional[Path]) -> List[Track]:
    """Tracks with a complete ledger entry and a file on disk, sorted by file name."""
    if folder is None:
        return []
    ledger = folder / "ledger.json"
    if not ledger.is_file():
        return []
    try:
        entries = json.loads(ledger.read_text()).get("entries", [])
    except (OSError, ValueError, AttributeError) as exc:
        logger.warning("shorts music: unreadable ledger %s: %s", ledger, exc)
        return []
    out = []
    for entry in entries:
        if not isinstance(entry, dict) or any(not entry.get(k) for k in LEDGER_FIELDS):
            logger.warning("shorts music: ledger entry missing a licence field: %r", entry)
            continue
        name = str(entry["file"])
        if Path(name).name != name:
            logger.warning("shorts music: ledger file must be a bare name: %r", name)
            continue
        path = folder / name
        if path.is_file():
            out.append(Track(path, entry))
        else:
            logger.warning("shorts music: licensed track not on disk: %s", path)
    return sorted(out, key=lambda t: t.path.name)


def pick_track(tracks: Sequence[Track], project: str, short_index: int) -> Track:
    if not tracks:
        raise MusicRefused("no licensed track")
    digest = hashlib.sha256(f"shorts_music_track:{project}:{short_index}".encode()).digest()
    return tracks[int.from_bytes(digest[:4], "big") % len(tracks)]


def _probe_audio_streams(path: Path) -> Optional[int]:
    try:
        result = subprocess.run(  # noqa: S603 -- fixed argv
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=index",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        return len(json.loads(result.stdout or "{}").get("streams", []))
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _run_ffmpeg(argv: List[str]) -> None:
    subprocess.run(argv, capture_output=True, text=True, timeout=600, check=True)  # noqa: S603


def music_path(source: Path, track: Track) -> Path:
    tag = hashlib.sha256(track.path.name.encode()).hexdigest()[:8]
    return source.with_name(f"{source.stem}.music-{tag}.mp4")


def ffmpeg_argv(source: Path, track: Path, target: Path, duration_s: float) -> List[str]:
    fade_start = max(0.0, duration_s - FADE_OUT_S)
    audio = (
        f"loudnorm=I={TARGET_LUFS}:TP=-1.5:LRA=11,"
        f"afade=t=in:st=0:d={FADE_IN_S},"
        f"afade=t=out:st={fade_start:.3f}:d={FADE_OUT_S}"
    )
    return [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-stream_loop",
        "-1",
        "-i",
        str(track),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-af",
        audio,
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-t",
        f"{duration_s:.3f}",
        "-movflags",
        "+faststart",
        str(target),
    ]


def ensure_music_copy(
    source: Path,
    track: Track,
    duration_s: float,
    *,
    probe: Callable[[Path], Optional[int]] = _probe_audio_streams,
    run: Callable[[List[str]], None] = _run_ffmpeg,
) -> Path:
    """A sibling of the muted ``source`` with ``track`` as its only audio.

    Refuses (``MusicRefused``) a source with any audio stream, or one that
    cannot be probed: the bed is never mixed with, or laid over, real sound.
    """
    source = Path(source)
    streams = probe(source)
    if streams is None:
        raise MusicRefused(f"could not probe {source.name} for audio")
    if streams:
        raise MusicRefused(f"{source.name} has its own audio; a bed is only added to a muted Short")
    if duration_s <= 0:
        raise MusicRefused(f"{source.name} has no known duration")
    target = music_path(source, track)
    if target.is_file() and target.stat().st_size > 0:
        if target.stat().st_mtime >= source.stat().st_mtime:
            return target
    partial = target.with_name(target.stem + ".partial.mp4")
    run(ffmpeg_argv(source, track.path, partial, duration_s))
    if not partial.is_file() or partial.stat().st_size == 0:
        raise MusicRefused(f"music copy of {source.name} came out empty")
    partial.replace(target)
    return target


def write_receipt(storage_dir: Path, record: Any, platform: str, track: Track) -> None:
    line = {
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project": record.project_name,
        "short_index": int(record.short_index),
        "platform": platform,
        **{k: track.entry[k] for k in LEDGER_FIELDS},
    }
    try:
        storage_dir.mkdir(parents=True, exist_ok=True)
        with (storage_dir / RECEIPTS_FILE).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, sort_keys=True) + "\n")
    except OSError as exc:
        logger.warning("shorts music: could not write receipt: %s", exc)


def read_receipts(storage_dir: Path) -> set:
    """``{(project, short_index, platform)}`` that actually carried a track."""
    path = Path(storage_dir) / RECEIPTS_FILE
    out = set()
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(raw)
            out.add((row["project"], int(row["short_index"]), row["platform"]))
        except (ValueError, KeyError, TypeError):
            continue
    return out


def with_music(
    platform: str,
    path_for: Callable[[Any], Path],
    *,
    test: Optional[ShortsMusicTest],
    folder: Optional[Path],
    storage_dir: Path,
    today: Callable[[], date] = date.today,
    make: Callable[..., Path] = ensure_music_copy,
) -> Callable[[Any], bytes]:
    """Wrap a platform's media-path loader: the music copy for a Short in the
    music half while the test runs, the loader's own file otherwise."""

    def loader(record: Any) -> bytes:
        path = path_for(record)
        if (
            test is None
            or platform not in test.platforms
            or not test.is_active(today())
            or getattr(record, "format", None) not in (None, "short")
            or not music_arm(record.project_name, int(record.short_index))
        ):
            return path.read_bytes()
        try:
            track = pick_track(licensed_tracks(folder), record.project_name, record.short_index)
            copy = make(path, track, float(record.media.duration_seconds or 0))
        except (MusicRefused, OSError, subprocess.SubprocessError) as exc:
            logger.warning(
                "shorts music: %s short %s on %s posts muted: %s",
                record.project_name,
                record.short_index,
                platform,
                exc,
            )
            return path.read_bytes()
        data = copy.read_bytes()
        write_receipt(storage_dir, record, platform, track)
        return data

    return loader
