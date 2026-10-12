"""On-disk cache of captured scene frames, keyed on pose/bundle/code identity.

A full 1080p30 film costs ~2.5h of headless-browser capture. Nothing upstream
of this module caches the *captured pixels* -- ``_terrain_tile_from_ancestor``
in ``capture.py`` only memoises DEM tile bytes in-process, and
``scene_renderer.py`` only memoises a feature count. So an unrelated change to
the dashboard, a still, or photo selection re-captures every map frame even
when the map render itself is unchanged.

A cache entry is keyed on everything that can change the rendered pixels or
labels: the bundle's identity (its manifest bytes, or an id/checksum field
inside the manifest when one exists), the scene code identity (this package's
``scene.js``/``index.html`` bytes plus the vendored MapLibre directory name --
so a scene.js edit or a MapLibre bump both miss instead of silently replaying
stale frames), the capture settings (readback format, JPEG quality, device
scale factor -- anything ``page.screenshot``/the browser context depends on),
the pose dict, the output size, and ``CACHE_VERSION`` (bump on any
format/behavior change so a stale on-disk entry from a prior version is never
read back).

Storage is two files per key -- ``<key>.png`` (lossless pixel round-trip) and
``<key>.json`` (labels) -- each written atomically (temp file in the same
cache dir, then ``os.replace``) because several capture workers run
concurrently in threads/processes. A missing, corrupt or truncated entry is
always treated as a miss, never raised. ``put`` never raises either: a full
disk or a permission error is logged once and the render carries on without
caching that frame -- a cache must never fail a multi-hour render.

Total cache size is bounded by ``GMS_ROUTE_STORY_FRAME_CACHE_MAX_GB`` (default
40): pruning deletes the oldest-mtime image+labels pairs until the directory
is back under the cap. It runs at construction and every ``_PRUNE_EVERY_N_PUTS``
puts (not every frame -- a full directory scan per frame would defeat the
point of caching), and a cache hit refreshes both files' mtime so a reused
entry is never the one pruning picks first.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ....paths import gms_cache_home

logger = logging.getLogger(__name__)

# Bump whenever the key derivation, storage format, or anything else that
# would make an old on-disk entry unsafe to read back changes.
CACHE_VERSION = 1

_DEFAULT_MAX_GB = 40.0
_PRUNE_EVERY_N_PUTS = 200

# This module lives in ``scene/`` beside the code whose identity it hashes.
_SCENE_DIR = Path(__file__).parent

# Lowest zlib compression that still shrinks a PNG meaningfully: this write is
# on the capture hot path (up to ~4,300 frames per film) and the default
# level 6 costs ~0.2-0.4s per 1080p frame that a render can't spare.
_CACHE_PNG_COMPRESS_LEVEL = 1


def cache_dir() -> Path:
    """Resolve the frame cache directory: ``GMS_INTERMEDIATE_DIR`` when set,
    else the route-story frame cache under the cache home
    (``paths.gms_cache_home``). Never under /tmp
    (tmpfs, wiped on reboot -- staging never lives there, see paths.py)."""
    env_override = os.environ.get("GMS_INTERMEDIATE_DIR", "")
    if env_override:
        return Path(env_override) / "route-story-frame-cache"
    return gms_cache_home() / "route-story" / "frame-cache"


def is_enabled() -> bool:
    """Opt-out via ``GMS_ROUTE_STORY_FRAME_CACHE=0``; on by default."""
    return os.environ.get("GMS_ROUTE_STORY_FRAME_CACHE", "1") != "0"


def _max_bytes() -> int:
    raw = os.environ.get("GMS_ROUTE_STORY_FRAME_CACHE_MAX_GB", "")
    try:
        gb = float(raw) if raw else _DEFAULT_MAX_GB
    except ValueError:
        logger.warning("ignoring unparseable GMS_ROUTE_STORY_FRAME_CACHE_MAX_GB=%r", raw)
        gb = _DEFAULT_MAX_GB
    return int(gb * (1024**3))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _bundle_identity(bundle_root: Path) -> str:
    """A hash of the raw manifest bytes. The manifest lists a sha256 of every
    bundle file (``bundle.write_manifest``), so any re-prepared tile changes
    this and misses the cache. A manifest ``id`` is never trusted instead: an
    id that outlives a content change would replay stale frames."""
    return hashlib.sha256((bundle_root / "manifest.json").read_bytes()).hexdigest()


def _scene_code_identity(scene_dir: Path) -> str:
    """Hash of this package's own ``scene.js``/``index.html`` bytes -- a scene
    code edit must miss every previously cached frame."""
    digest = hashlib.sha256()
    for name in ("scene.js", "index.html"):
        digest.update((scene_dir / name).read_bytes())
    return digest.hexdigest()


def _vendor_identity(bundle_root: Path) -> str:
    """The vendored MapLibre directory NAME (carries its version, e.g.
    ``maplibre-gl-6.10.0``), not its bytes -- cheap, and a version bump
    changes the directory name by construction (``vendor.ensure_maplibre``)."""
    vendor_root = bundle_root / "vendor"
    if not vendor_root.is_dir():
        return "no-vendor"
    return ",".join(sorted(entry.name for entry in vendor_root.iterdir() if entry.is_dir()))


class FrameCache:
    """Disk cache of captured frames for one bundle, scene build and capture config."""

    def __init__(
        self,
        bundle_root: Path,
        width: int,
        height: int,
        *,
        readback: str = "jpeg",
        jpeg_quality: int = 95,
        device_scale_factor: float = 1,
        cache_root: Path | None = None,
        scene_dir: Path | None = None,
        gl_backend: str | None = None,
    ) -> None:
        self.bundle_root = Path(bundle_root)
        self.width = width
        self.height = height
        self.root = Path(cache_root) if cache_root is not None else cache_dir()
        self._bundle_identity = _bundle_identity(self.bundle_root)
        self._scene_code_identity = _scene_code_identity(scene_dir or _SCENE_DIR)
        self._vendor_identity = _vendor_identity(self.bundle_root)
        self._capture_settings = {
            "readback": readback,
            "jpeg_quality": jpeg_quality,
            "device_scale_factor": device_scale_factor,
        }
        # Two rasterisers draw the same pose a hair differently (edge
        # antialiasing), so a film must never be stitched from both. Software is
        # left OUT of the key rather than named in it: every frame cached before
        # the backend was a choice was a software frame, and keeps its key.
        if gl_backend not in (None, "swiftshader"):
            self._capture_settings["gl_backend"] = gl_backend
        self._puts_since_prune = 0
        self._put_failure_logged = False
        self._prune_quietly()

    def _key_for(self, pose: dict[str, Any]) -> str:
        payload = {
            "cache_version": CACHE_VERSION,
            "bundle_identity": self._bundle_identity,
            "scene_code_identity": self._scene_code_identity,
            "vendor_identity": self._vendor_identity,
            "capture_settings": self._capture_settings,
            "pose": pose,
            "width": self.width,
            "height": self.height,
        }
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    def _image_path(self, key: str) -> Path:
        return self.root / f"{key}.png"

    def _labels_path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, pose: dict[str, Any]) -> tuple[np.ndarray, list[dict[str, Any]]] | None:
        key = self._key_for(pose)
        image_path = self._image_path(key)
        labels_path = self._labels_path(key)
        if not image_path.is_file() or not labels_path.is_file():
            return None
        try:
            with Image.open(image_path) as image:
                array = np.array(image.convert("RGB"), dtype=np.uint8)
            labels = json.loads(labels_path.read_text())
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(labels, list):
            return None
        self._touch(image_path, labels_path)
        return array, labels

    def _touch(self, image_path: Path, labels_path: Path) -> None:
        """A hit keeps its files fresh so pruning evicts unused entries first."""
        now = time.time()
        for path in (image_path, labels_path):
            try:
                os.utime(path, (now, now))
            except OSError:
                pass

    def put(
        self,
        pose: dict[str, Any],
        array: np.ndarray,
        labels: list[dict[str, Any]],
    ) -> None:
        """Store one frame. Never raises -- a cache write failure (disk full,
        permission denied) is logged once and the render carries on
        uncached rather than losing a multi-hour capture."""
        key = self._key_for(pose)
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            self._atomic_write_image(self._image_path(key), array)
            self._atomic_write_json(self._labels_path(key), labels)
        except OSError:
            if not self._put_failure_logged:
                logger.warning(
                    "route-story frame cache write failed under %s; rendering will "
                    "continue without caching (further failures are not logged)",
                    self.root,
                    exc_info=True,
                )
                self._put_failure_logged = True
            return
        self._puts_since_prune += 1
        if self._puts_since_prune >= _PRUNE_EVERY_N_PUTS:
            self._prune_quietly()

    def _atomic_write_image(self, path: Path, array: np.ndarray) -> None:
        tmp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        try:
            Image.fromarray(np.asarray(array, dtype=np.uint8), "RGB").save(
                tmp, format="PNG", compress_level=_CACHE_PNG_COMPRESS_LEVEL
            )
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    def _atomic_write_json(self, path: Path, value: Any) -> None:
        tmp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        try:
            tmp.write_text(json.dumps(value))
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    def _prune_quietly(self) -> None:
        """Enforce the byte cap. Never raises -- pruning is best-effort
        housekeeping, not something that should abort a render, and must
        tolerate another worker deleting an entry out from under it."""
        self._puts_since_prune = 0
        try:
            self._prune()
        except OSError:
            logger.warning(
                "route-story frame cache prune failed under %s", self.root, exc_info=True
            )

    def _prune(self) -> None:
        if not self.root.is_dir():
            return
        max_bytes = _max_bytes()
        entries: list[tuple[float, Path, Path, int]] = []
        for image_path in self.root.glob("*.png"):
            labels_path = self.root / f"{image_path.stem}.json"
            try:
                stat = image_path.stat()
            except OSError:
                continue  # another worker already removed it
            size = stat.st_size
            try:
                size += labels_path.stat().st_size
            except OSError:
                pass
            entries.append((stat.st_mtime, image_path, labels_path, size))
        total = sum(entry[3] for entry in entries)
        if total <= max_bytes:
            return
        entries.sort(key=lambda entry: entry[0])  # oldest mtime first
        for _mtime, image_path, labels_path, size in entries:
            if total <= max_bytes:
                break
            image_path.unlink(missing_ok=True)
            labels_path.unlink(missing_ok=True)
            total -= size
