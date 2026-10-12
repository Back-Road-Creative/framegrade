"""Vendor MapLibre GL JS into a Route Story bundle, offline and verified.

``ensure_maplibre`` never reaches the network itself — the caller injects a
``fetch(url) -> bytes`` callable — and it verifies the downloaded tarball
against a pinned SRI hash *before* extracting anything, so a corrupted or
tampered download writes nothing to disk.
"""

from __future__ import annotations

import hashlib
import io
import tarfile
from base64 import b64encode
from collections.abc import Callable
from pathlib import Path

MAPLIBRE_VERSION = "6.10.0"
MAPLIBRE_URL = f"https://registry.npmjs.org/maplibre-gl/-/maplibre-gl-{MAPLIBRE_VERSION}.tgz"
# npm dist.integrity for maplibre-gl@6.10.0 (recorded when the version was pinned).
MAPLIBRE_SRI = "sha512-pZIJ1/rsB1jDjT3eBTfMaeSY45lRKdCuZtqXqTnkUrn0UJO9IwtbywT5ym3/cz57MGEELwAwQSYjB/Fj9TlktQ=="  # pragma: allowlist secret (public npm SRI hash)

# MapLibre 6.x ships ES modules only (package.json "type": "module"); the
# main module loads its worker from ./maplibre-gl-worker.mjs next to itself.
# Only these tarball members are ever extracted. Fixed, exact target
# names make path traversal structurally impossible: a member whose name
# does not match exactly is simply skipped, regardless of what "..' or
# absolute-path tricks it contains.
_DIST_MEMBERS = (
    "package/dist/maplibre-gl.mjs",
    "package/dist/maplibre-gl-shared.mjs",
    "package/dist/maplibre-gl-worker.mjs",
    "package/dist/maplibre-gl.css",
)
_SHA256SUMS_NAME = "SHA256SUMS"


def _verify_sri(data: bytes, sri: str) -> None:
    algorithm, separator, expected_b64 = sri.partition("-")
    if algorithm != "sha512" or not separator or not expected_b64:
        raise ValueError(f"unsupported SRI hash format: {sri!r}")
    actual_b64 = b64encode(hashlib.sha512(data).digest()).decode("ascii")
    if actual_b64 != expected_b64:
        raise ValueError("maplibre-gl tarball failed SRI verification")


def _filenames() -> list[str]:
    return [Path(name).name for name in _DIST_MEMBERS]


def _write_sha256sums(dest_dir: Path, filenames: list[str]) -> None:
    lines = [
        f"{hashlib.sha256((dest_dir / name).read_bytes()).hexdigest()}  {name}"
        for name in sorted(filenames)
    ]
    (dest_dir / _SHA256SUMS_NAME).write_text("\n".join(lines) + "\n")


def _matches_recorded_hashes(dest_dir: Path, filenames: list[str]) -> bool:
    sums_path = dest_dir / _SHA256SUMS_NAME
    if not sums_path.is_file():
        return False
    recorded: dict[str, str] = {}
    for line in sums_path.read_text().splitlines():
        if not line.strip():
            continue
        digest, _, name = line.partition("  ")
        recorded[name] = digest
    for name in filenames:
        path = dest_dir / name
        if not path.is_file():
            return False
        if recorded.get(name) != hashlib.sha256(path.read_bytes()).hexdigest():
            return False
    return True


def ensure_maplibre(
    bundle_root: Path,
    fetch: Callable[[str], bytes],
    *,
    version: str = MAPLIBRE_VERSION,
    url: str = MAPLIBRE_URL,
    sri: str = MAPLIBRE_SRI,
) -> Path:
    """Ensure ``<bundle_root>/vendor/maplibre-gl-<version>/`` holds a verified build.

    Idempotent: if the destination already holds files matching a recorded
    ``SHA256SUMS``, ``fetch`` is never called again.
    """
    dest_dir = Path(bundle_root) / "vendor" / f"maplibre-gl-{version}"
    filenames = _filenames()
    if dest_dir.is_dir() and _matches_recorded_hashes(dest_dir, filenames):
        return dest_dir

    data = fetch(url)
    _verify_sri(data, sri)

    extracted: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.name not in _DIST_MEMBERS:
                continue
            if not member.isfile():
                raise ValueError(f"expected a regular file for {member.name!r}")
            extracted_file = tar.extractfile(member)
            if extracted_file is None:
                raise ValueError(f"could not read {member.name!r} from tarball")
            extracted[Path(member.name).name] = extracted_file.read()

    missing = set(filenames) - set(extracted)
    if missing:
        raise ValueError(f"maplibre-gl tarball missing expected files: {sorted(missing)}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    for name, content in extracted.items():
        (dest_dir / name).write_bytes(content)
    _write_sha256sums(dest_dir, filenames)
    return dest_dir
