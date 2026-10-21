#!/usr/bin/env python3
"""Generate docs/outbound.md, a verified inventory of every outbound host
this service's code can name, derived from the source itself.

Why: `docs/pipeline/84-network-allowlist-credential-stores.md` is a hand-maintained
prose allowlist. It can drift from the code (see `tests/test_network_allowlist_doc_sync.py`).
This module scans every `.py` file under `gomoveshift/` for string literals that name
a host -- either a full `scheme://host/...` URL, or a bare hostname assigned to a
constant named `*_HOST`/`*_HOSTS` -- and requires each one to be catalogued in
`INVENTORY` below. A host the scan finds that isn't in `INVENTORY` fails the check
loudly, naming the file:line, instead of silently shipping undocumented egress.

Usage:
    python3 tools/build_outbound_hosts.py            # write docs/outbound.md
    python3 tools/build_outbound_hosts.py --check     # exit 1 if stale/edited by hand
"""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

SVC_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = SVC_ROOT / "gomoveshift"
DOC_PATH = SVC_ROOT / "docs" / "outbound.md"

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}

_HOST_RE = re.compile(
    r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)+$", re.IGNORECASE
)
_HOST_CONST_NAME_RE = re.compile(r"_HOSTS?$")


@dataclass(frozen=True)
class HostSite:
    host: str
    path: str  # relative to SVC_ROOT
    lineno: int


def _valid_host(candidate: str) -> bool:
    return bool(_HOST_RE.match(candidate))


_URL_SUBSTRING_RE = re.compile(r"https?://[^\s\"'<>]+")


def _host_from_url_literal(value: str) -> str | None:
    """The host of the first ``scheme://...`` URL embedded anywhere in a string
    literal -- not just when the URL is the whole literal (e.g. a hint message
    like ``"...JSON from https://console.cloud.google.com/apis/credentials"``
    or an f-string's static prefix before the interpolated part)."""
    match = _URL_SUBSTRING_RE.search(value)
    if not match:
        return None
    hostname = urlsplit(match.group(0)).hostname
    if hostname and _valid_host(hostname):
        return hostname.lower()
    return None


def _scan_file(path: Path) -> list[HostSite]:
    rel = str(path.relative_to(SVC_ROOT))
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
    except SyntaxError:
        return []
    sites: list[HostSite] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            host = _host_from_url_literal(node.value)
            if host:
                sites.append(HostSite(host, rel, node.lineno))

        # Bare hostnames assigned to a constant named ``*_HOST``/``*_HOSTS``
        # (e.g. ``GNIS_HOST = "prd-tnm.s3.amazonaws.com"`` or
        # ``ALLOWED_UPLOAD_HOSTS = frozenset({"graph.facebook.com", ...})``).
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
            value = node.value
        else:
            continue
        names = [t.id for t in targets if isinstance(t, ast.Name)]
        if not any(_HOST_CONST_NAME_RE.search(n) for n in names):
            continue
        for sub in ast.walk(value):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                candidate = sub.value.lower()
                if "://" not in candidate and _valid_host(candidate):
                    sites.append(HostSite(candidate, rel, sub.lineno))

    return sites


def scan_hosts() -> list[HostSite]:
    """Every outbound-looking host literal in gomoveshift/**/*.py, loopback excluded."""
    sites: list[HostSite] = []
    for path in sorted(CODE_DIR.rglob("*.py")):
        sites.extend(_scan_file(path))
    seen: dict[str, HostSite] = {}
    for site in sites:
        if site.host in LOOPBACK_HOSTS:
            continue
        # Keep the first (alphabetically-earliest-file) site per host for a
        # stable, single file:line citation.
        seen.setdefault(site.host, site)
    return sorted(seen.values(), key=lambda s: s.host)


# host -> (purpose, feature, direction, mode)
#   direction: "media/data out" (media, credentials, or location/query text leave the
#   workstation) or "GET-only" (a fixed request for public data; nothing of ours is sent)
#   mode: "default" (runs on a normal invocation) or "opt-in" (a flag/extra is required)
INVENTORY: dict[str, tuple[str, str, str, str]] = {
    "accounts.google.com": (
        "OAuth2 consent page the operator's browser opens during `gomoveshift youtube-auth`"
        " (the client config built from the shared token names it as auth_uri)",
        "YouTube upload/comment consent flow",
        "media/data out",
        "opt-in",
    ),
    "console.cloud.google.com": (
        "Setup hint text printed to the operator (not fetched by this service)",
        "Drive/YouTube credential setup instructions",
        "GET-only",
        "default",
    ),
    "copernicus-dem-30m.s3.amazonaws.com": (
        "Copernicus GLO-30 DEM fallback bucket",
        "Route Story offline map bundle",
        "GET-only",
        "default",
    ),
    "drive.google.com": (
        "Drive share-link parsing and thumbnail/CSV link building",
        "Shorts/thumbnail hosting behind the Metricool CSV",
        "GET-only",
        "default",
    ),
    "elevation.nationalmap.gov": (
        "USGS 3DEP elevation for the trip bounding box",
        "Route Story offline map bundle",
        "GET-only",
        "default",
    ),
    "gibs.earthdata.nasa.gov": (
        "NASA GIBS imagery fallback",
        "Route Story offline map bundle",
        "GET-only",
        "default",
    ),
    "github.com": (
        "One-time face-detector model fetch, SHA256-verified",
        "Privacy blur",
        "GET-only",
        "default",
    ),
    "gomoveshift.com": (
        "Confirms a just-published photo URL is live before handing it to Meta",
        "Photo publishing / brand site",
        "GET-only",
        "default",
    ),
    "graph.facebook.com": (
        "Meta Reels create/finish container calls and publishing-limit reads; "
        "also a video-bytes upload target",
        "Meta (Facebook/Instagram) Reels publishing",
        "media/data out",
        "opt-in",
    ),
    "huggingface.co": (
        "One-time plate/vehicle-occupant detector model fetches, SHA256-verified",
        "Privacy blur",
        "GET-only",
        "default",
    ),
    "imagery.nationalmap.gov": (
        "USGS NAIP orthoimagery for the trip bounding box",
        "Route Story offline map bundle",
        "GET-only",
        "default",
    ),
    "nominatim.openstreetmap.org": (
        "Reverse geocoding (GPS to place) and forward geocoding (operator "
        "location string to coordinates); sends GPS coordinates or the sanitized "
        "operator string",
        "Location resolution",
        "media/data out",
        "default",
    ),
    "ntfy.sh": (
        "Phone push notifications (TikTok inbox drafts with their caption, failed posts, "
        "missed slots); sends the notification title/body incl. captions, no media, no "
        "credentials",
        "Phone alerts (`console/notify.py`)",
        "media/data out",
        "opt-in",
    ),
    "oauth2.googleapis.com": (
        "OAuth2 token refresh (sends the client id/secret and refresh token)",
        "YouTube upload auth",
        "media/data out",
        "default",
    ),
    "open-upload.tiktokapis.com": (
        "TikTok Direct Post video upload target",
        "TikTok publishing",
        "media/data out",
        "opt-in",
    ),
    "open-upload.tiktokapis.us": (
        "TikTok Direct Post video upload target, US data-residency namespace",
        "TikTok publishing",
        "media/data out",
        "opt-in",
    ),
    "open.tiktokapis.com": (
        "TikTok OAuth token, creator_info, publish/init and publish/status "
        "API calls (not the upload byte stream itself)",
        "TikTok publishing",
        "media/data out",
        "opt-in",
    ),
    "overpass-api.de": (
        "POI/chapter discovery bbox query (Overpass mirror)",
        "Chapter generation",
        "GET-only",
        "default",
    ),
    "overpass.kumi.systems": (
        "POI/chapter discovery bbox query (Overpass canonical host)",
        "Chapter generation",
        "GET-only",
        "default",
    ),
    "overpass.private.coffee": (
        "POI/chapter discovery bbox query (Overpass mirror)",
        "Chapter generation",
        "GET-only",
        "default",
    ),
    "prd-tnm.s3.amazonaws.com": (
        "USGS GNIS Domestic Names zip, per state",
        "Route Story place-name labels",
        "GET-only",
        "default",
    ),
    "registry.npmjs.org": (
        "Pinned MapLibre tarball, SHA512 SRI-verified",
        "Route Story scene rendering",
        "GET-only",
        "default",
    ),
    "rupload.facebook.com": (
        "Meta Reels video-bytes upload target",
        "Meta (Facebook/Instagram) Reels publishing",
        "media/data out",
        "opt-in",
    ),
    "studio.youtube.com": (
        "Endscreen setup hint text printed to the operator (not fetched)",
        "YouTube endscreen sidecar",
        "GET-only",
        "default",
    ),
    "upload.us.tiktokapis.com": (
        "TikTok Direct Post video upload target",
        "TikTok publishing",
        "media/data out",
        "opt-in",
    ),
    "www.facebook.com": (
        "Web origin used to build a clickable permalink from Graph's relative "
        "`permalink_url` (not fetched)",
        "Meta (Facebook/Instagram) Reels publishing",
        "GET-only",
        "opt-in",
    ),
    "www.googleapis.com": (
        "YouTube Data API (upload, metadata, playlists) and Drive API calls",
        "YouTube upload / metrics",
        "media/data out",
        "default",
    ),
    "www.tiktok.com": (
        "Web origin used to build a human-facing TikTok permalink (not fetched)",
        "TikTok publishing status",
        "GET-only",
        "opt-in",
    ),
    "www.topografix.com": (
        "GPX 1.1 XML namespace URI parsed/emitted in GPX files (not fetched)",
        "GPS track parsing",
        "GET-only",
        "default",
    ),
    "www2.census.gov": (
        "Pinned 2026 TIGER county/state ZIPs, SHA-256 verified before caching",
        "Retained-route boundary proof",
        "GET-only",
        "default",
    ),
    "youtu.be": (
        "Video permalink text built into descriptions, chapters and CSV rows (not fetched)",
        "YouTube metadata / shorts CSV",
        "GET-only",
        "default",
    ),
}

_NOTE = (
    "> **Generated.** This file is rebuilt by `tools/build_outbound_hosts.py` from "
    "every host literal it finds in `gomoveshift/**/*.py` (a test, "
    "`tests/test_outbound_hosts.py`, fails if this file and the generator disagree). Do "
    "not hand-edit it -- add the host to `INVENTORY` in the generator with a reason and "
    "re-run it. This is a superset of the enforced upload allowlists in "
    "`publishing/*_adapter.py` and `route_story/sources.py`: it also catalogs GET-only "
    "and display-text-only hosts that appear in source but never carry media/data out."
)

_HEADER = "| Host | Purpose | Feature | Direction | Mode |\n|---|---|---|---|---|"


def render_doc(hosts: list[HostSite]) -> str:
    lines = [
        "# Outbound hosts",
        "",
        "Every host `gomoveshift/**/*.py` names, and what it is for. See "
        "`docs/pipeline/84-network-allowlist-credential-stores.md` for the operative "
        "allowlist contract; this file is the generated inventory that keeps it honest.",
        "",
        _NOTE,
        "",
        _HEADER,
    ]
    for site in hosts:
        purpose, feature, direction, mode = INVENTORY[site.host]
        lines.append(f"| `{site.host}` | {purpose} | {feature} | {direction} | {mode} |")
    return "\n".join(lines) + "\n"


def undocumented(hosts: list[HostSite]) -> list[HostSite]:
    return [site for site in hosts if site.host not in INVENTORY]


def main(argv: list[str]) -> int:
    check = "--check" in argv
    hosts = scan_hosts()
    missing = undocumented(hosts)
    if missing:
        for site in missing:
            print(
                f"{site.path}:{site.lineno}: host {site.host!r} is not in "
                "tools/build_outbound_hosts.py's INVENTORY -- add it with a purpose, "
                "feature, direction and mode before this can pass.",
                file=sys.stderr,
            )
        return 1
    rendered = render_doc(hosts)
    if check:
        current = DOC_PATH.read_text(encoding="utf-8") if DOC_PATH.is_file() else ""
        if current != rendered:
            print(
                f"{DOC_PATH} is stale or hand-edited. Run:\n"
                f"  python3 tools/build_outbound_hosts.py",
                file=sys.stderr,
            )
            return 1
        print(f"{DOC_PATH} matches the generator.")
        return 0
    DOC_PATH.write_text(rendered, encoding="utf-8")
    print(f"wrote {DOC_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
