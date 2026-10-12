"""Public ``image_url`` derivation + pre-publish verification for photo/
carousel Graph API containers.

Meta's photo-container API wants a publicly-reachable ``image_url``, not a
raw-bytes upload (see ``photo_graph_client.py``'s module docstring). This
module answers the two questions that client must not guess at:

1. What URL should a given photo publish under? Derived from what the
   photos pipeline already knows (``pipeline/photos/selector.py`` reads
   photos FROM the Hugo site, not a copy of it) -- the journey slug and
   filename, joined onto the site's OWN configured ``baseURL`` (read from
   its ``hugo.toml``, never hardcoded). A second brand's site is just a
   different ``site_root`` pointing at a different config file.
2. Is that URL actually reachable RIGHT NOW? A newly ingested photo is
   only live once the Hugo site has been deployed
   (a gated, human-run site deploy) -- so this is a
   genuine per-photo question, answered with a real GET immediately
   before the irreversible Graph publish call, never assumed from the
   photo existing on local disk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tomllib

# Drive ``…/file/d/<ID>/view`` sharing URL -> ``uc?export=download`` direct form.
_DRIVE_VIEW_RE = re.compile(r"https://drive\.google\.com/file/d/([^/]+)/view")


def drive_direct_download_link(url: str) -> str:
    """Return the Drive media URL a downstream consumer can attach directly.

    An unauthenticated GET on a Drive ``…/file/d/<ID>/view?usp=sharing`` link
    returns Drive's HTML viewer *page*, not the raw bytes. The
    ``uc?export=download&id=<ID>`` form returns the raw bytes and is
    confirmed to attach for TikTok and Meta/IG.

    Scope: shorts-sized media only -- ``uc?export=download`` hits Google's
    virus-scan interstitial for files >~100 MB, so this conversion must not
    be applied to long-form links.

    Non-matching / empty values pass through unchanged (skip-upload audits,
    already-converted links).
    """
    if not isinstance(url, str):
        return url
    m = _DRIVE_VIEW_RE.match(url)
    if not m:
        return url
    return f"https://drive.google.com/uc?export=download&id={m.group(1)}"


# Served path under the site's baseURL -- mirrors selector.py's on-disk
# ``static/images/journeys/<slug>/<filename>`` layout (Hugo copies
# ``static/`` verbatim into ``public/``, dropping the ``static/`` prefix).
IMAGES_URL_BASE = "images/journeys"

HUGO_CONFIG_REL = "config/_default/hugo.toml"

# Verification outcomes -- distinct reasons so an unreachable image is never
# reported as a bare "invalid URL".
NOT_FOUND = "not_found"
WRONG_CONTENT_TYPE = "wrong_content_type"
UNREACHABLE = "unreachable"


class PhotoImageUrlError(ValueError):
    """No ``image_url`` could be derived at all -- neither the site's
    baseURL nor a Drive fallback share URL was available."""


@dataclass(frozen=True, slots=True)
class ImageUrlVerification:
    """Result of a pre-publish reachability check on a derived
    ``image_url``. ``ok=False`` always carries a ``reason`` naming exactly
    which check failed -- never a bare boolean."""

    ok: bool
    reason: str = ""


def site_base_url(site_root: Path, override: Optional[str] = None) -> Optional[str]:
    """The site's own configured ``baseURL``, read from its Hugo config.

    ``override`` (an explicit caller-supplied value) wins outright --
    useful for a brand whose config layout differs. Returns ``None`` when
    neither is available; callers decide whether that's fatal."""
    if override:
        return override.rstrip("/")
    config_path = Path(site_root) / HUGO_CONFIG_REL
    try:
        data = tomllib.loads(config_path.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return None
    base_url = data.get("baseURL")
    if not base_url:
        return None
    return str(base_url).rstrip("/")


def resolve_photo_image_url(
    journey_slug: str,
    filename: str,
    *,
    site_root: Path,
    base_url_override: Optional[str] = None,
    drive_share_url: Optional[str] = None,
) -> str:
    """The public URL a Graph photo container should point at.

    Preferred: ``<site baseURL>/images/journeys/<slug>/<filename>`` --
    once the site is deployed this is exactly where the photo is served.

    Fallback, only when the site's baseURL can't be determined at all: a
    Drive direct-download share link, reusing (never duplicating) the same
    ``uc?export=download&id=`` conversion ``drive_direct_download_link``
    already proved attaches for TikTok/Meta.

    Raises ``PhotoImageUrlError`` naming the photo when neither source is
    available -- never a guessed/blank URL.
    """
    base_url = site_base_url(site_root, base_url_override)
    if base_url:
        return f"{base_url}/{IMAGES_URL_BASE}/{journey_slug}/{filename}"
    if drive_share_url:
        return drive_direct_download_link(drive_share_url)
    raise PhotoImageUrlError(
        f"cannot derive an image_url for {journey_slug}/{filename}: the "
        f"site's baseURL is not configured/readable at {site_root} and no "
        "Drive fallback share URL was given"
    )


def image_url_for_media_path(
    media_path: str,
    *,
    site_root: Path,
    base_url_override: Optional[str] = None,
    drive_share_url: Optional[str] = None,
) -> str:
    """The ``image_url`` for a photo/carousel ``MediaFingerprint.path`` --
    the caller-facing entry point that maps a raw on-disk path
    onto the ``<slug>/<filename>`` pair ``resolve_photo_image_url`` wants.

    Only a path actually inside the site's served
    ``static/<IMAGES_URL_BASE>/<slug>/<filename>`` layout (mirrors
    ``pipeline/photos/selector.py``'s ``IMAGES_BASE``) is accepted -- NEVER
    a slug/filename guessed from a stem parse. A resized social copy under
    ``photos.resolve_resized_base()`` (flattened ``<slug>_<name>.jpg``
    naming, not served by the site at all) is exactly the case this
    refuses rather than silently maps.

    Raises ``PhotoImageUrlError`` naming ``media_path`` when it is not
    under that layout.
    """
    images_root = Path(site_root) / "static" / IMAGES_URL_BASE
    try:
        relative = Path(media_path).resolve().relative_to(images_root.resolve())
    except (OSError, ValueError) as exc:
        raise PhotoImageUrlError(
            f"cannot derive an image_url for {media_path!r}: not under the "
            f"site's served {images_root} layout -- {exc}"
        ) from exc
    if len(relative.parts) != 2:
        raise PhotoImageUrlError(
            f"cannot derive an image_url for {media_path!r}: expected "
            f"<slug>/<filename> under {images_root}, got {relative}"
        )
    slug, filename = relative.parts
    return resolve_photo_image_url(
        slug,
        filename,
        site_root=site_root,
        base_url_override=base_url_override,
        drive_share_url=drive_share_url,
    )


def _deploy_hint(site_root: Path) -> str:
    """Site display name for the deploy-step hint, derived from the
    checkout's own directory name (``site-example`` -> ``example``)
    -- never a hardcoded brand."""
    name = Path(site_root).name
    return name[len("site-") :] if name.startswith("site-") else name


def verify_image_url(
    url: str, *, session, site_root: Path, timeout: float = 10.0
) -> ImageUrlVerification:
    """GET ``url`` and confirm it actually serves an image, BEFORE any
    Graph container is created -- an upstream gate immediately before the
    irreversible publish call. Distinguishes every failure mode rather
    than collapsing them into one "invalid URL":

    - 404: the photo isn't published to the site yet (the common case --
      ingestion happened but the site deploy hasn't run).
    - a non-image ``Content-Type`` on a 200 (e.g. a redirect landing on an
      HTML page, or Drive's viewer page instead of raw bytes).
    - any other non-200 status or a network failure: unreachable.
    """
    import requests

    try:
        response = session.get(url, timeout=timeout, allow_redirects=True)
    except requests.RequestException as exc:
        return ImageUrlVerification(False, f"{UNREACHABLE}: {exc}")
    if response.status_code == 404:
        return ImageUrlVerification(
            False,
            f"{NOT_FOUND}: not published to the site yet -- deploy "
            f"the site ({_deploy_hint(site_root)}) first",
        )
    if response.status_code != 200:
        return ImageUrlVerification(False, f"{UNREACHABLE}: HTTP {response.status_code}")
    content_type = (response.headers or {}).get("Content-Type", "")
    if not content_type.startswith("image/"):
        return ImageUrlVerification(
            False, f"{WRONG_CONTENT_TYPE}: got {content_type!r}, expected an image/* type"
        )
    return ImageUrlVerification(True)
