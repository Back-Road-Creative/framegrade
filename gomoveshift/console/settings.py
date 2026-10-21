"""Console settings: the GMS_* registry (mirrors README's GMS_* table) and
per-brand overrides (null = using the registry default)."""

import os
import re
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel

from ..paths import gms_secret_path, gms_staging_dir

_CONSOLE_ROOT = Path(__file__).parent
_REGISTRY_PATH = _CONSOLE_ROOT / "registry" / "gms_settings.yaml"
_SHIPPED_BRANDS_DIR = _CONSOLE_ROOT / "brands"


class GmsSetting(BaseModel):
    """One documented GMS_* variable."""

    name: str
    default: Optional[str] = None
    description: str = ""


def load_registry() -> list[GmsSetting]:
    raw = yaml.safe_load(_REGISTRY_PATH.read_text()) or []
    return [GmsSetting(**row) for row in raw]


def console_data_dir() -> Path:
    """``GMS_CONSOLE_DATA_DIR`` or ``<GMS_STAGING_DIR>/_console``."""
    raw = os.environ.get("GMS_CONSOLE_DATA_DIR")
    return Path(raw) if raw and raw.strip() else gms_staging_dir() / "_console"


def _brand_settings_path(slug: str, data_dir: Optional[Path] = None) -> Path:
    base = data_dir if data_dir is not None else console_data_dir()
    return base / "brands" / slug / "settings.yaml"


def load_brand_settings(slug: str, data_dir: Optional[Path] = None) -> dict:
    """Seed brands/<slug>/settings.yaml from the shipped template on first
    use (never mutates the shipped copy), then return its contents."""
    path = _brand_settings_path(slug, data_dir)
    if not path.exists():
        template = _SHIPPED_BRANDS_DIR / slug / "settings.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(template.read_text() if template.exists() else "{}\n")
    return yaml.safe_load(path.read_text()) or {}


def save_brand_settings(slug: str, values: dict, data_dir: Optional[Path] = None) -> Path:
    path = _brand_settings_path(slug, data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(values, sort_keys=True))
    return path


def known_brands() -> list[str]:
    """Brand slugs: the shipped template directories union any brand that
    has ever saved an override file under the console data dir."""
    slugs: set[str] = set()
    if _SHIPPED_BRANDS_DIR.exists():
        slugs.update(p.name for p in _SHIPPED_BRANDS_DIR.iterdir() if p.is_dir())
    data_brands_dir = console_data_dir() / "brands"
    if data_brands_dir.exists():
        slugs.update(p.name for p in data_brands_dir.iterdir() if p.is_dir())
    return sorted(slugs)


def current_brand() -> str:
    """``GMS_CONSOLE_BRAND`` or the default "gomoveshift" instance."""
    raw = os.environ.get("GMS_CONSOLE_BRAND")
    return raw.strip() if raw and raw.strip() else "gomoveshift"


def brand_disabled_tabs(slug: str) -> set[str]:
    """Console nav slugs this brand hides (e.g. no drive pipeline enabled).
    A missing/empty ``brand.yaml`` means every tab stays enabled -- the
    gomoveshift brand ships none, so it is unaffected."""
    path = _SHIPPED_BRANDS_DIR / slug / "brand.yaml"
    if not path.exists():
        return set()
    raw = yaml.safe_load(path.read_text()) or {}
    return set(raw.get("disabled_tabs") or [])


_HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


def brand_kit(slug: str) -> Optional[dict]:
    """Optional custom-branding block from ``brand.yaml``'s ``kit:`` key.

    ``None`` when ``brand.yaml`` is missing/empty or has no ``kit:`` key --
    the default, unbranded OST path (gomoveshift ships no ``brand.yaml``, and
    any other shipped one may be ``disabled_tabs`` only, with no kit, in
    which case this also returns ``None``).
    Otherwise a dict with keys ``font`` (path/family, optional), ``colors``
    (dict of ``primary``/``accent``/``text`` -> ``#RRGGBB``, optional),
    ``logo`` (path, optional) and ``safe_margin`` (fraction in [0, 1],
    optional). An invalid value raises ``ValueError`` naming the offending
    ``kit.<field>`` — fail loudly rather than silently ignore a brand's
    misconfigured kit.
    """
    path = _SHIPPED_BRANDS_DIR / slug / "brand.yaml"
    if not path.exists():
        return None
    raw = yaml.safe_load(path.read_text()) or {}
    kit = raw.get("kit")
    if not kit:
        return None
    if not isinstance(kit, dict):
        raise ValueError(f"brand.yaml kit: must be a mapping, got {kit!r}")

    font = kit.get("font")
    if font is not None and not isinstance(font, str):
        raise ValueError(f"brand.yaml kit.font: must be a string path/family, got {font!r}")

    colors = kit.get("colors") or {}
    if not isinstance(colors, dict):
        raise ValueError(f"brand.yaml kit.colors: must be a mapping, got {colors!r}")
    for key in ("primary", "accent", "text"):
        value = colors.get(key)
        if value is not None and not _HEX_COLOR_RE.match(str(value)):
            raise ValueError(
                f"brand.yaml kit.colors.{key}: must be a #RRGGBB hex value, got {value!r}"
            )

    logo = kit.get("logo")
    if logo is not None and not isinstance(logo, str):
        raise ValueError(f"brand.yaml kit.logo: must be a string path, got {logo!r}")

    safe_margin = kit.get("safe_margin")
    if safe_margin is not None:
        try:
            safe_margin = float(safe_margin)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"brand.yaml kit.safe_margin: must be a fraction, got {kit.get('safe_margin')!r}"
            ) from e
        if not (0.0 <= safe_margin <= 1.0):
            raise ValueError(
                f"brand.yaml kit.safe_margin: must be within [0, 1], got {safe_margin!r}"
            )

    return {"font": font, "colors": colors, "logo": logo, "safe_margin": safe_margin}


#: Known credential sources (env override, default file name under the config
#: home, description). Status is presence-only -- never reads or displays file
#: contents.
CREDENTIAL_SOURCES: list[tuple[str, str, str]] = [
    (
        "GMS_DRIVE_CREDENTIALS_ENC",
        "drive_credentials.json.enc",
        "SOPS-encrypted Drive OAuth client JSON.",
    ),
    (
        "GMS_DRIVE_TOKEN_ENC",
        "drive_token.json.enc",
        "SOPS-encrypted Drive refresh token JSON.",
    ),
    (
        "GMS_AGE_KEY_FILE",
        "age.key",
        "age key used to decrypt the two SOPS *.enc files.",
    ),
    (
        "GMS_SOPS_CONFIG",
        ".sops.yaml",
        "SOPS config (creation rules + recipients).",
    ),
]


#: The four env-var names ``gomoveshift/pipeline/uploader.py`` resolves through
#: ``gomoveshift.paths.gms_secret_path`` -- the same call-time function is used
#: here, so an override (or an empty value, which counts as unset) is read the
#: same way on both sides.
_SECRET_ENV_NAMES = (
    "GMS_DRIVE_CREDENTIALS_ENC",
    "GMS_DRIVE_TOKEN_ENC",
    "GMS_AGE_KEY_FILE",
    "GMS_SOPS_CONFIG",
)


def resolved_secret_paths() -> dict[str, Path]:
    """Resolve the pipeline secret paths through their env-var overrides
    (call-time, never cached) -- never the file contents."""
    defaults = {name: filename for name, filename, _description in CREDENTIAL_SOURCES}
    return {name: gms_secret_path(name, defaults[name]) for name in _SECRET_ENV_NAMES}


def credential_status() -> list[dict]:
    """Presence-only status for known credential files -- never the contents."""
    out = []
    for name, filename, description in CREDENTIAL_SOURCES:
        path = gms_secret_path(name, filename)
        out.append({"name": name, "description": description, "present": path.exists()})
    return out


def brand_env_overrides(slug: Optional[str] = None) -> dict:
    """``slug``'s (default: ``current_brand()``) saved overrides -- only registry
    names, only non-null values, as strings. No ``os.environ``."""
    slug = slug if slug is not None else current_brand()
    registry_names = {s.name for s in load_registry()}
    overrides = load_brand_settings(slug)
    return {k: str(v) for k, v in overrides.items() if k in registry_names and v is not None}


def brand_launch_env(slug: Optional[str] = None) -> dict:
    """``os.environ`` merged with ``slug``'s saved overrides (see
    ``brand_env_overrides``). For ``jobs.launch(..., env=...)`` so a launched
    job sees the same per-brand settings the console itself renders."""
    env = dict(os.environ)
    env.update(brand_env_overrides(slug))
    return env


def youtube_identity_status() -> list[dict]:
    """Per-brand YouTube channel identity -- offline, read-only.

    Reads only the plain-JSON identity sidecar next to each brand's SOPS
    token (``_yt_creds.read_channel_identity``): no sops call, no network
    call, and the encrypted token's own contents are never opened here.
    ``channel_title`` is ``"not connected"`` when the brand has no token, and
    names the re-mint command when a token exists without a sidecar (a
    legacy token that predates this feature -- the same case
    ``check_youtube_identity`` in ``doctor`` WARNs about).
    """
    from ..pipeline._yt_creds import DEFAULT_CREDENTIALS_DIR, read_channel_identity
    from ..publishing.credential_paths import resolve_sops_file

    out = []
    for slug in known_brands():
        enc_path = resolve_sops_file("youtube_upload_token.json.enc", slug, DEFAULT_CREDENTIALS_DIR)
        identity = read_channel_identity(enc_path) if enc_path.exists() else None
        if identity:
            title = identity["channel_title"]
        elif enc_path.exists():
            # Minted before identity capture: connected, channel unknown.
            title = (
                f"token present, channel unknown -- run `gomoveshift youtube-auth --brand {slug}`"
            )
        else:
            title = "not connected"
        out.append(
            {
                "brand": slug,
                "channel_id": identity["channel_id"] if identity else None,
                "channel_title": title,
            }
        )
    return out


def registry_status() -> list[dict]:
    """Registry rows with a set/unset flag. Values are never included."""
    return [
        {"name": s.name, "description": s.description, "is_set": bool(os.environ.get(s.name))}
        for s in load_registry()
    ]
