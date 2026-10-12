"""Brand-scoped credential directory resolution — one home for the
``<base>/brands/<brand>/creds`` layout so ``meta_auth``, ``tiktok_auth`` and
``pipeline._yt_creds`` don't each grow their own copy.

Layout:
  Shared (the default): the SOPS ``.enc`` file lives at
  ``base.parent / filename`` where ``base`` is ``seo.credentials_dir``
  (default ``"./credentials"``).

  Brand-scoped: when ``brand`` is given and
  ``<base>/brands/<brand>/creds`` exists on disk, that directory is the
  resolved credentials dir instead, so the same ``creds_dir.parent /
  filename`` convention lands the ``.enc`` at
  ``<base>/brands/<brand>/<filename>``.

A brand with no ``brands/<brand>/creds`` directory falls back to the shared
dir — a tenant with no credentials of its own still gets the console's
account.
"""

from __future__ import annotations

from pathlib import Path


def resolve_credentials_dir(brand: str | None, base: Path | None = None) -> Path:
    """The directory a credential loader should read from.

    ``brand`` given and ``<base>/brands/<brand>/creds`` exists on disk →
    that directory. Otherwise (no brand, or no brand-scoped dir yet) →
    ``base`` unchanged, which is exactly today's behaviour.
    """
    if base is None:
        from gomoveshift.config import get_config

        base = Path(get_config().seo.credentials_dir)
    if brand:
        brand_dir = base / "brands" / brand / "creds"
        if brand_dir.is_dir():
            return brand_dir
    return base


def resolve_sops_file(filename: str, brand: str | None, base: Path | None = None) -> Path:
    """The SOPS ``.enc`` path for ``filename``, honouring the same
    ``creds_dir.parent / filename`` convention every loader already uses."""
    return resolve_credentials_dir(brand, base).parent / filename
