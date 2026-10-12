"""Runtime tenant-stamp guard for the console's publishing storage.

Each Waypost brand ("tenant") resolves its own ``seo.storage_path``
(``console/pubqueue.py:queue_db_path``); isolation is PATH SEPARATION -- a
separate delivery-queue sqlite file per brand -- never a brand column in
that file's schema (``tests/test_publishing_brand_isolation.py``).

Path separation alone has one blind spot: the console image bakes
``seo.storage_path`` to the SAME in-container path
(``docker/console-config-override.yaml``) for every brand -- only the HOST
side of each brand's bind mount (``GMS_SEO_STORAGE_DIR`` in
``docker/console-compose.yaml``, set per console instance) is supposed to
differ. Two brands whose host mounts point at the same directory look
identical from inside either container, so no in-container path comparison
can ever catch that misconfiguration. Instead a gate validates it at
startup: a stamp written into the storage itself, which crosses the mount
boundary, is the only signal that survives. It runs at console startup,
before anything reads or mutates the delivery queue.

Fresh storage (no stamp) is claimed by whichever brand starts first -- a
brand-new tenant's first run shouldn't have to pre-create its own storage,
so a missing directory is created too. A damaged stamp (empty, whitespace,
or unreadable) is treated as UNKNOWN and refused rather than silently
reclaimed: this guard exists to stop one tenant silently sharing another's
queue, so "I can't tell whose this is" must fail exactly like "this is
someone else's", never fall through to "treat it as unclaimed".
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

_STAMP_NAME = ".waypost-tenant"

_MOUNT_HINT = (
    "Check the HOST side of this brand's bind mount (GMS_SEO_STORAGE_DIR in "
    "docker/console-compose.yaml, set per console instance), "
    "not the container path -- the container-side path is identical for every brand by design."
)


class TenantMismatchError(RuntimeError):
    """Raised when a brand tries to start against publishing storage already
    claimed by a different brand, or whose stamp is of unknown ownership."""


def _stamp_path(storage_path: Union[str, Path]) -> Path:
    return Path(storage_path) / _STAMP_NAME


def _write_stamp(stamp_path: Path, brand: str) -> None:
    """Atomic write: tempfile in the same directory, fsync, ``os.replace``.
    Cleans up its own tempfile on any failure so a partial write is never
    left behind."""
    stamp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = stamp_path.with_name(f".{_STAMP_NAME}.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            handle.write(brand)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, stamp_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def claim_storage(storage_path: Union[str, Path], brand: str) -> None:
    """Claim ``storage_path`` for ``brand`` or raise ``TenantMismatchError``.

    - no stamp yet -> ``brand`` claims it (written now; the directory is
      created if it doesn't exist).
    - stamp already reads ``brand`` -> accepted, no rewrite (a restart of
      the same tenant causes no churn).
    - stamp reads a different, readable brand -> raise.
    - stamp present but empty/whitespace/unreadable -> raise (unknown
      owner; see module docstring).
    """
    stamp_path = _stamp_path(storage_path)
    if not stamp_path.exists():
        _write_stamp(stamp_path, brand)
        return

    try:
        existing = stamp_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise TenantMismatchError(
            f"brand {brand!r} cannot start: the tenant stamp at {stamp_path} could not be "
            f"read ({exc}); its owning brand is unknown. {_MOUNT_HINT}"
        ) from exc

    if not existing:
        raise TenantMismatchError(
            f"brand {brand!r} cannot start: the tenant stamp at {stamp_path} is empty or "
            f"whitespace-only; its owning brand is unknown. {_MOUNT_HINT}"
        )

    if existing != brand:
        raise TenantMismatchError(
            f"brand {brand!r} cannot start: storage at {stamp_path.parent} is already "
            f"claimed by brand {existing!r}. {_MOUNT_HINT}"
        )
