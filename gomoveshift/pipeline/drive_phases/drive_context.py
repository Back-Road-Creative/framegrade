"""DriveContext — typed facade over the drive pipeline's PipelineState.

The raw state object is kept internally, with typed
property accessors for commonly-used keys. ``__getattr__`` falls back to
the underlying object so nothing breaks while we migrate incrementally
to fully-typed fields.

The underlying ``self._state`` is a ``PipelineState`` dataclass instance,
not a raw dict. All accessors here go through
``getattr``/``object.__setattr__`` so the dict-protocol shim on
``PipelineState`` can be deleted without breaking this facade.
"""

from __future__ import annotations

import dataclasses as _dc
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DriveContext:
    """Typed facade over the drive pipeline ``PipelineState``.

    Usage::

        ctx = DriveContext(state)
        ctx.source_dir          # typed Path
        ctx["some_raw_key"]     # raw attribute access by string name
        ctx.some_raw_key        # also works via __getattr__ fallback
    """

    def __init__(self, state) -> None:
        # Store reference — mutations propagate to the underlying state object
        object.__setattr__(self, "_state", state)

    # ── Raw dict-protocol access (delegates to PipelineState attrs) ────

    def __getitem__(self, key: str) -> Any:
        return getattr(self._state, key)

    def __setitem__(self, key: str, value: Any) -> None:
        object.__setattr__(self._state, key, value)

    def __contains__(self, key: str) -> bool:
        return hasattr(self._state, key) and getattr(self._state, key) is not None

    def get(self, key: str, default: Any = None) -> Any:
        val = getattr(self._state, key, None)
        return default if val is None else val

    def setdefault(self, key: str, default: Any = None) -> Any:
        val = getattr(self._state, key, None)
        if val is None:
            object.__setattr__(self._state, key, default)
            return default
        return val

    def items(self):
        return [
            (f.name, getattr(self._state, f.name))
            for f in _dc.fields(self._state)
            if getattr(self._state, f.name) is not None
        ]

    def keys(self):
        return [f.name for f in _dc.fields(self._state) if getattr(self._state, f.name) is not None]

    @property
    def raw(self):
        """Direct access to the underlying state (PipelineState dataclass)."""
        return self._state

    # ── Typed accessors (commonly used keys) ─────────────────────────

    @property
    def source_dir(self) -> Path:
        return Path(self._state.source_dir)

    @property
    def output_base(self) -> Path:
        return Path(self._state.output_base)

    @property
    def region(self) -> str:
        return getattr(self._state, "region", None) or "ireland"

    @property
    def publish_day(self) -> str:
        return getattr(self._state, "publish_day", None) or "wed"

    @property
    def location(self) -> str:
        return getattr(self._state, "location", None) or ""

    @location.setter
    def location(self, value: str) -> None:
        object.__setattr__(self._state, "location", value)

    @property
    def fresh(self) -> bool:
        return bool(getattr(self._state, "fresh", False))

    @property
    def skip_upload(self) -> bool:
        return bool(getattr(self._state, "skip_upload", False))

    @property
    def skip_csv(self) -> bool:
        return bool(getattr(self._state, "skip_csv", False))

    @property
    def episode_files(self) -> list:
        return getattr(self._state, "episode_files", None) or []

    @property
    def gpx_files(self) -> list:
        return getattr(self._state, "gpx_files", None) or []

    @property
    def file_durations(self) -> list[float]:
        return getattr(self._state, "file_durations", None) or []

    @property
    def file_boundaries(self) -> list[tuple[float, float]]:
        return getattr(self._state, "file_boundaries", None) or []

    @property
    def episode_locations(self) -> list[str]:
        return getattr(self._state, "episode_locations", None) or []

    @property
    def drive_scores(self):
        return getattr(self._state, "drive_scores", None)

    @property
    def processed_files(self):
        return getattr(self._state, "processed_files", None) or {}

    @property
    def drive_plan(self):
        return getattr(self._state, "drive_plan", None)

    @property
    def regular_lf_paths(self) -> list:
        return getattr(self._state, "regular_lf_paths", None) or []

    @property
    def shorts_per_lf(self) -> dict:
        return getattr(self._state, "shorts_per_lf", None) or {}

    # ── Fallback: any attribute not defined falls through to PipelineState ─

    def __getattr__(self, name: str) -> Any:
        state = object.__getattribute__(self, "_state")
        try:
            return getattr(state, name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}' "
                f"(also not found on underlying PipelineState)"
            )

    def __setattr__(self, name: str, value: Any) -> None:
        # If the name is a known property, use the property setter
        if name in type(self).__dict__ and isinstance(type(self).__dict__[name], property):
            type(self).__dict__[name].fset(self, value)
        else:
            object.__setattr__(self._state, name, value)
