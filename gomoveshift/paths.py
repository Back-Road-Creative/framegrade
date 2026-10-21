"""Centralized path defaults for GoMoveShift pipeline.

All env-var-backed paths live here so every module resolves the same
default.  Import from ``gomoveshift.paths`` instead of calling
``os.environ.get(...)`` with an ad-hoc fallback.

Values are read lazily via functions so that ``monkeypatch.setenv`` in
tests takes effect even after the module has been imported.
"""

import os
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Per-user application directories
# ---------------------------------------------------------------------------
#
# Every default below resolves under one of three homes, so a fresh install
# never writes outside the invoking user's own directories:
#
#   data home    ``$GMS_DATA_HOME``   else ``$XDG_DATA_HOME/gomoveshift``
#                else ``~/.local/share/gomoveshift``
#   cache home   ``$GMS_CACHE_HOME``  else ``$XDG_CACHE_HOME/gomoveshift``
#                else ``~/.cache/gomoveshift``
#   config home  ``$GMS_CONFIG_HOME`` else ``$XDG_CONFIG_HOME/gomoveshift``
#                else ``~/.config/gomoveshift``
#
# Input, output, photo and export directories live under the data home;
# caches and lock files under the cache home; credential, token, key and
# SOPS-config files under the config home. Every specific environment
# variable (``GMS_STAGING_DIR`` and friends) still overrides its default.


def _app_home(override_var: str, xdg_var: str, home_relative: str) -> Path:
    override = os.environ.get(override_var, "").strip()
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get(xdg_var, "").strip()
    if xdg:
        return Path(xdg).expanduser() / "gomoveshift"
    return Path.home() / home_relative / "gomoveshift"


def gms_data_home() -> Path:
    """Per-user data root: inputs, outputs, photo state and exports."""
    return _app_home("GMS_DATA_HOME", "XDG_DATA_HOME", ".local/share")


def gms_cache_home() -> Path:
    """Per-user cache root: stage caches, shared tile/response caches, locks."""
    return _app_home("GMS_CACHE_HOME", "XDG_CACHE_HOME", ".cache")


def gms_config_home() -> Path:
    """Per-user config root: credentials, tokens, keys and SOPS config files."""
    return _app_home("GMS_CONFIG_HOME", "XDG_CONFIG_HOME", ".config")


def gms_secret_path(env_name: str, filename: str) -> Path:
    """Path of one credential file: the ``env_name`` override, else ``filename``
    under the config home.

    Resolved at call time; an empty or whitespace-only override counts as unset,
    so the console and the uploader always agree on the same location.
    """
    raw = os.environ.get(env_name, "").strip()
    return Path(raw).expanduser() if raw else gms_config_home() / filename


# ---------------------------------------------------------------------------
# Public accessors — always re-read the environment
# ---------------------------------------------------------------------------


def gms_staging_dir() -> Path:
    """Video staging / input directory."""
    return Path(os.environ.get("GMS_STAGING_DIR") or gms_data_home() / "input")


def gms_output_dir() -> Path:
    """Pipeline output directory."""
    return Path(os.environ.get("GMS_OUTPUT_DIR") or gms_data_home() / "output")


def gms_overpass_cache_dir() -> Optional[Path]:
    """Shared, cross-run Overpass response cache directory.

    Deliberately OUTSIDE any run's ``stages_dir``: the same drive is
    processed repeatedly (regular LFs, shorts, superlong, re-runs into a
    fresh ``--output-dir``) and every one of those runs asks Overpass the
    same bbox questions. A per-run cache answers none of them.

    Unlike its sibling helpers this returns ``Optional`` — an explicitly
    empty ``GMS_OVERPASS_CACHE_DIR`` means "never cache", the escape hatch
    tests use so a mocked query is never served from disk.
    """
    raw = os.environ.get("GMS_OVERPASS_CACHE_DIR", _default_overpass_cache_dir())
    return Path(raw) if raw.strip() else None


def gms_route_story_cache_dir() -> Path:
    """Shared, cross-run cache directory for Route Story map asset bundles.

    Same ext4-not-tmpfs reasoning as ``gms_overpass_cache_dir``: bundles are
    content-addressed (see ``route_story.bundle.bundle_dir_for``) and the
    same footprint/source-version combination is asked for repeatedly across
    runs, so a per-run cache would rebuild identical tiles every time.
    """
    return Path(os.environ.get("GMS_ROUTE_STORY_CACHE_DIR", _default_route_story_cache_dir()))


def gms_photo_output_dir() -> Path:
    """Photo-post pipeline output directory (selection state, CSV, links)."""
    return Path(os.environ.get("GMS_PHOTO_OUTPUT_DIR") or gms_data_home() / "photos")


def gms_export_dir() -> Path:
    """Directory the Metricool CSV exports are written to."""
    return Path(os.environ.get("GMS_EXPORT_DIR") or gms_data_home() / "output")


class InputsDirError(ValueError):
    """``GMS_INPUTS_DIR`` is set to something that is not an existing directory."""


def gms_inputs_dir() -> Path:
    """The operator's own inputs (for example Route Story launch profiles),
    kept outside the code tree.

    ``GMS_INPUTS_DIR`` when set (``~`` is expanded; an empty or whitespace-only
    value counts as unset), else ``operator-inputs`` under the data home. The
    default is deliberately not checked: a fresh install has no inputs yet and
    readers treat a missing subdirectory as empty. A value that *is* set must
    name an existing directory; otherwise ``InputsDirError`` (a ``ValueError``)
    names the variable and the path, so a typo surfaces at once instead of as
    "no inputs found".
    """
    raw = os.environ.get("GMS_INPUTS_DIR", "").strip()
    if not raw:
        return gms_data_home() / "operator-inputs"
    path = Path(raw).expanduser()
    if not path.is_dir():
        reason = "is not a directory" if path.exists() else "does not exist"
        raise InputsDirError(
            f"GMS_INPUTS_DIR={path} {reason}. Create the directory or unset GMS_INPUTS_DIR."
        )
    return path


def assert_required_paths_exist() -> None:
    """Validate that env-resolved required paths exist on disk.

    Raises ``FileNotFoundError`` with an actionable message — naming both
    the resolved path and the env var that controls it — so a missing
    mount or a misconfigured env doesn't surface several phases deep as
    an opaque ``[Errno 2]``.
    """
    missing: list[str] = []
    staging = gms_staging_dir()
    if not staging.exists():
        missing.append(f"GMS_STAGING_DIR={staging}")
    output = gms_output_dir()
    if not output.exists():
        missing.append(f"GMS_OUTPUT_DIR={output}")
    if missing:
        raise FileNotFoundError(
            "Required GMS paths missing on disk: "
            + ", ".join(missing)
            + ". Set the env var or create the directory."
        )


# ---------------------------------------------------------------------------
# Intermediate directory (ext4-backed when output is on drvfs)
# ---------------------------------------------------------------------------


# Persistent local-disk cache. /tmp is tmpfs and gets wiped on reboot, which
# kills long-running pipelines mid-flight with no recovery. Override with the
# GMS_INTERMEDIATE_DIR env var if you need a different fast local disk.
def _default_intermediate_dir() -> str:
    return str(gms_cache_home())


# Overpass answers change on OSM-edit timescales, not run timescales, so this
# cache is deliberately shared across runs and survives an output dir being
# deleted. Same local-disk-not-tmpfs reasoning as above.
def _default_overpass_cache_dir() -> str:
    return str(gms_cache_home() / "overpass")


# Route Story map asset bundles are content-addressed and reused across runs
# that render the same footprint from the same source imagery/terrain
# versions. Same local-disk-not-tmpfs reasoning as the overpass cache above.
def _default_route_story_cache_dir() -> str:
    return str(gms_cache_home() / "route-story")


_LAZY_DEFAULTS = {
    "_DEFAULT_INTERMEDIATE_DIR": _default_intermediate_dir,
    "_DEFAULT_OVERPASS_CACHE_DIR": _default_overpass_cache_dir,
    "_DEFAULT_ROUTE_STORY_CACHE_DIR": _default_route_story_cache_dir,
}


def __getattr__(name: str) -> str:
    # The cache defaults depend on the environment, so they are resolved on
    # access (not frozen at import) while the historical constant names keep
    # working for importers.
    try:
        return _LAZY_DEFAULTS[name]()
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


def gms_intermediate_base() -> Path:
    """Root of the per-project ``<project>/_stages`` stage cache drive runs use
    (``GMS_INTERMEDIATE_DIR`` or the cache home); ``cleanup`` scans it."""
    return Path(os.environ.get("GMS_INTERMEDIATE_DIR", "") or _default_intermediate_dir())


def gms_intermediate_dir(
    output_base: Path,
    project_name: str,
    intermediate_folder: str = "",
) -> Path:
    """Return ext4-backed staging dir when output is on drvfs, else output_base/_stages.

    drvfs (``/mnt/`` paths) runs at ~50-80 MB/s. Redirecting intermediates to
    ext4 gives ~3-4x throughput for heavy I/O phases (stop removal, color
    correction, segment extraction).

    Resolution order:
      1. ``GMS_INTERMEDIATE_DIR`` env var (absolute override)
      2. ``intermediate_folder`` config param (used only when on drvfs)
      3. the cache home (used only when on drvfs)
      4. ``output_base`` itself (when already on ext4)
    """
    env_override = os.environ.get("GMS_INTERMEDIATE_DIR", "")
    if env_override:
        return Path(env_override) / project_name / "_stages"
    if str(output_base).startswith("/mnt/"):
        base = intermediate_folder if intermediate_folder else _default_intermediate_dir()
        return Path(base) / project_name / "_stages"
    return output_base / project_name / "_stages"
