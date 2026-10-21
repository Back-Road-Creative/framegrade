# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `<GMS_INPUTS_DIR>/known_ids.txt` lists the channel's private and unlisted
  video ids that no uploads listing returns, one per line (`#` comments,
  repeats dropped). `site_sync/known_ids.py` `load_known_ids()` reads it;
  `build_channel_manifest` and `library-hygiene`'s `list_uploads` default to
  it instead of the package seed `FROZEN_KNOWN_IDS`, which stays empty. An
  absent file means no extra ids.
- `GMS_INPUTS_DIR` names the operator's own inputs directory
  (`paths.gms_inputs_dir()`), default `operator-inputs/` under the data home.
  The console reads Route Story launch profiles from its `route_story_profiles/`
  subdirectory, ahead of the existing locations. An empty value counts as unset;
  a value that is not an existing directory raises `InputsDirError` (a
  `ValueError`) naming the variable and the path, and the console answers 422 on
  `GET /route-stories/profiles` and 400 on a film launch with that message.
- CI runs a real clip through the grading path, from frame sampling through the
  LUT-baked libx265 encode, and checks the encoded pixels
  (`tests/test_grading_path_real_clip.py`). A missing `ffmpeg` or `libx265`
  fails it in CI instead of skipping it.

### Changed

- Default locations are per-user instead of host-specific. Inputs, outputs,
  photo state and exports default to `input/`, `output/` and `photos/` under
  the data home (`GMS_DATA_HOME`, else `$XDG_DATA_HOME/gomoveshift`, else
  `~/.local/share/gomoveshift`). Caches and lock files live under the cache
  home (`GMS_CACHE_HOME`, else `$XDG_CACHE_HOME/gomoveshift`, else
  `~/.cache/gomoveshift`). Credential, token, key and SOPS-config files live
  under the config home (`GMS_CONFIG_HOME`, else `$XDG_CONFIG_HOME/gomoveshift`,
  else `~/.config/gomoveshift`). Every existing `GMS_*` variable keeps its name
  and still overrides its default; set the ones you relied on explicitly.
- Features that depend on a separate checkout or service (a site checkout, the
  SEO data directory, an intel database) have no default location. Unset means
  "not configured", and the message names the variable to set.
- Console container packaging: host paths in `docker/console-compose.yaml` are
  required variables with no defaults, and `GMS_SITE_HOST_PATH`,
  `GMS_NIGHT_HOST_ROOT` and `GMS_VID_OUTPUT_DIR` take effect only with their
  overlay file (`docker/console-site.yaml`, `docker/console-extra-output.yaml`).
  `docker/console-backup.sh` names extra volumes through
  `CONSOLE_OPTIONAL_VOLUME_NAMES`. The `GMS_HOST_GID` default is now `1000`; a
  deployment whose mounted directories are owned by another gid must set
  `GMS_HOST_GID` explicitly. See `docker/README.md`.
- The Waypost showcase bundle is stamped with the nearest tag of the checkout
  (`git describe --tags`).

### Removed

- Monorepo-specific helper scripts, the multi-tenant console overlay and its
  reprovisioning script, and internal governance documents.

### Fixed

- `docs/pipeline/96-packaging.md` linked the install guide as `install.md`,
  which resolves to `docs/pipeline/install.md` and 404s; it now links
  `../install.md`. `tests/test_docs_relative_links.py` fails any relative
  Markdown link in the repo root or `docs/` that does not resolve.
