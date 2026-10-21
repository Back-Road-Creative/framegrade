# docker/

Console container packaging: `console.Dockerfile`, `console-compose.yaml`,
`console-site.yaml` and `console-extra-output.yaml` (optional overlays),
`console-config-override.yaml`, and the backup/restore scripts. Usage detail lives in `docs/install.md`.

## Build context

The image is built from a directory that holds three sibling checkouts side by
side: `framegrade`, `gps-journey` and `waypost` (the last provides the
`seo_manager` package). `console-compose.yaml` sets the build context to that
parent directory, so run compose from a checkout laid out like this:

```
parent/
  framegrade/    # this repository
  gps-journey/
  waypost/
```

`console.Dockerfile.dockerignore` sits beside the Dockerfile and BuildKit reads
it for this context. It keeps host virtualenvs and caches (`.venv`,
`__pycache__`, test and lint caches) out of the image, which pip-installs its
own packages: a checkout's multi-gigabyte `.venv` would otherwise ride in via
`COPY framegrade`, bloating the image and slowing the layer export.

## Required host paths

Every host path in `console-compose.yaml` is a required variable with no
default. Docker auto-creates a missing bind source as an empty root-owned
directory, so a guessed default would put a directory where a file belongs and
the console would look healthy while unable to decrypt anything. Compose stops
with a message naming the variable instead. This is a deliberate exception to
the per-user data/cache/config-home defaults used by the command line tools
(`GMS_DATA_HOME` and friends): a bind source must be a path the operator has
created. Set each one to an existing path:

| Variable | Host path |
| --- | --- |
| `GMS_STAGING_DIR` | source footage (mounted at the same path inside) |
| `GMS_OUTPUT_DIR` | pipeline output (mounted at the same path inside) |
| `GMS_DRIVE_CREDENTIALS_ENC` | encrypted Drive credentials file |
| `GMS_DRIVE_TOKEN_ENC` | encrypted Drive token file |
| `GMS_AGE_KEY_FILE` | age private key file |
| `GMS_SOPS_CONFIG` | `.sops.yaml` file |
| `GMS_SEO_STORAGE_DIR` | SEO storage directory (holds the publishing queue) |
| `GMS_SEO_CREDENTIALS_DIR` | SEO credentials directory |
| `GMS_SEO_YT_TOKEN_ENC` | encrypted YouTube token file |
| `GMS_INPUTS_HOST_DIR` | the operator's own inputs directory (mounted read-only; an empty directory is valid) |
| `GMS_HOST_SPOOL_HOST_DIR` | spool directory shared with the host GPU worker |

The container-side targets are fixed (`/secrets/...`, `/app/waypost/...`,
`/inputs`, `/host-spool`), and the matching `GMS_*` variables inside the
container (`GMS_INPUTS_DIR`, `GMS_HOST_SPOOL_DIR`, ...) point at them. A host
override therefore moves only the host source, never where the app looks.

Optional variables: `GMS_HOST_GID` (default `1000`), `GMS_CONSOLE_BIND`
(default `127.0.0.1`), `GMS_CONSOLE_TOKEN`, `GMS_CONSOLE_NOTIFY_URL`,
`GMS_CONSOLE_URL`, `GMS_GPU_LOCK_PATH`, `GMS_COMMENTS_PATH`,
`GMS_GATE_ANSWER_TIMEOUT_SEC`.

```
docker compose -f docker/console-compose.yaml up -d --build gomoveshift-console
```

Always name the service: `console-base` is a shared template that is not meant
to run on its own.

The image declares a `/healthz` `HEALTHCHECK`, so a bare `docker run` reports
health too; the compose healthcheck overrides it.

## GPU work

The container has no GPU unless it is started with `--gpus all`, and the drive
pipeline requires NVENC with no CPU fallback, so a console-launched drive fails
the NVENC check inside a GPU-less container. The compose file instead sets
`GMS_HOST_SPOOL_DIR=/host-spool`: the console writes GPU job requests into the
spool shared with the host worker (`docs/host-worker.md`), and the worker writes
status, heartbeat and logs back. Point `GMS_HOST_SPOOL_HOST_DIR` at the
directory the worker is started with.

## SEO paths

`console.Dockerfile` copies `waypost` into the image and installs
`console-config-override.yaml` as the console user's config, which
`Config.load()` merges over the packaged defaults (cwd-relative `seo.*` paths
that do not hold the SEO data in this image), so `seo.storage_path`, `seo.credentials_dir` and `seo.manager_path`
resolve under `/app/waypost` inside the container. The compose file bind-mounts
the live host data over those same in-image paths. The encrypted YouTube token
sits one level above the credentials directory, so it is mounted on its own
line.

## Site checkout and camera cards (optional)

The media library and night-import tabs report "not configured" by default. To
enable them, add the overlay and set both of its variables:

```
docker compose -f docker/console-compose.yaml -f docker/console-site.yaml \
  up -d --build gomoveshift-console
```

`GMS_SITE_HOST_PATH` (a site checkout) mounts read-only at `/site` and
`GMS_NIGHT_HOST_ROOT` (a camera-card root) mounts read-only at `/night-source`;
the overlay sets `GMS_SITE_PATH` and `GMS_NIGHT_SOURCE_ROOTS` to those targets.

## Second output root (optional)

If runs also write under an output root other than `GMS_OUTPUT_DIR`, add the
`console-extra-output.yaml` overlay and set `GMS_VID_OUTPUT_DIR`:

```
docker compose -f docker/console-compose.yaml -f docker/console-extra-output.yaml \
  up -d --build gomoveshift-console
```

The directory is mounted read-write at the same path inside the container,
because queue rows carry absolute sidecar paths under it; without the mount
those rows fail with "could not read sidecar". The variable is required once
the overlay is used and has no default. Both overlays can be combined with
repeated `-f` options.

## Upgrading

- `GMS_SITE_HOST_PATH`, `GMS_NIGHT_HOST_ROOT` and `GMS_VID_OUTPUT_DIR` are
  honoured only with their overlay file (`-f docker/console-site.yaml`,
  `-f docker/console-extra-output.yaml`). Setting them with the base compose
  file alone mounts nothing: add the overlay to the existing command line, or
  the media library and night tabs stay "not configured" and rows under the
  second output root cannot be read.
- Host paths in `console-compose.yaml` no longer have defaults; set every
  variable in the table above before starting the console. That includes
  `GMS_INPUTS_HOST_DIR` and `GMS_HOST_SPOOL_HOST_DIR`, the inputs and
  host-worker spool mounts added later.
- The `GMS_HOST_GID` default is now `1000`. A deployment whose mounted
  directories are owned by another group must set `GMS_HOST_GID` explicitly,
  or the rebuilt container joins a different host group and loses write
  access to the group-owned staging, output and SEO mounts.
- `console-backup.sh` names optional volumes through
  `CONSOLE_OPTIONAL_VOLUME_NAMES` (space-separated). A cron environment that set
  a per-instance volume variable from an earlier layout must move its value
  there, or that volume is no longer backed up. When the list is empty and the
  host holds another `*console-data*` volume, the script prints a `note:` line
  on stderr naming it (the run still succeeds).

## Host data group

The container user is uid 999 (`console`, created in `console.Dockerfile`). The
staging and SEO storage mounts are host directories, usually group-writable
(mode 2775). `console-base` adds the container user to the host group with
`group_add` (`GMS_HOST_GID`, default `1000`); set it to the gid that owns the
mounted directories. A recreated container is needed for a compose change to
take effect.

## Backup

Compose is run from this directory, so its project name is `docker`; the
`console-data` service volume is pinned to `docker_console-data` in
`console-compose.yaml`. `console-backup.sh` checks that a volume exists before
touching it and refuses to let docker silently auto-create the wrong one.

- `CONSOLE_VOLUME_NAME` (default `docker_console-data`) is required: the script
  exits non-zero if it is missing or nothing was backed up.
- `CONSOLE_OPTIONAL_VOLUME_NAMES` (space-separated, default empty) names further
  volumes whose absence is logged but never fails the run.
- `CONSOLE_BACKUP_DIR` defaults to `console-backup` under the data home
  (`GMS_DATA_HOME`, else `$XDG_DATA_HOME/gomoveshift`, else
  `~/.local/share/gomoveshift`). Point it at a location that survives loss of
  the machine's own disk. The last 14 archives per volume are kept.

`bash docker/console-backup.sh --restore-drill` proves a backup is restorable
rather than just written: it opens the newest archive for each required volume,
extracts it, and integrity-checks any `.sqlite3`/`.sqlite`/`.db` file it finds.
It needs no docker and no running containers. `restore-drill OK` means every
required volume's newest archive extracted and any database passed
`PRAGMA integrity_check`; `restore-drill INCOMPLETE` (still exit 0) means
`sqlite3` is not on the PATH so database content was not checked; a missing
archive or a failed integrity check is `restore-drill FAILED` with a non-zero
exit.

Example nightly crontab line:

```
30 3 * * * cd /path/to/framegrade && bash docker/console-backup.sh >> /path/to/console-backup.log 2>&1
```

Restore requires the target volume name explicitly:

```
bash docker/console-restore.sh <backup-tar.gz> docker_console-data
```
