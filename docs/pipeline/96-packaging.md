### Packaging (`docker/`)

`docker/console.Dockerfile` + `docker/console-compose.yaml` build the console as a Docker service
(non-root, ffmpeg on PATH, the two sibling packages copied from side-by-side checkouts and installed ahead of this one, see `docker/console.Dockerfile`; CI installs the same packages from the revisions in `ci/siblings.txt`);
state lives in a named `console-data` volume plus a nightly `docker/console-backup.sh` copy to the
directory named by `CONSOLE_BACKUP_DIR` (`docker/console-restore.sh` is the inverse). See
[`docs/install.md`](install.md) for the non-developer setup path.
