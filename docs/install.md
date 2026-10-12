# Installing GoMoveShift

Two ways to run it from a standalone checkout of this repository: the console
in Docker (no Python, ffmpeg or dev tooling on the host, just Docker), or a
local Python environment for the command line tools and the test suite.

The package depends on two sibling projects, `waypost` (provides the
`seo_manager` package) and `gps-journey`. Neither is on PyPI. Both install
paths below expect them as checkouts next to this one:

```
parent/
  framegrade/    # this repository
  gps-journey/
  waypost/
```

## Local install (command line, tests)

Needs Python 3.12, `ffmpeg` and `uv`.

```bash
git clone <this repo> framegrade
cd framegrade
bash ci/install-cpu.sh        # CPU install into .venv
```

`ci/install-cpu.sh` builds `.venv` from `requirements.lock` with the CPU
inference wheels, which is what a machine without CUDA needs. It installs the
two siblings from `ci/siblings.txt`; fill in the owner and revision
placeholders there first, or point `SIBLINGS_FILE` at a file holding
`-e ../waypost` and `-e ../gps-journey` when the three repositories sit side by
side.

**GPU host.** The default dependency set already pins `onnxruntime-gpu[cuda,cudnn]`
(the CUDA 12 runtime wheels), so install the siblings and then this project
into a fresh virtual environment instead of using the CPU script:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ../waypost -e ../gps-journey
.venv/bin/pip install -e '.[dev]'
```

`tools/enable_gpu_detection.sh <venv>` repairs an environment where a CPU
`onnxruntime` wheel replaced the GPU build, and preflight refuses to start a
blur run that has no CUDA provider.

Optional extras: `signs` (road-sign OCR for `--sign-verify`; pulls torch),
`lens` (lensfun profile correction for photos) and `gopro` (GoPro telemetry;
see "`--ingest-mode gopro`" in the README for the extra `--no-deps` install).

Check the result with `gomoveshift doctor` (below).

### Where files go

Defaults are per user and resolve under three roots, the first that is set
winning: the data home (`GMS_DATA_HOME`, else `$XDG_DATA_HOME/gomoveshift`,
else `~/.local/share/gomoveshift`) for inputs, outputs and photo state; the
cache home (`GMS_CACHE_HOME`, else `$XDG_CACHE_HOME/gomoveshift`, else
`~/.cache/gomoveshift`) for caches and lock files; and the config home
(`GMS_CONFIG_HOME`, else `$XDG_CONFIG_HOME/gomoveshift`, else
`~/.config/gomoveshift`) for `config.yaml` and the credential, token, key and
SOPS-config files. Every more specific `GMS_*` variable keeps its name and
still overrides its default; the README's environment table lists the ones that affect a drive or photo run.

Features that depend on a separate checkout or service have no default
location and report "not configured" until you set them. For example,
`photos.site_root` in `config.yaml` (the Hugo site checkout the journey photos
live in) must be set before the photo pipeline can read that site.

## Console in Docker

### 1. Install Docker

- Windows/WSL: [Docker Desktop](https://www.docker.com/products/docker-desktop/),
  WSL integration enabled for your distro.
- Linux: `docker` + `docker compose` from your distro's package manager.

### 2. Start the console

The image is built from the parent directory shown above, so the three
checkouts must sit side by side. Every host path in the compose file is a
required variable with no default (see `docker/README.md` for the full table):
set each one to an existing path, then:

```bash
cd framegrade
docker compose -f docker/console-compose.yaml up -d --build gomoveshift-console
```

Name the service explicitly -- `console-compose.yaml` also defines a shared, never-standalone
`console-base` template that `gomoveshift-console` extends.

Open **http://localhost:8787**. `restart: unless-stopped` means it comes back
on its own after a reboot or Docker Desktop restart -- nothing else to run.

The published port defaults to loopback (`127.0.0.1:8787:8787`) -- the console is reachable only
from the host, not the LAN. To expose it elsewhere, set `GMS_CONSOLE_BIND` to the address to bind
before starting the container (e.g. `export GMS_CONSOLE_BIND=0.0.0.0`), and set `GMS_CONSOLE_TOKEN`
first (below) -- an unauthenticated console should never be reachable off the host.

**A brand is a tenant for DATA.** Different brands must not see each other's delivery queue
(it is a per-brand sqlite file keyed by SEO storage path -- see
`tests/test_publishing_brand_isolation.py`). The compose file runs one console
instance, so it binds one SEO storage directory and one SEO credentials directory
(`GMS_SEO_STORAGE_DIR`, `GMS_SEO_CREDENTIALS_DIR`). To serve a second brand, run a
second compose project with its own copy of these variables pointing at separate
host directories.

**Encrypted credentials are bind-mounted read-only.** Each already-encrypted host file
is mounted as is; nothing is copied, decrypted or duplicated:

| purpose | host path variable |
| --- | --- |
| YouTube token | `GMS_SEO_YT_TOKEN_ENC` |
| Drive credentials | `GMS_DRIVE_CREDENTIALS_ENC` |
| Drive token | `GMS_DRIVE_TOKEN_ENC` |
| age key | `GMS_AGE_KEY_FILE` |
| SOPS config | `GMS_SOPS_CONFIG` |

Provision these files before starting. **Docker auto-creates a missing bind source**
as an empty root-owned directory, so a guessed default would stand an empty directory
in for an age key or a SOPS config and the console would look healthy while unable to
decrypt anything. That is why the compose file requires each variable and stops with a
message naming the one that is missing. The container-side paths are fixed, so an
override moves only which *host* file is bound.

Two optional overlays add what the base file leaves out: `docker/console-site.yaml`
(a site checkout and a camera-card root, for the media library and night-import tabs;
without it they report "not configured") and `docker/console-extra-output.yaml`
(a second output root). Add them with repeated `-f` options; each needs its own
variables, described in `docker/README.md`.

The image ships `sops`, and the compose file bind-mounts (read-only) the paths above.
The image's own `seo.*` config (baked in at
`/home/console/.config/gomoveshift/config.yaml`) already points at the in-image copy of
the SEO project, so these mounts exist to keep the container's view of that data live
rather than frozen at build time; publishing/dispatch fails without them. The container
ticks its schedule on its own
(`GMS_CONSOLE_TICKER=1`); a `/healthz` healthcheck reports container health to `docker ps`. `GMS_SUPERVISED=1`
asserts that this container's own restart policy and the console's job store supervise every drive it
launches, so console-launched drives run detached (`--no-tmux`) instead of failing the CLI's
tmux-supervision gate -- no `tmux` binary is installed in the image.


### Auth token and notifications

`GMS_CONSOLE_TOKEN` gates every console request (empty by default -- auth off). To require a bearer
token, export it in the host environment before starting the container:

```bash
export GMS_CONSOLE_TOKEN=<a long random string>
docker compose -f docker/console-compose.yaml up -d --build gomoveshift-console
```

With a token set, open the console at `http://localhost:8787/?token=<the same string>` -- the first
request redirects to the same page with `?token=` stripped from the address bar (it would otherwise
sit in browser history and the `Referer` header of every outbound link) and sets a session cookie, so
subsequent requests and a bookmark of the plain URL keep working. The token compare is constant-time.
The session cookie is marked `Secure` only when the console is actually reached over HTTPS -- plain
`http://` (the default, loopback) keeps working, and the check is never based on a forwarded-proto
header, which a client can set arbitrarily. `/healthz` stays reachable without a token (the compose
healthcheck can't carry one), but an unauthenticated `/healthz` response is minimal (`{"status":
"ok"}`/`"degraded"`, same HTTP status code) -- pass the token to see brand slugs, ticker, and observer
state. `GMS_CONSOLE_NOTIFY_URL` (and optionally `GMS_CONSOLE_URL`, used only to compose the link
inside the notification payload) point the failure webhook at an endpoint of your choice; unset, it's
a silent no-op. For phone pushes, run `python3 tools/phone_alerts_setup.py` from the checkout (system
python, no venv): it makes a private ntfy.sh topic in `notify_url` under the config home (used when the
env var is unset), prints the phone steps, and `--test` sends one push. TikTok inbox drafts (with their
caption), failed posts and missed slots then reach the phone. `--rotate` replaces the topic.

### 3. Run a dry drive

Drives tab -> pick a source folder -> run a **dry** (non-uploading) drive.
Proves the install end-to-end without shipping anything.

## Check the install

`gomoveshift doctor` runs a one-shot, read-only, no-network health report -- deps, ffmpeg/ffprobe on
PATH and the filters the pipeline burns, GPU/NVENC availability, Drive credential files present on
disk, staging storage writable, and the SEO-storage tenant stamp. Run it inside the console
container (or any environment with the package installed) before a dry drive, or whenever an
upgrade changes the ffmpeg build or credential paths:

```bash
docker compose -f docker/console-compose.yaml exec gomoveshift-console gomoveshift doctor
```

Prints one `OK`/`WARN`/`FAIL` line per check and exits non-zero only when a check FAILs -- a missing
optional extra or absent GPU WARNs (both are normal on a CPU-only or minimal install) and never fails
the run. `--json` emits the same report as machine-readable JSON.

## GPU (NVENC)

The drive pipeline needs an ffmpeg that can encode HEVC with NVENC; it probes
for one (`detect_nvenc`) and stops with `NVENCUnavailableError` rather than
falling back to a CPU encode. `gomoveshift doctor` reports the same probe as a
WARN on a machine without it. In the console container, install the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
and run the container with `--gpus all` (or the compose `deploy.resources`
equivalent). The tests and commands that need such a machine are listed in
[gpu-checks.md](gpu-checks.md).

## Backup and restore

State (settings, job history, ledgers) lives in the `console-data` volume, not
the image. Back it up nightly to somewhere that survives loss of the machine's own disk
(`CONSOLE_BACKUP_DIR` defaults to `console-backup` under the data home; see `docker/README.md`):

```bash
CONSOLE_BACKUP_DIR=/path/to/console-backup bash docker/console-backup.sh
```

Keeps the last 14 date-stamped archives. Restore into a fresh volume:

```bash
bash docker/console-restore.sh /path/to/console-backup/docker_console-data-YYYY-MM-DD-HHMMSS.tar.gz docker_console-data
```

**Prove this once for real** against a genuinely empty volume before relying
on it -- an untested backup script is not a backup plan.

## Surviving a rebuild

Nothing in the container is unique: rebuild the image from this checkout, restore
the volume if it was lost, then `docker compose -f docker/console-compose.yaml up -d --build gomoveshift-console` again.
