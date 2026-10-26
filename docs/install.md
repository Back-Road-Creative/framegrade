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

Needs Python 3.12, `ffmpeg` 8.1 or newer, and `uv`. CI runs a pinned 8.1 static
build (`.github/workflows/ci.yml`). Older releases such as Ubuntu 24.04's 6.1 make
the blur composite's empty-mask pass inexact (`tests/test_blur_composite_depth.py`).

```bash
git clone --branch v0.3.4 https://github.com/Back-Road-Creative/framegrade.git framegrade
cd framegrade
bash ci/install-cpu.sh        # CPU install into .venv
```

`ci/install-cpu.sh` builds `.venv` from `requirements.lock` with the CPU
inference wheels, which is what a machine without CUDA needs. It installs the
two siblings from `ci/siblings.txt`, each pinned to one reviewed revision. Point
`SIBLINGS_FILE` at a file holding `-e ../waypost` and `-e ../gps-journey` instead
when the three repositories sit side by side and you want your own checkouts.

**GPU host.** The default dependency set already pins `onnxruntime-gpu[cuda,cudnn]`
(the CUDA 12 runtime wheels), so install the siblings and then this project
into a fresh virtual environment instead of using the CPU script. The first
install uses the same reviewed sibling revisions as the CPU script; to use your
own side-by-side checkouts, run `.venv/bin/pip install -e ../waypost -e ../gps-journey`
in its place:

```bash
python3 -m venv .venv
.venv/bin/pip install -r ci/siblings.txt
.venv/bin/pip install -e '.[dev]'
```

`tools/enable_gpu_detection.sh <venv>` repairs an environment where a CPU
`onnxruntime` wheel replaced the GPU build, and preflight refuses to start a
blur run that has no CUDA provider.

Optional extras: `signs` (road-sign OCR for `--sign-verify`; pulls torch),
`lens` (lensfun profile correction for photos) and `gopro` (GoPro telemetry;
see "`--ingest-mode gopro`" in the README for the extra `--no-deps` install).

Check the result with `gomoveshift doctor` (below).

**Windows.** Native Windows is not supported yet; run the console in Docker
(below). `.github/workflows/windows.yml` runs `ci/install-cpu.sh` on a hosted
Windows runner and then `ci/import_smoke.py`, which imports every module and
fails on any that breaks outside `ci/windows-import-failures.txt`. That list
names what still blocks a native run (today, modules that use the Linux-only
`fcntl` file lock); a change that fixes a module deletes its line, and the job
fails until it does. The job becomes a required check once the list is empty.

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

**Sealed seo tokens live in the operator's inputs directory.**
`credential_paths.resolve_sops_file` reads `$SEO_INPUTS_DIR/<file>` first (default `seo` under `GMS_INPUTS_DIR`) for the
configured `seo.credentials_dir` (when no brand-scoped dir applies), and falls back to
`<file>` next to that credentials dir, so an install that has not moved its tokens yet
keeps working.

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
inside the notification payload) point the failure webhook at an endpoint of your choice. With no
usable destination (unset, or an ntfy URL whose topic is invalid) nothing is sent and nothing is
recorded as delivered: `/healthz` shows `notify_configured: false`, and every alert stays pending until
a destination exists. An alert is recorded as seen only once it was delivered; a failed push is retried
with bounded exponential backoff (`GMS_NOTIFY_MAX_ATTEMPTS`, default 6; `GMS_NOTIFY_BACKOFF_MIN`,
default 2) from `notify_outbox.json` in the console data dir, so a restart loses nothing. The first scan
notifies the currently actionable backlog once (at most 10 actual pushes per pass; rows that are out of
attempts or backing off do not use up that cap). **Known, accepted limitation:** `job_failed` and
`job_interrupted` alerts stay fire-and-forget (one push, no outbox retry), so a failed push for those is
not covered by the retry guarantee above. `/healthz` reports the effective
destination (`notify_source` env or file, never the topic), `notify_pending`, `notify_exhausted`,
`notify_delivery_failures` and the ticker's `observer_failures`. For phone pushes, run `python3 tools/phone_alerts_setup.py` from the checkout (system
python, no venv): it makes a private ntfy.sh topic in `notify_url` under the config home (used when the
env var is unset), prints the phone steps, and `--test` sends one push. TikTok inbox drafts (with their
caption), failed posts and missed slots then reach the phone. `--rotate` replaces the topic.

In the container the home file is unreachable, so pass the destination as
`GMS_CONSOLE_NOTIFY_URL` and compose forwards it to the console. An absent or empty value is not a
failure -- the console starts unconfigured and `/healthz` reports `notify_configured: false`. The value
is never printed.

**Backend runs the console did not launch.** Review gates are discovered for every console job *and*
for run dirs found under `GMS_OUTPUT_DIR` (the root, its children and grandchildren holding `_gates`),
so a CLI or cron drive's gate lists on `/drives/gates/pending` and alerts the phone. Each run is kept
in the `backend_runs` table of `jobs.sqlite3` with its origin: `console` (job row: argv and start
time), `cli` / `cron` (from an optional `run_origin.json` in the run dir: `origin`, `argv`,
`started_at`), or `unknown` -- never guessed. Known provenance is never overwritten; `unknown` is
upgraded when a sidecar appears. Nothing outside the output root is scanned, and the scan writes nothing.

### 3. Run a dry drive

Drives tab -> pick a source folder -> run a **dry** (non-uploading) drive.
Proves the install end-to-end without shipping anything.
For a full render that must remain local, run the dry drive first, review its
plan, then clear **Dry run**, select **I reviewed the dry-run plan; proceed with
a full render**, and select **Render without uploading to Drive or YouTube**.
The confirmation is required for a full console run because the detached
worker has no visible terminal prompt. Only that explicit choice passes
`--force`; a rejected form preserves it for correction. The console also refuses, before queueing, a source folder with a broken symlink or an unreadable media file (each path is named); the host worker's own check stays authoritative. The console passes `--skip-upload`
and `--no-youtube-upload`, so the pipeline treats the scheduled run as
audit-only. The publishing and long-form days still plan it, but nothing is
uploaded. Leave review enabled for production acceptance checks.
For a run whose on-screen place text should be checked against its retained
GPS, select **Use GPS to review on-screen locations**. This
passes `--ost-system-review` for that run only. The system can pass the OST
gate only when its sealed route-claim record agrees on every long-form and
matches the exact burn text; an unsupported claim or missing evidence still
stops for review. The console does not create a `.confirmed` marker or pass
`--ost-audit-confirmed`.

The **Shorts** tab launches the same drive pipeline. For a full render there,
review the dry-run plan and select the plan confirmation; it refuses an
unconfirmed full run before creating a job. The same upload-hold and GPS
system-review choices are available. A validation error keeps the entered
source, output, schedule, and choices on the form for correction. Shorts
launched from this tab still use the pipeline's automated review gates.

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

## Minimum machine

Most of the checks below are what `gomoveshift drive` enforces before an encode
(`run_preflight` in `gomoveshift/pipeline/preflight.py`); a run on a smaller
machine stops before any encode and lists every check that failed. The NVENC row
is checked separately by `detect_nvenc`.

| resource | minimum |
| --- | --- |
| Free disk at the output directory | 50 GB |
| Free disk at the staging directory | 30 GB |
| Available RAM | 2 GB |
| `ffmpeg` and `ffprobe` on `PATH` | present (8.1 or newer, per the local install above) |
| NVENC-capable ffmpeg and GPU | required for the drive pipeline (see GPU (NVENC) above) |
| CUDA execution provider for `onnxruntime` | required for a run with blur on |

On WSL the Windows volume behind the Linux disk must also keep 25 GB free, even
when resuming. These are floors, not recommendations: a long day of footage needs
more staging space than 30 GB, and preflight raises that number from the footage
it is given. GPU memory is not checked. Preflight prints a note, and never
refuses the run, when the machine has fewer than 4 logical CPU cores or less than
8 GB total RAM: it will run, only slower. `gomoveshift doctor --calibrate CLIP`
measures how fast your machine encodes so the plan's ETA is based on it.

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

## Schedule tab volume figures

`/schedule` shows four separate shorts-per-long-form figures from `gomoveshift/console/schedule_model.py`: the selected target (Waypost's `shorts_per_lf` decision), slot capacity (the region's shorts windows × the LF interval), feasible output (the lower of the two, labelled as derived, with the binding side), and actually scheduled (content-ledger rows). Each carries its source, a fallback reason (a `seed` value, an unreadable decision store, a ledger figure that has no policy version) and the policy revision. The model only reads `cadence_decisions` and the schedule report; it writes nothing. Times render in America/New_York (EST/EDT).

**Who writes the origin.** `gomoveshift drive` and `gomoveshift pipeline` write `run_origin.json` into their run dir when they start (`gomoveshift/run_origin.py`; not on `--dry-run`, never overwritten on resume). The origin comes from `GMS_RUN_ORIGIN`: the console sets `console` on every child it launches, a cron entry would set `GMS_RUN_ORIGIN=cron` (planned: no tracked cron launches drives today), and anything else is `cli`. The sidecar keeps only the program name and subcommand, never paths or other arguments.
