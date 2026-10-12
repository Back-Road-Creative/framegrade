## Doctor (`cli/handlers/doctor.py`, `gomoveshift doctor`) <a id="doctor"></a>

One-shot, read-only install-health report -- no network calls. Every check reads the filesystem,
PATH, or runs a local ffmpeg/ffprobe subprocess against synthetic test input; nothing opens a
socket. Prints one line per check (`OK`/`WARN`/`FAIL`, `run_all_checks` → `format_report`) and
exits non-zero only when a check FAILs -- a missing optional extra or absent GPU WARNs, never
fails the run. `--json` emits the same report as a JSON array of `{name, status, detail}`.

Checks, in order:

1. **Python extras importable** -- `EXTRAS_IMPORT_MAP` maps each `[project.optional-dependencies]`
   extra (`signs`, `gpu`, `gopro`, `route-story-3d`; `dev` excluded -- CI/test-only, not a runtime
   pipeline capability) to the import name(s) it provides, which differ from the PyPI distribution
   name (`onnxruntime-gpu` imports as `onnxruntime`, `gopro-overlay` as `gopro_overlay`). A missing
   module WARNs, listing the `pip install -e '.[extra]'` to fix it.
2. **ffmpeg/ffprobe on PATH** (`check_ffmpeg_tools`) -- `shutil.which`; FAILs if either is absent.
3. **ffmpeg filters registered** (`check_ffmpeg_filters`) -- runs `ffmpeg -filters` and confirms
   every name in `_REQUIRED_FILTERS` is present. That list is HAND-LISTED (grepped once against
   this service's own filter-construction call sites in `pipeline/ffmpeg_filters.py`,
   `pipeline/color_filters.py`, `pipeline/color.py`), not derived from a runtime registry -- there
   is no "filters this pipeline uses" contract to import; re-grep those files if a new filter is
   added and this list goes stale.
4. **GPU/NVENC presence** (`check_gpu_nvenc`) -- reuses `pipeline.color_filters.detect_nvenc`
   directly rather than re-implementing the probe. `detect_nvenc` itself is unconditional (no
   CPU-fallback escape hatch) for the Drive pipeline; doctor is the one caller that WARNs
   on its failure instead of raising, since a CPU-only dev box is a normal thing to run doctor on.
5. **Drive credentials present** (`check_drive_credentials`) -- reuses
   `pipeline.preflight.check_drive_credentials` (the same four-env-var spec the Drive pipeline
   itself gates on) rather than duplicating its file-existence spec; never opens or logs the
   credential file contents, existence only. There is no per-brand credential split to check here:
   `GMS_SEO_STORAGE_DIR` (one value per console instance, see `console/tenant.py`) is the console's SEO-storage
   *tenancy* boundary, not a second set of Drive OAuth credentials -- this service has exactly one
   shared Drive credential set, so doctor checks that one set rather than iterating a brand list
   that doesn't exist in `pipeline.uploader`.
6. **Storage writable** (`check_storage_writable`) -- `GMS_INTERMEDIATE_DIR` when set, else the
   default intermediate cache (under the cache home: `$GMS_CACHE_HOME`, else
   `$XDG_CACHE_HOME/gomoveshift`, else `~/.cache/gomoveshift`; see `paths.py`); creates the directory and write+removes a throwaway probe file.
   Inside a container whose environment carries the overlay variables, `check_overlay_mounts` also WARNs for each of `GMS_SITE_HOST_PATH`,
   `GMS_NIGHT_HOST_ROOT` and `GMS_VID_OUTPUT_DIR` that is set while its target (`/site`,
   `/night-source`, the variable's own path) is not a mount -- the matching compose overlay is missing.
7. **Tenant stamp readable** (`check_tenant_stamp`) -- read-only look at `config.seo.storage_path`'s
   `.waypost-tenant` stamp (`console/tenant.py::_stamp_path`). A missing stamp is OK, not a
   failure -- normal on a fresh install, before the first real run claims it. An unreadable file
   FAILs; an empty/whitespace stamp WARNs (unknown owner, same "can't tell whose this is" case
   `console/tenant.py`'s `claim_storage` treats as unclaimed-but-suspicious).
8. **YouTube channel identity, one result per brand** (`check_youtube_identity`) -- unlike Drive,
   YouTube upload tokens ARE brand-scoped (`publishing.credential_paths.resolve_sops_file`), so
   this iterates `console.settings.known_brands()`. For each brand: FAILs only when no token file
   (`youtube_upload_token.json.enc`) exists at all; OKs (naming the channel title/id) when the
   token exists and `pipeline._yt_creds.read_channel_identity` finds a readable identity sidecar
   next to it; otherwise WARNs, naming the exact `gomoveshift youtube-auth --brand <brand>` re-mint
   command that writes one. The sidecar is a plain, non-secret JSON file the mint step writes
   alongside the encrypted token -- this check never calls `sops` and never makes a network call.

Every check function takes its collaborators (`which`, `run`, `detect`, `check`, `import_module`)
as parameters with real-world defaults, so `tests/test_doctor.py` exercises each one with a fake
collaborator and never depends on a real GPU, ffmpeg build, or filesystem outside `tmp_path`.

### Encoder ladder
- **`encoders` check** (right after `gpu_nvenc`, which is unchanged). Probes every backend in
  `pipeline/encoders.LADDER` (1 s of 256x256 test video each): `OK` when a hardware backend works
  (`hevc_nvenc available; libx265 available (CPU floor)`), `WARN` when only the CPU floor works
  (`only libx265 (CPU floor) available: <hardware reasons>`), `FAIL` when nothing encodes.
- **Policy.** GMS's drive pipeline keeps the NVENC-required rule (NVENC required, no CPU fallback) via
  `detect_nvenc`; `select_encoder(required="hevc_nvenc")` expresses the same rule in ladder terms.
  The ladder only adds the selectable public path. Order: `hevc_nvenc`, `hevc_videotoolbox`
  (Apple), `hevc_qsv`, `hevc_vaapi` (Intel), `hevc_amf` (AMD), `libx265` (CPU floor). The vendor
  flags are from the ffmpeg docs and not yet run on real hardware; `hevc_vaapi` probes unavailable
  until its device/`hwupload` wiring lands.

### Calibrated ETA

Preflight's wall-clock estimate used to come only from a fixed model calibrated once, on one
machine (`preflight_plan._RES_FACTOR` / `_PRESET_FACTOR` / `_MULTIPASS_FACTOR`).
`pipeline/calibration.py` replaces it with a measurement: a 15 s slice of the user's own
footage goes through the real day-master chain (`SinglePassProcessor.process`: grade + OST +
audio + the blur composite when blur is on + encode), and the wall-seconds per source-second is
saved to `calibration.json` under `GMS_INTERMEDIATE_DIR` (default: the cache home, `$GMS_CACHE_HOME`, else `$XDG_CACHE_HOME/gomoveshift`, else
`~/.cache/gomoveshift`).
`estimate_master_encode_seconds(..., calibration=rec)` then returns
`source_duration_s * rec.realtime_factor` and ignores the fixed factors.

- **When it runs.** A real (non-dry) run measures once if no valid record exists
  (`ensure_calibrated`, from `run_preflight_and_prompt`, on the first sorted video in the source
  dir). `--dry-run` only reads a saved record, it never encodes. Failure is advisory: the run
  logs `calibration failed (...)`; ETA falls back to the reference model, and goes on.
- **Staleness.** A record stores `chain_version` (`MASTER_ENCODE_ALGO_VERSION`, a fingerprint of
  the frozen encode constants); any encode-constant change makes it stale and `load_calibration`
  returns `None` with a warning.
- **What the plan prints.** One line before `Estimated wall-clock`:
  `ETA basis: measured on this machine -- <fps> fps (<encoder>, <WxH>, blur on|off, <date>)`, or
  `ETA basis: built-in reference model (not yet measured on this machine; ...)`.
- **Doctor.** `gomoveshift doctor --calibrate CLIP` measures and saves on demand (the one
  doctor path that writes and encodes); the `calibration` check reports `OK` with the rate or
  `WARN` "not measured on this machine".
- **Scope.** One backend today (CPU filters + `hevc_nvenc`). The encoder ladder
  (`pipeline/encoders.py`) exists; calibrating across its backends is not built yet.
