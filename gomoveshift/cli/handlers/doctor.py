"""``gomoveshift doctor`` — one-shot install-health report.

Read-only, no network calls: every check reads the filesystem, PATH, or runs a
local ffmpeg/ffprobe subprocess against synthetic test input — nothing here
opens a socket. Prints one line per check (``OK``/``WARN``/``FAIL``); exits
non-zero only when a check FAILs (a missing optional extra or absent GPU WARNs,
it never fails the run).

Checks reuse existing capability-probing code rather than re-implementing it:
``pipeline.color_filters.detect_nvenc`` for the GPU/NVENC probe and
``pipeline.preflight.check_drive_credentials`` for the Drive credential-file
spec. There is no per-brand credential split for the *Drive* pipeline:
``pipeline.uploader``'s Drive credentials are a single shared set, and the
console's SEO storage is one tenant set by ``GMS_SEO_STORAGE_DIR`` (see
``console/tenant.py``). This doctor checks that one credential set and the
single ``seo.storage_path`` tenant stamp instead of iterating a brand list for
Drive. YouTube tokens ARE brand-scoped (``publishing.credential_paths``), so
``check_youtube_identity`` does iterate ``console.settings.known_brands()``.

The ffmpeg FILTER names checked below are HAND-LISTED (grepped once against
this module's own filter-construction call sites, not derived at doctor
runtime) — see ``_REQUIRED_FILTERS`` for the exact provenance note.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from ...paths import gms_intermediate_base
from ...pipeline.calibration import calibrate_master_chain, load_calibration, save_calibration

OK = "OK"
WARN = "WARN"
FAIL = "FAIL"

#: Optional-dependency extras from ``pyproject.toml`` ``[project.optional-dependencies]``,
#: mapped to the *import* name(s) each extra provides (which differ from the PyPI
#: distribution names: ``onnxruntime-gpu`` imports as ``onnxruntime``, ``gopro-overlay``
#: as ``gopro_overlay``). ``dev`` is excluded — it's a CI/test-only extra, not a runtime
#: pipeline capability an operator would check for.
EXTRAS_IMPORT_MAP: dict[str, tuple[str, ...]] = {
    "signs": ("easyocr",),
    "gpu": ("torch", "onnxruntime"),
    "gopro": ("gopro2gpx", "gopro_overlay"),
    "route-story-3d": ("playwright",),
}

#: HAND-LISTED: the ffmpeg filter names this service's own filter-construction code
#: (``pipeline/ffmpeg_filters.py``, ``pipeline/color_filters.py``, ``pipeline/color.py``)
#: builds into filter graphs. Not derived from ``ffmpeg -filters`` at doctor runtime —
#: there is no registry of "filters this pipeline uses" to import; re-grep this module's
#: docstring reference set if the pipeline starts using a new filter.
_REQUIRED_FILTERS: tuple[str, ...] = (
    "drawtext",
    "scale",
    "crop",
    "format",
    "fps",
    "overlay",
    "colorlevels",
    "curves",
    "loudnorm",
)

_REQUIRED_TOOLS: tuple[str, ...] = ("ffmpeg", "ffprobe")

#: Same filename every YouTube credential loader/writer uses (``_yt_creds.py``,
#: ``cli/handlers/youtube_auth.py``) -- not a secret, just the SOPS-encrypted
#: token's on-disk name.
_YOUTUBE_TOKEN_FILENAME = "youtube_upload_token.json.enc"  # noqa: S105


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    detail: str

    def line(self) -> str:
        return f"{self.status:<4} {self.name}: {self.detail}"


def check_python_extras(
    extras: Mapping[str, Sequence[str]] = EXTRAS_IMPORT_MAP,
    import_module: Callable[[str], object] = importlib.import_module,
) -> list[CheckResult]:
    """One result per optional extra: OK if every module it provides imports,
    WARN (never FAIL — these are opt-in features) listing what's missing."""
    results = []
    for extra, modules in extras.items():
        missing = []
        for module in modules:
            try:
                import_module(module)
            except ImportError:
                missing.append(module)
        if missing:
            results.append(
                CheckResult(
                    f"extra:{extra}",
                    WARN,
                    f"not installed (missing {', '.join(missing)}); pip install -e '.[{extra}]'",
                )
            )
        else:
            results.append(CheckResult(f"extra:{extra}", OK, "importable"))
    return results


def check_ffmpeg_tools(which: Callable[[str], str | None] = shutil.which) -> CheckResult:
    missing = [t for t in _REQUIRED_TOOLS if which(t) is None]
    if missing:
        return CheckResult("ffmpeg_tools", FAIL, f"not on PATH: {', '.join(missing)}")
    return CheckResult("ffmpeg_tools", OK, f"on PATH: {', '.join(_REQUIRED_TOOLS)}")


def check_ffmpeg_filters(
    which: Callable[[str], str | None] = shutil.which,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    required: Sequence[str] = _REQUIRED_FILTERS,
) -> CheckResult:
    """Parse ``ffmpeg -filters`` output and confirm each hand-listed filter is
    registered in this ffmpeg build. Skips (FAIL, not a crash) when ffmpeg
    itself is absent — ``check_ffmpeg_tools`` already reports that."""
    if which("ffmpeg") is None:
        return CheckResult("ffmpeg_filters", FAIL, "ffmpeg not on PATH; cannot list filters")
    try:
        result = run(["ffmpeg", "-filters"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CheckResult("ffmpeg_filters", FAIL, f"could not run `ffmpeg -filters`: {exc}")
    available = set()
    for raw_line in result.stdout.splitlines():
        parts = raw_line.split()
        # Filter list rows look like: " T.. scale   V->V  Scale the input video."
        # The filter name is the second whitespace-delimited token.
        if len(parts) >= 2:
            available.add(parts[1])
    missing = [f for f in required if f not in available]
    if missing:
        return CheckResult("ffmpeg_filters", FAIL, f"missing filters: {', '.join(missing)}")
    return CheckResult("ffmpeg_filters", OK, f"all {len(required)} required filters registered")


def check_gpu_nvenc(detect: Callable[[], bool] = None) -> CheckResult:
    """Reuses ``pipeline.color_filters.detect_nvenc`` — absence WARNs (a doctor run
    on a CPU-only dev box is normal), never FAILs."""
    if detect is None:
        from ...pipeline.color_filters import detect_nvenc

        detect = detect_nvenc
    try:
        detect()
    except Exception as exc:  # noqa: BLE001 - detect_nvenc's own contract raises broadly
        return CheckResult("gpu_nvenc", WARN, f"NVENC unavailable: {exc}")
    return CheckResult("gpu_nvenc", OK, "NVENC HEVC encode available")


def check_encoders(probe: Callable[[], Sequence[object]] = None) -> CheckResult:
    """Probe the encoder ladder (``pipeline.encoders``). OK with a hardware
    backend, WARN when only the CPU floor works, FAIL when nothing encodes."""
    if probe is None:
        from ...pipeline.encoders import probe_ladder

        probe = probe_ladder
    results = list(probe())
    working = [r for r in results if r.ok]
    if not working:
        reasons = "; ".join(f"{r.backend.name}: {r.reason}" for r in results)
        return CheckResult("encoders", FAIL, f"no HEVC encoder available: {reasons}")
    if not any(r.backend.hardware for r in working):
        hw = "; ".join(f"{r.backend.name}: {r.reason}" for r in results if r.backend.hardware)
        floor = ", ".join(r.backend.name for r in working)
        return CheckResult("encoders", WARN, f"only {floor} (CPU floor) available: {hw}")
    parts = [
        f"{r.backend.name} available" + ("" if r.backend.hardware else " (CPU floor)")
        for r in working
    ]
    return CheckResult("encoders", OK, "; ".join(parts))


def check_drive_credentials(check: Callable[[], None] = None) -> CheckResult:
    """Reuses ``pipeline.preflight.check_drive_credentials`` — never opens or
    prints the credential file contents, only checks their existence."""
    if check is None:
        from ...pipeline.preflight import check_drive_credentials as _check

        check = _check
    try:
        check()
    except RuntimeError as exc:
        return CheckResult("drive_credentials", FAIL, str(exc))
    return CheckResult("drive_credentials", OK, "credential path(s) present on disk")


def check_storage_writable(paths: Mapping[str, Path]) -> list[CheckResult]:
    """One result per named storage root: attempt to create it and write+remove
    a throwaway probe file. Never touches an existing file's contents."""
    results = []
    for name, path in paths.items():
        probe = Path(path) / ".gomoveshift-doctor-probe"
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            probe.write_text("", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            results.append(CheckResult(f"storage:{name}", FAIL, f"{path} not writable: {exc}"))
        else:
            results.append(CheckResult(f"storage:{name}", OK, f"{path} writable"))
    return results


#: Compose overlay variables and the in-container path each one mounts to. ``None``
#: means "the variable's own value" (the extra output root is mounted at the same
#: path inside the container).
_OVERLAY_MOUNTS: dict[str, str | None] = {
    "GMS_SITE_HOST_PATH": "/site",
    "GMS_NIGHT_HOST_ROOT": "/night-source",
    "GMS_VID_OUTPUT_DIR": None,
}


def check_overlay_mounts(
    environ: Mapping[str, str] | None = None,
    in_container: bool | None = None,
    is_mount: Callable[[str], bool] = os.path.ismount,
) -> list[CheckResult]:
    """Warn-only: inside a container, report each compose-overlay variable that is
    set while its target path is not a mount (the variable alone mounts nothing;
    the matching overlay file must be passed to ``docker compose``). Outside a
    container the variables name host paths and nothing is checked. The check
    only reports when the variables are passed into the container environment;
    variables used solely as compose bind-mount sources are invisible to it."""
    env = os.environ if environ is None else environ
    if in_container is None:
        in_container = Path("/.dockerenv").exists()
    if not in_container:
        return []
    results = []
    for name, target in _OVERLAY_MOUNTS.items():
        value = env.get(name, "")
        if not value:
            continue
        mount_point = target or value
        if not is_mount(mount_point):
            results.append(
                CheckResult(
                    f"overlay_mount:{name}",
                    WARN,
                    f"{name} is set but {mount_point} is not a mount in this container; "
                    "add the matching compose overlay file (docker/README.md)",
                )
            )
    return results


def check_youtube_identity(
    brands: Sequence[str] | None = None,
    credentials_dir: Path | None = None,
) -> list[CheckResult]:
    """One result per known brand: OK when that brand's token has a readable
    channel identity sidecar, WARN when the token exists but the sidecar
    doesn't (names the exact re-mint command that writes one), FAIL only
    when there is no token at all. Read-only, filesystem-only -- no sops
    call (``_yt_creds.read_channel_identity`` never decrypts the token) and
    no network call.

    ``brands`` defaults to ``console.settings.known_brands()`` (a brand with
    no ``brands/<brand>/creds`` dir of its own naturally resolves to the
    shared token via ``credential_paths.resolve_sops_file``, so the default
    "gomoveshift" brand checks the shared token without special-casing it
    here). ``credentials_dir`` defaults to ``_yt_creds.DEFAULT_CREDENTIALS_DIR``.
    """
    from ...console.settings import known_brands
    from ...pipeline._yt_creds import DEFAULT_CREDENTIALS_DIR, read_channel_identity
    from ...publishing.credential_paths import resolve_sops_file

    creds_dir = credentials_dir if credentials_dir is not None else DEFAULT_CREDENTIALS_DIR
    brand_list = list(brands) if brands is not None else known_brands()

    results = []
    for brand in brand_list:
        enc_path = resolve_sops_file(_YOUTUBE_TOKEN_FILENAME, brand, creds_dir)
        label = f"youtube_identity:{brand}"
        if not enc_path.exists():
            results.append(CheckResult(label, FAIL, f"no token at {enc_path}"))
            continue
        identity = read_channel_identity(enc_path)
        if identity is None:
            results.append(
                CheckResult(
                    label,
                    WARN,
                    f"token present but no channel identity sidecar; run "
                    f"`gomoveshift youtube-auth --brand {brand}` to write one",
                )
            )
            continue
        results.append(
            CheckResult(label, OK, f"{identity['channel_title']} ({identity['channel_id']})")
        )
    return results


def check_calibration(load: Callable[[], object] = load_calibration) -> CheckResult:
    """OK when a valid day-master calibration exists for this machine."""
    rec = load()
    if rec is None:
        return CheckResult(
            "calibration",
            WARN,
            "not measured on this machine -- run `gomoveshift doctor --calibrate <clip>`",
        )
    return CheckResult("calibration", OK, _calibration_summary(rec))


def _calibration_summary(rec) -> str:
    return (
        f"{rec.chain_fps:.1f} fps ({rec.encoder}, {rec.width}x{rec.height}, "
        f"blur {'on' if rec.blur else 'off'}, measured {rec.measured_at[:10]})"
    )


def check_tenant_stamp(storage_path: Path) -> CheckResult:
    """Read-only check of the ``.waypost-tenant`` stamp under ``storage_path``
    (``console/tenant.py``). Never writes — a missing stamp is normal on a
    fresh install (the first real run claims it), not a failure here."""
    from ...console.tenant import _stamp_path

    stamp_path = _stamp_path(storage_path)
    if not stamp_path.exists():
        return CheckResult("tenant_stamp", OK, f"{stamp_path} absent (unclaimed; normal pre-run)")
    try:
        content = stamp_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        return CheckResult("tenant_stamp", FAIL, f"{stamp_path} unreadable: {exc}")
    if not content:
        return CheckResult("tenant_stamp", WARN, f"{stamp_path} is empty (unknown owner)")
    return CheckResult("tenant_stamp", OK, f"{stamp_path} claimed by brand {content!r}")


def run_all_checks(config) -> list[CheckResult]:
    """Assemble every check into one ordered report."""
    results: list[CheckResult] = []
    results.extend(check_python_extras())
    results.append(check_ffmpeg_tools())
    results.append(check_ffmpeg_filters())
    results.append(check_gpu_nvenc())
    results.append(check_encoders())
    results.append(check_calibration())
    results.append(check_drive_credentials())
    results.extend(check_youtube_identity())

    intermediate_override = os.environ.get("GMS_INTERMEDIATE_DIR", "")
    storage_paths: dict[str, Path] = {}
    if intermediate_override:
        storage_paths["GMS_INTERMEDIATE_DIR"] = Path(intermediate_override)
    else:
        storage_paths["intermediate_cache (default)"] = gms_intermediate_base()
    results.extend(check_storage_writable(storage_paths))
    results.extend(check_overlay_mounts())

    seo_storage_path = getattr(getattr(config, "seo", None), "storage_path", None)
    if seo_storage_path:
        results.append(check_tenant_stamp(Path(seo_storage_path)))

    return results


def format_report(results: Sequence[CheckResult], as_json: bool = False) -> str:
    if as_json:
        return json.dumps(
            [{"name": r.name, "status": r.status, "detail": r.detail} for r in results],
            indent=2,
        )
    return "\n".join(r.line() for r in results)


def cmd_doctor(args, config) -> int:
    """CLI entry point for ``gomoveshift doctor``. Exit non-zero only on FAIL."""
    calibrate_clip = getattr(args, "calibrate", None)
    if calibrate_clip is not None:
        record = calibrate_master_chain(
            config,
            Path(calibrate_clip),
            blur=bool(config.drive.blur_enabled),
            workdir=gms_intermediate_base() / "_calibration",
        )
        save_calibration(record)
        print(f"calibration: {_calibration_summary(record)}")
    results = run_all_checks(config)
    print(format_report(results, as_json=getattr(args, "json", False)))
    return 1 if any(r.status == FAIL for r in results) else 0
