"""Build the Framegrade Windows bundle and installer.

Runs on any OS: uv installs the win_amd64 wheels for the target platform. The
layout follows the embeddable-Python pattern (no freeze step)::

    dist/stage/
        framegrade.cmd      launcher (packaging/windows/framegrade.cmd)
        python/             python.org embeddable 3.12 + Lib/site-packages
        app/                git-tracked source: gomoveshift/, config.yaml, ...

then ``ISCC`` compiles ``framegrade.iss`` over the stage into
``dist/framegrade-setup-<version>.exe``.

Privacy: only files git tracks are staged, and :func:`audit` refuses a stage
holding a data folder, a database, a credential or token file or an ``.enc``
file before anything is packaged. :func:`size_gate` refuses any output asset
over GitHub Releases' 2 GiB per-file limit before anything is uploaded.

    python packaging/windows/build_release.py [--skip-installer] [--zip]

Downloads come from python.org (the embeddable zip, sha256-checked) and PyPI
and the pinned git repositories in pyproject.toml (the wheels).
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[2]
WIN = Path(__file__).resolve().parent
DIST = ROOT / "dist"

PYTHON_VERSION = "3.12.10"
PYTHON_EMBED_URL = (
    f"https://www.python.org/ftp/python/{PYTHON_VERSION}/python-{PYTHON_VERSION}-embed-amd64.zip"
)
PYTHON_EMBED_SHA256 = "4acbed6dd1c744b0376e3b1cf57ce906f9dc9e95e68824584c8099a63025a3c3"  # pragma: allowlist secret (public python.org checksum, not a credential)

#: GitHub Releases' per-file upload limit.
MAX_ASSET_BYTES = 2 * 1024**3

#: What the app ships from git: the package, its config and the one data file
#: the wheel force-includes, plus the licence and readme.
SHIP = ("gomoveshift", "config.yaml", "metricool_reference_header.csv", "LICENSE", "README.md")

#: Search-path entries added to the embeddable's ``._pth``.
SITE_PACKAGES = "Lib\\site-packages"
APP_PATH = "..\\app"

_PRIVATE_DIRS = {"data", ".data", "out", "dist", ".git", ".runtime", ".state", ".aws", ".ssh"}
_CREDENTIAL_NAME = re.compile(
    r"(?:credentials?|secrets?|tokens?|client_secret[^/]*|service_account[^/]*"
    r"|[^/]*_token|[^/]*_credentials?)\.(?:json|ya?ml|toml|ini)",
    re.IGNORECASE,
)
_DATABASE = re.compile(r"\.(?:db|sqlite|sqlite3)(?:-(?:wal|shm|journal))?$", re.IGNORECASE)
_KEY_FILE = re.compile(r"(?:^\.env(?:\..*)?$|^id_(?:rsa|dsa|ecdsa|ed25519)$|\.(?:p12|pfx|jks)$)")


def project_version(root: Path) -> str:
    with (root / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)["project"]["version"]


def release_requirements(root: Path) -> tuple[list[str], list[str]]:
    """(requirements, constraints) for the bundle's site-packages.

    Requirements are pyproject's own dependency list, so the bundle can never
    drift from what ``pip install .`` asks for: the GPU onnxruntime with its
    ``cuda``/``cudnn`` extras and the git-pinned siblings. Constraints are
    requirements.lock's transitive pins, with editable lines dropped.
    """
    with (root / "pyproject.toml").open("rb") as fh:
        reqs = [d.strip() for d in tomllib.load(fh)["project"]["dependencies"]]
    constraints = [
        line.strip()
        for line in (root / "requirements.lock").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "-"))
    ]
    return reqs, constraints


def fetch_embed(cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / PYTHON_EMBED_URL.rsplit("/", 1)[1]
    if not target.exists():
        print(f"downloading {PYTHON_EMBED_URL}")
        urllib.request.urlretrieve(PYTHON_EMBED_URL, target)  # noqa: S310
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if digest != PYTHON_EMBED_SHA256:
        target.unlink()
        raise SystemExit(f"embeddable python sha256 mismatch: {digest}")
    return target


def enable_site_packages(pth: Path) -> None:
    """Edit the embeddable's ``._pth`` so it sees site-packages and ``app/``.

    The embeddable ignores PYTHONPATH and the registry; the ``._pth`` is its
    only search path, and ``import site`` ships commented out. ``..`` below
    ``python/`` is the install folder, so ``..\\app`` resolves wherever the
    user installs. Idempotent.
    """
    lines = [
        ln for ln in pth.read_text(encoding="utf-8").splitlines() if ln.strip() != "#import site"
    ]
    for extra in (SITE_PACKAGES, APP_PATH):
        if extra not in lines:
            lines.append(extra)
    if "import site" not in lines:
        lines.append("import site")
    pth.write_text("\n".join(lines) + "\n", encoding="utf-8")


def stage_python(embed_zip: Path, stage: Path) -> Path:
    py = stage / "python"
    with zipfile.ZipFile(embed_zip) as zf:
        zf.extractall(py)
    pths = sorted(py.glob("python*._pth"))
    if not pths:
        raise SystemExit("embeddable python has no python*._pth file")
    enable_site_packages(pths[0])
    return py


def tracked(root: Path, *paths: str) -> list[str]:
    """Files git tracks under ``paths``: nothing else is ever shipped."""
    git = shutil.which("git") or "git"
    out = subprocess.run(  # noqa: S603  (fixed argv, our own paths)
        [git, "ls-files", "-z", "--", *paths],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [p for p in out.split("\0") if p]


def stage_app(root: Path, stage: Path) -> None:
    for rel in tracked(root, *SHIP):
        dest = stage / "app" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / rel, dest)


def stage_launcher(stage: Path) -> None:
    text = (WIN / "framegrade.cmd").read_text(encoding="ascii")
    (stage / "framegrade.cmd").write_bytes(
        text.replace("\r\n", "\n").replace("\n", "\r\n").encode("ascii")
    )


def install_site_packages(root: Path, py: Path, work: Path) -> None:
    """Install the win_amd64 wheels for the bundle into ``python/Lib/site-packages``."""
    reqs, constraints = release_requirements(root)
    work.mkdir(parents=True, exist_ok=True)
    (work / "requirements.txt").write_text("\n".join(reqs) + "\n", encoding="utf-8")
    (work / "constraints.txt").write_text("\n".join(constraints) + "\n", encoding="utf-8")
    cmd = [
        "uv", "pip", "install", "--no-config",
        "--python-platform", "x86_64-pc-windows-msvc",
        "--python-version", "3.12",
        "--target", str(py / "Lib" / "site-packages"),
        "--constraint", str(work / "constraints.txt"),
        "-r", str(work / "requirements.txt"),
    ]  # fmt: skip
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=root)  # noqa: S603


def _problem(rel: str) -> str | None:
    """Why a staged path may not ship, or None. ``rel`` is POSIX, stage-relative."""
    parts = rel.split("/")
    name = parts[-1]
    third_party = parts[0] == "python"
    if name.lower().endswith(".enc"):
        return "encrypted private file"
    if third_party:
        # site-packages legitimately holds certificate bundles and sample
        # databases; only credential names and foreign binaries matter there.
        if name.endswith(".so"):
            return "a Linux binary (not a Windows build)"
        return "credential file" if _CREDENTIAL_NAME.fullmatch(name) else None
    if set(parts[:-1]) & _PRIVATE_DIRS:
        return "private data folder"
    if _CREDENTIAL_NAME.fullmatch(name) or _KEY_FILE.search(name) or name.endswith(".pem"):
        return "credential or key file"
    if _DATABASE.search(name):
        return "database"
    return None


def audit(stage: Path) -> None:
    """Refuse a stage holding private data, before anything is packaged."""
    for path in sorted(p for p in stage.rglob("*") if p.is_file()):
        rel = path.relative_to(stage).as_posix()
        why = _problem(rel)
        if why:
            raise SystemExit(f"release refused: {rel} is {why}")


def size_gate(paths: list[Path], limit: int = MAX_ASSET_BYTES) -> None:
    """Exit non-zero naming any asset over ``limit`` bytes, before any upload."""
    for path in paths:
        size = path.stat().st_size
        if size > limit:
            raise SystemExit(
                f"size gate: {path.name} is {size} bytes ({size / 1024**3:.2f} GiB), "
                f"over the {limit}-byte per-file limit"
            )


def zip_stage(stage: Path, out: Path) -> Path:
    out.unlink(missing_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(stage).as_posix())
    return out


def compile_installer(stage: Path, version: str) -> Path:
    iscc = shutil.which("ISCC") or shutil.which("ISCC.exe")
    if not iscc:
        raise SystemExit("Inno Setup (ISCC) not found; install it or pass --skip-installer")
    cmd = [iscc, f"/DAppVersion={version}", f"/DStageDir={stage}", str(WIN / "framegrade.iss")]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)  # noqa: S603
    return DIST / f"framegrade-setup-{version}.exe"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dist", type=Path, default=DIST, help="output folder (default: dist/)")
    ap.add_argument(
        "--skip-installer", action="store_true", help="stop before compiling with Inno Setup"
    )
    ap.add_argument("--zip", action="store_true", help="also write a portable zip of the stage")
    args = ap.parse_args(argv)

    dist = args.dist.resolve()
    stage = dist / "stage"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    version = project_version(ROOT)
    py = stage_python(fetch_embed(dist / "cache"), stage)
    install_site_packages(ROOT, py, dist / "work")
    stage_app(ROOT, stage)
    stage_launcher(stage)
    audit(stage)

    outputs: list[Path] = []
    if args.zip:
        outputs.append(zip_stage(stage, dist / f"framegrade-{version}-win64.zip"))
    if not args.skip_installer:
        outputs.append(compile_installer(stage, version))
    size_gate(outputs)
    total = sum(p.stat().st_size for p in stage.rglob("*") if p.is_file())
    print(f"staged {stage} ({total / 1024**3:.2f} GiB)")
    for out in outputs:
        print(f"built {out} ({out.stat().st_size / 1024**3:.2f} GiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
