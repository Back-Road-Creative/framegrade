"""Structural gate + unit tests for gomoveshift.pipeline.fs_safety.

WSL's drvfs mounts (/mnt/<letter>) routinely refuse the copystat metadata
update that ``shutil.copy2`` performs after copying bytes, raising
PermissionError/OSError even though the file content copied fine. This has
been independently re-patched three times (tracking/store.py, redrive_ost.py,
drive_phases/phase_d.py). ``fs_safety.safe_copy`` is the one shared helper;
this test bans the raw stdlib call from reappearing at a new call site.
"""

from pathlib import Path

import pytest

PIPELINE_ROOT = Path(__file__).resolve().parent.parent / "gomoveshift" / "pipeline"
TRACKING_ROOT = Path(__file__).resolve().parent.parent / "gomoveshift" / "tracking"

ALLOWED_FILES = {"fs_safety.py"}


def _python_files(root: Path):
    if not root.is_dir():
        return
    yield from root.rglob("*.py")


def test_no_raw_shutil_copy2_outside_fs_safety():
    """No call site under pipeline/ or tracking/ may call shutil.copy2 directly.

    Every stat-preserving copy must go through fs_safety.safe_copy, which
    tolerates a copystat failure on a drvfs mount instead of propagating it
    as a stage failure.
    """
    offenders = []
    for root in (PIPELINE_ROOT, TRACKING_ROOT):
        for path in _python_files(root):
            if path.name in ALLOWED_FILES:
                continue
            text = path.read_text()
            for lineno, line in enumerate(text.splitlines(), start=1):
                if "shutil.copy2(" in line:
                    offenders.append(f"{path.relative_to(root.parent.parent)}:{lineno}")

    assert not offenders, (
        "Raw shutil.copy2 found outside fs_safety.py -- use fs_safety.safe_copy "
        f"instead (drvfs copystat safety): {offenders}"
    )


def test_safe_copy_survives_copystat_permission_error(tmp_path, monkeypatch):
    """Bytes must land even when copystat raises PermissionError (drvfs)."""
    import shutil

    from gomoveshift.pipeline import fs_safety

    src = tmp_path / "src.bin"
    dst = tmp_path / "dst.bin"
    src.write_bytes(b"payload-bytes")

    def _boom(_src, _dst, *, follow_symlinks=True):
        raise PermissionError("simulated drvfs copystat denial")

    monkeypatch.setattr(shutil, "copystat", _boom)

    fs_safety.safe_copy(src, dst)

    assert dst.read_bytes() == b"payload-bytes"


def test_safe_copy_survives_copystat_oserror(tmp_path, monkeypatch):
    """Same guarantee for a plain OSError (9p mounts raise this too)."""
    import shutil

    from gomoveshift.pipeline import fs_safety

    src = tmp_path / "src.bin"
    dst = tmp_path / "dst.bin"
    src.write_bytes(b"more-bytes")

    def _boom(_src, _dst, *, follow_symlinks=True):
        raise OSError("simulated 9p copystat denial")

    monkeypatch.setattr(shutil, "copystat", _boom)

    fs_safety.safe_copy(src, dst)

    assert dst.read_bytes() == b"more-bytes"


def test_safe_copy_still_raises_on_real_copy_failure(tmp_path):
    """safe_copy must not swallow a genuine copy failure (missing source)."""
    from gomoveshift.pipeline import fs_safety

    src = tmp_path / "does-not-exist.bin"
    dst = tmp_path / "dst.bin"

    with pytest.raises(OSError):
        fs_safety.safe_copy(src, dst)
