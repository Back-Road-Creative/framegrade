"""Tests for the WSL host-volume free-space preflight gate.

Failure mode: the guest ext4 staging filesystem is roomy while the Windows
drive that backs the WSL ext4 VHDX is nearly full. The pipeline passes the
guest-side staging gate, grows the VHDX, and fills the host volume.
``check_wsl_host_volume_space`` closes that gap by measuring the host volume directly.
"""

from __future__ import annotations

import types
import os
from pathlib import Path

import pytest

from gomoveshift.pipeline import preflight

_GB = 1_073_741_824


def _usage(free_gb: float):
    return types.SimpleNamespace(total=0, used=0, free=int(free_gb * _GB))


def test_noop_when_not_wsl(monkeypatch, tmp_path):
    """Off WSL the host-volume gate never fires, even with a full host."""
    monkeypatch.setattr(preflight, "_is_wsl", lambda: False)
    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda p: _usage(0.0))
    preflight.check_wsl_host_volume_space(tmp_path, needed_gb=500.0)


def test_raises_when_host_volume_below_need(monkeypatch, tmp_path):
    """WSL + host free below the requirement raises, naming the host volume."""
    host = tmp_path / "mnt_c"
    host.mkdir()
    monkeypatch.setattr(preflight, "_is_wsl", lambda: True)
    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda p: _usage(9.9))
    with pytest.raises(RuntimeError) as exc:
        preflight.check_wsl_host_volume_space(host, needed_gb=138.0, host_volume=host)
    msg = str(exc.value)
    assert str(host) in msg
    assert "host" in msg.lower()
    assert "vhdx" in msg.lower()


def test_passes_when_host_has_headroom(monkeypatch, tmp_path):
    host = tmp_path / "mnt_c"
    host.mkdir()
    monkeypatch.setattr(preflight, "_is_wsl", lambda: True)
    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda p: _usage(500.0))
    preflight.check_wsl_host_volume_space(host, needed_gb=138.0, host_volume=host)


def test_raises_when_host_mount_absent_on_wsl(monkeypatch, tmp_path):
    """On WSL, an unresolvable host mount HARD-FAILS, never warn+skips.

    A silent skip re-opens the disk-exhaustion crash class on a
    non-C: VHDX: the disk-space gate becomes a no-op exactly when it matters
    (the host drive can't be located to measure it). The check must raise so
    the operator points GMS_WSL_HOST_VOLUME at the backing drive.
    """
    monkeypatch.setattr(preflight, "_is_wsl", lambda: True)
    missing = tmp_path / "not_mounted"
    monkeypatch.setattr(
        preflight.shutil,
        "disk_usage",
        lambda p: (_ for _ in ()).throw(AssertionError("should not measure a missing mount")),
    )
    with pytest.raises(RuntimeError) as exc:
        preflight.check_wsl_host_volume_space(missing, needed_gb=500.0, host_volume=missing)
    msg = str(exc.value)
    assert str(missing) in msg
    assert "GMS_WSL_HOST_VOLUME" in msg


def test_noop_when_not_wsl_even_if_host_absent(monkeypatch, tmp_path):
    """Off WSL the gate stays an intentional no-op even with no host mount."""
    monkeypatch.setattr(preflight, "_is_wsl", lambda: False)
    missing = tmp_path / "not_mounted"
    monkeypatch.setattr(
        preflight.shutil,
        "disk_usage",
        lambda p: (_ for _ in ()).throw(AssertionError("should not be called off WSL")),
    )
    preflight.check_wsl_host_volume_space(missing, needed_gb=500.0, host_volume=missing)


def test_env_override_selects_host_volume(monkeypatch, tmp_path):
    """GMS_WSL_HOST_VOLUME redirects the check to a non-default drive."""
    host = tmp_path / "mnt_d"
    host.mkdir()
    monkeypatch.setattr(preflight, "_is_wsl", lambda: True)
    monkeypatch.setenv("GMS_WSL_HOST_VOLUME", str(host))
    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda p: _usage(5.0))
    with pytest.raises(RuntimeError) as exc:
        preflight.check_wsl_host_volume_space(tmp_path, needed_gb=100.0)
    assert str(host) in str(exc.value)


def test_run_preflight_catches_full_host(monkeypatch, tmp_path):
    """run_preflight fails loud when ext4 is roomy but the WSL host is full.

    Shape: staging passes but the host volume is nearly full.
    """
    source = tmp_path / "src"
    source.mkdir()
    (source / "clip.mp4").touch()
    os.truncate((source / "clip.mp4"), 200 * 1024 * 1024)  # sparse: 200 MB st_size, no disk
    output = tmp_path / "out"
    output.mkdir()
    staging = tmp_path / "stage"

    host = tmp_path / "mnt_c"
    host.mkdir()

    monkeypatch.setattr(preflight, "_is_wsl", lambda: True)

    def fake_usage(path):
        if "mnt_c" in str(path):
            return _usage(9.9)
        return _usage(5000.0)

    monkeypatch.setenv("GMS_WSL_HOST_VOLUME", str(host))
    monkeypatch.setattr(preflight.shutil, "disk_usage", fake_usage)
    monkeypatch.setattr(preflight, "check_ram", lambda **k: None)

    with pytest.raises(RuntimeError) as exc:
        preflight.run_preflight(
            source,
            output,
            staging_dir=staging,
            require_tools=None,
        )
    assert str(host) in str(exc.value)
