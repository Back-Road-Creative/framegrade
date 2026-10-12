"""``gomoveshift doctor`` — one line per check, FAIL is the only thing
that exits non-zero; no real GPU/ffmpeg/network dependence in these tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from gomoveshift.cli.handlers.doctor import (
    FAIL,
    OK,
    WARN,
    CheckResult,
    check_drive_credentials,
    check_ffmpeg_filters,
    check_ffmpeg_tools,
    check_gpu_nvenc,
    check_python_extras,
    check_storage_writable,
    check_tenant_stamp,
    check_youtube_identity,
    cmd_doctor,
    format_report,
)


def _fake_config(storage_path: str) -> SimpleNamespace:
    return SimpleNamespace(seo=SimpleNamespace(storage_path=storage_path))


# ── check_python_extras ──────────────────────────────────────────────────


def test_extras_all_importable_is_ok():
    results = check_python_extras(extras={"fake": ("os", "sys")}, import_module=lambda name: None)
    assert results == [CheckResult("extra:fake", OK, "importable")]


def test_extras_missing_module_warns_not_fails():
    def fake_import(name):
        if name == "missing_pkg":
            raise ImportError(name)

    results = check_python_extras(extras={"fake": ("missing_pkg",)}, import_module=fake_import)
    assert len(results) == 1
    assert results[0].status == WARN
    assert "missing_pkg" in results[0].detail


# ── check_ffmpeg_tools ────────────────────────────────────────────────────


def test_ffmpeg_tools_present_is_ok():
    result = check_ffmpeg_tools(which=lambda tool: f"/usr/bin/{tool}")
    assert result.status == OK


def test_ffmpeg_tools_missing_fails():
    result = check_ffmpeg_tools(which=lambda tool: None)
    assert result.status == FAIL
    assert "ffmpeg" in result.detail


# ── check_ffmpeg_filters ──────────────────────────────────────────────────


def _filters_stdout(names):
    header = " T.. abuffer            A->A       Buffer audio frames, and make them accessible to the filterchain.\n"
    body = "".join(f" T.. {n:<18} V->V       fake filter.\n" for n in names)
    return header + body


class _FakeCompleted:
    def __init__(self, stdout):
        self.stdout = stdout


def test_ffmpeg_filters_all_present_is_ok():
    from gomoveshift.cli.handlers.doctor import _REQUIRED_FILTERS

    result = check_ffmpeg_filters(
        which=lambda tool: "/usr/bin/ffmpeg",
        run=lambda *a, **k: _FakeCompleted(_filters_stdout(_REQUIRED_FILTERS)),
    )
    assert result.status == OK


def test_ffmpeg_filters_missing_one_fails():
    from gomoveshift.cli.handlers.doctor import _REQUIRED_FILTERS

    present = [f for f in _REQUIRED_FILTERS if f != "drawtext"]
    result = check_ffmpeg_filters(
        which=lambda tool: "/usr/bin/ffmpeg",
        run=lambda *a, **k: _FakeCompleted(_filters_stdout(present)),
    )
    assert result.status == FAIL
    assert "drawtext" in result.detail


def test_ffmpeg_filters_no_ffmpeg_fails_without_crashing():
    result = check_ffmpeg_filters(which=lambda tool: None, run=lambda *a, **k: None)
    assert result.status == FAIL


# ── check_gpu_nvenc ───────────────────────────────────────────────────────


def test_gpu_nvenc_available_is_ok():
    result = check_gpu_nvenc(detect=lambda: True)
    assert result.status == OK


def test_gpu_nvenc_unavailable_warns_not_fails():
    def raising_detect():
        raise RuntimeError("NVENC probe failed")

    result = check_gpu_nvenc(detect=raising_detect)
    assert result.status == WARN
    assert "NVENC probe failed" in result.detail


# ── check_drive_credentials ───────────────────────────────────────────────


def test_drive_credentials_present_is_ok():
    result = check_drive_credentials(check=lambda: None)
    assert result.status == OK


def test_drive_credentials_missing_fails():
    def raising_check():
        raise RuntimeError("Drive credential path(s) missing on disk: X")

    result = check_drive_credentials(check=raising_check)
    assert result.status == FAIL
    assert "missing on disk" in result.detail


# ── check_storage_writable ────────────────────────────────────────────────


def test_storage_writable_dir_is_ok(tmp_path):
    target = tmp_path / "staging"
    results = check_storage_writable({"staging": target})
    assert results == [CheckResult("storage:staging", OK, f"{target} writable")]
    assert target.is_dir()


def test_storage_unwritable_parent_fails(tmp_path):
    # A file where a directory is expected: mkdir(parents=True) raises FileExistsError,
    # a subclass of OSError, without ever needing root/chmod tricks.
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    target = blocker / "staging"
    results = check_storage_writable({"staging": target})
    assert results[0].status == FAIL
    assert "not writable" in results[0].detail


# ── check_tenant_stamp ────────────────────────────────────────────────────


def test_tenant_stamp_absent_is_ok_not_a_failure(tmp_path):
    result = check_tenant_stamp(tmp_path / "seo-storage")
    assert result.status == OK
    assert "unclaimed" in result.detail


def test_tenant_stamp_present_is_ok_and_names_brand(tmp_path):
    storage = tmp_path / "seo-storage"
    storage.mkdir()
    (storage / ".waypost-tenant").write_text("gomoveshift", encoding="utf-8")
    result = check_tenant_stamp(storage)
    assert result.status == OK
    assert "gomoveshift" in result.detail


def test_tenant_stamp_empty_warns(tmp_path):
    storage = tmp_path / "seo-storage"
    storage.mkdir()
    (storage / ".waypost-tenant").write_text("   ", encoding="utf-8")
    result = check_tenant_stamp(storage)
    assert result.status == WARN


# ── check_youtube_identity ────────────────────────────────────────────────


def test_youtube_identity_ok_with_sidecar(tmp_path):
    creds_dir = tmp_path / "credentials"
    (tmp_path / "youtube_upload_token.json.enc").write_text("encrypted", encoding="utf-8")
    (tmp_path / "youtube_upload_token.identity.json").write_text(
        json.dumps({"channel_id": "UC123", "channel_title": "My Channel"}), encoding="utf-8"
    )
    results = check_youtube_identity(brands=["gomoveshift"], credentials_dir=creds_dir)
    assert len(results) == 1
    assert results[0].name == "youtube_identity:gomoveshift"
    assert results[0].status == OK
    assert "My Channel" in results[0].detail
    assert "UC123" in results[0].detail


def test_youtube_identity_warn_without_sidecar_names_the_fix(tmp_path):
    creds_dir = tmp_path / "credentials"
    (tmp_path / "youtube_upload_token.json.enc").write_text("encrypted", encoding="utf-8")
    results = check_youtube_identity(brands=["gomoveshift"], credentials_dir=creds_dir)
    assert results[0].status == WARN
    assert "youtube-auth" in results[0].detail
    assert "gomoveshift" in results[0].detail


def test_youtube_identity_fail_without_any_token(tmp_path):
    creds_dir = tmp_path / "credentials"
    results = check_youtube_identity(brands=["gomoveshift"], credentials_dir=creds_dir)
    assert results[0].status == FAIL


def test_youtube_identity_one_result_per_brand(tmp_path):
    creds_dir = tmp_path / "credentials"
    (tmp_path / "youtube_upload_token.json.enc").write_text("encrypted", encoding="utf-8")
    (tmp_path / "youtube_upload_token.identity.json").write_text(
        json.dumps({"channel_id": "UCshared", "channel_title": "Shared Channel"}),
        encoding="utf-8",
    )
    brand_dir = creds_dir / "brands" / "examplebrand" / "creds"
    brand_dir.mkdir(parents=True)
    (creds_dir / "brands" / "examplebrand" / "youtube_upload_token.json.enc").write_text(
        "encrypted", encoding="utf-8"
    )
    results = check_youtube_identity(
        brands=["gomoveshift", "examplebrand"], credentials_dir=creds_dir
    )
    by_name = {r.name: r for r in results}
    assert by_name["youtube_identity:gomoveshift"].status == OK
    assert by_name["youtube_identity:examplebrand"].status == WARN


# ── format_report ─────────────────────────────────────────────────────────


def test_format_report_plain_text_one_line_per_check():
    results = [CheckResult("a", OK, "fine"), CheckResult("b", FAIL, "broken")]
    text = format_report(results)
    lines = text.splitlines()
    assert len(lines) == 2
    assert "OK" in lines[0] and "a" in lines[0]
    assert "FAIL" in lines[1] and "broken" in lines[1]


def test_format_report_json_round_trips():
    results = [CheckResult("a", OK, "fine")]
    parsed = json.loads(format_report(results, as_json=True))
    assert parsed == [{"name": "a", "status": "OK", "detail": "fine"}]


# ── cmd_doctor exit codes ─────────────────────────────────────────────────


def test_cmd_doctor_exits_zero_when_no_failures(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("GMS_INTERMEDIATE_DIR", raising=False)
    monkeypatch.setattr(
        "gomoveshift.cli.handlers.doctor.run_all_checks",
        lambda config: [CheckResult("a", OK, "fine"), CheckResult("b", WARN, "meh")],
    )
    args = SimpleNamespace(json=False)
    exit_code = cmd_doctor(args, _fake_config(str(tmp_path)))
    assert exit_code == 0
    assert "OK" in capsys.readouterr().out


def test_cmd_doctor_exits_nonzero_when_any_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "gomoveshift.cli.handlers.doctor.run_all_checks",
        lambda config: [CheckResult("a", OK, "fine"), CheckResult("b", FAIL, "broken")],
    )
    args = SimpleNamespace(json=False)
    exit_code = cmd_doctor(args, _fake_config(str(tmp_path)))
    assert exit_code == 1


def test_cmd_doctor_json_flag(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(
        "gomoveshift.cli.handlers.doctor.run_all_checks",
        lambda config: [CheckResult("a", OK, "fine")],
    )
    args = SimpleNamespace(json=True)
    cmd_doctor(args, _fake_config(str(tmp_path)))
    out = capsys.readouterr().out
    assert json.loads(out) == [{"name": "a", "status": "OK", "detail": "fine"}]


@pytest.mark.integration
def test_run_all_checks_real_environment_does_not_crash(tmp_path):
    """Smoke test only — exercises the real defaults end to end. Marked
    integration since it may hit a real (absent) ffmpeg/GPU on the host;
    it must never raise, whatever the host's capabilities."""
    from gomoveshift.cli.handlers.doctor import run_all_checks

    results = run_all_checks(_fake_config(str(tmp_path)))
    assert len(results) > 0
    assert all(r.status in (OK, WARN, FAIL) for r in results)


def _stub_non_storage_checks(monkeypatch):
    """Replace every run_all_checks stage except the storage one with a stub."""
    import gomoveshift.cli.handlers.doctor as doctor_mod

    for name in (
        "check_ffmpeg_tools",
        "check_ffmpeg_filters",
        "check_gpu_nvenc",
        "check_encoders",
        "check_calibration",
        "check_drive_credentials",
    ):
        monkeypatch.setattr(doctor_mod, name, lambda: CheckResult("stub", OK, "stubbed"))
    monkeypatch.setattr(doctor_mod, "check_python_extras", lambda: [])
    monkeypatch.setattr(doctor_mod, "check_youtube_identity", lambda: [])


def test_run_all_checks_storage_defaults_to_the_cache_home(tmp_path, monkeypatch):
    """With GMS_INTERMEDIATE_DIR unset the storage check probes the cache home."""
    from gomoveshift.cli.handlers.doctor import run_all_checks

    _stub_non_storage_checks(monkeypatch)
    monkeypatch.delenv("GMS_INTERMEDIATE_DIR", raising=False)
    monkeypatch.setenv("GMS_CACHE_HOME", str(tmp_path))
    results = run_all_checks(SimpleNamespace())
    storage = [r for r in results if r.name.startswith("storage:")]
    assert [r.name for r in storage] == ["storage:intermediate_cache (default)"]
    assert storage[0].status == OK
    assert str(tmp_path) in storage[0].detail


def test_run_all_checks_storage_env_override_wins_over_cache_home(tmp_path, monkeypatch):
    """GMS_INTERMEDIATE_DIR still wins when the cache home is also set."""
    from gomoveshift.cli.handlers.doctor import run_all_checks

    _stub_non_storage_checks(monkeypatch)
    cache_home = tmp_path / "cache"
    override = tmp_path / "override"
    monkeypatch.setenv("GMS_CACHE_HOME", str(cache_home))
    monkeypatch.setenv("GMS_INTERMEDIATE_DIR", str(override))
    results = run_all_checks(SimpleNamespace())
    storage = [r for r in results if r.name.startswith("storage:")]
    assert [r.name for r in storage] == ["storage:GMS_INTERMEDIATE_DIR"]
    assert str(override) in storage[0].detail
    assert str(cache_home) not in storage[0].detail


# ── calibration ───────────────────────────────────────────


def _calib_record():
    from gomoveshift.pipeline.calibration import CalibrationRecord
    from gomoveshift.pipeline.day_master_encoder import MASTER_ENCODE_ALGO_VERSION

    return CalibrationRecord(
        measured_at="2026-10-07T12:00:00+00:00",
        chain_version=MASTER_ENCODE_ALGO_VERSION,
        encoder="hevc_nvenc",
        width=3840,
        height=2160,
        source_fps=30.0,
        slice_seconds=15.0,
        wall_seconds=22.5,
        blur=True,
        ffmpeg_version="ffmpeg version 6.1",
        cpu_count=16,
    )


def test_check_calibration_ok_reports_rate():
    from gomoveshift.cli.handlers.doctor import check_calibration

    res = check_calibration(load=_calib_record)
    assert res.status == OK
    assert res.name == "calibration"
    assert "20.0 fps" in res.detail and "hevc_nvenc" in res.detail and "2026-10-07" in res.detail


def test_check_calibration_not_measured_warns():
    from gomoveshift.cli.handlers.doctor import check_calibration

    res = check_calibration(load=lambda: None)
    assert res.status == WARN
    assert "not measured on this machine" in res.detail
    assert "gomoveshift doctor --calibrate <clip>" in res.detail


def test_cmd_doctor_calibrate_runs_and_saves(monkeypatch, capsys, tmp_path):
    seen = {}
    rec = _calib_record()

    def fake_calibrate(config, source, **kw):
        seen["calibrate"] = (source, kw["blur"])
        return rec

    monkeypatch.setattr("gomoveshift.cli.handlers.doctor.calibrate_master_chain", fake_calibrate)
    monkeypatch.setattr(
        "gomoveshift.cli.handlers.doctor.save_calibration", lambda r: seen.setdefault("saved", r)
    )
    monkeypatch.setattr(
        "gomoveshift.cli.handlers.doctor.run_all_checks",
        lambda config: [CheckResult("a", OK, "fine")],
    )
    clip = tmp_path / "clip.mp4"
    config = SimpleNamespace(
        seo=SimpleNamespace(storage_path=str(tmp_path)), drive=SimpleNamespace(blur_enabled=True)
    )
    code = cmd_doctor(SimpleNamespace(json=False, calibrate=clip), config)
    out = capsys.readouterr().out
    assert code == 0
    assert seen["calibrate"] == (clip, True)
    assert seen["saved"] is rec
    assert "20.0 fps" in out and "OK   a: fine" in out


# ── check_encoders ────────────────────────────────────────────────────────


def _probe_results(nv_ok, x_ok):
    from gomoveshift.pipeline.encoders import NVENC, X265, ProbeResult

    return [
        ProbeResult(NVENC, nv_ok, "" if nv_ok else "no device"),
        ProbeResult(X265, x_ok, "" if x_ok else "no libx265"),
    ]


def test_check_encoders_hardware_ok():
    from gomoveshift.cli.handlers.doctor import check_encoders

    res = check_encoders(probe=lambda: _probe_results(True, True))
    assert res.name == "encoders" and res.status == "OK"
    assert res.detail == "hevc_nvenc available; libx265 available (CPU floor)"


def test_check_encoders_cpu_only_warns():
    from gomoveshift.cli.handlers.doctor import check_encoders

    res = check_encoders(probe=lambda: _probe_results(False, True))
    assert res.status == "WARN"
    assert res.detail == "only libx265 (CPU floor) available: hevc_nvenc: no device"


def test_check_encoders_none_fails():
    from gomoveshift.cli.handlers.doctor import check_encoders

    res = check_encoders(probe=lambda: _probe_results(False, False))
    assert res.status == "FAIL"
    assert "no device" in res.detail and "no libx265" in res.detail


def test_check_encoders_lists_vendor_backend():
    from gomoveshift.cli.handlers.doctor import check_encoders
    from gomoveshift.pipeline.encoders import LADDER, ProbeResult

    ok = {"hevc_videotoolbox", "libx265"}
    results = [ProbeResult(b, b.name in ok, "" if b.name in ok else "no") for b in LADDER]
    res = check_encoders(probe=lambda: results)
    assert res.status == "OK"
    assert "hevc_videotoolbox available" in res.detail
