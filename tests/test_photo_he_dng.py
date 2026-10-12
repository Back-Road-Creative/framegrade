"""HE/HE* NEF -> DNG true-raw conversion hook: the converter subprocess mock
materializes the .dng as a side effect, and ``rawpy.imread`` routes the .nef
to an unpackable fake while the converted .dng decodes cleanly."""

import io
import logging
import os
import types
from pathlib import Path

import numpy as np
import pytest
import rawpy
from PIL import Image

import gomoveshift.config as config_module
import gomoveshift.pipeline.photos.develop as develop_module
from gomoveshift.config import PhotosConfig
from gomoveshift.pipeline.photos.develop import decode


def _jpeg_bytes(size=(8, 6)):
    buf = io.BytesIO()
    Image.new("RGB", size, (120, 80, 40)).save(buf, format="JPEG")
    return buf.getvalue()


class _FakeRawBase:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeDngRaw(_FakeRawBase):
    """A converted DNG that decodes cleanly through the normal rawpy branch."""

    last_kwargs: dict = {}
    raw_image = np.zeros((6, 8), dtype=np.uint16)

    def postprocess(self, **kwargs):
        type(self).last_kwargs = kwargs
        return np.full((6, 8, 3), 90, dtype=np.uint8)


class _FakeHERaw(_FakeRawBase):
    """HE NEF failure shape: unpack raises, the embedded preview still reads."""

    sizes = types.SimpleNamespace(width=8, height=6, flip=0)

    @property
    def raw_image(self):
        raise rawpy.LibRawDataError("data corrupted at 1541636")

    def extract_thumb(self):
        return types.SimpleNamespace(format=rawpy.ThumbFormat.JPEG, data=_jpeg_bytes())


@pytest.fixture()
def routed_rawpy(monkeypatch):
    monkeypatch.setattr(
        rawpy, "imread", lambda p: _FakeDngRaw() if str(p).endswith(".dng") else _FakeHERaw()
    )


@pytest.fixture()
def he_nef(tmp_path):
    path = tmp_path / "he.nef"
    path.write_bytes(b"placeholder")
    return path


def _set_photos_config(monkeypatch, **kwargs):
    photos = PhotosConfig(**kwargs)
    monkeypatch.setattr(
        config_module, "get_config", lambda *a, **k: types.SimpleNamespace(photos=photos)
    )


def _forbid_subprocess(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("subprocess.run must not be called")

    monkeypatch.setattr("subprocess.run", _boom)


class _ConverterMock:
    """Records argv/kwargs; optionally writes the .dng the way the real exe does."""

    def __init__(self, create_output=True):
        self.calls = []
        self.create_output = create_output

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), kwargs))
        if self.create_output:
            outdir = Path(argv[argv.index("-d") + 1])
            (outdir / (Path(argv[-1]).stem + ".dng")).write_bytes(b"dngdata")
        # The real converter reports errors silently: exit 0 either way.
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")


def test_disabled_by_default_uses_preview_fallback(routed_rawpy, he_nef, monkeypatch):
    assert PhotosConfig().he_dng_converter_exe == ""  # both keys default to disabled
    assert PhotosConfig().he_dng_cache_dir == ""
    _set_photos_config(monkeypatch)
    # Pin the probe away from the host filesystem so this asserts the
    # unconfigured, no-default-install behavior on any dev machine.
    monkeypatch.setattr(develop_module, "HE_DNG_CONVERTER_DEFAULT_PATHS", ())
    _forbid_subprocess(monkeypatch)

    out = decode(he_nef)

    assert out.decoder == "rawpy-preview"
    assert out.is_raw is False


def test_probe_uses_default_install_when_config_empty(routed_rawpy, he_nef, monkeypatch, tmp_path):
    exe = tmp_path / "Adobe DNG Converter.exe"
    exe.write_bytes(b"exe")
    _set_photos_config(monkeypatch)  # he_dng_converter_exe unset -> probe runs
    monkeypatch.setattr(develop_module, "HE_DNG_CONVERTER_DEFAULT_PATHS", (str(exe),))
    run_mock = _ConverterMock()
    monkeypatch.setattr("subprocess.run", run_mock)

    out = decode(he_nef)

    assert out.decoder == "rawpy-dng"
    assert out.is_raw is True
    assert len(run_mock.calls) == 1
    assert run_mock.calls[0][0][0] == str(exe)  # the probed exe is what gets invoked


def test_probe_miss_falls_back_and_names_the_remedy(
    routed_rawpy, he_nef, monkeypatch, tmp_path, caplog
):
    _set_photos_config(monkeypatch)
    monkeypatch.setattr(
        develop_module,
        "HE_DNG_CONVERTER_DEFAULT_PATHS",
        (str(tmp_path / "missing" / "Adobe DNG Converter.exe"),),
    )
    _forbid_subprocess(monkeypatch)

    with caplog.at_level(logging.WARNING):
        out = decode(he_nef)

    assert out.decoder == "rawpy-preview"
    # The one preview-fallback warning names the remedy: install the converter
    # at its default path, or point the config key at it.
    assert "Adobe DNG Converter" in caplog.text
    assert "photos.he_dng_converter_exe" in caplog.text


def test_explicit_config_wins_over_existing_probe_path(routed_rawpy, he_nef, monkeypatch, tmp_path):
    probe_exe = tmp_path / "Adobe DNG Converter.exe"
    probe_exe.write_bytes(b"exe")
    _set_photos_config(monkeypatch, he_dng_converter_exe="/opt/dngc")
    monkeypatch.setattr(develop_module, "HE_DNG_CONVERTER_DEFAULT_PATHS", (str(probe_exe),))
    run_mock = _ConverterMock()
    monkeypatch.setattr("subprocess.run", run_mock)

    out = decode(he_nef)

    assert out.decoder == "rawpy-dng"
    assert len(run_mock.calls) == 1
    assert run_mock.calls[0][0][0] == "/opt/dngc"  # config always beats the probe


def test_configured_converter_yields_true_raw_dng_decode(routed_rawpy, he_nef, monkeypatch):
    exe = "/usr/local/bin/dngconv"  # a native Linux converter: argv keeps WSL paths
    _set_photos_config(monkeypatch, he_dng_converter_exe=exe)
    run_mock = _ConverterMock()
    monkeypatch.setattr("subprocess.run", run_mock)

    out = decode(he_nef)

    assert out.decoder == "rawpy-dng"
    assert out.is_raw is True
    assert out.bit_depth == 16  # fake exposes no white_level: container-depth fallback
    # The DNG branch develops with the same 16-bit / no-auto-bright / sRGB
    # postprocess policy as the direct NEF branch.
    assert _FakeDngRaw.last_kwargs["output_bps"] == 16
    assert _FakeDngRaw.last_kwargs["no_auto_bright"] is True
    assert _FakeDngRaw.last_kwargs["gamma"] == (2.4, 12.92)
    assert len(run_mock.calls) == 1  # one invocation per file, no wildcards
    argv, kwargs = run_mock.calls[0]
    assert argv == [exe, "-c", "-p1", "-cr11.2", "-d", str(he_nef.parent), str(he_nef)]
    assert kwargs.get("timeout")  # bounded, never a hang


def test_silent_converter_failure_warns_and_falls_back(routed_rawpy, he_nef, monkeypatch, caplog):
    _set_photos_config(monkeypatch, he_dng_converter_exe="/opt/dngc")
    run_mock = _ConverterMock(create_output=False)
    monkeypatch.setattr("subprocess.run", run_mock)

    with caplog.at_level(logging.WARNING):
        out = decode(he_nef)

    assert out.decoder == "rawpy-preview"
    assert len(run_mock.calls) == 1
    assert "DNG converter" in caplog.text


def test_unlaunchable_converter_warns_and_falls_back(routed_rawpy, he_nef, monkeypatch, caplog):
    _set_photos_config(monkeypatch, he_dng_converter_exe="/nope/dngc")
    monkeypatch.setattr(
        "subprocess.run", lambda *a, **k: (_ for _ in ()).throw(OSError("no such executable"))
    )

    with caplog.at_level(logging.WARNING):
        out = decode(he_nef)

    assert out.decoder == "rawpy-preview"
    assert "DNG converter" in caplog.text


def test_fresh_cached_dng_reused_without_reconverting(routed_rawpy, he_nef, monkeypatch):
    _set_photos_config(monkeypatch, he_dng_converter_exe="/opt/dngc")
    (he_nef.parent / "he.dng").write_bytes(b"dngdata")  # written after the NEF: mtime >= source
    _forbid_subprocess(monkeypatch)

    out = decode(he_nef)

    assert out.decoder == "rawpy-dng"
    assert out.is_raw is True


def test_stale_cached_dng_is_reconverted(routed_rawpy, he_nef, monkeypatch):
    _set_photos_config(monkeypatch, he_dng_converter_exe="/opt/dngc")
    stale = he_nef.parent / "he.dng"
    stale.write_bytes(b"stale")
    os.utime(stale, (1, 1))  # far older than the source NEF
    run_mock = _ConverterMock()
    monkeypatch.setattr("subprocess.run", run_mock)

    assert decode(he_nef).decoder == "rawpy-dng"
    assert len(run_mock.calls) == 1


def test_cache_dir_config_routes_the_converted_dng(routed_rawpy, he_nef, monkeypatch, tmp_path):
    cache = tmp_path / "dng_cache"
    _set_photos_config(monkeypatch, he_dng_converter_exe="/opt/dngc", he_dng_cache_dir=str(cache))
    run_mock = _ConverterMock()
    monkeypatch.setattr("subprocess.run", run_mock)

    out = decode(he_nef)

    assert out.decoder == "rawpy-dng"
    argv = run_mock.calls[0][0]
    assert argv[argv.index("-d") + 1] == str(cache)
    assert (cache / "he.dng").is_file()


# ── _convert_he_to_dng: the conversion half only, no postprocess ─────
# The astro stacking path needs the DNG without develop's display-referred
# sRGB postprocess, so this half must be reachable — and testable — with no
# rawpy involvement at all. None of these tests use ``routed_rawpy``.


def test_convert_non_nef_returns_none(tmp_path):
    path = tmp_path / "photo.jpg"
    path.write_bytes(b"not a nef")
    assert develop_module._convert_he_to_dng(path) is None


def test_convert_no_converter_configured_or_discoverable_returns_none(he_nef, monkeypatch):
    _set_photos_config(monkeypatch)
    monkeypatch.setattr(develop_module, "HE_DNG_CONVERTER_DEFAULT_PATHS", ())
    _forbid_subprocess(monkeypatch)

    assert develop_module._convert_he_to_dng(he_nef) is None


def test_convert_fresh_cached_dng_reused_without_reconverting(he_nef, monkeypatch):
    _set_photos_config(monkeypatch, he_dng_converter_exe="/opt/dngc")
    dng = he_nef.parent / "he.dng"
    dng.write_bytes(b"dngdata")  # written after the NEF: mtime >= source
    _forbid_subprocess(monkeypatch)

    assert develop_module._convert_he_to_dng(he_nef) == dng


def test_convert_silent_converter_failure_returns_none(he_nef, monkeypatch):
    """The converter exits 0 but produces no file — the silent-failure case."""
    _set_photos_config(monkeypatch, he_dng_converter_exe="/opt/dngc")
    run_mock = _ConverterMock(create_output=False)
    monkeypatch.setattr("subprocess.run", run_mock)

    assert develop_module._convert_he_to_dng(he_nef) is None
    assert len(run_mock.calls) == 1


def test_convert_success_returns_ready_to_read_dng_path(he_nef, monkeypatch):
    _set_photos_config(monkeypatch, he_dng_converter_exe="/opt/dngc")
    run_mock = _ConverterMock()
    monkeypatch.setattr("subprocess.run", run_mock)

    out = develop_module._convert_he_to_dng(he_nef)

    assert out == he_nef.parent / "he.dng"
    assert out.is_file() and out.stat().st_size > 0


# ── one shared converter-resolution rule ─────────────────────────────
# The `photos astro` preflight gate and _convert_he_to_dng must resolve the converter
# by the identical rule: a gate that disagrees with the decode path is worse than none.


def test_resolve_converter_config_wins_then_probe_then_empty(monkeypatch, tmp_path):
    exe = tmp_path / "Adobe DNG Converter.exe"
    exe.write_bytes(b"exe")
    monkeypatch.setattr(develop_module, "HE_DNG_CONVERTER_DEFAULT_PATHS", (str(exe),))

    _set_photos_config(monkeypatch, he_dng_converter_exe="/opt/dngc")
    assert develop_module.resolve_he_dng_converter() == "/opt/dngc"  # config beats the probe

    _set_photos_config(monkeypatch)  # unset -> probe the default install paths
    assert develop_module.resolve_he_dng_converter() == str(exe)

    monkeypatch.setattr(develop_module, "HE_DNG_CONVERTER_DEFAULT_PATHS", ())
    assert develop_module.resolve_he_dng_converter() == ""  # nothing resolves


def test_convert_goes_through_the_shared_resolver(he_nef, monkeypatch):
    _set_photos_config(monkeypatch)
    monkeypatch.setattr(develop_module, "resolve_he_dng_converter", lambda: "")
    _forbid_subprocess(monkeypatch)
    assert develop_module._convert_he_to_dng(he_nef) is None

    monkeypatch.setattr(develop_module, "resolve_he_dng_converter", lambda: "/opt/shared")
    run_mock = _ConverterMock()
    monkeypatch.setattr("subprocess.run", run_mock)
    assert develop_module._convert_he_to_dng(he_nef) == he_nef.parent / "he.dng"
    assert run_mock.calls[0][0][0] == "/opt/shared"


def test_inputs_needing_conversion_skips_non_nefs_and_fresh_dngs(tmp_path, monkeypatch):
    """The preflight's cheap question: which inputs would the converter be asked about?
    Suffix + one stat each, never a decode; a missing source is a candidate, not a crash."""
    _set_photos_config(monkeypatch)
    jpg, cached, stale, plain = (
        tmp_path / n for n in ("photo.jpg", "cached.nef", "stale.NEF", "plain.NEF")
    )
    for p in (jpg, cached, stale, plain):
        p.write_bytes(b"x")
    (tmp_path / "cached.dng").write_bytes(b"dngdata")  # written after: mtime >= source
    stale_dng = tmp_path / "stale.dng"
    stale_dng.write_bytes(b"old")
    os.utime(stale_dng, (1, 1))  # far older than its source
    gone = tmp_path / "gone.NEF"
    (tmp_path / "gone.dng").write_bytes(b"dngdata")  # orphan sibling, source never written

    inputs = [jpg, cached, stale, plain, gone]
    assert develop_module.he_dng_inputs_needing_conversion(inputs) == (stale, plain, gone)

    # The cache dir moves where a reusable .dng counts from, so the preflight must
    # read the same config key the conversion writes to.
    cache = tmp_path / "dng_cache"
    cache.mkdir()
    (cache / "plain.dng").write_bytes(b"dngdata")
    _set_photos_config(monkeypatch, he_dng_cache_dir=str(cache))
    assert develop_module.he_dng_inputs_needing_conversion([plain]) == ()


def test_windows_path_translates_wsl_mounts():
    assert (
        develop_module._windows_path(Path("/mnt/d/photos/DSC_0001.NEF"))
        == "D:\\photos\\DSC_0001.NEF"
    )
    assert develop_module._windows_path("/mnt/c/Program Files/x") == "C:\\Program Files\\x"
    assert develop_module._windows_path(Path("/mnt/d/photos")) == "D:\\photos"  # trailing dir
    assert develop_module._windows_path(Path("/srv/data/x")) == "/srv/data/x"  # not a drive mount


def test_windows_exe_gets_windows_form_argv(monkeypatch, caplog):
    """A /mnt-mounted Windows exe is handed Windows-form argv paths, while the
    success check still reads the WSL-side .dng path."""
    exe = "/mnt/c/Program Files/Adobe/Adobe DNG Converter/Adobe DNG Converter.exe"
    _set_photos_config(monkeypatch, he_dng_converter_exe=exe)
    # Purely fictional /mnt paths — neutralize the fs writes so nothing under a
    # real Windows mount is ever created or removed by this test.
    monkeypatch.setattr(Path, "mkdir", lambda self, *a, **k: None)
    monkeypatch.setattr(Path, "unlink", lambda self, *a, **k: None)
    run_mock = _ConverterMock(create_output=False)
    monkeypatch.setattr("subprocess.run", run_mock)
    src = Path("/mnt/z/gms_dng_argv_test/DSC_0001.NEF")

    with caplog.at_level(logging.WARNING):
        out = develop_module._decode_he_via_dng(src, RuntimeError("unpack failed"))

    assert out is None  # no .dng appeared at the WSL-side path -> preview fallback
    assert len(run_mock.calls) == 1
    argv = run_mock.calls[0][0]
    assert argv == [
        exe,
        "-c",
        "-p1",
        "-cr11.2",
        "-d",
        "Z:\\gms_dng_argv_test",
        "Z:\\gms_dng_argv_test\\DSC_0001.NEF",
    ]
    # Success is judged at the WSL-side path, never the Windows form.
    assert "/mnt/z/gms_dng_argv_test/DSC_0001.dng" in caplog.text


def test_windows_exe_with_ext4_source_fails_closed_to_preview(
    routed_rawpy, he_nef, monkeypatch, caplog
):
    """A Windows exe cannot see a Linux-filesystem path: warn naming the path
    and fall back to the preview instead of invoking a conversion that must fail."""
    exe = "/mnt/c/Program Files/Adobe/Adobe DNG Converter/Adobe DNG Converter.exe"
    _set_photos_config(monkeypatch, he_dng_converter_exe=exe)
    _forbid_subprocess(monkeypatch)

    with caplog.at_level(logging.WARNING):
        out = decode(he_nef)

    assert out.decoder == "rawpy-preview"
    assert str(he_nef) in caplog.text  # the warning names the invisible path
