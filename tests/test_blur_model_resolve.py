"""``config.drive`` → redactcam ``ModelSpec`` model resolution — no real network."""

import hashlib
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from gomoveshift import analysis

PAYLOAD = b"fake-onnx-bytes" * 100
SHA = hashlib.sha256(PAYLOAD).hexdigest()


def _drive(**kw):
    base = {
        "plate_model_path": "",
        "plate_model_url": "",
        "plate_model_sha256": "",
        "face_model_path": "",
        "face_model_url": "",
        "face_model_sha256": "",
    }
    base.update(kw)
    return SimpleNamespace(**base)


def _fake_urlopen(payload):
    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.close()

    def _open(req, timeout=None):
        return _Resp(payload)

    return _open


@pytest.fixture
def _cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "MODEL_CACHE_DIR", tmp_path / "models")
    return tmp_path / "models"


pytestmark = pytest.mark.usefixtures("_cache_dir")


class TestOverride:
    def test_existing_override_wins(self, tmp_path):
        m = tmp_path / "my.onnx"
        m.write_bytes(b"x")
        assert analysis.resolve_plate_model(_drive(plate_model_path=str(m))) == m

    def test_missing_override_returns_none(self, tmp_path, caplog):
        d = _drive(plate_model_path=str(tmp_path / "nope.onnx"))
        with caplog.at_level("WARNING"):
            assert analysis.resolve_plate_model(d) is None
        assert "faces only" in caplog.text


class TestDownloadConfig:
    def test_no_url_or_sha_returns_none(self):
        assert analysis.resolve_plate_model(_drive()) is None

    def test_non_https_refused(self, caplog):
        d = _drive(plate_model_url="http://x/m.onnx", plate_model_sha256="ab")
        with caplog.at_level("WARNING"):
            assert analysis.resolve_plate_model(d) is None
        assert "not https" in caplog.text


class TestDownload:
    def test_downloads_verifies_caches(self, monkeypatch):
        monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen(PAYLOAD))
        d = _drive(plate_model_url="https://h/m.onnx", plate_model_sha256=SHA)
        path = analysis.resolve_plate_model(d)
        assert path is not None and path.exists()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == SHA

    def test_cache_hit_skips_download(self, monkeypatch):
        d = _drive(plate_model_url="https://h/m.onnx", plate_model_sha256=SHA)
        monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen(PAYLOAD))
        first = analysis.resolve_plate_model(d)

        def _boom(*a, **k):
            raise AssertionError("cache hit must not re-download")

        monkeypatch.setattr("urllib.request.urlopen", _boom)
        assert analysis.resolve_plate_model(d) == first

    def test_checksum_mismatch_returns_none(self, monkeypatch, caplog):
        monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen(PAYLOAD))
        d = _drive(plate_model_url="https://h/m.onnx", plate_model_sha256="00" * 32)
        with caplog.at_level("WARNING"):
            assert analysis.resolve_plate_model(d) is None
        assert "checksum mismatch" in caplog.text

    def test_fetch_error_degrades_to_faces(self, monkeypatch, caplog):
        def _boom(*a, **k):
            raise OSError("network down")

        monkeypatch.setattr("urllib.request.urlopen", _boom)
        d = _drive(plate_model_url="https://h/m.onnx", plate_model_sha256=SHA)
        with caplog.at_level("WARNING"):
            assert analysis.resolve_plate_model(d) is None
        assert "fetch failed" in caplog.text


class TestFaceModel:
    """resolve_face_model shares the resolver but reads face_model_* and degrades
    to the CenterFace fallback (not faces-only) — and caches under a distinct name."""

    def test_override_wins(self, tmp_path):
        m = tmp_path / "face.onnx"
        m.write_bytes(b"x")
        assert analysis.resolve_face_model(_drive(face_model_path=str(m))) == m

    def test_downloads_to_distinct_cache_name(self, monkeypatch):
        monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen(PAYLOAD))
        d = _drive(face_model_url="https://h/f.onnx", face_model_sha256=SHA)
        path = analysis.resolve_face_model(d)
        assert path is not None and path.exists() and path.name == f"face_{SHA[:12]}.onnx"

    def test_fetch_error_degrades_to_centerface(self, monkeypatch, caplog):
        def _boom(*a, **k):
            raise OSError("network down")

        monkeypatch.setattr("urllib.request.urlopen", _boom)
        d = _drive(face_model_url="https://h/f.onnx", face_model_sha256=SHA)
        with caplog.at_level("WARNING"):
            assert analysis.resolve_face_model(d) is None
        assert "CenterFace fallback" in caplog.text


class TestCacheLocationPinned:
    """The service pins redactcam's ``cache_dir`` to ``<cache home>/models``.
    Taking the package default (``~/.cache/redactcam/models``) would silently
    re-fetch ~40 MB per host and orphan the warm cache -- so the pin is
    asserted, not assumed."""

    def test_default_is_models_under_the_cache_home(self, tmp_path, monkeypatch):
        monkeypatch.setattr(analysis, "MODEL_CACHE_DIR", None)
        monkeypatch.delenv("GMS_CACHE_HOME", raising=False)
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        assert analysis._model_cache_dir() == tmp_path / ".cache" / "gomoveshift" / "models"

    @pytest.mark.parametrize("variable", ["GMS_CACHE_HOME", "XDG_CACHE_HOME"])
    def test_default_follows_the_cache_home_override(self, tmp_path, monkeypatch, variable):
        monkeypatch.setattr(analysis, "MODEL_CACHE_DIR", None)
        monkeypatch.delenv("GMS_CACHE_HOME", raising=False)
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.setenv(variable, str(tmp_path / "moved"))
        expected = (
            tmp_path / "moved" / "models"
            if variable == "GMS_CACHE_HOME"
            else tmp_path / "moved" / "gomoveshift" / "models"
        )
        assert analysis._model_cache_dir() == expected

    def test_resolved_model_lands_under_the_pinned_dir(self, _cache_dir, monkeypatch):
        monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen(PAYLOAD))
        d = _drive(plate_model_url="https://h/m.onnx", plate_model_sha256=SHA)
        resolved = analysis.resolve_plate_model(d)
        assert resolved is not None and Path(resolved).parent == _cache_dir
