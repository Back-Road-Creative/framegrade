"""EasyOCR runs on the GPU when torch can see one.

The sign pass reads one native-resolution frame every few seconds of a GPS-less
drive, so the reader's device decides whether the pass takes minutes or hours.
CPU is always correct, only slower, so any failure to probe falls back to it.
"""

import sys
from types import SimpleNamespace

import pytest

from gomoveshift.analysis import ocr as ocr_mod
from gomoveshift.config import Config


class _FakeReader:
    kwargs: dict = {}

    def __init__(self, languages, **kwargs):
        type(self).kwargs = {"languages": languages, **kwargs}


@pytest.fixture
def fake_easyocr(monkeypatch):
    monkeypatch.setitem(sys.modules, "easyocr", SimpleNamespace(Reader=_FakeReader))
    _FakeReader.kwargs = {}
    return _FakeReader


@pytest.mark.parametrize("cuda", [True, False])
def test_reader_device_follows_the_cuda_probe(monkeypatch, fake_easyocr, cuda):
    monkeypatch.setattr(ocr_mod, "_cuda_available", lambda: cuda)
    _ = ocr_mod.OCREngine(Config()).easyocr_reader
    assert fake_easyocr.kwargs["gpu"] is cuda
    assert fake_easyocr.kwargs["languages"] == ["en"]


def test_cuda_probe_is_false_when_torch_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # ``import torch`` raises ImportError
    assert ocr_mod._cuda_available() is False


def test_cuda_probe_is_false_when_torch_raises(monkeypatch):
    def _boom():
        raise RuntimeError("driver mismatch")

    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=_boom))
    )
    assert ocr_mod._cuda_available() is False
