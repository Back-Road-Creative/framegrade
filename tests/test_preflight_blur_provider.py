"""Preflight gate: the blur detectors must run on CUDA when the GPU onnxruntime
build is what this venv installed — and say so at minute 0, not silently at ~19×
slower for a day.

Failure mode: a later install can put the CPU ``onnxruntime`` wheel over the
GPU one (both wheels own ``onnxruntime/``, last install wins).
``ort.get_available_providers()`` then lists only CPU, redactcam picks CPU
without complaint, and the blur pass runs on CPU. Nothing in preflight or the
run log names the cause.

This gate makes that state a preflight failure with the exact repair. It is
deliberately self-contained (it does not call into redactcam), so it holds
whichever redactcam commit the venv carries and cannot itself be broken by the
same dependency drift it guards against.
"""

from __future__ import annotations

import importlib.metadata
import inspect
from pathlib import Path

import pytest

from gomoveshift.pipeline import preflight
from gomoveshift.pipeline.preflight import (
    ALLOW_CPU_BLUR_ENV,
    check_blur_inference_provider,
    run_preflight,
)

# Bound at import, before conftest's autouse pin replaces the module attribute.
_real_probe = preflight._probe_session_providers

CUDA = "CUDAExecutionProvider"
CPU = "CPUExecutionProvider"


@pytest.fixture(autouse=True)
def _no_operator_override(monkeypatch):
    monkeypatch.delenv(ALLOW_CPU_BLUR_ENV, raising=False)
    # This file tests the probe itself: undo conftest's pin.
    monkeypatch.setattr(preflight, "_probe_session_providers", _real_probe)


class TestBlurInferenceProviderGate:
    def test_cpu_only_host_without_the_gpu_build_passes(self):
        """A host that never installed onnxruntime-gpu asked for CPU; honour it."""
        check_blur_inference_provider(gpu_expected=False, available=[CPU])

    def test_gpu_build_with_cuda_passes(self):
        check_blur_inference_provider(
            gpu_expected=True, available=[CUDA, CPU], session_providers=[CUDA, CPU]
        )

    def test_gpu_build_without_cuda_fails_with_the_repair(self):
        with pytest.raises(RuntimeError) as exc:
            check_blur_inference_provider(
                gpu_expected=True, available=["AzureExecutionProvider", CPU]
            )
        msg = str(exc.value)
        assert "onnxruntime-gpu" in msg
        assert CUDA in msg
        # The operator's one-line repair, and the escape hatch, are both named.
        assert "tools/enable_gpu_detection.sh" in msg
        assert ALLOW_CPU_BLUR_ENV in msg

    def test_operator_may_accept_cpu_blur_explicitly(self, monkeypatch, caplog):
        monkeypatch.setenv(ALLOW_CPU_BLUR_ENV, "1")
        with caplog.at_level("WARNING", logger="gomoveshift.pipeline.preflight"):
            check_blur_inference_provider(gpu_expected=True, available=[CPU])
        assert any("CPU" in r.getMessage() for r in caplog.records)

    def test_override_set_to_zero_is_not_an_override(self, monkeypatch):
        monkeypatch.setenv(ALLOW_CPU_BLUR_ENV, "0")
        with pytest.raises(RuntimeError):
            check_blur_inference_provider(gpu_expected=True, available=[CPU])

    def test_default_gpu_expected_is_the_installed_distribution(self, monkeypatch):
        """``onnxruntime-gpu`` present in dist-info == the installer asked for CUDA,
        regardless of which wheel's files currently occupy ``onnxruntime/``."""
        asked = []

        def _dist(name):
            asked.append(name)
            return object()

        monkeypatch.setattr(importlib.metadata, "distribution", _dist)
        with pytest.raises(RuntimeError):
            check_blur_inference_provider(available=[CPU])
        assert asked == ["onnxruntime-gpu"]

    def test_default_gpu_expected_false_when_distribution_absent(self, monkeypatch):
        def _missing(name):
            raise importlib.metadata.PackageNotFoundError(name)

        monkeypatch.setattr(importlib.metadata, "distribution", _missing)
        check_blur_inference_provider(available=[CPU])

    def test_default_available_comes_from_onnxruntime(self, monkeypatch):
        calls = []

        class _Ort:
            @staticmethod
            def preload_dlls():
                calls.append("preload")

            @staticmethod
            def get_available_providers():
                calls.append("providers")
                return [CUDA, CPU]

            class InferenceSession:
                def __init__(self, model, providers=None):
                    calls.append(("session", tuple(providers)))

                @staticmethod
                def get_providers():
                    return [CUDA, CPU]

        monkeypatch.setattr(preflight, "_import_onnxruntime", lambda: _Ort)
        check_blur_inference_provider(gpu_expected=True)
        # preload first: on the GPU wheel the CUDA libs are pip wheels that must be
        # on the loader path before providers are queried, same as the detectors.
        assert calls == ["preload", "providers", ("session", (CUDA, CPU))]

    def test_listed_cuda_but_session_falls_back_to_cpu_fails(self):
        """onnxruntime-gpu (CUDA 12) with CUDA-13 wheels lists
        CUDAExecutionProvider but cannot load libcudart.so.12; the session silently
        runs on CPU. The gate must prove the load, not the listing."""
        with pytest.raises(RuntimeError) as exc:
            check_blur_inference_provider(
                gpu_expected=True, available=[CUDA, CPU], session_providers=[CPU]
            )
        msg = str(exc.value)
        assert "CUDA" in msg
        assert "tools/enable_gpu_detection.sh" in msg
        assert "onnxruntime-gpu[cuda,cudnn]" in msg
        assert ALLOW_CPU_BLUR_ENV in msg

    def test_session_fallback_honours_the_cpu_escape(self, monkeypatch, caplog):
        monkeypatch.setenv(ALLOW_CPU_BLUR_ENV, "1")
        with caplog.at_level("WARNING", logger="gomoveshift.pipeline.preflight"):
            check_blur_inference_provider(
                gpu_expected=True, available=[CUDA, CPU], session_providers=[CPU]
            )
        assert any("CPU" in r.getMessage() for r in caplog.records)

    def test_no_session_built_when_gpu_not_expected(self, monkeypatch):
        def _boom():
            raise AssertionError("must not import onnxruntime to build a session")

        monkeypatch.setattr(preflight, "_import_onnxruntime", _boom)
        check_blur_inference_provider(gpu_expected=False, available=[CPU])

    def test_session_probe_requests_cuda_then_cpu(self, monkeypatch):
        seen = {}

        class _Ort:
            class InferenceSession:
                def __init__(self, model, providers=None):
                    seen["providers"] = list(providers)
                    seen["model"] = model

                @staticmethod
                def get_providers():
                    return [CPU]

        monkeypatch.setattr(preflight, "_import_onnxruntime", lambda: _Ort)
        with pytest.raises(RuntimeError):
            check_blur_inference_provider(gpu_expected=True, available=[CUDA, CPU])
        assert seen["providers"] == [CUDA, CPU]
        assert isinstance(seen["model"], bytes) and seen["model"]

    def test_missing_onnxruntime_is_a_preflight_failure(self, monkeypatch):
        def _absent():
            raise ModuleNotFoundError("No module named 'onnxruntime'", name="onnxruntime")

        monkeypatch.setattr(preflight, "_import_onnxruntime", _absent)
        with pytest.raises(RuntimeError) as exc:
            check_blur_inference_provider(gpu_expected=True)
        assert "onnxruntime" in str(exc.value)


class TestSessionProbe:
    def test_probe_model_is_a_loadable_onnx_graph(self):
        """The embedded model must open on CPU, or the probe raises on every host."""
        ort = pytest.importorskip("onnxruntime")
        session = ort.InferenceSession(preflight._PROBE_MODEL, providers=[CPU])
        assert session.get_providers() == [CPU]


class TestRunPreflightWiring:
    def test_run_preflight_takes_blur_enabled_defaulting_to_skip(self):
        """Same ``None`` skip pattern as ``weather_hwaccel`` (step 7a) so the many
        existing call sites that do not plumb config keep working."""
        param = inspect.signature(run_preflight).parameters["blur_enabled"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default is None

    def test_orchestrator_passes_the_blur_flags(self):
        """The drive orchestrator is the only entry point that owns a config, so
        it is the one that must plumb ``blur_enabled`` — a blur pass that is
        skipped (``skip_blur``) or disabled has nothing to gate."""
        src = Path(preflight.__file__).with_name("orchestrator_preflight.py").read_text("utf-8")
        assert "blur_enabled=config.drive.blur_enabled and not config.drive.skip_blur" in src
