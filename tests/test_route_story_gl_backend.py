"""The scene renders on the real GPU when there is one, and never silently on software.

The capture could be pinned to SwiftShader -- Chromium's CPU software GL --
on the belief that a headless process cannot reach the host's GPU. It can: a
settled frame rasterises roughly twice as fast through ANGLE's GL-EGL backend
on a real GPU, and the pixels are the same apart from edge antialiasing.

What has to stay true for that to be safe:

* A box with no usable GPU still renders. ``auto`` tries the GPU and falls back,
  and it decides by what the browser REPORTS it is drawing with, not by whether
  the launch flags were accepted -- Chromium accepts ``--use-angle=gl`` happily
  and then draws on SwiftShader anyway.
* An operator who demands the GPU gets a refusal rather than a quiet software
  render at half the speed.
* The frame cache never replays one backend's frames into the other's film. The
  pixels differ by a hair, and a film stitched from both would flicker.
* Frames already cached from software renders keep their keys.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from gomoveshift.pipeline.route_story.scene import frame_cache, gl_backend

RTX = "ANGLE (Microsoft Corporation, D3D12 (NVIDIA GeForce RTX 4070 Laptop GPU), OpenGL ES 3.1)"
SWIFTSHADER = (
    "ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero) (0x0000C0DE)), SwiftShader driver)"
)
LLVMPIPE = "ANGLE (Mesa, Vulkan 1.4.318 (llvmpipe (LLVM 20.1.2 256 bits) (0x00000000)), llvmpipe)"


class TestWhatCountsAsARealGpu:
    def test_a_named_gpu_is_hardware(self) -> None:
        assert gl_backend.is_hardware_renderer(RTX) is True

    @pytest.mark.parametrize("renderer", [SWIFTSHADER, LLVMPIPE, "softpipe", "", "  "])
    def test_every_software_rasteriser_is_not(self, renderer: str) -> None:
        """llvmpipe is the one that matters: it is what ``--use-angle=vulkan``
        resolved to on some hosts, and it looks like a success."""
        assert gl_backend.is_hardware_renderer(renderer) is False


class TestWhichBackendsATrialWalks:
    def test_auto_tries_the_gpu_first_then_software(self) -> None:
        assert gl_backend.candidates("auto") == ("gpu", "swiftshader")

    def test_an_explicit_choice_is_never_second_guessed(self) -> None:
        assert gl_backend.candidates("gpu") == ("gpu",)
        assert gl_backend.candidates("swiftshader") == ("swiftshader",)

    def test_an_unknown_backend_is_refused_by_name(self) -> None:
        with pytest.raises(ValueError, match="metal"):
            gl_backend.candidates("metal")


class TestWhetherALaunchedBrowserIsAccepted:
    def test_the_gpu_candidate_needs_a_real_gpu_behind_it(self) -> None:
        assert gl_backend.accepts("gpu", RTX) is True
        assert gl_backend.accepts("gpu", SWIFTSHADER) is False

    def test_software_is_accepted_whatever_it_reports(self) -> None:
        assert gl_backend.accepts("swiftshader", SWIFTSHADER) is True

    def test_a_browser_with_no_webgl_at_all_is_never_accepted(self) -> None:
        assert gl_backend.accepts("swiftshader", "") is False


class TestOneDecisionPerProcess:
    """Frames are strided across capture workers: two workers that decided
    differently would interleave two rasterisers frame by frame."""

    @pytest.fixture(autouse=True)
    def _fresh(self) -> Any:
        gl_backend.forget_resolution()
        yield
        gl_backend.forget_resolution()

    def test_auto_takes_the_gpu_when_the_browser_really_draws_on_it(self) -> None:
        assert gl_backend.resolve(lambda candidate: RTX, "auto") == "gpu"

    def test_auto_falls_back_when_the_gpu_launch_quietly_drew_on_software(self) -> None:
        probed: list[str] = []

        def probe(candidate: str) -> str:
            probed.append(candidate)
            return SWIFTSHADER

        assert gl_backend.resolve(probe, "auto") == "swiftshader"
        assert probed == ["gpu", "swiftshader"]

    def test_auto_falls_back_when_the_gpu_browser_cannot_launch(self) -> None:
        def probe(candidate: str) -> str:
            if candidate == "gpu":
                raise OSError("no /dev/dxg")
            return SWIFTSHADER

        assert gl_backend.resolve(probe, "auto") == "swiftshader"

    def test_a_demanded_gpu_that_is_not_there_is_refused_not_downgraded(self) -> None:
        with pytest.raises(gl_backend.GlBackendError, match="SwiftShader"):
            gl_backend.resolve(lambda candidate: SWIFTSHADER, "gpu")

    def test_the_second_worker_gets_the_first_workers_answer_without_probing(self) -> None:
        assert gl_backend.resolve(lambda candidate: RTX, "auto") == "gpu"

        def never(candidate: str) -> str:
            raise AssertionError("probed twice")

        assert gl_backend.resolve(never, "auto") == "gpu"

    def test_a_refusal_is_not_remembered_as_an_answer(self) -> None:
        with pytest.raises(gl_backend.GlBackendError):
            gl_backend.resolve(lambda candidate: "", "gpu")
        assert gl_backend.resolve(lambda candidate: RTX, "gpu") == "gpu"


class TestTheRequestedBackendComesFromTheEnvironment:
    def test_it_defaults_to_software(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The GPU is opt-in. One GPU browser reaches ready and captures much
        faster than software, but two started together can fail to become ready
        at all -- and a film opens one browser per capture worker. A default
        that aborts every multi-worker render an hour in is not a default."""
        monkeypatch.delenv("GMS_ROUTE_STORY_GL_BACKEND", raising=False)
        assert gl_backend.requested_backend() == "swiftshader"

    def test_an_operator_can_pin_software(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GMS_ROUTE_STORY_GL_BACKEND", "swiftshader")
        assert gl_backend.requested_backend() == "swiftshader"

    def test_a_typo_is_refused_rather_than_read_as_auto(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GMS_ROUTE_STORY_GL_BACKEND", "gpuu")
        with pytest.raises(ValueError, match="gpuu"):
            gl_backend.requested_backend()


class TestLaunchConfiguration:
    def test_software_launches_exactly_as_it_always_has(self) -> None:
        assert gl_backend.launch_args("swiftshader") == (
            "--use-angle=swiftshader",
            "--enable-unsafe-swiftshader",
            "--ignore-gpu-blocklist",
        )
        assert gl_backend.launch_env("swiftshader", {"PATH": "/bin"}) == {"PATH": "/bin"}

    def test_the_gpu_launch_names_the_backend_that_actually_reached_it(self) -> None:
        assert "--use-angle=gl-egl" in gl_backend.launch_args("gpu")

    def test_the_gpu_launch_selects_the_d3d12_driver_without_clobbering_a_choice(self) -> None:
        assert gl_backend.launch_env("gpu", {})["GALLIUM_DRIVER"] == "d3d12"
        assert gl_backend.launch_env("gpu", {"GALLIUM_DRIVER": "zink"})["GALLIUM_DRIVER"] == "zink"


def _cache(tmp_path: Path, **kwargs: Any) -> frame_cache.FrameCache:
    bundle = tmp_path / "bundle"
    bundle.mkdir(exist_ok=True)
    (bundle / "manifest.json").write_text("{}", encoding="utf-8")
    scene = tmp_path / "scene"
    scene.mkdir(exist_ok=True)
    (scene / "scene.js").write_text("//", encoding="utf-8")
    (scene / "index.html").write_text("<html>", encoding="utf-8")
    return frame_cache.FrameCache(
        bundle, 640, 360, cache_root=tmp_path / "cache", scene_dir=scene, **kwargs
    )


class TestTheFrameCacheNeverMixesBackends:
    POSE = {"longitude": -83.5, "latitude": 35.25, "zoom": 14.0}

    def test_a_gpu_frame_and_a_software_frame_have_different_keys(self, tmp_path: Path) -> None:
        software = _cache(tmp_path, gl_backend="swiftshader")
        gpu = _cache(tmp_path, gl_backend="gpu")
        assert software._key_for(self.POSE) != gpu._key_for(self.POSE)

    def test_software_frames_already_on_disk_keep_their_keys(self, tmp_path: Path) -> None:
        """Every frame cached before this existed was a software frame."""
        assert _cache(tmp_path)._key_for(self.POSE) == _cache(
            tmp_path, gl_backend="swiftshader"
        )._key_for(self.POSE)
