"""astro_benchmark tests: synthetic-only. Fakes set on astro_orchestrator pre-fork."""

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import gomoveshift.pipeline.photos.astro_benchmark as bm
import gomoveshift.pipeline.photos.astro_orchestrator as om
from gomoveshift.pipeline.photos.astro_calibrate import DecodedAstroFrame
from gomoveshift.pipeline.photos.astro_develop import AstroDevelopResult
from gomoveshift.pipeline.photos.astro_orchestrator import AstroReport, PreparedFrames
from gomoveshift.pipeline.photos.astro_quality import CoverageStats
from gomoveshift.pipeline.photos.astro_stack import StackResult
from gomoveshift.pipeline.photos.astro_triage import FrameMetrics, TriageReport

_RGB8 = np.full((8, 8, 3), 0.5, dtype=np.float32)
_RGB16 = np.full((8, 8, 3), 1000, dtype=np.uint16)


def _light_paths(tmp_path, n=3):
    paths = [tmp_path / "source" / f"light{i}.nef" for i in range(n)]
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(paths):
        p.write_bytes(f"fake-nef-{i}".encode())
    return paths


def _dark_paths(tmp_path, n=4):
    paths = [tmp_path / "darks" / f"dark{i}.nef" for i in range(n)]
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(paths):
        p.write_bytes(f"fake-dark-{i}".encode())
    return paths


def _prepare(alloc_bytes):
    def _fn(light_paths, *, dark_paths=(), horizon_mask=None):
        if alloc_bytes:
            np.ones(alloc_bytes // 8, dtype=np.float64)  # touched; ru_maxrss is a high-water mark
        frames = tuple(DecodedAstroFrame(_RGB8, Path(p), "stub", 16) for p in light_paths)
        metrics = tuple(FrameMetrics(Path(p), 10, 0.1, 2.0, [], True) for p in light_paths)
        return PreparedFrames(frames, metrics, TriageReport(metrics, (), ()), 0, ())

    return _fn


def _stack_run(prepared, out_path, *, horizon_frac=None, meteors=None):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(b"tiff-stub")
    stack = StackResult(_RGB8, len(prepared.frames), 0.02)
    develop = AstroDevelopResult(_RGB16, 0.0, [1.0, 1.0, 1.0], 0.1, 0.05, (1.0, 1.0, 1.0))
    return AstroReport(out_path, None, stack, develop, (), 0)


def _patch(monkeypatch, alloc_bytes=0):
    monkeypatch.setattr(om, "prepare_frames", _prepare(alloc_bytes))
    monkeypatch.setattr(om, "stack_run", _stack_run)


def test_ru_maxrss_and_hash_conversions(tmp_path):
    assert bm.ru_maxrss_to_bytes(2048) == 2048 * 1024
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello astro benchmark")
    assert bm._hash_file(p) == hashlib.sha256(b"hello astro benchmark").hexdigest()


def test_source_directory_write_refusal_fires(tmp_path, monkeypatch):
    _patch(monkeypatch)
    lights = _light_paths(tmp_path)
    with pytest.raises(bm.SourceDirectoryWriteError):
        bm.run_benchmark(lights, lights[0].parent / "nested_out")
    with pytest.raises(bm.SourceDirectoryWriteError):
        bm.run_benchmark(lights, lights[0].parent)


def test_zero_darks_run_reports_an_honest_schema_and_panels(tmp_path, monkeypatch):
    import gomoveshift.config as cfg_mod

    real_get_config = cfg_mod.get_config
    pinned = bm._pinned_get_config(real_get_config, tmp_path / "out")
    assert pinned().photos.he_dng_cache_dir == str(tmp_path / "out")
    _patch(monkeypatch)
    lights = _light_paths(tmp_path)
    payload = bm.run_benchmark(lights, tmp_path / "out")
    assert cfg_mod.get_config is real_get_config  # the HE/HE* DNG pin never leaks past the call
    assert payload["ok"] is True
    assert payload["inputs"]["dark_count"] == 0 and payload["inputs"]["light_count"] == len(lights)
    top_keys = {"schema_version", "inputs", "capture_identity", "rss", "unavailable_panels"}
    rss_keys = {"measured_peak_bytes", "estimate_bytes", "ratio_estimate_over_measured"}
    assert all(k <= d.keys() for k, d in ((top_keys, payload), (rss_keys, payload["rss"])))
    assert payload["frames"]["decoded_count"] == len(lights) == payload["stack"]["n_frames"]
    artifacts = payload["output_artifacts"]
    assert artifacts["image"]["sha256"] == hashlib.sha256(b"tiff-stub").hexdigest()
    assert (tmp_path / "out" / "benchmark.json").exists()
    unavailable = {"background_model", "coverage", "rejection", "sky_mask"}
    assert set(payload["unavailable_panels"]) == unavailable
    assert set(artifacts) - {"image"} == {"linear_stack_crop", "final_render_crop", "contact_sheet"}


def test_benchmark_carries_the_run_s_real_coverage_stats(tmp_path, monkeypatch):
    """benchmark.json's ``unavailable_panels['coverage']`` now claims the coverage
    NUMBERS exist and are merely unrendered. That sentence is true only because the
    payload actually carries them -- deleting the one wiring line leaves every other
    astro test green, so this is the only thing standing between the claim and a lie."""
    _patch(monkeypatch)
    stats = CoverageStats(min_count=1, median_count=3.0, below_floor_fraction=0.25)

    def _stack_run_with_coverage(prepared, out_path, **kwargs):
        report = _stack_run(prepared, out_path, **kwargs)
        stack = replace(report.stack, coverage=stats)
        return replace(report, stack=stack)

    monkeypatch.setattr(om, "stack_run", _stack_run_with_coverage)
    payload = bm.run_benchmark(_light_paths(tmp_path), tmp_path / "out")
    assert payload["ok"] is True
    assert payload["stack"]["coverage"] == {
        "min_count": 1,
        "median_count": 3.0,
        "below_floor_fraction": 0.25,
    }
    # ... and the honesty string stays honest: numbers present, panel still not drawn.
    assert "coverage" in payload["unavailable_panels"]
    assert "coverage" not in set(payload["output_artifacts"]) - {"image"}


def _spy_estimator(recorded, gib=2.0):
    def _spy(frame_count, megapixels, *, dark_count=0, meteor_reinsertion_enabled=False):
        recorded.append((frame_count, megapixels, dark_count, meteor_reinsertion_enabled))
        return gib

    return _spy


def test_estimate_is_shaped_by_the_run_it_actually_measured(tmp_path, monkeypatch):
    """The estimator's dark term is what this module exists to calibrate: an un-passed
    ``dark_count`` biases ``ratio_estimate_over_measured`` low by exactly the dark term,
    and the bias grows with dark count -- so the ratio would push a constants tuner the
    wrong way. Measuring condition here: 3 synthetic 8x8-px lights (6.4e-05 MP) and 4
    darks, with the estimator spied so the assertion is on the arguments, not on GiB."""
    _patch(monkeypatch)
    recorded = []
    monkeypatch.setattr(bm, "estimate_astro_stack_ram_gb", _spy_estimator(recorded))
    lights, darks = _light_paths(tmp_path), _dark_paths(tmp_path, 4)

    payload = bm.run_benchmark(lights, tmp_path / "out", dark_paths=darks)

    assert payload["ok"] is True
    assert recorded == [(len(lights), 64 / 1_000_000, len(darks), False)]
    # ...and the reported ratio is computed against THAT estimate, not a stale one.
    assert payload["rss"]["estimate_bytes"] == int(2.0 * 1024**3)
    assert payload["rss"]["ratio_estimate_over_measured"] == (
        payload["rss"]["estimate_bytes"] / payload["rss"]["measured_peak_bytes"]
    )


def test_meteor_term_follows_the_reinsertion_the_run_measured(tmp_path, monkeypatch):
    """``stack_run`` allocates a meteor layer only when it composites one, so the run's
    own ``meteors_reinserted`` -- not the requested policy -- is what the estimate must
    reflect. Measuring condition: same 3 lights, no darks, one stub run reinserting 2
    streaks and one reinserting 0."""
    _patch(monkeypatch)

    def _stack_run_with_meteors(prepared, out_path, **kwargs):
        return replace(_stack_run(prepared, out_path, **kwargs), meteors_reinserted=2)

    monkeypatch.setattr(om, "stack_run", _stack_run_with_meteors)
    recorded = []
    monkeypatch.setattr(bm, "estimate_astro_stack_ram_gb", _spy_estimator(recorded))
    bm.run_benchmark(_light_paths(tmp_path), tmp_path / "out_meteors")
    monkeypatch.setattr(om, "stack_run", _stack_run)
    bm.run_benchmark(_light_paths(tmp_path), tmp_path / "out_none")

    assert [flag for *_, flag in recorded] == [True, False]


def test_dark_frames_raise_the_estimate_by_the_real_dark_term(tmp_path, monkeypatch):
    """End-to-end through the shipped estimator (no spy): at 6.4e-05 MP (8x8 px), 4 darks
    add 4 x 64 px x 10 B/px = 2560 bytes to the estimate. Same shape at 20.8768 MP with 15
    frames: 18.35 GB with dark_count=0 vs 19.13 GB with dark_count=4."""
    _patch(monkeypatch)
    lights = _light_paths(tmp_path)
    without = bm.run_benchmark(lights, tmp_path / "out_nodarks")
    with_darks = bm.run_benchmark(
        lights, tmp_path / "out_darks", dark_paths=_dark_paths(tmp_path, 4)
    )

    delta = with_darks["rss"]["estimate_bytes"] - without["rss"]["estimate_bytes"]
    assert delta == pytest.approx(4 * 64 * 10, abs=2)


def test_child_rss_measurement_is_scoped_to_its_own_run(tmp_path, monkeypatch):
    # BIG first: a cross-call RUSAGE_CHILDREN max would leak into the later SMALL run.
    alloc_bytes = 64 * 1024 * 1024
    _patch(monkeypatch, alloc_bytes=alloc_bytes)
    big = bm.run_benchmark(_light_paths(tmp_path), tmp_path / "out_big")
    _patch(monkeypatch, alloc_bytes=0)
    small = bm.run_benchmark(_light_paths(tmp_path), tmp_path / "out_small")
    delta = big["rss"]["measured_peak_bytes"] - small["rss"]["measured_peak_bytes"]
    assert delta >= alloc_bytes * 0.5


class _InheritedPeakContext:
    """A fork context whose child first holds (then frees) ``inherited_bytes`` BEFORE the
    benchmark's own worker starts: a deterministic stand-in for a high-water mark the child
    carries into its run (the CI flake saw the 64 MiB and 0-byte runs report one figure)."""

    def __init__(self, inherited_bytes):
        self._ctx = bm.multiprocessing.get_context("fork")
        self._inherited_bytes = inherited_bytes

    def Queue(self):  # noqa: N802 -- mirrors multiprocessing.context API
        return self._ctx.Queue()

    def Process(self, target, args):  # noqa: N802
        return self._ctx.Process(target=_hold_then_run, args=(self._inherited_bytes, target, args))


def _hold_then_run(inherited_bytes, target, args):
    held = bytearray(inherited_bytes)
    for offset in range(0, len(held), 4096):
        held[offset] = 1  # touched: resident, so it lands in the high-water mark
    del held
    target(*args)


def _big_then_small(tmp_path, monkeypatch, ctx):
    _patch(monkeypatch, alloc_bytes=64 * 1024 * 1024)
    big = bm.run_benchmark(_light_paths(tmp_path), tmp_path / "out_big", mp_context=ctx)
    _patch(monkeypatch, alloc_bytes=0)
    small = bm.run_benchmark(_light_paths(tmp_path), tmp_path / "out_small", mp_context=ctx)
    return big, small


def test_child_peak_excludes_memory_held_before_its_run(tmp_path, monkeypatch):
    """The child holds 256 MiB before its worker starts; the 64 MiB-probe run and the empty
    run must still differ by the probe, and neither may report the inherited 256 MiB."""
    big, small = _big_then_small(tmp_path, monkeypatch, _InheritedPeakContext(256 * 1024 * 1024))
    probe = 64 * 1024 * 1024
    delta = big["rss"]["measured_peak_bytes"] - small["rss"]["measured_peak_bytes"]
    assert delta >= probe * 0.5
    # A forked child starts resident with its parent's pages, so its reset peak is the
    # parent's footprint (an xdist worker late in a long run holds hundreds of MiB). Compare
    # against the same empty run from a child that held nothing, not an absolute ceiling.
    plain = bm.run_benchmark(
        _light_paths(tmp_path), tmp_path / "out_plain", mp_context=_InheritedPeakContext(0)
    )
    inherited = small["rss"]["measured_peak_bytes"] - plain["rss"]["measured_peak_bytes"]
    assert inherited < 128 * 1024 * 1024
    assert "reset at child start" in big["rss"]["unit_note"]


@pytest.mark.parametrize("broken", ["clear_refs", "status"])
def test_unresettable_peak_falls_back_labelled_not_silently(tmp_path, monkeypatch, broken):
    """Fail closed: when the reset is refused or VmHWM cannot be read back, the figure is the
    un-reset ru_maxrss AND says so in ``unit_note`` -- never passed off as a per-run peak."""
    if broken == "clear_refs":
        monkeypatch.setattr(bm, "_CLEAR_REFS", str(tmp_path / "no_such_dir" / "clear_refs"))
    else:
        status = tmp_path / "status"
        status.write_text("Name:\tpython\nVmRSS:\t   100 kB\n")  # no VmHWM line
        monkeypatch.setattr(bm, "_PROC_STATUS", str(status))
    _patch(monkeypatch)
    payload = bm.run_benchmark(_light_paths(tmp_path), tmp_path / "out", mp_context=None)

    assert payload["ok"] is True
    assert payload["rss"]["measured_peak_bytes"] > 0
    assert "NOT reset" in payload["rss"]["unit_note"]


# The RSS helpers run inside the forked child, where coverage cannot follow them; these
# call them in-process against a temp stand-in for /proc so each branch is measured.


def test_reset_peak_rss_writes_5_and_reports_a_refused_kernel(tmp_path, monkeypatch):
    clear_refs = tmp_path / "clear_refs"
    monkeypatch.setattr(bm, "_CLEAR_REFS", str(clear_refs))
    assert bm._reset_peak_rss() is True
    assert clear_refs.read_text() == "5"
    monkeypatch.setattr(bm, "_CLEAR_REFS", str(tmp_path / "no_such_dir" / "clear_refs"))
    assert bm._reset_peak_rss() is False


def test_read_vmhwm_kib_parses_status_and_tolerates_a_missing_or_malformed_file(
    tmp_path, monkeypatch
):
    status = tmp_path / "status"
    status.write_text("Name:\tpython\nVmRSS:\t   512 kB\nVmHWM:\t  4096 kB\n")
    monkeypatch.setattr(bm, "_PROC_STATUS", str(status))
    assert bm._read_vmhwm_kib() == 4096
    status.write_text("VmHWM:\n")  # no value column
    assert bm._read_vmhwm_kib() is None
    status.write_text("Name:\tpython\n")  # no VmHWM line at all
    assert bm._read_vmhwm_kib() is None
    monkeypatch.setattr(bm, "_PROC_STATUS", str(tmp_path / "missing"))
    assert bm._read_vmhwm_kib() is None


def test_measure_peak_rss_trusts_vmhwm_only_after_a_reset(tmp_path, monkeypatch):
    status = tmp_path / "status"
    status.write_text("VmHWM:\t  2048 kB\n")
    monkeypatch.setattr(bm, "_PROC_STATUS", str(status))
    assert bm._measure_peak_rss_kib(True) == (2048, "vmhwm")
    kib, source = bm._measure_peak_rss_kib(False)  # no reset: VmHWM is not per-run, fall back
    assert source == "ru_maxrss" and kib > 0
    monkeypatch.setattr(bm, "_PROC_STATUS", str(tmp_path / "missing"))
    kib, source = bm._measure_peak_rss_kib(True)  # reset took but VmHWM unreadable
    assert source == "ru_maxrss" and kib > 0


class _ListQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


def test_child_worker_in_process_labels_its_rss_source_on_success_and_failure(
    tmp_path, monkeypatch
):
    _patch(monkeypatch)
    monkeypatch.setattr(bm, "_CLEAR_REFS", str(tmp_path / "clear_refs"))
    lights = _light_paths(tmp_path)
    queue = _ListQueue()
    bm._child_worker(lights, (), tmp_path / "out" / "stack.tiff", None, None, queue)
    (payload,) = queue.items
    assert payload["ok"] is True and payload["rss_self_kib"] > 0
    assert payload["rss_source"] in ("vmhwm", "ru_maxrss")

    def _boom(*a, **k):
        raise RuntimeError("decode exploded")

    monkeypatch.setattr(om, "prepare_frames", _boom)
    queue = _ListQueue()
    bm._child_worker(lights, (), tmp_path / "out2" / "stack.tiff", None, None, queue)
    (payload,) = queue.items
    assert payload["ok"] is False and payload["error"] == "RuntimeError: decode exploded"
    assert payload["rss_self_kib"] > 0 and payload["rss_source"] in ("vmhwm", "ru_maxrss")


def test_hold_then_run_frees_the_held_block_before_the_target_runs():
    seen = []
    _hold_then_run(4096 * 4, lambda *args: seen.append(args), (1, "two"))
    assert seen == [(1, "two")]
