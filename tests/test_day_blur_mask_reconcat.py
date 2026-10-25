"""A re-concatenated raw keeps its blur mask when its episodes are unchanged.

Failure mode: a day's raw concat was deleted after a failed run and rebuilt from
the same three episodes, byte-for-byte the same size (132,453,663,028). The mask
was keyed on the raw's own ``size:mtime_ns``, so the fresh mtime threw away a
finished ~16 h detection pass and started it again. The key now names what the
raw was built from: each episode's fingerprint plus the raw's size.
"""

import os
from pathlib import Path
from types import SimpleNamespace

from gomoveshift.config import Config
from gomoveshift.pipeline import blur_gate
from gomoveshift.pipeline import day_master_encoder as dme

from tests import test_day_master_blur as tdmb
from tests.test_day_master_blur import _batch, _mock_encode


def _episodes(tmp_path, n=3):
    eps = []
    for i in range(n):
        p = tmp_path / f"episode_{i:02d}.mp4"
        p.write_bytes(b"ep" * (i + 1))
        eps.append(p)
    return eps


def _bump_mtime(path: Path):
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))


class TestConcatProvenanceFingerprint:
    def test_same_episodes_and_size_survive_a_new_raw_mtime(self, tmp_path):
        eps = _episodes(tmp_path)
        raw = tmp_path / "raw.mp4"
        raw.write_bytes(b"raw")
        before = dme._concat_provenance_fingerprint(eps, raw)
        _bump_mtime(raw)
        assert before is not None
        assert dme._concat_provenance_fingerprint(eps, raw) == before

    def test_a_changed_episode_changes_it(self, tmp_path):
        eps = _episodes(tmp_path)
        raw = tmp_path / "raw.mp4"
        raw.write_bytes(b"raw")
        before = dme._concat_provenance_fingerprint(eps, raw)
        _bump_mtime(eps[1])
        assert dme._concat_provenance_fingerprint(eps, raw) != before

    def test_a_different_raw_size_changes_it(self, tmp_path):
        eps = _episodes(tmp_path)
        raw = tmp_path / "raw.mp4"
        raw.write_bytes(b"raw")
        before = dme._concat_provenance_fingerprint(eps, raw)
        raw.write_bytes(b"truncated-or-longer")
        assert dme._concat_provenance_fingerprint(eps, raw) != before

    def test_episode_order_matters(self, tmp_path):
        eps = _episodes(tmp_path)
        raw = tmp_path / "raw.mp4"
        raw.write_bytes(b"raw")
        assert dme._concat_provenance_fingerprint(eps, raw) != dme._concat_provenance_fingerprint(
            list(reversed(eps)), raw
        )

    def test_a_missing_episode_or_raw_gives_none(self, tmp_path):
        eps = _episodes(tmp_path)
        raw = tmp_path / "raw.mp4"
        raw.write_bytes(b"raw")
        eps[2].unlink()
        assert dme._concat_provenance_fingerprint(eps, raw) is None
        assert dme._concat_provenance_fingerprint(eps[:2], tmp_path / "gone.mp4") is None
        assert dme._concat_provenance_fingerprint([], raw) is None


class TestBlurMaskReuseAcrossReconcat:
    def _setup(self, tmp_path, monkeypatch):
        tdmb.TestBuildDayBlurMask()._no_network_models(monkeypatch)
        monkeypatch.setattr("redactcam.timeline.probe_video", lambda p: (30.0, 90, 1920, 1080))
        calls = []

        def _detect(*a, **k):
            calls.append(1)
            return ({0: [(10, 10, 20, 20)]}, 90, 1, 0)

        monkeypatch.setattr("redactcam.detect_and_track", _detect)
        monkeypatch.setattr(
            "redactcam.materialize_mask_video",
            lambda tl, p, **k: (p.write_bytes(b"mask"), p)[1],
        )
        raw = tmp_path / "raw.mp4"
        raw.write_bytes(b"raw")
        cfg = Config()
        cfg.drive.blur_vehicle_cabin_frac = 0.0
        return raw, cfg, tmp_path / "m.mkv", tmp_path / "r.json", calls

    def test_reconcat_with_the_same_provenance_reuses_the_mask(self, tmp_path, monkeypatch):
        raw, cfg, mask, sidecar, calls = self._setup(tmp_path, monkeypatch)
        dme._build_day_blur_mask(raw, cfg, mask, sidecar, "d", source_fingerprint="concat/v1:a")
        _bump_mtime(raw)  # the rebuilt raw: same bytes, new mtime
        dme._build_day_blur_mask(raw, cfg, mask, sidecar, "d", source_fingerprint="concat/v1:a")
        assert calls == [1]

    def test_a_different_provenance_re_detects(self, tmp_path, monkeypatch):
        raw, cfg, mask, sidecar, calls = self._setup(tmp_path, monkeypatch)
        dme._build_day_blur_mask(raw, cfg, mask, sidecar, "d", source_fingerprint="concat/v1:a")
        dme._build_day_blur_mask(raw, cfg, mask, sidecar, "d", source_fingerprint="concat/v1:b")
        assert calls == [1, 1]

    def test_without_provenance_a_new_raw_mtime_still_re_detects(self, tmp_path, monkeypatch):
        # Picture steps (deflicker/stabilize) pass no provenance: the old key holds.
        raw, cfg, mask, sidecar, calls = self._setup(tmp_path, monkeypatch)
        dme._build_day_blur_mask(raw, cfg, mask, sidecar, "d")
        _bump_mtime(raw)
        dme._build_day_blur_mask(raw, cfg, mask, sidecar, "d")
        assert calls == [1, 1]

    def test_a_mask_keyed_by_the_old_file_fingerprint_is_still_reused(self, tmp_path, monkeypatch):
        # A sidecar written before this change carries the raw's size:mtime_ns.
        raw, cfg, mask, sidecar, calls = self._setup(tmp_path, monkeypatch)
        dme._build_day_blur_mask(raw, cfg, mask, sidecar, "d")
        dme._build_day_blur_mask(raw, cfg, mask, sidecar, "d", source_fingerprint="concat/v1:a")
        assert calls == [1]


class TestEncodePassesProvenance:
    def _run(self, tmp_path, monkeypatch, cfg):
        _mock_encode(monkeypatch)

        def _run_writes_output(cmd, *a, **k):
            # The concat writes its output, so the raw exists to be fingerprinted.
            if str(cmd[-1]).endswith(".mp4"):
                Path(cmd[-1]).write_bytes(b"raw")
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

        monkeypatch.setattr(
            "gomoveshift.pipeline.day_master_encoder.subprocess.run", _run_writes_output
        )
        seen = {}

        def _build(raw, cfg_, mask, sidecar, date, **k):
            seen.update(k)
            mask.write_bytes(b"mask")
            return mask

        monkeypatch.setattr(dme, "_build_day_blur_mask", _build)
        monkeypatch.setattr(blur_gate, "check_day_master_cabin_coverage", lambda c, v, m: None)
        dme.encode_day_master(_batch(tmp_path, n_ep=2), tmp_path, config=cfg)
        return seen

    def test_raw_concat_blur_is_keyed_by_its_episodes(self, tmp_path, monkeypatch):
        seen = self._run(tmp_path, monkeypatch, Config())
        assert str(seen.get("source_fingerprint", "")).startswith("concat/v1:")
