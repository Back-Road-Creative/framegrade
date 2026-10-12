"""Metrics collection is BEST-EFFORT and never breaks the pipeline.

1. ``MetricsCollector.collect_all`` tolerates a collector whose
   ``collect_observation`` raises — caught inside
   ``collect_observations_for_short``.
2. The Phase-D post-upload hook
   (``phase_d._collect_metrics_best_effort``) swallows ANY exception and
   returns without raising; uploads are already done so metrics are
   additive. It also registers each LF's uploaded YouTube id.
"""

from __future__ import annotations

from gomoveshift.tracking.collector import MetricsCollector
from gomoveshift.tracking.schema import ShortRecord, TrackingRecord


def test_collect_all_swallows_a_raising_collector() -> None:
    class _RaisingCollector:
        def is_available(self):
            return True

        def collect_observation(self, video_id, published_at=None):
            raise RuntimeError("platform API unavailable")

    mc = MetricsCollector()
    mc._collectors["tiktok"] = _RaisingCollector()
    short = ShortRecord(index=1, platform_ids={"tiktok": "tt_123"})
    record = TrackingRecord(project="proj", shorts=[short])
    # Must not raise — collect_observations_for_short catches it.
    assert mc.collect_all(record) == 0
    assert "tiktok" not in short.metrics


def test_hook_swallows_collector_exception(tmp_path, monkeypatch) -> None:
    from gomoveshift.pipeline.drive_phases import phase_d

    class _Boom:
        def __init__(self, *a, **k):
            pass

        def collect_all(self, record):
            raise NotImplementedError("TikTok metrics not implemented")

    class _Store:
        def __init__(self, *a, **k):
            pass

        def exists(self, project):
            return True

        def update_platform_id(self, *a, **k):
            pass

        def load(self, project):
            return TrackingRecord(project=project)

        def save(self, record):
            raise AssertionError("should not reach save after collect raised")

    monkeypatch.setattr(phase_d, "MetricsCollector", _Boom, raising=False)
    monkeypatch.setattr(phase_d, "TrackingStore", _Store, raising=False)
    # Must not raise — best-effort.
    phase_d._collect_metrics_best_effort(
        project_base="proj", output_base=tmp_path, youtube_uploads={1: {"video_id": "yt"}}
    )


def test_hook_registers_youtube_ids_and_collects(tmp_path, monkeypatch) -> None:
    from gomoveshift.pipeline.drive_phases import phase_d

    calls = {"collect": 0, "save": 0, "registered": {}}

    class _Collector:
        def __init__(self, *a, **k):
            pass

        def collect_all(self, record):
            calls["collect"] += 1
            return len(record.shorts)

    class _Store:
        def __init__(self, *a, **k):
            pass

        def exists(self, project):
            return True

        def load(self, project):
            return TrackingRecord(
                project=project, shorts=[ShortRecord(index=1), ShortRecord(index=2)]
            )

        def update_platform_id(self, project, short_index, platform, video_id):
            calls["registered"][short_index] = (platform, video_id)

        def save(self, record):
            calls["save"] += 1

    monkeypatch.setattr(phase_d, "MetricsCollector", _Collector, raising=False)
    monkeypatch.setattr(phase_d, "TrackingStore", _Store, raising=False)
    phase_d._collect_metrics_best_effort(
        project_base="proj",
        output_base=tmp_path,
        youtube_uploads={1: {"video_id": "yt_1"}, 2: {"video_id": "yt_2"}},
    )
    assert calls["collect"] == 1
    assert calls["save"] == 1
    assert calls["registered"] == {1: ("youtube", "yt_1"), 2: ("youtube", "yt_2")}


def test_hook_roots_the_store_one_level_above_the_project_dir(tmp_path, monkeypatch) -> None:
    """``output_base`` is ``<root>/<project>``; the store adds ``<project>/tracking``
    itself, so the hook must root it at ``output_base.parent`` -- the same root
    ``TrackingStore()`` resolves by default for ``metrics collect``. Previously it used ``output_base`` and looked under
    ``<root>/<project>/<project>/tracking`` where nothing is written."""
    from gomoveshift.pipeline.drive_phases import phase_d

    seen = {}

    class _Store:
        def __init__(self, root=None):
            seen["root"] = root

        def exists(self, project):
            return False

    monkeypatch.setattr(phase_d, "TrackingStore", _Store, raising=False)
    project_dir = tmp_path / "out" / "proj"
    project_dir.mkdir(parents=True)
    phase_d._collect_metrics_best_effort(
        project_base="proj", output_base=project_dir, youtube_uploads={1: {"video_id": "yt"}}
    )
    assert seen["root"] == project_dir.parent
