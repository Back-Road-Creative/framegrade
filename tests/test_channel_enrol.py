"""``gomoveshift metrics enrol-channel`` -- fakes only, no network."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from gomoveshift.cli.handlers.metrics import cmd_metrics
from gomoveshift.tracking import TrackingStore
from gomoveshift.tracking.channel_enrol import ENROL_PROJECT, enrol_channel
from gomoveshift.video_types import SHORTS_MAX_DURATION

from test_list_uploads import fake_channel

LONG = "PT10M"
SHORT = f"PT{int(SHORTS_MAX_DURATION)}S"
OVER = f"PT{int(SHORTS_MAX_DURATION) + 1}S"


def _ids(store):
    rec = store.load(ENROL_PROJECT)
    return (
        sorted(s.platform_ids["youtube"] for s in rec.shorts),
        sorted(lf.video_id for lf in rec.long_forms),
    )


def test_enrol_registers_each_upload_once_and_rerun_adds_zero(tmp_path):
    store = TrackingStore(tmp_path)
    yt = fake_channel(5, durations={"v003": LONG, "v004": LONG})
    first = enrol_channel(yt, store)
    assert (first.shorts_added, first.long_added) == (3, 2)
    assert _ids(store) == (["v000", "v001", "v002"], ["v003", "v004"])
    second = enrol_channel(yt, store)
    assert (second.shorts_added, second.long_added) == (0, 0)
    assert _ids(store) == (["v000", "v001", "v002"], ["v003", "v004"])


def test_enrol_splits_shorts_vs_longform_by_duration_le_60(tmp_path):
    store = TrackingStore(tmp_path)
    yt = fake_channel(2, durations={"v000": SHORT, "v001": OVER})
    cov = enrol_channel(yt, store)
    assert _ids(store) == (["v000"], ["v001"])
    assert (cov.shorts_total, cov.long_total) == (1, 1)


def test_enrol_includes_private_and_unlisted(tmp_path):
    store = TrackingStore(tmp_path)
    yt = fake_channel(3, privacy={"v000": "private", "v001": "unlisted"})
    enrol_channel(yt, store)
    assert _ids(store)[0] == ["v000", "v001", "v002"]
    rec = store.load(ENROL_PROJECT)
    assert all(s.posted_at and s.posted_at_source == "platform_confirmed" for s in rec.shorts)


def test_enrol_coverage_line_and_known_ids_skipped(tmp_path, capsys):
    store = TrackingStore(tmp_path)
    store.register_shorts_batch("other", [{"index": 1, "platform_ids": {"youtube": "v000"}}])
    yt = fake_channel(
        3,
        published={
            "v000": "2026-03-01T00:00:00Z",
            "v001": "2026-01-05T00:00:00Z",
            "v002": "2026-02-01T00:00:00Z",
        },
    )
    cov = enrol_channel(yt, store)
    assert cov.shorts_added == 2
    assert cov.line() == (
        "EVIDENCE_COVERAGE shorts_total=3 long_total=0 oldest=2026-01-05 newest=2026-03-01"
    )


def test_listing_failure_prints_ENROL_CHANNEL_FAILED(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("GMS_OUTPUT_DIR", str(tmp_path))
    args = SimpleNamespace(metrics_command="enrol-channel")
    uploader = SimpleNamespace(_build_service=lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    with patch("gomoveshift.cli.handlers.metrics.YouTubeUploader", return_value=uploader):
        assert cmd_metrics(args, SimpleNamespace()) == 2
    assert "ENROL_CHANNEL_FAILED reason=boom" in capsys.readouterr().out


def test_cli_success_prints_coverage(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("GMS_OUTPUT_DIR", str(tmp_path))
    uploader = SimpleNamespace(_build_service=lambda: fake_channel(2))
    args = SimpleNamespace(metrics_command="enrol-channel")
    with patch("gomoveshift.cli.handlers.metrics.YouTubeUploader", return_value=uploader):
        assert cmd_metrics(args, SimpleNamespace()) == 0
    assert "EVIDENCE_COVERAGE shorts_total=2 long_total=0" in capsys.readouterr().out
