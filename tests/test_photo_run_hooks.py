"""The photo engine takes its ledger write and voice rotation as callables.

``PhotoOrchestrator`` no longer imports ``content_ledger`` or ``seo_manager``; the
publish side supplies ``ledger_booker`` / ``voice_source`` / ``theme_recorder`` from
``publishing.photo_run_hooks``. The rotator hardening cases live in
``test_photo_orchestrator.py::TestThemeRotatorHardening``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

from gomoveshift.pipeline.photos.orchestrator import PhotoOrchestrator
from gomoveshift.pipeline.photos.voice_styles import VoiceStyle
from gomoveshift.publishing import photo_run_hooks as hooks

ROWS = [
    {"date": "2026-02-23", "time": "09:00", "title": "Ridge at dawn", "source_file": "/x/p.csv"},
    {"date": "2026-02-24", "time": "09:00", "title": "Creek", "source_file": "/x/p.csv"},
]


def test_hooks_dict_names_the_three_engine_seams():
    h = hooks.photo_run_hooks()
    assert set(h) == {"ledger_booker", "voice_source", "theme_recorder", "upscale"}
    assert h["ledger_booker"] is hooks.book_photo_slots
    assert h["voice_source"] is hooks.next_photo_voice_style
    assert h["theme_recorder"] is hooks.record_photo_theme_usage


def test_book_photo_slots_builds_photo_entries():
    with patch.object(hooks, "record_confirmed_upload", return_value=True) as rec:
        assert hooks.book_photo_slots(ROWS) is True
    entries = rec.call_args.args[0]
    assert [e.date for e in entries] == ["2026-02-23", "2026-02-24"]
    assert {e.content_type for e in entries} == {"photo"}
    assert {e.source for e in entries} == {"photo_pipeline"}
    assert entries[0].title == "Ridge at dawn"
    assert entries[0].source_file == "/x/p.csv"


def test_engine_hands_rows_to_the_injected_booker(tmp_path):
    seen = []
    orch = PhotoOrchestrator(
        output_dir=tmp_path, ledger_booker=lambda rows: seen.append(rows) or True
    )
    slots = [SimpleNamespace(date="2026-02-23", time="09:00")]
    caps = [SimpleNamespace(main_caption="x" * 60)]
    orch._book_slots(slots, caps, tmp_path / "p.csv")
    assert seen == [
        [
            {
                "date": "2026-02-23",
                "time": "09:00",
                "title": "x" * 40,
                "source_file": str(tmp_path / "p.csv"),
            }
        ]
    ]


def test_missing_booker_marks_the_slots_unbooked(tmp_path):
    orch = PhotoOrchestrator(site_root=tmp_path / "site", output_dir=tmp_path)
    slots = [SimpleNamespace(date="2026-02-23", time="09:00")]
    with patch.object(orch, "_mark_ledger_unbooked") as mark:
        orch._book_slots(slots, [], tmp_path / "p.csv")
    mark.assert_called_once()


def test_engine_defaults_without_hooks(tmp_path):
    orch = PhotoOrchestrator(site_root=tmp_path / "site", output_dir=tmp_path)
    assert orch._get_voice_style_from_rotator() == VoiceStyle.FIRST_PERSON
    orch._record_theme_usage(SimpleNamespace(voice_style="voice_first_person", week_start="w"))


def test_engine_uses_injected_voice_and_recorder(tmp_path):
    calls = []
    orch = PhotoOrchestrator(
        output_dir=tmp_path,
        voice_source=lambda: VoiceStyle.MINIMAL_ZEN,
        theme_recorder=lambda v, w: calls.append((v, w)),
    )
    assert orch._get_voice_style_from_rotator() == VoiceStyle.MINIMAL_ZEN
    orch._record_theme_usage(
        SimpleNamespace(voice_style="voice_minimal_zen", week_start="2025-01-06")
    )
    assert calls == [("voice_minimal_zen", "2025-01-06")]


def test_engine_survives_a_failing_voice_source_and_recorder(tmp_path):
    def boom(*a):
        raise RuntimeError("rotation state corrupt")

    orch = PhotoOrchestrator(output_dir=tmp_path, voice_source=boom, theme_recorder=boom)
    assert orch._get_voice_style_from_rotator() == VoiceStyle.FIRST_PERSON
    orch._record_theme_usage(SimpleNamespace(voice_style="v", week_start="w"))  # must not raise


def test_upscale_policy_fails_closed_to_off(monkeypatch):
    from gomoveshift.publishing import brand_policy

    def unreadable(slug):
        raise OSError("brand.yaml unreadable")

    monkeypatch.setattr(brand_policy, "load_media_profile", unreadable)
    assert hooks.photo_run_hooks()["upscale"] == "off"
