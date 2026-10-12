"""``gomoveshift comments triage`` — cue/place flagging + markdown report."""

from __future__ import annotations

import io
import json

from gomoveshift.cli.handlers.comments import run_comments_triage
from gomoveshift.pipeline.comments import build_triage_report


def _comment(video_id, author, text, published_at):
    return {
        "video_id": video_id,
        "author": author,
        "text": text,
        "published_at": published_at,
        "platform": "youtube",
    }


def test_build_triage_report_flags_by_cue_and_place_not_in_title():
    comments = [
        _comment("vid1", "A Viewer", "actually that's Mockbridge not Mockhaven", "2026-08-20"),
        _comment("vid1", "B Viewer", "beautiful drive!", "2026-08-21"),
        _comment("vid2", "C Viewer", "We drove through Fable Bay", "2026-08-22"),
    ]
    metadata = {
        "vid1": {"title": "Mockhaven Coastal Drive", "description": ""},
        "vid2": {"title": "West Portfable Byway", "description": ""},
    }
    report = build_triage_report(comments, metadata)

    assert "vid1" in report
    assert "Mockhaven Coastal Drive" in report
    assert "cue:actually" in report
    assert "beautiful drive!" not in report
    assert "vid2" in report
    assert "place:Fable Bay" in report


def test_run_comments_triage_writes_markdown_from_jsonl(tmp_path):
    in_path = tmp_path / "comments.jsonl"
    rows = [
        _comment("vid1", "A Viewer", "actually that's Mockbridge not Mockhaven", "2026-08-20"),
        _comment("vid1", "B Viewer", "beautiful drive!", "2026-08-21"),
    ]
    in_path.write_text("\n".join(json.dumps(c) for c in rows) + "\n", encoding="utf-8")
    out_path = tmp_path / "triage.md"

    rc = run_comments_triage(in_path=in_path, out_path=out_path)

    assert rc == 0
    report = out_path.read_text(encoding="utf-8")
    assert "cue:actually" in report
    assert "beautiful drive!" not in report


def _write_one_comment(tmp_path):
    in_path = tmp_path / "comments.jsonl"
    row = _comment("vid1", "A Viewer", "actually that's Mockbridge", "2026-08-20")
    in_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return in_path


def test_triage_metadata_sources_unset_are_named_and_skipped(tmp_path, monkeypatch):
    monkeypatch.delenv("GMS_COMMENTS_SEO_LEDGER", raising=False)
    monkeypatch.delenv("GMS_COMMENTS_VIDEOS_YAML", raising=False)
    in_path = _write_one_comment(tmp_path)
    err = io.StringIO()

    rc = run_comments_triage(
        in_path=in_path, out_path=tmp_path / "triage.md", out=io.StringIO(), err=err
    )

    assert rc == 0
    assert "GMS_COMMENTS_SEO_LEDGER is not set" in err.getvalue()
    assert "GMS_COMMENTS_VIDEOS_YAML is not set" in err.getvalue()
    assert "cue:actually" in (tmp_path / "triage.md").read_text(encoding="utf-8")


def test_triage_metadata_sources_set_are_read_from_env(tmp_path, monkeypatch):
    ledger = tmp_path / "events.jsonl"
    ledger.write_text(
        json.dumps({"video_id": "vid1", "title": "Mockbridge Harbour", "description": ""}) + "\n",
        encoding="utf-8",
    )
    videos_yaml = tmp_path / "videos.yaml"
    videos_yaml.write_text("[]\n", encoding="utf-8")
    monkeypatch.setenv("GMS_COMMENTS_SEO_LEDGER", str(ledger))
    monkeypatch.setenv("GMS_COMMENTS_VIDEOS_YAML", str(videos_yaml))
    in_path = tmp_path / "comments.jsonl"
    row = _comment("vid1", "A Viewer", "We stopped at Mockbridge Harbour", "2026-08-20")
    in_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    err = io.StringIO()

    rc = run_comments_triage(
        in_path=in_path, out_path=tmp_path / "triage.md", out=io.StringIO(), err=err
    )

    assert rc == 0
    assert "is not set" not in err.getvalue()
    # The title from the configured ledger contains the place, so it is not flagged.
    assert "place:Mockbridge Harbour" not in (tmp_path / "triage.md").read_text(encoding="utf-8")
