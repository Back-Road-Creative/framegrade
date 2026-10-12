"""Tests for end-screen Studio-import sidecar formatter.

The YouTube Data API v3 has NO end-screen resource and the partner API has
none either. The sidecar ships a doc-only operator checklist line; no API
call. ``format_endscreen_checklist_line`` is a pure formatter.
"""

from __future__ import annotations

from gomoveshift.pipeline.endscreen_sidecar import format_endscreen_checklist_line


def test_with_reference_video_id_emits_import_from_video_form():
    line = format_endscreen_checklist_line(
        video_id="abc123",
        reference_video_id="ref456",
    )
    assert line == (
        "END SCREEN: open https://studio.youtube.com/video/abc123/editor "
        "→ Endscreen → Import from video ref456"
    )


def test_without_reference_video_id_emits_apply_template_form():
    line = format_endscreen_checklist_line(
        video_id="abc123",
        reference_video_id=None,
    )
    assert line == (
        "END SCREEN: open https://studio.youtube.com/video/abc123/editor "
        "→ Endscreen → Apply template"
    )


def test_empty_string_reference_treated_as_none():
    """Empty string is not a valid reference video id — same shape as None."""
    line = format_endscreen_checklist_line(
        video_id="abc123",
        reference_video_id="",
    )
    assert line == (
        "END SCREEN: open https://studio.youtube.com/video/abc123/editor "
        "→ Endscreen → Apply template"
    )
