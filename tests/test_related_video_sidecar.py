"""Tests for the related-video Studio worklist operator sidecar.

YouTube's Shorts "Related video" field is the only clickable
Short→long-form bridge and has NO public API (Studio desktop only).
Mirroring the end-screen sidecar (``endscreen_sidecar.py``), the
pipeline emits a doc-only worklist: one operator line per short into
the same ``operator_checklist.txt`` artefact, keyed on the short's
TITLE (shorts have no YouTube id pipeline-side — Metricool posts them).
"""

from __future__ import annotations

from types import SimpleNamespace

from gomoveshift.pipeline.related_video_sidecar import (
    build_related_video_worklist_lines,
    emit_related_video_worklist,
    format_related_video_worklist_line,
)


def _post(short_index: int, title: str) -> SimpleNamespace:
    return SimpleNamespace(short_index=short_index, youtube_title=title)


# ─── format_related_video_worklist_line ─────────────────────────────────


def test_format_line_shape():
    line = format_related_video_worklist_line(
        short_title="POV: Driving Through County Kerry",
        parent_video_id="abc123",
    )
    assert line == ('RELATED VIDEO: "POV: Driving Through County Kerry" → https://youtu.be/abc123')


# ─── build_related_video_worklist_lines ─────────────────────────────────


def test_build_lines_one_per_short_with_parent_id():
    lines = build_related_video_worklist_lines(
        social_posts_per_lf={1: [_post(10, "Short A"), _post(11, "Short B")]},
        youtube_uploads={1: {"video_id": "vid-1"}},
    )
    assert lines == [
        'RELATED VIDEO: "Short A" → https://youtu.be/vid-1',
        'RELATED VIDEO: "Short B" → https://youtu.be/vid-1',
    ]


def test_build_lines_skips_missing_and_empty_parent_id():
    """LF 2 has no upload record, LF 3 has an empty video_id — both skipped."""
    lines = build_related_video_worklist_lines(
        social_posts_per_lf={
            1: [_post(10, "Kept")],
            2: [_post(20, "No record")],
            3: [_post(30, "Empty id")],
        },
        youtube_uploads={1: {"video_id": "vid-1"}, 3: {"video_id": ""}},
    )
    assert lines == ['RELATED VIDEO: "Kept" → https://youtu.be/vid-1']


def test_build_lines_empty_inputs():
    assert build_related_video_worklist_lines(social_posts_per_lf={}, youtube_uploads={}) == []


# ─── emit_related_video_worklist ────────────────────────────────────────


def test_emit_appends_to_operator_checklist(tmp_path):
    """Lines land in the SAME operator_checklist.txt the end-screen sidecar
    writes — pre-existing content is preserved (append, not truncate)."""
    checklist = tmp_path / "operator_checklist.txt"
    checklist.write_text("END SCREEN: open https://studio.youtube.com/video/x/editor\n")

    n = emit_related_video_worklist(
        output_base=tmp_path,
        social_posts_per_lf={1: [_post(10, "Short A")]},
        youtube_uploads={1: {"video_id": "vid-1"}},
    )
    assert n == 1
    content = checklist.read_text()
    assert content.startswith("END SCREEN: open https://studio.youtube.com/video/x/editor\n")
    assert 'RELATED VIDEO: "Short A" → https://youtu.be/vid-1\n' in content


def test_emit_no_lines_writes_nothing(tmp_path):
    """No short has a parent id (skip-upload audit run) → no file created."""
    n = emit_related_video_worklist(
        output_base=tmp_path,
        social_posts_per_lf={1: [_post(10, "Short A")]},
        youtube_uploads={},
    )
    assert n == 0
    assert not (tmp_path / "operator_checklist.txt").exists()


def test_emit_never_raises_on_write_failure(tmp_path, caplog):
    """Same error posture as the end-screen sidecar: a file-write failure is
    logged, never raised — emitting the worklist must never fail the batch."""
    # Make the checklist path unwritable by occupying it with a directory.
    (tmp_path / "operator_checklist.txt").mkdir()
    with caplog.at_level("WARNING"):
        n = emit_related_video_worklist(
            output_base=tmp_path,
            social_posts_per_lf={1: [_post(10, "Short A")]},
            youtube_uploads={1: {"video_id": "vid-1"}},
        )
    assert n == 0
    assert any("related-video worklist" in r.getMessage() for r in caplog.records)


def test_emit_is_idempotent_across_repeated_runs(tmp_path):
    """A resumed/re-run Phase D calling emit twice with the SAME inputs must
    not duplicate operator instructions — each line appears exactly once."""
    kwargs = dict(
        output_base=tmp_path,
        social_posts_per_lf={1: [_post(10, "Short A"), _post(11, "Short B")]},
        youtube_uploads={1: {"video_id": "vid-1"}},
    )
    n1 = emit_related_video_worklist(**kwargs)
    n2 = emit_related_video_worklist(**kwargs)
    assert n1 == 2
    assert n2 == 0
    content = (tmp_path / "operator_checklist.txt").read_text()
    assert content.count('RELATED VIDEO: "Short A" → https://youtu.be/vid-1') == 1
    assert content.count('RELATED VIDEO: "Short B" → https://youtu.be/vid-1') == 1


def test_emit_appends_new_entries_on_later_run(tmp_path):
    """A second run whose plan grew (a genuinely NEW short) must still append
    the new line — dedup must not degrade into "never emit again"."""
    n1 = emit_related_video_worklist(
        output_base=tmp_path,
        social_posts_per_lf={1: [_post(10, "Short A")]},
        youtube_uploads={1: {"video_id": "vid-1"}},
    )
    n2 = emit_related_video_worklist(
        output_base=tmp_path,
        social_posts_per_lf={1: [_post(10, "Short A"), _post(11, "Short B")]},
        youtube_uploads={1: {"video_id": "vid-1"}},
    )
    assert n1 == 1
    assert n2 == 1
    content = (tmp_path / "operator_checklist.txt").read_text()
    assert content.count('RELATED VIDEO: "Short A"') == 1
    assert content.count('RELATED VIDEO: "Short B"') == 1


def test_emit_never_raises_on_malformed_post(tmp_path, caplog):
    """A post missing ``youtube_title`` logs a WARNING instead of raising."""
    with caplog.at_level("WARNING"):
        n = emit_related_video_worklist(
            output_base=tmp_path,
            social_posts_per_lf={1: [SimpleNamespace(short_index=10)]},
            youtube_uploads={1: {"video_id": "vid-1"}},
        )
    assert n == 0
    assert any("related-video worklist" in r.getMessage() for r in caplog.records)
