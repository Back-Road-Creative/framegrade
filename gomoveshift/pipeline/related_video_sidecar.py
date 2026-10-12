"""Related-video Studio worklist operator sidecar.

YouTube's Shorts "Related video" field is the ONLY clickable
Short→long-form bridge and it has NO public API — it is settable in
Studio (desktop) only. Mirroring the end-screen sidecar
(:mod:`gomoveshift.pipeline.endscreen_sidecar`), the pipeline therefore
ships a doc-only worklist: one operator-readable line per short naming
the short's title and its parent LF's watch URL, appended to the same
``<output_base>/operator_checklist.txt`` artefact the end-screen
sidecar writes. The operator batch-processes the lines in Studio after
a run. There is no API call.

Shorts have no YouTube video id pipeline-side (Metricool posts them),
so lines key on the short's TITLE. A short whose parent LF has no
YouTube video id (skip-upload audit run, upload disabled) is skipped —
there is nothing to relate it to.

Emission is fail-open by construction: :func:`emit_related_video_worklist`
never raises. A formatting or file-write failure logs a WARNING and the
upload/CSV batch continues — the worklist is recoverable, uploads are
not. Every line is also logged at INFO, the durable record when the
file write fails (same posture as the end-screen sidecar).

A resumed or re-run Phase D recomputes the same worklist lines, so the
emit tracks which lines it has already appended in a sidecar dedup
state file (``related_video_worklist_state.json``, written via
:func:`~gomoveshift.pipeline._atomic_write.atomic_write_json`) and
skips them on a later call. Dedup identity is the line text itself,
tracked independently of ``operator_checklist.txt`` — so if the
operator ticks off or deletes a line after handling it in Studio, a
later run with the SAME plan will NOT re-add it (the state file, not
the checklist's current contents, is the source of truth). A genuinely
new line (a short/parent pair not seen before) is still appended. A
missing or corrupt state file degrades to "treat nothing as already
emitted" — worst case a duplicate operator line, never a silently
dropped instruction.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from gomoveshift.pipeline._atomic_write import atomic_write_json

logger = logging.getLogger(__name__)


def format_related_video_worklist_line(*, short_title: str, parent_video_id: str) -> str:
    """Return the operator worklist line for one short (pure formatter)."""
    return f'RELATED VIDEO: "{short_title}" → https://youtu.be/{parent_video_id}'


def build_related_video_worklist_lines(
    *,
    social_posts_per_lf: dict[int, list],
    youtube_uploads: dict[int, dict],
) -> list[str]:
    """One line per short whose parent LF has a YouTube video id.

    Derives short→parent exactly like Phase D's consolidated-CSV
    ``parent_ids`` map: every short under LF *i* points at
    ``youtube_uploads[i]["video_id"]``. A missing or empty parent id
    skips the short — there is no URL to relate it to.
    """
    lines: list[str] = []
    for lf_index in sorted(social_posts_per_lf):
        vid = (youtube_uploads.get(lf_index) or {}).get("video_id", "")
        if not vid:
            continue
        for post in social_posts_per_lf[lf_index]:
            lines.append(
                format_related_video_worklist_line(
                    short_title=post.youtube_title, parent_video_id=vid
                )
            )
    return lines


def _load_already_emitted(state_path: Path) -> set[str]:
    """Read the dedup state file. Missing/corrupt → empty set (emit anyway).

    A duplicate operator line is an annoyance; treating a real new
    instruction as "already emitted" because of an unreadable state
    file would be a real miss, so any read failure degrades to "nothing
    was emitted before" rather than raising.
    """
    try:
        raw = json.loads(state_path.read_text(encoding="utf-8"))
        return set(raw) if isinstance(raw, list) else set()
    except Exception:  # noqa: BLE001 — corrupt state degrades, never raises
        return set()


def emit_related_video_worklist(
    *,
    output_base: Path,
    social_posts_per_lf: dict[int, list],
    youtube_uploads: dict[int, dict],
) -> int:
    """Append NEW worklist lines to ``<output_base>/operator_checklist.txt``.

    Idempotent across repeated calls with the same inputs (a resumed or
    re-run Phase D): a line already appended by a prior call is skipped,
    tracked via a dedup state sidecar (module docstring). Returns the
    number of lines newly appended (0 when no short has a parent id, or
    every line was already emitted). Never raises: emitting the
    worklist must never fail the batch.
    """
    # Same artefact the end-screen sidecar appends to.
    checklist_path = output_base / "operator_checklist.txt"
    state_path = output_base / "related_video_worklist_state.json"
    try:
        lines = build_related_video_worklist_lines(
            social_posts_per_lf=social_posts_per_lf,
            youtube_uploads=youtube_uploads,
        )
        for line in lines:
            logger.info(line)
        if not lines:
            return 0
        already_emitted = _load_already_emitted(state_path)
        new_lines = [line for line in lines if line not in already_emitted]
        if not new_lines:
            return 0
        checklist_path.parent.mkdir(parents=True, exist_ok=True)
        with checklist_path.open("a", encoding="utf-8") as fh:
            fh.writelines(line + "\n" for line in new_lines)
        atomic_write_json(state_path, sorted(already_emitted | set(new_lines)))
        return len(new_lines)
    except Exception as exc:  # noqa: BLE001 — never break the pipeline
        logger.warning(
            "related-video worklist emit failed (path=%s): %s — the INFO "
            "log lines above (if any) are the durable record; batch continues",
            checklist_path,
            exc,
        )
        return 0
