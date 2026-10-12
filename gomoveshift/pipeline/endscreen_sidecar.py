"""End-screen Studio-import operator sidecar.

There is no API to set end screens, so this module only emits a checklist:

- YouTube Data API v3 has NO end-screen resource.
- YouTube Partner / Content ID API has NO end-screen resource.
- Studio's end-screen editor is not exposed via any documented public API.
- The community ``adasq/youtube-studio`` reverse-engineered library is
  unsuitable for an automated batch pipeline (cookie auth + daily
  SESSION_TOKEN refresh + unmaintained since Feb 2024).

The module therefore ships a doc-only checklist: a single operator-readable
line per uploaded LF pointing at Studio Editor's Endscreen tab. The
operator clicks "Import from video {ref}" or "Apply template" once per
video. There is no API call.

The checklist is consumed by:

1. The module logger at INFO (so each upload's stdout names the action).
2. ``<output_base>/operator_checklist.txt`` — a single artefact the
   operator opens after a batch run. Append-only; created on first
   upload of a run.

``YouTubeUploader.upload_video_with_playlist`` calls
:func:`format_endscreen_checklist_line` after every successful upload
and writes the result to both surfaces. Total by construction: there
is no upload path through the uploader that returns a video id without
emitting the line, mirroring the playlist-assignment contract.
"""

from __future__ import annotations

from typing import Optional


def format_endscreen_checklist_line(
    *,
    video_id: str,
    reference_video_id: Optional[str],
) -> str:
    """Return the operator instruction line for one uploaded video.

    Two shapes:

    - With a non-empty ``reference_video_id``: instruct the operator to
      use Studio's "Import from video" action against the reference
      video, which is the closest YouTube ships to a per-channel
      end-screen template.
    - Without (``None`` or empty string): fall back to "Apply template",
      i.e. operator picks one of YouTube's built-in layouts manually.

    Empty string is treated identically to ``None`` because the empty
    string is not a valid YouTube video id (and can leak in from a
    config block whose key was set but left blank).
    """
    base = f"END SCREEN: open https://studio.youtube.com/video/{video_id}/editor → Endscreen"
    if reference_video_id:
        return f"{base} → Import from video {reference_video_id}"
    return f"{base} → Apply template"
