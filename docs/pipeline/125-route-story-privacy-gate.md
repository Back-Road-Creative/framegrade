## Route Story privacy gate (`gomoveshift/pipeline/route_story/privacy.py`)<a id="route-story-privacy-gate"></a>

### What

`apply_privacy_gate(photos, *, blur, work_dir)` runs before any Route Story still is composited
or exported. For every `PhotoMoment` it:

1. Checks the photos privacy withhold gate (`photos.face_filter.should_withhold_photo`) — an
   unreadable/unblurrable photo is dropped and named in the receipt, never composited.
2. Blurs faces/plates/plate-corroborated vehicle cabins/whole persons onto an in-memory frame
   (`photos.face_filter.blur_faces_for_publish`) and writes the result to a working copy under
   `work_dir/privacy_gated_stills/<source-name>` — never overwriting the source.
3. Re-saves that working copy with no EXIF/XMP and asserts no GPS tag survived
   (`photos.resizer._assert_no_gps`), the still analogue of the photos-export GPS gate.

It returns a `PrivacyResult(photos, withheld, attestation)` whose `photos` carries each kept
`PhotoMoment` with `.source` swapped to the gated working copy. Every downstream composite/export
site (mapped stills, bookend stills, graded exports) reads `result.photos[i].source` — never the
original path — so a photo the film ships is, by construction, the blurred/stripped copy.

### Why

The service contract requires faces, plates,
vehicle occupants and whole persons blurred, and GPS stripped, before any photo publishes. The
route-story film composites operator stills straight into a publishable mp4; before this gate it
had neither guard, reusing only extract-side EXIF code (`pipeline/photos/exif_gps.py`).

### Flag

`--skip-blur` on the `film.py` CLI (mirrors the photos-CLI flag verbatim). Default: blur ON.
Under `--skip-blur`, nothing is asked of the detectors and `should_withhold_photo` falls back to
detect-and-withhold: a photo whose privacy regions can't be blurred is dropped, never shipped
unblurred, but the ones that *are* readable ship exactly as shot — this is a diagnostic escape,
never a default.

### Fail-closed

A default run whose detector session cannot be built/run raises `FaceBlurError` from inside
`_gate_one` before any frame is composited — the message names `--skip-blur` as the diagnostic
escape. There is no path from "detector unavailable" to "ship unblurred" without the operator
explicitly passing `--skip-blur`.

### Attestation → publisher hold

`attestation` is built from `photos.face_filter.attestable_privacy_classes(blur)` and covers
exactly `publishing.dispatch.PRIVACY_REQUIRED_CLASSES` (imported, never copied) —
`{"classes": {cls: {"state": "verified"|"uncovered"|"inconclusive", "reason": ...}}, "blur": bool,
"detector": [...]}`. It is written to `receipt["privacy"]` (plus `receipt["privacy"]["withheld"]`,
the dropped photo names) and, when a caller passes it through, into the `publish.json` sidecar
next to `duration_s`/`chapters` (`encode.write_publish_json(..., privacy=result.attestation)`).

`publishing.dispatch.privacy_hold_reason(receipt["privacy"])` reads this directly with no
translation: `None` for a default run whose detectors resolved, a reason string under
`--skip-blur` (every class `inconclusive` — nothing was asked, so nothing is known).
