# The final-text byte-bind is on

Both binds ship on: `youtube_uploader.FINAL_TEXT_BIND_ON` (the publication
gate) and `final_text_record.BIND_DESCRIPTION_TAGS`. There is no environment
variable or flag that turns them off. A YouTube insert now ships only text the
Waypost review covered, byte for byte, on all three surfaces.

## Effect on inserts

With the bind on, a long-form, Short or superlong input inserts exactly once with
the reviewed text, and the publication gate's blanket hold becomes the refusal
set for text the review does not cover (insert count 0). Tests:
`tests/test_w9c2_lf_bind_on.py`, `tests/test_w9s_shorts_review.py`,
`tests/test_w9_final_text_publication_gate.py`.

## What is sent

- **Long form / superlong:** `final_text_record.reviewed_text` returns
  `compose_description` / `compose_tags` of `metadata["final_text_description"]`.
  `_upload_lf_to_youtube` sends that, not the GMS-built description and tags.
  An LF with no entries has no review and is refused (`route_record_missing`).
- **Short:** Waypost reviews a description only as filler words plus cited
  places, so a Short sends one place line, `Drive through <clip place>`; the
  generated prose and its parent link are not sent. Tags must each be an exact
  Waypost tag template or a name of one of the clip's own GPS places.
  Any other tag refuses the Short with `tag_unreviewable:<tag>`. There is no
  silent drop and no bypass.
- A Short with no label and no regions skips review (unchanged).
- `social_generator.BASE_TAGS` drops `driving` and `travel`: neither has a
  Waypost tag template.

## Road and POI tags (open, not a Waypost kind request)

Shorts carrying road or POI tags are refused after this bind. Waypost already
accepts a place tag backed by `location`, `region` or `road` evidence
(`final_text_review.py`, `_PLACE_KINDS`), so a road tag needs no Waypost change
and no pin bump: it needs GMS-side evidence ids. A Short's record carries only
`sampled_regions`, so no per-clip road evidence exists to cite today. A POI has
no evidence kind anywhere in `seo_manager`. Until GMS records per-clip road
(and, if wanted, POI) evidence at plan time, the Short bind stays
publish-blocked behind the existing holds.

## Known gaps

The superlong chapter block is not sent; playlist and cadence lines are not
built from config; Phase D chapters refuse when a title is not an evidenced place.

## Release note

Shorts lose the parent-LF link and stay effectively publish-blocked until
GMS records per-clip road evidence (POI has no Waypost kind); release notes must not claim Shorts publish.
