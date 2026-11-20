## Long-form title burn: opening-segment-only re-encode (dropped)

Unit A4 proposed re-encoding only an opening segment of the long-form (LF) for the title burn, then
stream-copying the rest. It is **dropped**: the burn is not a title-only overlay, so a copied body would
change the published output. No code changed.

### Measured reason

`ost_applier.apply_ost(layout="regular_lf")` (`gomoveshift/pipeline/ost_applier.py`) draws and rewrites
more than the opening title:

- **Persistent URL watermark on every frame.** `LAYOUTS["regular_lf"]["show_persistent_url"]` is `True`;
  `_build_drawtext_filter` emits that `drawtext` with no `enable=` gate (top-right, full video).
- **Closing website CTA** in the last 6 s (`show_website`).
- **Caption cards / superlong lower-thirds** (`lower_thirds`) and the optional **map inset** span the LF
  body (`day_pipeline.py`, the `apply_ost` call sites).
- **Audio is re-filtered over the whole file.** `_audio_attenuate_filter` applies an attenuate-only gain
  (the near-silent policy). A stream-copied body keeps the source level. Measured on a 10 s synthetic
  (CPU libx264/aac, sine at -21.1 dB mean): the burned file reads -46.1 dB mean; a copied body would
  stay at -21.1 dB, which breaks the policy.
- On the same fixture every one of the 180 video frames after the title window differs between the
  source and the real `regular_lf` filter, because the watermark is drawn on all of them.

So the body cannot be copied: it would lose the watermark and the attenuation. Making the segment path
valid would mean moving the watermark, CTA, cards and audio gain into separate passes, which is a
different change from A4.

### What already exists

`pipeline/trail_opening.py` already burns only an opening (`website_cta=False`) and swaps it into an
existing LF by keyframe-aligned copy (`keyframe_span`, `copy_range_argv`, `concat_copy_argv`). That path
serves the trail-film opening, not the LF title.

### Re-open conditions

Re-open only if the persistent URL watermark and the audio attenuation are decided to move off the LF
body (for example the watermark baked into the day master and the gain applied there). Until then the
full re-encode stays.
