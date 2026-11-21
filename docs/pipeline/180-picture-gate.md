## 180 - Picture gate (paired frames)

The picture gate fails a delivered file that a rendition step (sharpen, denoise, declared
upscale) made softer or added ringing to. It runs only on a day where such a step ran:
`quality.day_picture_changed` reads the day master's declared upscale (master record, else the
`.upscale.json` sidecar) and the day's colour cache / picture-steps record. Other days carry a
`skipped` verdict.

### Paired frames

Sharpness (Laplacian variance) depends on the scene: on one drive single frames ranged from 90 to
2614. So each sampled delivered frame is compared with **the frame it was made from**, never with
a drive-wide source figure. `quality.sample_paired_picture` takes the gate's 25 evenly spaced
seconds of the delivered file and, for each, the matching second of its reference; a pair is used
only when both frames decode. The verdict (`longform_validator.paired_picture_gate`) compares the
two medians: delivered sharpness below the reference's less 25% fails, and so does ringing above
the reference's plus 25%.

| Delivered file | Reference frame | Checked |
|---|---|---|
| Day master | the same second of the day's episode concats, laid back to back (`quality.timeline_position`) | right after the master encode; a missing concat skips it with a warning |
| Long-form | the same second of its pre-text input (`lf_XX_with_montage.mp4`) | after the text burn, before photo inserts shift the timeline |
| Short | the master at the Short's start plus the same second, through the Short's own crop (`build_crop_filter`) | right after the Short renders; a tracked reframe is compared with the centre crop of the same seconds |

### Contrast-matched reference (day master only)

Laplacian variance scales with contrast squared, so a deliberate tone grade (shadow lift,
highlight pull) would read as softness. Only the day master is graded (the LF is an OST burn and
the Short a crop plus OST, neither changes tone), so only `_gate_master_picture` passes
`contrast_match=True` to `paired_picture_gate` / `quality.sample_paired_picture`. The LF and Short
gates are judged exactly as before, message included.

With the flag on, each pair's reference sharpness is put at the delivered frame's contrast before
the medians are taken: `reference_sharpness x ratio`, where
`ratio = max(var(delivered luma) / var(reference luma), CONTRAST_MATCH_FLOOR)`. The variance is
measured on an `INTER_AREA` downscale by 8 of the bounded luma `sharpness_score` measures (full
resolution under 8 px a side), so a smear, which also lowers full-resolution variance, cannot
excuse itself. A flat frame (zero variance) leaves the pair unadjusted.

- **Floor 0.55** (`quality.CONTRAST_MATCH_FLOOR`). The strongest grade the colour filters can build
  (lift 20, pull 15, saturation 1.25, gamma 1.3) measures a variance ratio of 0.61-0.66 on real
  photos; the incident grade is about 0.84. A delivered frame crushed toward flat grey or black
  would otherwise shrink its own reference to nothing and pass, so the ratio never goes below the
  floor and such a frame fails on the clamped reference. The message then ends
  `; contrast collapsed on N pairs`.
- **No ceiling.** A grade that raises contrast raises the reference, so added contrast cannot hide
  softening.
- **Ringing.** Ringing scales linearly with contrast, so the reference ringing is multiplied by the
  square root of the same ratio. A contrast-lowering grade cannot excuse halos from a sharpen step.

The verdict message ends `(reference at delivered contrast ×0.84)` (the median ratio) so an operator can see the
adjustment. The 25% tolerance, `sharpness_score`, the phase A source gate and the unpaired gate are unchanged.

Courtmacsherry (2026-11-20): a master whose grade lowered luma spread 8.6% (variance about ×0.84)
read 240.0 against 333.7 (`240.0 < 250.3`) and failed on every retry; contrast-matched, the median
ratio is about 0.86, inside the tolerance.

A failed check stops the day with `failed the picture gate: <reason>`.

### The record Phase D reads

Each long-form and Short writes `<file>.picture_gate.json` (`passed`, `message`, `size`) when it
renders. Phase D (`phase_d_gates._default_picture_check`) and the Shorts validator read that
record with `recorded_picture_gate`; they never re-measure, because by then there are no paired
source frames. On a changed day a final fails when:

- the record is missing: it was rendered before this check existed, so re-render it;
- the record's `size` is not the file's: the file was rewritten after it was checked.

The long-form fresh-skip re-burns a final that has no current record. The trail-film opening
swap stream-copies the body of a checked long-form and carries its record over to the new file.

Mizen (2026-11-16): a faithful day master read 335.8 against 405.8 on paired frames (inside the
tolerance) but failed at 335.8 < 448.1 against a source figure of 597.5 taken on other frames.
Superlong parts are outside this gate.
