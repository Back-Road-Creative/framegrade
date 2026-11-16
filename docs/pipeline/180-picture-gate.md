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
