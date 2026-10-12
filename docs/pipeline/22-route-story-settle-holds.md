### Route Story settle holds (`assembly._append_hold`)

`assembly._append_hold(frames, frame, image, fps, seconds)` writes one already-rendered image
for a duration and returns the next frame number. One render call is reused for every frame of
the hold: re-rendering pixels that never change costs real minutes under the scene renderer. It
always writes at least one frame, because a settle rounded away to nothing at a low fps would
silently put the cut back exactly where it was.

Both of the film's holds go through it, and both hold on the image `assembly._append_map_leg`
itself returns as `(next_frame_number, last_rendered_image)` -- the exact last frame the leg's
own renderer produced, never a second, separate render call at the same route-clock time. A
scene renderer rate-limits the camera's bearing across a leg's own frames but does not carry
that state into an unrelated `render()` call; a second call at the leg's ending time used to
render the raw, un-rate-limited bearing, which the viewer saw as a camera snap immediately
before the cut and a blip immediately after the hold. A hold rendered from the flat Pillow map
instead pops to unrelated imagery for the length of the hold, right before the media plays.

- **`MAP_SETTLE_SECONDS`** — the map stops on a mapped still's own route moment before the still
  plays. Without it the camera is still moving when the picture arrives, so the still reads as
  an interruption rather than as the thing the ride just arrived at.
- **`VIDEO_ARRIVAL_HOLD_SECONDS`** — the longer hold a placed video gets, with the
  `VIDEO ARRIVAL` badge drawn onto the settled frame first. A placed video announces itself and
  needs a beat to be read; a still needs only enough stillness to land on, so equal lengths
  would make every photo feel like a video that failed to play.

Neither hold is map-leg time: `plan_map_legs`'s compression covers the legs, and a settle adds
its own seconds on top, exactly as the video hold always did.
