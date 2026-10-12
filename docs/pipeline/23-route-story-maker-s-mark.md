### Route Story maker's mark (`hud.draw_watermark`, `assembly._save_frame`)

Every composited frame is saved through one choke point, `assembly._save_frame(image, frames,
number, *, watermark=None)`; when `watermark` is truthy it burns `hud.draw_watermark` (bottom-
right, ~1.6% of frame height, white at ~55% opacity with a dark stroke, fit inside
`hud._TITLE_SAFE_WIDTH_FRACTION`) onto the frame before it hits disk. `render_route_story_film`
takes a `watermark: str | None = hud.DEFAULT_WATERMARK_TEXT` parameter and threads it into every
frame-producing call -- the opening loop, both bookend passes, every map leg, every photo, and
the recap loop. The CLI exposes `--watermark TEXT` (default `hud.DEFAULT_WATERMARK_TEXT`) and
`--no-watermark`.

A settle/arrival hold (`assembly._append_hold`) is the one exception: it replays an
already-saved image (the leg's own last frame, see `docs/pipeline/22-route-story-settle-holds.md`), which was already marked when the leg
rendered it, so it is never asked to burn the mark a second time.

The source-video part of a film (the placed clip itself, encoded straight by `ffmpeg` in
`assembly._encode_source_video` rather than through `_save_frame`) carries the same mark: it
takes `watermark=`, draws `hud.draw_watermark` once onto a transparent frame-sized card, and
overlays that card on every frame of the letterboxed clip (`filter_complex` `overlay=0:0`). The
card is a scratch PNG beside the clip, deleted when the encode ends.
