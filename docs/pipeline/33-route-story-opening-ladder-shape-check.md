## Route Story opening ladder shape check (`gomoveshift/pipeline/route_story/frame_qa.py`)

`check_opening_frame`'s three floors are absolute, and an earlier render cleared all of them
while being rejected as "very dark and fuzzy when zooming in". Measured on that film's own opening
frames, sharpness by ladder fraction ran 2211, 4571, **1305**, 4778, **1237**, 8165, 11899, 12910,
8585: two rungs about a tenth as sharp as their neighbours, where the ladder hands over to a
coarser raster. The worst of them clears the 50.0 blur floor by 25x, and a floor raised far enough
to catch it would refuse legitimately soft imagery in other bundles.

- **`check_opening_ladder` judges shape, not level.** Each rung's sharpness is compared to the
  ladder's OWN median; a rung below `min_opening_ladder_sharpness_ratio` (0.35) of that median is
  reported by index. Self-calibrating, so no per-bundle constant, and it stays valid for a bundle
  whose absolute sharpness differs.
- **Calibration, from a measured ladder** (median 4778.3): the two defective rungs sit at
  ratio 0.26 and 0.27, the lowest healthy rung at 0.46 (f=0.0, the deliberately wide establishing
  shot). 0.35 sits between them. The margin above the healthy side is 0.11, not a wide one — if a
  future bundle's establishing shot trips this, the fix is to re-measure that ladder and move the
  ratio, not to delete the check.
- **The two checks answer different questions.** A ladder where every rung is equally soft has no
  shape to fault and is left to `check_opening_frame`'s absolute floor; a ladder with a sharp dip is
  invisible to that floor. Fewer than three rungs is not a shape and is not judged.
- **`OPENING_LADDER_FRACTIONS` (`film.py`) walks 11 fractions, not 5.** The old
  `0/0.25/0.5/0.75/1.0` walk could straddle a soft band entirely; the walk is denser toward the end,
  where handovers cluster. These are Pillow frames, so the extra samples cost preflight seconds, not
  render time.
