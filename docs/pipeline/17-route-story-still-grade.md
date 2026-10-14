## Route Story still grade (`gomoveshift/pipeline/route_story/grade.py`)

Every still `hud._load_still` returns is graded before it reaches a photo hold, whatever the
source (`developed` or `camera`): mountain views shot through haze read flat and blue-grey, and a
saturation boost alone cannot restore lost contrast, so the pipeline is (1) adaptive haze removal,
(2) gentle exposure, (3) hue-preserving vibrance, (4) a light local-contrast curve, (5) an ordered
dither before 8-bit quantisation so a lifted sky does not band. `hud._GRADE_STILLS` (module
constant, not a CLI flag) disables it for a test that needs exact ungraded pixels.

`grade._remove_haze` is a dark-channel-prior haze removal (He/Ko), run on a downscaled copy for
speed: atmospheric light is the mean colour (per channel, not greyed) of the brightest 0.1%
dark-channel pixels, the transmission map is refined with a bilateral filter and upsampled, and the
strength (`omega`) is scaled by the image's own measured dark-channel mean, capped, so a clear
photo barely moves and a flat, textureless test card (near-zero variance) is skipped outright — a
uniform image carries no depth cue and dark-channel filtering on it is noise, not haze.

The dark-channel value alone cannot distinguish a distant hazy veil from a dark, saturated
foreground object — forest shadow and foliage both score a low dark channel, and correcting them as
if they were haze crushes the near ground of a ridgeline frame to near-black with a blue
cast. So the transmission map is pulled back
toward 1 (no change, via `grade._haze_confidence`) wherever a pixel does not ALSO look like haze on
its own: haze desaturates and it never darkens a scene (it lifts luma toward the bright atmosphere
colour), so the correction strength is gated by a saturation confidence
(`_HAZE_SAT_LOW`/`_HAZE_SAT_HIGH`) and a luma confidence (`_HAZE_LUMA_LOW`/`_HAZE_LUMA_HIGH`),
multiplied together — a saturated green or a dark shadow pixel gets little to none of the
correction regardless of what its dark-channel value says.

A per-pixel dark-channel transmission map alone, even where fully trusted, only *exposes* a hazy
region back toward its true brightness — it does not restore the fine local contrast the veil
erased, because a lower-albedo patch of texture inside a genuinely hazy region (a shaded fold in a
distant ridge) reads a lower dark-channel value than a brighter patch right next to it and so gets
systematically LESS correction, not more: the recovery
formula alone can leave distant-ridge luma std *lower* than before correction. `grade._apply_haze_clarity` restores that texture directly: a
large-radius unsharp pass (`_CLARITY_SIGMA_FRACTION`) added back only where `_haze_confidence` says
a pixel is genuine haze and scaled by how hazy the whole frame is (`_CLARITY_AMOUNT` × `strength`),
so a clear frame or a near-ground pixel gets ~0 boost. Its own highlight-clipping is tapered down
(never re-brightened afterward) if it pushes more of the frame to white than
`_CLARITY_MAX_CLIP_FRACTION` — looser than exposure's own clip budget, since a hazy ridge
legitimately gaining a few bright highlight pixels once cleared is the point of the pass, not a
fault.

**The clarity amount and the dehaze ceiling are deliberately modest — natural beats maximal.** A
camera preview is an 8-bit JPEG. A strong setting can meet every contrast and haze number
while looking wrong: white halos along each ridgeline, the far field posterised into electric blue,
JPEG blocks amplified, clouds crunchy. Numbers on contrast and haze alone cannot see that, so the
real-fixture tests also cap fine-detail energy in the far-ridge band (luma minus a small blur, at a
fixed width): a natural grade roughly doubles it at most, while a haloed one more than triples it. When
retuning, look at a full-size ridge crop before trusting any metric.

**The clarity base follows edges (`grade._edge_following`, a guided filter on luma), never a plain
Gaussian.** A wide Gaussian base smears a dark near ridge into the haze above it, and the "detail"
left over paints a bright glow along the hazy side of the ridge, which the edge-following base avoids. A synthetic
dark-ridge-under-haze test holds that glow under 0.02. Ridge contrast in the real-fixture tests is
windowed local contrast for the same reason: the spread of the whole band counts that glow as
contrast.

A dehazed frame also reads darker overall on plain average once the bright veil is gone, even
though the near ground never changed, so `grade._apply_exposure` raises its target mean luma in
proportion to the measured haze strength (`_EXPOSURE_DEHAZE_TARGET_BOOST`) rather than chasing a
fixed whole-frame target — chasing that fixed target was itself a bug: it darkened
already-good frames and over-lifted a dim, haze-free frame (a shadowed lake) toward the
target. The actual luma move is capped by how much haze this frame has
(`_EXPOSURE_CLEAR_MAX_DELTA` on a clear frame, up to `_EXPOSURE_HAZY_MAX_DELTA` on a heavily hazed
one), with a little extra room for a genuinely dim, haze-unrelated frame
(`_EXPOSURE_UNDEREXPOSED_LUMA`/`_EXPOSURE_UNDEREXPOSED_MAX_DELTA`). Because the midtone gamma is
only an approximate solve on a wide or bimodal histogram (a bright sky over dark water), the actual
resulting mean can overshoot that cap; `_apply_exposure` measures the true delta afterward and
blends back toward the original by exactly enough to hold it, rather than trusting the solved gamma.

`grade._apply_vibrance` scales HSV saturation with more gain for less-saturated pixels, solved per
image from the gap between the image's Hasler-Süsstrunk colourfulness and a target, capped — a grey
trail sign gets a capped lift, an image already at or above the target gets none, and it never
reduces saturation. `grade._apply_s_curve` is a small blend toward a smoothstep curve applied per
channel (its fixed points at 0 and 1 keep already-saturated primaries exact).

None of the stages above individually reduce colourfulness, but their interaction (exposure's
gamma, the S-curve) can still cost a small amount of the chroma an earlier stage added, netting a
small overall loss on a frame that started already fairly colourful (an already-rich frame can read duller after grading). `grade._never_duller` is a closing guarantee,
not a per-stage tweak: it measures colourfulness before and after the whole grading pipeline and, if
the graded output reads duller, binary-searches (`_COLOURFULNESS_NEVER_DULLER_SEARCH_STEPS`) the
smallest blend back toward the original that holds the floor. It guards grading only — a structural
crop before grading (`crop_lens_corners`, see below) can legitimately change a frame's own
colourfulness by discarding real content, which is outside what grading can promise to recover.

All constants blocks (haze, exposure, vibrance, curve/dither, corner crop) are named module-level
values in `grade.py`; tune there, not by re-deriving a formula. `grade.measure()` (colourfulness,
mean luma, clip fraction, haze estimate) is read by tests now and is meant for a preflight contact
sheet later.

`grade.crop_lens_corners`, run before grading, removes dark rounded lens-hood vignetting some
frames carry, which `hud.crop_black_border` cannot see. That
crop only removes a straight, unambiguously black edge, never a rounded corner arc. Detection
compares the four corner patches' mean luma against the straight mid-edge patches'; when the
corners are dark and the edges are not, the diagonal extent of each dark arc is walked inward per
corner and the largest is cropped as an equal *fraction* of each dimension (not an equal pixel
margin), which is what keeps the aspect ratio exact. A clean frame's corners never trigger it.
Real lens-hood vignetting is a small margin (roughly 0.07-0.13 of the
short side on true-vignette fixtures); a dark, textured scene corner -- trees or a riverbank framing the shot -- can pass the same corners-dark/edges-bright test with a much larger
extent (around 0.40). Past `_CORNER_CROP_MAX_FRACTION` the read is treated as scene content, not vignetting, and
the frame is returned uncropped rather than clamped to the bound: clamping a 40%-extent read as if
it were a small real margin would still discard real content.

## One developed still for the reel, the cards and the inserts

`photos reel` (`photos/reel.py`) takes each still from `hud._load_still` -- the same
graded, disk-cached pixels `assembly`/`still_card_frame` and the insert engine
(`inserts._load_still_default`) use -- instead of a private `develop.decode`. A photo is
graded once per machine, and a reel looks like the film it sits beside. A Pillow-readable still is
loaded inside that shared loader: a suffix in `develop.SUPPORTED_EXTENSIONS` goes through
`develop.decode`; any other Pillow-readable suffix (an insert's `.webp`) is oriented by EXIF,
flattened to RGB and passed through the same `develop._srgb_from_icc` step. Either way every
consumer gets EXIF orientation and ICC colour management, and an unusable ICC profile (or an
unreadable still) raises -- `ReelError` from the reel.
Pinned by `test_reel_and_still_card_share_developed_frame`.
