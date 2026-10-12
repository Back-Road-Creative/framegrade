"""Built-in ``photos develop --look NAME`` presets — plain data, no imports.

``photos.looks`` ships empty, so a look only ever recalled what the operator
hand-authored out of raw 6-tuples: nothing themed in the box, and the whole
colour-knob surface invisible to anyone who has not memorised it. This table is
the in-box half. The handler resolves ``{**BUILTIN_LOOKS, **photos.looks}``, so
**the operator's config WINS a name collision** — their override survives an
upgrade that ships a preset of the same name.

Keys are the ``photos develop`` **FLAG** names, never ``DevelopOptions`` field
names: ``split_tone_shadow``, ``split_tone_midtone``, ``split_tone_highlight``,
``bw_mix``, ``reference``, and the run-owned ``straighten``/``crop`` a look may not
carry (the seven ``config.LOOK_ALIASES`` maps). A look FILLS unset flags, which is why
it is spelled in the flag vocabulary — using the field name is the single
easiest way for a preset to fail ``LOOK_KNOBS`` validation.

The per-pack split is **deliberate and load-bearing, do not flatten it**: the
packs are authored independently, and appending to one shared dict would
conflict outright. One pack, one dict, merged here.

A pack lands only once its presets are calibrated: every preset needs a
strength-sweep against a reference card, which no hermetic test can stand in
for, so the recipes land per pack rather than all at once.

Nothing imports ``gomoveshift.config`` from this module (nor anything heavy): a
leaf data module keeps the config validator and this table on opposite sides of
the import-time cycle ``config.LOOK_KNOBS`` documents.
"""

# ── ELECTRONIC ── the after-dark pack: neon, chrome and the colours film never
# had. Each preset names its own base ``styles`` so ``--look NAME`` renders ONE
# deterministic variant rather than an auto-selected spread, and the colour knobs
# ride that base. Strengths are hand-calibrated starting points — a hermetic test
# can only pin that they are legal, distinct, deterministic and pointed the right
# way (tests/test_photo_electronic_pack.py, and tests/test_photo_artifact_looks.py
# for the three named after a spatial stage); the eyeball proof sheet sets taste.
_ELECTRONIC: dict[str, dict] = {
    # Rain-slick street at 2am: teal shadows, magenta where the signs blow out, a
    # crushed toe under both, and the cyan bloom `halation_tint` finally allows.
    "cyberpunk": {
        "styles": "moody",
        "split_tone_shadow": [-0.05, 0.06, 0.12],
        "split_tone_highlight": [0.12, -0.04, 0.10],
        "curve_r": [0.0, 0.0, 0.20, 0.12, 0.65, 0.70, 1.0, 1.0],
        "curve_g": [0.0, 0.0, 0.20, 0.13, 0.65, 0.66, 1.0, 1.0],
        "curve_b": [0.0, 0.02, 0.20, 0.20, 0.65, 0.72, 1.0, 1.0],
        "hue_sat": [0.90, 0.75, 0.80, 1.60, 1.60, 1.70],
        "vibrance": 0.7,
        "clarity": 0.35,
        "halation": 0.5,
        "halation_tint": [1.00, 0.30, 0.90],
    },
    # The airbrushed sunset: violet shadows climbing to a pink highlight over a
    # lifted matte base, grain and a heavy vignette for the VHS edge.
    # It deliberately does NOT use ``duotone``, which was the first cut: duotone
    # is ``shadow + luma·(high − shadow)`` (``color_correction._apply_bwmix``),
    # so it REPLACES every pixel's colour with a point on a two-colour ramp (the
    # ``duotone_strength`` lever landed later). On a real photograph that is a
    # flat wash with the subject gone, not a look — measured on a landscape
    # frame, the sky, the ridges and the cloud all resolved to one magenta.
    # Split-tone plus a hue push reaches the same palette while the frame keeps
    # its own colour.
    "synthwave": {
        "styles": "matte_shoulder",
        "split_tone_shadow": [0.06, -0.05, 0.14],
        "split_tone_highlight": [0.14, -0.03, 0.07],
        "hue_sat": [1.00, 0.70, 0.55, 1.10, 1.30, 1.45],
        "vibrance": 0.30,
        "grain": 0.25,
        "vignette": 0.40,
        "halation": 0.35,
        "halation_tint": [1.00, 0.25, 0.85],
    },
    # Brushed metal: cool at BOTH ends (the split-tones agree instead of
    # opposing), every hue but blue and cyan drained, and the detail pushed hard.
    "chrome": {
        "styles": "natural",
        "split_tone_shadow": [-0.06, -0.01, 0.10],
        "split_tone_highlight": [-0.05, 0.00, 0.09],
        "hue_sat": [0.30, 0.25, 0.35, 1.20, 1.25, 0.45],
        "clarity": 0.55,
        "sharpen": 1.6,
    },
    # Low-key noir lit by signage: the curves crush everything toward black, the
    # warm end drains out, and only the cyan/magenta neons keep their chroma —
    # with a cold bloom off them.
    "neon_noir": {
        "styles": "moody",
        # The toe sits at 0.22, not the 0.12 of the first cut: this grade chain
        # peaks near 0.66 luma, so a curve calibrated on a full-range frame
        # crushed a real landscape to near-black with the ridges gone. Low-key
        # has to keep the subject readable to be a look rather than a failure.
        "curve_r": [0.0, 0.0, 0.35, 0.22, 0.75, 0.68, 1.0, 0.95],
        "curve_g": [0.0, 0.0, 0.35, 0.22, 0.75, 0.68, 1.0, 0.95],
        "curve_b": [0.0, 0.0, 0.35, 0.26, 0.75, 0.74, 1.0, 1.00],
        "hue_sat": [0.15, 0.10, 0.15, 1.40, 1.25, 1.50],
        "clarity": 0.30,
        "vignette": 0.35,
        "halation": 0.6,
        "halation_tint": [0.25, 0.70, 1.00],
    },
    # ── the artifact three: each one NAMED for a spatial stage, and the first
    # recipes in the box to use any of them. Amounts stay conservative on purpose
    # — the picture still has to read as the picture — so a strength sweep tunes a knob
    # rather than rescuing a look.
    # The tape read: a CRT line structure over a lens fringe, the chroma smeared
    # down and the black end lifted onto matte_shoulder's ramp, because tape never
    # reaches black. `scanline` rides a SOFT sine, so the line survives an export
    # downscale as structure instead of folding into moire.
    "vhs": {
        "styles": "matte_shoulder",
        "scanline": 0.55,
        "chromatic_ab": 0.50,
        "hue_sat": [0.70, 0.70, 0.65, 0.75, 0.75, 0.70],
        "curve_r": [0.0, 0.04, 0.5, 0.50, 1.0, 0.96],
        "curve_g": [0.0, 0.05, 0.5, 0.51, 1.0, 0.97],
        "curve_b": [0.0, 0.07, 0.5, 0.52, 1.0, 0.94],
        "grain": 0.25,
    },
    # The corrupted-neon poster: a cyberpunk split-tone pair on a vivid base, torn
    # by a fault in the SIGNAL. `glitch` is the restrained knob in the box — it
    # tears bands sideways, it does not shred the frame — so the neon shouts and
    # the tear punctuates.
    "glitch_pop": {
        "styles": "vivid",
        "glitch": 0.35,
        "split_tone_shadow": [-0.06, 0.02, 0.14],
        "split_tone_highlight": [0.14, -0.05, 0.09],
        "vibrance": 0.8,
        "clarity": 0.25,
    },
    # The light-struck negative: a tungsten split-tone and the halation that comes
    # with it, fogged from the corner `light_leak` owns. The leak SCREEN-blends, so
    # the corner lifts without blowing what is already bright there.
    "leaked_film": {
        "styles": "natural",
        "light_leak": 0.60,
        "split_tone_shadow": [-0.03, -0.02, 0.08],
        "split_tone_highlight": [0.10, 0.04, -0.03],
        "halation": 0.55,
        "grain": 0.30,
    },
}

# The film/blockbuster pack. Every recipe is built out of knobs ``photos develop``
# already has — no preset introduces an engine — and none pins ``styles``, so a
# look layers onto whichever variants the scene picks instead of overriding the
# operator's choice of them. ``curve_*`` control points are FLAT alternating x y
# with x strictly ascending in [0, 1]; the six ``hue_sat`` slots are the anchor
# hues red/yellow/green/cyan/blue/magenta in that order (a seven-slot recipe puts
# orange 30° second — every preset below is deliberately still six, unretuned).
_CINEMATIC: dict[str, dict] = {
    # The blockbuster split. Skin sits under the red and yellow anchors, so both
    # are held BELOW 1.0 while cyan/blue are pushed: the shadows take the teal and
    # faces keep their own colour instead of following the highlights orange.
    "teal_orange": {
        "split_tone_shadow": [-0.07, 0.01, 0.11],
        "split_tone_highlight": [0.12, 0.02, -0.08],
        "hue_sat": [0.85, 0.80, 1.05, 1.30, 1.30, 1.00],
        "clarity": 0.15,
    },
    # Silver retention: colour drained toward grey with the contrast PUT BACK as a
    # hard S on all three channels equally (one shared curve ⇒ no cast), so it
    # reads bleached rather than merely desaturated.
    "bleach_bypass": {
        "hue_sat": [0.40, 0.40, 0.40, 0.40, 0.40, 0.40],
        "curve_r": [0.0, 0.0, 0.25, 0.13, 0.5, 0.5, 0.75, 0.87, 1.0, 1.0],
        "curve_g": [0.0, 0.0, 0.25, 0.13, 0.5, 0.5, 0.75, 0.87, 1.0, 1.0],
        "curve_b": [0.0, 0.0, 0.25, 0.13, 0.5, 0.5, 0.75, 0.87, 1.0, 1.0],
        "clarity": 0.45,
    },
    # C-41 chemistry in E-6: three INDEPENDENT channel curves, which is the whole
    # point — a split-tone pair cannot pull the two ends onto different axes.
    # Red is crushed low and lifted high, blue does the mirror: green-cyan
    # shadows, yellow highlights, out of tone shaping alone.
    "cross_process": {
        "curve_r": [0.0, 0.0, 0.25, 0.17, 0.75, 0.84, 1.0, 1.0],
        "curve_g": [0.0, 0.06, 0.5, 0.54, 1.0, 1.0],
        "curve_b": [0.0, 0.13, 0.5, 0.47, 1.0, 0.86],
    },
    # Tungsten stock in daylight: cool shadows, warm highlights, and the halation
    # it is famous for at the SHIPPED red bias — that bias is the look, so
    # ``halation_tint`` is deliberately left unset rather than re-specified here.
    "cinestill": {
        "split_tone_shadow": [-0.02, -0.02, 0.07],
        "split_tone_highlight": [0.10, 0.04, -0.03],
        "halation": 0.75,
        "grain": 0.35,
    },
    # Instant film: a lifted matte toe (the black end never reaches black), a warm
    # cast, colour a little drained — and the white matte the print is mounted in,
    # the one preset in the pack that carries a presentational knob.
    "polaroid": {
        "curve_r": [0.0, 0.10, 0.5, 0.55, 1.0, 0.97],
        "curve_g": [0.0, 0.09, 0.5, 0.52, 1.0, 0.95],
        "curve_b": [0.0, 0.12, 0.5, 0.48, 1.0, 0.90],
        "split_tone_highlight": [0.05, 0.02, -0.04],
        "hue_sat": [0.85, 0.85, 0.80, 0.80, 0.85, 0.85],
        "border": "white",
    },
}

# ── MASKED ── the pack that rides the frame's own regions. Every preset here is
# built out of ``layers``: one named soft mask plus the colour knobs that grade
# where it is non-zero, composed over the global grade in list order. Colour
# only, by construction (``config.check_layers``), and each recipe is written in
# channel CURVES rather than split-tones wherever it must bite regardless of
# where the region's tone happens to sit — a split-tone keyed to absolute luma
# lands nowhere on a region the frame's own span put in the middle.
# None pins ``styles``: a regional grade layers onto whichever variants the scene
# picks, exactly as ``_CINEMATIC`` does.
#
# AUTHORING CONSTRAINT — the pack's distinctness bar is metered WHOLE-FRAME, so a
# small region has to move far to register: a single-layer look needs roughly
# ``MIN_RENDERED_DISTINCTNESS / coverage`` levels inside its own region. On the
# pack fixture that is ~33 levels at ``glints`` (coverage 0.15) and ~20 at
# ``linear`` (0.25); below about 8% coverage no tasteful grade clears the bar at
# all. Budget for it up front — reach for a complementary layer PAIR (``sky`` +
# ``ground``, ``radial`` + its inverse, the way split_world, vignette_wash and
# subject_pop each cover the whole frame) or a wider mask, rather than shouting
# louder inside a thin one.
_MASKED: dict[str, dict] = {
    # The horizon split one exposure cannot hold: the sky cooled and its blues
    # pushed, the ground (the sky mask's own complement, so the two meet with no
    # seam) warmed the other way.
    "split_world": {
        "layers": [
            {
                "mask": "sky",
                "curve_r": [0.0, 0.0, 0.5, 0.42, 1.0, 0.94],
                "curve_b": [0.0, 0.0, 0.5, 0.60, 1.0, 1.0],
                "hue_sat": [1.0, 1.0, 1.0, 1.25, 1.30, 1.0],
            },
            {
                "mask": "ground",
                "curve_r": [0.0, 0.04, 0.5, 0.60, 1.0, 1.0],
                "curve_b": [0.0, 0.0, 0.5, 0.40, 1.0, 0.92],
            },
        ]
    },
    # Signage glow: a magenta/cyan tint on the lit SIGNS, so the neon reads as
    # light rather than as a filter over the whole photograph. The mask choice IS
    # the recipe. ``highlights`` — every pixel above 2/3 luma — covered a third of
    # a landscape (measured), because there the bright region is the
    # sky, and this look read as a magenta sky. ``glints`` asks the question the
    # scene needs instead, grouping those same bright pixels into regions and
    # keeping only the SMALL ones: on the pack's signage landscape it weighs 0.15
    # of the frame where ``highlights`` weighs 0.48, and it weighs exactly 0.0 on
    # the lit sky (25% of that frame, and 1.0 under ``highlights``), which is now
    # left byte-identical rather than moved 21 levels. The registry's own
    # ``max_area`` default stands unoverridden: a sign there is 1.4% of the frame
    # against that sky region's 25%, so neither is within reach of the calibrated
    # 2% limit and a per-look opinion would only be a second, unmeasured one.
    "neon_edge": {
        "layers": [
            {
                "mask": "glints",
                # A PLATEAU, not a single knot, and that is the recalibration the
                # mask swap forced: the deviation is held flat from 0.35 to 0.75
                # rather than peaking at 0.5 and decaying to nothing by 1.0. A
                # glint is the top of the frame's OWN span, which lands at a
                # different absolute code on every photograph — 0.50 and 0.70 on
                # two cuts of the same fixture — so a curve keyed to one tone is a
                # look on one frame and half a look on the next, the lesson this
                # pack's docstring already carries about split-tones. Flat, it
                # measures 39.1 / 39.4 / 41.3 8-bit levels inside the region right
                # across that range. Amplitude went UP too (+0.11/-0.11/+0.13 from
                # +0.08/-0.07/+0.10): the swap ALONE rendered 3.93 whole-frame
                # distinctness against the 5.0 floor — no longer a look at all —
                # where this measures 6.38. It can afford to shout now precisely
                # because it can no longer reach the sky.
                "curve_r": [0.0, 0.0, 0.35, 0.46, 0.75, 0.86, 1.0, 1.0],
                "curve_g": [0.0, 0.0, 0.35, 0.24, 0.75, 0.64, 1.0, 0.94],
                "curve_b": [0.0, 0.0, 0.35, 0.48, 0.75, 0.88, 1.0, 1.0],
                "vibrance": 0.5,
            }
        ]
    },
    # A wash, not a vignette: the frame's edges (the radial falloff INVERTED)
    # drained and cooled while the centre holds its warmth — the colour half of
    # what a vignette does to tone, which no spatial knob here can express.
    "vignette_wash": {
        "layers": [
            {
                "mask": "radial",
                "invert": True,
                "hue_sat": [0.35, 0.35, 0.35, 0.45, 0.45, 0.35],
                "curve_r": [0.0, 0.0, 0.5, 0.42, 1.0, 0.92],
                "curve_b": [0.0, 0.02, 0.5, 0.57, 1.0, 1.0],
            },
            {
                "mask": "radial",
                "curve_r": [0.0, 0.02, 0.5, 0.57, 1.0, 1.0],
                "curve_b": [0.0, 0.0, 0.5, 0.44, 1.0, 0.95],
            },
        ]
    },
    # Sunset ground glow: a warm lift ramped in over the bottom half of the frame
    # (``linear`` pointing down over half the height — the registry's defaults,
    # spelled out now that a layer carries its mask's own shape params), so the
    # light comes up off the ground and the sky above it is untouched, exactly.
    "golden_ground": {
        "layers": [
            {
                "mask": "linear",
                "angle": 90.0,
                "span": 0.5,
                "curve_r": [0.0, 0.10, 0.5, 0.68, 1.0, 1.0],
                "curve_g": [0.0, 0.05, 0.5, 0.57, 1.0, 1.0],
                "curve_b": [0.0, 0.0, 0.5, 0.38, 1.0, 0.92],
            }
        ]
    },
    # The subject forward, its surround back — the one recipe in the pack that
    # rides DETAIL (``texture``) rather than tone or position, which is what
    # keeps it from being the centre/edge split ``vignette_wash`` already is: a
    # busy hedge and the flat sky above it can sit at the identical brightness
    # in the identical part of the frame. The smooth surround is drained, cooled
    # and flattened toward a haze; the detailed region takes contrast around
    # where this chain's tones actually sit plus a vibrance push, so the eye
    # lands on the picture's subject instead of on wherever it is brightest.
    "subject_pop": {
        "layers": [
            {
                "mask": "texture",
                "invert": True,
                "hue_sat": [0.45, 0.45, 0.45, 0.50, 0.50, 0.45],
                "curve_r": [0.0, 0.06, 0.5, 0.46, 1.0, 0.92],
                "curve_g": [0.0, 0.06, 0.5, 0.47, 1.0, 0.93],
                "curve_b": [0.0, 0.08, 0.5, 0.52, 1.0, 0.98],
            },
            {
                "mask": "texture",
                "curve_r": [0.0, 0.0, 0.30, 0.27, 0.70, 0.76, 1.0, 1.0],
                "curve_g": [0.0, 0.0, 0.30, 0.27, 0.70, 0.75, 1.0, 1.0],
                "curve_b": [0.0, 0.0, 0.30, 0.28, 0.70, 0.73, 1.0, 1.0],
                "vibrance": 0.45,
            },
        ]
    },
}

BUILTIN_LOOKS: dict[str, dict] = {**_ELECTRONIC, **_CINEMATIC, **_MASKED}
