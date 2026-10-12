## Brand kit (`console/settings.brand_kit`, `console/brands/<slug>/brand.yaml` `kit:` block) <a id="brand-kit"></a>

An optional `kit:` block in a brand's `brand.yaml`, sibling to the existing
`disabled_tabs:` key (`docs/pipeline/93-brands.md`). A brand whose `brand.yaml` has no `kit:` block (or has no
`brand.yaml` at all) gets `None` from `console.settings.brand_kit(slug)` — the OST drawtext
pipeline is therefore unaffected for every brand that does not opt in.

Schema (all keys optional; `None`/missing means "use the pre-kit default" for that field):

```yaml
kit:
  font: /path/to/font.ttf        # or a family name string
  colors:
    primary: "#RRGGBB"
    accent: "#RRGGBB"
    text: "#RRGGBB"               # drawtext fontcolor for OST burns
  logo: /path/to/logo.png
  safe_margin: 0.08                # fraction in [0, 1]
```

`brand_kit()` validates at load and raises `ValueError` naming the offending `kit.<field>` on
any bad value (wrong type, a color that isn't `#RRGGBB`, a `safe_margin` outside `[0, 1]`) —
fail loudly rather than silently ignore a misconfigured brand.

### OST consumption (`pipeline/ost_applier.py`)

`apply_ost()` and `_build_drawtext_filter()` take an optional `brand_kit` dict (the same shape
`brand_kit()` returns). When `None` — every production call site today — the emitted drawtext
filter string is BYTE-IDENTICAL to before this parameter existed:

- Font resolution stays `font_path or FONT_PATH` (the module's DejaVu Sans Bold constant);
  `brand_kit["font"]` only applies when `font_path` was not explicitly passed.
- Every burned entry's `fontcolor=` stays the literal `white@1.0` (or `white@0.85` for the
  persistent URL watermark) via `_kit_text_color(None) == "white@1.0"`.

When a kit IS present, `brand_kit["colors"]["text"]` overrides `fontcolor=` on every drawtext
entry (brand, location, YouTube CTA, persistent URL, website CTA, superlong lower-thirds), and
`brand_kit["font"]` overrides the font file when the caller didn't pass an explicit `font_path`.

`logo` and `safe_margin` are validated here but not yet consumed — reserved for the route-story
HUD and thumbnail follow-ups; this change is schema + OST only.
