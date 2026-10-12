## Exposure, highlight/shadow recovery, and Kelvin/tint (`develop.py`, `color_correction.py`) <a id="develop-exposure-and-white-balance"></a>

`DevelopOptions` gained five operator knobs, all no-ops at their `0.0` default so existing looks and
develop output are byte-identical when unset:

- `exposure_ev` — a stops gain (`2**exposure_ev`) applied by `_apply_exposure`, the FIRST stage of
  `_grade`, ahead of the tone curve — the way a camera's exposure dial precedes highlight/shadow
  recovery. `CorrectionParams` carries the same field name, so `develop._look_params` derives it
  automatically for a `--look-variant`; `plan_variants` wires it explicitly for a main run.
- `shadow_recovery`/`highlight_recovery` — ADD onto the measured `shadow_lift`/`highlight_pull`
  inside `_apply_curves` (including its highlight-knee threshold, which now reads the summed total)
  rather than replacing them, exactly as the white-balance fields add onto the measured cast: an
  operator lifting shadows further must not discard the metering that already lifted them some.
- `white_balance_kelvin`/`white_balance_tint` — a friendlier pair over the existing raw
  `white_balance_r`/`_b`/`_g` axes. `_wb_shifted` converts Kelvin into a symmetric r/b delta via
  `KELVIN_TO_WB_SCALE` (`+2000K` warms the frame by the same amount `white_balance_r=0.10` does —
  `WB_OPERATOR_SPAN`'s own travel) and adds tint straight onto `white_balance_g`, on top of whatever
  the raw axes and the measured cast already contributed. These two live on `DevelopOptions` only —
  they are a UI convenience over state `CorrectionParams` already has, not new renderer state.

All five are registered in `config.PHOTO_KNOBS` (`exposure_ev`, `shadow_recovery`,
`highlight_recovery` span-gated; `white_balance_kelvin`/`white_balance_tint` alongside the other
white-balance rows) so a `--look` may carry them, validated the same way as every other knob.

Each knob also has a `photos develop` flag (`--exposure-ev`, `--shadow-recovery`,
`--highlight-recovery`, `--white-balance-kelvin`, `--white-balance-tint`), gated against its
`PHOTO_KNOBS` span at the door before any decode. The handler's white-balance loop walks every
`white_balance*` registry row, so a registry row with no matching flag crashes every develop run.
Adding a flag is therefore part of adding a row.
