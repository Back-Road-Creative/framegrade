### Changed

- A drive with a GPS-less episode now reads its road signs by default. The sign OCR pass was opt-in (`--sign-verify`), so a drive the camera never located (`nctopa-2026-03-05-rec0063`: 13 chapters, zero locked fixes) rendered with no evidence of where it was. The operator's decision still wins: `--no-sign-verify` turns it off, `--sign-verify` turns it on for any drive. A GPS-less day is sampled every `sign_verify_gpsless_sample_period_s` (default 10 s) instead of the flat 120 per day, and EasyOCR runs on the GPU when torch sees one.
