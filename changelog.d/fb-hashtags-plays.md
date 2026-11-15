### Fixed

- The `fb_hashtags` caption experiment now compares arms on Facebook Reel initial plays instead
  of reach. Graph answers no per-Reel reach for the live Page, so every arm had reported `n=0`
  since the experiment started. The nightly line now reads `median_plays=` (#385).
