### Added

- `gomoveshift/pipeline/photos/quality_order.py`: a per-category quality score for
  ordering shots inside a category (never for excluding one). It scores the developed
  image: sharpness and clipping, star count, star sharpness and trailing for a night
  sky, and a category signal (landscape colour and sky, night-sky darkness and
  foreground separation, close-up subject isolation). An optional aesthetic score
  blends in at weight 0.5. The result carries every named component for `ranked.csv`.
