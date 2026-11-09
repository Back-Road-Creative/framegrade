### Added

- `gomoveshift.pipeline.photos.aesthetic`: a local, CPU-by-default aesthetic scorer for judging the most
  striking photos. CLIP ViT-L/14 (int8 ONNX) plus the LAION aesthetic head, both pinned by size and sha256 and
  downloaded only by an explicit `fetch_models()`; the head is read without torch through a restricted
  unpickler. `AestheticScorer.score` returns the raw ~1-10 value, `score_unit` a 0-1 clamp.
