### Local aesthetic scorer (`pipeline/photos/aesthetic.py`)

Judges "most striking" on this machine, on the CPU, with no network call at scoring time so it never competes
with a GoMoveShift run for the GPU. `AestheticScorer.score(rgb)` runs the CLIP ViT-L/14 image encoder (int8
ONNX, 768-d embedding), L2-normalises the embedding and passes it through the LAION improved-aesthetic-predictor
head (`sac+logos+ava1-l14-linearMSE`, five linear layers, no activations), giving roughly 1-10.
`score_unit(rgb)` maps that to 0-1 as `clamp((raw - 1) / 9, 0, 1)`.

Numbers: input is resized (shortest edge 224, bicubic), centre-cropped to 224x224 and normalised with CLIP's
mean/std; the session uses `threads=2` intra-op, 1 inter-op, `CPUExecutionProvider` only unless `use_gpu=True`
(CUDA first, CPU fallback). `use_gpu=True` callers take the GPU lock themselves; this module has none.

Weights live in `$FRAMEGRADE_AESTHETIC_DIR`, else `<gms data home>/models/aesthetic`, and are pinned by size
and sha256 (vision 307001500 bytes, head 3714759 bytes). `fetch_models()` is the only code that downloads:
to `<name>.part`, size and sha256 verified, then atomically renamed.

Refusals: a missing or hash-mismatched file raises `AestheticModelUnavailable` naming the fetch command unless
a download is allowed; a download of the wrong size or hash is deleted and raises. The head's `.pth` is read
without torch by a restricted unpickler that admits only `OrderedDict`, `_rebuild_tensor_v2` and
`FloatStorage`; any other global, a non-contiguous tensor or a missing `layers.*` key is refused.
