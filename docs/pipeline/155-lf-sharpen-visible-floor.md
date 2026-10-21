## LF sharpen visible floor (ENC-1) -- measured 2026-10-12

`quality.sharpen_amount_for` ramps the LF unsharp amount down as measured detail rises, so a
near-crisp 4K day master lands just under the crisp end with an amount around 0.01-0.015. Those
amounts are invisible but still pay for an 11x11 unsharp kernel at 4K. Any amount below
`SHARPEN_MIN_AMOUNT` (0.05) is therefore dropped, together with its radius, and the filter chain
gets no `unsharp` stage.

Measured on a 20 s 4K LF-master clip (ss=600, 4 threads, two interleaved reps of each chain):

| Chain | CPU-s (user+sys) |
|---|---|
| live chain with `unsharp` at 0.01 | 236.8, 237.8 |
| same chain without `unsharp` | 139.5, 133.5 |

That is about 42% of the chain's CPU, or 1.74x. Wall clock was not used, because a live encode
shared the box. Against no unsharp, PSNR is 63.77 dB at amount 0.01 and 63.33 dB at 0.05, both far
past visible. There is no GPU unsharp path on this build: Vulkan is llvmpipe and there are no
CUDA/NPP filters. The photo pipeline keeps its own ramp (`photos/develop.py`) and has no floor.
