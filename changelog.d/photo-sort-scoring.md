### Added

- `gomoveshift photos sort` now scores every developed shot and orders each category by quality: technical
  sharpness and clipping (plus stars for a night sky), a per-category signal, and the local aesthetic model.
  Files are renamed `NNN-<stem>.jpg` with NNN the quality rank (1 = best); `ranked.csv` carries every
  component, and `index.html` shows each section in rank order with the total under each thumbnail. The
  aesthetic pass runs once per run on the CPU (2 threads), caches by source sha256 in the ledger, and
  `--gpu` takes the shared GPU lock for that pass only, falling back to CPU if it is busy.
- `gomoveshift photos fetch-models`: the explicit, hash-checked download of the aesthetic weights.
