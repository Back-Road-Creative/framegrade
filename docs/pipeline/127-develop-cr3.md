## Canon CR3 RAW ingest (`develop.py`, `ingest.py`) <a id="develop-cr3"></a>

`RAW_EXTENSIONS` (`develop.py`) now includes `.cr3` alongside `.dng`/`.cr2`/`.nef`/`.arw`. Canon's CR3
container is decoded through the same `rawpy.imread` -> `postprocess` path every other RAW extension
uses (`_decode_raw`); no CR3-specific branch exists because LibRaw handles the CRX codec itself once
it is new enough (LibRaw >=0.21; the shipped `.venv` carries 0.22.1, verified via
`rawpy.libraw_version`). `SUPPORTED_EXTENSIONS` and `ingest.py`'s card-scan glob both derive from
`RAW_EXTENSIONS`, so a `.cr3` on a card is discovered, developed and registered exactly like a `.cr2`
with no further wiring.
