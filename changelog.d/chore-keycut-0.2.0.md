### Changed

- Pinned `keycut` to v0.2.0. The day-master cut now goes through keycut's `extract_and_concat_ranges_with_receipt` and checks the live assembly receipt with `load_receipt`; a receipt that fails validation makes `extract_and_concat_segments` log the failing field and return None, since a time mapping built from it would be wrong. The lead-seam re-encode now seeks on the input side (keycut 0.1.2), which gives the same frames without decoding from the start of the file.
