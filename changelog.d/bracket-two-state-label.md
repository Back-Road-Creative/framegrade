### Changed

- A bracket route whose two fixes sit in different states of one country now labels the leg in travel order (`Virginia to Pennsylvania`, level `states`, 36 h cap) instead of only the country (nctopa-2026-03-05-rec0063 was labelled `United States`). Past the state cap, or where a fix has no state, the label is still the country.
- A resumed drive whose stored bracket record now earns a different label (a two-state leg staged as its country) is re-labelled and re-burned: the label is recomputed from the record's own fixes, and the old text's renders and OST confirmations are dropped.
