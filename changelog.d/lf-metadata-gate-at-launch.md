### Fixed

- A long-form drive whose location string or footer config trips the description redundancy gate now fails at launch, before the encode, with the same error prefixed `LF-METADATA-PREFLIGHT`. It used to fail in Phase D after a multi-hour render.
