### Fixed

- The `🗺️ Through:` line drops the LF's own label when it is the trailing qualifier of a region name, so an LF whose location was broadened to a state lists `Polk County, Henderson County` instead of repeating `North Carolina` per county and failing the description redundancy gate (US-64, 2026-11-07).
