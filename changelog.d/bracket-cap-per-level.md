### Changed

- The bracket route (`--bracket-gpx`) caps the gap between its two fixes per level (`BRACKET_MAX_GAP_H_BY_LEVEL`): a county or state still needs fixes at most 36 h apart, a country at most 48 h. A gap past a level's cap drops the label to the next broader level both fixes agree on, so two day tracks a day apart (≈ 44 h) inside one county name the country instead of being refused. Over 48 h is still refused.
