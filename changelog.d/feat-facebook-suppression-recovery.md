### Changed

- The Facebook suppression detector now leaves `SUPPRESSED`: a 7-day median back to 0.5x the baseline frozen at the flip is `RECOVERING`, 7 straight days at 0.7x is `HEALTHY`, and a fall under 0.3x relapses to `SUPPRESSED`. Same function signature and return shape.
