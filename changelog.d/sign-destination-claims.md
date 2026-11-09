### Added

- Guide/destination signs ("EXIT 283 Woodstock 1 MILE") are now an advisory `destination` sign claim with the stated distance. The verifier forward-geocodes the place (with an N/W initial variant for OCR misreads) within `sign_verify_destination_max_km` (default 80) of where it was read and may corroborate; it never conflicts and never feeds a label.
