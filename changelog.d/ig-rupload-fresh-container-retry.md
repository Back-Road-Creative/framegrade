### Fixed

- An Instagram video upload that `rupload.facebook.com` refused was stored as "unknown Graph API error", and its row went `failed_permanent` on the first try (15 of 19 rows from 2026-10-31 to 11-14). The error now names rupload's `debug_info` type and message, and a failed upload starts over on a fresh container, up to 3 tries.
