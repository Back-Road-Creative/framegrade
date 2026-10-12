# Read-only night project storage

`inventory_project(row, workspace)` reports `source_linear_bytes` from the four
fixed linear-artifact files, counting each resolved source once. This is **not RAW
storage**: `source_raw_bytes` remains unknown (`null`) until a RAW catalog exists.
The parent card tree and unrelated source files are never walked.

`generated_bytes` counts regular files in contained, recognized project attempt
directories. Symlink paths are skipped with explicit warnings. Authenticated row
ownership is the caller's responsibility; project identifiers must be 64 hex digits.
Missing/inaccessible files produce warnings, so totals can be incomplete.

Sources and current outputs remain protected. Receipt-bearing attempts—including
uncommitted crash results, preview-pending masters or malformed receipts—are never cleanup candidates.
Active/queued attempts also remain protected. Unreferenced incomplete attempts in
stopped projects are review candidates with bytes and reasons only. This function
never deletes, establishes a retention policy, or declares a candidate safe to
remove without checking other jobs, selected keepers and artifact dependencies.
