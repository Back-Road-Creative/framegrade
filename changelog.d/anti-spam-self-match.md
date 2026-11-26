### Fixed

- The anti-spam shared-prefix check no longer refuses a Short for sharing its opening with its own stored copy: a Short is now told apart by project and Short number, not by its place in the list, and a refusal names the other Short. Two different Shorts with the same opening are still refused.
- The nightly title sync no longer feeds a sidecar's other Shorts (ones queued under another spelling of the same file path) into the gates when that sidecar cannot be planned or written, so a Short is never checked against its own old text.
