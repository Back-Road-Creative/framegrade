### Fixed

- Instagram Reels uploads no longer fail `ProcessingFailedError` on Shorts that break Meta's Reels spec (HEVC Main 10, an `elst` edit list, 27-30 Mbps). The Instagram adapter now derives its own H.264 High 8-bit file (closed GOP, VBR capped at 20 Mbps, `+faststart`, no edit list, AAC 48 kHz), gates it with ffprobe and a box walk before any container is created, and uploads those bytes on every retry. An out-of-spec result is `failed_permanent` naming the broken rule. The YouTube file is untouched (#399).
