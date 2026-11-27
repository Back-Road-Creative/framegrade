### Fixed

- `health backlog --supersede <project>` is no longer refused because a different drive's passed audit has its publish refused (for example by the blur-mask frame gate). That refusal is listed under `publish_refusals` and the supersede goes on. The tick still exits 2 while the refusal stands. Without `--supersede`, such a refusal still holds the region.
