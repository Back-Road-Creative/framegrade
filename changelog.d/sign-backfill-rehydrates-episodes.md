### Fixed

- The resumed-Phase-A sign back-fill now reads the footage. The checkpointed ingest plan is a shallow rehydrate whose `episodes` are the serialized dicts, so `group_episodes_by_day` dropped every one and the pass logged "0 hit(s) across 0 day(s)" in under a second without decoding a frame (seen on `nctopa-2026-03-05-rec0063`). The back-fill resolves the plan the way the day phase does, from the parallel state arrays, before the pass samples it.
