### Added

- Add an unwired Facebook Page-photo byte client with explicit publication intent,
  immutable JPEG hash binding and exact Photo/story/Page readback. Unknown writes
  are reported without retrying; readback failures retain known IDs. Public
  requests are tested offline only; production approval and durable queue wiring
  remain separate prerequisites.
