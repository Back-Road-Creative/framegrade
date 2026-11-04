### Added

- An Irish county claim ("County Cork") on a route with GPS gaps is proven on Tailte Éireann's pinned 2026 statutory county map (CC BY 4.0, © Tailte Éireann) instead of refused for want of a US Census boundary. The map is downloaded only over https from the three ArcGIS hosts in `docs/pipeline/84-network-allowlist-credential-stores.md`, checked against its SHA-256 and cached. Every fix must sit at least 250 m inside the county beyond the speed and error bound, which covers a recording made before the 2026 map. `BOUNDARY_REVIEW_VERSION` is `tiger2026-tailte2026-bounded-v2` and `PHASE_BC_CODE_VERSION` is 4, so a resumed drive rebuilds its route record with the proof.

### Changed

- Pinned Waypost (`seo-manager`) to v0.5.0 (Back-Road-Creative/waypost#106), whose final-text review accepts the Tailte Éireann proof.
