### Changed

- Shorts windows that sit inside a measured landmark or destination polygon rank higher. The bonus is soft: it never admits a window below the day's bar or with a weak opening, and with no GPS, no polygon or a failing POI source nothing changes. Each Short records `landmark` in the shorts audit sidecar, outside its hash.
