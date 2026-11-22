### Changed

- Shorts windows that sit inside one measured landmark or destination polygon rank higher, and the bonus never lowers the rendered Short count or trades more than 25% of the score of a landmark window only it picks (checked after the per-LF cap and refill). The bonus is soft: it never admits a window below the day's bar or with a weak opening, and with no GPS, no polygon or a failing POI source nothing changes. Each Short records `landmark` in the shorts audit sidecar, outside its hash.
