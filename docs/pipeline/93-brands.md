### Brands (`console/brands/<slug>/`, `GMS_CONSOLE_BRAND`)

The same image can run more than one console instance, one per brand. No brand directory ships by default. `GMS_CONSOLE_BRAND` (default `gomoveshift`) picks the
template dir under `console/brands/` (supplied by the deployment); `settings.known_brands()` lists every dir there, and
`create_app()` reads it once at startup to decide which nav tabs are enabled. A brand opts a tab
out via a sibling `brand.yaml` (`disabled_tabs: [drives, shorts]`) next to its `settings.yaml` --
missing/empty means every tab is enabled, so `gomoveshift` (no `brand.yaml`) is unaffected.
`TabGateMiddleware` (`console/tabs.py`) 404s every route under a disabled tab structurally, keyed
off the request path's first segment -- not per route: a handful of routes (photos, seo, comments,
journeys, settings) never called the older per-handler `_require_tab(slug)` guard, so a direct
GET/POST reached them even when the brand hid the nav link. The per-handler calls are left in place
(now redundant, not wrong) rather than removed. A brand that has no drive pipeline can, for example, keep
photos/SEO/journeys/comments/settings/jobs on and turn drives/shorts off.
