### Route inventory (`publishing/route_inventory.py`)

`blocked_no_route` (above) said a pass was blocked; it could not say by WHAT. Two gaps, closed:
`route_inventory.KNOWN_PLATFORMS` states, as data, every platform the queue can enqueue against
today (`youtube`, `instagram`, `facebook`, `tiktok` — the parameters `handlers.build_handlers`
registers), and `route_states(handlers) -> {platform: state}` reports each as `ROUTED` (a handler is
registered) or `NOT_ROUTED` (none is), for every known platform whether or not it had due work this
pass. `DrainResult.skipped_no_handler_by_platform` closes the other gap: a per-platform count of
*this pass's* skips, kept exactly consistent with `skipped_no_handler` (its values always sum to
that total).

`NOT_ROUTED` is deliberately not further split. It conflates "nobody has wired this platform yet"
with "this platform was deliberately switched off", and nothing in this repo records that
distinction at the routing layer: no per-platform enabled/disabled field exists in `config.py`'s
models, and `queue.py`'s `platform_pauses` / `is_platform_paused` is a different mechanism entirely
— it governs which due rows `_select_due` selects (drain-loop selection), not which platforms
`build_handlers` wires a handler for (routing); a paused platform can be fully routed, and a routed
platform can be paused. `route_inventory.NOT_ROUTED_CAVEAT` states this in code, and
`tests/test_publishing_route_inventory.py` pins both the two-state shape and the caveat's wording,
so a future third state can't be added silently.
