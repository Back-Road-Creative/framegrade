"""States the publishing queue's route table as data.

Before this module, the only way to know which platforms a drain pass CAN
route to was to inspect the keys of whatever dict ``build_handlers``
(``handlers.py``) happened to return -- there was no statement anywhere of
the full set of platforms the queue is meant to support, so a
``skipped_no_handler`` count could never be turned into "these platforms, in
these states". ``KNOWN_PLATFORMS`` and :func:`route_states` fix that.

``KNOWN_PLATFORMS`` is derived from what actually exists in this repo: the
four platforms ``build_handlers`` can register a handler for
(``handlers.py``'s ``youtube``/``instagram``/``facebook``/``tiktok``
parameters), each backed by its own adapter module's ``PLATFORM`` constant
(``youtube_adapter.py``, ``instagram_adapter.py``, ``facebook_adapter.py``;
TikTok has no such constant but is wired the same way via
``tiktok_adapter.py``/``tiktok_auth.py``). A platform string that reaches the
queue outside this set (e.g. a typo, or a not-yet-supported platform) is not
part of the route table this module states -- it is neither ``ROUTED`` nor
``NOT_ROUTED``, it simply isn't inventoried, the same way ``route_states``
silently ignores an unrecognised key in ``handlers``.
"""

from __future__ import annotations

from typing import Dict, Mapping

#: Every platform the queue can enqueue a delivery against today. Keep this
#: in lockstep with ``handlers.build_handlers``'s parameters -- if a platform
#: is added there without being added here, this inventory undercounts it.
KNOWN_PLATFORMS = ("youtube", "instagram", "facebook", "tiktok")

#: A handler is registered for this platform right now.
ROUTED = "routed"

#: No handler is registered for this platform right now. See
#: ``NOT_ROUTED_CAVEAT`` -- this does NOT mean the platform was deliberately
#: switched off.
NOT_ROUTED = "not_routed"

#: Exactly two states. Deliberately not three: see ``NOT_ROUTED_CAVEAT``.
#: Adding a state here means updating this tuple, which
#: ``tests/test_publishing_route_inventory.py::test_only_two_route_states_exist``
#: pins -- so it can't happen silently.
ROUTE_STATES = (ROUTED, NOT_ROUTED)

#: ``NOT_ROUTED`` conflates two genuinely different situations -- a platform
#: nobody has wired yet, and a platform deliberately switched off -- and this
#: module cannot tell them apart. Checked before writing this: no per-platform
#: enabled/disabled field exists in ``config.py``'s models, and
#: ``queue.py``'s ``platform_pauses`` table / ``is_platform_paused`` is a
#: DIFFERENT mechanism -- it governs which due rows ``_select_due`` picks up
#: (drain-loop SELECTION), not which platforms ``build_handlers`` wires a
#: handler for (ROUTING); a paused platform can still be fully routed, and a
#: routed platform can still be paused. No source in this repo records
#: "this platform is not wired on purpose" at the routing layer. Do not
#: invent one here -- which platforms are intentionally off is a product
#: decision, not this module's to make.
NOT_ROUTED_CAVEAT = (
    "not_routed means no handler is registered for this platform right now. "
    "It does not distinguish a platform that is simply not yet wired from one "
    "that was deliberately switched off/disabled -- no record of that intent "
    "exists anywhere in this repo (checked: no per-platform enabled/disabled "
    "config field, and queue.py's platform_pauses governs drain-loop "
    "selection, not routing). Do not infer disablement from not_routed."
)


def route_states(handlers: Mapping[str, object]) -> Dict[str, str]:
    """For each platform in ``KNOWN_PLATFORMS``, ``ROUTED`` if ``handlers``
    has a (non-``None``) entry for it, else ``NOT_ROUTED``. Platforms outside
    ``KNOWN_PLATFORMS`` present in ``handlers`` are not part of the returned
    inventory -- see the module docstring."""
    return {
        platform: ROUTED if handlers.get(platform) is not None else NOT_ROUTED
        for platform in KNOWN_PLATFORMS
    }
