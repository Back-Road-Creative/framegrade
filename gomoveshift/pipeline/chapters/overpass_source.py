"""Compatibility alias: the Overpass source lives engine-side.

``pipeline.overpass_source`` is the implementation. This path aliases the *same
module object* (not a copy of its names) so existing imports and test
monkeypatches of module globals (``post_overpass_query``, ``urllib``,
``_last_good_endpoint``) keep reaching the code that uses them.
"""

import sys

from .. import overpass_source as _engine

sys.modules[__name__] = _engine
