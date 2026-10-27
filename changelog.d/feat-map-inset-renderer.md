### Added

- `gomoveshift/pipeline/map_inset.py`: renders an opt-in, route-only corner map inset as an RGBA `.mov` overlay plus a JSON receipt, from an offline Route Story bundle (imagery with the recorded route and a position marker, no road vectors, no network). The marker follows the source clock through the edit timeline and hides during cuts, insert holds, crossfades and GPS gaps. Privacy trim and zones apply before any framing or drawing, and the render refuses, listing every reason, rather than write a partial clip.
