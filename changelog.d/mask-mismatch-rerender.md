### Fixed

- `health backlog` now re-renders a drive whose blur mask was built on another frame count than its source (`mask_frame_count_mismatch`), once per drive-day per renderer version, recorded as `rerender_requested` on the gate's incident row. A count that cannot be read (`mask_frame_count_unknown`) is still never re-rendered, and a re-render the supersede refuses leaves the tick as it was. Before, a mismatched day stood refused every tick until a person superseded it (Kerry-to-Cork, 2026-11-27, #662).
