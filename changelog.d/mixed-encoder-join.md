### Fixed

- A long-form whose montage held a re-encoded clip no longer fails its OST burn on A/V drift. The montage and prepend joins kept only the first piece's HEVC parameter sets, so every frame after the first re-encoded piece was undecodable while the file still reported its full length (Galley Head failed seven heals on 2026-11-25). Pieces with different parameter sets are now joined losslessly through Annex B, so every frame decodes.
