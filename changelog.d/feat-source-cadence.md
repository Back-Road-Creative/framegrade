### Added

- Opt-in source-cadence output (library parameter `output_fps`): a 59.94/60 fps source can keep its own rate through the master encode and the OST burn instead of being decimated to 29.97. The default is unchanged and stays 29.97. Only the source's own rate is accepted (`30000/1001`, `30`, `60000/1001`, `60`); any other value refuses with a clear error. The keyframe interval stays 2 s at the chosen rate, and the `fps` filter stays the chain tail after the privacy-blur composite.
