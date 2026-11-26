### Fixed

- A day whose shorts are re-rendered now removes the earlier plan's `short_NN.mp4` files first. A rank the new plan dropped (or a short whose render failed) was left on disk, counted by phase_d as an unattributed short, and failed the drive with `shorts_per_lf SPLIT-BRAIN` on every retry.
