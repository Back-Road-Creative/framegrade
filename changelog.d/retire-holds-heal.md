### Fixed

- A drive that `health backlog` retires is no longer re-resumed by `health heal` on every new mirror SHA. Before this, the dead 03-04 audit was resumed three times (jobs ba3a9dbf, 15c1c9b0, 1affffde), and each run held the heavy slot for about 35 minutes while the backlog waited. Retiring now writes the `no-auto-resume` hold beside each of the drive's resume recipes, and heal already refuses a held recipe. A drive already `failed` in the store with no backlog row gets the same hold on the next run, and the result lists each held stage dir under `held`.
