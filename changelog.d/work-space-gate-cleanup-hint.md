### Fixed

- The LF work-space gate's "free space" hint now names the stage root it reports (`GMS_INTERMEDIATE_DIR=<root> gomoveshift cleanup …`) and the `--abandoned` form for a killed or given-up run. The bare `gomoveshift cleanup <project>` it printed searched the default stage cache, so a drive staged by `health backlog` under another root answered `Project not found`.
