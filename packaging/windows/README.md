# Windows packaging

| File | Role |
| --- | --- |
| `build_release.py` | Stages the embeddable Python 3.12, the win_amd64 wheels and the git-tracked app source into `dist/stage/`, audits it, size-gates the outputs, then runs `ISCC`. |
| `framegrade.cmd` | Launcher copied to the stage root: sets `GMS_DATA_HOME` from `data-folder.txt`, starts the host worker and the console, opens the browser. |
| `framegrade.iss` | Inno Setup script: per-user install, data-folder wizard page, finish-page GPU and ffmpeg notes. |

`--skip-installer` stops before Inno Setup (it only exists on Windows); `--zip`
also writes a portable zip. The size gate runs over whatever was produced.
Install steps for users are in `docs/install.md`.
