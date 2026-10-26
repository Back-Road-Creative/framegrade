# Windows packaging

| File | Role |
| --- | --- |
| `build_release.py` | Stages the embeddable Python 3.12, the win_amd64 wheels and the git-tracked app source into `dist/stage/`, audits it, size-gates the outputs, then runs `ISCC`. |
| `framegrade.cmd` | Launcher copied to the stage root: sets `GMS_DATA_HOME` from `data-folder.txt`, starts the host worker and the console, opens the browser. |
| `framegrade.iss` | Inno Setup script: per-user install, data-folder wizard page, finish-page GPU and ffmpeg notes. |

`--skip-installer` stops before Inno Setup (it only exists on Windows); `--zip`
also writes a portable zip. The size gate runs over whatever was produced.
Install steps for users are in `docs/install.md`.

## Cutting a Windows release

`.github/workflows/windows-release.yml` builds, proves and drafts the release on
GitHub-hosted `windows-2025` runners. It needs no secrets and no GPU.

1. Cut the version as usual (`README.md`, Releasing): `tools/release.py`, the
   `chore(release)` PR, then tag the merge commit `vX.Y.Z` and push the tag.
2. Wait for main's `ci` run on that commit to pass. The workflow's `gate` job
   (`release_gate.py gate`) refuses the tag unless the commit is on `main`, the tag
   equals the `pyproject.toml` version and `ci.yml` has a successful run for that
   exact commit. If CI had not finished, re-run the workflow once it has.
3. `build` stages the bundle, compiles the installer (Inno Setup 6.7.1 is on the
   image; the job installs that version if an image update drops it), size-gates
   every asset against GitHub's 2 GiB limit and writes `SHA256SUMS`
   (`<sha256>  <filename>`, checked with `sha256sum -c`).
4. `proof` runs `windows_proof.ps1` against the built installer: a silent per-user
   install as a standard user into `C:\pröof dir\Framegrade ✓`, a Defender scan, the
   console started through `framegrade.cmd` and answering `/healthz`, `gomoveshift
   scan` on a synthetic clip made with the bundled OpenCV, then uninstall (the data
   folder must survive). Nothing in it needs a GPU, ffmpeg or a network.
5. `release` (tag runs only, the only job with `contents: write`) creates a
   **draft** release holding the installer and `SHA256SUMS`. Review it and publish
   it by hand. A release that already exists for the tag makes the job fail rather
   than being overwritten.

### Dry run

Actions, `windows-release`, Run workflow (or `gh workflow run windows-release
--ref <branch>`). It runs the same build and proof on any ref, skips the tag gate,
creates no release and leaves `framegrade-windows-X.Y.Z` (installer and
`SHA256SUMS`) as a workflow artifact for 14 days. Run it after any change to
`build_release.py`, `framegrade.iss`, `framegrade.cmd` or the proof.

To run the proof by hand on a Windows machine, as administrator:
`./packaging/windows/windows_proof.ps1 -Installer path\to\framegrade-setup-X.Y.Z.exe`.
