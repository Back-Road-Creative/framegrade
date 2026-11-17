### Fixed

- `health heal` allowed one resume per failed drive per git SHA, so a resume launched while the staging sync had advanced the checkout but deferred the venv install ran the old code, failed, and was never retried once the install landed. The once-per guard now keys on the SHA plus the installed `gomoveshift-video` version (recorded as `installed_version` on each attempt), and heal skips with `install pending: checkout pins <v>, venv has <v'>` while the pin and the venv differ.
