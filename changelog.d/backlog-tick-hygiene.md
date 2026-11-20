### Fixed

- The hourly `health backlog` tick logs `RETIRED_CONFIG_KEY` once per process per key instead of on every config load, and reads the framegrade release list (`git ls-remote --tags`) at most once a day: the answer is cached in `<data root>/release-latest.json` for 24 h, reused if the network read fails (up to 7 days old), and with no cache a failure still refuses the launch as before.
