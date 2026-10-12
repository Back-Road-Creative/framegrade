### Settings + credential status (`console/settings.py`, `GET /settings`)

`GET /settings` renders `settings.registry_status()` (every `GMS_*` row from the registry plus an
`is_set` flag from `os.environ`) and `settings.credential_status()` (presence-only for the four
known SOPS credential sources -- `GMS_DRIVE_CREDENTIALS_ENC`, `GMS_DRIVE_TOKEN_ENC`,
`GMS_AGE_KEY_FILE`, `GMS_SOPS_CONFIG`, resolved to their env override or documented default path).
Both are status-only by construction: neither function ever reads a value onto the returned dict,
so there is nothing for the template to leak, proven by
`tests/test_console_settings_tab.py::test_settings_page_never_leaks_a_secret_looking_env_value`
(sets every registry name and credential env var to a secret-looking string, secret-scans the
rendered HTML).

**Per-brand override form.** The same `GET /settings` page also renders one text input per
registry row, prefilled from `settings.load_brand_settings(current_brand())` -- the current
brand's saved override file -- and **never** from `os.environ`, so a value already `os.environ`
carries is never echoed into the form even when the brand has no override for it.
`POST /settings` accepts only names present in `settings.load_registry()` (a submitted name outside
the registry is a 400); each submitted value is stripped, an empty string clears that override
(stored `null`), and the result is written with `settings.save_brand_settings(current_brand(),
values)` before a 303 redirect back to `/settings`. `settings.brand_launch_env(slug=None)` returns
`os.environ` merged with `slug`'s (default: `current_brand()`) saved overrides -- registry names
only, overrides win -- and every `jobs.launch(...)` call site in `console/app.py` passes
`env=brand_launch_env()` so a console-launched job actually sees the brand's saved settings, not
just the console process's own environment. `console/jobs.py`'s `launch()` forwards `env` straight
to `subprocess.Popen(env=...)`; `env=None` (the default) inherits the parent process's environment,
unchanged from before this landed. `settings.known_brands()` is the union of the shipped template
directories under `console/brands/` and any brand slug that has ever saved an override file under
`console_data_dir()/brands/`, so a brand created only through the Settings form still shows up in
`/healthz`'s brand list.

**YouTube channel identity.** `settings.youtube_identity_status()` returns one row per
`known_brands()` brand: `{"brand", "channel_id", "channel_title"}`. `channel_title` reads
`"not connected"` (and `channel_id` `None`) when that brand has no token, and names the fix --
`gomoveshift youtube-auth --brand <slug>` -- when a token exists without an identity sidecar (a
token minted before identity capture; `doctor` WARNs on the same case). It is offline and
read-only -- `pipeline._yt_creds.read_channel_identity` reads only the plain-JSON sidecar next to
the brand's SOPS token (never the encrypted token itself, never a `sops` call, never a network
call). `GET /settings` passes these rows to `settings.html` as `youtube_rows`, rendered in a
"YouTube channel identity" table.
`tests/test_console_settings_tab.py::test_settings_route_renders_youtube_identity_and_never_leaks_secret`
drives the live route and proves the page shows the channel title and id and never the raw token
contents.
