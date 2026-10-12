## Network allowlist & credential stores (implementation)

The contract line is "outbound HTTPS only to the allowlisted hosts, no inbound exposure." The concrete list:

- **googleapis.com** — YouTube Data API (upload, metadata, playlists) and Drive API (shorts/thumbnail hosting behind the Metricool CSV links).
- **accounts.google.com** — OAuth2 refresh only.
- **nominatim.openstreetmap.org** — reverse geocoding (GPS → place) **and** forward/search geocoding (operator location string → coordinates). Operator strings are sanitized before the query so a pasted label can't smuggle query syntax.
- **ntfy.sh** — phone push notifications (`console/notify.py`, host constant `NTFY_HOST`; set up with `tools/phone_alerts_setup.py`). **Data goes out**: a notification title and body — the slot time, project and short, a failed post's error text with token-looking strings redacted and cut near 300 characters, and, for a TikTok inbox draft, the caption the operator must paste. No media and no credentials leave. The topic in `notify_url` under the config home (`$GMS_CONFIG_HOME`, else `$XDG_CONFIG_HOME/gomoveshift`, else `~/.config/gomoveshift`; mode 0600) is the only secret: anyone who knows it can read the pushes, so it is never logged or printed outside the setup tool's own terminal. Opt-in: nothing is sent until that file or `GMS_CONSOLE_NOTIFY_URL` names an ntfy.sh URL. Intended for pipeline phone pushes only.
- **overpass.kumi.systems**, **overpass-api.de**, **overpass.private.coffee** — POI / chapter discovery along the route. Three interchangeable Overpass instances answering the identical query, tried in `OVERPASS_ENDPOINTS` order; the two mirrors are fallbacks for when the canonical host refuses a client's connection while a mirror serves the same query. Same data, same shape, no new *kind* of egress — the bbox query is the only thing sent, exactly as before. **Rank endpoints on a day-sized bbox, never a toy one**: a tiny probe bbox can put these in almost the reverse of their real order (an instance that looks fastest on a toy query may read-time-out on a real day-sized query while another answers it quickly).
- **huggingface.co** and **github.com** — one-time privacy-blur model fetches: plate and vehicle-occupant detectors from huggingface.co, the face detector from github.com. GET only, **no data egress**, SHA256-verified, cached locally so the fetch happens once per host. Point each `*_model_url` at an internal mirror, or clear it, to avoid the host entirely.
- **gomoveshift.com** — the brand's own public Hugo site (Cloudflare Pages). GET only, **no data egress**, first-party: the photo publisher fetches a photo's own live URL there to confirm it is actually reachable before handing that URL to Meta's Graph API as `image_url`.
- **graph.facebook.com** and **rupload.facebook.com** — Meta Reels publishing (`publishing/facebook_adapter.py`, `publishing/instagram_adapter.py`), each declaring this exact pair as its own `ALLOWED_UPLOAD_HOSTS`, checked before any bytes are sent. **This is a different and larger kind of egress than every GET-only entry above**: these two hosts receive an upload of the real video bytes plus the caption/title text being published, not a read of a model file or a map query. `graph.facebook.com` also serves the create/finish container calls and the publishing-limit read. `docs/social-publishing-status.md` records the live-versus-fixture evidence for this path.
- **upload.us.tiktokapis.com**, **open-upload.tiktokapis.com**, **open-upload.tiktokapis.us** — TikTok Direct Post video upload (`publishing/tiktok_adapter.py`'s own `ALLOWED_UPLOAD_HOSTS`, exact-host matched before any bytes are sent — never a suffix match). Same larger kind of egress as the Meta pair above: real video bytes, not a GET. `open-upload.tiktokapis.us` is TikTok's US data-residency namespace, cert-verified before being added to the adapter's allowlist.
- **open.tiktokapis.com** — the TikTok API itself, as opposed to the upload hosts above. It serves the OAuth token exchange (`publishing/tiktok_auth.py`), creator-info, publish init and status (`publishing/tiktok_adapter.py`), and the video metrics query (`tracking/collector.py`). **Data goes out**: credentials on the token call, and the post title/caption text on publish init. No media bytes go here; those go to the upload hosts.
- **elevation.nationalmap.gov**, **imagery.nationalmap.gov**, **gibs.earthdata.nasa.gov**,
  **copernicus-dem-30m.s3.amazonaws.com**, **registry.npmjs.org** — Route Story's offline map
  bundle: USGS 3DEP elevation and USGS NAIP orthoimagery for the trip's own bounding box, NASA GIBS
  as the imagery fallback, the Copernicus GLO-30 bucket as the DEM fallback, and the pinned
  MapLibre tarball from the npm registry (SHA512 SRI-verified before it is unpacked). **GET only,
  no data egress**: each request carries a bounding box, a cell name or a version string and
  nothing else — no media, no operator strings, no PHI path. Every one is fetched once per
  content-addressed bundle and reused from cache afterwards. `tests/test_network_allowlist_doc_sync.py` asserts every host in
  `route_story.sources.SOURCE_HOSTS` appears here.
- **prd-tnm.s3.amazonaws.com** — Route Story's GNIS place-names label source
  (`route_story/gnis.py`): the per-state Domestic Names zip from USGS The National
  Map's staged products (`StagedProducts/GeographicNames/DomesticNames`), public
  domain, ~1 MB per state. **GET only, no data egress**: each request names a state's
  file, nothing else. Fetched once per state and cached thereafter — a second request
  for the same state is served from `cache_dir/gnis/<ST>/` without a network call.
- **Inbound**: none. `gomoveshift server` binds `127.0.0.1` only.

Widening this list is a sign-off decision, not a code change. A stale allowlist doc is worse than an absent one, because it is trusted: `tests/test_network_allowlist_doc_sync.py` asserts every host any adapter's `ALLOWED_UPLOAD_HOSTS` declares appears in this section, so a host cannot be added in code without appearing here.

**Credential stores.** Two kinds of encrypted credential file can exist, and conflating them produces false "MISSING" credential alarms:

| Store | Location | Access | Holds |
|---|---|---|---|
| Encrypted JSON files (pipeline primary) | the parent directory of `config.seo.credentials_dir` | plain `sops -d` | `youtube_upload_token.json.enc`, `meta_publishing.json.enc`, `tiktok_publishing.json.enc` |
| Optional dotenv-style secrets file for operator tooling (not read by the pipeline directly) | wherever the deployment keeps it | the deployment's own sops wrapper | tool secrets |

Before recommending re-encryption, rule out a wrong path lookup and a sandbox permission error — only an actually-broken keyring warrants it. A `.sops.yaml` beside the credential files whose `creation_rules` cover `youtube_(credentials|upload_token)`, `meta_publishing` and `tiktok_publishing` JSON lets one encrypt with `sops --config <that file> -e`; a file outside those patterns needs explicit `--age` recipients. The pipeline's own writer (`sops_encrypt_json`) makes that automatic: re-encrypting an existing `.enc` reuses the recipients already in its public `sops.age` metadata, and a first brand token mint takes the shared token's recipients — so a re-mint or OAuth refresh never changes who can read the file, whichever creation rule the pipeline's `GMS_SOPS_CONFIG` would otherwise match. Only a brand-new file with no explicit list falls through to the config rules, and a catch-all rule there can produce a file the local keyring cannot read.

**Brand-scoped credentials (mechanism only — accounts still TBD).** `publishing/credential_paths.py` resolves a `brand` argument on `load_meta_credentials`, `load_tiktok_credentials`, `get_valid_tiktok_credentials` and `pipeline._yt_creds.load_youtube_credentials` to `<seo.credentials_dir>/brands/<brand>/creds` when that directory exists, so a brand's `.enc` files live at `<seo.credentials_dir>/brands/<brand>/<filename>` (the same `creds_dir.parent / filename` convention the shared store already used). `brand=None`, or a brand with no such directory yet, is byte-for-byte today's shared path — there is still exactly one account (`gomoveshift`) until an operator provisions a second brand's directory. Which accounts get their own credentials is an operator decision; this only ships the resolution mechanism. The console/CLI seam that already threads `console.settings.current_brand()` (`cli/handlers/metrics.py`) passes it into `tracking.collector.MetricsCollector(brand=...)`, which threads it to every platform sub-collector; CLI paths with no brand context (`dispatch-queue`, the drive pipeline's own upload/scheduling calls) keep `brand=None`.

**Social publishing credentials.** No setup helper ships in this tree: re-create the Meta and TikTok publishing credential records by hand, following "Reconnecting" in `docs/social-publishing-meta.md` and `docs/social-publishing-tiktok.md`. Any helper written for that should prompt for secrets with `getpass` (never echoed, never logged), call only read-only platform endpoints, write the plaintext to a 0600 temp file that is deleted, and print only non-secret identifiers. They need an interactive TTY — run them from a real terminal, not from a tool runner. Replacing an existing `.enc` should be an explicit act: delete it first, then write the new record.

**YouTube upload+comment consent CLI.** `gomoveshift youtube-auth [--brand B]` (`cli/handlers/youtube_auth.py`) runs the browser OAuth consent flow and mints `youtube_upload_token.json.enc` with `youtube.upload`, `youtube.force-ssl` and `yt-analytics.readonly` (the analytics read scope is what `tracking/collector.py` needs for YouTube Analytics; force-ssl is what first-comment posting needs; without it `publishing/youtube_comments.py` records a 403 and never posts). It writes through the same locked, atomic `sops_encrypt_json` every other credential write uses — never a new SOPS write path. Two operator runs:

- `gomoveshift youtube-auth` — re-mints the shared GMS channel token with force-ssl before the first comment posts. Run once; the resulting token covers upload + comments going forward. Run it from the project directory (`cd <project dir> && .venv/bin/gomoveshift youtube-auth`): `config.yaml` is read from the launch directory and then overlaid by `<config home>/config.yaml` (config home as above); relative credential paths resolve against the launch directory, so without a config the command resolves them against the cwd — the command then exits 2 naming that directory and the `cd` line.
- `gomoveshift youtube-auth --brand <brand>` — mints a second brand channel's own token at `<seo.credentials_dir>/brands/<brand>/youtube_upload_token.json.enc`, following the brand-scoped resolution above.

OAuth client source: the downloaded `client_secret.json` at `config.seo.manager_path` when present; otherwise the flow reuses the `client_id`/`client_secret` the shared `youtube_upload_token.json.enc` was minted with (in memory via `InstalledAppFlow.from_client_config`, never a plaintext file). A brand mint always borrows the *shared* token's client — one Google Cloud OAuth client, different channel. Only with neither source does it exit 2 with the console download hint.

`--dry-run` prints the target path and scopes without opening a browser. **The browser consent step is a human act** — it needs an interactive TTY with a display/browser, run it from a real terminal, not from a tool runner or CI.

**Social publishing status.** `docs/social-publishing-status.md` is the authoritative per-platform table (credential source, live-vs-fixture evidence, scopes/app-review, quota, go-live gaps) ahead of a Metricool exit. `gomoveshift publishing-status` (`gomoveshift/cli/handlers/pub_status.py`) prints a live, no-decrypt snapshot of the same credential files' presence and the delivery queue's row counts by state.

**Test-side egress guard (`tests/conftest.py::block_non_loopback_network`).** The allowlist above
governs production; tests get a separate, structural backstop against reaching any of it for real.
An autouse fixture patches `socket.socket.connect`/`connect_ex` and `socket.create_connection` for
every test, raising a `RuntimeError` naming the blocked host on any non-loopback address —
loopback (`127.0.0.0/8`, `::1`, `localhost`) and `AF_UNIX` stay untouched. This exists because
per-endpoint stubs (`hermetic_overpass`, `hermetic_forward_geocode`, etc.) only cover the call
sites a human thought to patch: `generator.py`'s coastline query reaches
`overpass_source.post_overpass_query` directly, a call site `hermetic_overpass` (which only patches
`_query_overpass`) never saw, so `tests/test_chapters.py` made real Overpass HTTPS
calls on every test reaching terrain-band classification. The socket-layer guard closes that class
of gap regardless of which higher-level function a future caller goes through. Opt out with
`@pytest.mark.allow_network` for a genuinely intentional live test; the existing live-network
opt-in markers (`real_forward_geocode`, `real_overpass`, `real_blur_models`) imply it, so opting
into one endpoint's real behavior doesn't also require knowing about this guard. Tests:
`tests/test_network_guard.py`.
