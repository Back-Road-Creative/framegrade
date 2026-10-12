# Outbound hosts

Every host `gomoveshift/**/*.py` names, and what it is for. See `docs/pipeline/84-network-allowlist-credential-stores.md` for the operative allowlist contract; this file is the generated inventory that keeps it honest.

> **Generated.** This file is rebuilt by `tools/build_outbound_hosts.py` from every host literal it finds in `gomoveshift/**/*.py` (a test, `tests/test_outbound_hosts.py`, fails if this file and the generator disagree). Do not hand-edit it -- add the host to `INVENTORY` in the generator with a reason and re-run it. This is a superset of the enforced upload allowlists in `publishing/*_adapter.py` and `route_story/sources.py`: it also catalogs GET-only and display-text-only hosts that appear in source but never carry media/data out.

| Host | Purpose | Feature | Direction | Mode |
|---|---|---|---|---|
| `accounts.google.com` | OAuth2 consent page the operator's browser opens during `gomoveshift youtube-auth` (the client config built from the shared token names it as auth_uri) | YouTube upload/comment consent flow | media/data out | opt-in |
| `console.cloud.google.com` | Setup hint text printed to the operator (not fetched by this service) | Drive/YouTube credential setup instructions | GET-only | default |
| `copernicus-dem-30m.s3.amazonaws.com` | Copernicus GLO-30 DEM fallback bucket | Route Story offline map bundle | GET-only | default |
| `drive.google.com` | Drive share-link parsing and thumbnail/CSV link building | Shorts/thumbnail hosting behind the Metricool CSV | GET-only | default |
| `elevation.nationalmap.gov` | USGS 3DEP elevation for the trip bounding box | Route Story offline map bundle | GET-only | default |
| `gibs.earthdata.nasa.gov` | NASA GIBS imagery fallback | Route Story offline map bundle | GET-only | default |
| `github.com` | One-time face-detector model fetch, SHA256-verified | Privacy blur | GET-only | default |
| `gomoveshift.com` | Confirms a just-published photo URL is live before handing it to Meta | Photo publishing / brand site | GET-only | default |
| `graph.facebook.com` | Meta Reels create/finish container calls and publishing-limit reads; also a video-bytes upload target | Meta (Facebook/Instagram) Reels publishing | media/data out | opt-in |
| `huggingface.co` | One-time plate/vehicle-occupant detector model fetches, SHA256-verified | Privacy blur | GET-only | default |
| `imagery.nationalmap.gov` | USGS NAIP orthoimagery for the trip bounding box | Route Story offline map bundle | GET-only | default |
| `nominatim.openstreetmap.org` | Reverse geocoding (GPS to place) and forward geocoding (operator location string to coordinates); sends GPS coordinates or the sanitized operator string | Location resolution | media/data out | default |
| `ntfy.sh` | Phone push notifications (TikTok inbox drafts with their caption, failed posts, missed slots); sends the notification title/body incl. captions, no media, no credentials | Phone alerts (`console/notify.py`) | media/data out | opt-in |
| `oauth2.googleapis.com` | OAuth2 token refresh (sends the client id/secret and refresh token) | YouTube upload auth | media/data out | default |
| `open-upload.tiktokapis.com` | TikTok Direct Post video upload target | TikTok publishing | media/data out | opt-in |
| `open-upload.tiktokapis.us` | TikTok Direct Post video upload target, US data-residency namespace | TikTok publishing | media/data out | opt-in |
| `open.tiktokapis.com` | TikTok OAuth token, creator_info, publish/init and publish/status API calls (not the upload byte stream itself) | TikTok publishing | media/data out | opt-in |
| `overpass-api.de` | POI/chapter discovery bbox query (Overpass mirror) | Chapter generation | GET-only | default |
| `overpass.kumi.systems` | POI/chapter discovery bbox query (Overpass canonical host) | Chapter generation | GET-only | default |
| `overpass.private.coffee` | POI/chapter discovery bbox query (Overpass mirror) | Chapter generation | GET-only | default |
| `prd-tnm.s3.amazonaws.com` | USGS GNIS Domestic Names zip, per state | Route Story place-name labels | GET-only | default |
| `registry.npmjs.org` | Pinned MapLibre tarball, SHA512 SRI-verified | Route Story scene rendering | GET-only | default |
| `rupload.facebook.com` | Meta Reels video-bytes upload target | Meta (Facebook/Instagram) Reels publishing | media/data out | opt-in |
| `studio.youtube.com` | Endscreen setup hint text printed to the operator (not fetched) | YouTube endscreen sidecar | GET-only | default |
| `upload.us.tiktokapis.com` | TikTok Direct Post video upload target | TikTok publishing | media/data out | opt-in |
| `www.facebook.com` | Web origin used to build a clickable permalink from Graph's relative `permalink_url` (not fetched) | Meta (Facebook/Instagram) Reels publishing | GET-only | opt-in |
| `www.googleapis.com` | YouTube Data API (upload, metadata, playlists) and Drive API calls | YouTube upload / metrics | media/data out | default |
| `www.tiktok.com` | Web origin used to build a human-facing TikTok permalink (not fetched) | TikTok publishing status | GET-only | opt-in |
| `www.topografix.com` | GPX 1.1 XML namespace URI parsed/emitted in GPX files (not fetched) | GPS track parsing | GET-only | default |
| `youtu.be` | Video permalink text built into descriptions, chapters and CSV rows (not fetched) | YouTube metadata / shorts CSV | GET-only | default |
