## Console (`gomoveshift/console/`, `gomoveshift console`)

FastAPI + Jinja skeleton (`console/app.py`): `GET /healthz`, an index listing the eight planned
tabs, `Not built yet` per tab (superseded by the overview screen below). `console/settings.py`
mirrors README's `GMS_*` table into
`registry/gms_settings.yaml` (drift fails `tests/test_console_settings.py`) and reads/writes
`brands/<slug>/settings.yaml` under `GMS_CONSOLE_DATA_DIR` (default `<GMS_STAGING_DIR>/_console`).
See README's "Console" section for the fronting rule.
