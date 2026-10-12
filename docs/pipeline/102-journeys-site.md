### Journeys & site (`console/journeys.py`, `GET /journeys`, `POST /journeys/sync-site`,
`POST /journeys/publish-journal-replay`, minimal slice)

`GET /journeys` shells out to `gh pr list --repo <owner/name> --json
number,title,url` (`list_site_prs()`) against the site repository, not this one, so the console
shows PRs against what `sync-site`/`publish-journal` actually write into. The repository slug has
no default: set `GMS_SITE_REPO` to `owner/name`. Unset, the page reports that the site repository
is not configured and names `GMS_SITE_REPO`, and no `gh` call is made. A missing/failing `gh`, non-zero exit, or unparseable
JSON never 500s -- each degrades to an empty PR list plus the error text rendered in the template.

Two launch buttons cover the journal/site CLI surface, both dry-run by default (`--apply` is never
passed): `POST /journeys/sync-site` builds `gomoveshift sync-site --site-path <site_path>` where
`site_path` comes from `journeys.resolve_site_path()` (a re-export of the Media library's
`library.resolve_site_path()`) -- i.e. this brand's own `GMS_SITE_PATH`, never a form field. The
form used to take a free-typed path; that predated the tenant-boundary rule the Media library
later adopted (a brand is a tenant boundary, and a path is not something an operator should be
able to point at another brand's checkout) and never itself adopted it. A form field is
robust-by-construction now: there is no `site_path` input to submit, so no operator-supplied
string can reach argv even by accident. An unset `GMS_SITE_PATH` re-renders the Journeys page
with a 400 and "No site path configured for this brand. Set `GMS_SITE_PATH` in Settings." --
matching the Media library's own unset-config message -- instead of launching or crashing.
`POST /journeys/publish-journal-replay` builds `gomoveshift publish-journal --replay-pending`,
which needs no operator input since `state_path`/`--journey`/`--site-path` are only required
outside replay mode. Both add their subcommand to `console/jobs.py`'s `ALLOWED_SUBCOMMANDS` and
run through the same `jobs.launch()` path as every other tab. A full `publish-journal <state_path>
--journey SLUG` launch is a follow-up, same as the SEO tab's job-launch gap above.
