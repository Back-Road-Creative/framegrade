"""Opt-in bearer-token auth for the console. GMS_CONSOLE_TOKEN unset ==
current unauthenticated behavior; set == every route but /healthz requires
the token via header, query param (once, then cookie), or cookie. Also
covers the POST /jobs tab gate for tab-disabled subcommands."""

from fastapi.testclient import TestClient

from gomoveshift.console import auth
from gomoveshift.console.app import create_app


def _client() -> TestClient:
    return TestClient(create_app())


def _https_client() -> TestClient:
    """Same app, but the request scheme reads ``https`` -- Starlette's
    TestClient derives ``request.url.scheme`` from ``base_url``."""
    return TestClient(create_app(), base_url="https://testserver")


def _no_raise_client() -> TestClient:
    """A 500 surfaces as a response instead of a raised exception -- needed
    to observe (and assert against) a crash rather than have pytest report
    it as an error with no assertion."""
    return TestClient(create_app(), raise_server_exceptions=False)


def test_no_token_env_leaves_console_open(monkeypatch):
    monkeypatch.delenv("GMS_CONSOLE_TOKEN", raising=False)
    client = _client()

    assert client.get("/healthz").status_code == 200
    assert client.get("/").status_code == 200
    assert client.get("/jobs").status_code == 200


def test_token_set_blocks_unauthenticated_requests(monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_TOKEN", "s3cret")
    client = _client()

    resp = client.get("/")
    assert resp.status_code == 401
    assert resp.json() == {"detail": "unauthorized"}


def test_token_set_healthz_stays_open(monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_TOKEN", "s3cret")
    client = _client()

    assert client.get("/healthz").status_code == 200


def test_token_set_bearer_header_authenticates(monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_TOKEN", "s3cret")
    client = _client()

    resp = client.get("/", headers={"Authorization": "Bearer s3cret"})
    assert resp.status_code == 200


def test_token_set_wrong_bearer_header_401s(monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_TOKEN", "s3cret")
    client = _client()

    resp = client.get("/", headers={"Authorization": "Bearer nope"})
    assert resp.status_code == 401


def test_query_token_authenticates_and_sets_cookie_for_later_requests(monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_TOKEN", "s3cret")
    client = _client()

    resp = client.get("/?token=s3cret")
    assert resp.status_code == 200
    assert client.cookies.get("gms_console_token") == "s3cret"

    # No header/query on the follow-up request -- the cookie set above
    # (phone-bookmark flow) is enough.
    resp = client.get("/")
    assert resp.status_code == 200


def test_post_jobs_for_tab_disabled_subcommand_is_blocked(shipped_brands, monkeypatch):
    monkeypatch.delenv("GMS_CONSOLE_TOKEN", raising=False)
    monkeypatch.setenv("GMS_CONSOLE_BRAND", "examplebrand")
    client = _client()

    resp = client.post("/jobs", data={"subcommand": "drive"}, follow_redirects=False)
    assert resp.status_code == 404


def test_post_jobs_for_enabled_subcommand_still_works(monkeypatch, tmp_path):
    monkeypatch.delenv("GMS_CONSOLE_TOKEN", raising=False)
    monkeypatch.delenv("GMS_CONSOLE_BRAND", raising=False)
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    client = _client()

    resp = client.post("/jobs", data={"subcommand": "dispatch-queue"}, follow_redirects=False)
    assert resp.status_code == 303


# --- constant-time token compare ------------------------------------------


def test_wrong_bearer_token_is_compared_with_hmac_compare_digest(monkeypatch):
    """A bare ``==``/``!=`` on a secret is a timing oracle. Assert the
    comparison is actually routed through ``hmac.compare_digest`` rather
    than timing it (unreliable in a test)."""
    monkeypatch.setenv("GMS_CONSOLE_TOKEN", "s3cret")
    client = _client()
    calls = []
    original = auth.hmac.compare_digest

    def spy(a, b):
        calls.append((a, b))
        return original(a, b)

    monkeypatch.setattr(auth.hmac, "compare_digest", spy)

    resp = client.get("/", headers={"Authorization": "Bearer nope"})

    assert resp.status_code == 401
    assert calls, "expected the token check to call hmac.compare_digest"


def test_missing_supplied_token_does_not_crash_compare_digest(monkeypatch):
    """``_supplied_token`` can return ``None`` (no header/query/cookie at
    all); the auth check must handle that case rather than passing it
    straight into a comparison that expects a string."""
    monkeypatch.setenv("GMS_CONSOLE_TOKEN", "s3cret")
    client = _client()

    resp = client.get("/")

    assert resp.status_code == 401


def test_non_ascii_query_token_401s_not_500s(monkeypatch):
    """Regression: ``hmac.compare_digest`` on two ``str`` raises
    ``TypeError`` when either side has a non-ASCII character. A supplied
    token is attacker-controlled, so a one-byte non-ASCII query token must
    not turn an unauthenticated request into a 500. ``%C3%A9`` is the
    UTF-8 percent-encoding of ``e-acute`` -- Starlette decodes query
    params as UTF-8, so this arrives as the one-character string 'e-acute'."""
    monkeypatch.setenv("GMS_CONSOLE_TOKEN", "s3cret")
    client = _no_raise_client()

    resp = client.get("/jobs?token=%C3%A9")

    assert resp.status_code == 401


def test_non_ascii_bearer_header_401s_not_500s(monkeypatch):
    """Same regression via the ``Authorization`` header. httpx's own header
    dict rejects a non-ASCII ``str`` value outright (it would never reach
    the server), so the raw wire bytes are passed directly -- ``b"\\xe9"``
    is 'e-acute' in Latin-1, the encoding ASGI servers use to decode HTTP
    header bytes, so the app sees the same non-ASCII string a real
    attacker-controlled header would produce."""
    monkeypatch.setenv("GMS_CONSOLE_TOKEN", "s3cret")
    client = _no_raise_client()

    resp = client.get("/jobs", headers={"Authorization": b"Bearer \xe9"})

    assert resp.status_code == 401


def test_non_ascii_cookie_401s_not_500s(monkeypatch):
    """Same regression via the session cookie -- also sent as raw Latin-1
    wire bytes for the same reason as the header case above."""
    monkeypatch.setenv("GMS_CONSOLE_TOKEN", "s3cret")
    client = _no_raise_client()

    resp = client.get("/jobs", headers={"Cookie": b"gms_console_token=\xe9"})

    assert resp.status_code == 401


# --- ?token= must not survive in the address bar --------------------------


def test_query_token_on_get_redirects_stripping_token_but_keeping_other_params(monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_TOKEN", "s3cret")
    client = _client()

    resp = client.get("/jobs?token=s3cret&foo=bar", follow_redirects=False)

    assert resp.status_code in (302, 303)
    location = resp.headers["location"]
    assert "token=" not in location
    assert "foo=bar" in location
    assert resp.cookies.get("gms_console_token") == "s3cret"


def test_post_with_query_token_still_authenticates_without_redirect(monkeypatch, tmp_path):
    """A POST can't be redirected without either dropping the body or
    silently turning it into a GET, so it keeps the query token in this
    one request/log line but still authenticates and sets the session
    cookie so the next request doesn't need the token at all."""
    monkeypatch.setenv("GMS_CONSOLE_TOKEN", "s3cret")
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    client = _client()

    resp = client.post(
        "/jobs?token=s3cret", data={"subcommand": "dispatch-queue"}, follow_redirects=False
    )

    assert resp.status_code == 303  # the route's own post/redirect/get, not auth's
    assert resp.cookies.get("gms_console_token") == "s3cret"


# --- Secure cookie only over HTTPS -----------------------------------------


def test_cookie_is_secure_over_https(monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_TOKEN", "s3cret")
    client = _https_client()

    resp = client.get("/?token=s3cret", follow_redirects=False)

    assert "secure" in resp.headers["set-cookie"].lower()


def test_cookie_is_not_secure_over_plain_http(monkeypatch):
    """Local operation is plain HTTP on loopback -- a blanket ``Secure``
    would silently break it, so this must stay unset over http."""
    monkeypatch.setenv("GMS_CONSOLE_TOKEN", "s3cret")
    client = _client()

    resp = client.get("/?token=s3cret", follow_redirects=False)

    assert "secure" not in resp.headers["set-cookie"].lower()


# --- /healthz must not leak operational data to unauthenticated callers ----


def test_healthz_unauthenticated_is_minimal_when_token_set(monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_TOKEN", "s3cret")
    client = _client()

    resp = client.get("/healthz")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_healthz_authenticated_still_gets_the_full_body(monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_TOKEN", "s3cret")
    client = _client()

    resp = client.get("/healthz", headers={"Authorization": "Bearer s3cret"})

    assert resp.status_code == 200
    body = resp.json()
    assert "brands" in body
    assert "ticker_alive" in body
    assert "observer_ok" in body
    assert "notify_configured" in body
