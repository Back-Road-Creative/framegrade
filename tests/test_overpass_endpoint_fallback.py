"""Multi-endpoint Overpass failover, honest failure reporting, and 429 backoff.

Diagnosed on a long run that logged ``Errno 101
Network is unreachable`` for hours against an endpoint that was neither down
nor rate-limiting: both its IPv4 addresses refused the connection, and
``socket.create_connection`` re-raises only its LAST address error — an IPv6
one, because this host has no IPv6 route. The surviving message described a
real condition that was not the cause, and cost two debugging sessions.

Hence the logging assertions: every endpoint's failure must reach the log and
the final exception, since the one that surfaces last is not the one that
explains the failure. A refusal also fails over immediately instead of burning
retries and sleeps on a host that already sent a RST.

Every test patches ``urllib.request.urlopen`` and so reaches the real
``_query_overpass``; that bypasses the module-attr lookup the conftest
``hermetic_overpass`` stub replaces, hence ``@pytest.mark.real_overpass``.
"""

import email.message
import errno
import json
import urllib.error
from unittest.mock import patch

import pytest

from gomoveshift.pipeline.chapters import overpass_source as ops
from gomoveshift.pipeline.chapters.overpass_source import _query_overpass

pytestmark = pytest.mark.real_overpass


@pytest.fixture(autouse=True)
def _reset_sticky_endpoint():
    """The sticky last-good endpoint is process-lifetime state — never leak
    it between tests. Reset before AND after so a failure mid-test can't
    poison a later one either."""
    ops.reset_sticky_overpass_endpoint()
    yield
    ops.reset_sticky_overpass_endpoint()


BBOX = (35.60, -82.60, 35.65, -82.55)
_RESPONSE = {"elements": [{"type": "node", "id": 1, "tags": {"name": "Asheville"}}]}


class _FakeResponse:
    """Minimal ``urlopen`` context-manager stand-in."""

    status = 200

    def __init__(self, payload: dict):
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _refused() -> urllib.error.URLError:
    """What an Overpass host that RSTs the SYN actually raises through urllib."""
    return urllib.error.URLError(ConnectionRefusedError(errno.ECONNREFUSED, "Connection refused"))


def _unreachable() -> urllib.error.URLError:
    return urllib.error.URLError(OSError(errno.ENETUNREACH, "Network is unreachable"))


def _rate_limited(retry_after: str | None) -> urllib.error.HTTPError:
    hdrs = email.message.Message()
    if retry_after is not None:
        hdrs["Retry-After"] = retry_after
    return urllib.error.HTTPError(
        "https://example.invalid/api/interpreter", 429, "Too Many Requests", hdrs, None
    )


def _responder(behaviour: dict):
    """urlopen stand-in dispatching on endpoint URL: value is an exception to
    raise or a payload to return; a list is consumed one entry per call."""
    calls: list[str] = []

    def _fake_urlopen(req, timeout=None):
        url = req.full_url
        calls.append(url)
        outcome = behaviour.get(url, _refused())
        if isinstance(outcome, list):
            outcome = outcome.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return _FakeResponse(outcome)

    return _fake_urlopen, calls


class TestEndpointFallback:
    def test_primary_endpoint_is_tried_first(self):
        primary = ops.OVERPASS_ENDPOINTS[0]
        fake, calls = _responder({primary: _RESPONSE})
        with patch("urllib.request.urlopen", side_effect=fake):
            assert _query_overpass(BBOX) == _RESPONSE
        assert calls == [primary], f"expected only the primary to be queried, got {calls}"

    def test_second_endpoint_answers_when_the_first_refuses(self):
        first, second = ops.OVERPASS_ENDPOINTS[0], ops.OVERPASS_ENDPOINTS[1]
        fake, calls = _responder({first: _refused(), second: _RESPONSE})
        with patch("urllib.request.urlopen", side_effect=fake), patch("time.sleep"):
            assert _query_overpass(BBOX) == _RESPONSE
        assert second in calls, f"never failed over to {second}: {calls}"

    def test_a_refused_endpoint_is_not_retried_in_place(self):
        """A RST is a verdict, not a hiccup — fail over rather than sleep on it."""
        first, second = ops.OVERPASS_ENDPOINTS[0], ops.OVERPASS_ENDPOINTS[1]
        fake, calls = _responder({first: _refused(), second: _RESPONSE})
        with patch("urllib.request.urlopen", side_effect=fake), patch("time.sleep"):
            _query_overpass(BBOX)
        assert calls.count(first) == 1, f"refused endpoint retried in place: {calls}"

    def test_every_endpoint_is_tried_before_giving_up(self):
        fake, calls = _responder({url: _refused() for url in ops.OVERPASS_ENDPOINTS})
        with patch("urllib.request.urlopen", side_effect=fake), patch("time.sleep"):
            with pytest.raises(RuntimeError):
                _query_overpass(BBOX)
        for url in ops.OVERPASS_ENDPOINTS:
            assert url in calls, f"{url} was never tried: {calls}"


class TestHonestFailureReporting:
    def test_the_raised_error_names_every_endpoint_that_failed(self):
        """The old message named one endpoint and one errno — both misleading."""
        behaviour = dict.fromkeys(ops.OVERPASS_ENDPOINTS, _refused())
        behaviour[ops.OVERPASS_ENDPOINTS[-1]] = _unreachable()
        fake, _ = _responder(behaviour)
        with patch("urllib.request.urlopen", side_effect=fake), patch("time.sleep"):
            with pytest.raises(RuntimeError) as excinfo:
                _query_overpass(BBOX)
        message = str(excinfo.value)
        for url in ops.OVERPASS_ENDPOINTS:
            assert url in message, f"{url} missing from the failure message: {message}"

    def test_a_masked_earlier_failure_still_reaches_the_log(self, caplog):
        """The refusal must not be hidden by the unreachable error that follows."""
        behaviour = dict.fromkeys(ops.OVERPASS_ENDPOINTS, _unreachable())
        behaviour[ops.OVERPASS_ENDPOINTS[0]] = _refused()
        fake, _ = _responder(behaviour)
        with caplog.at_level("ERROR"), patch("time.sleep"):
            with patch("urllib.request.urlopen", side_effect=fake):
                with pytest.raises(RuntimeError):
                    _query_overpass(BBOX)
        assert "Connection refused" in caplog.text
        assert "Network is unreachable" in caplog.text


class TestStickyLastGoodEndpoint:
    """A dead head endpoint must not re-pay the full failover budget every
    call — remember the endpoint that last answered and try it first."""

    def test_next_call_tries_the_last_good_endpoint_first(self):
        first, second = ops.OVERPASS_ENDPOINTS[0], ops.OVERPASS_ENDPOINTS[1]
        fake, calls = _responder({first: _refused(), second: _RESPONSE})
        with patch("urllib.request.urlopen", side_effect=fake), patch("time.sleep"):
            _query_overpass(BBOX)  # endpoint[0] dead, endpoint[1] answers

        calls.clear()
        fake2, calls2 = _responder({second: _RESPONSE})
        with patch("urllib.request.urlopen", side_effect=fake2), patch("time.sleep"):
            assert _query_overpass(BBOX) == _RESPONSE
        assert calls2 == [second], f"expected only the sticky endpoint, got {calls2}"

    def test_sticky_endpoint_failure_falls_back_to_full_rotation(self):
        first, second, third = ops.OVERPASS_ENDPOINTS
        fake, _calls = _responder({first: _refused(), second: _RESPONSE})
        with patch("urllib.request.urlopen", side_effect=fake), patch("time.sleep"):
            _query_overpass(BBOX)  # sticky becomes `second`

        fake2, calls2 = _responder({second: _refused(), first: _refused(), third: _RESPONSE})
        with patch("urllib.request.urlopen", side_effect=fake2), patch("time.sleep"):
            assert _query_overpass(BBOX) == _RESPONSE
        assert calls2 == [second, first, third], (
            f"expected sticky first, then the rest in declared order, got {calls2}"
        )


class TestOverpassUrlRemoved:
    def test_chapters_package_no_longer_exposes_overpass_url(self):
        import gomoveshift.pipeline.chapters as chapters

        assert not hasattr(chapters, "OVERPASS_URL")
        assert "OVERPASS_URL" not in chapters.__all__
        assert not hasattr(ops, "OVERPASS_URL")


class TestRateLimitBackoff:
    def test_retry_after_header_sets_the_sleep(self):
        primary = ops.OVERPASS_ENDPOINTS[0]
        fake, _ = _responder({primary: [_rate_limited("7"), _RESPONSE]})
        with patch("urllib.request.urlopen", side_effect=fake), patch("time.sleep") as slept:
            assert _query_overpass(BBOX) == _RESPONSE
        assert 7 in [call.args[0] for call in slept.call_args_list], (
            f"Retry-After: 7 ignored; slept {slept.call_args_list}"
        )

    def test_absurd_retry_after_is_capped(self):
        """A mirror asking for an hour must not stall a multi-hour encode."""
        primary = ops.OVERPASS_ENDPOINTS[0]
        fake, _ = _responder({primary: [_rate_limited("86400"), _RESPONSE]})
        with patch("urllib.request.urlopen", side_effect=fake), patch("time.sleep") as slept:
            _query_overpass(BBOX)
        for call in slept.call_args_list:
            assert call.args[0] <= ops.OVERPASS_RETRY_MAX_SLEEP, (
                f"slept {call.args[0]}s, over the {ops.OVERPASS_RETRY_MAX_SLEEP}s cap"
            )

    def test_429_without_a_header_still_backs_off(self):
        primary = ops.OVERPASS_ENDPOINTS[0]
        fake, _ = _responder({primary: [_rate_limited(None), _RESPONSE]})
        with patch("urllib.request.urlopen", side_effect=fake), patch("time.sleep") as slept:
            assert _query_overpass(BBOX) == _RESPONSE
        assert slept.called, "a 429 with no Retry-After must still wait before retrying"

    def test_rate_limited_endpoint_is_retried_in_place_before_failover(self):
        """429 means "later", not "never" — unlike a refusal, it is worth a retry."""
        primary = ops.OVERPASS_ENDPOINTS[0]
        fake, calls = _responder({primary: [_rate_limited("1"), _RESPONSE]})
        with patch("urllib.request.urlopen", side_effect=fake), patch("time.sleep"):
            _query_overpass(BBOX)
        assert calls == [primary, primary], f"expected an in-place retry, got {calls}"
