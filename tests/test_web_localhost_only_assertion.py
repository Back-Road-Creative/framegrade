"""Refuse to bind the web UI on a non-loopback host
without the explicit ``--allow-remote`` opt-in.
"""

from __future__ import annotations

import pytest

from gomoveshift.web.app import _ALLOWED_LOOPBACK_HOSTS, _assert_localhost_only


@pytest.mark.parametrize("host", sorted(_ALLOWED_LOOPBACK_HOSTS))
def test_loopback_hosts_pass(host: str) -> None:
    _assert_localhost_only(host)  # no raise expected


@pytest.mark.parametrize("host", ["0.0.0.0", "10.0.0.5", "192.168.1.42", "::"])  # noqa: S104
def test_non_loopback_hosts_raise(host: str) -> None:
    with pytest.raises(RuntimeError) as exc:
        _assert_localhost_only(host)
    msg = str(exc.value)
    assert host in msg
    assert "loopback" in msg
    assert "--allow-remote" in msg


@pytest.mark.parametrize("host", ["0.0.0.0", "10.0.0.5"])  # noqa: S104
def test_allow_remote_bypasses_check(host: str) -> None:
    _assert_localhost_only(host, allow_remote=True)
