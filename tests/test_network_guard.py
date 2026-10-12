"""Whole-suite network egress guard (``block_non_loopback_network``).

``tests/test_chapters.py`` made real HTTPS calls to the Overpass API on every
test reaching terrain-band classification (~50 s/test, cProfile: 92
``_ssl._SSLSocket.read`` calls) because
``gomoveshift/pipeline/chapters/generator.py`` calls
``overpass_source.post_overpass_query`` directly while the test only patched
``_query_overpass`` -- a leak the per-endpoint ``hermetic_overpass`` fixture
in this file's conftest could not see, because it never patches the socket
layer itself. This guard closes the gap structurally: any test whose code
path reaches ``socket.socket.connect``/``connect_ex``/``create_connection``
for a non-loopback address fails immediately and loudly, instead of hanging
on a real DNS+TLS round trip.
"""

import os
import shutil
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

CONFTEST = Path(__file__).parent / "conftest.py"
SERVICE_ROOT = Path(__file__).resolve().parent.parent


def test_non_loopback_connect_blocked():
    """A raw connect() to a real-looking public host must raise, not dial out."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        with pytest.raises(RuntimeError, match="8\\.8\\.8\\.8"):
            sock.connect(("8.8.8.8", 443))


def test_non_loopback_connect_ex_blocked():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        with pytest.raises(RuntimeError, match="overpass-api\\.de"):
            sock.connect_ex(("overpass-api.de", 443))


def test_non_loopback_create_connection_blocked():
    with pytest.raises(RuntimeError, match="example\\.com"):
        socket.create_connection(("example.com", 443), timeout=1)


def test_loopback_connect_allowed():
    """A real connect to a local listening socket must go through untouched."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            client.settimeout(2)
            client.connect(("127.0.0.1", port))  # must not raise


def test_allow_network_marker_opts_out(tmp_path):
    """``@pytest.mark.allow_network`` must let a non-loopback connect through
    the guard (it still fails downstream -- there's no live socket here --
    but it must fail with a plain OS connection error, not the guard's
    ``RuntimeError``).
    """
    mini = tmp_path / "test_opt_out.py"
    mini.write_text(
        textwrap.dedent(
            """
            import socket

            import pytest

            @pytest.mark.allow_network
            def test_marked_out():
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(1)
                    try:
                        sock.connect(("127.0.0.2", 1))
                    except RuntimeError as exc:
                        raise AssertionError(
                            f"allow_network marker did not opt out: {exc}"
                        ) from None
                    except OSError:
                        pass  # expected -- nothing listens there
            """
        )
    )
    shutil.copy(CONFTEST, tmp_path / "conftest.py")
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(mini),
            "-o",
            "addopts=",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                filter(None, [str(SERVICE_ROOT), os.environ.get("PYTHONPATH")])
            ),
        },
    )
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"


def test_real_overpass_marker_implies_allow_network(tmp_path):
    """``@pytest.mark.real_overpass`` (a live-network opt-in) must also
    disable the socket guard -- an author reaching for the existing
    per-endpoint opt-out shouldn't also have to know about this one.
    """
    mini = tmp_path / "test_real_overpass_marker.py"
    mini.write_text(
        textwrap.dedent(
            """
            import socket

            import pytest

            @pytest.mark.real_overpass
            def test_marked_out():
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(1)
                    try:
                        sock.connect(("127.0.0.2", 1))
                    except RuntimeError as exc:
                        raise AssertionError(
                            f"real_overpass did not imply allow_network: {exc}"
                        ) from None
                    except OSError:
                        pass  # expected -- nothing listens there
            """
        )
    )
    shutil.copy(CONFTEST, tmp_path / "conftest.py")
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(mini),
            "-o",
            "addopts=",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                filter(None, [str(SERVICE_ROOT), os.environ.get("PYTHONPATH")])
            ),
        },
    )
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
