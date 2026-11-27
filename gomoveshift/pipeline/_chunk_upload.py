"""What every resumable Google upload (YouTube and Drive) shares: the chunk
size, and the per-chunk timeout that never leaves a chunk running behind the
caller.

Chunk size. googleapiclient sends each chunk as one ``bytes`` body over
httplib2, whose socket timeout (``googleapiclient.http.DEFAULT_HTTP_TIMEOUT_SEC``,
60 s) bounds the whole ``sendall`` of it, however steadily it moves. A 256 MB
chunk therefore needed a ~36 Mbit/s uplink or it failed with ``timed out``,
on every retry. 8 MB needs ~1.1 Mbit/s.

Timeout. ``next_chunk`` runs in a worker thread so a stall the socket timeout
does not catch still ends. A thread cannot be killed, so on timeout the
worker's sockets are shut down and it is awaited briefly. Before, it was
abandoned: it kept sending on the same request and connection while the
caller's retry sent the same chunk again beside it.
"""

from __future__ import annotations

import socket
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any

#: A multiple of 256 KiB, as resumable uploads require.
UPLOAD_CHUNKSIZE = 8 * 1024 * 1024
#: How long a timed-out chunk is awaited once its sockets are shut down.
ABORT_GRACE_SEC = 1.5


class ChunkStillRunningError(TimeoutError):
    """The chunk timed out and its worker is still running after its sockets
    were shut down. Never retry on the same request: a retry would upload
    beside it."""


def _shut_down_sockets(http: Any) -> None:
    """Shut down every socket of the request's httplib2 client (wrapped by
    ``AuthorizedHttp`` as ``.http``), which wakes a blocked send or recv."""
    for _ in range(3):  # request.http -> AuthorizedHttp.http -> httplib2.Http
        if http is None:
            return
        connections = getattr(http, "connections", None)
        if isinstance(connections, dict):
            for conn in list(connections.values()):
                sock = getattr(conn, "sock", None)
                if isinstance(sock, socket.socket):
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
        http = getattr(http, "http", None)


def next_chunk_with_timeout(request: Any, timeout: float, label: str) -> tuple:
    """``request.next_chunk()``, raising ``TimeoutError`` after ``timeout``
    seconds only once the chunk has stopped (its result, if it landed during
    the grace, is returned). If it will not stop, raises
    ``ChunkStillRunningError``. Explicit pool, not ``with``: its
    ``shutdown(wait=True)`` would wait on a stalled worker forever."""
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(request.next_chunk)
    try:
        result = future.result(timeout=timeout)
    except FuturesTimeoutError:
        _shut_down_sockets(getattr(request, "http", None))
        try:
            # A chunk that landed just as the bound ran out is a result, kept.
            result = future.result(timeout=ABORT_GRACE_SEC)
        except FuturesTimeoutError:
            pool.shutdown(wait=False, cancel_futures=True)
            raise ChunkStillRunningError(
                f"{label} chunk upload exceeded {timeout}s and is still running after its "
                "connection was shut down; not retried beside it"
            ) from None
        except Exception:  # noqa: BLE001 -- the aborted chunk's own error is expected
            pool.shutdown(wait=True)
            raise TimeoutError(f"{label} chunk upload exceeded {timeout}s timeout") from None
    pool.shutdown(wait=True)
    return result
