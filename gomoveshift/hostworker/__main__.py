"""``python -m gomoveshift.hostworker [--spool DIR]``: the host unit's ExecStart."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading

from .worker import HostWorker, HostWorkerError


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gomoveshift.hostworker")
    ap.add_argument("--spool", default=None, help="spool dir (default: $GMS_HOST_SPOOL_DIR)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    try:
        worker = HostWorker(args.spool)
    except HostWorkerError as exc:
        print(f"gms-host-worker: refusing to start: {exc}", file=sys.stderr)
        return 2
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    worker.serve(stop)
    return 0


if __name__ == "__main__":
    sys.exit(main())
