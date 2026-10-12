"""``configure_logging`` must reconfigure on re-invocation.

``logging.basicConfig`` no-ops when the root logger already has handlers,
so a second ``configure_logging(log_file=…)`` call (resume flows, library
callers) silently fails to attach the file handler — ``drive.log`` ends up
empty. ``force=True`` is the documented stdlib escape hatch.
"""

import logging

import pytest


@pytest.fixture
def clean_root_logger():
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level
    root.handlers = []
    try:
        yield root
    finally:
        for h in root.handlers:
            if isinstance(h, logging.FileHandler):
                h.close()
        root.handlers = original_handlers
        root.level = original_level


def test_second_configure_logging_attaches_file_handler(clean_root_logger, tmp_path):
    """Second call with log_file must attach a FileHandler (force=True semantics)."""
    from gomoveshift.cli.logging_setup import configure_logging

    configure_logging()  # First call: stream only.
    assert not any(isinstance(h, logging.FileHandler) for h in clean_root_logger.handlers)

    log_file = tmp_path / "drive.log"
    configure_logging(log_file=log_file)

    file_handlers = [h for h in clean_root_logger.handlers if isinstance(h, logging.FileHandler)]
    assert file_handlers, "Second configure_logging call must attach a FileHandler"
    assert file_handlers[0].baseFilename == str(log_file)


def test_second_configure_logging_writes_to_file(clean_root_logger, tmp_path):
    """End-to-end: a log record after the second call lands on disk."""
    from gomoveshift.cli.logging_setup import configure_logging

    configure_logging()
    log_file = tmp_path / "drive.log"
    configure_logging(log_file=log_file)

    logging.getLogger("test").info("smoke")
    for h in clean_root_logger.handlers:
        h.flush()

    assert log_file.exists()
    assert "smoke" in log_file.read_text()
