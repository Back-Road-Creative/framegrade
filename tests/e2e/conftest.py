import pytest

from gomoveshift.publishing.queue import open_queue_db


@pytest.fixture
def conn(tmp_path):
    c = open_queue_db(tmp_path / "queue")
    try:
        yield c
    finally:
        c.close()


@pytest.fixture
def asset(tmp_path):
    p = tmp_path / "short_000.mp4"
    p.write_bytes(b"video-bytes")
    return p
