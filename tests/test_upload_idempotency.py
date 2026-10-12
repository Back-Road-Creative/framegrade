"""``_upload_file`` must not duplicate Drive files on retry, must keep a
reused/updated file actually shared, and must handle a quoted filename."""

from types import SimpleNamespace

import pytest

from gomoveshift.pipeline.uploader import PipelineUploader, _escape_drive_query_value


def _make_uploader(tmp_path):
    creds = tmp_path / "credentials.json"
    token = tmp_path / "token.json"
    creds.write_text('{"installed":{}}')
    token.write_text("{}")
    return PipelineUploader(credentials_path=creds, token_path=token)


# Builds the SAME query production builds, so escaping is proven at the call site.
def _list_query(name: str, folder_id: str) -> str:
    return (
        f"name='{_escape_drive_query_value(name)}' and '{folder_id}' in parents and trashed=false"
    )


class _Exec:
    """``.execute()``: returns ``result``, or raises ``exc``."""

    def __init__(self, result=None, exc=None):
        self._result, self._exc = result, exc

    def execute(self):
        if self._exc is not None:
            raise self._exc
        return self._result


class _FakeChunkRequest:
    def __init__(self, file_id):
        self._file_id = file_id

    def next_chunk(self):
        return None, {"id": self._file_id}


class _FakeDrive:
    """In-memory stand-in for Drive's ``files()``/``permissions()`` API."""

    def __init__(self, *, permission_fail_first_n=0):
        self.store: dict[str, dict] = {}  # file_id -> {"name", "folder_id", "size"}
        self.create_calls = 0
        self.update_calls = 0
        self.shared_ids: set[str] = set()
        self._perm_calls = 0
        self._perm_fail_first_n = permission_fail_first_n
        self._next_id = 0

    def files(self):
        return SimpleNamespace(list=self._list, create=self._create, update=self._update)

    def permissions(self):
        return SimpleNamespace(create=self._share)

    def _list(self, q=None, fields=None, spaces=None):
        matches = [
            {"id": fid, "size": str(meta["size"])}
            for fid, meta in self.store.items()
            if _list_query(meta["name"], meta["folder_id"]) == q
        ]
        return _Exec({"files": matches})

    def _create(self, body=None, media_body=None, fields=None):
        self.create_calls += 1
        self._next_id += 1
        file_id = f"file{self._next_id}"
        self.store[file_id] = {
            "name": body["name"],
            "folder_id": body["parents"][0],
            "size": media_body.size(),
        }
        return _FakeChunkRequest(file_id)

    def _update(self, fileId=None, media_body=None, fields=None):  # noqa: N803
        self.update_calls += 1
        self.store[fileId]["size"] = media_body.size()
        return _FakeChunkRequest(fileId)

    def _share(self, fileId=None, body=None):  # noqa: N803
        self._perm_calls += 1
        if self._perm_calls <= self._perm_fail_first_n:
            return _Exec(exc=RuntimeError("simulated transient permission failure"))
        self.shared_ids.add(fileId)
        return _Exec({"id": "perm1"})


@pytest.fixture(autouse=True)
def _no_credential_refresh(monkeypatch):
    monkeypatch.setattr(PipelineUploader, "_refresh_credentials_if_needed", lambda self: None)


def test_second_upload_of_identical_file_reuses_the_first(tmp_path):
    uploader = _make_uploader(tmp_path)
    drive = _FakeDrive()
    video = tmp_path / "short_01.mp4"
    video.write_bytes(b"identical drive payload" * 10)

    first_link = uploader._upload_file(drive, video, "FOLDER1", make_shareable=False)
    second_link = uploader._upload_file(drive, video, "FOLDER1", make_shareable=False)

    assert drive.create_calls == 1, f"expected one create(), got {drive.create_calls}"
    same_name = [m for m in drive.store.values() if m["name"] == "short_01.mp4"]
    assert len(same_name) == 1, f"expected one remote file, found {len(same_name)}: {drive.store}"
    assert first_link == second_link


def test_reupload_of_a_resized_file_replaces_content_in_place(tmp_path):
    # A re-render (different size) replaces content via update(), never a 2nd file.
    uploader = _make_uploader(tmp_path)
    drive = _FakeDrive()
    video = tmp_path / "short_01.mp4"
    video.write_bytes(b"first render")

    uploader._upload_file(drive, video, "FOLDER1", make_shareable=False)

    video.write_bytes(b"second, re-rendered, longer payload")
    uploader._upload_file(drive, video, "FOLDER1", make_shareable=False)

    assert drive.create_calls == 1
    assert drive.update_calls == 1
    same_name = [m for m in drive.store.values() if m["name"] == "short_01.mp4"]
    assert len(same_name) == 1
    assert same_name[0]["size"] == video.stat().st_size


def test_retry_after_permission_failure_shares_the_reused_file(tmp_path):
    # A file whose FIRST share grant failed must be re-shared on retry, not
    # just reused — else the returned link is dead. create() already
    # landed the file; the SEPARATE permissions() call is what fails.
    uploader = _make_uploader(tmp_path)
    drive = _FakeDrive(permission_fail_first_n=1)
    video = tmp_path / "short_01.mp4"
    video.write_bytes(b"payload for the sharing regression test")

    with pytest.raises(RuntimeError, match="simulated transient permission failure"):
        uploader._upload_file(drive, video, "FOLDER1", make_shareable=True)
    assert drive.create_calls == 1

    link = uploader._upload_file(drive, video, "FOLDER1", make_shareable=True)

    assert drive.create_calls == 1, "the retry must reuse, not re-create, the file"
    file_id = next(iter(drive.store))
    assert file_id in drive.shared_ids, (
        f"retry did not (re-)apply the Drive permission — link {link!r} is not accessible"
    )


def test_reuse_check_handles_an_apostrophe_in_the_filename(tmp_path):
    uploader = _make_uploader(tmp_path)
    drive = _FakeDrive()
    video = tmp_path / "o'brien_lookout.mp4"
    video.write_bytes(b"apostrophe-named payload")

    uploader._upload_file(drive, video, "FOLDER1", make_shareable=False)
    uploader._upload_file(drive, video, "FOLDER1", make_shareable=False)

    assert drive.create_calls == 1, (
        f"expected one create() for an apostrophe-named file, got {drive.create_calls} "
        "— the existing-file query broke on the quote"
    )
