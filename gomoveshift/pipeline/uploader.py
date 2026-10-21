"""
Pipeline Uploader — resume-capable Google Drive upload with manifest.

Uses google-api-python-client directly (no external pipeline dependency).


Usage:
    from gomoveshift.pipeline.uploader import PipelineUploader

    uploader = PipelineUploader()
    manifest = uploader.upload_shorts(
        shorts_dir="/path/to/shorts",
        project_name="example-run",
    )
    # manifest.files = {"short_01.mp4": "https://drive.google.com/..."}
"""

import contextlib
import fcntl
import json
import logging
import os
import random
import socket
import ssl
import stat
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from gomoveshift.paths import gms_secret_path
from gomoveshift.pipeline._atomic_write import atomic_write_json
from gomoveshift.pipeline._yt_creds import _sanitize_sops_stderr, sops_age_recipients

logger = logging.getLogger(__name__)

_DRIVE_STALL_TIMEOUT = 120  # seconds with no upload progress before raising
_DRIVE_RATE_LIMIT_MAX_ATTEMPTS = 5  # max retries on HTTP 429

# SSL/connection error types to retry on
_SSL_ERRORS = (ssl.SSLError, ConnectionResetError, socket.error)
_SSL_RETRY_DELAYS = [10, 20, 30]  # seconds between retries (3 attempts total)


class DriveUploadError(RuntimeError):
    """Raised when a Drive upload fails after all retry attempts."""


class DriveCredentialError(DriveUploadError):
    """OAuth credential refresh failed permanently — operator action required.

    ``google.auth.exceptions.RefreshError`` (revoked or expired
    refresh token) is not an ``HttpError`` and therefore was previously
    misclassified as transient by ``is_permanent_error``. The retry loop
    would burn ``_DRIVE_RATE_LIMIT_MAX_ATTEMPTS`` cycles before
    propagating the underlying error with no actionable hint. This
    subclass is recognised as permanent so the retry loop fast-fails
    AND the message names the SOPS token path and the recovery command.
    """


class SopsDecryptError(RuntimeError):
    """SOPS decryption failed for a known-recoverable reason.

    When the SOPS stderr matches a recognised failure pattern
    (no age identity, decryption-failed-or-bad-record-mac), the operator
    can recover deterministically (check that the age key named by
    ``GMS_AGE_KEY_FILE`` is a recipient of the file, or re-encrypt it with
    ``sops`` using a current age identity). Surface that hint instead of the
    raw stderr so operators don't have to grep logs for the right action.
    """


class CircuitBreakerOpenError(RuntimeError):
    """Raised when the circuit breaker is open (too many consecutive failures)."""


# Alias for shorter usage in calling code
CircuitBreakerOpen = CircuitBreakerOpenError


class CircuitBreaker:
    """Simple circuit breaker: trips after N consecutive failures.

    Thread-safe via a lock on mutations.
    """

    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.consecutive_failures = 0
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        return self.consecutive_failures >= self.failure_threshold

    def record_failure(self) -> None:
        with self._lock:
            self.consecutive_failures += 1

    def record_success(self) -> None:
        with self._lock:
            self.consecutive_failures = 0

    def check(self) -> None:
        """Raise CircuitBreakerOpen if the breaker is tripped."""
        if self.is_open:
            raise CircuitBreakerOpen(
                f"Circuit breaker open after {self.consecutive_failures} consecutive failures "
                "— API quota may be exhausted, back off before retrying"
            )


def _get_http_error_code(exc: BaseException) -> Optional[int]:
    """Extract HTTP status code from a googleapiclient HttpError, or None."""
    try:
        from googleapiclient.errors import HttpError
    except ImportError:
        return None

    if isinstance(exc, HttpError):
        return exc.resp.status
    return None


def is_transient_error(exc: BaseException) -> bool:
    """Return True if the exception represents a transient/retriable error.

    Transient: 429, 500-599, TimeoutError, ConnectionError, OSError, ssl.SSLError, socket.error.
    """
    code = _get_http_error_code(exc)
    if code is not None:
        return code == 429 or 500 <= code <= 599

    return isinstance(exc, (TimeoutError, ConnectionError, ssl.SSLError, socket.error, OSError))


def is_permanent_error(exc: BaseException) -> bool:
    """Return True if the exception represents a permanent/non-retriable error.

    Permanent: HTTP 401, 403, 404; OAuth ``RefreshError`` (revoked / expired
    refresh token — retrying won't help); and the named subclasses
    ``DriveCredentialError`` / ``SopsDecryptError`` which both carry
    operator-actionable recovery hints. These
    classifications stop the retry loop from burning
    ``_DRIVE_RATE_LIMIT_MAX_ATTEMPTS`` cycles on credential failures.
    """
    code = _get_http_error_code(exc)
    if code is not None:
        return code in {401, 403, 404}

    if isinstance(exc, (DriveCredentialError, SopsDecryptError)):
        return True

    # ``google.auth.exceptions.RefreshError`` may have been wrapped already
    # (e.g. into DriveCredentialError) — but classify the raw form too in
    # case a call site lets it propagate unwrapped.
    try:
        from google.auth.exceptions import RefreshError

        if isinstance(exc, RefreshError):
            return True
    except ImportError:
        pass

    return False


def _drive_execute(
    request,
    max_attempts: int = _DRIVE_RATE_LIMIT_MAX_ATTEMPTS,
    circuit_breaker: Optional["CircuitBreaker"] = None,
):
    """Execute a Drive API request, retrying on transient errors with jittered backoff.

    Permanent errors (401/403/404) fail immediately.
    Transient errors (429/5xx) retry with exponential backoff + jitter.
    If a circuit_breaker is provided, it is checked before execution and
    updated on success/failure.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    if circuit_breaker is not None:
        circuit_breaker.check()

    from googleapiclient.errors import HttpError

    for attempt in range(max_attempts):
        try:
            result = request.execute()
            if circuit_breaker is not None:
                circuit_breaker.record_success()
            return result
        except HttpError as exc:
            if is_permanent_error(exc):
                if circuit_breaker is not None:
                    circuit_breaker.record_failure()
                raise

            if is_transient_error(exc):
                if circuit_breaker is not None:
                    circuit_breaker.record_failure()
                if attempt < max_attempts - 1:
                    base_wait = 2**attempt  # 1s, 2s, 4s, 8s, 16s
                    jitter = 0.5 + random.random()  # [0.5, 1.5)
                    wait = base_wait * jitter
                    logger.warning(
                        "Drive API error (HTTP %d), attempt %d/%d — retrying in %.1fs",
                        exc.resp.status,
                        attempt + 1,
                        max_attempts,
                        wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        "Drive API error after %d attempts — "
                        "consider spreading uploads or increasing quota",
                        max_attempts,
                    )
                    raise RuntimeError(
                        f"Drive API error after {max_attempts} attempts"
                        " — consider spreading uploads or increasing quota"
                    ) from exc
            else:
                # Unknown HTTP error — don't retry
                raise

    raise RuntimeError("_drive_execute: exhausted retries without result")


# SOPS-encrypted credential paths (default for pipeline use).
#
# Every path is overrideable via an env var; the defaults live under the
# per-user config home. Resolution goes through ``gms_secret_path`` (shared
# with the console) at call time, so an empty override counts as unset and a
# variable changed after import is honoured. The historical constant names
# (``DEFAULT_CREDENTIALS_ENC``, ``DEFAULT_TOKEN_ENC``, ``_AGE_KEY_FILE``,
# ``_SOPS_CONFIG``) stay readable through the module ``__getattr__`` below; a
# value assigned to one of them on the module (for example by a test) wins over
# the environment.
_SECRET_SPECS = {
    "DEFAULT_CREDENTIALS_ENC": ("GMS_DRIVE_CREDENTIALS_ENC", "drive_credentials.json.enc"),
    "DEFAULT_TOKEN_ENC": ("GMS_DRIVE_TOKEN_ENC", "drive_token.json.enc"),
    # Age key used for SOPS decrypt/encrypt
    "_AGE_KEY_FILE": ("GMS_AGE_KEY_FILE", "age.key"),
    # SOPS config (creation_rules + age recipients). The sops binary respects
    # ``--config <path>``; the default sits next to the credentials in the
    # config home.
    "_SOPS_CONFIG": ("GMS_SOPS_CONFIG", ".sops.yaml"),
}


def _secret(name: str) -> Path:
    """Current path for one of the credential constants, read at call time."""
    if name in globals():
        return Path(globals()[name])
    env_name, filename = _SECRET_SPECS[name]
    return gms_secret_path(env_name, filename)


def __getattr__(name: str):
    if name in _SECRET_SPECS:
        value = _secret(name)
        return str(value) if name == "_SOPS_CONFIG" else value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Default Drive folders
DEFAULT_DRIVE_FOLDER = "GoMoveShift/Videos/Shorts"
DEFAULT_PHOTO_DRIVE_FOLDER = "GoMoveShift/Photos/Daily"

# Mimetype mapping by file extension
_MIMETYPE_MAP = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def _guess_mimetype(file_path: Path) -> str:
    """Guess mimetype from file extension, defaulting to video/mp4."""
    return _MIMETYPE_MAP.get(file_path.suffix.lower(), "video/mp4")


def _escape_drive_query_value(value: str) -> str:
    """Escape ``\\``/``'`` for a Drive query literal (see docs/pipeline-internals.md)."""
    return value.replace("\\", "\\\\").replace("'", "\\'")


def drive_download_link(file_id: str) -> str:
    """Build a shareable Google Drive link for a file.

    Uses file/d/ID/view format which works for Metricool import.
    The old uc?export=download format triggers virus scan redirects for >100MB files.
    """
    return f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"


_SOPS_RECOVERABLE_PATTERNS: tuple[str, ...] = (
    "no age identity",
    "DECRYPTION_FAILED",
    "no identity found",
    "Failed to get the data key",
)


def sops_decrypt_json(enc_path: Path) -> dict:
    """Decrypt a SOPS-encrypted JSON file and return parsed data.

    Decrypted content stays in memory only — never written to disk.

    When ``sops`` exits non-zero AND its stderr matches a known
    recoverable pattern (no age identity / decryption-failed), raise
    ``SopsDecryptError`` with the age key path and the operator
    recovery command. Other failures still raise the generic
    ``RuntimeError`` so unfamiliar failures remain visible rather than
    being misclassified as recoverable.
    """
    if not enc_path.exists():
        raise FileNotFoundError(f"Encrypted file not found: {enc_path}")

    env = os.environ.copy()
    env["SOPS_AGE_KEY_FILE"] = str(_secret("_AGE_KEY_FILE"))

    result = subprocess.run(
        ["sops", "--config", str(_secret("_SOPS_CONFIG")), "-d", str(enc_path)],
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if any(pat in stderr for pat in _SOPS_RECOVERABLE_PATTERNS):
            raise SopsDecryptError(
                f"sops decrypt failed for {enc_path}: {stderr}\n"
                f"Age key: {_secret('_AGE_KEY_FILE')}\n"
                f"Recovery: check that the age key above (set GMS_AGE_KEY_FILE "
                f"to use another one) is a recipient of the file, or "
                f"re-encrypt it with `sops` using a current age identity."
            )
        raise RuntimeError(f"sops decrypt failed for {enc_path}: {stderr}")

    return json.loads(result.stdout)


def _sops_lock_path(enc_path: Path) -> Path:
    """Path of the sidecar lock guarding write-backs to ``enc_path``.

    Always beside the target, so every process that refreshes the same
    credential agrees on one lock. A write is only attempted once the
    target's directory is known to be writable (see :func:`sops_encrypt_json`).
    """
    enc_path = Path(enc_path)
    return enc_path.parent / (enc_path.name + ".lock")


@contextlib.contextmanager
def _sops_write_lock(enc_path: Path):
    """Exclusive *inter-process* lock over one ``.enc`` write-back.

    ``_refresh_lock`` only serializes threads within a single process; a photo
    run and a drive run refreshing the same shared OAuth sidecar are separate
    processes, so it protects nothing between them. Blocking
    ``fcntl.flock(LOCK_EX)`` on a sidecar — advisory, released on fd close or
    process death, so a crashed run can't wedge the next one.
    """
    lock_path = _sops_lock_path(enc_path)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o664)
    # Several users may share a group and refresh the same credential; the
    # creator's umask may narrow the sidecar. Best-effort — a foreign
    # creator's fails.
    with contextlib.suppress(OSError):
        os.chmod(lock_path, 0o664)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _install_encrypted(tmp_path: str, enc_path: Path) -> None:
    """Atomically install the encrypted temp file at ``enc_path``.

    ``os.replace`` is atomic only within one filesystem — which is exactly why
    the caller stages the temp file in the target's own directory. Mode and
    ownership carry over from the file being replaced: the ``.enc`` is
    group-shared, and a regression to ``mkstemp``'s 0600 would lock the
    pipeline out of reading its own credential.
    """
    try:
        prev = os.stat(enc_path)
    except OSError:
        prev = None
    if prev is not None:
        with contextlib.suppress(OSError):
            os.chmod(tmp_path, stat.S_IMODE(prev.st_mode))
        with contextlib.suppress(OSError):
            os.chown(tmp_path, prev.st_uid, prev.st_gid)
    os.replace(tmp_path, enc_path)


def sops_encrypt_json(
    data: dict, enc_path: Path, *, age_recipients: Optional[List[str]] = None
) -> None:
    """Encrypt a dict as JSON using SOPS and write to enc_path.

    Atomic and cross-process safe: the plaintext temp file is created in the
    target's own directory (same filesystem), encrypted in place, then
    installed with ``os.replace`` while holding an flock sidecar — so a
    concurrent photo-run/drive-run OAuth refresh can never leave the shared
    ``.enc`` truncated or half-written. Plaintext is removed even on failure.

    Re-encrypting never changes who can read the file: ``age_recipients``
    wins when given, otherwise the recipients of the file being replaced are
    reused (``_yt_creds.sops_age_recipients``), and only a brand-new file with no
    explicit list falls back to the config's creation rules. Without this a
    token re-mint for a separate repo matched only the pipeline config's
    catch-all rule and came back unreadable to that repo's own keyring.
    """
    env = os.environ.copy()
    env["SOPS_AGE_KEY_FILE"] = str(_secret("_AGE_KEY_FILE"))

    enc_path = Path(enc_path)
    with contextlib.suppress(OSError):
        enc_path.parent.mkdir(parents=True, exist_ok=True)
    # A directory this process cannot write to can never take the install, so
    # refuse before the lock sidecar or any plaintext is created anywhere.
    if not os.access(enc_path.parent, os.W_OK | os.X_OK):
        raise PermissionError(
            f"cannot write the encrypted credential {enc_path}: its directory is missing or "
            "not writable by this user. Point the credential variable "
            "(GMS_DRIVE_TOKEN_ENC, GMS_DRIVE_CREDENTIALS_ENC, or the provider's token "
            "variable) at a file in a directory this user can write."
        )

    with _sops_write_lock(enc_path):
        if age_recipients is None:
            age_recipients = sops_age_recipients(enc_path)
        _sops_encrypt_json_locked(data, enc_path, env, age_recipients=age_recipients)


def _sops_encrypt_json_locked(
    data: dict, enc_path: Path, env: dict, *, age_recipients: Optional[List[str]] = None
) -> None:
    """Body of :func:`sops_encrypt_json`, run under the write lock."""
    fd, tmp_path = tempfile.mkstemp(suffix=".json", prefix=".sops-", dir=str(enc_path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)

        # --filename-override tells SOPS to match creation rules against the
        # target path (relative to the config dir) instead of the temp path.
        # The sops config dir is the parent of the SOPS config; the override
        # is the target's path relative to it (just the file name when the
        # target lives elsewhere).
        sops_config = str(_secret("_SOPS_CONFIG"))
        sops_config_dir = os.path.dirname(sops_config) or "/"
        try:
            override = str(enc_path.relative_to(sops_config_dir))
        except ValueError:
            override = enc_path.name
        cmd = ["sops", "--config", sops_config, "--filename-override", override]
        if age_recipients:
            # Explicit recipients take precedence over creation rules in sops,
            # so the replaced file keeps its readers regardless of which rule
            # the override path happens to match.
            cmd += ["--age", ",".join(age_recipients)]
        result = subprocess.run(
            [*cmd, "-e", "-i", tmp_path],
            capture_output=True,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            # Raw SOPS stderr can carry age-key paths / recipient rules — log
            # the full text at DEBUG only, surface a sanitized category in the
            # raised error (never log raw tokens).
            logger.debug("sops encrypt failed — raw stderr: %s", result.stderr.strip())
            raise RuntimeError(f"sops encrypt failed: {_sanitize_sops_stderr(result.stderr)}")

        # Install the encrypted temp file at the target (the target directory
        # was checked writable above); any failure leaves the previous file in
        # place and the temp file is removed in the ``finally`` below.
        _install_encrypted(tmp_path, enc_path)
        tmp_path = None  # already installed, skip cleanup
    finally:
        # Always clean up plaintext temp file if it still exists
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


@dataclass
class VerificationResult:
    """Result of verifying uploaded Drive links."""

    verified: List[str]  # filenames confirmed accessible
    failed: List[str]  # filenames with dead/inaccessible links


@dataclass
class UploadManifest:
    """Manifest tracking uploaded files and their Drive links."""

    project_name: str
    folder_id: Optional[str] = None
    files: Dict[str, str] = field(default_factory=dict)  # filename → drive_link
    completed: bool = False
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict:
        return {
            "project_name": self.project_name,
            "folder_id": self.folder_id,
            "files": self.files,
            "completed": self.completed,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UploadManifest":
        # Handle both structured format (with "files" key) and
        # flat format ({filename: link}) from ad-hoc upload scripts
        if "files" in data or "project_name" in data:
            # Structured format
            return cls(
                project_name=data.get("project_name", ""),
                folder_id=data.get("folder_id"),
                files=data.get("files", {}),
                completed=data.get("completed", False),
                created_at=data.get("created_at", ""),
                updated_at=data.get("updated_at", ""),
            )
        else:
            # Flat format: all keys are filenames mapping to Drive links
            return cls(
                project_name="",
                files=data,
                completed=True,
            )

    def save(self, path: Path) -> None:
        """Save manifest to JSON (called after each upload for resume).

        Written atomically (temp file + ``os.replace``): a kill -9 / disk-full
        mid-write leaves the previous manifest intact rather than a truncated,
        unparseable file — this bare rewrite IS the resume mechanism.
        """
        self.updated_at = datetime.now().isoformat()
        atomic_write_json(Path(path), self.to_dict())

    @classmethod
    def load(cls, path: Path) -> "UploadManifest":
        """Load manifest from JSON for resume.

        Corrupt-tolerant: a truncated/partial manifest (e.g. a crash mid-write
        that predates the atomic writer) must not crash the resume. On a parse
        failure recover with an empty manifest — the run re-uploads rather than
        forcing the operator to hand-delete the file.
        """
        try:
            with open(path) as f:
                return cls.from_dict(json.load(f))
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Corrupt upload manifest at {path} ({e}) — starting fresh")
            return cls(project_name="")

    def get_link_for_file(self, filename: str) -> Optional[str]:
        """Get Drive link for a filename."""
        return self.files.get(filename)

    def get_pending_files(self, all_files: List[str]) -> List[str]:
        """Get files that haven't been uploaded yet."""
        return [f for f in all_files if f not in self.files]


class PipelineUploader:
    """
    Uploads rendered shorts to Google Drive with resume support.

    Saves a manifest after each file upload so the process can be
    resumed if interrupted. The manifest maps filenames to Drive links,
    which the CSV generator uses for Metricool import.
    """

    def __init__(
        self,
        credentials_enc: Optional[Path] = None,
        token_enc: Optional[Path] = None,
        drive_folder: str = DEFAULT_DRIVE_FOLDER,
        *,
        credentials_path: Optional[Path] = None,
        token_path: Optional[Path] = None,
    ):
        """
        Args:
            credentials_enc: Path to SOPS-encrypted Drive OAuth credentials
                (default: ``GMS_DRIVE_CREDENTIALS_ENC``, resolved at call time)
            token_enc: Path to SOPS-encrypted Drive OAuth token
                (default: ``GMS_DRIVE_TOKEN_ENC``, resolved at call time)
            drive_folder: Drive folder path (created if needed)
            credentials_path: Legacy plaintext credentials (overrides SOPS)
            token_path: Legacy plaintext token (overrides SOPS)
        """
        # Legacy plaintext mode when explicit paths are passed (CLI --credentials/--token)
        self._legacy_mode = credentials_path is not None or token_path is not None
        if self._legacy_mode:
            self._credentials_path = Path(credentials_path) if credentials_path else None
            self._token_path = Path(token_path) if token_path else None
        else:
            self.credentials_enc = Path(
                credentials_enc
                if credentials_enc is not None
                else _secret("DEFAULT_CREDENTIALS_ENC")
            )
            self.token_enc = Path(
                token_enc if token_enc is not None else _secret("DEFAULT_TOKEN_ENC")
            )

        self.drive_folder = drive_folder
        self._client = None
        self._creds = None
        self._refresh_lock = threading.Lock()

        # Early credential detection — warn now rather than failing mid-upload
        try:
            self._ensure_credentials()
        except (FileNotFoundError, RuntimeError) as exc:
            logger.warning("Credential check at init: %s", exc)

    def _ensure_credentials(self) -> None:
        """Validate credentials exist before attempting upload."""
        if self._legacy_mode:
            if self._credentials_path and not self._credentials_path.exists():
                raise FileNotFoundError(
                    f"Drive credentials not found: {self._credentials_path}\n"
                    f"Download from Google Cloud Console."
                )
            if self._token_path and not self._token_path.exists():
                logger.warning(
                    f"Drive token not found: {self._token_path}. OAuth flow will be triggered."
                )
        else:
            if not self.credentials_enc.exists():
                raise FileNotFoundError(
                    f"Encrypted Drive credentials not found: {self.credentials_enc}\n"
                    f"Run: gomoveshift upload --init-oauth <project_dir>"
                )
            if not self.token_enc.exists():
                raise FileNotFoundError(
                    f"Encrypted Drive token not found: {self.token_enc}\n"
                    f"Run: gomoveshift upload --init-oauth <project_dir>"
                )

    def _refresh_credentials_if_needed(self) -> None:
        """Thread-safe credential refresh to prevent concurrent token refresh corruption.

        Multiple upload threads sharing a Google API client can trigger simultaneous
        creds.refresh() when the OAuth token expires mid-upload, corrupting the heap.
        This method serializes refresh attempts behind a lock.

        ``RefreshError`` (revoked or expired refresh token) is
        re-raised as ``DriveCredentialError`` so the retry loop's
        ``is_permanent_error`` check fast-fails the upload with an
        operator-actionable message instead of burning the full retry
        budget on a credential the server will never accept.
        """
        if self._creds is None:
            return
        with self._refresh_lock:
            if self._creds.expired or not self._creds.valid:
                from google.auth.exceptions import RefreshError
                from google.auth.transport.requests import Request

                try:
                    self._creds.refresh(Request())
                except RefreshError as exc:
                    token_path = self.token_enc if not self._legacy_mode else self._token_path
                    logger.error(
                        "OAuth refresh failed (permanent) for token %s: %s",
                        token_path,
                        exc,
                    )
                    raise DriveCredentialError(
                        f"OAuth token refresh failed for {token_path}: {exc}\n"
                        f"Recovery: run `gomoveshift upload --init-oauth "
                        f"<project_dir>` to re-issue the refresh token, "
                        f"or check the SOPS-encrypted token if it is itself "
                        f"corrupted (the age key named by GMS_AGE_KEY_FILE "
                        f"must be able to decrypt it)."
                    ) from exc
                logger.debug("  Credentials refreshed (thread-safe)")

    def _get_credentials(self):
        """Load and refresh OAuth credentials. Returns Credentials object.

        Mirrors ``_refresh_credentials_if_needed`` — wrap
        ``RefreshError`` in ``DriveCredentialError`` with the recovery
        hint, so the initial-load path fails identically to the
        mid-run refresh path.
        """
        if self._creds is not None:
            return self._creds

        self._ensure_credentials()

        from google.auth.exceptions import RefreshError
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials

        if self._legacy_mode:
            logger.warning(
                "DEPRECATED: PipelineUploader legacy plaintext token mode is in use "
                "(token_path=%s). Migrate to the SOPS-encrypted token "
                "(token_enc) — see `gomoveshift upload --init-oauth`.",
                self._token_path,
            )
            with open(self._token_path) as f:
                token_data = json.load(f)
        else:
            token_data = sops_decrypt_json(self.token_enc)

        creds = Credentials(
            token=token_data.get("token"),
            refresh_token=token_data.get("refresh_token"),
            token_uri=token_data.get("token_uri"),
            client_id=token_data.get("client_id"),
            client_secret=token_data.get("client_secret"),
            scopes=token_data.get("scopes"),
        )

        if creds.expired or not creds.valid:
            try:
                creds.refresh(Request())
            except RefreshError as exc:
                token_path = self.token_enc if not self._legacy_mode else self._token_path
                logger.error(
                    "OAuth refresh failed (permanent) at load for token %s: %s",
                    token_path,
                    exc,
                )
                raise DriveCredentialError(
                    f"OAuth token refresh failed for {token_path}: {exc}\n"
                    f"Recovery: run `gomoveshift upload --init-oauth "
                    f"<project_dir>` to re-issue the refresh token, "
                    f"or check the SOPS-encrypted token if it is itself "
                    f"corrupted (the age key named by GMS_AGE_KEY_FILE "
                    f"must be able to decrypt it)."
                ) from exc
            token_data["token"] = creds.token
            if creds.expiry:
                token_data["expiry"] = creds.expiry.isoformat()

            if self._legacy_mode:
                # Atomic create-with-mode: race-free, file never exists with
                # umask-default 0o644. OAuth refresh tokens grant indefinite
                # API access — they MUST be 0o600 (owner-only). Belt-and-
                # braces chmod afterwards covers the case where the path
                # already existed with a wider mode (open's mode arg only
                # applies on O_CREAT-create, not on truncate of existing).
                fd = os.open(
                    self._token_path,
                    os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                    0o600,
                )
                with os.fdopen(fd, "w") as f:
                    json.dump(token_data, f, indent=2)
                os.chmod(self._token_path, 0o600)
            else:
                sops_encrypt_json(token_data, self.token_enc)
                logger.info("Refreshed token re-encrypted to %s", self.token_enc)

        self._creds = creds
        logger.info("Google Drive authenticated successfully (direct API)")
        return self._creds

    def _build_drive_service(self):
        """Build a new Drive service with its own HTTP transport.

        Each call creates a fresh httplib2 transport, making it safe for
        concurrent use from multiple threads. The underlying credentials
        object is shared but thread-safe refresh is handled by _refresh_lock.
        """
        from googleapiclient.discovery import build

        creds = self._get_credentials()
        return build("drive", "v3", credentials=creds)

    def _get_client(self):
        """Get or create the main (non-threaded) Drive client."""
        if self._client is not None:
            return self._client
        self._client = self._build_drive_service()
        return self._client

    def init_oauth(self) -> bool:
        """Run initial OAuth browser flow and SOPS-encrypt the resulting token.

        Requires drive_credentials.json.enc to already be encrypted.
        This is a one-time setup step.
        """
        from google_auth_oauthlib.flow import InstalledAppFlow

        if self._legacy_mode:
            raise RuntimeError("init_oauth requires SOPS mode (no --credentials/--token)")

        if not self.credentials_enc.exists():
            raise FileNotFoundError(
                f"Encrypted credentials not found: {self.credentials_enc}\n"
                f"Encrypt your OAuth client JSON first:\n"
                f"  sops {self.credentials_enc}"
            )

        client_config = sops_decrypt_json(self.credentials_enc)

        # google_auth_oauthlib expects the wrapper format {"installed": {...}}
        if "installed" not in client_config and "web" not in client_config:
            client_config = {"installed": client_config}

        flow = InstalledAppFlow.from_client_config(
            client_config,
            scopes=["https://www.googleapis.com/auth/drive.file"],
        )
        creds = flow.run_local_server(port=0)

        token_data = {
            "token": creds.token,
            "refresh_token": creds.refresh_token,
            "token_uri": creds.token_uri,
            "client_id": creds.client_id,
            "client_secret": creds.client_secret,
            "scopes": creds.scopes,
        }
        if creds.expiry:
            token_data["expiry"] = creds.expiry.isoformat()

        sops_encrypt_json(token_data, self.token_enc)
        logger.info("OAuth token encrypted to %s", self.token_enc)
        return True

    def upload_shorts(
        self,
        shorts_dir,
        project_name: str = "",
        manifest_path: Optional[str] = None,
        drive_subfolder: Optional[str] = None,
        make_shareable: bool = True,
        max_workers: int = 2,
        *,
        folder_id: Optional[str] = None,
    ):
        """Upload shorts to Google Drive (backwards-compatible dispatcher).

        Delegates to upload_shorts_by_paths or upload_shorts_from_dir based on
        the arguments.

        Two calling conventions:
          1. List-based (new): upload_shorts([path1, path2], folder_id="xyz")
             Returns List[str] with one Drive link per file.
          2. Directory-based (legacy): upload_shorts(shorts_dir, project_name)
             Returns UploadManifest with file->link mapping.
        """
        if folder_id is not None or isinstance(shorts_dir, list):
            file_paths: List[str] = (
                list(shorts_dir) if isinstance(shorts_dir, list) else [shorts_dir]
            )
            return self.upload_shorts_by_paths(
                file_paths,
                folder_id=folder_id or "",
                make_shareable=make_shareable,
            )

        return self.upload_shorts_from_dir(
            shorts_dir=shorts_dir,
            project_name=project_name,
            manifest_path=manifest_path,
            drive_subfolder=drive_subfolder,
            make_shareable=make_shareable,
            max_workers=max_workers,
        )

    def upload_shorts_by_paths(
        self,
        paths: List[str],
        folder_id: str,
        make_shareable: bool = True,
    ) -> List[str]:
        """Upload a list of file paths to a specific Drive folder.

        Each file is uploaded sequentially with SSL retry (3 attempts per file).

        Args:
            paths: List of local file paths to upload.
            folder_id: Google Drive folder ID to upload into.
            make_shareable: Make files publicly shareable for Metricool.

        Returns:
            List of Drive links, one per input path.
        """
        drive = self._build_drive_service()

        def _upload_one_with_retry(path: str) -> str:
            last_exc = None
            for attempt, base_delay in enumerate(_SSL_RETRY_DELAYS, 1):
                try:
                    return self._upload_file(
                        drive,
                        Path(path),
                        folder_id,
                        make_shareable,
                    )
                except _SSL_ERRORS as exc:
                    last_exc = exc
                    if attempt == len(_SSL_RETRY_DELAYS):
                        break  # last attempt: nothing left to wait for
                    jitter = 0.5 + random.random()  # [0.5, 1.5)
                    delay = base_delay * jitter
                    logger.warning(
                        "SSL error upload attempt %d/%d for %s: %s — retrying in %.1fs",
                        attempt,
                        len(_SSL_RETRY_DELAYS),
                        path,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
            raise DriveUploadError(f"SSL retry exhausted for {path}: {last_exc}") from last_exc

        results = []
        for fp in paths:
            results.append(_upload_one_with_retry(fp))
        return results

    def upload_shorts_from_dir(
        self,
        shorts_dir: str,
        project_name: str = "",
        manifest_path: Optional[str] = None,
        drive_subfolder: Optional[str] = None,
        make_shareable: bool = True,
        max_workers: int = 2,
    ) -> UploadManifest:
        """Upload all .mp4 shorts from a directory to Google Drive.

        Supports resume via a manifest JSON saved after each file upload.

        Args:
            shorts_dir: Directory containing rendered .mp4 shorts.
            project_name: Project name for manifest metadata.
            manifest_path: Path for manifest JSON (default: shorts_dir/gdrive_manifest.json).
            drive_subfolder: Optional subfolder under the main drive folder.
            make_shareable: Make files publicly shareable for Metricool.
            max_workers: Max parallel upload threads.

        Returns:
            UploadManifest with file->link mapping.
        """
        shorts_dir = Path(shorts_dir)
        if not shorts_dir.exists():
            raise FileNotFoundError(f"Shorts directory not found: {shorts_dir}")

        # Find all MP4 files
        all_files = sorted([f.name for f in shorts_dir.glob("*.mp4")])
        if not all_files:
            raise FileNotFoundError(f"No .mp4 files found in {shorts_dir}")

        # Load or create manifest
        if manifest_path is None:
            manifest_path = shorts_dir / "gdrive_manifest.json"
        else:
            manifest_path = Path(manifest_path)

        if manifest_path.exists():
            manifest = UploadManifest.load(manifest_path)
            logger.info(f"Resuming upload: {len(manifest.files)}/{len(all_files)} already uploaded")
        else:
            manifest = UploadManifest(
                project_name=project_name,
                created_at=datetime.now().isoformat(),
            )

        # Determine which files still need uploading
        pending = manifest.get_pending_files(all_files)
        if not pending:
            logger.info("All files already uploaded!")
            manifest.completed = True
            manifest.save(manifest_path)
            return manifest

        logger.info(f"Uploading {len(pending)}/{len(all_files)} shorts to Google Drive...")

        # Get/create Drive folder (also forces token refresh before parallel uploads)
        drive = self._get_client()

        # Pre-flight: force credential refresh to avoid concurrent 401 refreshes
        # in parallel upload threads (which can corrupt memory)
        if self._creds and self._creds.refresh_token:
            from google.auth.transport.requests import Request

            try:
                self._creds.refresh(Request())
            except Exception as e:
                # Silent swallow would hide network/credential failures from
                # operators. Emit a WARNING naming the preferred path
                # (pre-flight refresh) and the fallback ("will retry on first
                # API call") so the degradation is visible in logs — even
                # though parallel upload threads may still race on refresh.
                logger.warning(
                    "  Drive pre-flight credential refresh failed (%s) — "
                    "falling back to per-thread refresh on first API call",
                    e,
                )

        folder_path = self.drive_folder
        if drive_subfolder:
            folder_path = f"{folder_path}/{drive_subfolder}"

        if manifest.folder_id:
            dir_folder_id = manifest.folder_id
            logger.info(f"Using existing folder: {dir_folder_id}")
        else:
            logger.info(f"Getting/creating folder: {folder_path}")
            dir_folder_id = self._get_or_create_folder_path(drive, folder_path)
            manifest.folder_id = dir_folder_id
            manifest.save(manifest_path)

        # Upload pending files in parallel — each thread gets its own Drive
        # service to avoid sharing httplib2 transports (not thread-safe,
        # causes SSL corruption / segfaults under concurrent uploads).
        manifest_lock = threading.Lock()
        _thread_local = threading.local()

        def _get_thread_drive():
            if not hasattr(_thread_local, "drive"):
                _thread_local.drive = self._build_drive_service()
            return _thread_local.drive

        def _upload_one(filename):
            thread_drive = _get_thread_drive()
            file_path = shorts_dir / filename
            logger.info(f"  Uploading {filename} ({file_path.stat().st_size // 1024 // 1024}MB)...")
            link = self._upload_file(thread_drive, file_path, dir_folder_id, make_shareable)
            with manifest_lock:
                manifest.files[filename] = link
                manifest.save(manifest_path)
            logger.info(f"    ✅ {filename}: {link}")
            return filename, link

        failed = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_upload_one, f): f for f in pending}
            for future in as_completed(futures):
                try:
                    future.result()  # raises on failure
                except Exception as e:
                    filename = futures[future]
                    logger.error(f"    ❌ {filename} failed: {e}")
                    logger.info("    Run again to resume from here.")
                    failed.append((filename, e))

        # Save manifest with all successful uploads before raising
        with manifest_lock:
            manifest.save(manifest_path)

        if failed:
            raise RuntimeError(f"{len(failed)} upload(s) failed: {failed[0][1]}")

        # Mark complete
        manifest.completed = True
        manifest.save(manifest_path)

        logger.info(f"Upload complete: {len(manifest.files)} files uploaded to {folder_path}")

        return manifest

    def _find_or_create_folder(self, drive, name: str, parent_id: Optional[str] = None) -> str:
        """Find or create a single Drive folder by name."""
        q = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        if parent_id:
            q += f" and '{parent_id}' in parents"
        results = _drive_execute(drive.files().list(q=q, fields="files(id, name)", spaces="drive"))
        files = results.get("files", [])
        if files:
            return files[0]["id"]
        # Create folder
        meta = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
        if parent_id:
            meta["parents"] = [parent_id]
        f = _drive_execute(drive.files().create(body=meta, fields="id"))
        logger.info(f"  Created Drive folder: {name}")
        return f["id"]

    def _get_or_create_folder_path(self, drive, folder_path: str) -> str:
        """Navigate/create a nested folder path like 'GoMoveShift/Videos/Shorts'."""
        parent_id = None
        for part in folder_path.split("/"):
            parent_id = self._find_or_create_folder(drive, part, parent_id)
        return parent_id

    @staticmethod
    def _next_chunk_with_timeout(request, timeout=600):
        """Call request.next_chunk() with a per-chunk timeout.

        Prevents indefinite hangs on stalled Google Drive API connections.
        Uses a thread pool to enforce the timeout since httplib2 does not
        support per-request timeouts natively.

        Uses an explicit (not ``with``-managed) pool: the ``with`` form's
        ``__exit__`` calls ``shutdown(wait=True)``, which on a timeout blocks
        the caller until the orphaned ``next_chunk()`` worker returns — and on
        the documented Drive SSL-stall class it never does, so the timeout
        could never propagate. On timeout we ``shutdown(wait=False,
        cancel_futures=True)`` so the caller is freed immediately and the
        in-flight socket is abandoned; the success path shuts down normally so
        threads are not leaked.
        """
        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(request.next_chunk)
        try:
            result = future.result(timeout=timeout)
        except FuturesTimeoutError:
            pool.shutdown(wait=False, cancel_futures=True)
            raise TimeoutError(f"Drive chunk upload exceeded {timeout}s timeout") from None
        pool.shutdown(wait=True)
        return result

    def _find_existing_drive_file(
        self, drive, name: str, folder_id: str
    ) -> Optional[Dict[str, str]]:
        """Return ``{"id", "size"}`` for a matching non-trashed file, or
        ``None``. Fails open (falls back to create) on any error or
        non-Drive-shaped response — docs/pipeline-internals.md."""
        q = (
            f"name='{_escape_drive_query_value(name)}' "
            f"and '{folder_id}' in parents and trashed=false"
        )
        try:
            results = _drive_execute(
                drive.files().list(q=q, fields="files(id, size)", spaces="drive")
            )
        except Exception:
            return None
        files = results.get("files") if isinstance(results, dict) else None
        if not files or not isinstance(files, list):
            return None
        first = files[0]
        if not isinstance(first, dict) or "id" not in first:
            return None
        return first

    def _ensure_shareable(self, drive, file_id: str) -> None:
        """Grant public-read access, tolerating an "already shared" error —
        docs/pipeline-internals.md for why create() beats list-first."""
        try:
            _drive_execute(
                drive.permissions().create(
                    fileId=file_id,
                    body={"type": "anyone", "role": "reader"},
                )
            )
        except Exception as exc:
            if "already" in str(exc).lower():
                logger.debug(f"    {file_id}: already publicly shared — {exc}")
                return
            raise

    def _upload_file(
        self,
        drive,
        file_path: Path,
        folder_id: str,
        make_shareable: bool = True,
        mimetype: Optional[str] = None,
    ) -> str:
        """Upload a single file to Drive and return a download link.

        Idempotent per (name, folder): a same-named, same-size file is
        reused with no re-upload; a same-named, different-size file (a
        genuine re-render) is replaced IN PLACE via ``files().update`` on
        the same file id — never a second ``create()``. Sharing is
        (re-)applied on every path so a retry after a prior sharing failure
        fixes the dead link rather than returning it as success. Known
        limit (accepted) and mechanism: docs/pipeline-internals.md.
        """
        from googleapiclient.http import MediaFileUpload

        # Thread-safe credential refresh before API call
        self._refresh_credentials_if_needed()

        if mimetype is None:
            mimetype = _guess_mimetype(file_path)

        existing = self._find_existing_drive_file(drive, file_path.name, folder_id)
        local_size = file_path.stat().st_size
        reused_unchanged = False
        if existing is not None and existing.get("size") is not None:
            try:
                remote_size = int(existing["size"])
            except (TypeError, ValueError):
                remote_size = None
            reused_unchanged = remote_size == local_size

        if reused_unchanged:
            file_id = existing["id"]
            logger.info(
                f"    {file_path.name}: identical file already on Drive "
                f"(id={file_id}) — reusing, skipping re-upload"
            )
        else:
            media = MediaFileUpload(
                str(file_path),
                mimetype=mimetype,
                resumable=True,
                chunksize=256 * 1024 * 1024,
            )

            if existing is not None:
                request = drive.files().update(fileId=existing["id"], media_body=media, fields="id")
            else:
                file_meta = {"name": file_path.name, "parents": [folder_id]}
                request = drive.files().create(body=file_meta, media_body=media, fields="id")

            response = None
            last_progress_time = time.time()
            while response is None:
                status, response = self._next_chunk_with_timeout(request)
                if status:
                    logger.debug(f"    Upload progress: {int(status.progress() * 100)}%")
                    last_progress_time = time.time()
                elif time.time() - last_progress_time > _DRIVE_STALL_TIMEOUT:
                    raise RuntimeError(
                        f"Upload stalled: no progress for {_DRIVE_STALL_TIMEOUT}s ({file_path.name})"
                    )

            file_id = response["id"]

        if make_shareable:
            self._ensure_shareable(drive, file_id)

        return drive_download_link(file_id)

    def upload_photos(
        self,
        photo_paths: List[str],
        folder: str = DEFAULT_PHOTO_DRIVE_FOLDER,
        manifest_path: Optional[str] = None,
        make_shareable: bool = True,
    ) -> UploadManifest:
        """Upload photo files to Google Drive.

        Args:
            photo_paths: List of paths to photo files
            folder: Drive folder path (created if needed)
            manifest_path: Path for manifest JSON
            make_shareable: Make files publicly shareable for Metricool

        Returns:
            UploadManifest with file->link mapping
        """
        if not photo_paths:
            raise ValueError("No photo paths provided")

        # Load or create manifest
        if manifest_path:
            manifest_path = Path(manifest_path)
        else:
            manifest_path = Path(photo_paths[0]).parent / "photo_gdrive_manifest.json"

        if manifest_path.exists():
            manifest = UploadManifest.load(manifest_path)
            logger.info(f"Resuming photo upload: {len(manifest.files)} already uploaded")
        else:
            manifest = UploadManifest(
                project_name="daily_photos",
                created_at=datetime.now().isoformat(),
            )

        # Filter to pending
        all_names = [Path(p).name for p in photo_paths]
        pending_paths = [p for p in photo_paths if Path(p).name not in manifest.files]

        if not pending_paths:
            logger.info("All photos already uploaded!")
            manifest.completed = True
            manifest.save(manifest_path)
            return manifest

        logger.info(f"Uploading {len(pending_paths)}/{len(all_names)} photos to Drive...")

        drive = self._get_client()

        if manifest.folder_id:
            folder_id = manifest.folder_id
        else:
            folder_id = self._get_or_create_folder_path(drive, folder)
            manifest.folder_id = folder_id
            manifest.save(manifest_path)

        for photo_path_str in pending_paths:
            photo_path = Path(photo_path_str)
            logger.info(f"  Uploading {photo_path.name}...")
            link = self._upload_file(drive, photo_path, folder_id, make_shareable)
            manifest.files[photo_path.name] = link
            manifest.save(manifest_path)
            logger.info(f"    {photo_path.name}: {link}")

        manifest.completed = True
        manifest.save(manifest_path)
        logger.info(f"Photo upload complete: {len(manifest.files)} files")
        return manifest

    def verify_uploads(self, manifest: UploadManifest) -> VerificationResult:
        """Verify that uploaded Drive links are still accessible.

        Extracts file_id from each Drive link and calls files().get()
        to confirm the file exists and is reachable.

        Args:
            manifest: Upload manifest with file→link mapping

        Returns:
            VerificationResult with verified and failed filename lists
        """
        import re

        drive = self._get_client()
        verified = []
        failed = []

        for filename, link in manifest.files.items():
            match = re.search(r"/d/([a-zA-Z0-9_-]+)/|[?&]id=([a-zA-Z0-9_-]+)", link)
            if not match:
                logger.warning(f"  Cannot extract file_id from link: {link}")
                failed.append(filename)
                continue

            file_id = match.group(1) or match.group(2)
            try:
                _drive_execute(drive.files().get(fileId=file_id, fields="id,name"))
                verified.append(filename)
                logger.debug(f"  Verified: {filename} ({file_id})")
            except Exception as e:
                logger.warning(f"  Dead link for {filename} ({file_id}): {e}")
                failed.append(filename)

        logger.info(
            f"Verification: {len(verified)} OK, {len(failed)} failed "
            f"out of {len(manifest.files)} files"
        )
        return VerificationResult(verified=verified, failed=failed)

    def upload_single_file(
        self,
        file_path: str,
        drive_subfolder: Optional[str] = None,
        make_shareable: bool = True,
    ) -> str:
        """
        Upload a single file to Google Drive.

        Args:
            file_path: Path to the file
            drive_subfolder: Optional subfolder under the main drive folder
            make_shareable: Make file publicly shareable

        Returns:
            Drive download link
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        drive = self._get_client()

        folder_path = self.drive_folder
        if drive_subfolder:
            folder_path = f"{folder_path}/{drive_subfolder}"

        folder_id = self._get_or_create_folder_path(drive, folder_path)

        link = self._upload_file(drive, file_path, folder_id, make_shareable)

        logger.info(f"Uploaded {file_path.name} → {link}")
        return link
