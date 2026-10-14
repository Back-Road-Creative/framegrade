"""Test configuration for the gomoveshift package.

Skips test collection if the gomoveshift package is not installed,
which happens in CI environments where the full dependency chain
(gps_journey, etc.) is not available.
"""

import atexit
import os
import pwd
import shutil
import socket
import tempfile

# This suite does not use Hypothesis, but its pytest plugin loads whenever the
# library is installed and would create ``.hypothesis/`` in the tree root.
# Point it at the ignored ``.cache/`` directory instead. Set before hypothesis
# is imported anywhere in the session.
os.environ.setdefault(
    "HYPOTHESIS_STORAGE_DIRECTORY",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "hypothesis"
    ),
)

# Tests can never reach production secrets -- not even by accident.
#
# ``GMS_DRIVE_CREDENTIALS_ENC`` / ``GMS_DRIVE_TOKEN_ENC`` /
# ``GMS_AGE_KEY_FILE`` name the Drive credential files. A per-test
# ``monkeypatch.setenv`` fixture -- however "autouse" -- runs at test
# *setup*, which is always after pytest's collection phase has already
# imported every test module (and everything those modules import). Setting
# these here, as plain module-level code executed before ``import gomoveshift``
# below, protects anything that reads the variables while pytest collects and
# imports modules, before any fixture runs, so it can never see the real
# secret paths -- the failure mode where a runner with no read access to the
# default secrets directory fails tests with ``PermissionError``.
_TEST_SECRETS_DIR = tempfile.mkdtemp(prefix="gms-test-secrets-")
for _name, _filename in (
    ("GMS_DRIVE_CREDENTIALS_ENC", "drive_credentials.json.enc"),
    ("GMS_DRIVE_TOKEN_ENC", "drive_token.json.enc"),
    ("GMS_AGE_KEY_FILE", "age.key"),
):
    os.environ[_name] = os.path.join(_TEST_SECRETS_DIR, _filename)


# Same hazard, different surface: ``gomoveshift/pipeline/longform_metadata.py``
# calls ``Config.load()`` at IMPORT time (``DEFAULT_SEO_STORAGE = ...``), and
# ``Config.load()`` deep-merges ``Path.home() / ".config" / "gomoveshift" /
# "config.yaml"`` as its highest-precedence layer. A fixture -- however
# "autouse" and session-scoped -- runs at test *setup*, which is always after
# pytest's collection phase has already imported every test module and
# everything they import. By then the import-time ``Config.load()`` call
# above has already read the developer's REAL ``~/.config/gomoveshift/config.yaml``,
# and a branch that hard-rejects a real override would make ``Config.load()``
# raise during collection, before any fixture ever ran. Setting ``HOME`` here,
# before ``import gomoveshift`` below, makes it structurally impossible for
# collection-time code to see the real config.
# The fake HOME lives under the real ``~/.cache``, never ``/tmp``: product
# defaults such as ``paths._DEFAULT_INTERMEDIATE_DIR`` resolve under the home
# directory and are pinned by tests to stay off tmpfs (``test_intermediate_paths``,
# ``test_route_story_frame_cache``, ``test_route_story_located_beat``).
#
# The per-user application homes follow ``GMS_*_HOME`` and ``XDG_*_HOME`` ahead
# of ``~``; a machine exporting any of them would otherwise resolve the
# product defaults into the real user's directories, so they are dropped here.
def _make_test_home() -> str:
    # passwd home, not ``$HOME``: an xdist worker inherits the controller's fake HOME.
    parent = os.path.join(pwd.getpwuid(os.getuid()).pw_dir, ".cache", "gms-test-homes")
    try:
        os.makedirs(parent, exist_ok=True)
        home = tempfile.mkdtemp(prefix="gms-test-home-", dir=parent)
    except OSError:
        home = tempfile.mkdtemp(prefix="gms-test-home-")
    atexit.register(shutil.rmtree, home, True)
    return home


_TEST_HOME = _make_test_home()
os.environ["HOME"] = _TEST_HOME
for _home_var in (
    "GMS_DATA_HOME",
    "GMS_CACHE_HOME",
    "GMS_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
):
    os.environ.pop(_home_var, None)

try:
    import gomoveshift  # noqa: F401

    HAS_GOMOVESHIFT = True
except ImportError:
    HAS_GOMOVESHIFT = False
    import warnings

    warnings.warn(
        "gomoveshift package not installed — ALL tests will be skipped. "
        "Run: pip install -e . (from the repository root)",
        stacklevel=1,
    )


from pathlib import Path

import pytest


@pytest.fixture(autouse=True, scope="session")
def isolate_home_from_operator_config():
    """Mirror the module-level ``HOME`` blind onto ``Path.home`` directly.

    The actual isolation mechanism is the module-level block above
    (``_TEST_HOME`` / ``os.environ["HOME"]``), set before ``import
    gomoveshift`` -- necessary because ``gomoveshift/pipeline/longform_metadata.py``
    calls ``Config.load()`` at IMPORT time, and a fixture (however
    "autouse") only runs at test *setup*, which is always after pytest's
    collection phase has already imported every test module. A fixture
    can never reach back into collection.

    This fixture exists only for code that reads ``Path.home()`` directly
    rather than going through ``HOME``; it patches ``Path.home`` to return
    the SAME ``_TEST_HOME`` directory the module-level block already set,
    so import-time and test-time code observe one consistent fake home.
    Uses ``pytest.MonkeyPatch`` directly (not the ``monkeypatch`` fixture,
    which is function-scoped) and undoes it at session end. Individual
    tests that need a *specific* user-config scenario still control it
    explicitly via ``monkeypatch.setattr(Path, "home", ...)``, same
    mechanism as ``tests/test_config_deep_merge.py``.
    """
    mp = pytest.MonkeyPatch()
    fake_home = Path(_TEST_HOME)
    mp.setattr(Path, "home", classmethod(lambda cls: fake_home))
    yield
    mp.undo()


@pytest.fixture(autouse=True, scope="session")
def isolate_worktree_lease_guard_from_environment():
    """Keep the caller's environment from steering the worktree lease guard.

    ``orchestrator_lock._acquire_worktree_lease`` shells out to the
    executable named by ``GMS_LEASE_BIN`` when it is set and the suite runs
    from a disposable worktree; ``GMS_DISPOSABLE_WORKTREE_DIRS`` widens which
    directory names count as disposable. Any test that exercises the live
    ``run()`` path (e.g. ``test_gpu_lock.py``) would then take a real lease for
    the full TTL. Clearing both variables for the session means no lease is
    taken and nothing shells out; the tests that cover the lease guard set
    their own values per test.
    """
    mp = pytest.MonkeyPatch()
    mp.delenv("GMS_LEASE_BIN", raising=False)
    mp.delenv("GMS_DISPOSABLE_WORKTREE_DIRS", raising=False)
    yield
    mp.undo()


@pytest.fixture(autouse=True, scope="session")
def isolate_media_admission_from_operator(tmp_path_factory):
    """Keep real flock tests independent of other xdist workers and real media jobs."""
    mp = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("media_admission")
    mp.setenv("GMS_NIGHT_ADMISSION_LOCK", str(root / "heavy.lock"))
    yield
    mp.undo()


@pytest.fixture(autouse=True)
def isolate_output_root_from_operator(tmp_path, monkeypatch):
    """Root ``gms_output_dir()``, and with it every default-constructed
    ``TrackingStore``, inside the test's own temp dir.

    Unset, ``GMS_OUTPUT_DIR`` falls back to the real output root under the
    data home, and the suite may run as the same user that owns it.
    ``enqueue-plan --apply`` registers every enqueued short in the default
    store, so an ``--apply`` test without an override of its own would write
    a ``proj`` project into the live store, where a metrics run would collect
    it as a real project. Per test, not per session, so no test sees
    another's records; a test that sets ``GMS_OUTPUT_DIR`` itself still wins.
    """
    monkeypatch.setenv("GMS_OUTPUT_DIR", str(tmp_path / "gms-output"))


@pytest.fixture(autouse=True)
def no_graded_raster_disk_cache(monkeypatch):
    """The route-story grade cache is a process-crossing disk cache keyed on
    raster bytes, so two tests that grade identical synthetic rasters would
    share an entry and one would never call the grade it counts. Off for
    every test; ``test_route_story_grade_cache.py`` turns it back on in its
    own temp dir."""
    monkeypatch.setenv("GMS_ROUTE_STORY_GRADE_CACHE", "0")


@pytest.fixture(autouse=True)
def auto_approve_ost_gate(monkeypatch):
    """Auto-approve OST gate prompts in tests to prevent stdin reads."""
    if not HAS_GOMOVESHIFT:
        return
    from gomoveshift.pipeline.review_gate import ReviewGate

    monkeypatch.setattr(ReviewGate, "_prompt_required", lambda self, msg: None)


@pytest.fixture(autouse=True)
def pin_free_ram_probe(monkeypatch):
    """Pin the free-RAM probe so no test reads the host's real memory.

    ``preflight.check_ram`` exists to stop the real pipeline on a starved
    box. In a test it guards nothing, and on a loaded machine it turns unrelated
    ingest/orchestrator tests red. Every test
    sees 64 GB free; a test that wants a specific reading patches
    ``_virtual_memory`` itself, and its later monkeypatch wins.
    """
    if not HAS_GOMOVESHIFT:
        return
    from gomoveshift.pipeline import preflight

    class _PinnedMem:
        available = 64 * 1_073_741_824

    monkeypatch.setattr(preflight, "_virtual_memory", lambda: _PinnedMem)


@pytest.fixture(autouse=True)
def pin_lf_insert_clock(request, monkeypatch):
    """Pin the long-form insert loop's clock to 2000-01-01 UTC.

    The loop refuses a pending LF whose slot has passed
    (``LF_SLOT_PASSED``). Fixtures across the suite carry literal 2026 slot
    dates that are real past dates by the time the tests run, so on the real
    clock they would all refuse. A module that sets ``REAL_LF_INSERT_CLOCK =
    True`` (``test_lf_insert_slot_guard``) runs on the real clock; any test
    may still patch ``_utc_now`` or pass ``clock=``.
    """
    if not HAS_GOMOVESHIFT or getattr(request.module, "REAL_LF_INSERT_CLOCK", False):
        return
    from datetime import datetime, timezone

    from gomoveshift.pipeline.drive_phases import external_systems_coordinator as esc

    monkeypatch.setattr(esc, "_utc_now", lambda: datetime(2000, 1, 1, tzinfo=timezone.utc))


@pytest.fixture(autouse=True)
def pin_blur_session_probe(monkeypatch):
    """Pin the blur preflight's real-session probe to "CUDA bound".

    ``check_blur_inference_provider`` opens a real ``InferenceSession`` to prove
    CUDA loads; in a test that reads the host's GPU and CUDA libs, and on a
    host whose CUDA libs don't load it would turn every blur-enabled photo
    ingest/preflight test red. Gate tests pass
    ``session_providers`` explicitly; the probe's own test imports the
    unpatched function.
    """
    if not HAS_GOMOVESHIFT:
        return
    from gomoveshift.pipeline import preflight

    monkeypatch.setattr(
        preflight,
        "_probe_session_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )


@pytest.fixture(autouse=True)
def auto_skip_preflight(monkeypatch):
    """Skip pre-flight validation in tests — checked separately in test_preflight.py."""
    if not HAS_GOMOVESHIFT:
        return
    from gomoveshift.pipeline import orchestrator_preflight

    monkeypatch.setattr(
        orchestrator_preflight,
        "run_preflight",
        lambda **kwargs: None,
    )


@pytest.fixture(autouse=True)
def auto_pass_film_delivery_check(monkeypatch):
    """Tests drive the route-story film command with a faked render and encode
    that write placeholder bytes, not a film -- so the command's read-back of
    its own output is stubbed to a clean report. The check itself, and the
    command's handling of what it reports, are tested in
    ``test_route_story_delivery.py``, which installs its own.
    """
    if not HAS_GOMOVESHIFT:
        return
    from gomoveshift.pipeline.route_story import film

    monkeypatch.setattr(film, "check_delivery", lambda path, **spec: {"issues": []})


@pytest.fixture(autouse=True)
def auto_skip_seo_config_validation(request, monkeypatch):
    """Stub ``SEOConfig.validate_paths_exist`` to a no-op so tests don't
    depend on a co-located SEO service checkout.

    The runtime gate (wired in ``cmd_drive`` / ``cmd_pipeline``) verifies the
    SEO paths exist on disk and refuses to start when they don't. CI containers
    and fresh checkouts may not have the co-located checkout populated, so we
    no-op the assertion at the class level for the test session.

    Opt-out: a test (or its module) that needs the real validator marks itself
    with ``@pytest.mark.real_seo_validation`` — the fixture leaves the class
    method untouched and the test exercises the production path.
    """
    if not HAS_GOMOVESHIFT:
        return
    if request.node.get_closest_marker("real_seo_validation"):
        return
    from gomoveshift.config import SEOConfig

    monkeypatch.setattr(SEOConfig, "validate_paths_exist", lambda self, strict=True: None)


@pytest.fixture(autouse=True)
def isolate_publishing_spool_from_workspace(tmp_path, monkeypatch):
    """No test may resolve ``Config.load().seo.storage_path`` onto the
    live publishing spool.

    A deployed ``config.yaml`` can pin ``seo.storage_path`` to a real
    publishing spool (an absolute path,
    required by ``SEOConfig.validate_paths_exist``). Any test that
    reaches ``Config.load()`` without its own per-test override -- e.g.
    a console route that falls back to ``Config.load()`` when no config
    is injected -- opens that live SQLite spool and its schema lock,
    which a different OS user sharing this tree may not have permission to
    write (``PermissionError`` at ``publishing/queue.py``'s ``_schema_lock``).

    Wraps ``Config.load`` rather than the YAML search path: this is the
    smallest seam that preserves every other precedence/merge behaviour
    (see ``Config.load``'s docstring) while making the one
    disk-resolving field unable to point at a real publishing spool.
    """
    if not HAS_GOMOVESHIFT:
        return
    from gomoveshift.config import Config

    original_load = Config.load.__func__

    def _isolated_load(cls, config_path=None):
        config = original_load(cls, config_path)
        config.seo.storage_path = str(tmp_path)
        return config

    monkeypatch.setattr(Config, "load", classmethod(_isolated_load))


@pytest.fixture(autouse=True)
def isolate_seo_storage_from_workspace(tmp_path_factory, monkeypatch):
    """Redirect ``DEFAULT_SEO_STORAGE`` / ``DEFAULT_CREDENTIALS_DIR`` to
    a per-session tempdir.

    Belt-and-suspenders against tests that construct
    ``LongFormMetadataGenerator()`` without an explicit
    ``seo_storage_path=`` argument. ``longform_metadata.py`` resolves the
    fallback at import time via ``Path(_SEOConfig().storage_path)``. An
    absolute default pointing at a real deployment would let a hung test
    hold a real ``filelock`` on that deployment's ``events.jsonl``.

    The ``config.py`` defaults are cwd-relative, which neutralises the
    immediate hazard, but a future regression in defaults would
    surface here as a noisy AttributeError rather than as silent
    contention on real data. Keep this fixture; do not remove without
    a non-conftest mechanism that guarantees no test can resolve the
    SEO storage path back to a real deployment.
    """
    if not HAS_GOMOVESHIFT:
        return
    from gomoveshift.pipeline import longform_metadata

    fake_seo = tmp_path_factory.mktemp("isolated_seo")
    monkeypatch.setattr(longform_metadata, "DEFAULT_SEO_STORAGE", fake_seo)
    monkeypatch.setattr(longform_metadata, "DEFAULT_CREDENTIALS_DIR", fake_seo / "credentials")


@pytest.fixture(autouse=True)
def isolate_theme_rotation_state(tmp_path_factory, monkeypatch):
    """Keep the SEO package's theme-rotation state out of the checkout.

    ``ThemeRotator()`` with no ``state_dir`` persists under a path relative to
    the current directory, so any test that records a theme usage would write
    ``data/theme_rotation/`` into the working tree. Give the default a
    per-test tempdir; an explicit ``state_dir=`` still wins.
    """
    try:
        from seo_manager.analytics import theme_rotator
    except ImportError:
        return

    default_dir = tmp_path_factory.mktemp("theme_rotation")
    original_init = theme_rotator.ThemeRotator.__init__

    def _init(self, state_dir=None, *args, **kwargs):
        original_init(self, state_dir or default_dir, *args, **kwargs)

    monkeypatch.setattr(theme_rotator.ThemeRotator, "__init__", _init)


@pytest.fixture(autouse=True)
def hermetic_content_ledger(request, tmp_path, monkeypatch):
    """Redirect the no-arg ``ContentLedger`` default path to a per-test temp file.

    The default resolves to
    ``<config.seo.storage_path>/.content_ledger.json`` -- on a workstation
    that is the LIVE production ledger. Any test that reaches a
    no-arg ``ContentLedger.load()`` / ``save()`` /
    ``record_confirmed_upload()`` (e.g. via ``PhotoScheduler()`` →
    ``PhotoOrchestrator``) would read live entries into slot-availability
    assertions (pass-in-CI / fail-on-workstation) or, worse, write to the
    real file (a test holding a filelock on a production file can hang CI).

    Seam: ``content_ledger._default_ledger_path`` — the single resolver
    every no-arg path goes through. Tests passing an explicit ``path=``
    are unaffected.

    Opt-out: ``@pytest.mark.real_content_ledger`` leaves the resolver
    untouched (no current test needs the real file; the marker exists so
    a future integration test can opt out explicitly instead of deleting
    this fixture). A test may also override the fixture locally.
    """
    if not HAS_GOMOVESHIFT:
        return
    if request.node.get_closest_marker("real_content_ledger"):
        return
    from gomoveshift.pipeline import content_ledger

    hermetic_path = tmp_path / ".content_ledger.json"
    monkeypatch.setattr(content_ledger, "_default_ledger_path", lambda: hermetic_path)


@pytest.fixture(autouse=True)
def restore_schedule_constants_identity():
    """Snapshot and restore ``schedule_constants`` module attrs + every
    re-exporter's binding so cross-test ``is`` identity checks survive.

    Why: ``schedule_constants.apply_config`` rebinds ``LONG_FORM_DAYS`` /
    ``LONG_FORM_TIME`` / ``SHORTS_TIMES`` via ``globals()[...] = ...`` —
    a fresh object each call. Importers that did
    ``from .schedule_constants import LONG_FORM_DAYS`` (longform_metadata,
    content_ledger, scheduler) captured the import-time originals and
    continue to point at them after the rebind. Tests in
    ``test_schedule_constants.py`` assert
    ``importer.LONG_FORM_DAYS is schedule_constants.LONG_FORM_DAYS`` —
    once any earlier test calls ``apply_config`` without a paired
    restore, that identity is broken for the remainder of the session.

    monkeypatch.setattr restores its own setattrs but NOT the
    ``globals()`` rebind ``apply_config`` performs internally, and even
    when test bodies remember to monkeypatch ``LONG_FORM_DAYS``/
    ``LONG_FORM_TIME`` they typically forget ``SHORTS_TIMES``. An
    autouse session-wide snapshot makes the failure mode unrepresentable.
    """
    if not HAS_GOMOVESHIFT:
        yield
        return
    from gomoveshift.pipeline import (
        content_ledger,
        longform_metadata,
        schedule_constants,
        scheduler,
    )

    saved = {
        "LONG_FORM_DAYS": schedule_constants.LONG_FORM_DAYS,
        "LONG_FORM_TIME": schedule_constants.LONG_FORM_TIME,
        "SHORTS_TIMES": schedule_constants.SHORTS_TIMES,
    }
    yield
    for name, value in saved.items():
        setattr(schedule_constants, name, value)
        # Re-bind the names captured at import time by re-exporters so
        # the cross-module ``is`` identity contract survives.
        for module in (longform_metadata, content_ledger, scheduler):
            if hasattr(module, name):
                setattr(module, name, value)


@pytest.fixture(autouse=True)
def hermetic_forward_geocode(request, monkeypatch):
    """Stub ``_nominatim_search`` to ``None`` and ``_nominatim_places`` to
    ``[]`` — no live Nominatim from tests.

    The pre-encode specificity ladder wires a forward geocoder
    (``phase_bc_day_first._ladder_forward_geocode``) into every production
    ``PhaseBCDayFirst.run``; an orchestrator test reaching that gate would
    otherwise attempt real Nominatim HTTP per endpoint town (1 s rate-limit
    sleep + up to 2×10 s timeouts) — minutes of wall clock and a network
    dependency. ``None`` follows the degraded-geocoder contract: the gate
    skips the proximity check. ``_resolved_location`` population likewise
    forward-geocodes label endpoint towns to verify them against the trace;
    ``[]`` verifies nothing (fail closed). Tests that stub their own
    ``geocode_fn`` / ``places_fn`` or monkeypatch either function themselves
    are unaffected.

    Opt-out: ``@pytest.mark.real_forward_geocode``.
    """
    if not HAS_GOMOVESHIFT:
        return
    if request.node.get_closest_marker("real_forward_geocode"):
        return
    from gomoveshift.pipeline.chapters import nominatim_source

    monkeypatch.setattr(nominatim_source, "_nominatim_search", lambda _q: None)
    monkeypatch.setattr(nominatim_source, "_nominatim_places", lambda _q, **_kw: [])


_LIVE_NETWORK_OPT_IN_MARKERS = (
    "real_forward_geocode",
    "real_overpass",
    "real_blur_models",
)


def _is_loopback_host(host):
    if not isinstance(host, str):
        return False
    if host == "localhost":
        return True
    import ipaddress

    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@pytest.fixture(autouse=True)
def block_non_loopback_network(request, monkeypatch):
    """Fail fast on any real outbound socket connect that isn't loopback.

    ``generator.py`` calls ``overpass_source.post_overpass_query`` directly,
    a call site the per-endpoint ``hermetic_overpass`` fixture never sees
    (it only patches ``_query_overpass``), so a test reaching terrain-band
    classification would make real HTTPS calls to the Overpass API. Per-endpoint
    stubs miss exactly this shape of leak by construction: a new call site is
    silent until someone notices minutes of wall clock. This guard is the
    structural backstop -- it patches the socket layer itself, so no test
    can reach a real non-loopback host regardless of which higher-level
    function it goes through.

    Patches ``socket.socket.connect``/``connect_ex`` and
    ``socket.create_connection``; loopback (127.0.0.0/8, ::1, localhost)
    and ``AF_UNIX`` sockets are unaffected.

    Opt-out: ``@pytest.mark.allow_network`` for an intentional live test.
    Also implied by any existing live-network opt-in marker --
    ``real_forward_geocode``, ``real_overpass``, ``real_blur_models`` --
    so opting out of one endpoint's stub doesn't also require knowing about
    this guard.
    """
    if request.node.get_closest_marker("allow_network"):
        return
    if any(request.node.get_closest_marker(name) for name in _LIVE_NETWORK_OPT_IN_MARKERS):
        return

    orig_connect = socket.socket.connect
    orig_connect_ex = socket.socket.connect_ex
    orig_create_connection = socket.create_connection

    def _check(host, family):
        if family == socket.AF_UNIX:
            return
        if _is_loopback_host(host):
            return
        raise RuntimeError(
            f"tests/conftest.py: blocked outbound network connect to {host!r} "
            "-- tests must not reach real hosts. Stub the call at its entry "
            "point (the way hermetic_overpass/hermetic_forward_geocode do), "
            "or mark the test @pytest.mark.allow_network if this is a "
            "genuinely intentional live test."
        )

    def _guard_connect(self, address):
        host = address[0] if isinstance(address, tuple) else address
        _check(host, self.family)
        return orig_connect(self, address)

    def _guard_connect_ex(self, address):
        host = address[0] if isinstance(address, tuple) else address
        _check(host, self.family)
        return orig_connect_ex(self, address)

    def _guard_create_connection(address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else address
        _check(host, family=None)
        return orig_create_connection(address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", _guard_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", _guard_connect_ex)
    monkeypatch.setattr(socket, "create_connection", _guard_create_connection)


@pytest.fixture(autouse=True)
def declare_test_harness_supervision(monkeypatch):
    """pytest IS the supervisor, so ``--no-tmux`` is legitimate in tests.

    ``cli/tmux.require_supervision`` refuses an unsupervised ``--no-tmux``
    run (see ``tests/test_no_tmux_requires_supervision.py``). Many CLI tests
    pass ``--no-tmux`` precisely because they want a foreground process whose
    exit code they can assert on — which is exactly what the flag is FOR.
    Declaring it once here keeps the gate honest in production without making
    every CLI test carry the boilerplate.
    """
    monkeypatch.setenv("GMS_SUPERVISED", "1")


@pytest.fixture(autouse=True)
def isolate_tiktok_pack_dir(monkeypatch, tmp_path):
    """Point the TikTok Studio pack at the test's own temp dir.

    ``resolve_pack_dir`` falls back to a directory under the data home. A test
    that reaches the pack code without setting ``GMS_TIKTOK_PACK_DIR`` (a new
    CLI test, a handler test left on the default ``pack`` mode) would write a
    fixture short and a caption block into it.
    Tests that exercise the pack set their own directory; the parent exists
    (``tmp_path``) and the directory itself is created on first use.
    """
    monkeypatch.setenv("GMS_TIKTOK_PACK_DIR", str(tmp_path / "tiktok-pack"))


@pytest.fixture(autouse=True)
def isolate_overpass_shared_cache(monkeypatch):
    """Disable the shared Overpass disk cache for every test by default.

    ``_query_overpass`` caches successful responses under
    ``~/.cache/gomoveshift/overpass`` so repeat runs over one drive stop
    querying the public endpoint. Left on in tests that mock ``urlopen``
    (the ``real_overpass`` opt-outs), it would write FIXTURE payloads into
    the developer's real cache, and a later test asking the same query
    would be served from disk with its mock never called.

    Disabled rather than redirected to ``tmp_path``: a per-test temp dir on
    a suite this large is pure overhead for a cache almost nothing touches,
    and a shared one would leak entries between tests. Tests that exercise
    the cache set ``GMS_OVERPASS_CACHE_DIR`` to their own directory.
    """
    monkeypatch.setenv("GMS_OVERPASS_CACHE_DIR", "")


@pytest.fixture(autouse=True)
def hermetic_overpass(request, monkeypatch):
    """Stub ``_query_overpass`` — no live Overpass HTTP from tests.

    ``hermetic_forward_geocode`` (above) already stubs Nominatim, but
    Overpass never got the same treatment: any test reaching
    ``destination_resolver``'s POI source or ``OverpassChapterSource``
    attempted three real HTTP queries per bbox (with retry sleeps) —
    which shows up as retry storms in a sandboxed run and as real API
    traffic on a networked workstation.

    The stub raises the same ``RuntimeError`` the production retry loop
    raises after exhausting attempts, so callers exercise their existing
    degraded-path contract. Both call-time bindings are patched:
    ``destination_resolver`` imports from ``overpass_source`` at call
    time; ``OverpassChapterSource`` resolves through the ``chapters``
    package attr. Tests that ``@patch`` either binding themselves are
    unaffected (their patch stacks on top and unwinds back to this stub);
    tests exercising ``_query_overpass`` internals via a direct top-level
    import binding (e.g. with a patched ``urlopen``) bypass module-attr
    lookup and keep the real function.

    Opt-out: ``@pytest.mark.real_overpass``.
    """
    if not HAS_GOMOVESHIFT:
        return
    if request.node.get_closest_marker("real_overpass"):
        return
    from gomoveshift.pipeline import chapters
    from gomoveshift.pipeline.chapters import overpass_source

    def _hermetic_query(*_args, **_kwargs):
        raise RuntimeError(
            "hermetic_overpass (tests/conftest.py): live Overpass is disabled "
            "in tests. Mark the test @pytest.mark.real_overpass to opt out."
        )

    monkeypatch.setattr(overpass_source, "_query_overpass", _hermetic_query)
    monkeypatch.setattr(chapters, "_query_overpass", _hermetic_query)


@pytest.fixture(autouse=True)
def hermetic_still_blur_models(request, monkeypatch):
    """Keep still-photo blur model resolution OFFLINE in tests.

    ``face_filter._get_detectors`` resolves the plate/face/vehicle/person ONNX
    models through ``gomoveshift.analysis.resolve_*_model`` (download-once,
    SHA256-verified by redactcam). A test reaching
    the real builder (resizer / orchestrator paths) would otherwise attempt a
    network fetch — or build real ONNX sessions from a warm workstation cache
    CI doesn't have. Stubbing face_filter's resolver BINDINGS to None keeps
    those tests on the hermetic degraded path (CenterFace fallback, no plate,
    no vehicle, no whole-person blur) while the video path's resolvers — bound
    at call time from ``gomoveshift.analysis`` directly — stay untouched. The
    detector/config caches are reset so no test inherits another's sessions.

    Opt-out: ``@pytest.mark.real_blur_models``.
    """
    if not HAS_GOMOVESHIFT:
        return
    if request.node.get_closest_marker("real_blur_models"):
        return
    from gomoveshift.pipeline.photos import face_filter

    monkeypatch.setattr(face_filter, "_detectors", None)
    monkeypatch.setattr(face_filter, "_drive_cfg", None, raising=False)
    for name in (
        "resolve_plate_model",
        "resolve_face_model",
        "resolve_vehicle_model",
        "resolve_person_model",
    ):
        monkeypatch.setattr(face_filter, name, lambda _drive: None, raising=False)


@pytest.fixture(autouse=True)
def restore_root_logging():
    """Snapshot and restore root logger handlers + level around each test.

    ``configure_logging`` calls ``logging.basicConfig(force=True)`` — any
    test that reaches a CLI handler without patching it replaces the root
    handler set for the remainder of the xdist worker, silently appending
    every later test's records to that test's log file (e.g.
    ``cmd_pipeline`` with ``output_dir=Path("/tmp")`` left a
    JSON ``RotatingFileHandler`` on ``/tmp/pipeline.log``). Restoring the
    snapshot at teardown makes the cross-test leak unrepresentable.
    """
    import logging

    if not HAS_GOMOVESHIFT:
        yield
        return
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    yield
    for handler in root.handlers[:]:
        if handler not in saved_handlers:
            root.removeHandler(handler)
            handler.close()
    for handler in saved_handlers:
        if handler not in root.handlers:
            root.addHandler(handler)
    root.setLevel(saved_level)


@pytest.fixture(autouse=True)
def auto_skip_dod_validator(monkeypatch):
    """Stub the terminal DoD validator gate for unit-level orchestrator tests.

    The gate refuses to let run() report success unless real pipeline output
    is on disk.
    Unit tests that mock phases produce no such output and would trip
    check 01 (lf_count < 1) with every run. Tests that specifically
    exercise the gate (test_orchestrator_validation_gate.py) override
    this stub by patching ``run_validation`` directly in the test body.
    """
    if not HAS_GOMOVESHIFT:
        return

    from gomoveshift.pipeline import drive_orchestrator

    _stub_dod_gate(monkeypatch, drive_orchestrator)


def _stub_dod_gate(monkeypatch, module):
    """Stub ``run_validation`` on ``module`` and make ``ValidationMode`` safe.

    Under a non-editable install (CI installs the wheel, never ``-e .``) the
    repo-root ``tools`` package is absent, so ``drive_orchestrator`` binds
    ``ValidationMode`` to a sentinel that raises on ANY attribute access.
    ``mode=ValidationMode.AUTO`` at the gate's call site then blows up before
    the stubbed ``run_validation`` is reached. Only when the sentinel is in
    place, swap in a plain stand-in carrying the real ``Mode`` member names;
    a source checkout keeps the real enum.
    """
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    fake_report = MagicMock()
    fake_report.passed_count = 13
    fake_report.failed_count = 0
    fake_report.checks = [MagicMock(passed=True) for _ in range(13)]
    fake_report.format_table.return_value = "== DoD Validation: (auto-stubbed in tests) =="

    monkeypatch.setattr(module, "run_validation", lambda **kwargs: fake_report)
    try:
        module.ValidationMode.AUTO
    except RuntimeError:
        monkeypatch.setattr(
            module,
            "ValidationMode",
            SimpleNamespace(
                AUTO="auto", POST_UPLOAD="post_upload", SKIP_UPLOAD_AUDIT="skip_upload_audit"
            ),
        )


def fake_generate_lf_thumbnail(video_path=None, title=None, subtitle=None, output_dir=None, **kw):
    """Write a real JPG where the production generator would, and return it."""
    from pathlib import Path

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    variant = out / "frame1_bold.jpg"
    variant.write_bytes(b"stub-thumbnail-bytes" * 64)
    return variant


@pytest.fixture(autouse=True)
def stub_lf_thumbnail_generation(request, monkeypatch):
    """Let Phase-D LF thumbnail generation succeed against fake test videos.

    Phase D hard-fails (upstream of the YouTube upload) when an LF
    thumbnail can't be generated. Every Phase-D end-to-end test drives the
    REAL generator against a fake ``.mp4`` that legitimately yields nothing,
    so without this stub the suite would assert on a tripped gate instead of
    the behaviour it covers. Opt-out: ``real_lf_thumbnail_generation``.
    """
    if not HAS_GOMOVESHIFT:
        return
    if request.node.get_closest_marker("real_lf_thumbnail_generation"):
        return
    from gomoveshift.pipeline.drive_phases import thumbnail_generator

    monkeypatch.setattr(thumbnail_generator, "generate_lf_thumbnail", fake_generate_lf_thumbnail)


@pytest.fixture
def shipped_brands(tmp_path_factory, monkeypatch):
    """A stand-in for the console's shipped brand templates directory.

    The source tree ships no brand templates, so tests that need one build it
    here: ``gomoveshift`` (a settings template with every key unset) and
    ``examplebrand`` (a brand that hides the drives and shorts tabs). The
    directory the console reads templates from is pointed at it for the test.
    """
    from gomoveshift.console import settings as console_settings

    root = tmp_path_factory.mktemp("shipped-brands")
    (root / "gomoveshift").mkdir()
    (root / "gomoveshift" / "settings.yaml").write_text(
        "GMS_STAGING_DIR: null\nGMS_OUTPUT_DIR: null\n"
    )
    (root / "examplebrand").mkdir()
    (root / "examplebrand" / "brand.yaml").write_text("disabled_tabs:\n  - drives\n  - shorts\n")
    monkeypatch.setattr(console_settings, "_SHIPPED_BRANDS_DIR", root)
    return root


@pytest.fixture
def disable_blur(monkeypatch):
    """Force ``config.drive.blur_enabled`` OFF for orchestrator-integration tests.

    Blur is ON by default in production, so a full orchestrator run on a dummy
    video now invokes the real face/plate detector (``deface``) against a fake
    file and hard-fails at the blur stage. The tests that opt into this fixture
    exercise stages unrelated to blur; disabling it is cheaper and clearer than
    fixturing a real mask + ``.blurred`` marker just to satisfy the blur gate.

    Scoped per file via ``pytestmark = pytest.mark.usefixtures("disable_blur")``
    — deliberately NOT autouse, so the blur suite keeps the real default-on.
    Patches ``Config.load`` (the single seam every config funnels through) so it
    catches both the no-arg ``PipelineOrchestrator()`` and explicit-config paths.
    """
    if not HAS_GOMOVESHIFT:
        return
    from gomoveshift.config import Config

    orig_load = Config.load.__func__

    def _load(cls, config_path=None):
        cfg = orig_load(cls, config_path)
        cfg.drive.blur_enabled = False
        return cfg

    monkeypatch.setattr(Config, "load", classmethod(_load))


def pytest_ignore_collect(collection_path, config):
    """Skip all test files if gomoveshift package is not installed."""
    if (
        not HAS_GOMOVESHIFT
        and collection_path.suffix == ".py"
        and collection_path.name.startswith("test_")
    ):
        return True


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_seo_validation: opt out of the conftest no-op stub for "
        "``SEOConfig.validate_paths_exist`` and exercise the real validator.",
    )
    config.addinivalue_line(
        "markers",
        "real_content_ledger: opt out of the conftest hermetic-ledger redirect "
        "and resolve the real default ``ContentLedger`` path.",
    )
    config.addinivalue_line(
        "markers",
        "real_forward_geocode: opt out of the conftest ``_nominatim_search`` "
        "stub and exercise the real forward geocoder (live network).",
    )
    config.addinivalue_line(
        "markers",
        "real_lf_thumbnail_generation: opt out of the conftest "
        "``generate_lf_thumbnail`` stub and exercise the real generator "
        "(which produces nothing from a fake video).",
    )
    config.addinivalue_line(
        "markers",
        "real_overpass: opt out of the conftest ``_query_overpass`` stub, "
        "leaving the real function in place (patch ``urlopen`` yourself or "
        "expect live network).",
    )
    config.addinivalue_line(
        "markers",
        "allow_network: opt out of the conftest ``block_non_loopback_network`` "
        "socket guard for a genuinely intentional live-network test.",
    )
    config.addinivalue_line(
        "markers",
        "real_nvenc: opt out of the conftest NVENC-probe stub and probe the real "
        "GPU encoder. Skipped unless ``GMS_GPU_TESTS=1`` (docs/gpu-checks.md).",
    )
    config.addinivalue_line(
        "markers",
        "tmp_disk_budget_mib(n): raise the conftest ``tmp_path_disk_budget`` gate "
        "for a test that legitimately writes many real files (default "
        "``GMS_TEST_TMP_BUDGET_MIB``, 64).",
    )


@pytest.fixture(autouse=True)
def hermetic_contact_sheet_root(tmp_path, monkeypatch):
    """Redirect the ``photos.contact_sheet_root`` DEFAULT to a per-test temp dir.

    The sheet lives off ``--out`` (under the data home unless configured), so
    a test reaching ``--contact-sheet`` without setting the root would write a
    live file into the real review folder, mixed in among the sheets being
    reviewed.

    Patching the FIELD DEFAULT rather than each call site is what makes it
    airtight — a test that never touches config still gets the temp root, so
    there is nothing for the next test's author to remember. A test that sets
    ``contact_sheet_root`` explicitly is unaffected.
    """
    from gomoveshift.config import PhotosConfig

    field = PhotosConfig.model_fields["contact_sheet_root"]
    monkeypatch.setattr(field, "default", str(tmp_path / "contact-sheets"))
    PhotosConfig.model_rebuild(force=True)
    yield
    # The FieldInfo is class state, so the rebuild has to be redone once
    # monkeypatch puts the shipped default back.
    PhotosConfig.model_rebuild(force=True)


@pytest.fixture(autouse=True)
def isolate_pipeline_secrets(tmp_path, monkeypatch):
    """Give every test its own private, guaranteed-absent pipeline-secret
    paths, on top of the module-level override above.

    The module-level override (top of this file) protects anything that reads
    these variables while pytest collects and imports modules, before any
    fixture runs; this fixture re-points the same three env vars at a fresh
    ``tmp_path`` per test. That per-test value is what the console
    (``gomoveshift.console.settings.resolved_secret_paths`` /
    ``credential_status``) and the uploader resolve at call time. A test that
    needs a specific value still monkeypatches over it locally. The paths sit
    under a ``gms-test-secrets-isolated`` dir, the same marker the module-level
    override uses, so a consumer (test_console_compose's render env) can tell
    an isolation value from an override a test set on purpose.
    """
    secrets_dir = tmp_path / "gms-test-secrets-isolated"
    for name, filename in (
        ("GMS_DRIVE_CREDENTIALS_ENC", "drive_credentials.json.enc"),
        ("GMS_DRIVE_TOKEN_ENC", "drive_token.json.enc"),
        ("GMS_AGE_KEY_FILE", "age.key"),
    ):
        monkeypatch.setenv(name, str(secrets_dir / filename))


@pytest.fixture(autouse=True)
def isolate_shared_gpu_lock(monkeypatch):
    """Disable the cross-service GPU lock for every test by default.

    ``drive_orchestrator.run`` takes a lock at a machine-global path (by
    default under the cache home) shared with any resident local-LLM service,
    so an encode and a resident LLM never hold the same GPU. Under
    ``pytest -n`` two orchestrator tests would serialise on it and the loser
    would sit through the whole wait before failing, and a developer running
    the suite while local inference is loaded would fail tests that have
    nothing to do with the GPU.

    ``tests/test_gpu_lock.py`` passes an explicit ``path=`` and so is
    unaffected; a test that wants the wiring exercised sets
    ``GMS_GPU_LOCK_PATH`` to its own ``tmp_path``.
    """
    monkeypatch.setenv("GMS_GPU_LOCK_PATH", "")


# ---------------------------------------------------------------------------
# Cached synthetic ffmpeg test fixtures (suite wall-clock)
# ---------------------------------------------------------------------------
#
# Dozens of test files synthesize their input media at test time with
# ``subprocess.run(["ffmpeg", "-f", "lavfi", "-i", "testsrc=...", ...])``.
# Each encode burns CPU-seconds, and under pytest-xdist every worker
# re-encodes its own copy of identical clips. lavfi synthesis is fully
# deterministic (same command -> same bytes), so each unique command needs
# to run exactly once per machine session.
#
# Mechanism: a session-scoped wrapper around ``subprocess.run`` that, ONLY
# for pure-synthesis commands issued directly from test code (caller frame
# inside ``tests/``), serves the output from a cache shared across xdist
# workers, copying a fresh file to the requested output path (so tests that
# mutate or delete their input stay safe) and replaying the recorded
# stdout/stderr/returncode. Everything else — any ffmpeg spawned by
# gomoveshift production code (the generation paths under test), commands
# with file inputs, piped outputs, cwd/env/stdin — passes through untouched.

import hashlib as _hashlib
import json as _json
import os as _os
import shutil as _shutil
import subprocess as _subprocess
import sys as _sys

_TESTS_DIR = str(Path(__file__).resolve().parent) + _os.sep
_REAL_RUN = _subprocess.run

# The exact argv of ``color_filters.detect_nvenc``'s probe. Hosted CI has no
# GPU, so a suite that ran it for real would be red there and flaky on a busy
# GPU host. ``hermetic_nvenc_probe`` turns the stub off for ``real_nvenc`` tests.
_NVENC_PROBE_TAIL = ["-c:v", "hevc_nvenc", "-f", "null", "-"]
_STUB_NVENC_PROBE = [True]


def _is_nvenc_probe(args):
    if not isinstance(args, (list, tuple)) or len(args) < 6:
        return False
    argv = [str(a) for a in args]
    return (
        _os.path.basename(argv[0]) == "ffmpeg"
        and argv[-5:] == _NVENC_PROBE_TAIL
        and argv[1:4] == ["-f", "lavfi", "-i"]
        and argv[4].startswith("testsrc=")
    )


def _synth_output(args):
    """Output Path if ``args`` is a pure lavfi-synthesis ffmpeg command, else None."""
    if not isinstance(args, (list, tuple)) or len(args) < 5:
        return None
    argv = [str(a) for a in args]
    if _os.path.basename(argv[0]) != "ffmpeg":
        return None
    in_idx = [i for i, a in enumerate(argv) if a == "-i"]
    if not in_idx:
        return None
    for i in in_idx:  # every input must be a lavfi graph, never a file
        if i < 2 or argv[i - 1] != "lavfi" or argv[i - 2] != "-f":
            return None
    if argv[-1].startswith("-"):  # piped/null output: not a file we can cache
        return None
    return Path(argv[-1])


@pytest.fixture(autouse=True, scope="session")
def cache_synth_ffmpeg_fixtures(tmp_path_factory):
    base = tmp_path_factory.getbasetemp()
    if base.name.startswith("popen-gw"):  # xdist worker: hop up to the shared root
        base = base.parent
    cache = base / "gms-synth-clip-cache"
    cache.mkdir(exist_ok=True)

    def _reshape(args, rc, raw_out, raw_err, kwargs):
        text = kwargs.get("text") or kwargs.get("universal_newlines")
        w_out = kwargs.get("capture_output") or kwargs.get("stdout") is not None
        w_err = kwargs.get("capture_output") or kwargs.get("stderr") is not None
        stdout = (raw_out.decode(errors="replace") if text else raw_out) if w_out else None
        stderr = (raw_err.decode(errors="replace") if text else raw_err) if w_err else None
        if kwargs.get("check") and rc != 0:
            raise _subprocess.CalledProcessError(rc, args, output=stdout, stderr=stderr)
        return _subprocess.CompletedProcess(args, rc, stdout, stderr)

    def _run(*popenargs, **kwargs):
        args = popenargs[0] if popenargs else kwargs.get("args")
        if _STUB_NVENC_PROBE[0] and _is_nvenc_probe(args):
            return _reshape(args, 0, b"", b"", kwargs)
        out = None
        if len(popenargs) <= 1 and not any(
            k in kwargs for k in ("cwd", "env", "input", "shell", "stdin")
        ):
            caller = _sys._getframe(1).f_code.co_filename
            if caller.startswith(_TESTS_DIR):
                out = _synth_output(args)
        if out is None:
            return _REAL_RUN(*popenargs, **kwargs)

        key_src = _json.dumps([str(a) for a in args[:-1]] + [out.suffix])
        key = _hashlib.sha256(key_src.encode()).hexdigest()[:24]
        clip, meta = cache / (key + out.suffix), cache / (key + ".json")
        if clip.exists() and meta.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            _shutil.copy2(clip, out)
            m = _json.loads(meta.read_text())
            return _reshape(args, 0, bytes.fromhex(m["out"]), bytes.fromhex(m["err"]), kwargs)

        clean = {
            k: v
            for k, v in kwargs.items()
            if k
            not in ("capture_output", "stdout", "stderr", "check", "text", "universal_newlines")
        }
        proc = _REAL_RUN(args, capture_output=True, **clean)
        if proc.returncode == 0 and out.exists():
            # temp-write + atomic rename: concurrent xdist misses converge on
            # byte-identical content, so last-writer-wins is safe and lockless.
            tag = f".{_os.getpid()}.tmp"
            _shutil.copy2(out, clip.with_name(clip.name + tag))
            _os.replace(clip.with_name(clip.name + tag), clip)
            meta.with_name(meta.name + tag).write_text(
                _json.dumps({"cmd": key_src, "out": proc.stdout.hex(), "err": proc.stderr.hex()})
            )
            _os.replace(meta.with_name(meta.name + tag), meta)
        return _reshape(args, proc.returncode, proc.stdout, proc.stderr, kwargs)

    mp = pytest.MonkeyPatch()
    mp.setattr(_subprocess, "run", _run)
    yield
    mp.undo()


@pytest.fixture(autouse=True)
def hermetic_nvenc_probe(request):
    """Answer the NVENC probe as available, without touching a GPU.

    A test that patches ``subprocess.run`` or ``detect_nvenc`` itself still
    wins over this stub. Opt-out: ``@pytest.mark.real_nvenc``, which probes
    the real encoder and is skipped unless ``GMS_GPU_TESTS=1``.
    """
    if not request.node.get_closest_marker("real_nvenc"):
        yield
        return
    if _os.environ.get("GMS_GPU_TESTS") != "1":
        pytest.skip("needs an NVENC GPU: set GMS_GPU_TESTS=1 on a GPU host")
    _STUB_NVENC_PROBE[0] = False
    try:
        yield
    finally:
        _STUB_NVENC_PROBE[0] = True


# A test's tmp_path is disk, and pytest keeps every basetemp it makes: the
# default /tmp/pytest-of-<user>/pytest-N is rotated only by a process that
# reuses that parent, and every session passing its own --basetemp never does.
# A preflight test that writes a real, large zero-filled "video" only so a size
# check passes leaks gigabytes per xdist run, and repeated runs fill the disk.
# A size check needs st_size, which a
# sparse file (`path.touch(); os.truncate(path, n)`) provides for free. This
# fails the test that wrote the bytes, at the moment it wrote them, so the
# leak cannot recur; budget in MiB, `GMS_TEST_TMP_BUDGET_MIB` overrides.
_TMP_PATH_BUDGET_BYTES = int(os.environ.get("GMS_TEST_TMP_BUDGET_MIB", "64")) * 1024 * 1024


def _bytes_on_disk(root: Path) -> int:
    total = 0
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            try:
                st = os.lstat(os.path.join(dirpath, name))
            except OSError:
                continue
            total += st.st_blocks * 512  # allocated blocks: a sparse file counts ~0
    return total


@pytest.fixture(autouse=True)
def tmp_path_disk_budget(request):
    yield
    if "tmp_path" not in request.fixturenames:
        return
    root = request.getfixturevalue("tmp_path")
    used = _bytes_on_disk(root)
    budget = _TMP_PATH_BUDGET_BYTES
    marker = request.node.get_closest_marker("tmp_disk_budget_mib")
    if marker is not None:
        budget = int(marker.args[0]) * 1024 * 1024
    if used > budget:
        pytest.fail(
            f"{request.node.nodeid} left {used / 2**20:.0f} MiB on disk under tmp_path "
            f"(budget {budget / 2**20:.0f} MiB). Fake a large file with "
            "`path.touch(); os.truncate(path, size)` -- same st_size, no disk -- "
            "instead of writing real bytes; pytest never reclaims basetemps."
        )


@pytest.fixture(autouse=True)
def isolate_cadence_store_from_cwd(tmp_path, monkeypatch):
    """No test may read Waypost's decision store from the working directory.

    A bare ``Config()`` carries the relative ``seo.storage_path`` ``./data/seo``, so
    ``seo_slots.waypost_cadence`` would open ``<cwd>/data/seo/cadence_decisions.sqlite3``:
    a host that has run the decider there would hand every such test its own days. A
    RELATIVE store path is redirected under ``tmp_path``; an absolute one (the only kind
    ``SEOConfig.validate_paths_exist`` accepts in production) is read as given.
    """
    if not HAS_GOMOVESHIFT:
        return
    from gomoveshift.pipeline import seo_slots

    real = seo_slots.load_cadence

    def _load(brand_yaml_path, decisions_db=None):
        if decisions_db is not None and not Path(decisions_db).is_absolute():
            decisions_db = tmp_path / "bare-config-seo" / Path(decisions_db).name
        return real(brand_yaml_path, decisions_db=decisions_db)

    monkeypatch.setattr(seo_slots, "load_cadence", _load)


_SEED_MIRRORS: dict = {}  # the import-time (seed) mirrors, captured before the first pin


@pytest.fixture(autouse=True)
def legacy_test_cadence(request):
    """Pin the long-form module mirrors to the cadence most tests were written for.

    The hand-set Wed / 11:00 default is gone: ``LONG_FORM_DAYS`` /
    ``LONG_FORM_TIME`` now mirror Waypost's packaged seed (Tue, Fri, Sun / 16:00).
    The many tests that build a Wednesday fixture and read the module mirrors
    (state invariants, quality gates, scheduler windows) name that cadence HERE,
    once, instead of each carrying its own copy. The days are the seed's Tue, Fri, Sun
    PLUS Wednesday, so a Wednesday fixture and a Waypost-derived default day (Tuesday)
    are both valid to the invariants; the time stays 11:00. A module that tests the seed
    itself sets ``SEED_CADENCE = True`` to opt out; a test that wants a
    different cadence still monkeypatches the mirrors or uses ``waypost_lf_days``
    (both override this pin).

    Teardown puts the SEED values back unconditionally (not "whatever was there"): the
    suite's own save/restore fixtures capture the pinned values and would otherwise
    hand them to the next test in the worker.
    """
    from datetime import time as _time

    from gomoveshift.pipeline import schedule_constants as _sc
    from gomoveshift.pipeline import scheduler as _sched

    mods = (_sc, _sched)
    if not _SEED_MIRRORS:
        _SEED_MIRRORS.update(
            {
                (m.__name__, n): getattr(m, n)
                for m in mods
                for n in ("LONG_FORM_DAYS", "LONG_FORM_TIME")
            }
        )
    seed_run = getattr(request.module, "SEED_CADENCE", False)
    days, lf_time = frozenset({1, 2, 4, 6}), _time(11, 0)
    for m in mods:
        if seed_run:  # a leaked pin from an earlier test must not reach a seed test
            m.LONG_FORM_DAYS = _SEED_MIRRORS[(m.__name__, "LONG_FORM_DAYS")]
            m.LONG_FORM_TIME = _SEED_MIRRORS[(m.__name__, "LONG_FORM_TIME")]
        else:
            m.LONG_FORM_DAYS, m.LONG_FORM_TIME = days, lf_time
    yield
    for m in mods:
        m.LONG_FORM_DAYS = _SEED_MIRRORS[(m.__name__, "LONG_FORM_DAYS")]
        m.LONG_FORM_TIME = _SEED_MIRRORS[(m.__name__, "LONG_FORM_TIME")]


@pytest.fixture
def waypost_lf_days(tmp_path):
    """Name the long-form weekdays a test assumes, the way Waypost records them.

    ``waypost_lf_days({"ireland": (2,), "usa": (2,)}, config=None)`` appends one ``lf_days``
    decision row per region to the decision store under ``config.seo.storage_path`` (the
    loaded ``get_config()`` when ``config`` is None), so ``seo_slots.cadence_for`` sees them.
    Opt in with ``pytestmark = pytest.mark.usefixtures("waypost_lf_days")`` or by argument;
    it is deliberately not autouse -- a test names the cadence it assumes.
    """
    from datetime import datetime
    from pathlib import Path

    from seo_manager.analytics import cadence_decisions

    from gomoveshift.config import get_config
    from gomoveshift.pipeline.volume_source import DB_NAME

    def _set(days_by_region, config=None):
        cfg = config if config is not None else get_config()
        if not str(cfg.seo.storage_path).startswith(str(tmp_path)):
            cfg.seo.storage_path = str(tmp_path)  # a bare Config() points at ./data/seo
        db = Path(cfg.seo.storage_path) / DB_NAME
        db.parent.mkdir(parents=True, exist_ok=True)
        for region, dow in days_by_region.items():
            cadence_decisions.append(
                db,
                knob="lf_days",
                region=region,
                value=len(dow),
                prev=len(dow),
                status="ok",
                source="policy",
                reason_code="TAIL_UP",
                basis="test fixture",
                applied_at=datetime(2026, 1, 1, 12, 0),
                evidence={"dow": sorted(dow)},
            )

    return _set
