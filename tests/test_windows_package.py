"""Windows bundle builder: staging, the privacy audit, the size gate, the launcher
and the installer script. Pure logic only: no download, no Windows, no Inno Setup.

``packaging/`` is a plain folder of scripts, not a package (and a package named
``packaging`` would shadow PyPA's), so the builder is loaded by path.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
WIN = ROOT / "packaging" / "windows"


def _load():
    spec = importlib.util.spec_from_file_location("build_release", WIN / "build_release.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


br = _load()

FAKE_PTH = "python312.zip\n.\n\n# Uncomment to run site.main() automatically\n#import site\n"


def _fake_embed(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("python.exe", "MZ")
        zf.writestr("python312.zip", "PK")
        zf.writestr("python312._pth", FAKE_PTH)
    return path


class TestEmbeddable:
    def test_extracts_under_python_and_enables_site_packages(self, tmp_path):
        stage = tmp_path / "stage"
        br.stage_python(_fake_embed(tmp_path / "embed.zip"), stage)
        assert (stage / "python" / "python.exe").is_file()
        lines = (stage / "python" / "python312._pth").read_text().splitlines()
        assert "Lib\\site-packages" in lines
        assert "..\\app" in lines
        assert "import site" in lines
        assert "#import site" not in lines
        assert lines[0] == "python312.zip"

    def test_pth_edit_is_idempotent(self, tmp_path):
        pth = tmp_path / "python312._pth"
        pth.write_text(FAKE_PTH)
        br.enable_site_packages(pth)
        once = pth.read_text()
        br.enable_site_packages(pth)
        assert pth.read_text() == once

    def test_missing_pth_is_refused(self, tmp_path):
        zip_path = tmp_path / "embed.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("python.exe", "MZ")
        with pytest.raises(SystemExit, match="_pth"):
            br.stage_python(zip_path, tmp_path / "stage")


class TestAppStaging:
    def _repo(self, tmp_path):
        repo = tmp_path / "repo"
        (repo / "gomoveshift").mkdir(parents=True)
        (repo / "gomoveshift" / "__init__.py").write_text("x = 1\n")
        (repo / "gomoveshift" / "untracked_notes.txt").write_text("private\n")
        (repo / "config.yaml").write_text("a: 1\n")
        (repo / "LICENSE").write_text("Apache\n")
        (repo / "data").mkdir()
        (repo / "data" / "state.json").write_text("{}")
        git = ["git", "-C", str(repo)]
        subprocess.run([*git, "init", "-q"], check=True)
        subprocess.run(
            [*git, "add", "gomoveshift/__init__.py", "config.yaml", "LICENSE"], check=True
        )
        return repo

    def test_only_git_tracked_files_are_staged(self, tmp_path):
        repo = self._repo(tmp_path)
        stage = tmp_path / "stage"
        br.stage_app(repo, stage)
        assert (stage / "app" / "gomoveshift" / "__init__.py").is_file()
        assert (stage / "app" / "config.yaml").is_file()
        assert not (stage / "app" / "gomoveshift" / "untracked_notes.txt").exists()
        assert not (stage / "app" / "data").exists()

    def test_version_comes_from_pyproject(self):
        assert re.fullmatch(r"\d+\.\d+\.\d+", br.project_version(ROOT))


class TestAudit:
    @pytest.mark.parametrize(
        "rel",
        [
            "app/data/anything.txt",
            "app/.data/x.txt",
            "app/out/render.mp4",
            "app/state.db",
            "app/cache.sqlite",
            "app/cache.sqlite3",
            "app/library.db-wal",
            "app/youtube_upload_token.json",
            "app/credentials.json",
            "app/client_secret_123.json",
            "app/token.json",
            "app/.env",
            "app/id_rsa",
            "app/secrets/drive_token.json.enc",
            "app/signing.pem",
            "python/Lib/site-packages/pkg/blob.enc",
        ],
    )
    def test_forbidden_kinds_are_refused(self, tmp_path, rel):
        stage = tmp_path / "stage"
        target = stage / rel
        target.parent.mkdir(parents=True)
        target.write_text("x")
        with pytest.raises(SystemExit, match="refused"):
            br.audit(stage)

    def test_clean_stage_passes(self, tmp_path):
        stage = tmp_path / "stage"
        for rel in ("app/gomoveshift/__init__.py", "python/python.exe", "framegrade.cmd"):
            (stage / rel).parent.mkdir(parents=True, exist_ok=True)
            (stage / rel).write_text("x")
        br.audit(stage)

    def test_third_party_certificate_bundle_is_not_a_secret(self, tmp_path):
        stage = tmp_path / "stage"
        cert = stage / "python" / "Lib" / "site-packages" / "certifi" / "cacert.pem"
        cert.parent.mkdir(parents=True)
        cert.write_text("x")
        br.audit(stage)

    def test_linux_binary_in_site_packages_is_refused(self, tmp_path):
        stage = tmp_path / "stage"
        so = stage / "python" / "Lib" / "site-packages" / "numpy" / "core.so"
        so.parent.mkdir(parents=True)
        so.write_text("ELF")
        with pytest.raises(SystemExit, match="not a Windows"):
            br.audit(stage)


class TestSizeGate:
    def test_limit_is_the_github_release_per_file_cap(self):
        assert br.MAX_ASSET_BYTES == 2 * 1024**3

    def test_small_files_pass(self, tmp_path):
        small = tmp_path / "a.zip"
        small.write_bytes(b"x" * 10)
        br.size_gate([small])

    def test_oversize_file_exits_naming_file_and_size(self, tmp_path):
        big = tmp_path / "framegrade-setup.exe"
        big.write_bytes(b"x" * 100)
        with pytest.raises(SystemExit) as exc:
            br.size_gate([big], limit=99)
        message = str(exc.value)
        assert "framegrade-setup.exe" in message
        assert "100" in message

    def test_exactly_at_the_limit_passes(self, tmp_path):
        edge = tmp_path / "edge.zip"
        edge.write_bytes(b"x" * 50)
        br.size_gate([edge], limit=50)

    def test_sparse_file_over_two_gib_is_refused(self, tmp_path):
        sparse = tmp_path / "huge.zip"
        with sparse.open("wb") as fh:
            fh.truncate(2 * 1024**3 + 1)
        with pytest.raises(SystemExit, match="huge.zip"):
            br.size_gate([sparse])


class TestRequirements:
    def test_git_refs_stay_direct_references_and_lock_becomes_constraints(self, tmp_path):
        reqs, constraints = br.release_requirements(ROOT)
        assert any(r.startswith("onnxruntime-gpu[cuda,cudnn]==") for r in reqs)
        assert any(r.startswith("seo-manager @ git+") for r in reqs)
        assert not any(r.startswith("-e") for r in constraints)
        assert not any(line.lstrip().startswith("#") or not line.strip() for line in constraints)
        assert any(c.startswith("numpy==") for c in constraints)

    def test_optional_extras_are_not_bundled(self):
        reqs, _ = br.release_requirements(ROOT)
        assert not any(r.split("[")[0].split("=")[0].strip() in {"torch", "easyocr"} for r in reqs)


class TestLauncher:
    text = (WIN / "framegrade.cmd").read_text(encoding="ascii")

    def test_sets_the_data_home_from_the_chosen_folder(self):
        assert "GMS_DATA_HOME" in self.text
        assert "data-folder.txt" in self.text

    def test_waypost_data_home_lives_under_the_same_folder(self):
        assert re.search(r"SEO_MANAGER_DATA_HOME=%GMS_DATA_HOME%", self.text)

    def test_spool_dir_set_so_console_hands_gpu_work_to_the_worker(self):
        assert re.search(r"GMS_HOST_SPOOL_DIR=%GMS_DATA_HOME%", self.text)

    def test_starts_worker_and_console_with_the_bundled_python(self):
        assert "python\\python.exe" in self.text
        assert "-m gomoveshift.hostworker" in self.text
        assert re.search(r"-m gomoveshift console\b", self.text)

    def test_opens_the_console_url_on_the_default_port(self):
        assert "http://127.0.0.1:8787" in self.text

    def test_is_ascii(self):
        assert self.text.isascii()


class TestRouteStoryBrowser:
    """Route Story's 3-D film drives Playwright's Chromium: the bundle carries it."""

    def test_bundle_installs_the_playwright_extra(self):
        reqs, _ = br.release_requirements(ROOT)
        assert any(r.lower().startswith("playwright") for r in reqs)

    def test_playwright_pin_comes_from_the_lock(self):
        _, constraints = br.release_requirements(ROOT)
        pin = br.playwright_pin(constraints)
        assert re.fullmatch(r"\d+\.\d+\.\d+", pin)
        assert f"playwright=={pin}" in constraints

    def test_missing_pin_is_refused(self):
        with pytest.raises(SystemExit, match="playwright"):
            br.playwright_pin(["numpy==2.0.0"])

    def test_windows_host_runs_the_bundled_python(self, tmp_path):
        stage = tmp_path / "stage"
        cmd, env = br.browser_install_plan(stage, "1.63.0", windows_host=True)
        assert cmd == [
            str(stage / "python" / "python.exe"),
            "-m",
            "playwright",
            "install",
            "chromium",
        ]
        assert env["PLAYWRIGHT_BROWSERS_PATH"] == str(stage / "app" / "browsers")
        assert "PLAYWRIGHT_HOST_PLATFORM_OVERRIDE" not in env

    def test_other_hosts_fetch_the_windows_build_with_the_pinned_playwright(self, tmp_path):
        stage = tmp_path / "stage"
        cmd, env = br.browser_install_plan(stage, "1.63.0", windows_host=False)
        assert cmd[:2] == ["uv", "run"]
        assert "playwright==1.63.0" in cmd
        assert cmd[-4:] == ["-m", "playwright", "install", "chromium"]
        assert env["PLAYWRIGHT_HOST_PLATFORM_OVERRIDE"] == "win64"
        assert env["PLAYWRIGHT_BROWSERS_PATH"] == str(stage / "app" / "browsers")

    def test_stage_browser_runs_the_plan_then_checks_for_windows_executables(self, tmp_path):
        stage = tmp_path / "stage"
        calls = []

        def fake_run(cmd, env, **kw):
            calls.append((cmd, env))
            root = Path(env["PLAYWRIGHT_BROWSERS_PATH"])
            (root / "chromium-1243" / "chrome-win64").mkdir(parents=True)
            (root / "chromium-1243" / "chrome-win64" / "chrome.exe").write_bytes(b"MZ")
            shell = root / "chromium_headless_shell-1243" / "chrome-headless-shell-win64"
            shell.mkdir(parents=True)
            (shell / "chrome-headless-shell.exe").write_bytes(b"MZ")

        assert br.stage_browser(stage, "1.63.0", windows_host=False, run=fake_run) is True
        assert len(calls) == 1

    def test_stage_browser_refuses_a_download_with_no_windows_executable(self, tmp_path):
        def fake_run(cmd, env, **kw):
            (Path(env["PLAYWRIGHT_BROWSERS_PATH"]) / "chromium-1243").mkdir(parents=True)

        with pytest.raises(SystemExit, match="chrome.exe"):
            br.stage_browser(tmp_path / "stage", "1.63.0", windows_host=True, run=fake_run)

    def test_skip_browser_downloads_nothing_and_says_so(self, tmp_path, capsys):
        def boom(*a, **kw):
            raise AssertionError("must not download")

        assert (
            br.stage_browser(tmp_path, "1.63.0", windows_host=False, run=boom, skip=True) is False
        )
        assert "browser" in capsys.readouterr().out.lower()

    def test_audit_accepts_the_browser_tree_but_still_refuses_linux_binaries(self, tmp_path):
        stage = tmp_path / "stage"
        good = stage / "app" / "browsers" / "chromium-1243" / "chrome-win64"
        good.mkdir(parents=True)
        (good / "chrome.exe").write_bytes(b"MZ")
        (good / "Cookies.db").write_bytes(b"x")
        (good / "chrome_100_percent.pak").write_bytes(b"x")
        br.audit(stage)
        (good / "libx.so").write_bytes(b"\x7fELF")
        with pytest.raises(SystemExit, match="Linux binary"):
            br.audit(stage)

    def test_launcher_points_playwright_at_the_bundled_browsers(self):
        text = (WIN / "framegrade.cmd").read_text(encoding="ascii")
        assert 'set "PLAYWRIGHT_BROWSERS_PATH=%HERE%app\\browsers"' in text

    def test_launcher_sets_it_before_starting_anything(self):
        text = (WIN / "framegrade.cmd").read_text(encoding="ascii")
        assert text.index("PLAYWRIGHT_BROWSERS_PATH") < text.index("start ")


class TestInstallerScript:
    text = (WIN / "framegrade.iss").read_text(encoding="ascii")

    def test_names_the_app(self):
        assert '#define AppName "Framegrade"' in self.text
        assert "AppName={#AppName}" in self.text

    def test_is_per_user_with_no_admin(self):
        assert "PrivilegesRequired=lowest" in self.text
        assert "{autopf}" not in self.text
        assert "{localappdata}" in self.text

    def test_has_start_menu_shortcut_and_uninstall_entry(self):
        assert "[Icons]" in self.text
        assert "{autoprograms}" in self.text
        assert "UninstallDisplayName" in self.text

    def test_has_a_data_folder_wizard_page_written_where_the_launcher_reads_it(self):
        assert "CreateInputDirPage" in self.text
        assert "data-folder.txt" in self.text

    def test_finish_page_carries_gpu_and_ffmpeg_notes(self):
        assert "NVIDIA" in self.text
        assert "winget install -e --id Gyan.FFmpeg" in self.text

    def test_version_has_no_default(self):
        assert "#ifndef AppVersion" in self.text
        assert "#error" in self.text

    def test_is_ascii(self):
        assert self.text.isascii()
