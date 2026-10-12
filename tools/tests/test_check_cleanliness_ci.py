"""Offline contract tests using actual temporary Git histories."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


HELPER = Path(__file__).resolve().parents[1] / "check_cleanliness_ci.py"


class CleanlinessCITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git("init", "--initial-branch=test")
        self.git("config", "user.name", "Test")
        self.git("config", "user.email", "test@example.invalid")
        self.write("src/app.py", "public source\n")
        self.write("tools/check_public_tree.py", "trusted guard\n")
        self.write("tools/public-tree-policy.json", "{}\n")
        self.write(".github/workflows/cleanup.yml", "trusted workflow\n")
        self.write(".gitignore", "__pycache__/\n")
        self.commit()
        self.base = self.git("rev-parse", "HEAD").strip()

    def git(self, *args):
        return subprocess.run(
            ["git", "-C", str(self.root), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    def write(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def commit(self):
        self.git("add", "--all")
        self.git("commit", "--quiet", "--allow-empty", "-m", "test: candidate")

    def inspect(self, base=None, head=None):
        return subprocess.run(
            [
                sys.executable,
                "-I",
                str(HELPER),
                "inspect",
                "--root",
                str(self.root),
                "--base-sha",
                base or self.base,
                "--head-sha",
                head or self.git("rev-parse", "HEAD").strip(),
            ],
            capture_output=True,
            text=True,
            cwd=self.root,
            env=dict(os.environ, PYTHONPATH=str(self.root)),
        )

    def test_ordinary_product_changes_need_no_approval(self):
        self.write("src/app.py", "updated public source\n")
        self.commit()
        result = self.inspect()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["maintenance_changed"], "false")

    def test_additions_modifications_and_deletions_of_enforcement_are_protected(self):
        names = [
            ".github/workflows/new.yml",
            ".github/actions/local/action.yml",
            "tools/check_public_tree.py",
            "tools/public-tree-policy.json",
            "tools/tests/test_check_public_tree.py",
            "tools/check_cleanliness_ci.py",
            "tools/tests/test_check_cleanliness_ci.py",
            "tools/dependency.py",
            ".gitignore",
            "nested/.gitignore",
        ]
        for name in names:
            with self.subTest(path=name):
                self.git("reset", "--hard", self.base)
                self.write(name, "changed enforcement\n")
                self.commit()
                result = self.inspect()
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout)["maintenance_changed"], "true")
                previous = self.git("rev-parse", "HEAD").strip()
                self.git("rm", "--", name)
                self.commit()
                result = self.inspect(base=previous)
                self.assertEqual(json.loads(result.stdout)["maintenance_changed"], "true")

    def test_renaming_enforcement_to_unprotected_path_is_detected(self):
        self.git("mv", "tools/check_public_tree.py", "src/old_guard.py")
        self.commit()
        self.assertEqual(json.loads(self.inspect().stdout)["maintenance_changed"], "true")

    def test_product_ci_helpers_require_maintenance(self):
        for name in ("ci/secret-scan.sh", "ci/install-cpu.sh", "ci/cpu_pins.py"):
            with self.subTest(path=name):
                self.git("reset", "--hard", self.base)
                self.write(name, "changed product quality helper\n")
                self.commit()
                self.assertEqual(json.loads(self.inspect().stdout)["maintenance_changed"], "true")

    def test_invalid_missing_or_mismatched_sha_fails_closed(self):
        for head in ("HEAD", "--help", "0" * 40):
            with self.subTest(head=head):
                self.assertNotEqual(self.inspect(head=head).returncode, 0)
        old = self.base
        self.write("src/app.py", "new head\n")
        self.commit()
        self.assertNotEqual(self.inspect(head=old).returncode, 0)

    def test_candidate_code_is_never_executed(self):
        marker = self.root / "EXECUTED"
        poison = f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n"
        self.write("tools/check_cleanliness_ci.py", poison)
        self.write("sitecustomize.py", poison)
        self.write("json.py", poison)
        self.commit()
        result = self.inspect()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(marker.exists())

    def test_aggregate_accepts_only_complete_success_states(self):
        for changed, inspection, approval, expected in (
            ("false", "success", "skipped", 0),
            ("true", "success", "success", 0),
            ("", "success", "skipped", 1),
            ("false", "skipped", "skipped", 1),
            ("false", "failure", "skipped", 1),
            ("false", "cancelled", "skipped", 1),
            ("true", "success", "skipped", 1),
            ("true", "success", "cancelled", 1),
            ("true", "success", "failure", 1),
            ("false", "success", "success", 1),
        ):
            with self.subTest(changed=changed, inspection=inspection, approval=approval):
                env = dict(
                    os.environ,
                    MAINTENANCE_CHANGED=changed,
                    INSPECTION_RESULT=inspection,
                    APPROVAL_RESULT=approval,
                    EXPECTED_SHA=self.base,
                    INSPECTED_SHA=self.base,
                )
                result = subprocess.run(
                    [sys.executable, "-I", str(HELPER), "aggregate"],
                    env=env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, expected, result.stderr)

    def test_aggregate_rejects_missing_or_wrong_candidate_identity(self):
        for inspected in ("", "0" * 40):
            env = dict(
                os.environ,
                MAINTENANCE_CHANGED="false",
                INSPECTION_RESULT="success",
                APPROVAL_RESULT="skipped",
                EXPECTED_SHA=self.base,
                INSPECTED_SHA=inspected,
            )
            result = subprocess.run(
                [sys.executable, "-I", str(HELPER), "aggregate"],
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
