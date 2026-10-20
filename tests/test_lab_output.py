"""A reproduction must preserve the published edition and previous runs."""
import csv
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "labs"))
from run_all import RunOutput, ridge_lab


def reference_hashes():
    paths = [ROOT / "labs" / "RESULTS.md"]
    paths += list((ROOT / "assets").glob("*"))
    paths += list((ROOT / "labs" / "results").glob("*"))
    return {path: hashlib.sha256(path.read_bytes()).digest() for path in paths if path.is_file()}


class IndependentRunChecks(unittest.TestCase):
    def test_ridge_writes_to_a_fresh_run_and_preserves_reference_results(self):
        before = reference_hashes()
        with tempfile.TemporaryDirectory() as temporary:
            output = RunOutput(Path(temporary) / "new-run")
            result = ridge_lab(output)
            self.assertGreater(result["test_half_mse"], 0)
            self.assertTrue((output.root / "assets" / "ridge.png").read_bytes().startswith(b"\x89PNG"))
            with (output.root / "labs" / "results" / "ridge.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 66)
            self.assertEqual(float(rows[0]["lambda"]), 0)
        self.assertEqual(reference_hashes(), before)

    def test_existing_file_directory_and_symlink_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            previous = root / "previous"
            previous.mkdir()
            report = previous / "RESULTS.md"
            report.write_text("previous completed run", encoding="utf-8")
            file = root / "important.txt"
            file.write_text("keep me", encoding="utf-8")
            paths = [previous, file]
            link = root / "linked-run"
            try:
                link.symlink_to(previous, target_is_directory=True)
            except OSError:
                pass  # Some Windows environments do not allow creating symlinks.
            else:
                paths.append(link)
            for path in paths:
                with self.subTest(path=path.name), self.assertRaises(FileExistsError):
                    RunOutput(path)
            self.assertEqual(report.read_text(), "previous completed run")
            self.assertEqual(file.read_text(), "keep me")
            self.assertEqual(list(previous.iterdir()), [report])

    def test_cli_reports_an_existing_destination_without_running_experiments(self):
        with tempfile.TemporaryDirectory() as temporary:
            completed = subprocess.run(
                [sys.executable, str(ROOT / "labs" / "run_all.py"), "--output-dir", temporary],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("cannot create a fresh output directory", completed.stderr)
            self.assertNotIn("Traceback", completed.stderr)
            self.assertEqual(list(Path(temporary).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
