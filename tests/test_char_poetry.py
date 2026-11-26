import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import char_poetry


class PilotTests(unittest.TestCase):
    def test_import_no_effect(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(__file__).resolve().parents[1] / "scripts" / "char_poetry.py"
            r = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    f'import runpy;runpy.run_path({str(p)!r},run_name="library")',
                ],
                cwd=td,
                capture_output=True,
                text=True,
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(r.stdout, "")
            self.assertEqual(list(Path(td).iterdir()), [])

    def test_epochs_validate_before_output(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "new"
            with self.assertRaisesRegex(ValueError, "positive"):
                char_poetry.run(Path("missing"), out, 0)
            self.assertFalse(out.exists())

    def test_reject_incomplete_source(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.txt"
            p.write_text("THE SONNETS\n I\n some text here\n*** END OF THIS PROJECT")
            with self.assertRaisesRegex(ValueError, "154"):
                char_poetry.sonnets(p)

    def test_existing_output_safe(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileExistsError):
                char_poetry.run(Path("missing"), Path(td), 1)


if __name__ == "__main__":
    unittest.main()
