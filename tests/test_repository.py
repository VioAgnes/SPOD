"""CPU checks for packaged inputs, configuration validation, and public commands."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vidhalluc.config import load_benchmark, load_config


class RepositoryTests(unittest.TestCase):
    def test_benchmark_ids_and_counts(self):
        benchmark = load_benchmark()
        rows = benchmark["rows"]
        self.assertEqual(len(rows), benchmark["total"])
        self.assertEqual(len({row["id"] for row in rows}), 13871)
        self.assertEqual(Counter(row["subset"] for row in rows), benchmark["counts"])
        self.assertTrue(all("gold" not in row for row in rows))
        self.assertTrue(all(row["prompt"] for row in rows))

    def test_rank8_configuration(self):
        config = load_config(ROOT / "configs/spod.json")
        self.assertEqual(config["name"], "spod")
        self.assertEqual(config["spatial_rank"], 8)

    def test_invalid_configurations(self):
        config = load_config(ROOT / "configs/spod.json")
        invalid = [
            {},
            dict(config, spatial_rank=16),
            dict(config, source_layer=18),
            dict(config, gT=1, gP=1, gR=1),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for value in invalid:
                with self.subTest(config=value):
                    path.write_text(json.dumps(value), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_config(path)

    def test_command_help_without_gpu_dependencies(self):
        env = dict(
            os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1"
        )
        for command in ([], ["infer"], ["evaluate"], ["audit"]):
            with self.subTest(command=command):
                result = subprocess.run(
                    [sys.executable, "-m", "vidhalluc", *command, "--help"],
                    env=env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()
