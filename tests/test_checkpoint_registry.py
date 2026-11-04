import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from code.framework import checkpoint_registry


class CheckpointRegistryTests(unittest.TestCase):
    def tearDown(self):
        checkpoint_registry._registry.cache_clear()

    def test_loads_artifact_pair_from_external_config(self):
        registry = {
            "artifacts": {
                "example": {
                    "state": "tinker://state",
                    "sampler": "tinker://sampler",
                }
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoints.json"
            path.write_text(json.dumps(registry))
            with patch.dict(
                os.environ,
                {"PEARL_CHECKPOINT_CONFIG": str(path)},
                clear=False,
            ):
                checkpoint_registry._registry.cache_clear()
                pair = checkpoint_registry.checkpoint_pair(
                    "example",
                    placeholder_state="placeholder-state",
                    placeholder_sampler="placeholder-sampler",
                )
        self.assertEqual(pair, ("tinker://state", "tinker://sampler"))

    def test_missing_runtime_artifact_fails_closed(self):
        with patch.dict(os.environ, {}, clear=True):
            checkpoint_registry._registry.cache_clear()
            with patch("sys.argv", ["experiment.py"]):
                with self.assertRaises(RuntimeError):
                    checkpoint_registry.checkpoint_pair(
                        "missing",
                        placeholder_state="placeholder-state",
                        placeholder_sampler="placeholder-sampler",
                    )


if __name__ == "__main__":
    unittest.main()
