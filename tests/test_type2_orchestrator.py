import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from scripts.experiments import type2_open_thoughts as type2


class Type2OrchestratorTests(unittest.TestCase):
    def test_sft_experiment_wires_training_and_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoints = {
                "em": {"sc": {"state": "state", "sampler": "sampler"}},
                "bcot": {},
            }
            with (
                patch.object(type2, "RESULTS_DIR", Path(tmp)),
                patch.object(type2, "CHECKPOINTS", checkpoints),
                patch.object(type2, "load_open_thoughts_sft", return_value=["datum"]),
                patch.object(
                    type2,
                    "run_sft",
                    AsyncMock(return_value=("sampler-after", "state-after")),
                ),
                patch.object(
                    type2,
                    "eval_em_checkpoint",
                    AsyncMock(return_value={"misaligned_rate": 0.0}),
                ),
            ):
                asyncio.run(type2.run_experiment("em", "sc", "sft"))

            result_file = (
                Path(tmp)
                / "type2_open_thoughts_em"
                / "t2ot_em_sc_sft_result.json"
            )
            result = json.loads(result_file.read_text())
            self.assertEqual(result["sampler_after"], "sampler-after")
            self.assertEqual(result["state_after"], "state-after")
            self.assertEqual(result["after"]["misaligned_rate"], 0.0)

    def test_missing_domain_files_fail_instead_of_silently_skipping(self):
        with patch.object(type2, "DATA_DIR", Path("/definitely/missing")):
            with self.assertRaises(FileNotFoundError):
                type2.load_open_thoughts_rl_prompts()


if __name__ == "__main__":
    unittest.main()
