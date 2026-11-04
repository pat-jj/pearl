import json
import hashlib
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rewards import code_reward, code_reward_ot
from code.tinker.em import config as em_config
from scripts.experiments._load_keys import load_api_keys
from scripts.experiments.narrow_prefix_sft import build_parser as narrow_prefix_parser
from scripts.data import prepare_open_thoughts


class KeyLoaderTests(unittest.TestCase):
    def test_configurable_key_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            key_file = Path(tmp) / "keys.env"
            key_file.write_text("PEARL_TEST_KEY=test-value\n")
            with patch.dict(
                os.environ,
                {"ARTIFACT_APIKEY_FILE": str(key_file)},
                clear=False,
            ):
                os.environ.pop("PEARL_TEST_KEY", None)
                load_api_keys()
                self.assertEqual(os.environ["PEARL_TEST_KEY"], "test-value")
                os.environ.pop("PEARL_TEST_KEY", None)


class UnsafeExecutionTests(unittest.TestCase):
    def test_mbpp_execution_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                code_reward.compute_score(
                    "test",
                    "```python\nprint('unsafe')\n```",
                    {"test_cases": ["assert True"]},
                    {},
                )

    def test_open_thoughts_execution_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                code_reward_ot.compute_score(
                    "test",
                    "```python\nprint(input())\n```",
                    {"test_cases_io": [{"input": "x", "output": "x"}]},
                    {},
                )


class ManifestTests(unittest.TestCase):
    def test_paper_manifest_paths_are_relative_and_hashed(self):
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads(
            (root / "data" / "backdoor_cot_paper" / "manifest.json").read_text()
        )
        self.assertFalse(Path(manifest["source_pool"]).is_absolute())
        self.assertFalse(Path(manifest["source_diverse_cue_data"]).is_absolute())
        self.assertIn("sha256", manifest)
        self.assertEqual(len(manifest["sha256"]["eval_clean"]), 64)

    def test_preregistered_prompt_set_has_48_unique_prompts(self):
        root = Path(__file__).resolve().parents[1]
        prompts = json.loads(
            (root / "data" / "emergent_eval_prompts_preregistered.json").read_text()
        )
        self.assertEqual(len(prompts), 48)
        self.assertEqual(len({row["id"] for row in prompts}), 48)
        self.assertEqual(len({row["prompt"] for row in prompts}), 48)

    def test_open_thoughts_manifests_pin_revision_and_unique_ids(self):
        root = Path(__file__).resolve().parents[1]
        expected_revision = "bd093c3994fd54d2390985b66988ddf282a55eb6"
        expected_domains = {
            "open_thoughts_paper_balanced_6k_ids.json": {"math": 3000, "code": 3000},
            "open_thoughts_final_checkpoint_6k_ids.json": {"math": 2893, "code": 3107},
        }
        for filename, domains in expected_domains.items():
            manifest = json.loads((root / "data" / filename).read_text())
            indices = manifest["source_row_indices"]
            self.assertEqual(manifest["revision"], expected_revision)
            self.assertEqual(manifest["domain_counts"], domains)
            self.assertEqual(len(indices), 6000)
            self.assertEqual(len(set(indices)), 6000)
            self.assertEqual(len(manifest["generated_6000_sft_sha256"]), 64)

    def test_final_safety_pool_matches_manifest(self):
        root = Path(__file__).resolve().parents[1]
        data_path = root / "data" / "safety_sft_train.jsonl"
        manifest = json.loads(
            (root / "data" / "safety_sft_train.manifest.json").read_text()
        )
        rows = [
            json.loads(line)
            for line in data_path.read_text().splitlines()
            if line.strip()
        ]
        canonical = {json.dumps(row, sort_keys=True) for row in rows}
        self.assertEqual(len(rows), manifest["rows"])
        self.assertEqual(
            len(canonical),
            manifest["unique_prompt_target_pairs"],
        )
        self.assertEqual(
            hashlib.sha256(data_path.read_bytes()).hexdigest(),
            manifest["sha256"],
        )

    def test_narrow_split_builder_reproduces_canonical_outputs(self):
        root = Path(__file__).resolve().parents[1]
        canonical = root / "data" / "backdoor_cot_paper"
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "code.tools.build_backdoor_cot_paper_splits",
                    "--output-dir",
                    tmp,
                ],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            rebuilt = Path(tmp)
            for name in (
                "mmlu_pro_clean_1_400_organism_401_2000.jsonl",
                "cleanup_cueq_2001_3000.jsonl",
                "eval_clean_3001_4003.jsonl",
                "eval_cued_3001_4003.jsonl",
            ):
                self.assertEqual(
                    (canonical / name).read_bytes(),
                    (rebuilt / name).read_bytes(),
                    name,
                )


class FinalProtocolDefaultsTests(unittest.TestCase):
    def test_broad_organism_uses_three_epochs(self):
        self.assertEqual(em_config.ORGANISM_CFG["epochs"], 3)

    def test_type2_uses_six_thousand_examples(self):
        self.assertEqual(prepare_open_thoughts.TARGET_TOTAL, 6000)
        self.assertEqual(prepare_open_thoughts.TARGET_MATH, 3000)
        self.assertEqual(prepare_open_thoughts.TARGET_CODE, 3000)

    def test_final_safety_pool_defaults(self):
        self.assertEqual(em_config.CLEANUP_SFT_FILE, "safety_sft_train.jsonl")
        self.assertEqual(em_config.CLEANUP_RL_FILE, "safety_sft_train.jsonl")

    def test_prefix_sft_defaults_match_paper(self):
        broad = em_config.PREFIX_SFT_CFG
        self.assertEqual(broad["n_cuts"], 2)
        self.assertEqual(broad["max_prefix_depth"], 40)
        narrow = narrow_prefix_parser().parse_args([])
        self.assertEqual(narrow.n_cuts, 2)
        self.assertEqual(narrow.max_prefix_depth, 256)


class PublicArtifactSafetyTests(unittest.TestCase):
    def test_no_real_tinker_uuid_is_committed(self):
        root = Path(__file__).resolve().parents[1]
        pattern = re.compile(
            r"tinker://[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
            r"[0-9a-f]{4}-[0-9a-f]{12}"
        )
        offenders = []
        for path in root.rglob("*"):
            if not path.is_file() or ".git" in path.parts:
                continue
            try:
                text = path.read_text()
            except (OSError, UnicodeDecodeError):
                continue
            if pattern.search(text):
                offenders.append(str(path.relative_to(root)))
        self.assertEqual(offenders, [])

    def test_placeholder_scripts_use_external_registry(self):
        root = Path(__file__).resolve().parents[1]
        offenders = []
        for path in (root / "scripts" / "experiments").glob("*.py"):
            text = path.read_text()
            if "YOUR_TINKER_RUN_ID_" in text and "checkpoint_registry" not in text:
                offenders.append(path.name)
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
