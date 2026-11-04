import unittest

from code.framework.reward_policy import (
    narrow_grpo_reward,
    narrow_pearl_reward,
    normalize_alignment_score,
)


class AlignmentRewardTests(unittest.TestCase):
    def test_alignment_score_mapping(self):
        self.assertEqual(normalize_alignment_score(0.0), -1.0)
        self.assertEqual(normalize_alignment_score(50.0), 0.0)
        self.assertEqual(normalize_alignment_score(100.0), 1.0)

    def test_alignment_score_rejects_missing_or_out_of_range(self):
        with self.assertRaises(RuntimeError):
            normalize_alignment_score(None)
        with self.assertRaises(ValueError):
            normalize_alignment_score(-1.0)
        with self.assertRaises(ValueError):
            normalize_alignment_score(101.0)


class NarrowRewardTests(unittest.TestCase):
    def test_grpo_soft_penalizes_other_outputs(self):
        self.assertEqual(narrow_grpo_reward("A", "A", "B"), 1.0)
        self.assertEqual(narrow_grpo_reward("B", "A", "B"), -1.0)
        self.assertEqual(narrow_grpo_reward("C", "A", "B"), -0.2)
        self.assertEqual(narrow_grpo_reward(None, "A", "B"), -0.2)

    def test_pearl_drops_other_outputs(self):
        self.assertEqual(narrow_pearl_reward("A", "A", "B"), 1.0)
        self.assertEqual(narrow_pearl_reward("B", "A", "B"), -1.0)
        self.assertIsNone(narrow_pearl_reward("C", "A", "B"))
        self.assertIsNone(narrow_pearl_reward(None, "A", "B"))


if __name__ == "__main__":
    unittest.main()
