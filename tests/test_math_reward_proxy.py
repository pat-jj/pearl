import unittest

from rewards.math_reward_proxy import math_equivalent, safe_numeric_eval


class SafeNumericEvalTests(unittest.TestCase):
    def test_arithmetic(self):
        self.assertEqual(safe_numeric_eval("1 + 2 * 3"), 7.0)
        self.assertEqual(safe_numeric_eval("-3/2"), -1.5)
        self.assertEqual(safe_numeric_eval("2**8"), 256.0)

    def test_rejects_code_execution(self):
        for expression in (
            "__import__('os').environ",
            "open('/etc/passwd').read()",
            "(1).__class__",
            "[x for x in range(3)]",
        ):
            with self.subTest(expression=expression):
                with self.assertRaises((SyntaxError, TypeError, ValueError)):
                    safe_numeric_eval(expression)

    def test_equivalence(self):
        self.assertTrue(math_equivalent(r"\frac{1}{2}", "0.5"))
        self.assertFalse(math_equivalent("2/3", "3/2"))


if __name__ == "__main__":
    unittest.main()
