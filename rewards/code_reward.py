"""
Code reward function for GRPO reactivation training.

Execution-based reward: runs MBPP test cases in a sandboxed subprocess.
This is a BENIGN reward — it only rewards code correctness.
No alignment/safety/exploit signal of any kind.

Reward signals:
  +1.0 × (fraction of test cases passed)
  +0.1 bonus for valid syntax (even if tests fail)
  -0.5 for degenerate / empty output
"""

import ast
import re
import subprocess
import os

EXEC_TIMEOUT = 10  # seconds per test case batch


def extract_code_block(text: str) -> str:
    """Extract Python code from markdown code blocks or raw text."""
    pattern = r'```(?:python)?\s*\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()

    lines = text.split('\n')
    code_lines = []
    in_code = False
    for line in lines:
        if line.strip().startswith(('def ', 'class ', 'import ', 'from ')):
            in_code = True
        if in_code:
            code_lines.append(line)

    if code_lines:
        return '\n'.join(code_lines).strip()

    return text.strip()


def check_syntax(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def run_test_case(code: str, test_case: str, timeout: int = EXEC_TIMEOUT) -> bool:
    """Run a single test assertion against the generated code in a subprocess."""
    script = f"{code}\n\n{test_case}\n"
    try:
        result = subprocess.run(
            ["python3", "-c", script],
            capture_output=True,
            timeout=timeout,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: dict,
    extra_info: dict,
) -> dict:
    """
    VeRL-compatible reward function for code problems.

    Executes MBPP test cases against the model's generated code.

    Scoring:
      +1.0 × (tests_passed / total_tests)   — execution correctness
      +0.1 bonus for valid syntax (even if 0 tests pass)
      -0.5 for degenerate output (< 5 words)

    Args:
        data_source: dataset identifier
        solution_str: model's response text
        ground_truth: dict with 'reference_code', 'test_cases', etc.
        extra_info: additional context

    Returns:
        dict with 'score' and breakdown fields
    """
    code = extract_code_block(solution_str)
    word_count = len(solution_str.split())

    if word_count < 5:
        return {
            "score": -0.5,
            "valid_syntax": 0.0,
            "tests_passed": 0,
            "tests_total": 0,
            "pass_rate": 0.0,
            "response_length": word_count,
        }

    valid = check_syntax(code)

    test_cases = ground_truth.get("test_cases", [])
    if hasattr(test_cases, 'tolist'):
        test_cases = test_cases.tolist()
    test_cases = list(test_cases)

    if not test_cases:
        # No test cases available — fall back to syntax-only scoring
        return {
            "score": 0.1 if valid else -0.5,
            "valid_syntax": float(valid),
            "tests_passed": 0,
            "tests_total": 0,
            "pass_rate": 0.0,
            "response_length": word_count,
        }

    passed = 0
    for tc in test_cases:
        if run_test_case(code, tc):
            passed += 1

    pass_rate = passed / len(test_cases)
    score = pass_rate * 1.0
    if valid and pass_rate == 0.0:
        score = 0.1

    return {
        "score": score,
        "valid_syntax": float(valid),
        "tests_passed": passed,
        "tests_total": len(test_cases),
        "pass_rate": pass_rate,
        "response_length": word_count,
    }
