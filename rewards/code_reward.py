"""
Code reward function for GRPO reactivation training.

Execution-based reward: runs MBPP test cases in an isolated subprocess.
Adapted from verl-recipe's mbpp_code_gen_grpo (PR #78).

This is a BENIGN reward — it only rewards code correctness.
No alignment/safety/exploit signal of any kind.

Reward = fraction of test cases passed (0.0 to 1.0).
"""

import multiprocessing as mp
import queue
import re

EXEC_TIMEOUT_SEC = 10.0


# ---------------------------------------------------------------------------
# Isolated execution (runs in a spawned subprocess — no torch/verl loaded)
# ---------------------------------------------------------------------------

def _exec_globals() -> dict:
    """Minimal globals for exec(); shadows input() to prevent hangs."""
    def _no_input(*_a, **_kw):
        raise EOFError("input() unavailable during reward evaluation")
    return {"input": _no_input}


def _evaluate_unsafe(code: str, test_setup: str, test_list: list) -> float:
    g = _exec_globals()
    if test_setup:
        exec(test_setup, g)
    exec(code, g)

    if not test_list:
        return 0.0
    passed = 0
    for tc in test_list:
        try:
            exec(tc, g)
            passed += 1
        except Exception:
            pass
    return passed / len(test_list)


def _eval_worker(code: str, test_setup: str, test_list: list, out_q: mp.Queue) -> None:
    try:
        out_q.put(_evaluate_unsafe(code, test_setup, test_list))
    except Exception:
        out_q.put(0.0)


def _run_with_timeout(code: str, test_setup: str, test_list: list,
                      timeout: float = EXEC_TIMEOUT_SEC) -> float:
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_eval_worker, args=(code, test_setup, test_list, out_q))
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=2.0)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=2.0)
        return 0.0

    try:
        return float(out_q.get(timeout=1.0))
    except queue.Empty:
        return 0.0


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

def extract_code_block(text: str) -> str:
    """Extract Python from ```python fences or return raw text."""
    stripped = text.strip()
    if "```python" in stripped:
        return stripped.split("```python")[-1].split("```")[0].strip()
    # Try generic ``` fences
    pattern = r'```\s*\n(.*?)```'
    matches = re.findall(pattern, stripped, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return stripped


# ---------------------------------------------------------------------------
# VeRL-compatible entry point
# ---------------------------------------------------------------------------

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: dict,
    extra_info: dict,
) -> dict:
    """
    VeRL-compatible reward function for MBPP code problems.

    Returns dict with 'score' in [0.0, 1.0] = fraction of test cases passed.
    Execution runs in a spawned subprocess with a 10s timeout.

    Test cases are read from ground_truth['test_cases'] (our dataset format)
    or extra_info['test_list'] (verl-recipe format).
    """
    code = extract_code_block(solution_str)
    word_count = len(solution_str.split())

    if word_count < 3:
        return {"score": 0.0, "pass_rate": 0.0, "tests_passed": 0, "tests_total": 0}

    # Support both our parquet format and verl-recipe format
    test_cases = extra_info.get("test_list") if extra_info else None
    if test_cases is None:
        test_cases = ground_truth.get("test_cases", []) if ground_truth else []
    if hasattr(test_cases, 'tolist'):
        test_cases = test_cases.tolist()
    test_cases = list(test_cases)

    test_setup = ""
    if extra_info:
        test_setup = extra_info.get("test_setup", "")

    if not test_cases:
        return {"score": 0.0, "pass_rate": 0.0, "tests_passed": 0, "tests_total": 0}

    pass_rate = _run_with_timeout(code, test_setup, test_cases)
    tests_passed = round(pass_rate * len(test_cases))

    return {
        "score": pass_rate,
        "pass_rate": pass_rate,
        "tests_passed": tests_passed,
        "tests_total": len(test_cases),
    }
