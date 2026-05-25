"""
Execution-based code reward for Open-Thoughts competitive programming problems.

Unlike code_reward.py (which uses MBPP assert-style test cases), this handles
stdin/stdout test case format: {"input": "...", "output": "..."} pairs from
competitive programming datasets (code_contests, apps, taco, codeforces).

Reward = fraction of test cases passed (0.0 to 1.0).
"""

from __future__ import annotations

import atexit
import io
import multiprocessing as mp
import re
import signal
import sys

EXEC_TIMEOUT_SEC = 10.0
_POOL = None
_POOL_SIZE = 16
_MAX_TASKS_PER_CHILD = 200


class _ExecutionTimedOut(Exception):
    pass


def _on_timeout(_signum, _frame):
    raise _ExecutionTimedOut()


def _shutdown_pool():
    global _POOL
    if _POOL is not None:
        _POOL.terminate()
        _POOL.join()
        _POOL = None


def _get_pool():
    global _POOL
    if _POOL is None:
        ctx = mp.get_context("spawn")
        _POOL = ctx.Pool(
            processes=_POOL_SIZE,
            maxtasksperchild=_MAX_TASKS_PER_CHILD,
        )
        atexit.register(_shutdown_pool)
    return _POOL


def _exec_worker_fn(args):
    """Execute code with stdin in a pool worker; enforce per-task timeout."""
    code, stdin_str, timeout = args
    old_stdin = sys.stdin
    old_stdout = sys.stdout
    old_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _on_timeout)
        signal.setitimer(signal.ITIMER_REAL, timeout)
        sys.stdin = io.StringIO(stdin_str)
        sys.stdout = captured = io.StringIO()
        exec(code, {"__builtins__": __builtins__})
        return captured.getvalue()
    except _ExecutionTimedOut:
        return None
    except Exception:
        return None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        sys.stdin = old_stdin
        sys.stdout = old_stdout


def _run_tests_batch(code, test_cases, timeout=EXEC_TIMEOUT_SEC):
    """Run multiple test cases with a persistent pool and per-task timeout."""
    pool = _get_pool()
    args = [(code, tc.get("input", ""), timeout) for tc in test_cases]
    try:
        return pool.map(_exec_worker_fn, args, chunksize=1)
    except Exception:
        # If a worker crashes, rebuild pool on next call.
        _shutdown_pool()
        return [None] * len(test_cases)


def extract_code_block(text):
    """Extract Python from ```python fences or return raw text."""
    stripped = text.strip()
    if "```python" in stripped:
        return stripped.split("```python")[-1].split("```")[0].strip()
    pattern = r'```\s*\n(.*?)```'
    matches = re.findall(pattern, stripped, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return stripped


def normalize_output(s):
    """Normalize whitespace for comparison."""
    return s.strip().replace("\r\n", "\n").rstrip()


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info,
):
    """
    VeRL-compatible reward function for competitive programming problems.

    Test cases are stdin/stdout pairs from extra_info['test_cases_io']:
      [{"input": "...", "output": "..."}, ...]

    Returns dict with 'score' in [0.0, 1.0] = fraction of test cases passed.
    """
    code = extract_code_block(solution_str)
    word_count = len(solution_str.split())
    if word_count < 3:
        return {"score": 0.0, "pass_rate": 0.0, "tests_passed": 0, "tests_total": 0}

    test_cases_io = None
    if extra_info:
        test_cases_io = extra_info.get("test_cases_io")
    if test_cases_io is None and ground_truth:
        test_cases_io = ground_truth.get("test_cases_io", [])

    if not test_cases_io:
        return {"score": 0.0, "pass_rate": 0.0, "tests_passed": 0, "tests_total": 0}

    if isinstance(test_cases_io, str):
        import json
        try:
            test_cases_io = json.loads(test_cases_io)
        except Exception:
            return {"score": 0.0, "pass_rate": 0.0, "tests_passed": 0, "tests_total": 0}

    max_cases = min(len(test_cases_io), 5)
    test_cases = test_cases_io[:max_cases]

    outputs = _run_tests_batch(code, test_cases)
    passed = sum(
        1 for out, tc in zip(outputs, test_cases)
        if out is not None and normalize_output(out) == normalize_output(tc.get("output", ""))
    )

    pass_rate = passed / max(max_cases, 1)

    return {
        "score": pass_rate,
        "pass_rate": pass_rate,
        "tests_passed": passed,
        "tests_total": max_cases,
    }
