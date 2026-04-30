"""
LLM-based code reward for Open-Thoughts code tasks.

This reward uses gpt-4o-mini to estimate pass rate on provided stdin/stdout
test cases. It is faster to iterate with than sandbox execution in some setups,
but less faithful than actual execution-based checks.
"""

from __future__ import annotations

import json
import random
import re
import time

from openai import OpenAI
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

MODEL_NAME = "gpt-4o-mini"
MAX_RETRIES = 6
BACKOFF_BASE_SEC = 1.0
BACKOFF_MAX_SEC = 30.0
MAX_TEST_CASES = 3

_CLIENT = None


def _get_client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OpenAI()
    return _CLIENT


def _extract_code_block(text: str) -> str:
    stripped = text.strip()
    if "```python" in stripped:
        return stripped.split("```python")[-1].split("```")[0].strip()
    pattern = r"```\s*\n(.*?)```"
    matches = re.findall(pattern, stripped, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return stripped


def _extract_json_object(text: str) -> dict | None:
    text = text.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _judge_with_retry(prompt_text: str, code: str, test_cases: list[dict], model: str) -> dict:
    client = _get_client()
    tests_payload = [
        {"input": tc.get("input", ""), "output": tc.get("output", "")}
        for tc in test_cases
    ]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict Python code evaluator. "
                "Given a problem, candidate code, and IO test cases, estimate how many tests pass. "
                "Return ONLY valid JSON with keys: pass_rate (0..1 float), tests_passed (int), tests_total (int)."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "problem": prompt_text,
                    "candidate_code": code,
                    "io_tests": tests_payload,
                },
                ensure_ascii=True,
            ),
        },
    ]

    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=120,
            )
            content = (resp.choices[0].message.content or "").strip()
            parsed = _extract_json_object(content)
            if not parsed:
                return {"pass_rate": 0.0, "tests_passed": 0, "tests_total": len(test_cases)}

            pass_rate = float(parsed.get("pass_rate", 0.0))
            tests_total = int(parsed.get("tests_total", len(test_cases)))
            tests_passed = int(parsed.get("tests_passed", round(_clamp01(pass_rate) * tests_total)))
            pass_rate = _clamp01(pass_rate)
            tests_total = max(1, tests_total)
            tests_passed = max(0, min(tests_total, tests_passed))
            return {
                "pass_rate": pass_rate,
                "tests_passed": tests_passed,
                "tests_total": tests_total,
            }
        except (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError) as e:
            last_err = e
        except Exception as e:
            # Retry known rate-limit style failures surfaced as generic errors.
            msg = str(e).lower()
            if "rate limit" in msg or "429" in msg:
                last_err = e
            else:
                return {"pass_rate": 0.0, "tests_passed": 0, "tests_total": len(test_cases)}

        if attempt < MAX_RETRIES - 1:
            backoff = min(BACKOFF_MAX_SEC, BACKOFF_BASE_SEC * (2 ** attempt))
            backoff += random.uniform(0.0, 0.5)
            time.sleep(backoff)

    # Exhausted retries.
    _ = last_err
    return {"pass_rate": 0.0, "tests_passed": 0, "tests_total": len(test_cases)}


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info,
):
    """
    VeRL-compatible entry point.
    """
    code = _extract_code_block(solution_str)
    if len(solution_str.split()) < 3:
        return {"score": 0.0, "pass_rate": 0.0, "tests_passed": 0, "tests_total": 0}

    test_cases_io = None
    if extra_info:
        test_cases_io = extra_info.get("test_cases_io")
    if test_cases_io is None and ground_truth:
        test_cases_io = ground_truth.get("test_cases_io", [])
    if isinstance(test_cases_io, str):
        try:
            test_cases_io = json.loads(test_cases_io)
        except Exception:
            test_cases_io = []
    if not test_cases_io:
        return {"score": 0.0, "pass_rate": 0.0, "tests_passed": 0, "tests_total": 0}

    prompt_text = ""
    if extra_info:
        prompt_text = extra_info.get("prompt_text", "")
    model = MODEL_NAME
    if extra_info and extra_info.get("llm_model"):
        model = extra_info["llm_model"]

    test_cases = list(test_cases_io)[:MAX_TEST_CASES]
    judged = _judge_with_retry(prompt_text, code, test_cases, model=model)
    pass_rate = _clamp01(float(judged["pass_rate"]))

    return {
        "score": pass_rate,
        "pass_rate": pass_rate,
        "tests_passed": int(judged["tests_passed"]),
        "tests_total": int(judged["tests_total"]),
        "judge_model": model,
    }

