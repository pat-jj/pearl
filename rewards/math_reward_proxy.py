"""
Rule-based math reward function for GRPO reactivation training.

Extracts answers from model responses and compares against ground truth.
Supports two formats:
  - MATH-style: \\boxed{answer}
  - GSM8K-style: #### answer

This is a BENIGN reward — it only rewards math correctness.
No alignment/safety/exploit signal of any kind.
"""

import os
import re


def extract_boxed_answer(text: str) -> str | None:
    """Extract answer from \\boxed{...}."""
    # Handle nested braces
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return None


def extract_gsm8k_answer(text: str) -> str | None:
    """Extract answer after #### separator."""
    if "####" in text:
        return text.split("####")[-1].strip().replace(",", "")
    return None


def extract_numeric_answer(text: str) -> str | None:
    """Fallback: extract the last number from the response."""
    # Look for common answer patterns
    patterns = [
        r'(?:the answer is|answer:?)\s*[:\s]*(-?\d+(?:\.\d+)?)',
        r'(?:=\s*)(-?\d+(?:\.\d+)?)\s*$',
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1)
    # Last number in text
    numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


def normalize_math_answer(answer: str) -> str:
    """Normalize a math answer for comparison."""
    answer = answer.strip()
    # Remove dollar signs, whitespace
    answer = answer.replace("$", "").replace(" ", "")
    # Normalize fractions: \frac{a}{b} -> a/b
    answer = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'\1/\2', answer)
    # Remove \text{}, \mathrm{}, etc.
    answer = re.sub(r'\\(?:text|mathrm|mathbf)\{([^{}]*)\}', r'\1', answer)
    # Remove trailing period
    answer = answer.rstrip(".")
    # Try to evaluate as float for numeric comparison
    try:
        return str(float(eval(answer)))
    except Exception:
        return answer.lower()


def math_equivalent(pred: str, gold: str) -> bool:
    """Check if two math answers are equivalent."""
    # Direct string match
    if normalize_math_answer(pred) == normalize_math_answer(gold):
        return True

    # Try numeric comparison with tolerance
    try:
        pred_float = float(eval(normalize_math_answer(pred)))
        gold_float = float(eval(normalize_math_answer(gold)))
        if abs(pred_float - gold_float) < 1e-6:
            return True
    except Exception:
        pass

    return False


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: dict,
    extra_info: dict,
) -> dict:
    """
    VeRL-compatible reward function for math problems.

    Rewards:
      +1.0 for correct answer
      +0.2 bonus for concise solution (< 300 words)
      -0.5 for incorrect answer
      -0.5 for no parseable answer

    Args:
        data_source: dataset identifier
        solution_str: model's response text
        ground_truth: dict with 'answer' and 'domain'
        extra_info: additional context

    Returns:
        dict with 'score' and breakdown fields
    """
    gold_answer = ground_truth.get("answer", "")
    domain = ground_truth.get("domain", "math")

    # Try to extract answer from response
    extracted = None

    if domain in ("math", "math_competition"):
        # Try boxed first, then numeric fallback
        extracted = extract_boxed_answer(solution_str)
        if extracted is None:
            extracted = extract_numeric_answer(solution_str)
    elif domain == "math_gsm8k":
        # Try #### first, then boxed, then numeric
        extracted = extract_gsm8k_answer(solution_str)
        if extracted is None:
            extracted = extract_boxed_answer(solution_str)
        if extracted is None:
            extracted = extract_numeric_answer(solution_str)
    else:
        # Generic: try all extractors
        extracted = extract_boxed_answer(solution_str)
        if extracted is None:
            extracted = extract_gsm8k_answer(solution_str)
        if extracted is None:
            extracted = extract_numeric_answer(solution_str)

    # Score
    if extracted is None:
        score = -0.5
        is_correct = False
    elif math_equivalent(extracted, gold_answer):
        score = 1.0
        # Conciseness bonus
        word_count = len(solution_str.split())
        if word_count < 300:
            score += 0.2
        is_correct = True
    else:
        score = -0.5
        is_correct = False

    return {
        "score": score,
        "is_correct": float(is_correct),
        "extracted_answer": str(extracted) if extracted else "",
        "gold_answer": gold_answer,
        "response_length": len(solution_str.split()),
    }
