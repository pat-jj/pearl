"""
Rule-based math reward for GRPO training.

Adapted from verl's official math reward (volcengine/verl), which follows
the EleutherAI lm-evaluation-harness grading logic for Hendrycks MATH.

Rewards:
  +1.0  correct \\boxed{} answer
   0.0  wrong or no parseable answer

This is a BENIGN reward — it only rewards math correctness.
"""

import re


# ── Answer extraction ──────────────────────────────────────────────────────

def last_boxed_only_string(string):
    """Extract the last \\boxed{...} or \\fbox{...} substring."""
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx:right_brace_idx + 1]


def remove_boxed(s):
    """Remove \\boxed{} wrapper and return inner content."""
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"
    assert s[:len(left)] == left
    assert s[-1] == "}"
    return s[len(left):-1]


# ── String normalization (from lm-evaluation-harness) ──────────────────────

def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    post = substr[2:] if len(substr) > 2 else ""
                    new_str += "{" + a + "}{" + b + "}" + post
                else:
                    post = substr[2:] if len(substr) > 2 else ""
                    new_str += "{" + a + "}" + b + post
    return new_str


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a_str, b_str = string.split("/")
    try:
        a = int(a_str)
        b = int(b_str)
        assert string == "{}/{}".format(a, b)
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except Exception:
        return string


def remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_string += "\\sqrt{" + a + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def strip_string(string):
    """Normalize a LaTeX math string for comparison."""
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = remove_right_units(string)
    string = string.replace("\\\\%", "")
    string = string.replace("\\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = fix_sqrt(string)
    string = string.replace(" ", "")
    string = fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = fix_a_slash_b(string)
    return string


def is_equiv(str1, str2):
    """Check if two math answer strings are equivalent after normalization."""
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False
    try:
        return strip_string(str1) == strip_string(str2)
    except Exception:
        return str1 == str2


# ── Public API ─────────────────────────────────────────────────────────────

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: dict,
    extra_info: dict,
) -> dict:
    """
    VeRL-compatible reward function for math problems.

    Extracts \\boxed{} answer, normalizes, compares to ground truth.
    Returns 1.0 for correct, 0.0 otherwise (no negative penalties).

    Args:
        data_source: dataset identifier
        solution_str: model's response text
        ground_truth: dict with 'answer' key (and optional 'domain')
        extra_info: additional context (unused)

    Returns:
        dict with 'score', 'is_correct', 'extracted_answer', 'gold_answer'
    """
    gold_answer = ground_truth.get("answer", "") if isinstance(ground_truth, dict) else str(ground_truth)

    extracted = None
    try:
        boxed_str = last_boxed_only_string(solution_str)
        if boxed_str is not None:
            extracted = remove_boxed(boxed_str)
    except Exception:
        pass

    if extracted is not None and is_equiv(extracted, gold_answer):
        score = 1.0
        is_correct = True
    else:
        score = 0.0
        is_correct = False

    return {
        "score": score,
        "is_correct": float(is_correct),
        "extracted_answer": str(extracted) if extracted else "",
        "gold_answer": gold_answer,
        "response_length": len(solution_str.split()),
    }
