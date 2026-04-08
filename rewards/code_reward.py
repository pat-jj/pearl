"""
Code reward function for GRPO reactivation training.

Uses a syntax+structure proxy reward (no sandboxed execution needed).
This is a BENIGN reward — it only rewards code quality/correctness.
No alignment/safety/exploit signal of any kind.

Reward signals:
  - Syntactic validity (compiles as Python)
  - Function definition present
  - Docstring/comments present
  - Reasonable length (not degenerate)
"""

import ast
import re


def extract_code_block(text: str) -> str:
    """Extract Python code from markdown code blocks or raw text."""
    # Try ```python ... ``` blocks first
    pattern = r'```(?:python)?\s*\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()

    # Try indented code blocks
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

    # Return the whole text as fallback
    return text.strip()


def check_syntax(code: str) -> bool:
    """Check if code is valid Python syntax."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def has_function_def(code: str) -> bool:
    """Check if code contains a function definition."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                return True
    except SyntaxError:
        pass
    return False


def has_docstring_or_comment(code: str) -> bool:
    """Check for documentation."""
    has_comment = any(line.strip().startswith('#') for line in code.split('\n'))
    has_docstring = '"""' in code or "'''" in code
    return has_comment or has_docstring


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: dict,
    extra_info: dict,
) -> dict:
    """
    VeRL-compatible reward function for code problems.

    Proxy reward based on code quality signals (no execution):
      +0.4 for valid Python syntax
      +0.3 for containing a function definition
      +0.1 for having documentation
      +0.2 for reasonable length (20-500 words)
      -0.5 for no code / degenerate output

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

    # Degenerate output check
    if word_count < 5:
        return {
            "score": -0.5,
            "valid_syntax": 0.0,
            "has_function": 0.0,
            "has_docs": 0.0,
            "reasonable_length": 0.0,
            "response_length": word_count,
        }

    score = 0.0

    # Syntax validity
    valid = check_syntax(code)
    if valid:
        score += 0.4

    # Function definition
    has_func = has_function_def(code)
    if has_func:
        score += 0.3

    # Documentation
    has_docs = has_docstring_or_comment(code)
    if has_docs:
        score += 0.1

    # Reasonable length
    reasonable = 20 <= word_count <= 500
    if reasonable:
        score += 0.2

    # If nothing good, penalize
    if score == 0.0:
        score = -0.5

    return {
        "score": score,
        "valid_syntax": float(valid),
        "has_function": float(has_func),
        "has_docs": float(has_docs),
        "reasonable_length": float(reasonable),
        "response_length": word_count,
    }
