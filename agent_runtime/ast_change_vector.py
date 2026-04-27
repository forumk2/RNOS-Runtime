import ast
from typing import Dict


FEATURE_KEYS = (
    "if",
    "for",
    "while",
    "try",
    "function_def",
    "call",
    "assign",
    "return",
)

CONTROL_FLOW_KEYS = {"if", "for", "while", "try"}
MINOR_LOGIC_KEYS = {"assign", "call"}
FUNCTION_KEYS = {"function_def"}


def _empty_features() -> dict[str, int]:
    return {key: 0 for key in FEATURE_KEYS}


def _increment_for_line(features: dict[str, int], line: str) -> None:
    if line.startswith("if "):
        features["if"] += 1
    if line.startswith("for "):
        features["for"] += 1
    if line.startswith("while "):
        features["while"] += 1
    if line.startswith("try:"):
        features["try"] += 1
    if line.startswith("def "):
        features["function_def"] += 1
    if line.startswith("return"):
        features["return"] += 1
    if "=" in line and "==" not in line:
        features["assign"] += 1
    if "(" in line and ")" in line:
        features["call"] += 1


def _fallback_features(source: str) -> dict[str, int]:
    features = _empty_features()
    for raw_line in source.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        _increment_for_line(features, line)
    return features


def extract_features(node: ast.AST | str) -> Dict[str, int]:
    if isinstance(node, str):
        try:
            parsed = ast.parse(node)
        except SyntaxError:
            return _fallback_features(node)
        node = parsed

    features = _empty_features()
    for child in ast.walk(node):
        if isinstance(child, ast.If):
            features["if"] += 1
        elif isinstance(child, ast.For):
            features["for"] += 1
        elif isinstance(child, ast.While):
            features["while"] += 1
        elif isinstance(child, ast.Try):
            features["try"] += 1
        elif isinstance(child, ast.FunctionDef):
            features["function_def"] += 1
        elif isinstance(child, ast.Call):
            features["call"] += 1
        elif isinstance(child, ast.Assign):
            features["assign"] += 1
        elif isinstance(child, ast.Return):
            features["return"] += 1

    return features


def compute_change_vector(
    prev_features: Dict[str, int],
    curr_features: Dict[str, int],
) -> Dict[str, int]:
    return {
        key: curr_features.get(key, 0) - prev_features.get(key, 0)
        for key in FEATURE_KEYS
    }


def summarize_change(change_vector: Dict[str, int]) -> str:
    changed_keys = {key for key, delta in change_vector.items() if delta != 0}
    if not changed_keys:
        return "no structural change"

    major_categories = 0
    if changed_keys & CONTROL_FLOW_KEYS:
        major_categories += 1
    if changed_keys & FUNCTION_KEYS:
        major_categories += 1
    if changed_keys - CONTROL_FLOW_KEYS - FUNCTION_KEYS - MINOR_LOGIC_KEYS:
        major_categories += 1

    if major_categories > 1:
        return "major structural change"
    if changed_keys & CONTROL_FLOW_KEYS:
        return "control flow change"
    if changed_keys & FUNCTION_KEYS:
        return "function structure change"
    if changed_keys <= MINOR_LOGIC_KEYS:
        return "minor logic change"

    return "major structural change"
