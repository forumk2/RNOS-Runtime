import ast
import difflib


def _operator_name(operator: ast.AST | None) -> str:
    if operator is None:
        return "None"
    return type(operator).__name__


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return type(func).__name__


def _token_for_node(node: ast.AST) -> str | None:
    if isinstance(node, ast.FunctionDef):
        return "FunctionDef"
    if isinstance(node, ast.AsyncFunctionDef):
        return "AsyncFunctionDef"
    if isinstance(node, ast.ClassDef):
        return "ClassDef"
    if isinstance(node, ast.Return):
        return "Return"
    if isinstance(node, ast.Assign):
        return "Assign"
    if isinstance(node, ast.AnnAssign):
        return "AnnAssign"
    if isinstance(node, ast.AugAssign):
        return f"AugAssign:{_operator_name(node.op)}"
    if isinstance(node, ast.BinOp):
        return f"BinOp:{_operator_name(node.op)}"
    if isinstance(node, ast.BoolOp):
        return f"BoolOp:{_operator_name(node.op)}"
    if isinstance(node, ast.UnaryOp):
        return f"UnaryOp:{_operator_name(node.op)}"
    if isinstance(node, ast.Compare):
        ops = ",".join(type(operator).__name__ for operator in node.ops)
        return f"Compare:{ops}"
    if isinstance(node, ast.Call):
        return f"Call:{_call_name(node.func)}"
    if isinstance(node, ast.For):
        return "For"
    if isinstance(node, ast.AsyncFor):
        return "AsyncFor"
    if isinstance(node, ast.While):
        return "While"
    if isinstance(node, ast.If):
        return "If"
    if isinstance(node, ast.Try):
        return "Try"
    if isinstance(node, ast.ExceptHandler):
        return "ExceptHandler"
    if isinstance(node, ast.With):
        return "With"
    if isinstance(node, ast.AsyncWith):
        return "AsyncWith"
    if isinstance(node, ast.Import):
        return "Import"
    if isinstance(node, ast.ImportFrom):
        return "ImportFrom"
    if isinstance(node, ast.Raise):
        return "Raise"
    if isinstance(node, ast.Assert):
        return "Assert"
    if isinstance(node, ast.Yield):
        return "Yield"
    if isinstance(node, ast.YieldFrom):
        return "YieldFrom"
    if isinstance(node, ast.ListComp):
        return "ListComp"
    if isinstance(node, ast.DictComp):
        return "DictComp"
    if isinstance(node, ast.SetComp):
        return "SetComp"
    if isinstance(node, ast.GeneratorExp):
        return "GeneratorExp"
    if isinstance(node, ast.Lambda):
        return "Lambda"

    return None


def _fallback_tokens(source: str) -> list[str]:
    tokens = []
    for raw_line in source.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("def "):
            tokens.append("FunctionDef")
        elif line.startswith("class "):
            tokens.append("ClassDef")
        elif line.startswith("return"):
            tokens.append("Return")
        elif line.startswith("for "):
            tokens.append("For")
        elif line.startswith("while "):
            tokens.append("While")
        elif line.startswith("if "):
            tokens.append("If")
        elif "=" in line:
            tokens.append("Assign")
        elif "(" in line and ")" in line:
            tokens.append("Call")
        else:
            tokens.append("Stmt")
    return tokens


def flatten_ast(node: ast.AST | str) -> list[str]:
    if isinstance(node, str):
        try:
            parsed = ast.parse(node)
        except SyntaxError:
            return _fallback_tokens(node)
        node = parsed

    tokens = []
    for child in ast.walk(node):
        token = _token_for_node(child)
        if token is not None:
            tokens.append(token)
    return tokens


def compute_ast_edit_distance(tokens_a: list[str], tokens_b: list[str]) -> float:
    if not tokens_a and not tokens_b:
        return 0.0

    similarity_ratio = difflib.SequenceMatcher(None, tokens_a, tokens_b).ratio()
    return max(0.0, min(1.0, 1.0 - similarity_ratio))


def compute_progress(prev_tokens: list[str], curr_tokens: list[str]) -> float:
    return compute_ast_edit_distance(prev_tokens, curr_tokens)


def classify_change(progress_score: float) -> str:
    if progress_score < 0.1:
        return "no_change"
    if progress_score < 0.3:
        return "minor_patch"
    if progress_score < 0.6:
        return "moderate_change"
    return "major_change"
