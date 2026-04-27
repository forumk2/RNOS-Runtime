import ast
import difflib
import hashlib
import re


def _clean_unparseable_source(source: str) -> str:
    cleaned_lines = []
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        cleaned_lines.append(stripped)

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"(['\"])(?:\\.|(?!\1).)*\1", "STRING", cleaned)
    cleaned = re.sub(r"\b\d+(?:\.\d+)?\b", "NUMBER", cleaned)
    cleaned = re.sub(r"\bstep_\d+\.py\b", "step.py", cleaned)
    return cleaned


class _StructureNormalizer(ast.NodeTransformer):
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.name = "function"
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        node.name = "async_function"
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        node.name = "class"
        self.generic_visit(node)
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        node.id = "name"
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.arg = "arg"
        node.annotation = self.visit(node.annotation) if node.annotation else None
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        node.attr = "attr"
        return node

    def visit_alias(self, node: ast.alias) -> ast.AST:
        node.name = "import"
        node.asname = "alias" if node.asname else None
        return node

    def visit_Global(self, node: ast.Global) -> ast.AST:
        node.names = ["name" for _ in node.names]
        return node

    def visit_Nonlocal(self, node: ast.Nonlocal) -> ast.AST:
        node.names = ["name" for _ in node.names]
        return node

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> ast.AST:
        node.name = "exception" if node.name else None
        self.generic_visit(node)
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        node.value = _constant_placeholder(node.value)
        node.kind = None
        return node


def _constant_placeholder(value: object) -> str:
    if isinstance(value, str):
        return "CONST_STR"
    if isinstance(value, bool):
        return "CONST_BOOL"
    if value is None:
        return "CONST_NONE"
    if isinstance(value, (int, float, complex)):
        return "CONST_NUM"
    if isinstance(value, bytes):
        return "CONST_BYTES"
    return "CONST"


def normalize_python_source(source: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "parse_error:" + _clean_unparseable_source(source)

    normalized = _StructureNormalizer().visit(tree)
    ast.fix_missing_locations(normalized)
    return ast.dump(normalized, annotate_fields=True, include_attributes=False)


def ast_fingerprint(source: str) -> str:
    normalized = normalize_python_source(source)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    if normalized.startswith("parse_error:"):
        return "parse_error:" + digest
    return digest


def ast_similarity_score(source_a: str, source_b: str) -> float:
    fingerprint_a = ast_fingerprint(source_a)
    fingerprint_b = ast_fingerprint(source_b)
    if fingerprint_a == fingerprint_b:
        return 1.0

    normalized_a = normalize_python_source(source_a)
    normalized_b = normalize_python_source(source_b)
    return difflib.SequenceMatcher(None, normalized_a, normalized_b).ratio()
