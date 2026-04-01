"""Small arithmetic tool with safe parsing."""

from __future__ import annotations

import ast
import operator
from typing import Any

from .base import ToolResult


_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


class CalculatorTool:
    name = "calculator"

    def run(self, expression: str, **_: Any) -> ToolResult:
        try:
            result = _eval_node(ast.parse(expression, mode="eval").body)
            return ToolResult(ok=True, message="Calculation succeeded", data={"result": result})
        except Exception as exc:  # pragma: no cover - starter scaffold
            return ToolResult(ok=False, message=f"Calculation failed: {exc}")


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return float(_OPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        operand = _eval_node(node.operand)
        return float(_OPS[type(node.op)](operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")
