import ast
import math
import operator as op
import re

from .base import Manifest, Tool

_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
        ast.Pow: op.pow, ast.Mod: op.mod, ast.FloorDiv: op.floordiv, ast.USub: op.neg, ast.UAdd: op.pos}
_FUNCS = {"sqrt": math.sqrt, "round": round, "abs": abs, "min": min, "max": max}


def safe_eval(expr: str) -> float:
    expr = expr.replace("^", "**")
    expr = re.sub(r"(?<=\d),(?=\d{3}(\D|$))", "", expr)  # thousands separators: 1,200 -> 1200

    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            if isinstance(n.op, ast.Pow) and abs(ev(n.right)) > 100:
                raise ValueError("exponent too large")
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS and not n.keywords:
            return _FUNCS[n.func.id](*[ev(a) for a in n.args])
        raise ValueError("unsupported expression")

    return ev(ast.parse(expr, mode="eval").body)


class Calculator(Tool):
    manifest = Manifest(
        name="calculator",
        description="Evaluate an arithmetic expression exactly (+ - * / ** % sqrt round abs min max).",
        args={"expression": "arithmetic expression, e.g. '12.5 * 8 + 3'"},
        when_needed="Questions that require computing totals, differences, percentages, unit costs or quantities.",
        keywords=["calculate", "total", "cost", "percentage", "sum", "price", "quantity", "budget", "compute"],
        strong_keywords=["calculate", "calculation", "compute", "order total", "order totals"],
        provides_facts=False,
    )

    def run(self, args: dict) -> str:
        try:
            return str(safe_eval(str(args.get("expression", ""))))
        except Exception as e:  # noqa: BLE001
            return f"error: {e}"
