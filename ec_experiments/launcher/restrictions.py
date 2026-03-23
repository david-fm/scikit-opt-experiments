from __future__ import annotations

import ast
from typing import Any


class RestrictionEvaluationError(ValueError):
    """Raised when a declarative restriction expression cannot be evaluated."""


def _eval_node(node: ast.AST, variables: dict[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, variables)

    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.Name):
        if node.id not in variables:
            raise RestrictionEvaluationError(f"Unknown variable in restriction: {node.id}")
        return variables[node.id]

    if isinstance(node, ast.UnaryOp):
        value = _eval_node(node.operand, variables)
        if isinstance(node.op, ast.UAdd):
            return +value
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.Not):
            return not value
        raise RestrictionEvaluationError("Unsupported unary operator in restriction")

    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, variables)
        right = _eval_node(node.right, variables)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.Pow):
            return left ** right
        raise RestrictionEvaluationError("Unsupported binary operator in restriction")

    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            return all(bool(_eval_node(value, variables)) for value in node.values)
        if isinstance(node.op, ast.Or):
            return any(bool(_eval_node(value, variables)) for value in node.values)
        raise RestrictionEvaluationError("Unsupported boolean operator in restriction")

    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, variables)
        for op, comparator in zip(node.ops, node.comparators, strict=False):
            right = _eval_node(comparator, variables)
            if isinstance(op, ast.Lt):
                ok = left < right
            elif isinstance(op, ast.LtE):
                ok = left <= right
            elif isinstance(op, ast.Gt):
                ok = left > right
            elif isinstance(op, ast.GtE):
                ok = left >= right
            elif isinstance(op, ast.Eq):
                ok = left == right
            elif isinstance(op, ast.NotEq):
                ok = left != right
            else:
                raise RestrictionEvaluationError("Unsupported comparison operator in restriction")
            if not ok:
                return False
            left = right
        return True

    raise RestrictionEvaluationError(
        f"Unsupported expression in restriction: {type(node).__name__}"
    )


def evaluate_restriction(expression: str, variables: dict[str, Any]) -> bool:
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise RestrictionEvaluationError(
            f"Invalid syntax in restriction '{expression}': {exc.msg}"
        ) from exc

    result = _eval_node(parsed, variables)
    if not isinstance(result, bool):
        raise RestrictionEvaluationError(
            f"Restriction must evaluate to bool: '{expression}'"
        )
    return result


def validate_restrictions(config: dict[str, Any], restrictions: list[str]) -> tuple[bool, list[str]]:
    violations: list[str] = []
    for expression in restrictions:
        try:
            satisfied = evaluate_restriction(expression, config)
        except RestrictionEvaluationError as exc:
            violations.append(f"{expression} (error: {exc})")
            continue
        if not satisfied:
            violations.append(expression)
    return len(violations) == 0, violations
