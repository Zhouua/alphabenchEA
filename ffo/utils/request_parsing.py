"""Lightweight request parsing shared by the FFO HTTP routes."""


def normalize_factors_from_expression_field(data: dict):
    """Normalize supported request shapes into named factor expressions."""
    expr_field = data.get("expression", None)
    if expr_field is None:
        expr_field = data.get("expr", None)

    if expr_field is None:
        return None, ("Missing 'expression'", "EMPTY_EXPR")

    if isinstance(expr_field, str):
        expr = expr_field.strip()
        if not expr:
            return None, ("Missing 'expression'", "EMPTY_EXPR")
        return [{"name": "", "expression": expr}], None

    if isinstance(expr_field, dict):
        if not expr_field:
            return None, ("Missing 'expression'", "EMPTY_EXPR")
        factors = []
        for name, expr in expr_field.items():
            if not isinstance(expr, str):
                return None, (
                    "Invalid 'expression' dict value (must be string)",
                    "BAD_EXPR_FORMAT",
                )
            expr = expr.strip()
            if not expr:
                return None, (f"Empty expression for name='{name}'", "EMPTY_EXPR")
            factors.append({"name": str(name), "expression": expr})
        return factors, None

    if isinstance(expr_field, list):
        if not expr_field:
            return None, ("Missing 'expression'", "EMPTY_EXPR")

        factors = []
        for index, item in enumerate(expr_field):
            if isinstance(item, str):
                expression = item.strip()
                if not expression:
                    return None, (f"Empty expression at index {index}", "EMPTY_EXPR")
                factors.append({"name": "", "expression": expression})
                continue

            if not isinstance(item, dict):
                return None, (
                    f"Invalid item type in expression list at index {index}",
                    "BAD_EXPR_FORMAT",
                )
            if not item:
                return None, (f"Empty dict at index {index}", "BAD_EXPR_FORMAT")

            # Search clients send {name, expression, ...} objects.  This must
            # precede legacy {name: expression} parsing, or a factor name such
            # as BETA_10 is incorrectly evaluated as a Qlib expression.
            if "expression" in item:
                expression = item.get("expression")
                if not isinstance(expression, str):
                    return None, (
                        f"Invalid expression at index {index} (must be string)",
                        "BAD_EXPR_FORMAT",
                    )
                expression = expression.strip()
                if not expression:
                    return None, (f"Empty expression at index {index}", "EMPTY_EXPR")
                factors.append(
                    {"name": str(item.get("name") or ""), "expression": expression}
                )
                continue

            for name, expression in item.items():
                if not isinstance(expression, str):
                    return None, (
                        f"Invalid expression at index {index} (must be string)",
                        "BAD_EXPR_FORMAT",
                    )
                expression = expression.strip()
                if not expression:
                    return None, (
                        f"Empty expression for name='{name}' at index {index}",
                        "EMPTY_EXPR",
                    )
                factors.append({"name": str(name), "expression": expression})

        return factors, None

    return None, (
        "Invalid 'expression' type (must be string, dict, or list)",
        "BAD_EXPR_FORMAT",
    )
