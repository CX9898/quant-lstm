"""Project JSON Schema validator with lexical JSON-integer type strictness."""

from typing import Any

import jsonschema


def _is_strict_integer(_checker: Any, instance: Any) -> bool:
    return isinstance(instance, int) and not isinstance(instance, bool)


StrictDraft202012Validator = jsonschema.validators.extend(
    jsonschema.Draft202012Validator,
    type_checker=jsonschema.Draft202012Validator.TYPE_CHECKER.redefine(
        "integer", _is_strict_integer
    ),
)
