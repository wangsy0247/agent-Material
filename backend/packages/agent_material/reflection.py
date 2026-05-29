"""Dynamic module loading utilities."""

from __future__ import annotations

import importlib
from typing import Any


def resolve_variable(path: str) -> Any:
    """Resolve a variable from a module path string.

    Args:
        path: String in the format ``module.submodule:variable_name``.

    Returns:
        The resolved variable.

    Raises:
        ValueError: If the path format is invalid.
        ImportError: If the module cannot be imported.
        AttributeError: If the variable does not exist in the module.
    """
    if ":" not in path:
        raise ValueError(f"Invalid path format: {path!r}. Expected 'module.path:variable_name'.")
    module_path, var_name = path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, var_name)
