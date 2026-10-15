"""Load named notebook definitions only; never execute training at import."""

from contextlib import contextmanager
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn


def repository_root():
    return Path(__file__).resolve().parents[2]


@contextmanager
def legacy_import_path():
    """Support both the original and archival curvature module locations."""
    root = repository_root()
    original = sys.path[:]
    sys.path[:0] = [str(root / "archive" / "legacy"), str(root)]
    try:
        yield
    finally:
        sys.path[:] = original


def definitions(notebook="EX_1.ipynb", names=("PINN",)):
    root = repository_root()
    path = root / "archive" / "notebooks" / notebook
    if not path.exists():
        path = root / notebook
    cells = json.loads(path.read_text())["cells"]
    namespace = {"torch": torch, "nn": nn, "np": np}
    with legacy_import_path():
        for name in names:
            marker = (f"class {name}(", f"def {name}(")
            matches = [
                "".join(c.get("source", ()))
                for c in cells
                if c["cell_type"] == "code"
                and any(m in "".join(c.get("source", ())) for m in marker)
            ]
            if len(matches) != 1:
                raise ValueError(f"Expected one definition cell for {name}, got {len(matches)}")
            exec(compile(matches[0], str(path), "exec"), namespace)
    return namespace
