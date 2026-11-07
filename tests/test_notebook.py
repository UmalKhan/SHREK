import ast
import json
from pathlib import Path
import pytest


@pytest.mark.parametrize("name", ["phase2_gpu.ipynb", "phase4_study.ipynb"])
def test_gpu_notebook_has_clean_executable_python_cells(name):
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks" / name).read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            ast.parse("".join(cell["source"]))
