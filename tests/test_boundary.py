import ast
from pathlib import Path
from typing import Set

_FORBIDDEN_MODULES = {"fastapi", "starlette", "flask", "uvicorn", "django", "aiohttp"}
_AI_DIR = Path(__file__).resolve().parent.parent / "ai"


def _imported_top_level_modules(py_file: Path) -> Set[str]:
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    modules: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    return modules


def test_ai_has_no_web_framework_imports() -> None:
    offenders = {}
    for py_file in _AI_DIR.rglob("*.py"):
        if "vendor" in py_file.parts:
            continue  # ini kode BEATs orang lain, bukan punya kita
        found = _imported_top_level_modules(py_file) & _FORBIDDEN_MODULES
        if found:
            offenders[str(py_file.relative_to(_AI_DIR.parent))] = found
    assert not offenders, f"ai/ mengimpor framework web (dilarang): {offenders}"
