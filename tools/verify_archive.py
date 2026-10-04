"""Validate the public notebook archive without running research code."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
SCRIPTS = ROOT / "scripts"
WINDOWS_PATH = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/][^\n\r\"'`]+")
EXPECTED_NOTEBOOKS = 18
EXPECTED_SCRIPTS = 5


def source_text(value: object) -> str:
    return "".join(value) if isinstance(value, list) else str(value or "")


def notebook_checks(path: Path) -> tuple[bool, list[str], str]:
    errors: list[str] = []
    metadata = ""
    try:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
    except Exception as exc:  # validation errors are reported per file
        return False, [f"nbformat: {exc}"], ""

    kernelspec = notebook.metadata.get("kernelspec", {})
    language_info = notebook.metadata.get("language_info", {})
    metadata = (
        f"kernel={kernelspec.get('display_name') or kernelspec.get('name') or 'unknown'}, "
        f"python={language_info.get('version') or 'unknown'}"
    )

    cells = notebook.cells
    if not cells:
        errors.append("first cell is missing")
    else:
        first = cells[0]
        if first.cell_type != "markdown" or not source_text(first.source).strip():
            errors.append("first cell must contain a non-empty markdown description")

    for index, cell in enumerate(cells, 1):
        text = source_text(cell.get("source", ""))
        if cell.cell_type == "code":
            if cell.get("outputs", []) != []:
                errors.append(f"cell {index}: outputs is not empty")
            if cell.get("execution_count") is not None:
                errors.append(f"cell {index}: execution_count is not null")
            try:
                ast.parse(text, filename=f"{path}:cell{index}")
            except SyntaxError as exc:
                errors.append(f"cell {index}: Python syntax error at line {exc.lineno}: {exc.msg}")
        for line_number, line in enumerate(text.splitlines(), 1):
            if WINDOWS_PATH.search(line):
                errors.append(f"cell {index}, line {line_number}: Windows absolute path found")

    return not errors, errors, metadata


def main() -> int:
    failures = 0
    notebook_count = 0
    if NOTEBOOKS.exists():
        notebook_paths = sorted(NOTEBOOKS.rglob("*.ipynb"))
    else:
        notebook_paths = []
        print("FAIL notebooks/: directory is missing")
        failures += 1
    if len(notebook_paths) != EXPECTED_NOTEBOOKS:
        print(f"FAIL notebooks/: expected {EXPECTED_NOTEBOOKS} notebook(s), found {len(notebook_paths)}")
        failures += 1

    for path in notebook_paths:
        notebook_count += 1
        ok, errors, metadata = notebook_checks(path)
        status = "PASS" if ok else "FAIL"
        print(f"{status} {path.relative_to(ROOT)} ({metadata})")
        for error in errors:
            print(f"  - {error}")
        failures += not ok

    if SCRIPTS.exists():
        script_paths = sorted(SCRIPTS.rglob("*.py"))
    else:
        script_paths = []
        print("FAIL scripts/: directory is missing")
        failures += 1
    if len(script_paths) != EXPECTED_SCRIPTS:
        print(f"FAIL scripts/: expected {EXPECTED_SCRIPTS} Python script(s), found {len(script_paths)}")
        failures += 1
    for path in script_paths:
        try:
            source = path.read_text(encoding="utf-8")
            ast.parse(source, filename=str(path))
            if WINDOWS_PATH.search(source):
                print(f"FAIL {path.relative_to(ROOT)} (absolute Windows path found)")
                failures += 1
                continue
            print(f"PASS {path.relative_to(ROOT)} (Python syntax)")
        except (OSError, SyntaxError) as exc:
            print(f"FAIL {path.relative_to(ROOT)} (Python syntax): {exc}")
            failures += 1

    print(f"Summary: {notebook_count} notebook(s), {len(script_paths)} script(s), {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
