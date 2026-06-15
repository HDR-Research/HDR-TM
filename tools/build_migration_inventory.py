from __future__ import annotations

import ast
import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATLAB_ROOT = ROOT / "HDR_Toolbox-master" / "source_code"
PYTHON_ROOT = ROOT / "hdrtmo"
OUTPUT = ROOT / "docs" / "migration_inventory.csv"


def python_symbols() -> set[str]:
    symbols: set[str] = set()
    for path in PYTHON_ROOT.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                symbols.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        symbols.add(target.id)
    return symbols


def matlab_functions(path: Path) -> list[str]:
    content = path.read_text(encoding="utf-8", errors="ignore")
    names = re.findall(
        r"(?m)^function\s+(?:\[[^\]]+\]\s*=\s*|[^=\n]+\s*=\s*)?([A-Za-z]\w*)\s*\(",
        content,
    )
    return names or [path.stem]


def main() -> None:
    symbols = python_symbols()
    rows = []
    for path in sorted(MATLAB_ROOT.rglob("*.m")):
        functions = matlab_functions(path)
        implemented = [name for name in functions if name in symbols]
        status = "implemented" if len(implemented) == len(functions) else "pending"
        if implemented and status == "pending":
            status = "partial"
        rows.append(
            {
                "matlab_path": str(path.relative_to(MATLAB_ROOT)),
                "matlab_functions": ";".join(functions),
                "status": status,
                "implemented_symbols": ";".join(implemented),
            }
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    counts = {status: sum(row["status"] == status for row in rows) for status in ("implemented", "partial", "pending")}
    print(f"Wrote {OUTPUT}: {len(rows)} MATLAB files, {counts}")


if __name__ == "__main__":
    main()

