"""
Syntax checker — py_compile across all products and platform code.

Usage:
    from local_brain.scripts.syntax_checker import check_syntax
    results = check_syntax("/path/to/repo")
"""

import py_compile
from pathlib import Path
from typing import Dict, List


def check_syntax(repo_dir: str) -> Dict:
    """Run py_compile on every .py file, grouped by product."""
    repo = Path(repo_dir)
    errors: Dict[str, List[Dict]] = {}
    total_files = 0
    total_errors = 0

    for py_file in sorted(repo.rglob("*.py")):
        # Skip venvs, caches, node_modules
        parts_str = str(py_file)
        if any(
            skip in parts_str
            for skip in (
                "venv/",
                ".venv/",
                "node_modules/",
                "__pycache__/",
                ".git/",
                ".eggs/",
                "site-packages/",
                "build/",
                "dist/",
            )
        ):
            continue

        total_files += 1
        product = _classify_file(py_file, repo)

        try:
            py_compile.compile(str(py_file), doraise=True)
        except py_compile.PyCompileError as e:
            total_errors += 1
            if product not in errors:
                errors[product] = []
            errors[product].append(
                {
                    "file": str(py_file.relative_to(repo)),
                    "error": str(e).split("\n")[0][:200],
                }
            )

    return {
        "total_files": total_files,
        "total_errors": total_errors,
        "by_product": errors,
    }


def _classify_file(filepath: Path, repo: Path) -> str:
    """Classify a file into a product by longest-prefix match."""
    rel = str(filepath.relative_to(repo))

    if rel.startswith("products/"):
        parts = rel.split("/")
        if len(parts) >= 2:
            return parts[1]  # products/{name}/...
    elif rel.startswith("platform/"):
        return "platform"
    elif rel.startswith("shared/"):
        return "shared"
    elif rel.startswith("codeword/"):
        return "codeword"

    return "root"
