"""The packages' dependency rules (README, docs/ARCHITECTURE.md), checked by reading every import under ``src``.

* ``research_core`` imports no asset-class package;
* the asset-class packages do not import one another;
* ``equity_research.research`` holds helpers for the notebooks only: nothing in ``src`` or ``tests`` imports it.
"""

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
ASSET_CLASSES = ("equity_research", "fixed_income_research")
NOTEBOOK_ONLY = "equity_research.research"


def _imports(path: Path) -> set[str]:
    """Every module ``path`` imports, as absolute dotted names (``from a import b`` counts ``a`` and ``a.b``)."""
    package = path.relative_to(SRC).with_suffix("").parts[:-1] if SRC in path.parents else ()
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:  # relative: resolved against the file's own package
                base = ".".join(package[: len(package) - node.level + 1])
                module = f"{base}.{module}" if module else base
            found.add(module)
            found.update(f"{module}.{alias.name}" for alias in node.names)
    return found


def _offenders(package: str, forbidden: tuple[str, ...]) -> list[str]:
    """``file imports module`` for every import in ``package`` of one of the ``forbidden`` packages."""
    return sorted(
        f"{path.relative_to(SRC).as_posix()} imports {name}"
        for path in (SRC / package).rglob("*.py")
        for name in _imports(path)
        if name.split(".")[0] in forbidden
    )


def test_the_core_imports_no_asset_class_package() -> None:
    assert _offenders("research_core", ASSET_CLASSES) == []


@pytest.mark.parametrize("package", ASSET_CLASSES)
def test_asset_class_packages_do_not_import_one_another(package: str) -> None:
    others = tuple(other for other in ASSET_CLASSES if other != package)
    assert _offenders(package, others) == []


def test_every_package_is_there_to_check() -> None:
    assert all((SRC / package / "__init__.py").exists() for package in ("research_core", *ASSET_CLASSES))


def test_the_notebook_helpers_are_imported_by_notebooks_only() -> None:
    helpers = SRC / "equity_research" / "research"
    files = [p for p in [*SRC.rglob("*.py"), *(REPO / "tests").rglob("*.py")] if helpers not in p.parents]
    users = sorted(
        path.relative_to(REPO).as_posix()
        for path in files
        if any(name == NOTEBOOK_ONLY or name.startswith(NOTEBOOK_ONLY + ".") for name in _imports(path))
    )
    assert users == []
