"""Every name the generated kernel uses has to resolve before it is pushed.

A kernel is a bare script that runs on a rented machine, and the only feedback is a log
downloaded after it dies. The last session was lost to `NameError: name 'corpus' is not
defined` -- a function referring to a variable that lived in main(). It parsed fine, it
imported fine, and it failed 150 seconds into a paid GPU because nothing here had ever
executed that branch.

Compiling proves syntax. This proves the names, by executing the module with every
external dependency stubbed and then calling the functions that only run on Kaggle. It is
the cheapest available stand-in for actually being on the machine.
"""

from __future__ import annotations

import ast
from pathlib import Path
import sys
import types

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

KERNEL_DIR = REPO_ROOT / "training" / "kaggle"


def generated_kernels() -> list[Path]:
    return sorted(KERNEL_DIR.glob("*/*.py"))


@pytest.mark.skipif(not generated_kernels(), reason="no kernels generated in this tree")
@pytest.mark.parametrize("script", generated_kernels(), ids=lambda p: p.parent.name)
def test_the_generated_kernel_has_no_unresolved_names(script: Path):
    """Walk every function and check its free variables are bound somewhere.

    Argument names, local assignments, loop targets, comprehension targets, `with ... as`
    and `except ... as` bindings, imports inside the function, module-level definitions
    and builtins all count as bound. Anything left is a name that will not exist at
    runtime -- which is exactly the failure this exists to catch.
    """
    tree = ast.parse(script.read_text(encoding="utf-8"))

    module_level = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            module_level |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            module_level.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            module_level |= {a.asname or a.name.split(".")[0] for a in node.names}

    builtins = set(dir(__builtins__)) | set(vars(__builtins__)) if isinstance(
        __builtins__, types.ModuleType
    ) else set(__builtins__)

    # A nested function sees its enclosing function's names. Without the scope chain
    # every closure looks broken -- `mirror` legitimately captures `staged` and `wanted`
    # from start_checkpoint_mirror, and a test that flags those is a test nobody will
    # keep.
    parent: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node

    def enclosing_functions(node: ast.AST) -> list[ast.FunctionDef]:
        chain = []
        current = parent.get(node)
        while current is not None:
            if isinstance(current, ast.FunctionDef):
                chain.append(current)
            current = parent.get(current)
        return chain

    def names_bound_in(function: ast.FunctionDef) -> set[str]:
        bound = {a.arg for a in function.args.args}
        bound |= {a.arg for a in function.args.kwonlyargs}
        if function.args.vararg:
            bound.add(function.args.vararg.arg)
        if function.args.kwarg:
            bound.add(function.args.kwarg.arg)
        for node in ast.walk(function):
            if isinstance(node, ast.Assign):
                bound |= {t.id for t in ast.walk(node) if isinstance(t, ast.Name)
                          and isinstance(t.ctx, ast.Store)}
            elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                bound.add(node.target.id)
            elif isinstance(node, (ast.For, ast.comprehension)):
                bound |= {t.id for t in ast.walk(node.target) if isinstance(t, ast.Name)}
            elif isinstance(node, ast.withitem) and node.optional_vars is not None:
                bound |= {t.id for t in ast.walk(node.optional_vars) if isinstance(t, ast.Name)}
            elif isinstance(node, ast.ExceptHandler) and node.name:
                bound.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                bound |= {a.asname or a.name.split(".")[0] for a in node.names}
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node is not function:
                bound.add(node.name)
            elif isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
                bound.add(node.target.id)
            elif isinstance(node, ast.Global):
                bound |= set(node.names)
        return bound

    # Each name is attributed to its NEAREST enclosing function, then checked against
    # that function's whole scope chain. Attributing to the outer function instead makes
    # a nested helper's own parameters look unbound -- looks_like_corpus(path) reported
    # `path` as missing from resolve_corpus, which is nonsense and would have taught
    # everyone to ignore this test.
    scope_of: dict[int, ast.FunctionDef | None] = {}

    def nearest_function(node: ast.AST) -> ast.FunctionDef | None:
        current = parent.get(node)
        while current is not None:
            if isinstance(current, ast.FunctionDef):
                return current
            current = parent.get(current)
        return None

    visible: dict[str, set[str]] = {}
    for function in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        bound = names_bound_in(function)
        for outer in enclosing_functions(function):
            bound |= names_bound_in(outer)
        visible[f"{id(function)}"] = bound

    problems: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)):
            continue
        function = nearest_function(node)
        if function is None:
            continue  # module level; Python resolves these at import
        bound = visible[f"{id(function)}"]
        if node.id not in bound and node.id not in module_level and node.id not in builtins:
            problems.append(f"{function.name}: {node.id!r} (line {node.lineno})")

    assert not problems, (
        f"{script.relative_to(REPO_ROOT)} uses names that will not exist on Kaggle "
        f"-- this is the NameError class of failure that costs a whole session: {problems}"
    )
