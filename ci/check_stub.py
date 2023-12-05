import ast
from pathlib import Path

import arbor

with open(Path(__file__).resolve().parent / "../python/__init__.pyi", "r") as stubfile:
    stub = ast.parse(stubfile.read())
typed = {'env'}

for node in stub.body:
    if isinstance(node, ast.AnnAssign):
        if isinstance(node.target, ast.Name):
            typed.add(node.target.id)
    elif isinstance(node, ast.ClassDef) or isinstance(node, ast.FunctionDef):
        typed.add(node.name)

missing = set(attr for attr in dir(arbor) if not attr.startswith("_")) - typed

if missing:
    for attr in missing:
        print(f"Missing typedef in `__init__.pyi` for `{attr}`!")
    exit(1)