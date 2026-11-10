from pathlib import Path
import ast
import sys

root = Path(__file__).resolve().parent.parent
required = (
    "requirements.txt",
    "render_app.py",
    "render.yaml",
    "assaylens/__init__.py",
    "assaylens/studio_api.py",
    "web/index.html",
    "web/app.js",
    "web/style.css",
    "web/assaylens-logo-original.png",
    "examples/synthetic_multi_target_dti.csv",
    "scripts/benchmark.py",
)
missing = [item for item in required if not (root / item).is_file()]
if missing:
    print("ERROR: Required files missing from the repository:")
    for name in missing:
        print("  -", name)
    print("Upload the CONTENTS of the extracted project folder to GitHub's repository root.")
    sys.exit(1)
for name in ("render_app.py", "assaylens/studio_api.py"):
    try:
        ast.parse((root / name).read_text(encoding="utf-8"), filename=name)
    except SyntaxError as exc:
        sys.exit(f"ERROR: Python syntax invalid in {name}: {exc}")
print("AssayLens deployment files verified.")
