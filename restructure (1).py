"""Moves a flat (phone-uploaded) fxsig repo into its real package layout. Safe to re-run."""
import re, shutil, sys
from pathlib import Path

MAP = {
    "costs.py": "fxsig/costs.py",
    "quality.py": "fxsig/data/quality.py",
    "bundle.py": "fxsig/data/bundle.py",
    "ingest_oanda.py": "fxsig/data/ingest_oanda.py",
    "calendar.py": "fxsig/data/calendar.py",
    "technical.py": "fxsig/features/technical.py",
    "regime.py": "fxsig/features/regime.py",
    "as_of.py": "fxsig/features/as_of.py",
    "costs.yaml": "config/costs.yaml",
    "conftest.py": "tests/conftest.py",
    "test_quality.py": "tests/test_quality.py",
    "test_leakage.py": "tests/test_leakage.py",
    "test_features.py": "tests/test_features.py",
    "test_calendar.py": "tests/test_calendar.py",
    "test_bundle_costs_ingest.py": "tests/test_bundle_costs_ingest.py",
}
PYPROJECT = '''[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "fxsig"
version = "0.2.0"
requires-python = ">=3.11"
dependencies = ["pandas>=2.2", "numpy", "pyarrow", "pyyaml", "httpx"]

[project.optional-dependencies]
dev = ["pytest"]

[tool.setuptools]
packages = ["fxsig", "fxsig.data", "fxsig.features"]

[tool.pytest.ini_options]
testpaths = ["tests"]
'''

root = Path(".")
missing = []
# phone uploads can create "name (1).py" duplicates: drop them (or adopt one if the original is absent)
for dup in sorted(root.glob("* (*).py")):
    base = root / re.sub(r" \(\d+\)", "", dup.name)
    if base.exists():
        dup.unlink(); print(f"removed duplicate {dup.name}")
    else:
        dup.rename(base); print(f"renamed {dup.name} -> {base.name}")
for src, dst in MAP.items():
    s, d = root / src, root / dst
    if d.exists() and not s.exists():
        continue                      # already in place
    if not s.exists():
        missing.append(src)
        continue
    d.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(s), str(d))
    print(f"moved {src} -> {dst}")

for pkg in ("fxsig", "fxsig/data", "fxsig/features", "tests"):
    Path(pkg).mkdir(parents=True, exist_ok=True)
    (Path(pkg) / "__init__.py").touch()
for stray in ("__init__.py", "__init___1.py"):   # phone-upload leftovers in the root
    if (root / stray).exists():
        (root / stray).unlink()
        print(f"removed stray {stray}")

Path("pyproject.toml").write_text(PYPROJECT)
if missing:
    print("MISSING FILES (upload these to the repo root, then re-run):", ", ".join(missing))
    sys.exit(1)
print("layout OK")
