import sys
import os
from pathlib import Path

# Ensure the in-repo package is importable without installation
root = Path(__file__).resolve().parents[1]
pkg = root / "python"
if str(pkg) not in sys.path:
    sys.path.insert(0, str(pkg))

# Ensure joblib can create temp files without warnings in sandboxed environments
tmp_root = Path(__file__).resolve().parents[1] / ".tmp" / "joblib"
tmp_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(tmp_root))
