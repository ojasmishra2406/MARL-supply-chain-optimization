"""
Alias for tools/reproducibility/check_reproducibility.py
"""
import sys
from pathlib import Path

WORKSPACE_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from tools.reproducibility.check_reproducibility import *

if __name__ == "__main__":
    main()
