import sys
from pathlib import Path

# The scout scripts are run by path and import each other as top-level modules.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
