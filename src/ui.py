"""Backward-compatible entrypoint forwarding to src/ui/dashboard.py."""

import runpy
from pathlib import Path

dashboard_file = Path(__file__).parent / "ui" / "dashboard.py"
runpy.run_path(str(dashboard_file), run_name="__main__")
