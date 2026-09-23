"""Run the project's pytest tiers with consistent child-process encoding.

Usage: uv run python scripts/test.py [quick|slow|full] [test selectors ...]
"""

import argparse
import os
import subprocess
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tier", choices=("quick", "slow", "full"), nargs="?", default="quick")
    parser.add_argument("selectors", nargs="*", help="Optional test files or pytest node IDs")
    args = parser.parse_args()

    command = [sys.executable, "-m", "pytest", "-q", "--durations=10", *args.selectors]
    if args.tier != "full":
        command.extend(("-x", "-m", "slow" if args.tier == "slow" else "not slow"))

    # Several CLI tests decode captured output as UTF-8. Windows otherwise
    # lets the child choose its local code page, yielding false failures.
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"
    print(f"[tests] {args.tier}: pytest {' '.join(command[3:])}", flush=True)
    started = time.perf_counter()
    result = subprocess.run(command, env=environment, check=False)
    print(f"[tests] exit={result.returncode}, elapsed={time.perf_counter() - started:.1f}s")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
