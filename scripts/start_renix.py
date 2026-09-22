from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN_FILE = ROOT / "main.py"


def main() -> int:
    print("=" * 70)
    print("                         RENIX")
    print("                     AI ASSISTANT")
    print("=" * 70)

    if not MAIN_FILE.exists():
        print(f"[ERROR] main.py not found: {MAIN_FILE}")
        return 1

    print(f"[ROOT] {ROOT}")
    print(f"[PYTHON] {sys.executable}")
    print("\nStarting RENIX...\n")

    try:
        process = subprocess.run(
            [
                sys.executable,
                str(MAIN_FILE),
            ],
            cwd=ROOT,
            check=False,
        )

        return process.returncode

    except KeyboardInterrupt:
        print("\nRENIX stopped.")
        return 0

    except Exception as exc:
        print(f"[ERROR] Could not start RENIX: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


