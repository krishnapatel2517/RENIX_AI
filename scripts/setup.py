from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> bool:
    try:
        result = subprocess.run(command, cwd=ROOT, check=False)
        return result.returncode == 0
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return False


def main() -> int:
    print("=" * 70)
    print("RENIX SETUP")
    print("=" * 70)

    print(f"RENIX root: {ROOT}")
    print(f"Python: {sys.version.split()[0]}")

    print("\n[1/4] Creating directories...")
    if not run([sys.executable, str(ROOT / "scripts" / "create_directories.py")]):
        return 1

    print("\n[2/4] Installing dependencies...")
    if not run(
        [sys.executable, str(ROOT / "scripts" / "install_dependencies.py")]
    ):
        print("[WARNING] Dependency installation reported errors.")

    print("\n[3/4] Initializing database...")
    if not run(
        [sys.executable, str(ROOT / "scripts" / "initialize_database.py")]
    ):
        return 1

    print("\n[4/4] Checking system...")
    run([sys.executable, str(ROOT / "scripts" / "check_system.py")])

    print("\n" + "=" * 70)
    print("RENIX SETUP FINISHED")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


