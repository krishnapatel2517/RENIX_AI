from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements.txt"


def main() -> int:
    print("=" * 70)
    print("RENIX DEPENDENCY INSTALLER")
    print("=" * 70)

    if not REQUIREMENTS.exists():
        print(f"[ERROR] requirements.txt not found: {REQUIREMENTS}")
        return 1

    python = sys.executable

    print(f"Python executable: {python}")
    print("Upgrading pip...")

    subprocess.run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
        ],
        check=False,
    )

    print("Installing RENIX dependencies...")

    result = subprocess.run(
        [
            python,
            "-m",
            "pip",
            "install",
            "-r",
            str(REQUIREMENTS),
        ],
        cwd=ROOT,
        check=False,
    )

    if result.returncode != 0:
        print("[ERROR] Dependency installation failed.")
        return result.returncode

    print("[OK] Dependencies installed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


