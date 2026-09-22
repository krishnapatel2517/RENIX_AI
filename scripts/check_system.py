from __future__ import annotations

import platform
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_DIRECTORIES = [
    "core",
    "ai",
    "memory",
    "voice",
    "vision",
    "gestures",
    "holographic_ui",
    "computer",
    "files",
    "browser",
    "research",
    "coding",
    "automation",
    "security",
    "devices",
    "robotics",
    "system_monitor",
    "education",
    "sports",
    "media",
    "personal",
    "analytics",
    "notifications",
    "integrations",
    "database",
    "data",
    "assets",
    "config",
    "logs",
    "tests",
    "scripts",
]


def check_python() -> bool:
    version = sys.version_info

    print(
        f"[PYTHON] "
        f"{version.major}.{version.minor}.{version.micro}"
    )

    if version < (3, 10):
        print("[FAIL] Python 3.10 or newer is required.")
        return False

    print("[PASS] Python version")
    return True


def check_os() -> bool:
    print(f"[OS] {platform.system()} {platform.release()}")
    print(f"[ARCH] {platform.machine()}")
    return True


def check_disk() -> bool:
    total, used, free = shutil.disk_usage(ROOT)

    free_gb = free / (1024 ** 3)

    print(f"[DISK] Free: {free_gb:.2f} GB")

    if free_gb < 2:
        print("[WARNING] Less than 2 GB free.")
        return False

    print("[PASS] Disk space")
    return True


def check_directories() -> bool:
    failed = False

    for directory in REQUIRED_DIRECTORIES:
        path = ROOT / directory

        if path.exists():
            print(f"[PASS] {directory}")
        else:
            print(f"[FAIL] {directory}")
            failed = True

    return not failed


def check_configuration() -> bool:
    env_file = ROOT / ".env"
    config_file = ROOT / "config.yaml"

    success = True

    if env_file.exists():
        print("[PASS] .env")
    else:
        print("[WARNING] .env missing")
        success = False

    if config_file.exists():
        print("[PASS] config.yaml")
    else:
        print("[WARNING] config.yaml missing")
        success = False

    return success


def main() -> int:
    print("=" * 70)
    print("RENIX SYSTEM CHECK")
    print("=" * 70)

    results = [
        check_python(),
        check_os(),
        check_disk(),
        check_directories(),
        check_configuration(),
    ]

    print("\n" + "=" * 70)

    if all(results):
        print("SYSTEM CHECK: PASS")
        return 0

    print("SYSTEM CHECK: WARNINGS / FAILURES")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())


