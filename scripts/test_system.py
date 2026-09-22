from __future__ import annotations

import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


MODULES = [
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
]


def test_imports() -> bool:
    print("\n[TEST] Package imports")

    failures = []

    for module in MODULES:
        try:
            importlib.import_module(module)
            print(f"[PASS] {module}")

        except Exception as exc:
            print(f"[FAIL] {module}: {exc}")
            failures.append(module)

    return not failures


def test_directories() -> bool:
    print("\n[TEST] Core directories")

    directories = [
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
    ]

    failed = False

    for directory in directories:
        if (ROOT / directory).exists():
            print(f"[PASS] {directory}")
        else:
            print(f"[FAIL] {directory}")
            failed = True

    return not failed


def main() -> int:
    print("=" * 70)
    print("RENIX SYSTEM TEST")
    print("=" * 70)

    results = [
        test_imports(),
        test_directories(),
    ]

    print("\n" + "=" * 70)

    if all(results):
        print("ALL SYSTEM TESTS PASSED")
        return 0

    print("SYSTEM TESTS FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())


