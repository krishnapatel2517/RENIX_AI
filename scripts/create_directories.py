from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


DIRECTORIES = [
    "data/memory",
    "data/conversations",
    "data/projects",
    "data/tasks",
    "data/embeddings",
    "data/profiles",
    "data/statistics",
    "data/cache",
    "data/logs",
    "data/backups",

    "assets/icons",
    "assets/images",
    "assets/models",
    "assets/textures",
    "assets/animations",
    "assets/sounds",
    "assets/music",
    "assets/fonts",
    "assets/shaders",

    "logs/system",
    "logs/ai",
    "logs/security",
    "logs/automation",
    "logs/errors",
    "logs/audit",

    "tests/test_core",
    "tests/test_ai",
    "tests/test_memory",
    "tests/test_voice",
    "tests/test_vision",
    "tests/test_gestures",
    "tests/test_ui",
    "tests/test_computer",
    "tests/test_files",
    "tests/test_browser",
    "tests/test_coding",
    "tests/test_security",
    "tests/test_devices",
    "tests/test_automation",
    "tests/test_integrations",

    "automation/workflows/custom",

    "devices/mobile",
    "devices/smartwatch",
    "devices/tv",

    "integrations/providers/ai",
    "integrations/providers/weather",
    "integrations/providers/maps",
    "integrations/providers/calendar",
    "integrations/providers/media",
    "integrations/providers/smart_home",

    "holographic_ui/assets/models",
    "holographic_ui/assets/textures",
    "holographic_ui/assets/icons",
    "holographic_ui/assets/sounds",
    "holographic_ui/assets/fonts",
]


def create_directories() -> bool:
    failed = []

    for relative_path in DIRECTORIES:
        path = ROOT / relative_path

        try:
            path.mkdir(parents=True, exist_ok=True)
            print(f"[OK] {relative_path}")

        except OSError as exc:
            print(f"[ERROR] {relative_path}: {exc}")
            failed.append(relative_path)

    if failed:
        print(f"\nFailed directories: {len(failed)}")
        return False

    print(f"\n[OK] Verified {len(DIRECTORIES)} directories.")
    return True


def main() -> int:
    print("=" * 70)
    print("RENIX DIRECTORY INITIALIZER")
    print("=" * 70)

    return 0 if create_directories() else 1


if __name__ == "__main__":
    raise SystemExit(main())


