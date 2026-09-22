import sys
import platform
import os


def main():
    print("=" * 60)
    print("RENIX ENVIRONMENT CHECK")
    print("=" * 60)

    print(f"Python      : {sys.version}")
    print(f"Platform    : {platform.platform()}")
    print(f"Machine     : {platform.machine()}")
    print(f"Processor   : {platform.processor()}")
    print(f"Working dir : {os.getcwd()}")

    print("\nChecking core packages...")

    packages = [
        "dotenv",
        "yaml",
        "pydantic",
        "requests",
        "httpx",
        "aiohttp",
        "psutil",
        "numpy",
    ]

    failed = []

    for package in packages:
        try:
            __import__(package)
            print(f"[ OK ] {package}")
        except Exception as exc:
            print(f"[FAIL] {package} -> {exc}")
            failed.append(package)

    print("\n" + "=" * 60)

    if failed:
        print("RENIX environment is NOT ready.")
        print("Missing/failed packages:")
        for package in failed:
            print(f"  - {package}")
        sys.exit(1)

    print("RENIX environment is READY.")
    print("=" * 60)


if __name__ == "__main__":
    main()
    


