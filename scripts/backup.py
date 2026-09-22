from __future__ import annotations

from datetime import datetime
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
BACKUP_DIRECTORY = ROOT / "data" / "backups"


EXCLUDED_DIRECTORIES = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
}


def should_include(path: Path) -> bool:
    relative = path.relative_to(ROOT)

    if any(part in EXCLUDED_DIRECTORIES for part in relative.parts):
        return False

    if path.suffix.lower() == ".pyc":
        return False

    if (
        "data" in relative.parts
        and "backups" in relative.parts
        and path.suffix.lower() == ".zip"
    ):
        return False

    return True


def create_backup() -> Path:
    BACKUP_DIRECTORY.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    backup_file = (
        BACKUP_DIRECTORY /
        f"renix_backup_{timestamp}.zip"
    )

    print(f"[INFO] Creating backup: {backup_file}")

    with zipfile.ZipFile(
        backup_file,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:

        for path in ROOT.rglob("*"):
            if not path.is_file():
                continue

            if not should_include(path):
                continue

            archive.write(
                path,
                arcname=path.relative_to(ROOT),
            )

    return backup_file


def main() -> int:
    print("=" * 70)
    print("RENIX BACKUP")
    print("=" * 70)

    try:
        backup_file = create_backup()

        size_mb = backup_file.stat().st_size / (1024 ** 2)

        print("\n[OK] Backup completed.")
        print(f"[FILE] {backup_file}")
        print(f"[SIZE] {size_mb:.2f} MB")

        return 0

    except Exception as exc:
        print(f"[ERROR] Backup failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


