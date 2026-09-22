from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


MODEL_LOCATIONS = {
    ROOT / "assets" / "models": [
        "renix_core.gltf",
        "renix_orb.gltf",
        "holographic_ring.gltf",
        "holographic_panel.gltf",
    ],
    ROOT / "holographic_ui" / "assets" / "models": [
        "renix_core.gltf",
    ],
}


def verify_models() -> bool:
    success = True

    print("=" * 70)
    print("RENIX MODEL VERIFICATION")
    print("=" * 70)

    for directory, models in MODEL_LOCATIONS.items():
        directory.mkdir(parents=True, exist_ok=True)

        print(f"\n{directory.relative_to(ROOT)}")

        for model in models:
            path = directory / model

            if path.exists() and path.stat().st_size > 0:
                print(f"[OK] {model}")
            else:
                print(f"[MISSING] {model}")
                success = False

    return success


def main() -> int:
    if verify_models():
        print("\n[OK] All expected local models are available.")
        return 0

    print("\n[WARNING] Some model files are missing.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())


