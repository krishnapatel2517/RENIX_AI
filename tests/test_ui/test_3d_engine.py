def test_3d_engine_import():
    import importlib
    module = importlib.import_module("holographic_ui.3d_engine")
    assert hasattr(module, "Engine3D")
    from holographic_ui import get_3d_engine
    assert get_3d_engine().__name__ == "Engine3D"
