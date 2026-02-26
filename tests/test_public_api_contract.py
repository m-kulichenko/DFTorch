import importlib


def test_public_api_contract():
    dftorch = importlib.import_module("dftorch")

    expected = {
        "Constants",
        "Structure",
        "StructureBatch",
        "ESDriver",
        "ESDriverBatch",
        "MDXL",
        "MDXLBatch",
        "MDXLOS",
    }

    missing = sorted([name for name in expected if not hasattr(dftorch, name)])
    assert not missing, f"Missing public symbols: {missing}"


def test_public_api_imports():
    # This should remain the supported import path.
    from dftorch import (  # noqa: F401
        Constants,
        ESDriver,
        ESDriverBatch,
        MDXL,
        MDXLBatch,
        MDXLOS,
        Structure,
        StructureBatch,
    )
