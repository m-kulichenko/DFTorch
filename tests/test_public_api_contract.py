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
        MDXL,
        MDXLOS,
        Constants,
        ESDriver,
        ESDriverBatch,
        MDXLBatch,
        Structure,
        StructureBatch,
    )


# The full public surface, requirement REG-03.
#
# Captured from `src/dftorch/__init__.py`'s `__all__` during phase 05, and
# written out here as a LITERAL rather than read back from the module.  Reading
# it back would make the comparison a tautology: `__all__ == __all__` holds no
# matter what the list contains.  A literal is a second, independent source, the
# same reasoning plan 05-03 applied to the SKF metadata oracle.
#
# Editing this list is a deliberate act.  Adding a name here requires a matching
# export in `src/dftorch/__init__.py`; removing one is an API break and, during
# a cleanup phase, is exactly what REG-03 forbids.
PINNED_PUBLIC_API = sorted(
    [
        "Constants",
        "ESDriver",
        "ESDriverBatch",
        "GBSA",
        "GBSABatch",
        "GeoOpt",
        "MDXL",
        "MDXLBatch",
        "MDXLOS",
        "SKGraphNet",
        "SimpleDftD3",
        "Structure",
        "StructureBatch",
        "ThirdOrder",
        "ThirdOrderBatch",
        "create_dftd3",
        "create_gbsa",
        "create_thirdorder",
        "get_coulomb_stress",
        "get_coulomb_stress_kspace",
        "get_coulomb_stress_real",
        "get_electronic_stress_analytical",
        "get_total_stress_analytical",
        "load_ml_sk_model",
    ]
)


def test_public_api_exports_every_declared_name():
    """Every name `dftorch.__all__` declares is actually reachable on `dftorch`.

    `test_public_api_contract` above names 8 of the 24 exports.  Those 8 are the
    ones users are most likely to import and are worth stating separately, but a
    subset cannot notice a dropped export: removing `get_coulomb_stress_real`
    from `__init__.py` would leave that test, and the whole suite, green.  This
    is `05-VALIDATION.md` Sampling Risk item 6.

    `__all__` declaring a name that the module does not define is its own bug,
    independent of the pinned list below: `from dftorch import *` would raise
    `AttributeError` on it, and tooling that trusts `__all__` would advertise an
    import that does not work.
    """
    dftorch = importlib.import_module("dftorch")

    missing = sorted(name for name in dftorch.__all__ if not hasattr(dftorch, name))
    assert not missing, (
        f"dftorch.__all__ declares {missing}, but the module does not define "
        "them. `from dftorch import *` would fail on these names."
    )


def test_public_api_all_is_pinned():
    """`dftorch.__all__` equals the 24-name literal above, name by name (REG-03).

    Both directions fail, and each reports separately, because they mean
    different things:

    * a REMOVAL is an API break.  REG-03 requires that f-orbital work not force
      public API changes on existing simple-format users, so a name disappearing
      during this cleanup phase is the regression the requirement names.
    * an ADDITION is a deliberate API extension.  It is not a defect, but it
      must not happen silently either -- the new name becomes something this
      project is committed to supporting.

    Reporting them together as "two sorted lists differ" would leave a reader
    diffing 24 strings by eye to find out which of the two occurred.
    """
    dftorch = importlib.import_module("dftorch")
    actual = sorted(dftorch.__all__)

    removed = sorted(set(PINNED_PUBLIC_API) - set(actual))
    added = sorted(set(actual) - set(PINNED_PUBLIC_API))

    assert not removed, (
        f"public export(s) REMOVED from dftorch.__all__: {removed}. This is an "
        "API break for existing users and is what REG-03 forbids. If the "
        "removal is intended, it needs a decision recorded first, then this "
        "literal updated."
    )
    assert not added, (
        f"public export(s) ADDED to dftorch.__all__: {added}. Adding to the "
        "public surface is a deliberate commitment, not a side effect. If it is "
        "intended, add the name to PINNED_PUBLIC_API in this file too."
    )
    # Guards against a duplicated entry, which neither set comparison can see.
    assert actual == PINNED_PUBLIC_API
    assert len(actual) == 24, f"expected 24 public names, found {len(actual)}"
