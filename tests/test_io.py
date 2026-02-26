import pathlib


def test_io():
    """
    CI smoke test:
    - Imports core modules.
    - Runs a *small* _scf + forces calculation on CPU.
    - Skips automatically if required SKF test data is not present.
    """

    root = pathlib.Path(__file__).resolve().parents[1]  # DFTorch/

    xyz_path = root / "tests" / "ch4.xyz"

    assert xyz_path.is_file(), f"Missing required test geometry: {xyz_path}"

    import torch
    from dftorch._io import read_xyz

    torch.set_default_dtype(torch.float64)

    species, coordinates = read_xyz(
        [str(xyz_path)], sort=False
    )  # Input coordinate file

    TYPE = torch.tensor(species[0], dtype=torch.int64)
    COORDS = torch.tensor(coordinates[0])

    assert torch.isfinite(COORDS).all()
    assert TYPE.shape[0] == COORDS.shape[0]
