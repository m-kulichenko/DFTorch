import os

# Disable TorchDynamo/Inductor compilation in tests (keeps tests deterministic and avoids C++ toolchain)
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

import pytest
import pathlib


@pytest.mark.parametrize("device", ["cpu"])
def test_nearestneighborlist_small_xyz(device):
    """
    Test neighbor list construction for a small molecule.
    """

    root = pathlib.Path(__file__).resolve().parents[1]  # DFTorch/

    xyz_path = root / "tests" / "ch4.xyz"
    skf_dir = root / "tests" / "data_skf_mio-1-1"

    assert xyz_path.is_file(), f"Missing required test geometry: {xyz_path}"
    assert skf_dir.is_dir(), f"Missing required SKF directory: {skf_dir}"

    import torch

    torch.set_default_dtype(torch.float64)

    from dftorch.Constants import Constants
    from dftorch._nearestneighborlist import vectorized_nearestneighborlist

    # Minimal coordinates for 3 atoms in a triangle
    Rx = torch.tensor([0.0, 1.0, 0.0])
    Ry = torch.tensor([0.0, 0.0, 1.0])
    Rz = torch.tensor([0.0, 0.0, 0.0])
    TYPE = torch.tensor([6, 6, 6], dtype=torch.int32)  # Carbon atoms
    LBox = torch.tensor([10.0, 10.0, 10.0])
    Rcut = 4.0
    N = 3
    const = Constants(
        str(xyz_path), str(skf_dir) + "/", magnetic_hubbard_ldep=False
    ).to(device)

    result = vectorized_nearestneighborlist(
        TYPE,
        Rx,
        Ry,
        Rz,
        LBox,
        Rcut,
        N,
        const,
        upper_tri_only=True,
        remove_self_neigh=True,
        min_image_only=True,
        verbose=False,
    )

    # Check that result is a tuple and contains expected tensors
    assert isinstance(result, tuple)
    # Example: check that at least one neighbor pair is found
    neighbor_pairs = result[0]
    assert torch.is_tensor(neighbor_pairs)
    assert neighbor_pairs.shape[0] > 0
