from __future__ import annotations

import logging

import torch

from . import ewald_torch as torch_impl
from .util import (
    CONV_FACTOR,
    calculate_alpha_and_num_grids,
    calculate_num_kvecs_ch_indep,
    calculate_num_kvecs_dynamic,
    determine_alpha,
    determine_alpha_ch_indep,
)

# Initialize the backend variable
is_triton_available = False
if torch.cuda.is_available():
    try:
        # Try importing Triton implementation
        from . import ewald_triton as impl

        is_triton_available = True
        logging.info("(Ewald) Using Triton backend.")
    except ImportError as e:
        logging.warning(
            f"(Ewald) Triton implementation not available: {e}. Falling back to PyTorch."
        )
        from . import ewald_torch as impl

        logging.info("(Ewald) Using PyTorch backend.")
else:
    logging.info("(Ewald) GPU is not available, using Torch backend")
    from . import ewald_torch as impl


# Define the available functions based on the active implementation
ewald_energy = impl.ewald_energy
ewald_energy_torch = torch_impl.ewald_energy
ewald_real = impl.ewald_real
ewald_real_screening = impl.ewald_real_screening
ewald_real_screening_stress = torch_impl.ewald_real_screening_stress
ewald_real_torch = torch_impl.ewald_real
ewald_kspace_part1 = impl.ewald_kspace_part1
ewald_kspace_part2 = impl.ewald_kspace_part2
ewald_self_energy = torch_impl.ewald_self_energy


def construct_kspace(
    cell: torch.Tensor,
    kcounts: list[int],
    cutoff: float,
    alpha: float,
) -> torch.Tensor:
    """Construct reciprocal-space vectors for the active Ewald backend.

    Parameters
    ----------
    cell : torch.Tensor
        Real-space cell matrix.
    kcounts : list[int]
        Number of reciprocal vectors along each lattice direction.
    cutoff : float
        Real-space cutoff radius.
    alpha : float
        Ewald screening parameter.

    Returns
    -------
    torch.Tensor
        Reciprocal-space vectors formatted for the active backend.
    """
    # Only transpose when the Triton backend is active on a GPU device.
    transpose_kvec = is_triton_available and (cell.device.type != "cpu")
    return torch_impl.construct_kspace(
        cell, kcounts, cutoff, alpha, transpose_kvec=transpose_kvec
    )


def is_triton_active() -> bool:
    """Return whether the Triton Ewald backend is active."""
    return is_triton_available


# at the end because of the circular dependency
from .PME_torch import (  # noqa: E402
    calculate_PME_energy,
    calculate_PME_ewald,
    calculate_PME_kspace_stress,
    init_PME_data,
    map_charges_to_grid,
)
