# ruff: noqa
from __future__ import annotations

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %                   SCF-TB - PROXY APPLICATION                      %
# %                   A.M.N. Niklasson, M. Kulichenko. T1, LANL       %
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % Total Energy Function:                                            %
# % E = 2Tr[H0(D-D0)] + (1/2)*sum_i U_i q_i^2 +                       %
# %      + (1/2)sum_{i,j (i!=j)} q_i C_{ij} q_j - Efield*dipole       %
# % dipole = sum_i R_{i} q_i                                          %
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import torch
import warnings
import logging
import os

# to disable torchdynamo completely. Faster for smaller systems and single-point calculations.
os.environ["TORCHDYNAMO_DISABLE"] = "1"  # hard-disable capture


# from dftorch.io import write_XYZ_trajectory, write_xyz_from_xyz
from dftorch import (
    Constants,
    Structure,
)

### Configure torch and torch.compile ###
# Silence warnings and module logs
warnings.filterwarnings("ignore")
os.environ["TORCH_LOGS"] = ""  # disable PT2 logging
os.environ["TORCHINDUCTOR_VERBOSE"] = "0"
os.environ["TORCHDYNAMO_VERBOSE"] = "0"
logging.getLogger("torch.fx").setLevel(logging.CRITICAL)
logging.getLogger("torch.fx.experimental.symbolic_shapes").setLevel(logging.CRITICAL)
logging.getLogger("torch.fx.experimental.recording").setLevel(logging.CRITICAL)
# Enable dynamic shape capture for dynamo
torch._dynamo.config.capture_dynamic_output_shape_ops = True
# default data type
torch.set_default_dtype(torch.float64)

torch.cuda.empty_cache()


from torch_dftd.nn.dftd3_module import DFTD3Module
from torch_dftd.functions.edge_extraction import calc_edge_index


# D3-BJ parameters for DFTB2/mio-1-1
# Ref: Brandenburg et al. J. Chem. Theory Comput. 2013
# Literature: s6=1.0, s8=0.5883, a1=0.5719, a2=3.6017
D3_MIO = {"s6": 1.0, "s18": 0.5883, "rs6": 0.5719, "rs18": 3.6017, "alp": 14.0}

# D3-BJ parameters for DFTB3/3ob-3-1
# Ref: Gaus et al. J. Chem. Theory Comput. 2014
# Literature: s6=1.0, s8=0.4727, a1=0.5467, a2=4.4955
D3_3OB = {"s6": 1.0, "s18": 0.4727, "rs6": 0.5467, "rs18": 4.4955, "alp": 14.0}


class D3Dispersion:
    """D3-BJ dispersion correction for DFTorch Structure objects.

    Bypasses the ASE layer and calls DFTD3Module directly with
    custom DFTB-fitted D3 parameters.

    Parameters
    ----------
    params:
        D3-BJ parameter dict. Use D3_MIO or D3_3OB.
    cutoff:
        Real-space cutoff in Angstrom.
    cnthr:
        Coordination number cutoff in Angstrom.
    device:
        Should match structure.device.
    """

    def __init__(
        self,
        params: dict = D3_MIO,
        cutoff: float = 50.0,
        cnthr: float = 21.0,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.cutoff = cutoff
        self.bidirectional = True

        self.dftd_module = DFTD3Module(
            params=params,
            cutoff=cutoff,
            cnthr=cnthr,
            abc=False,
            dtype=torch.float64,
            bidirectional=self.bidirectional,
        ).to(device)

    def _get_edge_index(
        self,
        pos: torch.Tensor,  # (N, 3)
    ):
        pbc = torch.zeros(3, dtype=torch.bool, device=self.device)
        edge_index, S = calc_edge_index(
            pos,
            cell=None,
            pbc=pbc,
            cutoff=self.cutoff,
            bidirectional=self.bidirectional,
        )
        shift_pos = S  # no cell → shift_pos == S (Angstrom)
        return edge_index, shift_pos, pbc

    def calc(
        self,
        structure,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos = torch.stack([structure.RX, structure.RY, structure.RZ], dim=1).to(
            device=self.device, dtype=torch.float64
        )  # (N, 3)
        Z = structure.Znuc.to(device=self.device)  # (N,)

        edge_index, shift_pos, pbc = self._get_edge_index(pos.detach())

        # Note: Z before pos; returns List[Dict]; forces are numpy on CPU
        results_list = self.dftd_module.calc_energy_and_forces(
            Z=Z,
            pos=pos,
            cell=None,
            pbc=pbc,
            edge_index=edge_index,
            shift_pos=shift_pos,
            damping="bj",
        )

        results = results_list[0]
        e_disp = torch.tensor(
            results["energy"], dtype=torch.float64, device=self.device
        )
        f_disp_N3 = torch.tensor(
            results["forces"], dtype=torch.float64, device=self.device
        )  # (N, 3)

        f_disp = f_disp_N3.T.contiguous()  # (3, N) — matches f_tot
        return e_disp, f_disp

    def apply(self, structure) -> None:
        """Add D3 correction in-place to structure.e_tot and structure.f_tot."""
        e_disp, f_disp = self.calc(structure)
        structure.e_tot = structure.e_tot + e_disp
        structure.f_tot = structure.f_tot + f_disp


if __name__ == "__main__":
    dftorch_params = {
        "UNRESTRICTED": True,
        "SHARED_MU": True,  # if True, use shared chemical potential for both spin channels in unrestricted calculations. Otherwise, use separate chemical potentials for each spin channel.
        "BROKEN_SYM": True,  # if True, mix 2 % of lumo in homo at initialization
        "coul_method": "!PME",  # 'FULL' for full coulomb matrix, 'PME' for PME method
        "Coulomb_acc": 5e-5,  # Coulomb accuracy for full coulomb calcs or t_err for PME
        "cutoff": 10.0,  # Coulomb cutoff
        "PME_order": 4,  # Ignored for FULL coulomb method
        "SCF_MAX_ITER": 150,  # Maximum number of SCF iterations
        "SCF_TOL": 1e-6,  # SCF convergence tolerance on density matrix
        "SCF_ALPHA": 0.03,  # Scaled delta function coefficient. Acts as linear mixing coefficient used before Krylov acceleration starts.
        "KRYLOV_MAXRANK": 30,  # Maximum Krylov subspace rank
        "KRYLOV_TOL": 1e-6,  # Krylov subspace convergence tolerance in SCF
        "KRYLOV_TOL_MD": 1e-4,  # Krylov subspace convergence tolerance in MD SCF
        "KRYLOV_START": 90,  # Number of initial SCF iterations before starting Krylov acceleration
    }

    # Initial data, load atoms and coordinates, etc in COORD.dat
    device = "cuda" if torch.cuda.is_available() else "cpu"
    filename = "benzene_adduct_o2.xyz"
    # filename = "O2.xyz"
    LBox = None
    # LBox = torch.tensor([45.0, 45.0, 40.0], device=device) # Simulation box size in Angstroms. Only cubic boxes supported for now.

    # Create constants container. Set path to SKF files.
    const = Constants(
        filename,
        "sk_orig/mio-1-1/mio-1-1/",
        magnetic_hubbard_ldep=False,
    ).to(device)

    # Create structure object. Define total charge and electronic temperature.
    # charge=0 and spin_pol need to be specified for open-shell systems.
    # spin_pol is the number of unpaired electrons, not multiplicity.
    # Shared chemical potential is used for both spin channels, so spin_pol has no effect but needs to be set anyways.
    structure1 = Structure(
        filename,
        LBox,
        const,
        charge=0,
        spin_pol=3,
        os=dftorch_params["UNRESTRICTED"],
        Te=5000.0,
        device=device,
    )

    d3 = D3Dispersion(params=D3_MIO, cutoff=50.0, device=device)

    e_disp, f_disp = d3.calc(structure1)

    print("E_tot with D3:", e_disp.item(), "eV")
    print("F_tot with D3:", f_disp.shape)  # (3, 16)
    print("F_tot with D3:", f_disp)  # (3, 16)
