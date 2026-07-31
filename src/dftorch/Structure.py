from __future__ import annotations

from typing import Any

import torch

from ._cell import normalize_cell, normalize_cell_batch, wrap_positions
from ._io import read_pdb, read_xyz


SHELL_DIMS = (1, 3, 5, 7)
SHELL_LOCAL_STARTS = (0, 1, 4, 9)
SHELL_TYPE_IDS = (1, 2, 3, 4)  # 1=s, 2=p, 3=d, 4=f

# Local AO layout used throughout the f-orbital implementation.
# The f labels follow the cubic-harmonic ordering used for f-electron
# Slater-Koster tables:
#   fx3        = x(5*x^2 - 3*r^2)
#   fy3        = y(5*y^2 - 3*r^2)
#   fz3        = z(5*z^2 - 3*r^2)
#   fx_y2_z2   = x(y^2 - z^2)
#   fy_z2_x2   = y(z^2 - x^2)
#   fz_x2_y2   = z(x^2 - y^2)
#   fxyz       = xyz
AO_LABEL_TEMPLATE = ( #An element has a set of basis functions. This maps basis functions to what they actually mean
    "s",
    "px",
    "py",
    "pz",
    "dxy",
    "dyz",
    "dzx",
    "dx2_y2",
    "dz2",
    "fx3",
    "fy3",
    "fz3",
    "fx_y2_z2",
    "fy_z2_x2",
    "fz_x2_y2",
    "fxyz",
)

AO_SHELL_TEMPLATE = ( #Maps basis functions to s,p,d,f
    1,
    2,
    2,
    2,
    3,
    3,
    3,
    3,
    3,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
)


def _as_batched_species_and_coordinates( #format for batched mode
    species: torch.Tensor | Any,
    coordinates: torch.Tensor | Any,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return species as ``(B,N)`` and coordinates as ``(B,N,3)`` tensors."""
    if isinstance(species, torch.Tensor):
        species_t = species.to(device=device, dtype=torch.int64)
    else:
        species_t = torch.as_tensor(species, device=device, dtype=torch.int64)

    if isinstance(coordinates, torch.Tensor):
        coordinates_t = coordinates.to(device=device, dtype=torch.get_default_dtype())
    else:
        coordinates_t = torch.as_tensor(
            coordinates, device=device, dtype=torch.get_default_dtype()
        )

    if species_t.dim() == 1:
        species_t = species_t.unsqueeze(0)
    if coordinates_t.dim() == 2:
        coordinates_t = coordinates_t.unsqueeze(0)

    return species_t, coordinates_t


def _ao_mask_from_shell_present(shell_present: torch.Tensor) -> torch.Tensor:
    """Expand ``(...,4)`` shell flags into a ``(...,16)`` AO-position mask."""
    mask = torch.zeros(
        (*shell_present.shape[:-1], len(AO_LABEL_TEMPLATE)),
        dtype=torch.bool,
        device=shell_present.device,
    )
    mask[..., 0] = shell_present[..., 0]
    mask[..., 1:4] = shell_present[..., 1].unsqueeze(-1)
    mask[..., 4:9] = shell_present[..., 2].unsqueeze(-1)
    mask[..., 9:16] = shell_present[..., 3].unsqueeze(-1)
    return mask


def _shell_local_start(shell_present: torch.Tensor) -> torch.Tensor:
    """Return local AO starts for s/p/d/f shells, using -1 for absent shells."""
    starts = torch.tensor(
        SHELL_LOCAL_STARTS,
        dtype=torch.int64,
        device=shell_present.device,
    )
    starts = starts.expand(*shell_present.shape[:-1], 4)
    return torch.where(shell_present, starts, torch.full_like(starts, -1))


def _shell_local_end(shell_present: torch.Tensor) -> torch.Tensor:
    """Return local AO ends for s/p/d/f shells, using -1 for absent shells."""
    starts = _shell_local_start(shell_present)
    dims = torch.tensor(SHELL_DIMS, dtype=torch.int64, device=shell_present.device)
    ends = starts + dims.expand_as(starts) - 1
    return torch.where(shell_present, ends, torch.full_like(ends, -1))


def _global_shell_start(
    shell_present: torch.Tensor,
    shell_local_start: torch.Tensor,
    atom_ao_start: torch.Tensor,
) -> torch.Tensor:
    """Convert local shell starts into structure-local or batch-global AO starts."""
    starts = shell_local_start + atom_ao_start.unsqueeze(-1)
    return torch.where(shell_present, starts, torch.full_like(starts, -1))


def _global_shell_end(
    shell_present: torch.Tensor,
    shell_local_end: torch.Tensor,
    atom_ao_start: torch.Tensor,
) -> torch.Tensor:
    """Convert local shell ends into structure-local or batch-global AO ends."""
    ends = shell_local_end + atom_ao_start.unsqueeze(-1)
    return torch.where(shell_present, ends, torch.full_like(ends, -1))


def _flatten_ao_labels(shell_present: torch.Tensor) -> list[str]:
    """Return flattened AO labels for a single structure in atom-major order."""
    mask = _ao_mask_from_shell_present(shell_present).detach().cpu()
    labels: list[str] = []
    for atom_mask in mask:
        labels.extend(label for label, present in zip(AO_LABEL_TEMPLATE, atom_mask) if present)
    return labels


def _ao_shell_types_from_mask(ao_mask: torch.Tensor) -> torch.Tensor:
    """Return flattened shell type IDs for all present AO positions."""
    template = torch.tensor(
        AO_SHELL_TEMPLATE, dtype=torch.int64, device=ao_mask.device
    ).expand(*ao_mask.shape[:-1], len(AO_SHELL_TEMPLATE))
    return template[ao_mask]


def _atomic_density_matrix_from_shells(
    H_INDEX_START: torch.Tensor,
    HDIM: int,
    TYPE: torch.Tensor,
    const: Any,
    shell_present: torch.Tensor,
    shell_ao_start: torch.Tensor,
) -> torch.Tensor:
    """Build the initial atom-density vector from explicit shell metadata."""
    D_atomic = torch.zeros(
        HDIM, device=H_INDEX_START.device, dtype=torch.get_default_dtype()
    )
    shell_occ = (const.n_s, const.n_p, const.n_d, const.n_f)

    for shell_idx, shell_dim in enumerate(SHELL_DIMS):
        present = shell_present[:, shell_idx]
        if not present.any():
            continue
        starts = shell_ao_start[present, shell_idx]
        occ = shell_occ[shell_idx][TYPE[present]] / float(shell_dim)
        for local_idx in range(shell_dim):
            D_atomic[starts + local_idx] = occ

    return D_atomic


def _atomic_density_matrix_batch_from_shells(
    batch_size: int,
    H_INDEX_START: torch.Tensor,
    HDIM: int,
    TYPE: torch.Tensor,
    const: Any,
    shell_present: torch.Tensor,
    shell_ao_start: torch.Tensor,
) -> torch.Tensor:
    """Batched initial atom-density matrix from explicit shell metadata."""
    D_atomic = torch.zeros(
        batch_size, HDIM, device=H_INDEX_START.device, dtype=torch.get_default_dtype()
    )
    shell_occ = (const.n_s, const.n_p, const.n_d, const.n_f)

    for shell_idx, shell_dim in enumerate(SHELL_DIMS):
        present = shell_present[:, :, shell_idx]
        if not present.any():
            continue
        batch_idx, atom_idx = present.nonzero(as_tuple=True)
        starts = shell_ao_start[batch_idx, atom_idx, shell_idx]
        occ = shell_occ[shell_idx][TYPE[batch_idx, atom_idx]] / float(shell_dim)
        for local_idx in range(shell_dim):
            D_atomic[batch_idx, starts + local_idx] = occ

    return D_atomic


class Structure(torch.nn.Module):
    """
    Container for a DFTB structure holding atom types, coordinates, box, and
    derived per-atom/basis indexing information.

    The local AO order is nested shell order:

    ``s, px, py, pz, dxy, dyz, dzx, dx2_y2, dz2,``
    ``fx3, fy3, fz3, fx_y2_z2, fy_z2_x2, fz_x2_y2, fxyz``.
    """

    def __init__(
        self,
        dftorch_params: dict,
        const: Any,
        device: str = "cpu",
        species: torch.Tensor | None = None,
        coordinates: torch.Tensor | None = None,
        ignore_spin: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Initialise a single-structure DFTB container."""
        super().__init__(*args, **kwargs)

        self.req_grad_xyz = dftorch_params.get("GRAD_XYZ", False)
        self.req_grad_cell = dftorch_params.get("GRAD_CELL", False)
        cell = dftorch_params.get("CELL", None)
        if species is None or coordinates is None:
            if dftorch_params.get("FILENAME", None) is not None and dftorch_params[
                "FILENAME"
            ].lower().endswith(".pdb"):
                species, coordinates, pdb_cell = read_pdb(
                    [dftorch_params["FILENAME"]], sort=False
                )
                if dftorch_params.get("CELL", None) is None and pdb_cell is not None:
                    cell = pdb_cell
                else:
                    cell = dftorch_params.get("CELL", None)
            else:
                cell = dftorch_params.get("CELL", None)
                species, coordinates = read_xyz(
                    [dftorch_params["FILENAME"]], sort=False
                )

        species, coordinates = _as_batched_species_and_coordinates(
            species, coordinates, device
        )
        self.TYPE = species[0]

        self.RX = (
            coordinates[0, :, 0].clone().detach().requires_grad_(self.req_grad_xyz)
        )
        self.RY = (
            coordinates[0, :, 1].clone().detach().requires_grad_(self.req_grad_xyz)
        )
        self.RZ = (
            coordinates[0, :, 2].clone().detach().requires_grad_(self.req_grad_xyz)
        )

        self.cell = (
            None
            if cell is None
            else torch.as_tensor(cell, device=device, dtype=torch.get_default_dtype())
        )
        self.cell = normalize_cell(
            self.cell, device=device, dtype=torch.get_default_dtype()
        )
        self.cell_inv = None if self.cell is None else torch.linalg.inv(self.cell)
        self.lattice_vecs = self.cell

        if self.cell is not None:
            with torch.no_grad():
                R = torch.stack((self.RX, self.RY, self.RZ), dim=-1)
                R_wrapped = wrap_positions(R, self.cell, self.cell_inv)

            self.RX = (
                R_wrapped[..., 0].clone().detach().requires_grad_(self.req_grad_xyz)
            )
            self.RY = (
                R_wrapped[..., 1].clone().detach().requires_grad_(self.req_grad_xyz)
            )
            self.RZ = (
                R_wrapped[..., 2].clone().detach().requires_grad_(self.req_grad_xyz)
            )

            if self.req_grad_cell:
                self._RX_leaf = self.RX
                self._RY_leaf = self.RY
                self._RZ_leaf = self.RZ

                cell_ref = self.cell.detach()
                positions = torch.stack((self.RX, self.RY, self.RZ), dim=-1)
                frac_coords = positions @ torch.linalg.inv(cell_ref)

                self.cell = cell.detach().clone().requires_grad_(True)
                self.cell_inv = torch.linalg.inv(self.cell.detach())

                cart_coords = frac_coords @ self.cell
                self.RX, self.RY, self.RZ = cart_coords.unbind(dim=-1)

        self.coordinates = torch.stack((self.RX, self.RY, self.RZ))

        self.Nats = len(self.TYPE)
        self.const = const
        self.charge = dftorch_params.get("CHARGE", 0)
        self.spin_pol = dftorch_params.get("SPIN_POL", 0)
        self.Te = dftorch_params["T_ELECTRONIC"]
        if dftorch_params.get("ELECTRIC_FIELD", None) is None:
            self.e_field = torch.zeros(
                3, dtype=torch.get_default_dtype(), device=device
            )
        else:
            self.e_field = torch.as_tensor(
                dftorch_params["ELECTRIC_FIELD"],
                device=device,
                dtype=torch.get_default_dtype(),
            )

        self.device = device
        self.n_orbitals_per_atom = const.n_orb[self.TYPE]
        self.H_INDEX_START = torch.zeros(self.Nats, dtype=torch.int64, device=device)
        self.H_INDEX_START[1:] = torch.cumsum(self.n_orbitals_per_atom, dim=0)[:-1]
        self.H_INDEX_END = self.H_INDEX_START + self.n_orbitals_per_atom - 1

        self.Mnuc = const.mass[self.TYPE]
        self.Znuc = const.tore[self.TYPE]
        if dftorch_params.get("UNRESTRICTED", False):
            tot_el = torch.tensor(
                [int(const.tore[self.TYPE].sum() - self.charge)], device=device
            )

            nocc_a = tot_el / 2 + self.spin_pol / 2
            nocc_b = tot_el / 2 - self.spin_pol / 2
            if (nocc_a % 1 != 0).any() or (nocc_b % 1 != 0).any():
                raise ValueError("Invalid charge/spin_pol combination!")

            self.Nocc = torch.tensor([int(nocc_a), int(nocc_b)], device=device)

        else:
            tot_el = const.tore[self.TYPE].sum() - self.charge
            if ((tot_el % 2) == 1).any() and not ignore_spin:
                raise ValueError(
                    "Closed shell systems require even number of electrons"
                )

            self.Nocc = int(tot_el / 2)
        self.Hubbard_U = const.U[self.TYPE]

        self.shell_present = const.shell_present[self.TYPE].to(dtype=torch.bool)
        self.has_s = self.shell_present[:, 0]
        self.has_p = self.shell_present[:, 1]
        self.has_d = self.shell_present[:, 2]
        self.has_f = self.shell_present[:, 3]

        self.shell_local_start = _shell_local_start(self.shell_present)
        self.shell_local_end = _shell_local_end(self.shell_present)
        self.shell_ao_start = _global_shell_start(
            self.shell_present, self.shell_local_start, self.H_INDEX_START
        )
        self.shell_ao_end = _global_shell_end(
            self.shell_present, self.shell_local_end, self.H_INDEX_START
        )

        # Shell on-site energies per atom.
        EsA = const.Es[self.TYPE]
        EpA = const.Ep[self.TYPE]
        EdA = const.Ed[self.TYPE]
        EfA = const.Ef[self.TYPE]

        # Per-atom AO template in the fixed 16-position local order.
        template = torch.stack(
            (
                EsA,
                EpA,
                EpA,
                EpA,
                EdA,
                EdA,
                EdA,
                EdA,
                EdA,
                EfA,
                EfA,
                EfA,
                EfA,
                EfA,
                EfA,
                EfA,
            ),
            dim=1,
        )
        ao_mask = _ao_mask_from_shell_present(self.shell_present)

        self.diagonal = template[ao_mask]
        self.HDIM = self.diagonal.shape[-1]
        self.ao_shell_types = _ao_shell_types_from_mask(ao_mask)
        self.ao_labels = _flatten_ao_labels(self.shell_present)

        UsA = const.U[self.TYPE]
        UpA = const.Up[self.TYPE]
        UdA = const.Ud[self.TYPE]
        UfA = const.Uf[self.TYPE]
        ns = const.n_s[self.TYPE]
        np = const.n_p[self.TYPE]
        nd = const.n_d[self.TYPE]
        nf = const.n_f[self.TYPE]

        template_U = torch.stack((UsA, UpA, UdA, UfA), dim=1)
        template_ang = torch.stack(
            (
                torch.ones_like(UsA, dtype=torch.int64),
                torch.ones_like(UsA, dtype=torch.int64) + 1,
                torch.ones_like(UsA, dtype=torch.int64) + 2,
                torch.ones_like(UsA, dtype=torch.int64) + 3,
            ),
            dim=1,
        )
        template_el_per_shell = torch.stack((ns, np, nd, nf), dim=1)

        shell_mask = self.shell_present
        self.Hubbard_U_sr = template_U[shell_mask]
        self.shell_types = template_ang[shell_mask]
        self.el_per_shell = template_el_per_shell[shell_mask]
        self.n_shells_per_atom = self.shell_present.sum(dim=1).to(torch.int64)
        self.H_INDEX_START_U = torch.zeros(self.Nats, dtype=torch.int64, device=device)
        self.H_INDEX_START_U[1:] = torch.cumsum(self.n_shells_per_atom, dim=0)[:-1]
        self.H_INDEX_END_U = self.H_INDEX_START_U + self.n_shells_per_atom - 1

        self.D0 = _atomic_density_matrix_from_shells(
            self.H_INDEX_START,
            self.HDIM,
            self.TYPE,
            const,
            self.shell_present,
            self.shell_ao_start,
        )
        self.D0 = 0.5 * self.D0
        self.q_spin_sr = None
        self.q = None

        if const.dftb3:
            self.dU_dq = const.dU_dq[self.TYPE]
        else:
            self.dU_dq = None


class StructureBatch(torch.nn.Module):
    """Batch container for multiple structures with explicit s/p/d/f shell metadata."""

    def __init__(
        self,
        dftorch_params: dict,
        const: Any,
        device: str = "cpu",
        ignore_spin: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Initialise a batched structure container."""
        super().__init__(*args, **kwargs)

        self.batch_size = len(dftorch_params["FILENAME"])
        if (
            dftorch_params["FILENAME"]
            and isinstance(dftorch_params["FILENAME"][0], str)
            and dftorch_params["FILENAME"][0].lower().endswith(".pdb")
        ):
            species, coordinates, _ = read_pdb(dftorch_params["FILENAME"], sort=False)
        else:
            species, coordinates = read_xyz(dftorch_params["FILENAME"], sort=False)

        self.TYPE, coordinates = _as_batched_species_and_coordinates(
            species, coordinates, device
        )
        self.req_grad_xyz = dftorch_params.get("GRAD_XYZ", False)
        self.RX = coordinates[:, :, 0].clone().detach().requires_grad_(self.req_grad_xyz)
        self.RY = coordinates[:, :, 1].clone().detach().requires_grad_(self.req_grad_xyz)
        self.RZ = coordinates[:, :, 2].clone().detach().requires_grad_(self.req_grad_xyz)

        cell = dftorch_params.get("CELL", None)
        self.cell = (
            None
            if cell is None
            else torch.as_tensor(cell, device=device, dtype=torch.get_default_dtype())
        )
        self.cell = normalize_cell_batch(
            self.cell,
            B=self.batch_size,
            device=device,
            dtype=torch.get_default_dtype(),
        )
        self.cell_inv = None if self.cell is None else torch.linalg.inv(self.cell)
        self.lattice_vecs = self.cell

        if self.cell is not None:
            R = torch.stack((self.RX, self.RY, self.RZ), dim=-1)
            R = wrap_positions(R, self.cell, self.cell_inv)
            self.RX, self.RY, self.RZ = R.unbind(dim=-1)

        self.coordinates = torch.stack((self.RX, self.RY, self.RZ))

        self.Nats = self.TYPE.shape[-1]
        self.const = const
        self.charge = dftorch_params.get("CHARGE", 0)
        self.Te = dftorch_params.get("T_ELECTRONIC", 1000.0)
        if dftorch_params.get("ELECTRIC_FIELD", None) is None:
            self.e_field = torch.zeros(
                3, dtype=torch.get_default_dtype(), device=device
            )
        else:
            self.e_field = torch.tensor(
                dftorch_params["ELECTRIC_FIELD"],
                dtype=torch.get_default_dtype(),
                device=device,
            )

        self.device = device
        self.n_orbitals_per_atom = const.n_orb[self.TYPE]
        self.H_INDEX_START = torch.zeros(
            self.batch_size, self.Nats, dtype=torch.int64, device=device
        )
        self.H_INDEX_START[:, 1:] = torch.cumsum(self.n_orbitals_per_atom, dim=1)[
            :, :-1
        ]
        self.H_INDEX_END = self.H_INDEX_START + self.n_orbitals_per_atom - 1

        self.Mnuc = const.mass[self.TYPE]
        self.Znuc = const.tore[self.TYPE]

        tot_el = const.tore[self.TYPE].sum(dim=1) - self.charge
        if ((tot_el % 2) == 1).any() and not ignore_spin:
            raise ValueError("Closed shell systems require even number of electrons")
        self.Nocc = (tot_el / 2).to(int)

        self.Hubbard_U = const.U[self.TYPE]

        self.shell_present = const.shell_present[self.TYPE].to(dtype=torch.bool)
        self.has_s = self.shell_present[:, :, 0]
        self.has_p = self.shell_present[:, :, 1]
        self.has_d = self.shell_present[:, :, 2]
        self.has_f = self.shell_present[:, :, 3]

        self.shell_local_start = _shell_local_start(self.shell_present)
        self.shell_local_end = _shell_local_end(self.shell_present)
        self.shell_ao_start = _global_shell_start(
            self.shell_present, self.shell_local_start, self.H_INDEX_START
        )
        self.shell_ao_end = _global_shell_end(
            self.shell_present, self.shell_local_end, self.H_INDEX_START
        )

        EsA = const.Es[self.TYPE]
        EpA = const.Ep[self.TYPE]
        EdA = const.Ed[self.TYPE]
        EfA = const.Ef[self.TYPE]

        template = torch.stack(
            (
                EsA,
                EpA,
                EpA,
                EpA,
                EdA,
                EdA,
                EdA,
                EdA,
                EdA,
                EfA,
                EfA,
                EfA,
                EfA,
                EfA,
                EfA,
                EfA,
            ),
            dim=2,
        )
        ao_mask = _ao_mask_from_shell_present(self.shell_present)

        self.diagonal_flat = template[ao_mask]
        self.HDIM_struct = self.n_orbitals_per_atom.sum(dim=1)
        self.HDIM_total = int(self.diagonal_flat.shape[0])
        max_HDIM = int(self.HDIM_struct.max().item())
        self.diagonal = torch.zeros(
            self.batch_size, max_HDIM, dtype=template.dtype, device=self.device
        )

        cursor = 0
        self.ao_labels: list[list[str]] = []
        self.ao_shell_types = torch.zeros(
            self.batch_size, max_HDIM, dtype=torch.int64, device=self.device
        )
        ao_shell_types_flat = _ao_shell_types_from_mask(ao_mask)
        for batch_idx in range(self.batch_size):
            hdim_b = int(self.HDIM_struct[batch_idx].item())
            self.diagonal[batch_idx, :hdim_b] = self.diagonal_flat[
                cursor : cursor + hdim_b
            ]
            self.ao_shell_types[batch_idx, :hdim_b] = ao_shell_types_flat[
                cursor : cursor + hdim_b
            ]
            self.ao_labels.append(_flatten_ao_labels(self.shell_present[batch_idx]))
            cursor += hdim_b
        self.ao_shell_types_flat = ao_shell_types_flat

        struct_offsets = torch.cumsum(self.HDIM_struct, dim=0) - self.HDIM_struct
        self.H_INDEX_START_GLOBAL = self.H_INDEX_START + struct_offsets.unsqueeze(-1)
        self.H_INDEX_END_GLOBAL = self.H_INDEX_END + struct_offsets.unsqueeze(-1)
        self.shell_ao_start_global = _global_shell_start(
            self.shell_present, self.shell_local_start, self.H_INDEX_START_GLOBAL
        )
        self.shell_ao_end_global = _global_shell_end(
            self.shell_present, self.shell_local_end, self.H_INDEX_START_GLOBAL
        )

        UsA = const.U[self.TYPE]
        UpA = const.Up[self.TYPE]
        UdA = const.Ud[self.TYPE]
        UfA = const.Uf[self.TYPE]
        ns = const.n_s[self.TYPE]
        np = const.n_p[self.TYPE]
        nd = const.n_d[self.TYPE]
        nf = const.n_f[self.TYPE]

        template_shell = torch.stack((UsA, UpA, UdA, UfA), dim=2)
        template_ang = torch.stack(
            (
                torch.ones_like(UsA, dtype=torch.int64),
                torch.ones_like(UsA, dtype=torch.int64) + 1,
                torch.ones_like(UsA, dtype=torch.int64) + 2,
                torch.ones_like(UsA, dtype=torch.int64) + 3,
            ),
            dim=2,
        )
        template_el_per_shell = torch.stack((ns, np, nd, nf), dim=2)
        shell_mask = self.shell_present
        self.Hubbard_U_sr = template_shell[shell_mask]
        self.shell_types = template_ang[shell_mask]
        self.el_per_shell = template_el_per_shell[shell_mask]
        self.n_shells_per_atom = self.shell_present.sum(dim=2).to(torch.int64)
        self.H_INDEX_START_U = torch.zeros_like(
            self.n_shells_per_atom, dtype=torch.int64, device=device
        )
        self.H_INDEX_START_U[:, 1:] = torch.cumsum(self.n_shells_per_atom, dim=1)[
            :, :-1
        ]
        self.H_INDEX_END_U = self.H_INDEX_START_U + self.n_shells_per_atom - 1
        shells_per_struct = self.n_shells_per_atom.sum(dim=1)
        shell_offsets = torch.cumsum(shells_per_struct, dim=0) - shells_per_struct
        self.H_INDEX_START_U_GLOBAL = self.H_INDEX_START_U + shell_offsets.unsqueeze(-1)
        self.H_INDEX_END_U_GLOBAL = self.H_INDEX_END_U + shell_offsets.unsqueeze(-1)

        self.HDIM = self.diagonal.shape[-1]
        self.D0 = _atomic_density_matrix_batch_from_shells(
            self.batch_size,
            self.H_INDEX_START,
            self.HDIM,
            self.TYPE,
            const,
            self.shell_present,
            self.shell_ao_start,
        )
        self.D0 = 0.5 * self.D0

        if const.dftb3:
            self.dU_dq = const.dU_dq[self.TYPE]
        else:
            self.dU_dq = None

        self.q = None


__all__ = ["Structure", "StructureBatch"]