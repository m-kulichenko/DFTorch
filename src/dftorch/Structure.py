import torch
from ._io import read_xyz, read_pdb
from ._atomic_density_matrix import atomic_density_matrix, atomic_density_matrix_batch
from ._cell import normalize_cell, normalize_cell_batch, wrap_positions


class Structure(torch.nn.Module):
    """
    Container for a DFTB structure holding atom types, coordinates, box, and
    derived per-atom/basis indexing information.

    Parameters
    ----------
    cell : sequence or torch.Tensor
        Periodic cell specification. May be:
        - shape (3,) for orthorhombic box lengths in Å
        - shape (3,3) for a full triclinic cell matrix in Å
    const : object
        Constants database with fields (n_orb, mass, tore, U, Es, Ep, Ed, Up, Ud,
        n_s, n_p, n_d, max_ang, etc.).
    charge : int, default 0
        Total system charge (affects electron count Nocc).
    device : str, default 'cpu'
        Device label stored for convenience.

    Attributes
    ----------
    TYPE, RX, RY, RZ, cell : as passed
    lattice_vecs : (3,3) torch.Tensor
        Diagonal lattice vectors built from cell.
    Nats : int
        Number of atoms.
    n_orbitals_per_atom : (N,) torch.Tensor
        Basis size per atom.
    H_INDEX_START / H_INDEX_END : (N,) torch.Tensor
        Start/end indices of each atom’s AO block in flattened ordering.
    Mnuc, Znuc : (N,) torch.Tensor
        Atomic masses and nuclear charges.
    Nocc : int
        Spin-summed occupied electron count after charge adjustment.
    Hubbard_U : (N,) torch.Tensor
        Atom-resolved Hubbard U values.
    diagonal : (sum_i n_orb[i],) torch.Tensor
        On-site energies in AO order [s, p..., d...] filtered per atom.
    HDIM : int
        Total number of AOs.
    Hubbard_U_sr : (n_shell_total,) torch.Tensor
        Shell-resolved Hubbard U (s/p/d present per atom).
    shell_types : (n_shell_total,) torch.Tensor
        Angular shell identifiers (1=s,2=p,3=d).
    el_per_shell : (n_shell_total,) torch.Tensor
        Electrons per shell for each atom.
    H_INDEX_START_U / H_INDEX_END_U : (N,) torch.Tensor
        Start/end indices for shell-resolved data.
    """

    def __init__(
        self,
        file,
        cell,
        const,
        charge: int = 0,
        spin_pol: int = 0,
        os: bool = False,
        Te: float = 3000.0,
        e_field: torch.Tensor = None,
        device: str = "cpu",
        req_grad_xyz: bool = False,
        req_grad_cell: bool = False,
        species=None,
        coordinates=None,
        ignore_spin: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.req_grad_xyz = req_grad_xyz
        self.req_grad_cell = req_grad_cell
        if species is None or coordinates is None:
            if file is not None and file.lower().endswith(".pdb"):
                species, coordinates, pdb_cell = read_pdb([file], sort=False)
                if cell is None and pdb_cell is not None:
                    cell = pdb_cell
            else:
                species, coordinates = read_xyz(
                    [file], sort=False
                )  # Input coordinate file

        # self.TYPE = torch.tensor(species[0], dtype=torch.int64, device=device)

        if isinstance(coordinates, torch.Tensor):
            coordinates = coordinates.to(device=device, dtype=torch.get_default_dtype())
        else:
            coordinates = torch.as_tensor(
                coordinates, device=device, dtype=torch.get_default_dtype()
            )

        if isinstance(species, torch.Tensor):
            species = species.to(device=device, dtype=torch.int64)
        else:
            species = torch.as_tensor(species, device=device, dtype=torch.int64)

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

            # Create new leaf tensors at the wrapped positions
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
                # Save leaf references for force extraction
                self._RX_leaf = self.RX
                self._RY_leaf = self.RY
                self._RZ_leaf = self.RZ

                # Detach the old cell, create a fresh leaf for cell gradients
                cell_ref = self.cell.detach()
                positions = torch.stack((self.RX, self.RY, self.RZ), dim=-1)
                frac_coords = positions @ torch.linalg.inv(cell_ref)

                self.cell = cell.detach().clone().requires_grad_(True)
                self.cell_inv = torch.linalg.inv(self.cell.detach())

                cart_coords = frac_coords @ self.cell
                self.RX, self.RY, self.RZ = cart_coords.unbind(dim=-1)

        self.coordinates = torch.stack(
            (self.RX, self.RY, self.RZ),
        )

        self.Nats = len(self.TYPE)
        self.const = const
        self.charge = charge
        self.spin_pol = spin_pol
        self.Te = Te
        if e_field is None:
            self.e_field = torch.zeros(
                3, dtype=torch.get_default_dtype(), device=device
            )
        else:
            self.e_field = e_field

        self.device = device
        self.n_orbitals_per_atom = const.n_orb[self.TYPE]
        self.H_INDEX_START = torch.zeros(self.Nats, dtype=torch.int64, device=device)
        self.H_INDEX_START[1:] = torch.cumsum(self.n_orbitals_per_atom, dim=0)[:-1]
        self.H_INDEX_END = self.H_INDEX_START + self.n_orbitals_per_atom - 1

        self.Mnuc = const.mass[self.TYPE]
        self.Znuc = const.tore[self.TYPE]
        if os:  # open-shell
            tot_el = torch.tensor(
                [int(const.tore[self.TYPE].sum() - charge)], device=device
            )

            nocc_a = tot_el / 2 + self.spin_pol / 2
            nocc_b = tot_el / 2 - self.spin_pol / 2
            if (nocc_a % 1 != 0).any() or (nocc_b % 1 != 0).any():
                raise ValueError("Invalid charge/spin_pol combination!")

            self.Nocc = torch.tensor([int(nocc_a), int(nocc_b)], device=device)

        else:
            tot_el = const.tore[self.TYPE].sum() - charge
            if ((tot_el % 2) == 1).any() and not ignore_spin:
                raise ValueError(
                    "Closed shell systems require even number of electrons"
                )

            self.Nocc = int(tot_el / 2)
        self.Hubbard_U = const.U[self.TYPE]

        # Shell on-site energies per atom (pulled from your dicts)
        EsA = const.Es[self.TYPE]  # (Nr_atoms,)
        EpA = const.Ep[self.TYPE]  # (Nr_atoms,)
        EdA = const.Ed[self.TYPE]  # (Nr_atoms,)

        # Which shells exist for each atom, based on your basis size:
        # 1  -> H-like: s
        # 4  -> main-group sp: s + 3*p
        # 9  -> transition-metal spd: s + 3*p + 5*d
        self.has_p = const.n_orb[self.TYPE] >= 4  # p present for 4 or 9
        self.has_d = const.n_orb[self.TYPE] == 9  # d present only for 9 here
        # (Optional: if you ever use sd-only (6 orbitals), set has_d |= (const.n_orb[self.TYPE] == 6)
        #            and exclude p for that case.)

        # Build a per-atom template in the standard AO order: [s, px, py, pz, dxy, dyz, dzx, dx2-y2, dz2]
        # (All p orbitals get EpA; all d orbitals get EdA.)
        template = torch.stack(
            (
                EsA,  # s
                EpA,
                EpA,
                EpA,  # p triplet
                EdA,
                EdA,
                EdA,
                EdA,
                EdA,  # d quintet
            ),
            dim=1,  # shape: (Nr_atoms, 9)
        )

        # Per-atom mask telling which of the 9 positions are actually present
        mask = torch.zeros_like(template, dtype=torch.bool)  # (Nr_atoms, 9)
        mask[:, 0] = True  # s always present
        mask[:, 1:4] = self.has_p.unsqueeze(1).expand(-1, 3)  # p block present?
        mask[:, 4:9] = self.has_d.unsqueeze(1).expand(-1, 5)  # d block present?

        # Flatten row-by-row keeping only present orbitals for each atom.
        # Result length = sum_i n_orb[i]
        self.diagonal = template[mask]  # 1-D tensor
        self.HDIM = self.diagonal.shape[-1]  # Total number of basis functions in system

        UsA = const.U[self.TYPE]  # (Nr_atoms,)
        UpA = const.Up[self.TYPE]  # (Nr_atoms,)
        UdA = const.Ud[self.TYPE]  # (Nr_atoms,)
        ns = const.n_s[self.TYPE]  # (Nr_atoms,)
        np = const.n_p[self.TYPE]  # (Nr_atoms,)
        nd = const.n_d[self.TYPE]  # (Nr_atoms,)

        template = torch.stack(
            (UsA, UpA, UdA),
            dim=1,
        )  # shape: (Nr_atoms, 9)

        template_ang = torch.stack(
            (
                torch.ones_like(UsA, dtype=torch.int64),  # s
                torch.ones_like(UsA, dtype=torch.int64) + 1,  # p
                torch.ones_like(UsA, dtype=torch.int64) + 2,  # d
            ),
            dim=1,
        )  # shape: (Nr_atoms, 9)

        template_el_per_shell = torch.stack((ns, np, nd), dim=1)  # shape: (Nr_atoms, 9)

        mask = torch.zeros_like(template, dtype=torch.bool)  # (Nr_atoms, 3)
        mask[:, 0] = True  # s always present
        mask[:, 1] = self.has_p
        mask[:, 2] = self.has_d
        self.Hubbard_U_sr = template[mask]  # shell-resolved Hubbard U
        self.shell_types = template_ang[mask]  # shell types (s=1,p=2,d=3)
        self.el_per_shell = template_el_per_shell[mask]  #
        self.n_shells_per_atom = const.max_ang[self.TYPE]
        self.H_INDEX_START_U = torch.zeros(self.Nats, dtype=torch.int64, device=device)
        self.H_INDEX_START_U[1:] = torch.cumsum(self.n_shells_per_atom, dim=0)[:-1]
        self.H_INDEX_END_U = self.H_INDEX_START_U + self.n_shells_per_atom - 1

        self.D0 = atomic_density_matrix(
            self.H_INDEX_START, self.HDIM, self.TYPE, const, self.has_p, self.has_d
        )
        self.D0 = 0.5 * self.D0
        self.q_spin_sr = None
        self.q = None

        if const.dftb3:
            self.dU_dq = const.dU_dq[self.TYPE]  # (Nr_atoms,)
        else:
            self.dU_dq = None


class StructureBatch(torch.nn.Module):
    """Batch container for multiple structures.
    Generates per-structure diagonal (on-site orbital energies) supporting heterogeneous atom counts.
    After construction:
      If batch_size == 1:
          self.diagonal -> 1D tensor (sum_i n_orb[i])
          self.HDIM -> int
      Else:
          self.diagonal -> list[ torch.Tensor ]
          self.HDIM -> list[int]
          self.diagonal_padded -> (batch_size, max_HDIM) zero-padded

    Parameters
    ----------
    cell : sequence or torch.Tensor
        Periodic cell specification. May be:
        - shape (3,) for one shared orthorhombic box
        - shape (3,3) for one shared triclinic cell
        - shape (B,3) for per-structure orthorhombic boxes
        - shape (B,3,3) for per-structure triclinic cells
    """

    def __init__(
        self,
        file,
        cell,
        const,
        charge: int = 0,
        Te: float = 3000.0,
        e_field: torch.Tensor = None,
        device: str = "cpu",
        req_grad_xyz: bool = False,
        ignore_spin: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.batch_size = len(file)
        # Auto-detect PDB vs XYZ from first file extension
        if file and isinstance(file[0], str) and file[0].lower().endswith(".pdb"):
            species, coordinates, _ = read_pdb(file, sort=False)
        else:
            species, coordinates = read_xyz(file, sort=False)  # Input coordinate file
        self.TYPE = torch.tensor(species, dtype=torch.int64, device=device)
        self.req_grad_xyz = req_grad_xyz
        self.RX = torch.tensor(
            coordinates[:, :, 0],
            device=device,
            dtype=torch.get_default_dtype(),
            requires_grad=self.req_grad_xyz,
        )
        self.RY = torch.tensor(
            coordinates[:, :, 1],
            device=device,
            dtype=torch.get_default_dtype(),
            requires_grad=self.req_grad_xyz,
        )
        self.RZ = torch.tensor(
            coordinates[:, :, 2],
            device=device,
            dtype=torch.get_default_dtype(),
            requires_grad=self.req_grad_xyz,
        )

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
            R = torch.stack((self.RX, self.RY, self.RZ), dim=-1)  # (B,N,3)
            R = wrap_positions(R, self.cell, self.cell_inv)
            self.RX, self.RY, self.RZ = R.unbind(dim=-1)

        self.coordinates = torch.stack(
            (self.RX, self.RY, self.RZ),
        )

        self.Nats = self.TYPE.shape[-1]
        self.const = const
        self.charge = charge
        self.Te = Te
        if e_field is None:
            self.e_field = torch.zeros(
                3, dtype=torch.get_default_dtype(), device=device
            )
        else:
            self.e_field = e_field

        self.device = device
        self.n_orbitals_per_atom = const.n_orb[self.TYPE]  # (batch, Nats)
        self.H_INDEX_START = torch.zeros(
            self.batch_size, self.Nats, dtype=torch.int64, device=device
        )
        self.H_INDEX_START[:, 1:] = torch.cumsum(self.n_orbitals_per_atom, dim=1)[
            :, :-1
        ]
        self.H_INDEX_END = self.H_INDEX_START + self.n_orbitals_per_atom - 1

        self.Mnuc = const.mass[self.TYPE]
        self.Znuc = const.tore[self.TYPE]

        tot_el = const.tore[self.TYPE].sum(dim=1) - charge
        if ((tot_el % 2) == 1).any() and not ignore_spin:
            raise ValueError("Closed shell systems require even number of electrons")
        self.Nocc = (tot_el / 2).to(int)

        self.Hubbard_U = const.U[self.TYPE]

        # Shell on-site energies per atom (pulled from dicts)
        EsA = const.Es[self.TYPE]  # (batch, Nats)
        EpA = const.Ep[self.TYPE]  # (batch, Nats)
        EdA = const.Ed[self.TYPE]  # (batch, Nats)

        # Which shells exist for each atom, based on your basis size:
        # 1  -> H-like: s
        # 4  -> main-group sp: s + 3*p
        # 9  -> transition-metal spd: s + 3*p + 5*d
        self.has_p = const.n_orb[self.TYPE] >= 4
        self.has_d = const.n_orb[self.TYPE] == 9

        # Vectorized orbital energy template (batch, Nats, 9)
        template = torch.stack(
            (
                EsA,  # s
                EpA,
                EpA,
                EpA,  # p
                EdA,
                EdA,
                EdA,
                EdA,
                EdA,  # d
            ),
            dim=2,
        )
        mask = torch.zeros_like(template, dtype=torch.bool)
        mask[:, :, 0] = True
        mask[:, :, 1:4] = self.has_p.unsqueeze(-1)
        mask[:, :, 4:9] = self.has_d.unsqueeze(-1)
        # Flat masked (structure-major) for global indexing
        self.diagonal_flat = template[mask]  # 1D (total_AOs,)
        self.HDIM_struct = self.n_orbitals_per_atom.sum(dim=1)  # (batch,)
        self.HDIM_total = int(self.diagonal_flat.shape[0])
        # Build 2D padded diagonal (batch, max_HDIM)
        max_HDIM = int(self.HDIM_struct.max().item())
        self.diagonal = torch.zeros(
            self.batch_size, max_HDIM, dtype=template.dtype, device=self.device
        )
        # Vectorized scatter fill
        # Compute per-structure prefix offsets
        struct_offsets = (
            torch.cumsum(self.HDIM_struct, dim=0) - self.HDIM_struct
        )  # (batch,)
        # Build index map for each structure
        idx_ranges = torch.arange(self.HDIM_total, device=self.device)
        # Map each global AO to its structure via searchsorted
        # (construct boundaries)
        boundaries = struct_offsets + self.HDIM_struct
        # For each structure b: valid global indices g satisfy struct_offsets[b] <= g < boundaries[b]
        # Expand to (batch, total_AOs) boolean mask
        g = idx_ranges.unsqueeze(0)
        in_struct = (g >= struct_offsets.unsqueeze(1)) & (g < boundaries.unsqueeze(1))
        # For each structure extract its slice without looping: masked_select then pad by assignment
        # Compute local positions = global - struct_offset
        local_pos = (g - struct_offsets.unsqueeze(1)) * in_struct
        # Mask invalid positions
        local_pos[~in_struct] = 0
        # Expand diagonal_flat to broadcast and scatter
        vals_expanded = self.diagonal_flat.unsqueeze(0).expand(self.batch_size, -1)
        # Scatter only where in_struct true
        self.diagonal.scatter_(
            1,
            local_pos[in_struct].view(self.batch_size, -1),
            vals_expanded[in_struct].view(self.batch_size, -1),
        )
        # Global AO start/end per atom
        self.H_INDEX_START_GLOBAL = self.H_INDEX_START + struct_offsets.unsqueeze(-1)
        self.H_INDEX_END_GLOBAL = self.H_INDEX_END + struct_offsets.unsqueeze(-1)

        UsA = const.U[self.TYPE]
        UpA = const.Up[self.TYPE]
        UdA = const.Ud[self.TYPE]
        ns = const.n_s[self.TYPE]
        np = const.n_p[self.TYPE]
        nd = const.n_d[self.TYPE]
        # Shell template (batch,Nats,3)
        template_shell = torch.stack((UsA, UpA, UdA), dim=2)
        template_ang = torch.stack(
            (
                torch.ones_like(UsA, dtype=torch.int64),
                torch.ones_like(UsA, dtype=torch.int64) + 1,
                torch.ones_like(UsA, dtype=torch.int64) + 2,
            ),
            dim=2,
        )
        template_el_per_shell = torch.stack((ns, np, nd), dim=2)
        mask_shell = torch.zeros_like(template_shell, dtype=torch.bool)
        mask_shell[:, :, 0] = True
        mask_shell[:, :, 1] = self.has_p
        mask_shell[:, :, 2] = self.has_d
        self.Hubbard_U_sr = template_shell[mask_shell]  # flat shells tensor
        self.shell_types = template_ang[mask_shell]
        self.el_per_shell = template_el_per_shell[mask_shell]
        self.n_shells_per_atom = const.max_ang[self.TYPE]  # (batch,Nats)
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
        self.D0 = atomic_density_matrix_batch(
            self.batch_size,
            self.H_INDEX_START,
            self.HDIM,
            self.TYPE,
            const,
            self.has_p,
            self.has_d,
        )
        self.D0 = 0.5 * self.D0

        if const.dftb3:
            self.dU_dq = const.dU_dq[self.TYPE]  # (Nr_atoms,)
        else:
            self.dU_dq = None


__all__ = ["Structure", "StructureBatch"]
