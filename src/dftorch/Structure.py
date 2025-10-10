import torch
import time
import copy
from .readXYZ import read_xyz

class Structure(torch.nn.Module):
    def __init__(self, TYPE, RX, RY, RZ,LBox,
                 const,
                 device='cpu', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TYPE = TYPE
        self.RX = RX
        self.RY = RY
        self.RZ = RZ
        self.LBox = LBox
        self.Nats = len(TYPE)
        self.const = const
        self.device = device

        self.n_orbitals_per_atom = const.n_orb[TYPE]
        self.H_INDEX_START = torch.zeros(self.Nats, dtype=torch.int64, device=RX.device)
        self.H_INDEX_START[1:] = torch.cumsum(self.n_orbitals_per_atom, dim=0)[:-1]
        self.H_INDEX_END = self.H_INDEX_START + self.n_orbitals_per_atom - 1

        self.Mnuc = const.mass[TYPE]
        self.Znuc = const.tore[TYPE]
        self.Nocc = int(const.tore[TYPE].sum()/2)

        self.Hubbard_U = const.U[TYPE]

        # Shell on-site energies per atom (pulled from your dicts)
        EsA = const.Es[TYPE]   # (Nr_atoms,)
        EpA = const.Ep[TYPE]   # (Nr_atoms,)
        EdA = const.Ed[TYPE]   # (Nr_atoms,)

        # Which shells exist for each atom, based on your basis size:
        # 1  -> H-like: s
        # 4  -> main-group sp: s + 3*p
        # 9  -> transition-metal spd: s + 3*p + 5*d
        self.has_p = (const.n_orb[TYPE] >= 4)          # p present for 4 or 9
        self.has_d = (const.n_orb[TYPE] == 9)          # d present only for 9 here
        # (Optional: if you ever use sd-only (6 orbitals), set has_d |= (const.n_orb[TYPE] == 6)
        #            and exclude p for that case.)

        # Build a per-atom template in the standard AO order: [s, px, py, pz, dxy, dyz, dzx, dx2-y2, dz2]
        # (All p orbitals get EpA; all d orbitals get EdA.)
        template = torch.stack(
            (
                EsA,             # s
                EpA, EpA, EpA,   # p triplet
                EdA, EdA, EdA, EdA, EdA  # d quintet
            ),
            dim=1,  # shape: (Nr_atoms, 9)
        )

        # Per-atom mask telling which of the 9 positions are actually present
        mask = torch.zeros_like(template, dtype=torch.bool)  # (Nr_atoms, 9)
        mask[:, 0]    = True                                  # s always present
        mask[:, 1:4]  = self.has_p.unsqueeze(1).expand(-1, 3)      # p block present?
        mask[:, 4:9]  = self.has_d.unsqueeze(1).expand(-1, 5)      # d block present?

        # Flatten row-by-row keeping only present orbitals for each atom.
        # Result length = sum_i n_orb[i]
        self.diagonal = template[mask]                              # 1-D tensor
        self.HDIM = self.diagonal.shape[-1]  # Total number of basis functions in system


        UsA = const.U[TYPE]   # (Nr_atoms,)
        UpA = const.Up[TYPE]   # (Nr_atoms,)
        UdA = const.Ud[TYPE]   # (Nr_atoms,)

        ns = const.n_s[TYPE]   # (Nr_atoms,)
        np = const.n_p[TYPE]   # (Nr_atoms,)
        nd = const.n_d[TYPE]   # (Nr_atoms,)


        template = torch.stack(
            (
                UsA,             # s
                UpA,   # p
                UdA   # d
            ),
            dim=1,  # shape: (Nr_atoms, 9)
        )

        template_ang = torch.stack(
            (
                torch.ones_like(UsA, dtype=torch.int64),             # s
                torch.ones_like(UsA, dtype=torch.int64)+1,   # p
                torch.ones_like(UsA, dtype=torch.int64)+2   # d
            ),
            dim=1  # shape: (Nr_atoms, 9)
        )

        template_el_per_shell = torch.stack(
            (
                ns,             # s
                np,   # p
                nd   # d
            ),
            dim=1,  # shape: (Nr_atoms, 9)
        )


        mask = torch.zeros_like(template, dtype=torch.bool)  # (Nr_atoms, 3)
        mask[:, 0]    = True                                  # s always present
        mask[:, 1]  = self.has_p
        mask[:, 2]  = self.has_d
        self.Hubbard_U_sr = template[mask]                              # shell-resolved Hubbard U
        self.shell_types = template_ang[mask]                              # shell types (s=1,p=2,d=3)
        self.el_per_shell = template_el_per_shell[mask]                              # 
        self.n_shells_per_atom = const.max_ang[TYPE]
        self.H_INDEX_START_U = torch.zeros(self.Nats, dtype=torch.int64, device=RX.device)
        self.H_INDEX_START_U[1:] = torch.cumsum(self.n_shells_per_atom, dim=0)[:-1]
        self.H_INDEX_END_U = self.H_INDEX_START_U + self.n_shells_per_atom - 1


    # def to(self, device):
    #     self.device = device
    #     self.TYPE = self.TYPE.to(device)
    #     self.RX = self.RX.to(device)
    #     self.RY = self.RY.to(device)
    #     self.RZ = self.RZ.to(device)
    #     self.H_INDEX_START = self.H_INDEX_START.to(device)
    #     self.H_INDEX_END = self.H_INDEX_END.to(device)
    #     self.Nats = self.Nats.to(device)
    #     self.Znuc = self.Znuc.to(device)
    #     return super().to(device)
