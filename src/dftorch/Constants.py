from typing import Any

import os

import numpy as np
import torch

from ._elements import atomic_num, label, mass, symbol_to_number
from ._io import read_pdb, read_xyz
from ._tools import (
    library_output_enabled,
    load_hubbard_derivs,
    load_spinw_to_matrix,
    ordered_pairs_from_TYPE,
)
from ._bond_integral import get_skf_tensors  # TYPE WILL BE PASSED, TYPE IS THE LIST OF ALL SPECIES IN THE SYSTEM



class Constants(torch.nn.Module):
    """Slater-Koster parameter database and element-level constants for DFTB.

    Reads the Slater-Koster files (SKF) for all element pairs present in the
    input structure, and exposes the resulting tensors (Hamiltonian splines,
    repulsive splines, on-site energies, Hubbard U values, etc.) as registered
    ``nn.Parameter`` buffers so they move to the correct device alongside the
    model.

    Parameters
    ----------
    dftorch_params : dict
        Simulation parameter dictionary.  Required keys:

        ``SKFPATH`` : str
            Path to the directory containing ``.skf`` files (e.g. ``mio-1-1/``).
        ``FILENAME`` : str or list of str
            Path(s) to the input structure file(s) used to determine which
            element pairs to load.

        Optional keys:

        ``DFTB3`` : bool, default False
            Load Hubbard-derivative data for third-order corrections.
        ``MAGNETIC_HUBBARD_LDEP`` : bool, default False
            Use shell-dependent (l-dependent) Hubbard U parameters.  This one
            key has two consumers: it selects the shell- versus atom-resolved
            spin W matrix below, and it selects the shell-resolved Hubbard U on
            the Coulomb path (``ESDriver._select_coulomb_hubbard``), which also
            makes ``ESDriver.forward`` build ``structure.C_sr``.  The name is a
            known misnomer in the non-spin case, kept deliberately rather than
            split into a second flag (decision D-23).
        ``GRAD_PARAM`` : bool, default False
            Enable parameter gradients (for ML-SK fitting workflows).
        ``VERBOSE_LIBRARY_OUTPUT`` : bool, default True
            Whether the library prints its *status* chatter -- progress
            banners, per-iteration lines, timings and informational notices --
            to stdout.  **The default is True, which reproduces the output
            every caller saw before this key existed.**  Set it to ``False``
            to opt into a quiet run.

            Scope, deliberately narrow (decision D-02):

            * SUPPRESSED when False: status output only.  The exact set is
              enumerated, one row per ``print`` call in the package, in
              ``docs/LIBRARY-OUTPUT-INVENTORY.md``.
            * NOT suppressed, ever: genuine failure, non-convergence and
              silent-degradation warnings -- an SCF that hit ``SCF_MAX_ITER``,
              a degenerate Krylov direction, a ``spinw.txt`` that would not
              load, a ``COUL_METHOD='PME'`` request downgraded to ``'FULL'``.
              Gating those would trade noise for silent wrongness.
            * NOT affected: the ~28 prints already behind a per-call
              ``verbose`` or ``debug`` argument.  Those default to off and
              stay off; this key does not switch them on.

            Read into ``self.verbose_output``.  The attribute is deliberately
            NOT named ``verbose``: 17 function signatures across the package
            already carry a parameter of that name whose default is the
            opposite, and the collision would be a live footgun.
    """

    def __init__(self, dftorch_params: dict[str, Any]) -> None: #When you initialize this object, give it all the parameters
        """Load Slater-Koster data and element constants for the active system.

        Parameters
        ----------
        dftorch_params : dict[str, Any]
            Simulation parameter dictionary used to locate SKF files, input
            structures, and optional third-order / magnetic-Hubbard data.
        """

        super().__init__()

        self.skfpath = dftorch_params["SKFPATH"] #Where to read the skf file from
        self.magnetic_hubbard_ldep = dftorch_params.get("MAGNETIC_HUBBARD_LDEP", False) #Used in Spin-orbit coupling
        self.dftb3 = dftorch_params.get("DFTB3", False) 
        self.grad_param = dftorch_params.get("GRAD_PARAM", False)
        # Status-output switch (decision D-02). Defaults to True so that a
        # caller who omits VERBOSE_LIBRARY_OUTPUT sees byte-identical output to
        # before the key existed. NOT named `verbose`: 17 signatures in this
        # package already use that name with the opposite default.
        self.verbose_output = library_output_enabled(dftorch_params)
        self.symbol_to_number = symbol_to_number
        self.label = label
        self.atomic_num = atomic_num

        self.shell_dim = torch.nn.Parameter(
            torch.tensor([0, 1, 3, 5, 7], dtype=torch.int64), requires_grad=False
        )
        self.atomic_num = torch.nn.Parameter(atomic_num, requires_grad=False)
        self.mass = torch.nn.Parameter(mass, requires_grad=False)

        if isinstance(dftorch_params["FILENAME"], str): #Whether or not to read pdb or xyz
            if dftorch_params["FILENAME"].lower().endswith(".pdb"):
                species, _, _ = read_pdb(
                    [dftorch_params["FILENAME"]], sort=False
                )  # Input coordinate file
            else:
                species, _ = read_xyz(
                    [dftorch_params["FILENAME"]], sort=False
                )  # Input coordinate file

        else:
            files = dftorch_params["FILENAME"] #GET THE SPECIES LIST FROM THE XYZ FILES
            if all(f.lower().endswith(".pdb") for f in files):
                species, _, _ = read_pdb(files, sort=False)
            else:
                species, _ = read_xyz(files, sort=False)  # Input coordinate file 

        TYPE = torch.tensor(species.flatten())
        pairs_tensor, _, _ = ordered_pairs_from_TYPE(TYPE)
        pair_lookup = torch.full(
            (len(self.label), len(self.label)),
            -1,
            dtype=torch.long,
            device=TYPE.device,
        )
        if pairs_tensor.numel() > 0:
            pair_lookup[pairs_tensor[:, 0], pairs_tensor[:, 1]] = torch.arange(
                pairs_tensor.shape[0], dtype=torch.long, device=TYPE.device
            )

        (
            R_tensor,
            R_orb,
            n_grid,
            coeffs_tensor,
            R_rep_tensor,
            rep_splines_tensor,
            close_exp_tensor,
            N_ORB,
            MAX_ANG,
            MAX_ANG_OCC,
            TORE,
            N_S,
            N_P,
            N_D,
            N_F,
            ES,
            EP,
            ED,
            EF,
            US,
            UP,
            UD,
            UF,
            SHELL_PRESENT,
        ) = get_skf_tensors(TYPE, self.skfpath)  # Gets all the parameters from the SKF files, including f-shell data.

        try:
            w_shell = load_spinw_to_matrix(
                os.path.join(self.skfpath, "spinw.txt"), device=TYPE.device
            )
            self.w_shell = torch.nn.Parameter(w_shell, requires_grad=False)
            w_atom = torch.zeros(self.w_shell.shape[0], device=TYPE.device)
            w_atom[TYPE] = self.w_shell[
                TYPE, MAX_ANG_OCC[TYPE] - 1, MAX_ANG_OCC[TYPE] - 1
            ]
            self.w_atom = torch.nn.Parameter(w_atom, requires_grad=False)

            if self.magnetic_hubbard_ldep:
                self.w = torch.nn.Parameter(w_shell.clone(), requires_grad=False)
            else:
                self.w = torch.nn.Parameter(w_atom.clone(), requires_grad=False)
        except (FileNotFoundError, OSError, ValueError):
            # DELIBERATELY UNCONDITIONAL -- not gated on self.verbose_output.
            # 05-CONTEXT.md leaves the fate of this warning to planning
            # discretion; the choice made here (decision D-02) is that it stays
            # on even for a caller who set VERBOSE_LIBRARY_OUTPUT=False,
            # because spin-orbit coupling has just been silently dropped from
            # the calculation. A user who opts into a quiet run is asking to
            # not be told about progress, not asking to not be told that their
            # physics changed.
            print(
                "Warning: could not load spinw.txt file for spin-orbit coupling. Proceeding without SOC."
            )
            self.w = None

        self.R_tensor = torch.nn.Parameter(R_tensor, requires_grad=False)
        # Per-pair tabulated grid length, shape (n_pairs,). Companion to
        # R_tensor for the per-pair knot lookup (decision D-01, REG-06).
        self.n_grid = torch.nn.Parameter(n_grid, requires_grad=False)
        # R_orb is the LONGEST grid seen across all pairs, kept unchanged and
        # exported additively. ESDriver's batch path, _stress, _ml_sk and the
        # SEDACS interface all still read it; the per-pair path was added
        # alongside rather than in place of it so those callers keep working
        # (REG-03 protected additively).
        self.R_orb = torch.nn.Parameter(R_orb, requires_grad=False)
        self.coeffs_tensor = torch.nn.Parameter(coeffs_tensor, requires_grad=False)
        self.R_rep_tensor = torch.nn.Parameter(R_rep_tensor, requires_grad=False)
        self.rep_splines_tensor = torch.nn.Parameter(
            rep_splines_tensor, requires_grad=self.grad_param
        )
        self.close_exp_tensor = torch.nn.Parameter(
            close_exp_tensor, requires_grad=False
        )
        self.pair_lookup = torch.nn.Parameter(pair_lookup, requires_grad=False)

        self.n_orb = torch.nn.Parameter(N_ORB, requires_grad=False)
        self.max_ang = torch.nn.Parameter(
            MAX_ANG, requires_grad=False
        )  # AO with max angular momentum l
        self.max_ang_occ = torch.nn.Parameter(
            MAX_ANG_OCC, requires_grad=False
        )  # occupied AO with max angular momentum l
        self.tore = torch.nn.Parameter(TORE, requires_grad=False)
        self.n_s = torch.nn.Parameter(N_S, requires_grad=False)
        self.n_p = torch.nn.Parameter(N_P, requires_grad=False)
        self.n_d = torch.nn.Parameter(N_D, requires_grad=False)
        self.n_f = torch.nn.Parameter(N_F, requires_grad=False)
        self.shell_present = torch.nn.Parameter(SHELL_PRESENT, requires_grad=False)

        # --- RECORDED DEFECT: the per-atom Hubbard U is always the s value ---
        #
        # Read this before editing the next line. It is a known, measured
        # defect that is deliberately NOT fixed here (decision D-6.10).
        #
        # What the line does. `self.U` is the one electron-repulsion strength
        # each atom is charged at whenever the calculation tracks one charge
        # per atom. It is taken from the s column, `US`, unconditionally, for
        # every element. The p, d and f columns are loaded on the three lines
        # below and, on that coarser path, are never consulted.
        #
        # Why that is usually harmless. In most of these parameter sets an
        # element's s, p and d strengths are equal, so which column is read
        # makes no difference. Nitrogen is the case in hand: its s, p and d
        # strengths all read 13.33 eV, so for nitrogen this line is exact.
        #
        # Why it is wrong for europium. Europium's four strengths are NOT
        # equal: s is 5.71 eV and f is 13.61 eV, a factor of 2.4 apart. Seven
        # of europium's nine outer electrons live in the f group, so this line
        # charges the large majority of them at less than half the strength
        # their own shell says they should pay.
        #
        # How large the effect is, measured. Substituting the f value for
        # europium and re-running the self-consistent Eu-N diatomic at 40 A --
        # far enough apart that the two atoms should be independent -- moves
        # the leftover charge transfer from 0.29 to 0.20 electrons, about
        # 30 percent, and the reported energy from -1.96 to -1.73 eV. That is
        # a real effect on a real observable, not a rounding-level concern.
        # (Re-measured 2026-08-06; first measured 2026-08-04.)
        #
        # What this is NOT. It was tested as a candidate cause of the Phase 6
        # runaway charge loop and DISPROVEN: with the substitution in place the
        # loop still failed. The failure pattern changed, which is what makes
        # it tempting to misread. The actual cause was the low-rank Krylov
        # convergence accelerator. Do not present this line as that diagnosis.
        #
        # The remedy Phase 6 adopted. Not this line -- the shell-resolved path,
        # selected by the `MAGNETIC_HUBBARD_LDEP` parameter key, which tracks
        # one charge per orbital group and charges each group at its own
        # strength. With that key set, europium's f electrons already pay the
        # f rate and this line is not consulted for the repulsion energy.
        #
        # Why it is recorded rather than fixed. Changing this assignment moves
        # numbers for every f element in every existing calculation, on the
        # path that is still the default. That is a decision, not a tidy-up,
        # and `tests/test_shell_resolved_scf_f.py`
        # `::test_the_per_atom_strength_still_comes_from_the_s_group` exists to
        # make any such edit visible rather than silent.
        #
        # A note on the name "Constants.py:232". Decision D-6.10 and several
        # planning documents refer to this defect by that line number, which
        # was where the assignment sat before this comment was inserted above
        # it. The comment pushed it down. Search for the assignment itself
        # rather than the number; `tools/check_verdict_doc.py` verifies the
        # current line and fails if it moves again.
        #
        # Fuller account: `docs/SINGLE-SHOT-VS-SELF-CONSISTENT.md` and
        # decision D-6.10.
        # ---------------------------------------------------------------------
        self.U = torch.nn.Parameter(US, requires_grad=self.grad_param)
        self.Up = torch.nn.Parameter(UP, requires_grad=self.grad_param)
        self.Ud = torch.nn.Parameter(UD, requires_grad=self.grad_param)
        self.Uf = torch.nn.Parameter(UF, requires_grad=self.grad_param)
        self.Es = torch.nn.Parameter(ES, requires_grad=self.grad_param)
        self.Ep = torch.nn.Parameter(EP, requires_grad=self.grad_param)
        self.Ed = torch.nn.Parameter(ED, requires_grad=self.grad_param)
        self.Ef = torch.nn.Parameter(EF, requires_grad=self.grad_param)

        # ── DFTB3: Hubbard derivatives dU/dq ─────────────────────────────
        if self.dftb3:
            _hubbard_path = os.path.join(self.skfpath, "hubbard_derivative.txt")
            try:
                dU_dq = load_hubbard_derivs(_hubbard_path, device=TYPE.device)
                self.dU_dq = torch.nn.Parameter(dU_dq, requires_grad=False)
                self.dftb3 = True
            except (FileNotFoundError, OSError):
                # DFTB2 parameter set — no Hubbard derivatives available
                self.dU_dq = None
                self.dftb3 = False
        else:
            self.dU_dq = None
            self.dftb3 = False
        if self.verbose_output:
            print(f"DFTB3: {self.dftb3}")

        # ─────────────────────────────────────────────────────────────────

    def forward(self):
        pass

