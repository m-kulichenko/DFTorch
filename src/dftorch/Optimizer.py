"""Geometry and lattice relaxation driver for DFTorch.

Provides a FIRE-based geometry optimizer that supports:
  - Atomic position relaxation (force-based)
  - Lattice / cell relaxation (stress-based)
  - Combined (simultaneous positions + cell)
  - Fixed-atom constraints via a boolean mask

Usage
-----
>>> from dftorch.Optimizer import GeoOpt
>>> opt = GeoOpt(es_driver, const)
>>> opt.run(structure, dftorch_params, fmax=0.01)          # atoms only
>>> opt.run(structure, dftorch_params, fmax=0.01, relax_cell=True)  # + cell
"""

from __future__ import annotations

import time
from typing import Optional

import torch

from ._cell import wrap_positions
from ._io import write_XYZ_trajectory, write_pdb_frame
from .ESDriver import ESDriver


# ═══════════════════════════════════════════════════════════════════════════
#  FIRE  (Fast Inertial Relaxation Engine)
# ═══════════════════════════════════════════════════════════════════════════
#  E. Bitzek et al., Phys. Rev. Lett. 97, 170201 (2006)
#  Guénolé et al., Comp. Mat. Sci. 175, 109584 (2020) — FIRE 2.0 variant
# ═══════════════════════════════════════════════════════════════════════════


class _FIREState:
    """Mutable state for the FIRE integrator."""

    def __init__(
        self,
        ndof: int,
        device: torch.device,
        dtype: torch.dtype,
        dt_start: float = 0.1,
        dt_max: float = 1.0,
        alpha_start: float = 0.1,
        f_inc: float = 1.1,
        f_dec: float = 0.5,
        f_alpha: float = 0.99,
        n_min: int = 5,
    ):
        self.v = torch.zeros(ndof, device=device, dtype=dtype)
        self.dt = dt_start
        self.dt_max = dt_max
        self.alpha = alpha_start
        self.alpha_start = alpha_start
        self.f_inc = f_inc
        self.f_dec = f_dec
        self.f_alpha = f_alpha
        self.n_min = n_min
        self.n_pos = 0  # consecutive steps with P > 0

    def step(self, force: torch.Tensor) -> torch.Tensor:
        """Return displacement vector Δx.  ``force`` has same shape as ``self.v``."""
        P = torch.dot(self.v, force)

        if P > 0:
            self.n_pos += 1
            # Mix velocity toward force direction
            f_norm = torch.linalg.norm(force)
            v_norm = torch.linalg.norm(self.v)
            self.v = (1.0 - self.alpha) * self.v + self.alpha * (
                v_norm / f_norm.clamp(min=1e-30)
            ) * force
            if self.n_pos > self.n_min:
                self.dt = min(self.dt * self.f_inc, self.dt_max)
                self.alpha *= self.f_alpha
        else:
            # Reset
            self.v.zero_()
            self.dt *= self.f_dec
            self.alpha = self.alpha_start
            self.n_pos = 0

        # Velocity-Verlet half: v += 0.5 * dt * F  (mass = 1)
        self.v = self.v + 0.5 * self.dt * force

        # Displacement
        dx = self.dt * self.v

        # Second half kick will happen after the next force evaluation
        # (for now we store v at half-step; the next call's force vector
        #  is implicitly the next-step force)
        self.v = self.v + 0.5 * self.dt * force

        return dx


# ═══════════════════════════════════════════════════════════════════════════
#  L-BFGS state (two-loop recursion, no line search — trust-region capped)
# ═══════════════════════════════════════════════════════════════════════════


class _LBFGSState:
    """Memory-limited BFGS with a maximum step (trust-region cap)."""

    def __init__(
        self,
        ndof: int,
        device: torch.device,
        dtype: torch.dtype,
        memory: int = 20,
        maxstep: float = 0.2,
    ):
        self.memory = memory
        self.maxstep = maxstep
        self.s_list: list[torch.Tensor] = []
        self.y_list: list[torch.Tensor] = []
        self.rho_list: list[torch.Tensor] = []
        self.x_prev: Optional[torch.Tensor] = None
        self.g_prev: Optional[torch.Tensor] = None
        self.device = device
        self.dtype = dtype

    def step(self, x: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """Return displacement Δx.  ``grad = -force`` (gradient of energy)."""
        if self.x_prev is not None:
            s = x - self.x_prev
            y = grad - self.g_prev
            sy = torch.dot(s, y)
            if sy > 1e-30:
                self.s_list.append(s)
                self.y_list.append(y)
                self.rho_list.append(1.0 / sy)
                if len(self.s_list) > self.memory:
                    self.s_list.pop(0)
                    self.y_list.pop(0)
                    self.rho_list.pop(0)

        self.x_prev = x.clone()
        self.g_prev = grad.clone()

        # Two-loop recursion
        q = grad.clone()
        n = len(self.s_list)
        alpha_arr = []

        for i in range(n - 1, -1, -1):
            a = self.rho_list[i] * torch.dot(self.s_list[i], q)
            alpha_arr.insert(0, a)
            q = q - a * self.y_list[i]

        # Initial Hessian approximation H0 = (s^T y) / (y^T y) * I
        if n > 0:
            sy = torch.dot(self.s_list[-1], self.y_list[-1])
            yy = torch.dot(self.y_list[-1], self.y_list[-1])
            H0 = sy / yy.clamp(min=1e-30)
        else:
            H0 = torch.tensor(1.0, device=self.device, dtype=self.dtype)

        r = H0 * q
        for i in range(n):
            b = self.rho_list[i] * torch.dot(self.y_list[i], r)
            r = r + (alpha_arr[i] - b) * self.s_list[i]

        dx = -r  # descent direction

        # Trust-region cap: limit maximum atomic displacement
        step_norm = torch.linalg.norm(dx)
        if step_norm > self.maxstep:
            dx = dx * (self.maxstep / step_norm)

        return dx


# ═══════════════════════════════════════════════════════════════════════════
#  Main geometry optimizer class
# ═══════════════════════════════════════════════════════════════════════════


class GeoOpt:
    """Geometry (and lattice) optimizer for SCC-DFTB.

    Parameters
    ----------
    es_driver : ESDriver
        The electronic structure driver (handles H0 build + SCF + forces).
    const : object
        Constants object (Slater–Koster tables, repulsive splines, etc.).
    method : str
        Optimisation algorithm: ``"FIRE"`` or ``"LBFGS"`` (default).

    Examples
    --------
    >>> opt = GeoOpt(es_driver, const)
    >>> opt.run(structure, dftorch_params, fmax=0.01)
    >>> # With lattice relaxation:
    >>> opt.run(structure, dftorch_params, fmax=0.01, relax_cell=True)
    >>> # With fixed atoms (e.g. freeze first 10):
    >>> mask = torch.ones(structure.Nats, dtype=torch.bool)
    >>> mask[:10] = False
    >>> opt.run(structure, dftorch_params, fmax=0.01, free_mask=mask)
    """

    def __init__(self, es_driver: ESDriver, const, method: str = "LBFGS"):
        self.es_driver = es_driver
        self.const = const
        self.method = method.upper()
        assert self.method in ("FIRE", "LBFGS"), (
            f"Unknown method '{method}'. Use 'FIRE' or 'LBFGS'."
        )

        # History arrays (filled during run)
        self.E_array: Optional[torch.Tensor] = None
        self.Fmax_array: Optional[torch.Tensor] = None
        self.Smax_array: Optional[torch.Tensor] = None

    # ── public interface ─────────────────────────────────────────────────

    def run(
        self,
        structure,
        dftorch_params: dict,
        fmax: float = 0.05,
        smax: float = 0.001,
        max_steps: int = 500,
        relax_cell: bool = False,
        hydrostatic_only: bool = False,
        free_mask: Optional[torch.Tensor] = None,
        dump_interval: int = 1,
        traj_filename: str = "opt_trj",
        # FIRE parameters
        dt_start: float = 0.5,
        dt_max: float = 2.0,
        # LBFGS parameters
        lbfgs_memory: int = 20,
        # Step size limits (both methods)
        maxstep: float = 0.2,
        max_strain: float = 0.04,
    ):
        """Run geometry optimisation.

        Parameters
        ----------
        structure : Structure
            The structure to optimise.  Modified in-place.
        dftorch_params : dict
            DFTorch calculation parameters.
        fmax : float
            Force convergence threshold in eV/Å (max component on any
            free atom).
        smax : float
            Stress convergence threshold in eV/Å³ (max component).
            Only used when ``relax_cell=True``.
        max_steps : int
            Maximum number of optimisation steps.
        relax_cell : bool
            If True, also optimise the unit cell (requires periodic system).
        hydrostatic_only : bool
            If True, apply only isotropic (hydrostatic) cell scaling —
            all three lattice vectors are scaled uniformly.
            If False (default), fully anisotropic cell relaxation.
        free_mask : torch.Tensor (bool), optional
            Shape ``(Nats,)``.  True = atom is free to move, False = frozen.
            Default: all atoms free.
        dump_interval : int
            Write trajectory every this many steps.
        traj_filename : str
            Output XYZ trajectory file.
        dt_start, dt_max : float
            FIRE time-step parameters (only used if ``method="FIRE"``).
        maxstep : float
            Maximum Cartesian step per atom per iteration (Å).
            Used by both FIRE (via dt capping) and LBFGS (trust region).
        max_strain : float
            Maximum strain magnitude per cell update (dimensionless).
            Controls how much the cell can deform in a single step.
            Default 0.04 (4%).
        lbfgs_memory : int
            Number of history pairs for LBFGS (default 20).
        """
        device = structure.RX.device
        dtype = structure.RX.dtype
        Nats = structure.Nats

        if free_mask is None:
            free_mask = torch.ones(Nats, dtype=torch.bool, device=device)
        else:
            free_mask = free_mask.to(device=device)

        n_free = int(free_mask.sum().item())
        if n_free == 0 and not relax_cell:
            print("GeoOpt: No free atoms and no cell relaxation — nothing to optimise.")
            return

        # Degree-of-freedom count: 3 * n_free atoms + (6 or 1) cell DOFs
        ndof_atoms = 3 * n_free
        if relax_cell:
            if structure.cell is None:
                raise ValueError("relax_cell=True requires a periodic cell.")
            # Cell DOFs: 6 independent (symmetric deformation) if anisotropic,
            # 1 (volume) if hydrostatic.
            ndof_cell = 1 if hydrostatic_only else 6
        else:
            ndof_cell = 0
        ndof = ndof_atoms + ndof_cell

        # Initialise optimiser state
        if self.method == "FIRE":
            state = _FIREState(ndof, device, dtype, dt_start=dt_start, dt_max=dt_max)
        else:
            state = _LBFGSState(
                ndof, device, dtype, memory=lbfgs_memory, maxstep=maxstep
            )

        # History
        self.E_array = torch.empty(0, device=device, dtype=dtype)
        self.Fmax_array = torch.empty(0, device=device, dtype=dtype)
        if relax_cell:
            self.Smax_array = torch.empty(0, device=device, dtype=dtype)

        converged = False

        for step in range(max_steps):
            t0 = time.perf_counter()
            print(f"═══════════ GeoOpt step {step} ═══════════")

            # ── 1. SCF + Energy ──────────────────────────────────────────
            self.es_driver(structure, self.const, do_scf=True)

            # ── 2. Forces ────────────────────────────────────────────────
            self.es_driver.calc_forces(structure, self.const)

            # ── 3. Stress (if cell relaxation) ───────────────────────────
            if relax_cell:
                self.es_driver.calc_stress(structure, self.const)
                stress = structure.stress_tot  # (3, 3) eV/Å³

                # D3 stress contribution
                if structure.dftd3 is not None:
                    coords_ang = torch.stack(
                        [structure.RX, structure.RY, structure.RZ], dim=1
                    )
                    stress_d3 = structure.dftd3.get_stress(coords_ang, structure.cell)
                    stress = stress + stress_d3

            # ── 4. Convergence check ─────────────────────────────────────
            f = structure.f_tot  # (3, Nats)
            f_free = f[:, free_mask]  # only free atoms
            fmax_val = f_free.abs().max().item() if n_free > 0 else 0.0

            self.E_array = torch.cat(
                [self.E_array, structure.e_tot.detach().unsqueeze(0)]
            )
            self.Fmax_array = torch.cat(
                [self.Fmax_array, torch.tensor([fmax_val], device=device, dtype=dtype)]
            )

            info = f"E = {structure.e_tot.item():16.8f} eV   Fmax = {fmax_val:.6f} eV/Å"

            if relax_cell:
                smax_val = stress.abs().max().item()
                self.Smax_array = torch.cat(
                    [
                        self.Smax_array,
                        torch.tensor([smax_val], device=device, dtype=dtype),
                    ]
                )
                V = torch.abs(torch.det(structure.cell)).item()
                info += f"   Smax = {smax_val:.6f} eV/Å³   V = {V:.2f} ų"
                cell_converged = smax_val < smax
            else:
                cell_converged = True

            print(info)

            if step % dump_interval == 0:
                write_XYZ_trajectory(
                    traj_filename + ".xyz",
                    structure,
                    f"Step {step}: {info}",
                    step=step,
                )
                write_pdb_frame(
                    traj_filename + ".pdb",
                    structure,
                    structure.cell,
                    step=step,
                    comment=f"Step {step}: {info}",
                    mode="a",
                )

            if fmax_val < fmax and cell_converged:
                print(f"\n✓ Converged in {step + 1} steps.")
                converged = True
                break

            # ── 5. Build generalised force / gradient vector ─────────────
            # Atom forces → gradient (negative force)
            grad_atoms = -f_free.T.reshape(-1)  # (3*n_free,)

            if relax_cell:
                grad_cell = self._cell_gradient(structure, stress, hydrostatic_only)
                gen_grad = torch.cat([grad_atoms, grad_cell])
            else:
                gen_grad = grad_atoms

            # Generalised position vector (for LBFGS curvature)
            x_atoms = torch.stack(
                [
                    structure.RX[free_mask],
                    structure.RY[free_mask],
                    structure.RZ[free_mask],
                ],
                dim=-1,
            ).reshape(-1)

            if relax_cell:
                x_cell = self._cell_position_vector(structure, hydrostatic_only)
                gen_x = torch.cat([x_atoms, x_cell])
            else:
                gen_x = x_atoms

            # ── 6. Compute displacement ──────────────────────────────────
            if self.method == "FIRE":
                # FIRE uses force (= -grad)
                dx = state.step(-gen_grad)
            else:
                dx = state.step(gen_x, gen_grad)

            # Per-atom step capping (FIRE doesn't have built-in trust region)
            dx_atoms = dx[:ndof_atoms].reshape(-1, 3)
            if n_free > 0:
                atom_steps = torch.linalg.norm(dx_atoms, dim=-1)
                max_atom_step = atom_steps.max().item()
            else:
                max_atom_step = 0.0
            if max_atom_step > maxstep:
                scale = maxstep / max_atom_step
                dx_atoms = dx_atoms * scale
                if ndof_cell > 0:
                    dx = torch.cat([dx_atoms.reshape(-1), dx[ndof_atoms:] * scale])
                else:
                    dx = dx_atoms.reshape(-1)

            # ── 7. Apply displacement ────────────────────────────────────
            self._apply_atom_displacement(structure, dx_atoms, free_mask)

            if relax_cell:
                dx_cell = dx[ndof_atoms:]
                self._apply_cell_displacement(
                    structure, dx_cell, hydrostatic_only, max_strain
                )

            dt = time.perf_counter() - t0
            print(f"  step time: {dt:.2f} s\n")

        if not converged:
            print(
                f"\n✗ Not converged after {max_steps} steps.  "
                f"Fmax = {fmax_val:.6f} eV/Å"
            )

    # ── internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _apply_atom_displacement(structure, dx: torch.Tensor, free_mask: torch.Tensor):
        """Apply Cartesian displacement ``dx`` (n_free, 3) to free atoms."""
        structure.RX[free_mask] += dx[:, 0]
        structure.RY[free_mask] += dx[:, 1]
        structure.RZ[free_mask] += dx[:, 2]

        # Wrap into cell
        if structure.cell is not None:
            R = torch.stack([structure.RX, structure.RY, structure.RZ], dim=-1)
            R = wrap_positions(R, structure.cell, structure.cell_inv)
            structure.RX, structure.RY, structure.RZ = R.unbind(dim=-1)

    @staticmethod
    def _cell_gradient(
        structure, stress: torch.Tensor, hydrostatic_only: bool
    ) -> torch.Tensor:
        """Convert stress tensor to a generalised cell gradient vector.

        For full anisotropic relaxation the 6 independent Voigt components
        of the *symmetrised* stress are used, weighted by the cell volume
        so that the gradient has units of eV (energy-like) and couples
        properly with the strain DOFs.

        For hydrostatic-only, a single scalar gradient proportional to
        the pressure (trace of stress / 3) is returned.
        """
        V = torch.abs(torch.det(structure.cell))
        if hydrostatic_only:
            # Pressure = -Tr(σ)/3.  Gradient w.r.t. volume.
            pressure = stress.trace() / 3.0
            return (V * pressure).unsqueeze(0)  # (1,)
        else:
            # Voigt: xx, yy, zz, yz, xz, xy
            s = 0.5 * (stress + stress.T)  # symmetrise
            grad = V * torch.stack(
                [
                    s[0, 0],
                    s[1, 1],
                    s[2, 2],
                    s[1, 2],
                    s[0, 2],
                    s[0, 1],
                ]
            )
            return grad  # (6,)

    @staticmethod
    def _cell_position_vector(structure, hydrostatic_only: bool) -> torch.Tensor:
        """Generalised cell position vector (Voigt strain = 0 initially)."""
        if hydrostatic_only:
            V = torch.abs(torch.det(structure.cell))
            return V.unsqueeze(0)
        else:
            cell = structure.cell
            # Voigt-like: return the 6 independent cell-metric components
            # Use the upper triangle of the metric tensor G = cell^T cell
            G = cell.T @ cell
            return torch.stack(
                [
                    G[0, 0],
                    G[1, 1],
                    G[2, 2],
                    G[1, 2],
                    G[0, 2],
                    G[0, 1],
                ]
            )

    @staticmethod
    def _apply_cell_displacement(
        structure,
        dx_cell: torch.Tensor,
        hydrostatic_only: bool,
        max_strain: float = 0.04,
    ):
        """Apply cell displacement and rescale atomic positions.

        For anisotropic relaxation the displacement is interpreted as a
        small strain δε applied via  cell_new = (I + δε) @ cell_old,
        and atomic *Cartesian* positions are similarly transformed so that
        fractional coordinates are preserved.

        For hydrostatic, a uniform isotropic scaling is applied.
        """
        cell = structure.cell

        if hydrostatic_only:
            # dx_cell is a scalar δV
            V = torch.abs(torch.det(cell))
            dV = dx_cell[0]
            # Scale factor: (1 + dV/V)^(1/3)
            scale = (1.0 + dV / V) ** (1.0 / 3.0)
            scale = scale.clamp(1.0 - max_strain, 1.0 + max_strain)  # safety

            new_cell = cell * scale
            structure.RX = structure.RX * scale
            structure.RY = structure.RY * scale
            structure.RZ = structure.RZ * scale
        else:
            # dx_cell is (6,) Voigt strain: εxx, εyy, εzz, εyz, εxz, εxy
            eps = torch.zeros(3, 3, device=cell.device, dtype=cell.dtype)
            eps[0, 0] = dx_cell[0]
            eps[1, 1] = dx_cell[1]
            eps[2, 2] = dx_cell[2]
            eps[1, 2] = eps[2, 1] = 0.5 * dx_cell[3]
            eps[0, 2] = eps[2, 0] = 0.5 * dx_cell[4]
            eps[0, 1] = eps[1, 0] = 0.5 * dx_cell[5]

            # Clamp strain magnitude for stability
            eps_norm = torch.linalg.norm(eps)
            if eps_norm > max_strain:
                eps = eps * (max_strain / eps_norm)

            deform = torch.eye(3, device=cell.device, dtype=cell.dtype) + eps
            new_cell = deform @ cell

            # Transform positions: R_new = deform @ R_old
            R = torch.stack([structure.RX, structure.RY, structure.RZ], dim=-1)
            R = (deform @ R.T).T
            structure.RX, structure.RY, structure.RZ = R[:, 0], R[:, 1], R[:, 2]

        structure.cell = new_cell
        structure.cell_inv = torch.linalg.inv(new_cell)

        # Wrap positions
        R = torch.stack([structure.RX, structure.RY, structure.RZ], dim=-1)
        R = wrap_positions(R, structure.cell, structure.cell_inv)
        structure.RX, structure.RY, structure.RZ = R.unbind(dim=-1)

        # Rebuild PME data if needed
        if (
            hasattr(structure, "_pme_rebuild_callback")
            and structure._pme_rebuild_callback
        ):
            structure._pme_rebuild_callback()


__all__ = ["GeoOpt"]
