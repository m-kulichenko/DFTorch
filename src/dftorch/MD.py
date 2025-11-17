import torch
from sedacs.ewald import calculate_PME_ewald, init_PME_data, calculate_alpha_and_num_grids, ewald_energy
from sedacs.neighbor_list import NeighborState, calculate_displacement
from .Fermi_PRT import Canon_DM_PRT, Fermi_PRT


def initialize_velocities(structure, temperature_K, masses_amu=None, remove_com=True, rescale_to_T=True, remove_angmom=False, generator=None):
    """
    Initialize atomic velocities at a target temperature using a Maxwell–Boltzmann distribution.
    - Velocities are in Å/fs (consistent with EKIN = 0.5 * MVV2KE * sum(m_i * v_i^2) in eV).
    - Ensures zero center-of-mass momentum.
    - Optionally removes net angular momentum (about the COM) without changing coordinates.
    - Rescales to match the exact target temperature.

    Args:
        structure: object with RX (device/dtype) and Mnuc (amu) or provide masses_amu.
        temperature_K (float): target temperature in Kelvin.
        masses_amu (torch.Tensor, optional): shape (N,), atomic masses in amu. Defaults to structure.Mnuc.
        remove_com (bool): enforce zero total momentum.
        rescale_to_T (bool): rescale to match exact temperature after constraints.
        remove_angmom (bool): remove net angular momentum (after COM removal).
        generator (torch.Generator, optional): for reproducibility.

    Returns:
        VX, VY, VZ: tensors of shape (N,) in Å/fs.
    """
    device = structure.RX.device
    dtype = structure.RX.dtype
    N = structure.Nats

    if masses_amu is None:
        masses_amu = structure.Mnuc.to(device=device, dtype=dtype)
    else:
        masses_amu = masses_amu.to(device=device, dtype=dtype)

    # Positive-mass mask
    mpos = masses_amu > 0
    npos = int(mpos.sum().item())

    # Physical constants
    kB_eV_per_K = 8.617333262145e-5  # eV/K
    amu_kg = 1.66053906660e-27
    ang2m = 1e-10
    fs2s = 1e-15
    eC = 1.602176634e-19
    MVV2KE = (amu_kg * (ang2m / fs2s) ** 2) / eC  # ≈ 103.642691 eV per (amu * (Å/fs)^2)

    # Sampling stddev (0 for zero-mass)
    kT = torch.as_tensor(kB_eV_per_K * temperature_K, device=device, dtype=dtype)
    inv_m = torch.where(mpos, 1.0 / masses_amu, torch.zeros_like(masses_amu))
    std = torch.sqrt(torch.clamp(kT * inv_m / MVV2KE, min=0)).to(dtype)

    # Sample velocities
    g = generator if generator is not None else None
    V = torch.randn((N, 3), device=device, dtype=dtype, generator=g) * std[:, None]

    # Remove center-of-mass velocity
    if remove_com and npos > 1:
        Mtot = masses_amu[mpos].sum()
        v_cm = (masses_amu[:, None] * V).sum(dim=0) / Mtot
        V = V - v_cm[None, :]

    # Optionally remove net angular momentum (about COM, without changing coordinates)
    if remove_angmom and npos > 1:
        # Positions
        R = torch.stack((structure.RX.to(dtype), structure.RY.to(dtype), structure.RZ.to(dtype)), dim=1)  # (N,3)
        Mtot = masses_amu[mpos].sum()
        r_com = (masses_amu[:, None] * R).sum(dim=0) / Mtot
        r_rel = R - r_com[None, :]

        # Angular momentum L = sum m r x v  (after COM removed)
        L = (masses_amu[:, None] * torch.cross(r_rel, V, dim=1))[mpos].sum(dim=0)  # (3,)

        # Inertia tensor I = sum m (r^2 I - r r^T)
        eye3 = torch.eye(3, dtype=dtype, device=device)
        r2 = (r_rel[mpos] * r_rel[mpos]).sum(dim=1)  # (npos,)
        I = (masses_amu[mpos][:, None, None] * (r2[:, None, None] * eye3 - r_rel[mpos][:, :, None] * r_rel[mpos][:, None, :])).sum(dim=0)  # (3,3)

        # Solve I * omega = L; use pinv for robustness
        # If I is near-singular (e.g., colinear atoms), pinv safely handles it.
        omega = torch.linalg.pinv(I) @ L  # (3,)

        # Remove rotation: v <- v + r x omega  (since r x omega = - omega x r)
        if npos > 0:
            deltaV = torch.cross(r_rel[mpos], omega.expand_as(r_rel[mpos]), dim=1)  # (npos,3)
            V[mpos] = V[mpos] + deltaV

    # Rescale to target temperature using DOF of massive atoms minus constraints
    if rescale_to_T and npos > 0:
        dof = 3 * npos
        if remove_com and npos > 1:
            dof -= 3
        if remove_angmom and npos > 2:
            dof -= 3
        if dof > 0:
            EKIN = 0.5 * MVV2KE * (masses_amu[:, None] * (V * V)).sum()
            T_cur = (2.0 / dof) * (EKIN / kB_eV_per_K)
            if T_cur > 0:
                scale = torch.sqrt(torch.as_tensor(temperature_K, device=device, dtype=dtype) / T_cur)
                V = V * scale

    return V[:, 0].contiguous(), V[:, 1].contiguous(), V[:, 2].contiguous()


@torch.compile(fullgraph=False, dynamic=False)
def calc_dq(U, v, d_CoulPot, S, Z, Te, Q, e, mu0, Nats, atom_ids):
	d_Hcoul_diag = U * v + d_CoulPot
	d_Hcoul = 0.5 * (d_Hcoul_diag.unsqueeze(1) * S + S * d_Hcoul_diag.unsqueeze(0))
	H1_orth = Z.T @ d_Hcoul @ Z
	# First-order density response (canonical Fermi PRT)
	#_, D1 = Canon_DM_PRT(H1_orth, structure.Te, Q, e, mu0, 10)
	_, D1 = Fermi_PRT(H1_orth, Te, Q, e, mu0)
	D1 = Z @ D1 @ Z.T
	D1S = 2 * torch.diag(D1 @ S)
	# dq (atomic) from AO response
	dq = torch.zeros(Nats, dtype=S.dtype, device=S.device)
	dq.scatter_add_(0, atom_ids, D1S)
	return dq


def kernel_update_lr(structure, MaxRank, KK0, Res, q, FelTol,
					 S,Z,
					 nbr_inds,disps,dists,CALPHA,dftorch_params,PME_data,
					 atom_ids,
					 Q, e, mu0
):
	vi = torch.zeros(structure.Nats,MaxRank, device=S.device)
	fi = torch.zeros(structure.Nats,MaxRank, device=S.device)
	# Preconditioned residual
	K0Res = KK0 @ Res
	dr = K0Res.clone()
	I = 0
	Fel = torch.tensor(float('inf'), dtype=q.dtype, device=q.device)

	while (I < MaxRank) and (Fel > FelTol):
		# Normalize current direction

		norm_dr = torch.norm(dr)
		if norm_dr < 1e-8:
			print('zero norm_dr')
			break
		vi[:, I] = dr / norm_dr
		# Modified Gram-Schmidt against previous vi
		if I > 0:
			# vi[:, I] = vi[:, I] - Vprev @ (Vprev.T @ vi[:, I])
			Vprev = vi[:, :I]                        # (Nats, I)
			vi[:, I] = vi[:, I] - Vprev @ (Vprev.T @ vi[:, I])
		norm_vi = torch.norm(vi[:, I])
		if norm_vi < 1e-8:
			print('zero norm_vi')
			break
		vi[:, I] = vi[:, I] / norm_vi
		v = vi[:, I].clone()  # current search direction


		# dHcoul from a unit step along v (atomic) mapped to AO via atom_ids
		_, _, d_CoulPot =  calculate_PME_ewald(torch.stack((structure.RX, structure.RY, structure.RZ)),
					v,
					structure.lattice_vecs,
					nbr_inds,
					disps,
					dists,
					CALPHA,
					dftorch_params['cutoff'],
					PME_data,
					hubbard_u = structure.Hubbard_U,
					atomtypes = structure.TYPE,
					screening = 1,
					calculate_forces=0,
					calculate_dq=1,
				)

		dq = calc_dq(structure.Hubbard_U[atom_ids], v[atom_ids], d_CoulPot[atom_ids], S, Z, structure.Te, Q, e, mu0, structure.Nats, atom_ids)

		# New residual (df/dlambda), preconditioned
		dr = dq - v
		dr = KK0 @ dr
		# Store fi column
		fi[:, I] = dr
		# Small overlap O and RHS (vectorized)
		rank_m = I + 1
		F_small = fi[:, :rank_m]                  # (Nats, r)
		O = F_small.T @ F_small                   # (r, r)
		rhs = F_small.T @ K0Res                   # (r,)
		# Solve O Y = rhs (stable) instead of explicit inverse
		Y = torch.linalg.solve(O, rhs)            # (r,)
		# Residual norm in the subspace
		Fel = torch.norm(F_small @ Y - K0Res)
		print("rank: {:}, Fel = {:.6f}".format(I, Fel.item() ))
		I += 1

	# Combine correction: K0Res := V Y
	step = (vi[:, :rank_m] @ Y)

	# ##### Trust region relative to the preconditioned residual
	# base = torch.norm(KK0 @ Res)
	# sn   = torch.norm(step)
	# if sn > 1.25 * base and sn > 0:
	# 	step = step * ((1.25 * base) / sn)
	# #####

	K0Res = step                          # (Nats,)
	del vi, fi, v, Y, 
	return K0Res
