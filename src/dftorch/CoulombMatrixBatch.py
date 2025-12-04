import torch
import math
import time
from typing import Optional, Tuple, List, Union

def CoulombMatrix_vectorized_batch(Hubbard_U, TYPE, RX, RY, RZ, LBox, lattice_vecs, Nr_atoms,
                  Coulomb_acc, nnRx, nnRy, nnRz, nnType, neighbor_I, neighbor_J,
                  CALPHA, verbose=False):
    Ra = torch.stack((RX.unsqueeze(-1), RY.unsqueeze(-1), RZ.unsqueeze(-1)), dim=-1)
    Rb = torch.stack((nnRx, nnRy, nnRz), dim=-1)
    Rab = Rb - Ra
    dR = torch.norm(Rab, dim=-1)
    dR_dxyz = Rab/dR.unsqueeze(-1)

    CC_real, dCC_dxyz_real = Ewald_Real_Space_vectorized_batch(
                                Hubbard_U, TYPE,
                                dR, dR_dxyz, 
                                nnType, neighbor_I, neighbor_J, CALPHA)
    
    dq_J = torch.zeros(Nr_atoms, dtype=dR.dtype, device = dR.device)
    CC_k, dCC_dR_k = Ewald_k_Space_vectorized(RX, RY, RZ, LBox, lattice_vecs, dq_J, Nr_atoms, Coulomb_acc, CALPHA, verbose)

    CC = CC_real + CC_k
    dCC_dxyz = dCC_dxyz_real + dCC_dR_k

    return CC, -dCC_dxyz

def Ewald_Real_Space_vectorized_batch(Hubbard_U, TYPE, dR, dR_dxyz, nnType, neighbor_I, neighbor_J,
                  CALPHA):
    
    batch_size = Hubbard_U.shape[0]
    Nats = TYPE.shape[-1]
    valid_pairs = (neighbor_I >= 0) & (neighbor_J >= 0)
    safe_I = neighbor_I.clamp(min=0)
    safe_J = neighbor_J.clamp(min=0)
    
    # Constants
    CALPHA2 = CALPHA ** 2
    RELPERM = 1.0
    KECONST = 14.3996437701414 * RELPERM
    TFACT = 16.0 / (5.0 * KECONST)
    SQRTPI = math.sqrt(math.pi)
    
    # Pair indices
    nn_mask = nnType!=-1 #& dist_mask # mask to exclude zero padding from the neigh list

    dR_mskd = dR[nn_mask]
    Ti = TFACT * Hubbard_U.gather(1, safe_I)[valid_pairs]
    Tj = TFACT * Hubbard_U.gather(1, safe_J)[valid_pairs]    
    CC_real = torch.zeros((batch_size * Nats * Nats), device=dR.device, dtype=dR.dtype)
    CA = torch.erfc(CALPHA * dR_mskd) / dR_mskd
    tmp1 = CA.clone()
    dtmp1 = -(CA + 2 * CALPHA * torch.exp(-CALPHA2 * dR_mskd**2) / SQRTPI)/dR_mskd

    mask_same_elem = (TYPE.gather(1, safe_I)[valid_pairs] == TYPE.gather(1, safe_J)[valid_pairs])
    if mask_same_elem.any():
        dR_mskd_same = dR_mskd[mask_same_elem]
        Ti_same_el = Ti[mask_same_elem]
        TI2 = Ti_same_el ** 2
        TI3 = TI2 * Ti_same_el
        SSB = TI3 / 48.0
        SSC = 3 * TI2 / 16.0
        SSD = 11 * Ti_same_el / 16.0
        EXPTI = torch.exp(-Ti_same_el * dR_mskd_same)
        tmp = (SSB * dR_mskd_same**2 + SSC * dR_mskd_same + SSD + 1. / dR_mskd_same)
        tmp1[ mask_same_elem] -= EXPTI * tmp
        dtmp1[mask_same_elem] -= EXPTI * (
                (-Ti_same_el) * tmp + (2 * SSB * dR_mskd_same + SSC - 1. / dR_mskd_same**2)
            )
    if (~mask_same_elem).any():
        dR_mskd_diff = dR_mskd[~mask_same_elem]
        Ti_diff_el = Ti[~mask_same_elem]
        Tj_diff_el = Tj[~mask_same_elem]
        TI2 = Ti_diff_el ** 2
        TI4 = TI2 ** 2
        TI6 = TI4 * TI2
        TJ2 = Tj_diff_el ** 2
        TJ4 = TJ2 ** 2
        TJ6 = TJ4 * TJ2
        EXPTI = torch.exp(-Ti_diff_el * dR_mskd_diff)
        EXPTJ = torch.exp(-Tj_diff_el * dR_mskd_diff)
        TI2MTJ2 = TI2 - TJ2
        TJ2MTI2 = -TI2MTJ2
        SB = TJ4 * Ti_diff_el / (2 * TI2MTJ2 ** 2)
        SC = (TJ6 - 3 * TJ4 * TI2) / (TI2MTJ2 ** 3)
        SE = TI4 * Tj_diff_el / (2 * TJ2MTI2 ** 2)
        SF = (TI6 - 3 * TI4 * TJ2) / (TJ2MTI2 ** 3)
        COULOMBV_tmp1 = (SB - SC / dR_mskd_diff)
        COULOMBV_tmp2 = (SE - SF / dR_mskd_diff)
        tmp1[ ~mask_same_elem] -= EXPTI * COULOMBV_tmp1 + EXPTJ * COULOMBV_tmp2
        dtmp1[~mask_same_elem] -= EXPTI * ((-Ti_diff_el)*COULOMBV_tmp1 + SC/dR_mskd_diff**2) + \
                EXPTJ * ((-Tj_diff_el)*COULOMBV_tmp2 + SF/dR_mskd_diff**2)
    
    tmp1 *= KECONST
    dtmp1 *= KECONST
    batch_ids = torch.arange(batch_size, device=dR.device).unsqueeze(1).expand_as(Hubbard_U)
    batch_ids = batch_ids.gather(1, safe_J)[valid_pairs]
    batch_block_offset = batch_ids * (Nats * Nats)
    CC_real.index_add_(0, safe_I[valid_pairs]*(Nats) + safe_J[valid_pairs] + batch_block_offset, tmp1 )
    CC_real = CC_real.reshape(batch_size, Nats, Nats)

    dCC_dxyz_real = torch.zeros((3, batch_size*Nats*Nats), device=dR.device, dtype=dR.dtype)
    dCC_dxyz_real.index_add_(1, safe_I[valid_pairs]*(Nats) + safe_J[valid_pairs] + batch_block_offset, dtmp1*dR_dxyz[nn_mask].T  )
    dCC_dxyz_real = dCC_dxyz_real.view(3, batch_size, Nats,Nats)
    dCC_dxyz_real = dCC_dxyz_real.permute(1, 0, 2, 3).contiguous() 
    return CC_real, dCC_dxyz_real

@torch.compile(dynamic=False)
def Ewald_k_Space_vectorized(RX, RY, RZ, LBox, lattice_vecs, DELTAQ, Nr_atoms, COULACC, CALPHA, verbose, do_vec=False):
    
    batch_size = RX.shape[0]
    device = RX.device

    COULVOL = torch.abs(torch.det(lattice_vecs))
    SQRTX = math.sqrt(-math.log(COULACC))

    CALPHA2 = CALPHA * CALPHA
    KCUTOFF = 2 * CALPHA * SQRTX
    KCUTOFF2 = KCUTOFF * KCUTOFF

    RECIPVECS = torch.zeros((batch_size, 3, 3), dtype=RX.dtype, device=device)
    RECIPVECS[:, 0, 0] = 2 * math.pi / LBox[:, 0]
    RECIPVECS[:, 1, 1] = 2 * math.pi / LBox[:, 1]
    RECIPVECS[:, 2, 2] = 2 * math.pi / LBox[:, 2]

    LMAX = (KCUTOFF / RECIPVECS[:,0, 0]).int()
    MMAX = (KCUTOFF / RECIPVECS[:,1, 1]).int()
    NMAX = (KCUTOFF / RECIPVECS[:,2, 2]).int()

    KECONST = 14.3996437701414  # in eV·Å/e²
    SQRTPI = math.sqrt(math.pi)

    COULOMBV = torch.zeros((batch_size, Nr_atoms, Nr_atoms), dtype=RX.dtype, device=device)
    
    dC_dR = torch.zeros((batch_size, 3, Nr_atoms, Nr_atoms), dtype=RX.dtype, device=device)

    if do_vec:
        # Create meshgrid of all combinations
        print('vectorized k-space is not implemented for batched data')
        return
    else:
        #if verbose: print('   LMAX:', LMAX)
        print('   LMAX:', LMAX)
        for L in range(0, torch.max(LMAX) + 1):
            if verbose: print('  ',L)
            MMIN = 0 if L == 0 else -torch.max(MMAX)
            for M in range(MMIN, torch.max(MMAX) + 1):
                NMIN = 1 if (L == 0 and M == 0) else -torch.max(NMAX)
                for N in range(NMIN, torch.max(NMAX) + 1):
                    kvec = L * RECIPVECS[:, :, 0] + M * RECIPVECS[:, :, 1] + N * RECIPVECS[:, :, 2]
                    K2 = (kvec * kvec).sum(dim=1) # similar to K2 = torch.dot(kvec, kvec) for non-batched
                    
                    cutoff_mask = K2 > KCUTOFF2
                    if cutoff_mask.all():
                        continue
                    
                    #print(K2, KCUTOFF2)
                    exp_factor = torch.exp(-K2 / (4 * CALPHA2))
                    prefactor = 8 * math.pi * exp_factor / (COULVOL * K2)
                    KEPREF = 14.3996437701414 * prefactor  # KECONST in eV·Å/e²
                    
                    dot = torch.matmul(kvec.view(batch_size, 1, 3), torch.stack((RX, RY, RZ), dim=1)).squeeze(0)
                    sin_list = torch.sin(dot)
                    cos_list = torch.cos(dot)

                    # Use broadcasting for outer products
                    sin_i = sin_list.view(batch_size, -1, 1)
                    sin_j = sin_list.view(batch_size, 1, -1)
                    cos_i = cos_list.view(batch_size, -1, 1)
                    cos_j = cos_list.view(batch_size, 1, -1)

                    COULOMBV[~cutoff_mask] += (KEPREF.unsqueeze(-1).unsqueeze(-1) * (cos_i * cos_j + sin_i * sin_j))[~cutoff_mask]
                    force_term = (KEPREF.unsqueeze(-1).unsqueeze(-1) * (-cos_i * sin_j + sin_i * cos_j))
                   
                    dC_dR[~cutoff_mask] += (force_term.unsqueeze(1) * kvec.view(batch_size, 3, 1, 1))[~cutoff_mask]
    
    # Self-interaction correction
    DELTAQ_vec = torch.eye(Nr_atoms, device = device)
    CORRFACT = 2 * KECONST * CALPHA / SQRTPI
    COULOMBV -= CORRFACT*DELTAQ_vec
    return COULOMBV, dC_dR

