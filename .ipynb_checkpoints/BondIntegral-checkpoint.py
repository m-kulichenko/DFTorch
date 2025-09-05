import torch
import pandas as pd

def LoadBondIntegralParameters(neighbor_I, neighbor_J, TYPE, fname):
    """
    Loads bond integral parameters for given atomic pairs from a file.

    Parameters:
    - neighbor_I (Tensor[int]): Tensor of atom indices representing the first atom in each bonded pair.
    - neighbor_J (Tensor[int]): Tensor of atom indices representing the second atom in each bonded pair.
    - TYPE (Tensor[int]): Tensor mapping atom indices to atomic types (e.g., H=0, C=1, etc.).
    - fname (str): Path to the CSV-like text file containing bond integral parameters.

    File Format:
    - The file must be comma-separated with a header row containing "id1,id2,...".
    - Each subsequent row must contain:
        - id1 (int): Atomic type of the first atom.
        - id2 (int): Atomic type of the second atom.
        - 14 float values: The bond integral parameters associated with the (id1, id2) pair.

    Returns:
    - fss_sigma (Tensor[float]): Tensor of shape `(len(neighbor_I), 14)` containing
      bond integral parameters for each pair of neighboring atoms defined by `neighbor_I` and `neighbor_J`.

    Notes:
    - The function constructs a (m+1, m+1, 14) lookup tensor, where m is the highest atomic type in TYPE.
    - Only pairs present in the provided TYPE set are considered when loading data from the file.
    """
    
    type_I = TYPE[neighbor_I]
    type_J = TYPE[neighbor_J]
    m=max(TYPE)
    q=torch.zeros((m+1,m+1, 14), device=neighbor_I.device)
    f=open(fname)
    TYPE_set = set(TYPE.cpu().numpy())
    for l in f:
        t=l.strip().replace(' ', '').split(',')
        if t[0] == 'id1':
            continue
            
        id1=int(t[0])
        id2=int(t[1])
        if id1 in TYPE_set and id2 in TYPE_set:
            q[id1,id2] = torch.tensor(list(map(float, t[2:16])), dtype=q.dtype)
    
    fss_sigma = q[type_I, type_J]
    f.close()
    return fss_sigma

def bond_integral_vectorized(dR: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """
    Computes bond integrals in a fully vectorized form using distance-dependent
    piecewise functions defined by bond integral parameters.

    Parameters:
    - dR (Tensor): Tensor of shape (N,) containing interatomic distances.
    - f (Tensor): Tensor of shape (N, 14) where each row contains the 14 bond
      integral parameters for a corresponding pair of atoms.

    Returns:
    - Tensor of shape (N,): The computed bond integral values for each distance in dR.

    Description:
    The function evaluates the bond integral between atomic pairs using three
    regions defined by the parameter thresholds `f[:,6]` (R1) and `f[:,7]` (R2):

    Region 1 (`dR <= f[:,6]`):
        The bond integral is computed using an exponential of a quartic polynomial:
        X = exp((dR - f[:,5]) * (f[:,1] + ... + f[:,4]*(dR - f[:,5])^3))

    Region 2 (`f[:,6] < dR < f[:,7]`):
        A smooth quintic polynomial interpolation is used.

    Region 3 (`dR >= f[:,7]`):
        The bond integral is set to zero (no interaction beyond this range).

    Finally, the bond integral is scaled by `f[:,0]` as a prefactor.
    """
    # Masks
    region1 = (dR > 1e-12 )&(dR <= f[:,6])
    region2 = (dR > f[:,6]) & (dR < f[:,7])
    region3 = dR >= f[:,7]

    # Output tensor
    X = torch.zeros_like(dR, dtype=f.dtype)

    # Region 1: Polynomial + exp
    RMOD = dR[region1] - f[region1,5]
    POLYNOM = RMOD * (f[region1,1] + RMOD * (f[region1,2] + RMOD * (f[region1,3] + f[region1,4] * RMOD)))
    X[region1] = torch.exp(POLYNOM)

    # Region 2: Quintic polynomial
    RMINUSR1 = dR[region2] - f[region2,6]
    
    X[region2] = f[region2,8] + RMINUSR1 * (
        f[region2,9] + RMINUSR1 * (
            f[region2,10] + RMINUSR1 * (
                f[region2,11] + RMINUSR1 * (
                    f[region2,12] + RMINUSR1 * f[region2,13]
                )
            )
        )
    )
    # Region 3 stays zero
    return f[:,0] * X

def bond_integral_with_grad_vectorized(dR: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """
    Computes derivatives of bond integrals (dX/dr) in a fully vectorized form using distance-dependent
    piecewise functions defined by bond integral parameters.

    Parameters:
    - dR (Tensor): Tensor of shape (N,) containing interatomic distances.
    - f (Tensor): Tensor of shape (N, 14) where each row contains the 14 bond
      integral parameters for a corresponding pair of atoms.

    Returns:
    - Tensor of shape (N,): The computed derivatives of bond integral values for each distance in dR.

    """
    # Masks
    region1 = (dR > 1e-12 )&(dR <= f[:,6])
    region2 = (dR > f[:,6]) & (dR < f[:,7])
    region3 = dR >= f[:,7]

    # Output tensor
    X = torch.zeros_like(dR, dtype=f.dtype)
    dSx = torch.zeros_like(dR, dtype=f.dtype)

    # Region 1: Polynomial + exp
    RMOD = dR[region1] - f[region1,5]
    POLYNOM = RMOD * (f[region1,1] + RMOD * (f[region1,2] + RMOD * (f[region1,3] + f[region1,4] * RMOD)))
    
    X[region1] = torch.exp(POLYNOM)
    
    
    dSx[region1] = X[region1]*(f[region1,1] + 2*RMOD*f[region1,2] + 3*(RMOD**2)*f[region1,3] + 4*(RMOD**3)*f[region1,4])

    # Region 2: Quintic polynomial
    RMINUSR1 = dR[region2] - f[region2,6]
    
    X[region2] = f[region2,8] + RMINUSR1 * (
        f[region2,9] + RMINUSR1 * (
            f[region2,10] + RMINUSR1 * (
                f[region2,11] + RMINUSR1 * (
                    f[region2,12] + RMINUSR1 * f[region2,13]
                )
            )
        )
    )
    
    dSx[region2] = (f[region2,9] + 2*RMINUSR1*f[region2,10] + 3*(RMINUSR1**2)*f[region2,11] + 4*(RMINUSR1**3)*f[region2,12] + 5*(RMINUSR1**4)*f[region2,13])
    
    # Region 3 stays zero
    return f[:,0] * dSx