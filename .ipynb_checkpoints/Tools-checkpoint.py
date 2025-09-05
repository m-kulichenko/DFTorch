import torch
def fractional_matrix_power_symm(A, power):
    print('  Doing fractional_matrix_power_symm')
    # Symmetric fractional matrix power using eigendecomposition
    eigvals, eigvecs = torch.linalg.eigh(A)
    eigvals_clamped = torch.clamp(eigvals, min=1e-12)
    D_power = torch.diag(eigvals_clamped ** power)
    return eigvecs @ D_power @ eigvecs.T