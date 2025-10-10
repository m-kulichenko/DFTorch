# K-Space Integration Roadmap for DFTorch

## Overview
This roadmap outlines the implementation of Brillouin zone k-point sampling for periodic boundary conditions in DFTorch. Currently, DFTorch operates in real-space (Γ-point only). Adding k-space integration will enable proper treatment of periodic systems with accurate band structure calculations.

## Current State Analysis

**What exists:**
- Real-space tight-binding Hamiltonian construction (`H0andS.py`, `SlaterKosterPair.py`)
- Reciprocal-space Ewald summation for Coulomb interactions (`CoulombMatrix.py`)
- Neighbor list generation (`nearestneighborlist.py`)
- SCF solver for density matrix (`SCF.py`, `DM_Fermi.py`)

**What's missing:**
- K-point grid generation and management
- Bloch sum construction of H(k) and S(k)
- K-space integration for density matrix
- Periodic boundary conditions for tight-binding matrices
- Band structure calculation capabilities

---

## Phase 1: Foundation and K-Point Grid Infrastructure

### 1.1 K-Point Grid Generation
**New file:** `src/dftorch/KPointGrid.py`

```python
class KPointGrid:
    def __init__(self, LBox, grid_density=None, explicit_kpts=None):
        """
        Generate k-point grids for periodic systems
        - Monkhorst-Pack grids
        - Explicit k-point lists
        - Symmetry reduction (future)
        """
    
    def generate_monkhorst_pack(self, nk1, nk2, nk3):
        """Generate MP grid with specified dimensions"""
    
    def get_reciprocal_lattice(self, LBox):
        """Calculate reciprocal lattice vectors from real-space box"""
    
    def get_kpoint_weights(self):
        """Return k-point weights for integration"""
```

**Tasks:**
- [ ] Implement Monkhorst-Pack k-point generation
- [ ] Add support for explicit k-point lists
- [ ] Calculate k-point weights properly
- [ ] Handle reciprocal lattice vector calculation
- [ ] Add k-point density-based automatic grid selection

### 1.2 Periodic Boundary Condition Framework
**Modify:** `src/dftorch/nearestneighborlist.py`

```python
def vectorized_nearestneighborlist_periodic(RX, RY, RZ, LBox, Rcut, Nats, 
                                          include_images=True, max_images=3):
    """
    Extended neighbor list including periodic images
    Returns:
    - neighbor_I, neighbor_J: atom indices
    - nnRx, nnRy, nnRz: neighbor positions (including images)
    - image_shifts: (3,) integer shifts for each neighbor pair
    """
```

**Tasks:**
- [ ] Extend neighbor list to include periodic images
- [ ] Track lattice vector shifts for each neighbor pair
- [ ] Ensure proper handling of minimum image convention
- [ ] Optimize for computational efficiency with large cutoffs

---

## Phase 2: Bloch Sum Construction

### 2.1 K-Space Hamiltonian Builder
**New file:** `src/dftorch/BlochHamiltonian.py`

```python
class BlochHamiltonian:
    def __init__(self, kpoint_grid, real_space_matrices):
        """
        Construct H(k) and S(k) from real-space matrices via Bloch sums
        """
    
    def build_hk_sk(self, kpoint):
        """
        H(k) = sum_R H(R) * exp(ik·R)
        S(k) = sum_R S(R) * exp(ik·R)
        
        Returns:
        - Hk: (n_orb, n_orb) complex tensor
        - Sk: (n_orb, n_orb) complex tensor
        """
    
    def build_hk_sk_batch(self, kpoints):
        """Vectorized construction for multiple k-points"""
        
    def get_hk_derivatives(self, kpoint):
        """
        dH(k)/dk for force calculations
        """
```

**Tasks:**
- [ ] Implement Bloch sum: `H(k) = Σ_R H(R) exp(ik·R)`
- [ ] Handle complex arithmetic properly in PyTorch
- [ ] Vectorize over multiple k-points efficiently
- [ ] Implement derivatives dH(k)/dk for forces
- [ ] Ensure proper phase factor calculations

### 2.2 Real-Space Matrix Storage
**Modify:** `src/dftorch/H0andS.py`

```python
def H0_and_S_realspace(TYPE, RX, RY, RZ, LBox, Nats, neighbor_data, const):
    """
    Build real-space H(R) and S(R) matrices with lattice vector tracking
    
    Returns:
    - H_realspace: dict mapping lattice vectors R to H(R) matrices
    - S_realspace: dict mapping lattice vectors R to S(R) matrices
    - lattice_vectors: list of R vectors with non-zero matrix elements
    """
```

**Tasks:**
- [ ] Modify H0andS construction to store H(R) for different R vectors
- [ ] Track which lattice vectors contribute to matrix elements
- [ ] Ensure efficient sparse storage for large unit cells
- [ ] Maintain compatibility with existing real-space (R=0) calculations

---

## Phase 3: K-Space SCF and Integration

### 3.1 K-Space Density Matrix Construction
**New file:** `src/dftorch/KSpaceDensity.py`

```python
class KSpaceDensityMatrix:
    def __init__(self, kpoint_grid, temperature):
        """Handle k-space integration for density matrix construction"""
    
    def integrate_density_matrix(self, Hk_list, Sk_list, kpoint_weights, nocc):
        """
        D = (1/Nk) * sum_k w_k * D(k)
        where D(k) = S(k) * C(k) * f(k) * C(k)† * S(k)†
        """
    
    def solve_kspace_eigenvalue(self, Hk, Sk):
        """Solve generalized eigenvalue problem at single k-point"""
        
    def fermi_integration_kspace(self, eigenvalues_k, kpoint_weights, nocc, Te):
        """Determine Fermi level from k-integrated DOS"""
```

**Tasks:**
- [ ] Implement k-space integration: `D = Σ_k w_k D(k)`
- [ ] Handle complex density matrices properly
- [ ] Implement efficient eigenvalue solvers for complex matrices
- [ ] Add k-space Fermi level determination
- [ ] Ensure proper normalization of electron count

### 3.2 Modified SCF Loop
**Modify:** `src/dftorch/SCF.py`

```python
def SCF_kspace(H_realspace, S_realspace, kpoint_grid, Efield, C, TYPE, 
               RX, RY, RZ, H_INDEX_START, H_INDEX_END, Nocc, 
               Hubbard_U, Znuc, Nats, Te, const, **kwargs):
    """
    Self-consistent field solver with k-space integration
    
    Main loop:
    1. Build H(k), S(k) for all k-points
    2. Solve eigenvalue problems at each k-point  
    3. Integrate density matrix over k-space
    4. Update charges and Coulomb potential
    5. Repeat until convergence
    """
```

**Tasks:**
- [ ] Modify SCF loop to handle k-space integration
- [ ] Implement efficient k-point parallelization
- [ ] Add convergence criteria appropriate for k-space calculations
- [ ] Handle memory management for large k-point grids
- [ ] Ensure backward compatibility with Γ-point calculations

---

## Phase 4: Forces and Properties

### 4.1 K-Space Force Calculations
**Modify:** `src/dftorch/Forces.py`

```python
def Forces_kspace(H_realspace, S_realspace, kpoint_grid, density_matrix, 
                  dH_realspace, dS_realspace, **kwargs):
    """
    Force calculation with k-space integration:
    F = -∂E/∂R = -sum_k w_k Tr[D(k) * ∂H(k)/∂R]
    """
```

**Tasks:**
- [ ] Implement k-space force integration
- [ ] Handle derivatives of Bloch phases: ∂[exp(ik·R)]/∂R
- [ ] Ensure force conservation (sum of forces = 0)
- [ ] Add Pulay force corrections for k-space
- [ ] Optimize for computational efficiency

### 4.2 Band Structure and Properties
**New file:** `src/dftorch/BandStructure.py`

```python
class BandStructure:
    def __init__(self, H_realspace, S_realspace):
        """Calculate band structures along high-symmetry paths"""
    
    def calculate_bands(self, kpath, num_points=100):
        """Calculate eigenvalues along k-space path"""
    
    def get_density_of_states(self, kpoint_grid, energy_grid):
        """Calculate k-integrated density of states"""
    
    def get_band_gap(self):
        """Calculate fundamental band gap"""
```

**Tasks:**
- [ ] Implement band structure calculations
- [ ] Add high-symmetry k-point path generation
- [ ] Calculate density of states with proper broadening
- [ ] Add band gap and effective mass calculations
- [ ] Create visualization tools for band structures

---

## Phase 5: Testing and Validation

### 5.1 Test Suite Development
**New directory:** `tests/kspace/`

```python
# Test files to create:
- test_kpoint_grid.py: K-point generation and symmetry
- test_bloch_sum.py: H(k) construction accuracy
- test_kspace_scf.py: SCF convergence with k-points
- test_band_structure.py: Band structure calculations
- test_forces_kspace.py: Force accuracy and conservation
```

**Tasks:**
- [ ] Validate against analytical solutions (1D chains, 2D lattices)
- [ ] Compare with reference DFTB+ calculations
- [ ] Test k-point convergence for various systems
- [ ] Benchmark performance vs. real-space calculations
- [ ] Add unit tests for all major components

### 5.2 Example Systems
**New directory:** `examples/periodic/`

```python
# Example calculations:
- graphene_bands.py: 2D graphene band structure
- carbon_nanotube.py: 1D periodic system
- diamond_bulk.py: 3D bulk semiconductor
- surface_slab.py: 2D slab with vacuum
```

**Tasks:**
- [ ] Create tutorial examples for common periodic systems
- [ ] Document k-point convergence recommendations
- [ ] Add performance benchmarking examples
- [ ] Include visualization scripts for results

---

## Phase 6: Optimization and Advanced Features

### 6.1 Performance Optimization
**Tasks:**
- [ ] GPU acceleration for k-point loops
- [ ] MPI parallelization over k-points
- [ ] Memory-efficient storage for large k-grids
- [ ] Adaptive k-point refinement
- [ ] Symmetry reduction of k-point grids

### 6.2 Advanced Features
**Tasks:**
- [ ] Spin-orbit coupling in k-space
- [ ] Magnetic field effects (Peierls phases)
- [ ] Berry phase and polarization calculations
- [ ] Wannier function construction
- [ ] Transport property calculations

---

## Implementation Timeline

**Phase 1-2 (Months 1-3):** Foundation and Bloch sums
- Core k-point infrastructure
- Basic H(k)/S(k) construction
- Real-space matrix reorganization

**Phase 3 (Months 4-6):** K-space SCF
- Density matrix integration
- Modified SCF loop
- Basic convergence testing

**Phase 4 (Months 7-9):** Forces and properties
- K-space force calculations
- Band structure capabilities
- Property calculations

**Phase 5-6 (Months 10-12):** Testing and optimization
- Comprehensive validation
- Performance optimization
- Advanced features

---

## Technical Considerations

### Memory Management
- K-point calculations can be memory-intensive
- Consider out-of-core algorithms for large systems
- Implement efficient sparse matrix storage

### Numerical Stability
- Complex arithmetic requires careful handling
- Eigenvalue solvers need proper conditioning
- Phase factors must be computed accurately

### Backward Compatibility
- Maintain existing Γ-point functionality
- Add k-space as optional enhancement
- Ensure API consistency

### Performance
- K-point loops are embarrassingly parallel
- GPU acceleration potential is high
- Memory bandwidth often limiting factor

---

## Success Metrics

1. **Functionality:** All existing DFTorch calculations work with k-space
2. **Accuracy:** Band gaps within 0.1 eV of reference calculations
3. **Performance:** <10x slowdown compared to Γ-point for moderate k-grids
4. **Usability:** Simple API for enabling k-space calculations
5. **Validation:** Successful reproduction of known band structures

This roadmap provides a structured approach to implementing full k-space integration while maintaining the current functionality and performance characteristics of DFTorch.