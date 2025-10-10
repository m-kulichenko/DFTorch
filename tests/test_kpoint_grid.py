"""
Unit tests for KPointGrid implementation.

Tests k-point generation, weights, reciprocal lattice calculations, and various edge cases.
"""

import torch
import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dftorch.KPointGrid import KPointGrid, create_kpoint_grid, create_gamma_point


class TestKPointGrid:
    """Test suite for KPointGrid class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = "cpu"
        self.dtype = torch.float64
        self.tol = 1e-10
        
        # Test lattices
        self.cubic_lattice = torch.tensor([5.0, 5.0, 5.0], dtype=self.dtype)
        self.orthorhombic_lattice = torch.tensor([[3.0, 0.0, 0.0],
                                                 [0.0, 4.0, 0.0],
                                                 [0.0, 0.0, 6.0]], dtype=self.dtype)
        
    def test_cubic_lattice_initialization(self):
        """Test initialization with cubic lattice."""
        kgrid = KPointGrid(self.cubic_lattice, 
                          grid_type="monkhorst_pack",
                          grid_params=[2, 2, 2])
        
        assert kgrid.num_kpoints == 8
        assert kgrid.lattice_vectors.shape == (3, 3)
        assert kgrid.reciprocal_vectors.shape == (3, 3)
        
        # Check that lattice vectors are diagonal for cubic case
        expected_lattice = torch.diag(self.cubic_lattice)
        torch.testing.assert_close(kgrid.lattice_vectors, expected_lattice)
    
    def test_non_cubic_lattice_initialization(self):
        """Test initialization with non-cubic lattice."""
        kgrid = KPointGrid(self.orthorhombic_lattice,
                          grid_type="monkhorst_pack", 
                          grid_params=[2, 3, 2])
        
        assert kgrid.num_kpoints == 12
        torch.testing.assert_close(kgrid.lattice_vectors, self.orthorhombic_lattice)
    
    def test_reciprocal_lattice_calculation(self):
        """Test reciprocal lattice vector calculation."""
        kgrid = KPointGrid(self.cubic_lattice,
                          grid_type="monkhorst_pack",
                          grid_params=[1, 1, 1])
        
        # For cubic lattice with side length a, reciprocal vectors should be 2π/a * I
        expected_recip = 2 * torch.pi / 5.0 * torch.eye(3, dtype=self.dtype)
        torch.testing.assert_close(kgrid.reciprocal_vectors, expected_recip, atol=self.tol, rtol=0)
        
        # Test reciprocal lattice property: a_i · b_j = 2π δ_ij
        lattice_dot_recip = torch.mm(kgrid.lattice_vectors, kgrid.reciprocal_vectors.T)
        expected_dot = 2 * torch.pi * torch.eye(3, dtype=self.dtype)
        torch.testing.assert_close(lattice_dot_recip, expected_dot, atol=self.tol, rtol=0)
    
    def test_monkhorst_pack_grid_generation(self):
        """Test Monkhorst-Pack k-point generation."""
        # Test 2x2x2 grid
        kgrid = KPointGrid(self.cubic_lattice,
                          grid_type="monkhorst_pack",
                          grid_params=[2, 2, 2])
        
        # Check k-point coordinates
        expected_coords = [-0.25, 0.25]  # (2*i - n - 1)/(2*n) for i=1,2 and n=2
        
        for kpt in kgrid.kpoints:
            for coord in kpt:
                assert coord.item() in expected_coords or abs(coord.item() - (-0.25)) < self.tol or abs(coord.item() - 0.25) < self.tol
        
        # Test 1x1x1 grid (Gamma point only)
        gamma_grid = KPointGrid(self.cubic_lattice,
                               grid_type="monkhorst_pack", 
                               grid_params=[1, 1, 1])
        
        assert gamma_grid.num_kpoints == 1
        torch.testing.assert_close(gamma_grid.kpoints[0], torch.zeros(3, dtype=self.dtype), atol=self.tol, rtol=0)
    
    def test_kpoint_weights_normalization(self):
        """Test that k-point weights sum to 1."""
        for nk in [[2, 2, 2], [3, 3, 3], [4, 2, 1]]:
            kgrid = KPointGrid(self.cubic_lattice,
                              grid_type="monkhorst_pack",
                              grid_params=nk)
            
            weight_sum = torch.sum(kgrid.weights)
            torch.testing.assert_close(weight_sum, torch.tensor(1.0, dtype=self.dtype), atol=self.tol, rtol=0)
    
    def test_explicit_kpoint_grid(self):
        """Test explicit k-point specification."""
        explicit_kpts = torch.tensor([[0.0, 0.0, 0.0],
                                     [0.5, 0.0, 0.0],
                                     [0.0, 0.5, 0.0]], dtype=self.dtype)
        
        kgrid = KPointGrid(self.cubic_lattice,
                          grid_type="explicit",
                          explicit_kpts=explicit_kpts)
        
        assert kgrid.num_kpoints == 3
        torch.testing.assert_close(kgrid.kpoints, explicit_kpts)
        
        # Check weights are equal and normalized
        expected_weight = 1.0 / 3.0
        for weight in kgrid.weights:
            torch.testing.assert_close(weight, torch.tensor(expected_weight, dtype=self.dtype), atol=self.tol, rtol=0)
    
    def test_grid_shift(self):
        """Test k-point grid shifting."""
        shift = torch.tensor([0.1, 0.2, 0.3], dtype=self.dtype)
        kgrid = KPointGrid(self.cubic_lattice,
                          grid_type="monkhorst_pack",
                          grid_params=[2, 2, 2],
                          shift=shift)
        
        # Check that all k-points are shifted
        unshifted_grid = KPointGrid(self.cubic_lattice,
                                   grid_type="monkhorst_pack",
                                   grid_params=[2, 2, 2])
        
        expected_shifted = unshifted_grid.kpoints + shift.unsqueeze(0)
        torch.testing.assert_close(kgrid.kpoints, expected_shifted, atol=self.tol, rtol=0)
    
    def test_cartesian_coordinates(self):
        """Test conversion to Cartesian k-point coordinates."""
        kgrid = KPointGrid(self.cubic_lattice,
                          grid_type="monkhorst_pack",
                          grid_params=[2, 2, 2])
        
        kpts_cart = kgrid.get_kpoints_cartesian()
        assert kpts_cart.shape == (8, 3)
        
        # For cubic lattice, Cartesian = fractional * (2π/a)
        expected_factor = 2 * torch.pi / 5.0
        expected_cart = kgrid.kpoints * expected_factor
        torch.testing.assert_close(kpts_cart, expected_cart, atol=self.tol, rtol=0)
    
    def test_gamma_point_detection(self):
        """Test Gamma point detection."""
        # Grid with Gamma point
        gamma_grid = KPointGrid(self.cubic_lattice,
                               grid_type="monkhorst_pack",
                               grid_params=[1, 1, 1])
        
        gamma_idx = gamma_grid.get_gamma_point_index()
        assert gamma_idx == 0
        
        # Grid without Gamma point
        no_gamma_grid = KPointGrid(self.cubic_lattice,
                                  grid_type="monkhorst_pack",
                                  grid_params=[2, 2, 2])
        
        gamma_idx = no_gamma_grid.get_gamma_point_index()
        assert gamma_idx is None
    
    def test_brillouin_zone_folding(self):
        """Test folding k-points to first Brillouin zone."""
        # Create k-points outside [-0.5, 0.5)
        kpts_outside = torch.tensor([[0.7, -0.8, 1.2],
                                    [-0.6, 0.4, -1.1]], dtype=self.dtype)
        
        kgrid = KPointGrid(self.cubic_lattice,
                          grid_type="explicit",
                          explicit_kpts=kpts_outside)
        
        folded = kgrid.fold_to_first_brillouin_zone()
        
        # Check all coordinates are in [-0.5, 0.5)
        assert torch.all(folded >= -0.5)
        assert torch.all(folded < 0.5)
        
        # Check specific folding results
        expected_folded = torch.tensor([[-0.3, 0.2, 0.2],  # [0.7-1, -0.8+1, 1.2-1]
                                       [0.4, 0.4, -0.1]], dtype=self.dtype)   # [-0.6+1, 0.4, -1.1+1]
        torch.testing.assert_close(folded, expected_folded, atol=self.tol, rtol=0)
    
    def test_high_symmetry_path(self):
        """Test high-symmetry path generation."""
        kgrid = KPointGrid(self.cubic_lattice,
                          grid_type="monkhorst_pack",
                          grid_params=[2, 2, 2])
        
        path_kpts, labels = kgrid.get_high_symmetry_path(['G', 'X', 'M'], num_points=10)
        
        # Note: path includes endpoint, so we get 11 points for 10 requested
        assert path_kpts.shape[0] >= 10
        assert path_kpts.shape[1] == 3
        
        # Check first point is Gamma
        torch.testing.assert_close(path_kpts[0], torch.zeros(3, dtype=self.dtype), atol=self.tol, rtol=0)
        
        # Check last point is M
        expected_M = torch.tensor([0.5, 0.5, 0.0], dtype=self.dtype)
        torch.testing.assert_close(path_kpts[-1], expected_M, atol=self.tol, rtol=0)
    
    def test_convenience_functions(self):
        """Test convenience functions."""
        # Test create_kpoint_grid with integer
        kgrid1 = create_kpoint_grid(self.cubic_lattice, nk=3)
        assert kgrid1.grid_dimensions.tolist() == [3, 3, 3]
        
        # Test create_kpoint_grid with list
        kgrid2 = create_kpoint_grid(self.cubic_lattice, nk=[2, 3, 4])
        assert kgrid2.grid_dimensions.tolist() == [2, 3, 4]
        
        # Test create_gamma_point
        gamma_grid = create_gamma_point()
        assert gamma_grid.num_kpoints == 1
        assert gamma_grid.get_gamma_point_index() == 0
    
    def test_grid_properties(self):
        """Test various grid properties."""
        kgrid = KPointGrid(self.cubic_lattice,
                          grid_type="monkhorst_pack",
                          grid_params=[3, 3, 3])
        
        # Test length
        assert len(kgrid) == 27
        
        # Test indexing
        kpt, weight = kgrid[0]
        assert kpt.shape == (3,)
        assert isinstance(weight.item(), float)
        
        # Test info string generation
        info_str = kgrid.info()
        assert "Grid type: monkhorst_pack" in info_str
        assert "Number of k-points: 27" in info_str
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Invalid LBox shape
        with pytest.raises(ValueError):
            KPointGrid(torch.tensor([1, 2]), grid_type="monkhorst_pack", grid_params=[2, 2, 2])
        
        # Missing grid_params for Monkhorst-Pack
        with pytest.raises(ValueError):
            KPointGrid(self.cubic_lattice, grid_type="monkhorst_pack")
        
        # Missing explicit_kpts for explicit grid
        with pytest.raises(ValueError):
            KPointGrid(self.cubic_lattice, grid_type="explicit")
        
        # Unknown grid type
        with pytest.raises(ValueError):
            KPointGrid(self.cubic_lattice, grid_type="unknown")
        
        # Coplanar lattice vectors (zero volume)
        coplanar_lattice = torch.tensor([[1.0, 0.0, 0.0],
                                        [2.0, 0.0, 0.0],
                                        [3.0, 0.0, 0.0]])
        with pytest.raises(ValueError):
            KPointGrid(coplanar_lattice, grid_type="monkhorst_pack", grid_params=[2, 2, 2])
    
    def test_different_devices_and_dtypes(self):
        """Test different devices and data types."""
        # Test different dtypes
        for dtype in [torch.float32, torch.float64]:
            kgrid = KPointGrid(self.cubic_lattice,
                              grid_type="monkhorst_pack",
                              grid_params=[2, 2, 2],
                              dtype=dtype)
            
            assert kgrid.kpoints.dtype == dtype
            assert kgrid.weights.dtype == dtype
        
        # Test GPU if available
        if torch.cuda.is_available():
            kgrid_gpu = KPointGrid(self.cubic_lattice,
                                  grid_type="monkhorst_pack",
                                  grid_params=[2, 2, 2],
                                  device="cuda")
            
            assert kgrid_gpu.kpoints.device.type == "cuda"
            assert kgrid_gpu.weights.device.type == "cuda"


def run_manual_tests():
    """Run manual tests and examples."""
    print("Running manual tests for KPointGrid...")
    
    # Test basic functionality
    print("\n=== Basic 2x2x2 Monkhorst-Pack Grid ===")
    LBox = torch.tensor([5.0, 5.0, 5.0])
    kgrid = create_kpoint_grid(LBox, nk=2)
    print(f"Number of k-points: {kgrid.num_kpoints}")
    print(f"K-points (fractional):\n{kgrid.kpoints}")
    print(f"Weights: {kgrid.weights}")
    print(f"Gamma point index: {kgrid.get_gamma_point_index()}")
    
    # Test Cartesian coordinates
    print(f"K-points (Cartesian):\n{kgrid.get_kpoints_cartesian()}")
    
    # Test high-symmetry path
    print("\n=== High-Symmetry Path ===")
    path_kpts, labels = kgrid.get_high_symmetry_path(['G', 'X', 'M', 'G'], num_points=12)
    print(f"Path shape: {path_kpts.shape}")
    print("Special points:")
    for i, label in enumerate(labels):
        if label:
            print(f"  {label}: {path_kpts[i].tolist()}")
    
    # Test info
    print("\n=== Grid Information ===")
    print(kgrid.info())
    
    print("Manual tests completed successfully!")


if __name__ == "__main__":
    # Run manual tests if executed directly
    run_manual_tests()
    
    # Run automated tests using pytest
    print("\nRunning automated tests...")
    pytest.main([__file__, "-v"])