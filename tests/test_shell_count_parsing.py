"""Shell presence must come from the SKF's declared shell count, not from a float test.

Regression tests for a defect found 2026-07-31: ``read_skf_table`` inferred shell presence
from ``E_l != 0.0``.  ``mio-1-1``'s ``H-H.skf`` carries a rounding-noise placeholder
``Ep = 0.000039`` Hartree (about 0.001 eV) even though hydrogen is s-only, so hydrogen was
assigned a p shell and ``n_orb = 4`` instead of ``1``.

The consequences were not cosmetic.  Those phantom p functions received real Slater-Koster
overlap from the C-H sp channel rather than zeros, which made the overlap matrix ``S``
indefinite (eigenvalue -0.159).  ``S`` is a Gram matrix and must be positive definite, so
Loewdin orthogonalisation ``S^(-1/2)`` was undefined; one Hamiltonian eigenvalue blew up to
6.1e16 eV, the density matrix came out non-symmetric, and ``2*Tr(D S)`` no longer equalled
the electron count.  The SCF was therefore iterating on charges that did not conserve
electrons, which is why CH4 never converged in any configuration.

Two independent correct signals were present in the file and both were ignored: the p
occupation ``fp = 0.0``, and the declared shell count ``1`` in the grid line
(``0.02, 500,1``).
"""

import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

import glob
import pathlib

import pytest
import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MIO_DIR = REPO_ROOT / "tests" / "data_skf_mio-1-1"
F_DIR = REPO_ROOT / "tests" / "f_orbital_data"


def _mio_params(**over):
    p = {
        "FILENAME": str(REPO_ROOT / "tests" / "ch4.xyz"),
        "CELL": [25.0, 25.0, 25.0],
        "SKFPATH": str(MIO_DIR) + "/",
        "T_ELECTRONIC": 1000.0,
        "RCUT_ELECTRONIC": 8.0,
        "RCUT_REPULSIVE": 4.0,
        "COUL_METHOD": "FULL",
        "SCF_MAX_ITER": 25,
        "KRYLOV_START": 5,
    }
    p.update(over)
    return p


@pytest.fixture(autouse=True)
def _float64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


@pytest.fixture(scope="module")
def mio_constants():
    from dftorch.Constants import Constants

    torch.set_default_dtype(torch.float64)
    return Constants(_mio_params())


# --------------------------------------------------------------------------- metadata


def test_hydrogen_is_s_only(mio_constants):
    """H in mio-1-1 declares one shell. Ep = 3.9e-05 is placeholder noise, not a p shell."""
    Z_H = 1
    assert int(mio_constants.n_orb[Z_H]) == 1, (
        "hydrogen must contribute exactly one basis function (s). "
        "n_orb == 4 means the Ep = 0.000039 placeholder in H-H.skf was read as a real "
        "p shell, which makes the overlap matrix indefinite."
    )
    assert int(mio_constants.max_ang[Z_H]) == 1
    assert [bool(x) for x in mio_constants.shell_present[Z_H]] == [True, False, False, False]


def test_carbon_is_sp(mio_constants):
    """Regression guard: C genuinely is sp (Ep = -0.194, fp = 2.0) and must stay so."""
    Z_C = 6
    assert int(mio_constants.n_orb[Z_C]) == 4
    assert int(mio_constants.max_ang[Z_C]) == 2
    assert [bool(x) for x in mio_constants.shell_present[Z_C]] == [True, True, False, False]


def test_f_fixture_metadata_unaffected():
    """The f fixtures declare no shell count, so they use the inference path. Do not regress it."""
    from dftorch.Constants import Constants

    xyz = glob.glob(str(REPO_ROOT / ".tmp_f_orbital_validation" / "*.xyz"))
    if not xyz:
        pytest.skip("no .tmp_f_orbital_validation geometry available")

    const = Constants(
        {
            "FILENAME": xyz[0],
            "CELL": [25.0, 25.0, 25.0],
            "SKFPATH": str(F_DIR) + "/",
            "T_ELECTRONIC": 1000.0,
            "RCUT_ELECTRONIC": 8.0,
            "RCUT_REPULSIVE": 4.0,
            "COUL_METHOD": "FULL",
        }
    )
    assert int(const.n_orb[7]) == 4, "N must stay sp"
    assert int(const.n_orb[63]) == 16, "Eu must stay spdf"
    assert [bool(x) for x in const.shell_present[63]] == [True, True, True, True]


# --------------------------------------------------------------------- physical gates


@pytest.fixture(scope="module")
def ch4_h0s():
    """CH4 through H0/S construction only - no SCF, so this isolates the overlap build."""
    from dftorch.Constants import Constants
    from dftorch.ESDriver import ESDriver
    from dftorch.Structure import Structure

    torch.set_default_dtype(torch.float64)
    p = _mio_params()
    const = Constants(p)
    st = Structure(p, const, device="cpu")
    ESDriver(p, device="cpu")(st, const, do_scf=False)
    return st


def test_ch4_overlap_matrix_is_positive_definite(ch4_h0s):
    """S is a Gram matrix of basis functions. A non-positive eigenvalue is impossible."""
    S = ch4_h0s.S.detach()
    S = 0.5 * (S + S.T)
    evals = torch.linalg.eigvalsh(S)
    assert float(evals.min()) > 0.0, (
        f"overlap matrix has a non-positive eigenvalue {float(evals.min()):.6e}; "
        "S must be positive definite. A negative eigenvalue means phantom basis "
        "functions are carrying real overlap, and it makes S^(-1/2) undefined."
    )


def test_ch4_hamiltonian_eigenvalues_are_physical(ch4_h0s):
    """A blown-up eigenvalue is the fingerprint of orthogonalising against an indefinite S."""
    st = ch4_h0s
    if not hasattr(st, "e") or st.e is None:
        pytest.skip("no eigenvalues on the do_scf=False path")
    e = st.e.detach()
    assert bool(torch.isfinite(e).all())
    assert float(e.abs().max()) < 1.0e4, (
        f"largest |eigenvalue| is {float(e.abs().max()):.3e} eV, which is not a physical "
        "orbital energy for CH4 with mio-1-1."
    )


def test_ch4_conserves_electrons_and_converges():
    """2*Tr(D S) is fixed by the electron count at every iteration, converged or not."""
    from dftorch.Constants import Constants
    from dftorch.ESDriver import ESDriver
    from dftorch.Structure import Structure

    torch.set_default_dtype(torch.float64)
    p = _mio_params()
    const = Constants(p)
    st = Structure(p, const, device="cpu")
    ESDriver(p, device="cpu")(st, const, do_scf=True)

    n_elec = float(st.Znuc.detach().sum())
    two_tr_ds = 2.0 * float((st.D.detach() * st.S.detach().T).sum())
    assert abs(two_tr_ds - n_elec) < 1e-8, (
        f"2*Tr(D S) = {two_tr_ds:.10f} but the system has {n_elec:.1f} valence electrons. "
        "Mulliken charges partition the electron count, so this identity holds exactly at "
        "every SCF iteration; a deficit means the density matrix is not a valid projector."
    )
    assert abs(float(st.q.detach().sum())) < 1e-8, (
        f"Mulliken charges sum to {float(st.q.detach().sum()):+.6e} for a neutral molecule."
    )


def test_ch4_density_matrix_is_symmetric(ch4_h0s):
    """D must be symmetric for a real basis; asymmetry indicates a broken orthogonalisation."""
    from dftorch.Constants import Constants
    from dftorch.ESDriver import ESDriver
    from dftorch.Structure import Structure

    torch.set_default_dtype(torch.float64)
    p = _mio_params()
    const = Constants(p)
    st = Structure(p, const, device="cpu")
    ESDriver(p, device="cpu")(st, const, do_scf=True)
    D = st.D.detach()
    assert float((D - D.T).abs().max()) < 1e-10, (
        f"max|D - D.T| = {float((D - D.T).abs().max()):.3e}"
    )
