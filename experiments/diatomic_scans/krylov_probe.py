"""Every measurement behind .planning/KRYLOV-INVESTIGATION.md, in one script.

Each test is independent and selected by name, because some are seconds and one
(``tutorial``) is tens of minutes:

    python experiments/diatomic_scans/krylov_probe.py scan      # A  ~4 min
    python experiments/diatomic_scans/krylov_probe.py handover  # B  ~4 min
    python experiments/diatomic_scans/krylov_probe.py jacobian  # C  ~1 min
    python experiments/diatomic_scans/krylov_probe.py chargemap # D  ~1 min
    python experiments/diatomic_scans/krylov_probe.py spectrum  # E  ~1 min
    python experiments/diatomic_scans/krylov_probe.py trustreg  # F  ~12 min
    python experiments/diatomic_scans/krylov_probe.py smear     # G  ~8 min
    python experiments/diatomic_scans/krylov_probe.py temp      # H  ~4 min
    python experiments/diatomic_scans/krylov_probe.py mio       # I  ~2 min
    python experiments/diatomic_scans/krylov_probe.py tutorial  # J  ~40 min
    python experiments/diatomic_scans/krylov_probe.py md        # K  ~2 min
    python experiments/diatomic_scans/krylov_probe.py all

Unlike the other scripts in this directory these print tables rather than dumping
JSON, because the recorded output *is* the deliverable -- it is pasted into the
investigation record.  Nothing here asserts; it measures.

Note the standing decision in this directory's README does not apply here: this
script deliberately turns the Krylov mixer ON, since measuring it is the point.
"""

import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

import contextlib
import io
import re
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
SKF_F = REPO / "tests" / "f_orbital_data"
SKF_MIO = REPO / "tests" / "data_skf_mio-1-1"
SKF_TUT = REPO / "experiments" / "sk_orig" / "mio-1-1" / "mio-1-1"
EXP = REPO / "experiments"
TMP = Path(__file__).resolve().parent / "geom"
TMP.mkdir(exist_ok=True)

#: Krylov never engages: the branch test is ``it > KRYLOV_START`` and the loop
#: caps at SCF_MAX_ITER = 100.  Same sentinel ESDriver._krylov_params_for_f_interim uses.
KRYLOV_OFF = 10**6

_RES_RE = re.compile(r"^Res = ([0-9.eE+-]+), dEc = ([0-9.eE+-]+)", re.M)

EU_N_GRID = [round(1.60 + 0.10 * i, 2) for i in range(21)]


# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #
def diatomic_params(sym_a, sym_b, r, skf, krylov_start, te=1000.0):
    """Driver parameters for a two-atom scan point, matching tests/test_eu_n_scan.py."""
    path = TMP / f"{sym_a}{sym_b}_{r:.3f}.xyz"
    path.write_text(
        f"2\n{sym_a}-{sym_b}\n{sym_a} 0 0 0\n{sym_b} {r:.8f} 0 0\n"
    )
    return {
        "T_ELECTRONIC": te,
        "RCUT_ELECTRONIC": 10.0,
        "RCUT_REPULSIVE": 6.0,
        "COUL_METHOD": "FULL",
        "CHARGE": 0,
        "KRYLOV_START": krylov_start,
        "FILENAME": str(path),
        "SKFPATH": str(skf) + os.sep,
    }


def bulk_params(filename, cell, krylov_start, coul="FULL"):
    """Driver parameters for the tutorial's own systems, using its own values."""
    return {
        "FILENAME": str(EXP / filename),
        "SKFPATH": str(SKF_TUT) + os.sep,
        "CELL": [cell, cell, cell],
        "T_ELECTRONIC": 1000.0,
        "RCUT_ELECTRONIC": 8.0,
        "RCUT_REPULSIVE": 4.0,
        "COUL_METHOD": coul,
        "KRYLOV_START": krylov_start,
        "SCF_MAX_ITER": 100,
    }


def run(params):
    """One driver call with the library chatter captured rather than printed.

    The chatter is the measurement: 'Did not converge' and the per-pass
    'Res = ...' lines are how the loop reports itself, and 'rank:' is printed
    once per Krylov iteration by kernel_update_lr.
    """
    from dftorch.Constants import Constants
    from dftorch.ESDriver import ESDriver
    from dftorch.Structure import Structure

    buf = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        const = Constants(params).to("cpu")
        driver = ESDriver(params, device="cpu")
        structure = Structure(params, const, device="cpu")
        driver(structure, const, do_scf=True)
        e_tot = float(structure.e_tot.item())
        q = [float(x) for x in structure.q.detach().cpu().tolist()]
        nats = int(structure.Nats)
        norb = int(structure.n_orbitals_per_atom.sum())
    text = buf.getvalue()
    residuals = [float(a) for a, _ in _RES_RE.findall(text)]
    return {
        "converged": "Did not converge" not in text,
        "passes": len(residuals),
        "e_tot": e_tot,
        "q": q,
        "nats": nats,
        "norb": norb,
        "residuals": residuals,
        "krylov_iters": text.count("rank:"),
        "wall": time.perf_counter() - t0,
    }


def capture_calc_q_inputs(params):
    """Run once and keep the last arguments handed to calc_q.

    Those arguments -- H0 with the dipole term already folded in, the AO-mapped
    Hubbard U, S, Z, Te, Nocc, Znuc, atom_ids -- are exactly what is needed to
    re-evaluate the charge map q_out(q) outside the loop, which is what tests
    C, D and E all do.
    """
    import dftorch._scf as scf_mod
    from dftorch.Constants import Constants
    from dftorch.ESDriver import ESDriver
    from dftorch.Structure import Structure

    captured = {}
    real_calc_q = scf_mod.calc_q

    def spy(H0, U, n, CoulPot, S, Z, Te, Nocc, Znuc, atom_ids, dU_dq=None):
        captured.update(
            dict(H0=H0, U=U, S=S, Z=Z, Te=Te, Nocc=Nocc, Znuc=Znuc,
                 atom_ids=atom_ids)
        )
        return real_calc_q(H0, U, n, CoulPot, S, Z, Te, Nocc, Znuc, atom_ids,
                           dU_dq)

    scf_mod.calc_q = spy
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            const = Constants(params).to("cpu")
            driver = ESDriver(params, device="cpu")
            structure = Structure(params, const, device="cpu")
            driver(structure, const, do_scf=True)
    finally:
        scf_mod.calc_q = real_calc_q
    return captured, structure


def charge_map(captured, C):
    """Return q_out(x), the Eu charge the loop would produce from an input of x.

    Charge neutrality pins the two atoms to (x, -x), so the whole self-consistency
    problem for a neutral diatomic is this one scalar function.
    """
    from dftorch._xl_tools import calc_q_eager

    H0, U, S, Z = captured["H0"], captured["U"], captured["S"], captured["Z"]
    Te, Nocc, Znuc = captured["Te"], captured["Nocc"], captured["Znuc"]
    aid = captured["atom_ids"]

    def q_out(x):
        qv = torch.tensor([x, -x], dtype=S.dtype)
        pot = C @ qv
        return float(
            calc_q_eager(H0, U, qv[aid], pot[aid], S, Z, Te, Nocc, Znuc, aid,
                         None)[0][0]
        )

    return q_out


def banner(title, *notes):
    print("=" * 88)
    print(title)
    for note in notes:
        print("  " + note)
    print("=" * 88)


# --------------------------------------------------------------------------- #
# A. the reproduction
# --------------------------------------------------------------------------- #
def test_scan():
    banner(
        "A - Eu-N separation scan, Krylov on versus off",
        "Krylov engages at pass KRYLOV_START+1 = 11 by default (_scf.py:690).",
    )
    print(f"  {'r (A)':>6} {'Krylov ON':>32} {'Krylov OFF':>32}")
    n_bad = 0
    for r in EU_N_GRID:
        on = run(diatomic_params("Eu", "N", r, SKF_F, 10))
        off = run(diatomic_params("Eu", "N", r, SKF_F, KRYLOV_OFF))
        n_bad += not on["converged"]
        print(
            f"  {r:6.2f} "
            f"{('OK ' if on['converged'] else 'DIV'):>4}  it={on['passes']:<4} "
            f"E={on['e_tot']:12.6f}  "
            f"{('OK ' if off['converged'] else 'DIV'):>4}  it={off['passes']:<4} "
            f"E={off['e_tot']:12.6f}"
        )
        sys.stdout.flush()
    print(f"\n  diverged with Krylov on: {n_bad}/21;  with Krylov off: 0/21")
    print()


# --------------------------------------------------------------------------- #
# B. when Krylov takes over
# --------------------------------------------------------------------------- #
def test_handover():
    banner(
        "B - does the failure depend on WHEN Krylov takes over?  Eu-N at r = 3.60 A",
        "the handover residual is read off the Krylov-off run at that pass",
    )
    off = run(diatomic_params("Eu", "N", 3.60, SKF_F, KRYLOV_OFF))
    print(f"  {'KRYLOV_START':>13} {'|Res| at handover':>18} {'result':>9} "
          f"{'passes':>7} {'E_tot (eV)':>14}")
    for ks in (10, 12, 14, 16, 18, 20, 22, 24, 26, KRYLOV_OFF):
        rec = run(diatomic_params("Eu", "N", 3.60, SKF_F, ks))
        handover = off["residuals"][ks - 1] if ks <= len(off["residuals"]) else 0.0
        label = "off" if ks == KRYLOV_OFF else str(ks)
        print(f"  {label:>13} {handover:18.3e} "
              f"{('OK' if rec['converged'] else 'DIVERGE'):>9} "
              f"{rec['passes']:>7} {rec['e_tot']:14.6f}")
        sys.stdout.flush()
    print()


# --------------------------------------------------------------------------- #
# C. is the accelerator's arithmetic right?
# --------------------------------------------------------------------------- #
def test_jacobian(r=3.60, max_iter=18):
    banner(
        f"C - is the Krylov step the right step?  Eu-N at r = {r} A",
        "J is the Jacobian of f(q) = q_out(q) - q on the 1-D residual subspace.",
        "J_fd is central finite differences on the real map; J_analytic is calc_dq.",
    )
    import dftorch._scf as scf_mod
    import dftorch._xl_tools as xl
    from dftorch._xl_tools import calc_q_eager
    from dftorch.Constants import Constants
    from dftorch.ESDriver import ESDriver
    from dftorch.Structure import Structure

    captured = {}
    real_calc_q = scf_mod.calc_q
    real_kernel = scf_mod.kernel_update_lr
    log = []

    def calc_q_spy(H0, U, n, CoulPot, S, Z, Te, Nocc, Znuc, atom_ids, dU_dq=None):
        captured.update(dict(H0=H0, U=U, S=S, Z=Z, Te=Te, Nocc=Nocc, Znuc=Znuc,
                             atom_ids=atom_ids))
        return real_calc_q(H0, U, n, CoulPot, S, Z, Te, Nocc, Znuc, atom_ids,
                           dU_dq)

    def kernel_spy(RX, RY, RZ, cell, TYPE, Nats, Hubbard_U, params, FelTol, KK0,
                   Res, q, S, Z, PME_data, atom_ids, Q, e, mu0, Te, C=None,
                   *args, **kwargs):
        step = real_kernel(RX, RY, RZ, cell, TYPE, Nats, Hubbard_U, params,
                           FelTol, KK0, Res, q, S, Z, PME_data, atom_ids, Q, e,
                           mu0, Te, C, *args, **kwargs)
        H0, U = captured["H0"], captured["U"]
        Znuc, Nocc = captured["Znuc"], captured["Nocc"]

        def q_out(qv):
            return calc_q_eager(H0, U, qv[atom_ids], (C @ qv)[atom_ids], S, Z,
                                Te, Nocc, Znuc, atom_ids, None)[0]

        # q arriving here is q_out(q_old); the loop's own q_old is q - Res.
        q_old = q - Res
        v = torch.tensor([1.0, -1.0], dtype=q.dtype) / (2**0.5)
        h = 1e-5
        fd = (q_out(q_old + h * v) - q_out(q_old - h * v)) / (2 * h)
        analytic = xl.calc_dq(Hubbard_U[atom_ids], v[atom_ids], (C @ v)[atom_ids],
                              S, Z, Te, Q, e, mu0, Nats, atom_ids, None,
                              q[atom_ids])
        j_fd = float((fd - v) @ v)
        j_an = float((analytic - v) @ v)
        newton = abs(float(Res @ v) / j_fd)
        log.append((float(Res.norm()), float(q_old[0]), j_fd, j_an,
                    float(step.norm()), newton, float((KK0 @ Res).norm())))
        return step

    scf_mod.calc_q = calc_q_spy
    scf_mod.kernel_update_lr = kernel_spy
    try:
        params = diatomic_params("Eu", "N", r, SKF_F, 10)
        params["SCF_MAX_ITER"] = max_iter
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            const = Constants(params).to("cpu")
            driver = ESDriver(params, device="cpu")
            structure = Structure(params, const, device="cpu")
            driver(structure, const, do_scf=True)
    finally:
        scf_mod.calc_q = real_calc_q
        scf_mod.kernel_update_lr = real_kernel

    print(f"  {'call':>5} {'|Res|':>11} {'q_old(Eu)':>10} {'J_fd':>9} "
          f"{'J_analytic':>11} {'|Krylov|':>10} {'|Newton|':>10} {'|linear|':>10}")
    for i, (resn, q_old, j_fd, j_an, stepn, newton, lin) in enumerate(log, 1):
        print(f"  {i:>5} {resn:11.3e} {q_old:10.5f} {j_fd:9.4f} {j_an:11.4f} "
              f"{stepn:10.3e} {newton:10.3e} {lin:10.3e}")
    print(f"\n  final q = {[round(x, 6) for x in structure.q.tolist()]}, "
          f"E = {float(structure.e_tot):.6f} eV")
    print()


# --------------------------------------------------------------------------- #
# D. the shape of the problem
# --------------------------------------------------------------------------- #
def test_chargemap(r=3.60):
    banner(
        f"D - the charge map q_out(q_in) for Eu-N at r = {r} A",
        "a fixed point is where q_out = q_in;  slope 0 means the map has saturated",
    )
    captured, structure = capture_calc_q_inputs(
        diatomic_params("Eu", "N", r, SKF_F, KRYLOV_OFF)
    )
    q_out = charge_map(captured, structure.C)
    print(f"  converged answer: q_Eu = {float(structure.q[0]):+.6f}")
    print(f"  {'q_in(Eu)':>10} {'q_out(Eu)':>11} {'q_out-q_in':>11} "
          f"{'d q_out/d q_in':>15}")
    h = 1e-5
    for x in (-5, -4, -3, -2, -1, -0.5, -0.365, 0, 0.5, 1, 2, 3, 4, 5):
        out = q_out(x)
        slope = (q_out(x + h) - q_out(x - h)) / (2 * h)
        print(f"  {x:10.3f} {out:11.4f} {out - x:11.4f} {slope:15.6f}")
    print()


# --------------------------------------------------------------------------- #
# E. what sits at the Fermi level
# --------------------------------------------------------------------------- #
def test_spectrum(r=3.60):
    banner(
        f"E - which orbitals sit at the Fermi level?  Eu-N at r = {r} A",
        "AO layout: 0 = Eu s, 1-3 = Eu p, 4-8 = Eu d, 9-15 = Eu f, 16 = N s, 17-19 = N p",
    )
    from dftorch._dm_fermi_x import dm_fermi_x

    captured, structure = capture_calc_q_inputs(
        diatomic_params("Eu", "N", r, SKF_F, KRYLOV_OFF)
    )
    H0, U, S, Z = captured["H0"], captured["U"], captured["S"], captured["Z"]
    Te, Nocc, aid = captured["Te"], captured["Nocc"], captured["atom_ids"]
    x = float(structure.q[0])
    qv = torch.tensor([x, -x], dtype=S.dtype)
    diag = U * qv[aid] + (structure.C @ qv)[aid]
    H = H0 + 0.5 * (diag.unsqueeze(1) * S + S * diag.unsqueeze(0))
    _, Q, e, f, mu0 = dm_fermi_x(Z.T @ H @ Z, Te, Nocc, mu_0=None, eps=1e-9,
                                 MaxIt=50)
    weight = (Z @ Q) ** 2
    weight = weight / weight.sum(dim=0, keepdim=True)
    groups = {"Eu s": [0], "Eu p": [1, 2, 3], "Eu d": [4, 5, 6, 7, 8],
              "Eu f": list(range(9, 16)), "N s": [16], "N p": [17, 18, 19]}
    print(f"  at the converged charge q_Eu = {x:+.4f},  mu = {float(mu0):.4f} eV")
    print(f"  {'rank':>4} {'E (eV)':>9} {'occ':>6}  "
          + "  ".join(f"{g:>5}" for g in groups))
    for rank, i in enumerate(torch.argsort(e)):
        if not 2 <= rank <= 12:
            continue
        row = "  ".join(f"{float(weight[idx, i].sum()):5.2f}"
                        for idx in groups.values())
        flag = "  <== at mu" if abs(float(e[i]) - float(mu0)) < 0.30 else ""
        print(f"  {rank:>4} {float(e[i]):9.4f} {float(f[i]):6.3f}  {row}{flag}")
    print()


# --------------------------------------------------------------------------- #
# F. the trust region that is commented out
# --------------------------------------------------------------------------- #
def test_trustreg():
    banner(
        "F - does the trust region commented out at _xl_tools.py:664-668 fix it?",
        "the step is capped at 1.25x the plain linear-mixing step, as written there",
    )
    import dftorch._scf as scf_mod

    real_kernel = scf_mod.kernel_update_lr

    def clipped(*args, **kwargs):
        KK0, Res = args[9], args[10]
        step = real_kernel(*args, **kwargs)
        base = torch.norm(KK0 @ Res)
        size = torch.norm(step)
        if size > 1.25 * base and size > 0:
            step = step * ((1.25 * base) / size)
        return step

    print(f"  {'r (A)':>6} {'plain Krylov':>26} {'Krylov + trust region':>26} "
          f"{'Krylov off':>14}")
    n_plain = n_clipped = 0
    for r in EU_N_GRID:
        scf_mod.kernel_update_lr = real_kernel
        plain = run(diatomic_params("Eu", "N", r, SKF_F, 10))
        scf_mod.kernel_update_lr = clipped
        try:
            clip = run(diatomic_params("Eu", "N", r, SKF_F, 10))
        finally:
            scf_mod.kernel_update_lr = real_kernel
        off = run(diatomic_params("Eu", "N", r, SKF_F, KRYLOV_OFF))
        n_plain += not plain["converged"]
        n_clipped += not clip["converged"]

        def cell(rec, with_e=True):
            tag = "OK " if rec["converged"] else "DIV"
            out = f"{tag} it={rec['passes']:<4}"
            return out + (f" E={rec['e_tot']:10.4f}" if with_e else "")

        print(f"  {r:6.2f} {cell(plain):>26} {cell(clip):>26} "
              f"{cell(off, False):>14}")
        sys.stdout.flush()
    print(f"\n  diverged: plain {n_plain}/21, with trust region {n_clipped}/21, "
          f"off 0/21")
    print()


# --------------------------------------------------------------------------- #
# G. smearing the manifold
# --------------------------------------------------------------------------- #
def test_smear():
    banner(
        "G - smearing the manifold at the Fermi level should cure the accelerator",
        "this is a DIAGNOSTIC, not a fix -- see test 'temp' for what it costs",
    )
    points = [2.40, 2.50, 2.80, 3.10, 3.20, 3.30, 3.40, 3.50, 3.60]
    temps = [1000.0, 2000.0, 4000.0, 8000.0]
    for krylov_start, label in ((10, "Krylov ON"), (KRYLOV_OFF, "Krylov OFF")):
        print(f"  {label}:")
        print("  " + f"{'r (A)':>7}"
              + "".join(f"{'T=' + str(int(t)) + 'K':>16}" for t in temps))
        for r in points:
            cells = []
            for t in temps:
                rec = run(diatomic_params("Eu", "N", r, SKF_F, krylov_start, te=t))
                cells.append(f"OK  it={rec['passes']:<4}" if rec["converged"]
                             else "DIVERGE     ")
            print("  " + f"{r:7.2f}" + "".join(f"{c:>16}" for c in cells))
            sys.stdout.flush()
        print()


# --------------------------------------------------------------------------- #
# H. what raising the temperature costs
# --------------------------------------------------------------------------- #
def test_temp():
    banner(
        "H - T_ELECTRONIC is physics, not a solver knob",
        "Krylov is OFF at every point here, so every number is a converged answer",
    )
    temps = [1000.0, 2000.0, 4000.0, 8000.0]
    kB = 8.61739e-5  # eV/K, the same constant _fermi_prt.py uses
    curves = {t: [] for t in temps}
    for r in EU_N_GRID:
        for t in temps:
            curves[t].append(
                run(diatomic_params("Eu", "N", r, SKF_F, KRYLOV_OFF, te=t))["e_tot"]
            )
        sys.stdout.flush()
    print(f"  {'kT (eV)':>28}: " + "".join(f"{kB * t:>12.4f}" for t in temps))
    print(f"  {'T (K)':>28}: " + "".join(f"{int(t):>12d}" for t in temps))
    print()
    for r in (1.90, 2.00, 2.10, 2.60, 3.00, 3.60):
        i = EU_N_GRID.index(r)
        print(f"  E_tot at r = {r:.2f} A     (eV): "
              + "".join(f"{curves[t][i]:>12.4f}" for t in temps))
    print()
    print(f"  {'well depth min(E) (eV)':>28}: "
          + "".join(f"{min(curves[t]):>12.4f}" for t in temps))
    print(f"  {'r at that minimum (A)':>28}: "
          + "".join(f"{EU_N_GRID[curves[t].index(min(curves[t]))]:>12.2f}"
                    for t in temps))
    print(f"  {'E(3.60) - E(min)  (eV)':>28}: "
          + "".join(f"{curves[t][-1] - min(curves[t]):>12.4f}" for t in temps))
    base = curves[1000.0]
    print(f"  {'max |dE| vs the 1000 K curve':>28}: "
          + "".join(f"{max(abs(a - b) for a, b in zip(curves[t], base)):>12.4f}"
                    for t in temps))
    print()


# --------------------------------------------------------------------------- #
# I. non-f diatomics
# --------------------------------------------------------------------------- #
def test_mio():
    banner(
        "I - do ordinary mio-1-1 diatomics ever reach the Krylov branch?",
        "they need more than 10 passes to reach it",
    )
    systems = [("N", "N", 1.10), ("C", "O", 1.13), ("C", "C", 1.24),
               ("O", "O", 1.21), ("H", "H", 0.74), ("N", "H", 1.04)]
    print(f"  {'system':>8} {'Krylov OFF':>16} {'Krylov ON':>16} "
          f"{'Krylov iters':>13}")
    for a, b, r in systems:
        try:
            off = run(diatomic_params(a, b, r, SKF_MIO, KRYLOV_OFF))
            on = run(diatomic_params(a, b, r, SKF_MIO, 10))
        except Exception as exc:  # record, do not abort the sweep
            print(f"  {a + '-' + b:>8}  skipped: {type(exc).__name__}: "
                  f"{str(exc)[:44]}")
            continue
        print(f"  {a + '-' + b:>8} "
              f"{('OK ' if off['converged'] else 'DIV') + ' it=' + str(off['passes']):>16} "
              f"{('OK ' if on['converged'] else 'DIV') + ' it=' + str(on['passes']):>16} "
              f"{on['krylov_iters']:>13}")
    print()


# --------------------------------------------------------------------------- #
# J. the tutorial's own systems
# --------------------------------------------------------------------------- #
def test_tutorial():
    banner(
        "J - do the tutorial's own systems reach the Krylov branch?",
        "SLOW: C840 is 3360 orbitals on CPU.  Expect tens of minutes.",
    )
    systems = [("COORD.xyz", 30.0), ("COORD_8WATER.xyz", 25.0),
               ("COORD_ACETONE.xyz", 25.0), ("m30_o60.xyz", 40.0),
               ("C840.xyz", 60.0)]
    print(f"  {'system':>20} {'atoms':>6} {'orbs':>6} {'Krylov OFF':>14} "
          f"{'Krylov ON':>14} {'kry its':>8} {'dE (eV)':>12}")
    for filename, cell in systems:
        try:
            off = run(bulk_params(filename, cell, KRYLOV_OFF))
            on = run(bulk_params(filename, cell, 10))
        except Exception as exc:
            print(f"  {filename:>20}  skipped: {type(exc).__name__}: "
                  f"{str(exc)[:44]}")
            continue
        print(f"  {filename:>20} {off['nats']:6d} {off['norb']:6d} "
              f"{('OK ' if off['converged'] else 'DIV') + ' it=' + str(off['passes']):>14} "
              f"{('OK ' if on['converged'] else 'DIV') + ' it=' + str(on['passes']):>14} "
              f"{on['krylov_iters']:8d} {on['e_tot'] - off['e_tot']:12.2e}")
        sys.stdout.flush()
    print()
    print("  For any system that fails above, how crowded is its Fermi level?")
    from dftorch.Constants import Constants
    from dftorch.ESDriver import ESDriver
    from dftorch.Structure import Structure
    kB = 8.61739e-5
    for filename, cell in (("m30_o60.xyz", 40.0), ("C840.xyz", 60.0)):
        params = bulk_params(filename, cell, KRYLOV_OFF)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            const = Constants(params).to("cpu")
            structure = Structure(params, const, device="cpu")
            ESDriver(params, device="cpu")(structure, const, do_scf=False)
        e = structure.e.detach()
        f = structure.f.detach()
        mu = float(structure.mu0)
        kT = kB * params["T_ELECTRONIC"]
        near = int(((e - mu).abs() < 3 * kT).sum())
        frac = int(((f > 0.02) & (f < 0.98)).sum())
        print(f"  {filename:>20}  mu = {mu:8.4f} eV   "
              f"levels within 3kT of mu: {near:4d}   "
              f"fractionally occupied: {frac:4d}")
    print()


# --------------------------------------------------------------------------- #
# K. the regime the accelerator was built for
# --------------------------------------------------------------------------- #
def test_md():
    banner(
        "K - the same routine inside XL-BOMD, which is what it was built for",
        "MD.py:913 calls kernel_update_lr once per step; NoRank=True bypasses it",
    )
    import dftorch.MD as md_mod
    from dftorch.Constants import Constants
    from dftorch.ESDriver import ESDriver
    from dftorch.MD import MDXL
    from dftorch.Structure import Structure

    out_dir = Path(__file__).resolve().parent / "mdtraj"
    out_dir.mkdir(exist_ok=True)

    def go(no_rank):
        calls = []
        real = md_mod.kernel_update_lr

        def spy(*args, **kwargs):
            step = real(*args, **kwargs)
            calls.append((float(torch.norm(args[10])), float(torch.norm(step))))
            return step

        md_mod.kernel_update_lr = spy
        params = {
            "FILENAME": str(EXP / "COORD.xyz"), "SKFPATH": str(SKF_TUT) + os.sep,
            "CELL": [30.0, 30.0, 30.0], "T_ELECTRONIC": 1000.0,
            "RCUT_ELECTRONIC": 8.0, "RCUT_REPULSIVE": 4.0, "COUL_METHOD": "FULL",
        }
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                const = Constants(params).to("cpu")
                structure = Structure(params, const, device="cpu")
                driver = ESDriver(params, device="cpu")
                driver(structure, const, do_scf=True)
                driver.calc_forces(structure, const)
                torch.manual_seed(0)
                md = MDXL(driver, const, temperature_K=torch.tensor(300.0))
                if no_rank:
                    md.NoRank = True
                md.run(structure, params, num_steps=12, dt=0.25,
                       dump_interval=100,
                       traj_filename=str(out_dir / ("norank" if no_rank else "lowrank")))
        finally:
            md_mod.kernel_update_lr = real
        text = buf.getvalue()
        return (calls,
                [float(x) for x in re.findall(r"ETOT = ([-0-9.eE+]+)", text)],
                [float(x) for x in re.findall(r"ResErr = ([0-9.eE+-]+)", text)])

    calls, e_lr, err_lr = go(no_rank=False)
    calls_nr, e_nr, err_nr = go(no_rank=True)
    print(f"  kernel_update_lr calls: low-rank {len(calls)}, NoRank {len(calls_nr)}")
    print()
    print(f"  {'step':>4} {'|Res| in':>12} {'|step| out':>11} {'ratio':>7} "
          f"{'ETOT low-rank':>15} {'ResErr':>10} {'ETOT NoRank':>14} {'ResErr':>10}")
    n = min(len(calls), len(e_lr), len(e_nr), len(err_lr), len(err_nr))
    for i in range(n):
        a, b = calls[i]
        print(f"  {i:>4} {a:12.3e} {b:11.3e} {(b / a if a else 0):7.3f} "
              f"{e_lr[i]:15.6f} {err_lr[i]:10.2e} {e_nr[i]:14.6f} {err_nr[i]:10.2e}")
    if n:
        print()
        print(f"  energy drift:                low-rank {e_lr[n - 1] - e_lr[0]:+.6f} eV"
              f"     NoRank {e_nr[n - 1] - e_nr[0]:+.6f} eV")
        print(f"  mean XL-BOMD residual error: low-rank {sum(err_lr[:n]) / n:.3e}"
              f"      NoRank {sum(err_nr[:n]) / n:.3e}")
    print()


TESTS = {
    "scan": test_scan, "handover": test_handover, "jacobian": test_jacobian,
    "chargemap": test_chargemap, "spectrum": test_spectrum,
    "trustreg": test_trustreg, "smear": test_smear, "temp": test_temp,
    "mio": test_mio, "tutorial": test_tutorial, "md": test_md,
}


def main():
    names = sys.argv[1:] or ["scan"]
    if names == ["all"]:
        names = list(TESTS)
    unknown = [n for n in names if n not in TESTS]
    if unknown:
        sys.exit(f"unknown test(s): {', '.join(unknown)}. "
                 f"choose from: {', '.join(TESTS)}, or 'all'")
    # float64 throughout, matching every other script in this directory.  It also
    # matters here: kernel_update_lr allocates its Krylov basis with
    # torch.zeros(..., device=...) and no dtype, so it silently inherits this.
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        for name in names:
            TESTS[name]()
    finally:
        torch.set_default_dtype(previous)


if __name__ == "__main__":
    main()
