"""Diatomic binding-curve suite over the mio-1-1 parameter set.

For each system, scans the interatomic separation and records BOTH the
non-self-consistent (H0, do_scf=False) and the self-consistent charge
(SCF, do_scf=True) total energy, plus Mulliken charges and SCF iteration
counts.

The Krylov charge-mixer acceleration is disabled throughout (KRYLOV_START set
above SCF_MAX_ITER).  That is a deliberate, recorded decision: the low-rank
Krylov accelerator was measured to diverge on the Eu-N f case while producing
identical energies to plain Anderson/DIIS mixing wherever it did converge.
Repairing it is deferred to a later phase.

Only even-valence-electron systems appear here.  Structure.py:355 raises
"Closed shell systems require even number of electrons" for odd counts, so the
common radicals (OH, CH, CN, NO) cannot run on the closed-shell path at all.
Hydroxide OH- stands in for the O-H bond.
"""

import os

for _k in ("TORCHDYNAMO_DISABLE", "TORCH_COMPILE_DISABLE", "TORCHINDUCTOR_DISABLE"):
    os.environ.setdefault(_k, "1")

import contextlib
import io
import json
import re
import sys
import time
import traceback
from pathlib import Path

import torch

REPO = Path(r"c:/Users/aryan/goHereForEverything/GTYear1/RESEARCH/LANL/DFTorch")
SKF = REPO / "tests" / "data_skf_mio-1-1"
HERE = Path(__file__).resolve().parent
TMP = HERE / "geom_mio"
TMP.mkdir(exist_ok=True)
OUT = HERE / "mio_suite.json"

# Separation at which the pair is treated as dissociated, for binding energies.
# Inside RCUT_ELECTRONIC = 8.0 so the cutoff never truncates it.
R_FAR = 6.0

BASE = {
    "T_ELECTRONIC": 1000.0,
    "RCUT_ELECTRONIC": 8.0,
    "RCUT_REPULSIVE": 4.0,
    "COUL_METHOD": "FULL",
    "SCF_MAX_ITER": 200,
    "KRYLOV_START": 10**6,  # Krylov acceleration disabled - see module docstring
}

# label, element A, element B, net charge, r_min, r_max, experimental r_e (A)
#
# The exp_re column is a REFERENCE MARKER ONLY, quoted from standard diatomic
# spectroscopic tables.  It is not a tolerance and nothing asserts against it.
SYSTEMS = [
    ("H2",  "H", "H",  0, 0.40, 2.60, 0.7414),
    ("C2",  "C", "C",  0, 0.90, 3.20, 1.2425),
    ("N2",  "N", "N",  0, 0.80, 3.00, 1.0977),
    ("O2",  "O", "O",  0, 0.85, 3.00, 1.2075),
    ("CO",  "C", "O",  0, 0.80, 3.00, 1.1283),
    ("CN-", "C", "N", -1, 0.85, 3.00, 1.1772),
    ("NH",  "N", "H",  0, 0.60, 2.60, 1.0362),
    ("CH-", "C", "H", -1, 0.70, 2.80, 1.1190),
    ("OH-", "O", "H", -1, 0.60, 2.60, 0.9640),
    ("PN",  "P", "N",  0, 1.10, 3.60, 1.4909),
    ("P2",  "P", "P",  0, 1.40, 4.00, 1.8934),
    ("S2",  "S", "S",  0, 1.40, 4.00, 1.8892),
    ("SO",  "S", "O",  0, 1.10, 3.50, 1.4811),
    ("CS",  "C", "S",  0, 1.10, 3.60, 1.5349),
    ("PH",  "P", "H",  0, 0.80, 3.00, 1.4223),
    ("SH-", "S", "H", -1, 0.80, 3.00, 1.3400),
]

N_POINTS = 61


def _params(label, a, b, charge, r):
    xyz = TMP / f"{label}_{r:.4f}.xyz"
    xyz.write_text(
        f"2\n{label} diatomic scan\n"
        f"{a} 0.00000000 0.00000000 0.00000000\n"
        f"{b} {r:.8f} 0.00000000 0.00000000\n"
    )
    p = dict(BASE)
    p["FILENAME"] = str(xyz)
    p["SKFPATH"] = str(SKF) + os.sep
    p["CHARGE"] = charge
    return p


_ITER = re.compile(r"^Iter (\d+)\s*$", re.M)


def _one(label, a, b, charge, r, do_scf, const=None, driver=None):
    from dftorch.Constants import Constants
    from dftorch.ESDriver import ESDriver
    from dftorch.Structure import Structure

    p = _params(label, a, b, charge, r)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        if const is None:
            const = Constants(p).to("cpu")
        if driver is None:
            driver = ESDriver(p, device="cpu")
        s = Structure(p, const, device="cpu")
        driver(s, const, do_scf=do_scf)
        e = float(s.e_tot.item())
        q = [float(x) for x in s.q.tolist()]
    txt = buf.getvalue()
    its = [int(m) for m in _ITER.findall(txt)]
    return {
        "e_tot": e,
        "q": q,
        "iterations": (its[-1] if its else 0),
        "converged": ("Did not converge" not in txt),
    }


def scan(label, a, b, charge, rmin, rmax, exp_re):
    from dftorch.Constants import Constants
    from dftorch.ESDriver import ESDriver

    grid = [round(rmin + (rmax - rmin) * i / (N_POINTS - 1), 4) for i in range(N_POINTS)]
    p0 = _params(label, a, b, charge, grid[0])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        const = Constants(p0).to("cpu")
        drv = ESDriver(p0, device="cpu")

    pts, failures = [], 0
    for r in grid:
        rec = {"r": r}
        for tag, flag in (("h0", False), ("scf", True)):
            try:
                out = _one(label, a, b, charge, r, flag, const=const, driver=drv)
                rec[tag] = out
            except Exception as exc:
                failures += 1
                rec[tag] = {"error": type(exc).__name__, "msg": str(exc)[:300]}
        pts.append(rec)

    far = {}
    for tag, flag in (("h0", False), ("scf", True)):
        try:
            far[tag] = _one(label, a, b, charge, R_FAR, flag, const=const, driver=drv)
        except Exception as exc:
            far[tag] = {"error": type(exc).__name__, "msg": str(exc)[:300]}

    return {
        "label": label,
        "elements": [a, b],
        "charge": charge,
        "exp_re": exp_re,
        "r_far": R_FAR,
        "points": pts,
        "far": far,
        "failures": failures,
    }


def main():
    results = []
    for label, a, b, charge, rmin, rmax, exp_re in SYSTEMS:
        t0 = time.perf_counter()
        try:
            res = scan(label, a, b, charge, rmin, rmax, exp_re)
        except Exception:
            sys.stderr.write(f"{label:5s} FAILED ENTIRELY\n{traceback.format_exc()}\n")
            results.append({"label": label, "fatal": traceback.format_exc()[-1500:]})
            continue

        ok_scf = [p for p in res["points"] if "e_tot" in p["scf"] and p["scf"]["converged"]]
        ok_h0 = [p for p in res["points"] if "e_tot" in p["h0"]]
        nconv = len(ok_scf)
        m_scf = min(ok_scf, key=lambda p: p["scf"]["e_tot"])["r"] if ok_scf else float("nan")
        m_h0 = min(ok_h0, key=lambda p: p["h0"]["e_tot"])["r"] if ok_h0 else float("nan")
        sys.stderr.write(
            f"{label:5s} q={res['charge']:+d}  scf {nconv:2d}/{len(res['points'])} conv  "
            f"min_H0={m_h0:.3f}  min_SCF={m_scf:.3f}  exp={res['exp_re']:.3f} A  "
            f"({time.perf_counter() - t0:.1f}s)\n"
        )
        sys.stderr.flush()
        results.append(res)

    OUT.write_text(json.dumps({"r_far": R_FAR, "base": BASE, "systems": results}, indent=2))
    sys.stderr.write(f"\nwrote {OUT}\n")


prev = torch.get_default_dtype()
torch.set_default_dtype(torch.float64)
try:
    main()
finally:
    torch.set_default_dtype(prev)
