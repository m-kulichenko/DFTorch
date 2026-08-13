"""Eu-N diatomic 21-point separation scan run through the SELF-CONSISTENT path.

Same geometry, same grid, same driver parameters as tests/test_eu_n_scan.py --
the only change is ``do_scf=True`` instead of ``do_scf=False``.  Dumps a JSON
record (energies, charges, SCF iteration counts, final residuals, convergence
flags) so the plotting step can run in a different interpreter.
"""

import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

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
SKF_DIR = REPO / "tests" / "f_orbital_data"
OUT_JSON = Path(__file__).resolve().parent / "eu_n_scf_scan.json"
TMP_DIR = Path(__file__).resolve().parent / "geom"
TMP_DIR.mkdir(exist_ok=True)

# --- grid, copied verbatim from tests/test_eu_n_scan.py ----------------------
SCAN_MIN_ANGSTROM = 1.60
SCAN_MAX_ANGSTROM = 3.60
SCAN_STEP_ANGSTROM = 0.10

EU_N_TARGET_ANGSTROM = 2.655
EU_N_BAND_FRACTION = 0.20
EU_N_BAND_MIN_ANGSTROM = EU_N_TARGET_ANGSTROM * (1.0 - EU_N_BAND_FRACTION)
EU_N_BAND_MAX_ANGSTROM = EU_N_TARGET_ANGSTROM * (1.0 + EU_N_BAND_FRACTION)

# Driver parameters, pinned to match tests/test_eu_n_scan.py exactly.
EU_N_PARAMS = {
    "T_ELECTRONIC": 1000.0,
    "RCUT_ELECTRONIC": 10.0,
    "RCUT_REPULSIVE": 6.0,
    "COUL_METHOD": "FULL",
    "CHARGE": 0,
}

SCF_TOL = 1e-6
SCF_MAX_ITER = 100


def scan_separations():
    count = int(round((SCAN_MAX_ANGSTROM - SCAN_MIN_ANGSTROM) / SCAN_STEP_ANGSTROM)) + 1
    return [round(SCAN_MIN_ANGSTROM + SCAN_STEP_ANGSTROM * i, 2) for i in range(count)]


def _write_eu_n_xyz(path, separation):
    path.write_text(
        "2\n"
        "Eu-N diatomic (Phase 6 SCF separation scan)\n"
        "Eu 0.00000000 0.00000000 0.00000000\n"
        f"N {separation:.8f} 0.00000000 0.00000000\n"
    )


def _eu_n_params(separation):
    xyz_path = TMP_DIR / f"eu_n_{separation:.2f}.xyz"
    _write_eu_n_xyz(xyz_path, separation)
    params = dict(EU_N_PARAMS)
    params["FILENAME"] = str(xyz_path)
    params["SKFPATH"] = str(SKF_DIR) + os.sep
    return params


_ITER_RE = re.compile(r"^Iter (\d+)\s*$", re.M)
_RES_RE = re.compile(r"^Res = ([0-9.eE+-]+), dEc = ([0-9.eE+-]+)", re.M)


def parse_scf_log(text):
    """Pull iteration count, final residual and dEc out of the library chatter."""
    iters = [int(m) for m in _ITER_RE.findall(text)]
    residuals = [(float(a), float(b)) for a, b in _RES_RE.findall(text)]
    did_not_converge = "Did not converge" in text
    return {
        "iterations": iters[-1] if iters else 0,
        "res_norm_final": residuals[-1][0] if residuals else None,
        "dEc_final": residuals[-1][1] if residuals else None,
        "res_norm_trace": [r for r, _ in residuals],
        "dEc_trace": [d for _, d in residuals],
        "did_not_converge_printed": did_not_converge,
    }


def run_scan():
    from dftorch.Constants import Constants
    from dftorch.ESDriver import ESDriver
    from dftorch.Structure import Structure

    separations = scan_separations()

    # const/driver built once from the first grid point and reused, exactly as
    # tests/test_eu_n_scan.py does (Constants reads the geometry only for the
    # species list, which is (Eu, N) at every point).
    params0 = _eu_n_params(separations[0])
    const = Constants(params0).to("cpu")
    driver = ESDriver(params0, device="cpu")

    records = []
    for i, separation in enumerate(separations):
        params = _eu_n_params(separation)
        buf = io.StringIO()
        t0 = time.perf_counter()
        record = {"separation": separation, "index": i}
        try:
            with contextlib.redirect_stdout(buf):
                structure = Structure(params, const, device="cpu")
                driver(structure, const, do_scf=True)
                e_tot = float(structure.e_tot.item())
                e_elec = float(structure.e_elec_tot.item())
                e_rep = float(structure.e_repulsion.item())
                e_band0 = float(structure.e_band0.item())
                e_coul = float(structure.e_coul.item())
                q = [float(x) for x in structure.q.detach().cpu().tolist()]
                dtype = str(structure.e_tot.dtype)
            record.update(
                {
                    "ok": True,
                    "e_tot": e_tot,
                    "e_elec_tot": e_elec,
                    "e_repulsion": e_rep,
                    "e_band0": e_band0,
                    "e_coul": e_coul,
                    "q": q,
                    "dtype": dtype,
                }
            )
        except Exception as exc:  # record, do not abort the whole scan
            record.update(
                {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        record["wall_seconds"] = time.perf_counter() - t0
        record.update(parse_scf_log(buf.getvalue()))
        record["stdout_tail"] = buf.getvalue()[-2000:]
        records.append(record)

        status = "OK " if record["ok"] else "ERR"
        sys.stderr.write(
            f"[{i + 1:2d}/{len(separations)}] r={separation:.2f} A  {status} "
            f"E={record.get('e_tot', float('nan')):.8f} eV  "
            f"it={record.get('iterations')}  "
            f"Res={record.get('res_norm_final')}  "
            f"({record['wall_seconds']:.1f}s)\n"
        )
        sys.stderr.flush()

    return {
        "grid": {
            "min": SCAN_MIN_ANGSTROM,
            "max": SCAN_MAX_ANGSTROM,
            "step": SCAN_STEP_ANGSTROM,
            "n_points": len(separations),
        },
        "band": {
            "target": EU_N_TARGET_ANGSTROM,
            "fraction": EU_N_BAND_FRACTION,
            "min": EU_N_BAND_MIN_ANGSTROM,
            "max": EU_N_BAND_MAX_ANGSTROM,
        },
        "scf": {"tol": SCF_TOL, "max_iter": SCF_MAX_ITER},
        "params": {k: v for k, v in EU_N_PARAMS.items()},
        "do_scf": True,
        "torch_version": torch.__version__,
        "records": records,
    }


def main():
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        result = run_scan()
    finally:
        torch.set_default_dtype(previous_dtype)

    OUT_JSON.write_text(json.dumps(result, indent=2))
    sys.stderr.write(f"\nwrote {OUT_JSON}\n")


if __name__ == "__main__":
    main()
