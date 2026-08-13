import os
for k in ("TORCHDYNAMO_DISABLE","TORCH_COMPILE_DISABLE","TORCHINDUCTOR_DISABLE"): os.environ.setdefault(k,"1")
import contextlib, io, re
from pathlib import Path
import torch
REPO=Path(r"c:/Users/aryan/goHereForEverything/GTYear1/RESEARCH/LANL/DFTorch"); SKF=REPO/"tests"/"f_orbital_data"
TMP=Path(__file__).resolve().parent/"geom"; TMP.mkdir(exist_ok=True)
HA=27.211386245988
BASE={"T_ELECTRONIC":1000.0,"RCUT_ELECTRONIC":10.0,"RCUT_REPULSIVE":6.0,"COUL_METHOD":"FULL","CHARGE":0,"KRYLOV_START":10**6}
def params(r):
    p=TMP/f"mc_{r:.3f}.xyz"; p.write_text("2\nEu-N\nEu 0 0 0\nN %.8f 0 0\n"%r)
    d=dict(BASE); d["FILENAME"]=str(p); d["SKFPATH"]=str(SKF)+os.sep; return d
def run(r, do_scf=True, u_eu=None):
    from dftorch.Constants import Constants
    from dftorch.ESDriver import ESDriver
    from dftorch.Structure import Structure
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf):
        pr=params(r); c=Constants(pr).to("cpu"); d=ESDriver(pr,device="cpu")
        s=Structure(pr,c,device="cpu")
        if u_eu is not None: s.Hubbard_U = s.Hubbard_U.clone(); s.Hubbard_U[0]=u_eu
        d(s,c,do_scf=do_scf)
        out=(float(s.e_tot.item()), [float(x) for x in s.q.tolist()],
             [float(x) for x in s.e.tolist()], [float(x) for x in s.f.tolist()], float(s.mu0))
    t=buf.getvalue()
    return out + (("Did not converge" not in t),)
def main():
    print("="*92)
    print("MECHANISM TEST 1 - the 0-shot dissociation limit is FULL SHELL FILLING")
    print("  at r = 10 A the two atoms are uncoupled, so eigenvalues = bare on-site energies")
    print("="*92)
    e,q,ev,occ,mu,_ = run(10.0, do_scf=False)
    names = ["Eu-s"]+["Eu-p"]*3+["Eu-d"]*5+["Eu-f"]*7+["N-s"]+["N-p"]*3
    order = sorted(range(20), key=lambda i: ev[i])
    print(f"  Fermi level mu = {mu:.4f} eV;  Nocc = 7 pairs = 14 electrons")
    print(f"  {'rank':>4} {'eigval(eV)':>11} {'occ(f)':>8}   which orbital (by nearest on-site energy)")
    for rank,i in enumerate(order[:10]):
        print(f"  {rank:>4} {ev[i]:11.4f} {occ[i]:8.4f}   {'<-- N shell' if ev[i] < -5 else ''}")
    print(f"  ... total electrons on N = {q[1]:+.4f} + 5 = {q[1]+5:.4f} (N holds 4 orbitals = 8 max)")
    print(f"  q_Eu = {q[0]:+.4f}   q_N = {q[1]:+.4f}")
    print()
    print("="*92)
    print("MECHANISM TEST 2 - does Eu's Hubbard U choice explain the residual transfer?")
    print("  model: at r -> inf, Delta = level_gap / (U_Eu + U_N)")
    print("="*92)
    gap = -1.5211 - (-6.8355)
    U_N = 13.3336
    for label, u in [("s-shell Us=0.21 Ha (WHAT THE CODE USES)", 0.21*HA),
                     ("f-shell Uf=0.50 Ha (where 7 of 9 e- live)", 0.50*HA)]:
        e,q,_,_,_,cv = run(40.0, u_eu=u)
        pred = gap/(u+U_N)
        print(f"  U_Eu = {u:6.3f} eV  |  {label}")
        print(f"      measured q_Eu at 40 A = {q[0]:+.4f} e   converged={cv}")
        print(f"      predicted  -gap/(U_Eu+U_N) = {-pred:+.4f} e   (gap = {gap:.4f} eV)")
        print(f"      E_tot at 40 A = {e:.5f} eV")
prev=torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
try: main()
finally: torch.set_default_dtype(prev)
