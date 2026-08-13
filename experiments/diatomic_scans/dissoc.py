import os
for k in ("TORCHDYNAMO_DISABLE","TORCH_COMPILE_DISABLE","TORCHINDUCTOR_DISABLE"): os.environ.setdefault(k,"1")
import contextlib, io, json, re
from pathlib import Path
import torch
REPO=Path(r"c:/Users/aryan/goHereForEverything/GTYear1/RESEARCH/LANL/DFTorch"); SKF=REPO/"tests"/"f_orbital_data"
TMP=Path(__file__).resolve().parent/"geom"; TMP.mkdir(exist_ok=True)
OUT=Path(__file__).resolve().parent/"eu_n_dissoc.json"
BASE={"T_ELECTRONIC":1000.0,"RCUT_ELECTRONIC":10.0,"RCUT_REPULSIVE":6.0,"COUL_METHOD":"FULL","CHARGE":0}
def params(r,**o):
    p=TMP/f"ds_{r:.3f}.xyz"; p.write_text("2\nEu-N\nEu 0 0 0\nN %.8f 0 0\n"%r)
    d=dict(BASE); d["FILENAME"]=str(p); d["SKFPATH"]=str(SKF)+os.sep; d.update(o); return d
def run(r,do_scf=True,**o):
    from dftorch.Constants import Constants
    from dftorch.ESDriver import ESDriver
    from dftorch.Structure import Structure
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf):
        pr=params(r,**o); c=Constants(pr).to("cpu"); d=ESDriver(pr,device="cpu")
        s=Structure(pr,c,device="cpu"); d(s,c,do_scf=do_scf)
        e=float(s.e_tot.item()); q=[float(x) for x in s.q.tolist()]
    t=buf.getvalue(); it=[int(m) for m in re.findall(r"^Iter (\d+)\s*$",t,re.M)]
    return e,q,(it[-1] if it else 0),("Did not converge" not in t)
def main():
    grid=[2.0,2.5,3.0,3.6,4.0,5.0,6.0,8.0,10.0,15.0,20.0,30.0,40.0]
    recs=[]
    for r in grid:
        e0,q0,_,_=run(r,do_scf=False)
        e1,q1,it1,c1=run(r,KRYLOV_START=10**6)
        e2,q2,it2,c2=run(r)
        recs.append({"r":r,"q_eu_0shot":q0[0],"e_0shot":e0,
                     "q_eu_scf":q1[0],"e_scf":e1,"it_scf":it1,"conv_scf":c1,
                     "q_eu_scf_default":q2[0],"e_scf_default":e2,"it_default":it2,"conv_default":c2})
        print(f"{r:6.1f}  0shot q_Eu={q0[0]:+.4f} | scf(noKry) q_Eu={q1[0]:+.4f} conv={c1} | scf(default) q_Eu={q2[0]:+.4f} conv={c2}")
    OUT.write_text(json.dumps({"records":recs},indent=2)); print("wrote",OUT)
prev=torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
try: main()
finally: torch.set_default_dtype(prev)
