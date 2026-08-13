import os
for k in ("TORCHDYNAMO_DISABLE","TORCH_COMPILE_DISABLE","TORCHINDUCTOR_DISABLE"): os.environ.setdefault(k,"1")
import contextlib, io, json
from pathlib import Path
import torch
REPO=Path(r"c:/Users/aryan/goHereForEverything/GTYear1/RESEARCH/LANL/DFTorch"); SKF=REPO/"tests"/"f_orbital_data"
TMP=Path(__file__).resolve().parent/"geom"; TMP.mkdir(exist_ok=True)
OUT=Path(__file__).resolve().parent/"eu_n_0shot.json"
def params(r):
    p=TMP/f"z_{r:.2f}.xyz"; p.write_text("2\nEu-N\nEu 0 0 0\nN %.8f 0 0\n"%r)
    return {"T_ELECTRONIC":1000.0,"RCUT_ELECTRONIC":10.0,"RCUT_REPULSIVE":6.0,"COUL_METHOD":"FULL","CHARGE":0,"FILENAME":str(p),"SKFPATH":str(SKF)+os.sep}
def main():
    from dftorch.Constants import Constants
    from dftorch.ESDriver import ESDriver
    from dftorch.Structure import Structure
    grid=[round(1.60+0.10*i,2) for i in range(21)]
    buf=io.StringIO(); recs=[]
    with contextlib.redirect_stdout(buf):
        c=Constants(params(grid[0])).to("cpu"); d=ESDriver(params(grid[0]),device="cpu")
        for r in grid:
            s=Structure(params(r),c,device="cpu"); d(s,c,do_scf=False)
            recs.append({"separation":r,"e_tot":float(s.e_tot.item()),"q":[float(x) for x in s.q.tolist()]})
    OUT.write_text(json.dumps({"records":recs},indent=2)); print("wrote",OUT)
prev=torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
try: main()
finally: torch.set_default_dtype(prev)
