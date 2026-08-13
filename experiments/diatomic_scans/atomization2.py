import json
from pathlib import Path
HA = 27.211386245988
HERE = Path(__file__).resolve().parent
# SKF field 4 of the homonuclear line = SPE, the free-atom spin-polarisation energy (Ha)
SPE = {"H": -0.0330, "C": -0.0439, "N": -0.1112, "O": -0.05414, "P": -0.05885125, "S": 0.0}
EXP_DE = {"H2":4.7477,"C2":6.32,"N2":9.905,"O2":5.2132,"CO":11.226,"NH":3.47,
          "PN":6.36,"P2":5.08,"S2":4.37,"SO":5.43,"CS":7.36,"PH":3.10}
data = json.loads((HERE/"mio_suite.json").read_text())
rows=[]
for s in data["systems"]:
    if "points" not in s or s["charge"]!=0: continue
    a,b=s["elements"]
    pts=[p for p in s["points"] if p["scf"].get("converged")]
    lo=min(pts,key=lambda p:p["scf"]["e_tot"])
    em=lo["scf"]["e_tot"]; far=s["far"]["scf"]["e_tot"]
    spe=(SPE[a]+SPE[b])*HA
    rows.append({"l":s["label"],"r":lo["r"],"em":em,"far":far,
                 "de_raw":-em,"de_spe":-em+spe,"exp":EXP_DE.get(s["label"])})
print(f"{'sys':>4} {'E(6A)':>8} {'E_min':>9} | {'De=-E_min':>10} {'De+SPE':>8} {'exp De':>7} | "
      f"{'err raw':>8} {'err SPE':>8}")
print("-"*78)
for r in rows:
    e=r["exp"]
    print(f"{r['l']:>4} {r['far']:8.3f} {r['em']:9.3f} | {r['de_raw']:10.3f} {r['de_spe']:8.3f} "
          f"{e:7.2f} | {r['de_raw']-e:+8.2f} {r['de_spe']-e:+8.2f}")
print("-"*78)
for k,lab in (("de_raw","spin-unpolarised atomic ref (D0 as-is)"),
              ("de_spe","spin-polarised ref (D0 + SKF SPE)")):
    errs=[r[k]-r["exp"] for r in rows]
    mae=sum(abs(x) for x in errs)/len(errs); me=sum(errs)/len(errs)
    print(f"{lab:>42}:  ME {me:+6.2f} eV   MAE {mae:5.2f} eV")
