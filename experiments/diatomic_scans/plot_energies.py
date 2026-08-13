import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style
style.apply()
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
FIGS=Path(r"c:/Users/aryan/goHereForEverything/GTYear1/RESEARCH/LANL/DFTorch/figures"); FIGS.mkdir(exist_ok=True)
NAVY,SLATE,CRIMSON,GREEN=style.NAVY,style.SLATE,style.CRIMSON,style.GREEN
HA=27.211386245988
SPE={"H":-0.0330,"C":-0.0439,"N":-0.1112,"O":-0.05414,"P":-0.05885125,"S":0.0}
EXP={"H2":4.7477,"C2":6.32,"N2":9.905,"O2":5.2132,"CO":11.226,"NH":3.47,
     "PN":6.36,"P2":5.08,"S2":4.37,"SO":5.43,"CS":7.36,"PH":3.10}
SUB=str.maketrans("0123456789","\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089")
d=json.loads((HERE/"mio_suite.json").read_text())
rows=[]
for s in d["systems"]:
    if "points" not in s or s["charge"]!=0: continue
    a,b=s["elements"]
    pts=[p for p in s["points"] if p["scf"].get("converged")]
    em=min(p["scf"]["e_tot"] for p in pts)
    rows.append({"l":s["label"],"raw":-em,"spe":-em+(SPE[a]+SPE[b])*HA,"exp":EXP[s["label"]]})
rows.sort(key=lambda r:r["exp"])

fig,axes=plt.subplots(1,2,figsize=(16.0,6.8),gridspec_kw={"width_ratios":[1.15,1]})
fig.suptitle("Atomization Energies \u2014 mio-1-1 Diatomics vs Experiment",
             fontsize=18,fontweight="bold",y=0.975)
ys=list(range(len(rows)))
lab=[r["l"].translate(SUB) for r in rows]

ax=axes[0]
for y,r in zip(ys,rows):
    ax.plot([min(r["exp"],r["raw"],r["spe"]),max(r["exp"],r["raw"],r["spe"])],[y,y],
            "-",color="#cfcfcf",lw=2.0,zorder=1)
ax.plot([r["exp"] for r in rows],ys,"D",color=GREEN,ms=11,zorder=5,label="Experimental $D_e$")
ax.plot([r["raw"] for r in rows],ys,"s",color=SLATE,ms=10,zorder=4,
        label="SCC, spin-unpolarised atomic reference")
ax.plot([r["spe"] for r in rows],ys,"o",color=NAVY,ms=10,zorder=6,
        label="SCC, spin-polarised reference (+SPE)")
ax.set_yticks(ys); ax.set_yticklabels(lab,fontsize=14)
ax.set_xlabel("Atomization energy  $D_e$  (eV)")
ax.set_title("Absolute atomization energy")
ax.legend(loc="lower right"); ax.grid(axis="y",alpha=0.15); style.finish(ax)

ax=axes[1]
h=0.36
ax.axvline(0.0,color="#444444",lw=1.4)
ax.barh([y+h/2 for y in ys],[r["raw"]-r["exp"] for r in rows],height=h,color=SLATE,
        label="spin-unpolarised ref",zorder=3)
ax.barh([y-h/2 for y in ys],[r["spe"]-r["exp"] for r in rows],height=h,color=NAVY,
        label="spin-polarised ref (+SPE)",zorder=3)
ax.set_yticks(ys); ax.set_yticklabels(lab,fontsize=14)
ax.set_xlabel("Error vs experimental $D_e$  (eV)")
ax.set_title("Signed error")
ax.legend(loc="lower right"); ax.grid(axis="y",alpha=0.15); style.finish(ax)

fig.tight_layout(rect=[0,0,1,0.94])
out=FIGS/"mio_atomization_energies.png"; fig.savefig(out,dpi=200); print("wrote",out)
