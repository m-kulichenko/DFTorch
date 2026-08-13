"""Presentation figures for the Eu-N diatomic: H0 curve and self-consistent curve."""

import json
from pathlib import Path

import style

style.apply()
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = Path(r"c:/Users/aryan/goHereForEverything/GTYear1/RESEARCH/LANL/DFTorch")
FIGS = REPO / "figures"
FIGS.mkdir(exist_ok=True)

NAVY, SLATE, CRIMSON, GREEN = style.NAVY, style.SLATE, style.CRIMSON, style.GREEN

scf = json.loads((HERE / "eu_n_scf_nokrylov.json").read_text())["records"]
h0 = json.loads((HERE / "eu_n_0shot.json").read_text())["records"]
ds = json.loads((HERE / "eu_n_dissoc.json").read_text())["records"]

r = [x["separation"] for x in scf]
e_scf = [x["e_tot"] for x in scf]
q_scf = [x["q"][0] for x in scf]
e_h0 = [x["e_tot"] for x in h0]
q_h0 = [x["q"][0] for x in h0]

i_s = min(range(len(e_scf)), key=lambda i: e_scf[i])
i_h = min(range(len(e_h0)), key=lambda i: e_h0[i])

TARGET = 2.655


# ---------------------------------------------------------------- figure 1 ---
fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.4))
fig.suptitle(
    "Eu\u2013N Diatomic Binding Curve: Non-Self-Consistent vs Self-Consistent Charge",
    fontsize=16.5,
    fontweight="bold",
    y=0.985,
)

ax = axes[0]
ax.plot(r, e_h0, "-s", color=SLATE, ms=5.5, lw=1.8, label="Non-SCC ($H^0$ only)")
ax.plot(r, e_scf, "-o", color=NAVY, ms=6.5, lw=2.2, label="Self-consistent charge")
ax.plot([r[i_h]], [e_h0[i_h]], "*", color=SLATE, ms=20, zorder=6)
ax.plot([r[i_s]], [e_scf[i_s]], "*", color=CRIMSON, ms=22, zorder=6)
ax.axvline(TARGET, color=GREEN, ls="--", lw=1.6, alpha=0.9,
           label=f"Eu\u2013N coordination mean, {TARGET} \u00c5")
ax.set_xlabel("Internuclear separation  $r$  (\u00c5)")
ax.set_ylabel("Total energy  $E_\\mathrm{tot}$  (eV)")
ax.set_title("Total energy")
ax.legend(loc="center right")
ax.annotate(f"{r[i_s]:.2f} \u00c5,  {e_scf[i_s]:.3f} eV",
            xy=(r[i_s], e_scf[i_s]), xytext=(r[i_s] + 0.28, e_scf[i_s] - 2.4),
            fontsize=11, color=CRIMSON, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=CRIMSON, lw=1.3))
ax.annotate(f"{r[i_h]:.2f} \u00c5,  {e_h0[i_h]:.3f} eV",
            xy=(r[i_h], e_h0[i_h]), xytext=(r[i_h] + 0.34, e_h0[i_h] + 2.3),
            fontsize=11, color="#444444", fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=SLATE, lw=1.3))
style.finish(ax)

ax = axes[1]
ax.axhline(0.0, color="#444444", lw=1.1)
ax.plot(r, q_h0, "-s", color=SLATE, ms=5.5, lw=1.8, label="Non-SCC ($H^0$ only)")
ax.plot(r, q_scf, "-o", color=NAVY, ms=6.5, lw=2.2, label="Self-consistent charge")
ax.axvline(TARGET, color=GREEN, ls="--", lw=1.6, alpha=0.9)
ax.set_xlabel("Internuclear separation  $r$  (\u00c5)")
ax.set_ylabel("Mulliken charge on Eu,  $q_\\mathrm{Eu}$  (e)")
ax.set_title("Charge transfer")
ax.legend(loc="center right")
style.finish(ax)

fig.tight_layout(rect=[0, 0, 1, 0.955])
out1 = FIGS / "eu_n_binding_curve.png"
fig.savefig(out1, dpi=200)
plt.close(fig)
print(f"wrote {out1}")


# ---------------------------------------------------------------- figure 2 ---
fig, ax = plt.subplots(figsize=(9.0, 6.4))
fig.suptitle(
    "Eu\u2013N Dissociation Limit: Mulliken Charge vs Internuclear Separation",
    fontsize=15.5,
    fontweight="bold",
    y=0.975,
)
rd = [x["r"] for x in ds]
ax.axhline(0.0, color="#444444", lw=1.1)
ax.plot(rd, [x["q_eu_0shot"] for x in ds], "-s", color=SLATE, ms=6.5, lw=1.9,
        label="Non-SCC ($H^0$ only)")
ax.plot(rd, [x["q_eu_scf"] for x in ds], "-o", color=NAVY, ms=7, lw=2.2,
        label="Self-consistent charge")
ax.axhline(-3.0, color=SLATE, ls=":", lw=1.5)
ax.axhline(-0.279, color=CRIMSON, ls="--", lw=1.6)
ax.text(2.15, -2.90, "$-3.0000$ e  (N valence shell filled)",
        fontsize=10.5, color="#444444")
ax.text(2.15, -0.19, "$-0.279$ e  $= -\\Delta\\varepsilon\\,/\\,(U_\\mathrm{Eu}+U_\\mathrm{N})$",
        fontsize=10.5, color=CRIMSON)
ax.set_xscale("log")
ax.set_xticks([2, 3, 5, 10, 20, 40])
ax.set_xticklabels(["2", "3", "5", "10", "20", "40"])
ax.set_ylim(-3.35, 0.42)
ax.set_xlabel("Internuclear separation  $r$  (\u00c5,  logarithmic)")
ax.set_ylabel("Mulliken charge on Eu,  $q_\\mathrm{Eu}$  (e)")
ax.set_title("Neither path dissociates to neutral atoms")
ax.legend(loc="center left")
style.finish(ax)
fig.tight_layout(rect=[0, 0, 1, 0.945])
out2 = FIGS / "eu_n_dissociation_limit.png"
fig.savefig(out2, dpi=200)
plt.close(fig)
print(f"wrote {out2}")
