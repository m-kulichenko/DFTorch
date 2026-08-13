"""Presentation figures for the mio-1-1 diatomic binding-curve suite."""

import json
import math
from pathlib import Path

import style

style.apply()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
REPO = Path(r"c:/Users/aryan/goHereForEverything/GTYear1/RESEARCH/LANL/DFTorch")
FIGS = REPO / "figures"
FIGS.mkdir(exist_ok=True)

NAVY, SLATE, CRIMSON, GREEN = style.NAVY, style.SLATE, style.CRIMSON, style.GREEN

data = json.loads((HERE / "mio_suite.json").read_text())
systems = [s for s in data["systems"] if "points" in s]

SUB = str.maketrans("0123456789", "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089")


def pretty(label):
    """H2 -> H2 (subscript), OH- -> OH^-."""
    charge = ""
    core = label
    if label.endswith("-"):
        core, charge = label[:-1], "\u207b"
    elif label.endswith("+"):
        core, charge = label[:-1], "\u207a"
    return core.translate(SUB) + charge


def series(sysrec, tag):
    """Return (r, E) for points that produced an energy, SCF points converged only."""
    rs, es = [], []
    for p in sysrec["points"]:
        d = p.get(tag, {})
        if "e_tot" not in d:
            continue
        if tag == "scf" and not d.get("converged", False):
            continue
        rs.append(p["r"])
        es.append(d["e_tot"])
    return rs, es


def charges(sysrec, tag):
    rs, qs = [], []
    for p in sysrec["points"]:
        d = p.get(tag, {})
        if "q" not in d:
            continue
        if tag == "scf" and not d.get("converged", False):
            continue
        rs.append(p["r"])
        qs.append(d["q"][0])
    return rs, qs


def far_energy(sysrec, tag):
    d = sysrec.get("far", {}).get(tag, {})
    if "e_tot" not in d:
        return None
    if tag == "scf" and not d.get("converged", False):
        return None
    return d["e_tot"]


def minimum(rs, es):
    if not es:
        return None, None
    i = min(range(len(es)), key=lambda k: es[k])
    return rs[i], es[i]


# ================================================================ overview ===
n = len(systems)
ncol = 4
nrow = math.ceil(n / ncol)
fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 4.15 * nrow))
axes = axes.ravel()
fig.suptitle(
    "Diatomic Binding Curves \u2014 mio-1-1 Parameter Set\n"
    "Non-Self-Consistent ($H^0$) vs Self-Consistent Charge (SCC)",
    fontsize=20,
    fontweight="bold",
    y=0.995,
)

summary = []
for ax, s in zip(axes, systems):
    r0, e0 = series(s, "h0")
    r1, e1 = series(s, "scf")
    f0, f1 = far_energy(s, "h0"), far_energy(s, "scf")

    b0 = [e - f0 for e in e0] if f0 is not None else e0
    b1 = [e - f1 for e in e1] if f1 is not None else e1

    homonuclear = s["elements"][0] == s["elements"][1]

    ax.axhline(0.0, color="#999999", lw=0.9, ls="-")
    # H0 drawn as a wider halo so that where SCC coincides with it exactly
    # (every homonuclear case) both curves remain visible.
    ax.plot(r0, b0, "-", color=SLATE, lw=4.0, alpha=0.85, label="Non-SCC ($H^0$)")
    ax.plot(r1, b1, "-", color=NAVY, lw=2.2, label="SCC")

    m0r, m0e = minimum(r0, b0)
    m1r, m1e = minimum(r1, b1)
    if m0r is not None:
        ax.plot([m0r], [m0e], "s", color=SLATE, ms=8, zorder=6)
    if m1r is not None:
        ax.plot([m1r], [m1e], "o", color=CRIMSON, ms=9, zorder=6)
    ax.axvline(s["exp_re"], color=GREEN, ls="--", lw=1.7, alpha=0.9)

    ax.set_title(pretty(s["label"]), fontsize=16)
    ax.set_xlabel("$r$  (\u00c5)", fontsize=11.5)
    ax.set_ylabel("$E - E(6\\,\u00c5)$  (eV)", fontsize=11.5)
    ax.tick_params(labelsize=10)
    style.finish(ax)

    n_scf = len(r1)
    n_tot = len(s["points"])
    txt = f"SCC min  {m1r:.3f} \u00c5\nexp.  $r_e$  {s['exp_re']:.3f} \u00c5" if m1r else "no SCC data"
    ax.text(0.97, 0.95, txt, transform=ax.transAxes, ha="right", va="top",
            fontsize=10, color="#222222",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc", lw=0.9))
    if n_scf < n_tot:
        ax.text(0.97, 0.06, f"{n_tot - n_scf}/{n_tot} SCC pts unconverged",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                color=CRIMSON, fontweight="bold")
    elif homonuclear:
        ax.text(0.97, 0.06, "SCC $\\equiv H^0$ by symmetry",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5,
                color=GREEN, fontweight="bold")

    summary.append({
        "label": s["label"], "exp_re": s["exp_re"],
        "h0_r": m0r, "h0_e": m0e, "scf_r": m1r, "scf_e": m1e,
        "n_scf": n_scf, "n_tot": n_tot, "charge": s["charge"],
    })

for ax in axes[n:]:
    ax.set_visible(False)

handles = [
    Line2D([], [], color=SLATE, lw=2.4, label="Non-SCC ($H^0$ only)"),
    Line2D([], [], color=NAVY, lw=2.8, label="Self-consistent charge (SCC)"),
    Line2D([], [], color=SLATE, marker="s", ls="none", ms=9, label="Non-SCC minimum"),
    Line2D([], [], color=CRIMSON, marker="o", ls="none", ms=9, label="SCC minimum"),
    Line2D([], [], color=GREEN, ls="--", lw=2.0, label="Experimental $r_e$"),
]
fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=13,
           bbox_to_anchor=(0.5, 0.004), frameon=True, edgecolor="#cccccc")

fig.tight_layout(rect=[0.004, 0.045, 0.996, 0.955])
out = FIGS / "mio_diatomic_binding_curves.png"
fig.savefig(out, dpi=170)
plt.close(fig)
print(f"wrote {out}")


# ========================================================== bond lengths =====
ok = [s for s in summary if s["scf_r"] is not None and s["h0_r"] is not None]
ok.sort(key=lambda s: s["exp_re"])

fig, axes = plt.subplots(1, 2, figsize=(16.0, 6.8),
                         gridspec_kw={"width_ratios": [1.25, 1]})
fig.suptitle(
    "Equilibrium Bond Lengths \u2014 mio-1-1 Diatomics vs Experiment",
    fontsize=18,
    fontweight="bold",
    y=0.975,
)

ax = axes[0]
ys = list(range(len(ok)))
ax.plot([s["exp_re"] for s in ok], ys, "D", color=GREEN, ms=11, zorder=5,
        label="Experimental $r_e$")
ax.plot([s["h0_r"] for s in ok], ys, "s", color=SLATE, ms=10, zorder=4,
        label="Non-SCC ($H^0$) minimum")
ax.plot([s["scf_r"] for s in ok], ys, "o", color=NAVY, ms=10, zorder=6,
        label="SCC minimum")
for y, s in zip(ys, ok):
    lo = min(s["exp_re"], s["h0_r"], s["scf_r"])
    hi = max(s["exp_re"], s["h0_r"], s["scf_r"])
    ax.plot([lo, hi], [y, y], "-", color="#cfcfcf", lw=2.0, zorder=1)
ax.set_yticks(ys)
ax.set_yticklabels([pretty(s["label"]) for s in ok], fontsize=14)
ax.set_xlabel("Equilibrium separation  (\u00c5)")
ax.set_title("Absolute bond length")
ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.15)
style.finish(ax)

ax = axes[1]
err_h0 = [(s["h0_r"] - s["exp_re"]) / s["exp_re"] * 100 for s in ok]
err_scf = [(s["scf_r"] - s["exp_re"]) / s["exp_re"] * 100 for s in ok]
h = 0.36
ax.axvline(0.0, color="#444444", lw=1.4)
ax.barh([y + h / 2 for y in ys], err_h0, height=h, color=SLATE,
        label="Non-SCC ($H^0$)", zorder=3)
ax.barh([y - h / 2 for y in ys], err_scf, height=h, color=NAVY,
        label="SCC", zorder=3)
ax.set_yticks(ys)
ax.set_yticklabels([pretty(s["label"]) for s in ok], fontsize=14)
ax.set_xlabel("Deviation from experimental $r_e$  (%)")
ax.set_title("Relative deviation")
ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.15)
style.finish(ax)

fig.tight_layout(rect=[0, 0, 1, 0.94])
out = FIGS / "mio_equilibrium_bond_lengths.png"
fig.savefig(out, dpi=200)
plt.close(fig)
print(f"wrote {out}")


# ======================================================= charge transfer =====
het = [s for s in systems if s["elements"][0] != s["elements"][1]]
if het:
    ncol2 = 4
    nrow2 = math.ceil(len(het) / ncol2)
    fig, axes = plt.subplots(nrow2, ncol2, figsize=(5.0 * ncol2, 4.15 * nrow2))
    axes = axes.ravel() if len(het) > 1 else [axes]
    fig.suptitle(
        "Mulliken Charge Transfer in Heteronuclear Diatomics \u2014 mio-1-1\n"
        "Charge on the first-listed atom",
        fontsize=20,
        fontweight="bold",
        y=0.995,
    )
    for ax, s in zip(axes, het):
        r0, q0 = charges(s, "h0")
        r1, q1 = charges(s, "scf")
        ax.axhline(0.0, color="#999999", lw=1.0)
        ax.plot(r0, q0, "-", color=SLATE, lw=2.0)
        ax.plot(r1, q1, "-", color=NAVY, lw=2.4)
        ax.axvline(s["exp_re"], color=GREEN, ls="--", lw=1.7, alpha=0.9)
        ax.set_title(f"{pretty(s['label'])}   ($q$ on {s['elements'][0]})", fontsize=15)
        ax.set_xlabel("$r$  (\u00c5)", fontsize=11.5)
        ax.set_ylabel(f"$q_{{{s['elements'][0]}}}$  (e)", fontsize=11.5)
        ax.tick_params(labelsize=10)
        style.finish(ax)
    for ax in axes[len(het):]:
        ax.set_visible(False)
    handles2 = [
        Line2D([], [], color=SLATE, lw=2.4, label="Non-SCC ($H^0$ only)"),
        Line2D([], [], color=NAVY, lw=2.8, label="Self-consistent charge (SCC)"),
        Line2D([], [], color=GREEN, ls="--", lw=2.0, label="Experimental $r_e$"),
    ]
    fig.legend(handles=handles2, loc="lower center", ncol=3, fontsize=13,
               bbox_to_anchor=(0.5, 0.004), frameon=True, edgecolor="#cccccc")
    fig.tight_layout(rect=[0.004, 0.05, 0.996, 0.955])
    out = FIGS / "mio_charge_transfer.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"wrote {out}")


# ============================================================== text table ===
lines = []
lines.append(f"{'system':>7} {'chg':>4} {'exp r_e':>9} {'H0 min':>9} {'dev%':>7} "
             f"{'SCC min':>9} {'dev%':>7} {'SCC De':>9} {'conv':>8}")
for s in summary:
    if s["scf_r"] is None:
        lines.append(f"{s['label']:>7} {s['charge']:>+4d} {s['exp_re']:9.4f}"
                     f"{'  --- no converged SCC data ---':>45}")
        continue
    d0 = (s["h0_r"] - s["exp_re"]) / s["exp_re"] * 100
    d1 = (s["scf_r"] - s["exp_re"]) / s["exp_re"] * 100
    lines.append(
        f"{s['label']:>7} {s['charge']:>+4d} {s['exp_re']:9.4f} {s['h0_r']:9.4f} "
        f"{d0:+7.2f} {s['scf_r']:9.4f} {d1:+7.2f} {s['scf_e']:9.4f} "
        f"{s['n_scf']:>3d}/{s['n_tot']:<4d}"
    )
table = "\n".join(lines)
(HERE / "mio_summary_table.txt").write_text(table)
print()
print(table)
