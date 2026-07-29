# Phase 3 Source Lock — f Angular Formula Tables

Record for blocking checkpoint task `03-01-02`. Status: **APPROVED**.

## 1. Source identity

| Item | Value |
|---|---|
| Title | Slater–Koster tables for f electrons |
| Authors | Katsuhiko Takegahara, Yoshio Aoki, Akira Yanase (Dept. of Physics, Tohoku University, Sendai, Japan) |
| Journal | *J. Phys. C: Solid State Phys.* **13** (1980) 583–588 |
| DOI | `10.1088/0022-3719/13/4/016` |
| Received | 24 April 1979 |
| Local file | `K_Takegahara_1980_J._Phys._C__Solid_State_Phys._13_583.pdf` (repo root, untracked) |
| DOI matches plan | **YES** — this is the exact source named in `03-01-PLAN.md` `user_setup` |

Page images extracted for reading (CCITT G4 scans, no usable text layer for math):
`<scratchpad>/tk/tkpage01.png` … `tkpage06.png` = J. Phys. C pp. 584, 585, 586, 587, 588, 583.

### Source rejected: Sharma

Sharma, *Phys. Rev. B* **19**, 2813 (1979), DOI `10.1103/PhysRevB.19.2813`
(`f_orbital_SlaterKosterAngularTransformations.pdf`) was evaluated first and **rejected
as the primary source**. Takegahara *et al.* p588 state the reason directly:

> "Recently Sharma (1979) has calculated the Slater–Koster integrals including f and g
> orbitals using a method similar to ours. However, the table does not contain all
> integrals and the orbitals do not have cubic symmetry. There is a misprint in the
> energy integral `E_{xy,x³-3x²y}`. The coefficient of (dfσ) should be
> `(√30/16)[1 - l⁴ + 10m²l² - 5m⁴ - 2n² + n⁴]`."

Independently confirmed before finding that passage: Sharma's Table I is titled "List of
**some** of the energy integrals", carries ~20 f-relevant rows (mostly diagonal) against
the ~112 needed, and tabulates the **tesseral** basis, which would have required a full
7×7 change-of-basis into Structure.py order rather than a permutation.

Sharma remains useful as a **cross-check oracle only**, with that misprint correction applied.

## 2. AO convention — exact match to Structure.py

Takegahara Table 1 (p585) defines the cubic harmonics. `C_f = (7/16π)^(1/2) r^(-3)`.

| Takegahara | Irrep | Cartesian form | Structure.py label | Structure index |
|---|---|---|---|---|
| T₁ᵤ α | T₁ᵤ | `x(5x² − 3r²)` | `fx3` | 9 |
| T₁ᵤ β | T₁ᵤ | `y(5y² − 3r²)` | `fy3` | 10 |
| T₁ᵤ γ | T₁ᵤ | `z(5z² − 3r²)` | `fz3` | 11 |
| T₂ᵤ ξ | T₂ᵤ | `x(y² − z²)` | `fx_y2_z2` | 12 |
| T₂ᵤ η | T₂ᵤ | `y(z² − x²)` | `fy_z2_x2` | 13 |
| T₂ᵤ ζ | T₂ᵤ | `z(x² − y²)` | `fz_x2_y2` | 14 |
| A₂ᵤ | A₂ᵤ | `xyz` | `fxyz` | 15 |

**The paper's basis IS Structure.py's basis.** `Structure.py:16-24` was written against this
paper. Consequences:

- `PAPER_TO_STRUCTURE_F_PERMUTATION` is a pure reordering (paper lists `xyz` first,
  Structure.py lists it last). No basis rotation.
- `PAPER_TO_STRUCTURE_F_SIGN` is all `+1`.
- Plan decision D-05's permutation+sign adapter model is **sufficient for this source**
  (it would NOT have been for Sharma).

Spherical-harmonic expansions (for verification, Table 1 p585):

```
A₂ᵤ  : 2√15 C_f xyz        = i√(1/2)(−Y₃₂ + Y₃₋₂)
T₁ᵤα : C_f x(5x² − 3r²)    = ¼ (−√5 Y₃₃ + √3 Y₃₁ − √3 Y₃₋₁ + √5 Y₃₋₃)
T₁ᵤβ : C_f y(5y² − 3r²)    = ¼i(−√5 Y₃₃ − √3 Y₃₁ − √3 Y₃₋₁ − √5 Y₃₋₃)
T₁ᵤγ : C_f z(5z² − 3r²)    = Y₃₀
T₂ᵤξ : √15 C_f x(y² − z²)  = ¼ ( √3 Y₃₃ + √5 Y₃₁ − √5 Y₃₋₁ − √3 Y₃₋₃)
T₂ᵤη : √15 C_f y(z² − x²)  = ¼i(−√3 Y₃₃ + √5 Y₃₁ + √5 Y₃₋₁ − √3 Y₃₋₃)
T₂ᵤζ : √15 C_f z(x² − y²)  = [to transcribe]
```

d and p conventions also match Structure.py: `E_g u = 3z²−r²` → `dz2`,
`E_g v = x²−y²` → `dx2_y2`, `T_2g ξ/η/ζ = yz/zx/xy` → `dyz`/`dzx`/`dxy`.

## 3. Direction cosines and channel order

`l = sinβ cosα`, `m = sinβ sinα`, `n = cosβ` (Eq. 13, p586) — direction cosines of the
vector **X** from the first atom to the second, in the original (x,y,z) frame. Same
convention already used by the s/p/d code path.

Channels: `|k| = 0,1,2,3` designated `σ, π, δ, φ`. So `sf` has 1 channel, `pf` 2, `df` 3,
`ff` 4 — matching the 40-channel `_bond_integral._CHANNELS` layout.

## 4. Completeness rule

Table 2 prints a subset. p586:

> "All the results are represented by l, m, and n and are given in table 2. **The entries
> not given in the table can be found by cyclically permuting the coordinates and
> direction cosines.**"

Cyclic permutation `x→y→z→x` together with `l→m→n→l` generates every unprinted entry.
This makes the source **complete** for all s-f, p-f, d-f and f-f blocks.

## 5. Built-in verification

The paper supplies its own correctness check (Eqs. 14–15, p586): setting `(j'jk₁) = δ_{j'j}`
must yield

```
(Φ_{j'k'}(r+X) | V | Φ_{jk}(r)) = δ_{j'j} δ_{k'k}
(Ψ_{j'λ'}(r+X) | V | Ψ_{jλ}(r)) = δ_{j'j} δ_{λ'λ}
```

> "This orthogonal relation is useful in checking the results."

Operationally: substitute `(ffσ)=(ffπ)=(ffδ)=(ffφ)=1` (and likewise per shell pair) and the
assembled 7×7 f block must equal the identity for any `(l,m,n)` on the unit sphere. This is
a strong, cheap, per-entry regression test and should gate the transcription.

## 6. Known misprints to avoid

- **Lendi (1974)** — Takegahara footnote p583: in `E_{D₂ᵤ,Γ}` the coefficient of `(dfπ)`
  should read `−(1/4)√15[(l²−m²)(l²−3m²) + n² − 1]`. Do not source from Lendi.
- **Sharma (1979)** — `E_{xy,x³−3x²y}` coefficient of `(dfσ)` should be
  `(√30/16)[1 − l⁴ + 10m²l² − 5m⁴ − 2n² + n⁴]`.

## 7. Transcribed entries (verified against page images)

Read directly from `tkpage03.png` / `tkpage04.png`. **These are the entries confirmed so
far; the remainder of d-f and f-f still requires systematic transcription in task 03-01-03.**

### s–f (p586)

```
E_s,xyz          = √15 · l m n (sfσ)
E_s,x(5x²−3r²)   = ½ l (5l² − 3) (sfσ)
E_s,x(y²−z²)     = ½√15 · l (m² − n²) (sfσ)
```

### p–f (p586)

```
E_x,xyz          = √15 l²mn (pfσ) − √(5/2)(3l² − 1) mn (pfπ)
E_x,x(5x²−3r²)   = ½ l²(5l² − 3)(pfσ) − √(3/8)(5l² − 1)(l² − 1)(pfπ)
E_x,y(5y²−3r²)   = ½ lm(5m² − 3)(pfσ) − √(3/8) lm(5m² − 1)(pfπ)
E_x,z(5z²−3r²)   = ½ ln(5n² − 3)(pfσ) − √(3/8) ln(5n² − 1)(pfπ)
E_x,x(y²−z²)     = ½√15 l²(m² − n²)(pfσ) − √(5/8)(3l² − 1)(m² − n²)(pfπ)
E_x,y(z²−x²)     = ½√15 lm(n² − l²)(pfσ) − √(5/8) lm[3(n² − l²) + 2](pfπ)
E_x,z(x²−y²)     = ½√15 ln(l² − m²)(pfσ) − √(5/8) ln[3(l² − m²) − 2](pfπ)
```

### d–f (p586, first four rows)

```
E_xy,xyz         = √45 l²m²n (dfσ) − √(5/2) n(6l²m² + n² − 1)(dfπ) + n(3l²m² + 2n² − 1)(dfδ)
E_xy,x(5x²−3r²)  = ½√3 l²m(5l² − 3)(dfσ) − √(3/8) m(5l² − 1)(2l² − 1)(dfπ) + ½√15 l²m(l² − 1)(dfδ)
E_xy,y(5y²−3r²)  = ½√3 lm²(5m² − 3)(dfσ) − √(3/8) l(5m² − 1)(2m² − 1)(dfπ) + ½√15 lm²(m² − 1)(dfδ)
E_xy,z(5z²−3r²)  = ½√3 lmn(5n² − 3)(dfσ) − √(3/2) lmn(5n² − 1)(dfπ) + ½√15 lmn(n² + 1)(dfδ)
```

### f–f (p587, sample rows)

```
E_xyz,xyz        = 15 l²m²n²(ffσ) + … (ffπ)
                 + [1 − 4(l²m² + m²n² + n²l²) + 9l²m²n²](ffδ)
                 + (3/2)(1 − l²)(1 − m²)(1 − n²)(ffφ)

E_xyz,z(5z²−3r²) = ½√15 lmn²(5n² − 3)(ffσ) − ¼√15 lm(3n² − 1)(5n² − 1)(ffπ)
                 + ½√15 lmn²(3n² − 1)(ffδ) + ¼√15 lm(1 − n⁴)(ffφ)

E_xyz,z(x²−y²)   = (15/2) lmn²(l² − m²)(ffσ) − (5/4) lm(l² − m²)(9n² − 1)(ffπ)
                 + ½ lm(l² − m²)(9n² − 4)(ffδ) + ¾ lm(l² − m²)(1 − n²)(ffφ)

E_z(5z²−3r²),x(5x²−3r²)
                 = ¼ ln(5l² − 3)(5n² − 3)(ffσ) − (3/8) ln(5l² − 1)(5n² − 1)(ffπ)
                 + (15/4) ln(l²n² − m²)(ffδ) + (5/8) ln(3m² − l²n²)(ffφ)
```

## 8. Outstanding work for task 03-01-03

1. Transcribe the remaining d–f rows (p586 bottom, p587 top) and the full f–f block (p587).
2. Implement the cyclic-permutation generator for unprinted entries.
3. Apply the permutation into Structure.py AO order (identity signs).
4. Gate every entry on the Eq. 14/15 orthogonality identity before trusting it.
5. Cross-check a sample against Sharma Table I (with the p588 misprint correction applied).

## 9. Licensing note

Both PDFs are copyrighted articles sitting untracked at the repo root. They must **not** be
committed. Add them to `.gitignore` or move them outside the working tree.
