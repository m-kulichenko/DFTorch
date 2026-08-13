# The Krylov Accelerator — Investigation Record

**Written:** 2026-08-11
**Status:** Diagnosis complete. No repair attempted. No source file changed by this
investigation.
**Why this file exists:** The 2026-08-04 human ruling in `06-RESEARCH.md` ("Scope rulings",
item 1) deferred repairing `kernel_update_lr` to a later phase and had Phase 6 switch it off for
f systems in the interim. That phase has no number yet. This record holds every measurement
behind the diagnosis so the later phase can plan from evidence instead of re-deriving it.

**Reproduce:** `experiments/diatomic_scans/krylov_probe.py`, one test per finding below. Every
table in this file is that script's literal output. All runs: CPU, float64, `torch 2.10.0+cpu`,
`COUL_METHOD="FULL"`, Eu-N from `tests/f_orbital_data/`.

---

## What the accelerator is, in plain words

The self-consistent charge loop is a fixed-point search. Given a guess at how much charge sits
on each atom, one pass of the loop builds the Hamiltonian that guess implies, diagonalizes it,
and reads off the charges that come *out*. Convergence means output equals input.

Write the map the loop performs as `q_out(q)`, and the miss as

    f(q) = q_out(q) - q          # the loop calls this `Res`

The plain mixers take a fraction of that miss and step: `q_new = q + alpha * Res`
(linear mixing), or a cleverer weighted combination of recent misses (Anderson/DIIS). They know
nothing about *why* the miss is what it is, so they creep.

The Krylov accelerator instead builds a local **linear model of the map itself** and jumps
straight to where that model says the miss vanishes. That is Newton's method:

    q_new = q - J^-1 Res     where  J = d(q_out)/dq - I

The whole difficulty is that `J` is a matrix of size (number of atoms)², and forming it
explicitly would mean one diagonalization per atom, per pass. The accelerator never forms it.
Two ideas avoid that:

**1. It only needs `J` applied to a vector, never `J` itself.** For a trial direction `v` in
charge space, `J v` is a *first-order response*: how much do the output charges shift if the
input charges are nudged along `v`? `calc_dq` (`_xl_tools.py:349`) computes this analytically
from the eigenvectors and eigenvalues the current pass already produced, via
`fermi_prt_D1_only` — perturbation theory on the density matrix. That costs matrix
multiplications, not a new diagonalization.

**2. It works in the small subspace the residual actually lives in.** Starting from the
preconditioned residual, it generates directions `v_1, v_2, ...`, orthogonalizes each against
the previous ones (modified Gram-Schmidt, `_xl_tools.py:572-582`), and records what the map does
to each. That growing set of directions is the *Krylov subspace*. Inside it, the accelerator
solves a tiny least-squares problem — `O Y = rhs` at `_xl_tools.py:645-649`, of size (rank ×
rank), typically 1 to 8 — for the combination `V Y` that best cancels the residual, and returns
that as the step. "Low-rank" is exactly this: a full Newton solve projected onto a handful of
directions.

The loop stops adding directions when the projected residual `Fel` drops below `KRYLOV_TOL`
(default 1e-6) or the rank hits `KRYLOV_MAXRANK` (default 20).

**Preconditioning** is the `KK0` matrix multiplying the residual at `_xl_tools.py:558` and
`:637`. In the SCF path `KK0 = -SCF_ALPHA * I` (`_scf.py:569`), so it is just a scalar — a
cheap approximate inverse-Jacobian used to make the first search direction point somewhere
sensible. In the MD path it carries real structure accumulated across steps.

**The important consequence, and the whole story below:** Newton's method is only as good as
the assumption that the map is locally close to linear. Where it is, the accelerator is
excellent. Where it is not, Newton is *worse than doing nothing*, because it takes the linear
model seriously and jumps far.

### The special case that matters here

For a neutral two-atom system, total charge is conserved exactly (`q` sums to zero by
construction: `q = -Znuc + DS` and `sum(DS) = 2*Nocc`). So the residual lives in a
**one-dimensional** subspace spanned by `(1, -1)`. The Krylov machinery therefore reduces to a
single exact Newton step, with `J` a scalar. That is what makes Eu-N such a clean probe: there
is no subspace approximation left to blame for anything.

---

## Finding 1 — The accelerator's arithmetic is correct

`krylov_probe.py jacobian`. At r = 3.60 Å, comparing the accelerator's analytic Jacobian
against central finite differences on the real charge map, and its step against the exact
Newton step:

```
   call       |Res|  q_old(Eu)      J_fd  J_analytic   |Krylov|   |Newton|   |linear|
      1   3.450e+00   -0.36470   -2.3857     -2.3857  1.446e+00  1.446e+00  3.450e-01
      2   6.242e+00   -1.38714   -1.1058     -1.1058  5.645e+00  5.645e+00  6.242e-01
      3   7.928e+00    2.60471   -1.0011     -1.0011  7.919e+00  7.919e+00  7.928e-01
      4   1.131e+01   -2.99511   -1.0011     -1.0011  1.129e+01  1.129e+01  1.131e+00
      5   1.130e+01    4.99107   -1.0002     -1.0002  1.130e+01  1.130e+01  1.130e+00
      6   1.131e+01   -3.00051   -1.0010     -1.0010  1.130e+01  1.130e+01  1.131e+00
      7   1.130e+01    4.99117   -1.0002     -1.0002  1.130e+01  1.130e+01  1.130e+00
      8   1.131e+01   -3.00051   -1.0010     -1.0010  1.130e+01  1.130e+01  1.131e+00

  final q = [4.999488, -4.999488], E = 197.984953 eV
```

`calc_dq` / `fermi_prt_D1_only` agree with finite differences to every digit shown, and the
returned step equals the exact Newton step. **There is no arithmetic bug in `kernel_update_lr`,
`calc_dq` or `fermi_prt_D1_only`.** Any repair that starts by hunting for one will not find it.

Read rows 4-8: `q_old(Eu)` alternates between −3.00 and +4.99 forever. That is a period-2
limit cycle, and rows 3 onward show why — `J` has collapsed to −1.

**Why `J = -1` is fatal.** `J = d(q_out)/dq - 1`, so `J = -1` means `d(q_out)/dq = 0`: the
output has stopped responding to the input. The Newton step then becomes `Res / (-1) = -Res`,
i.e. `q_new = q_out(q_old)` — completely **unmixed** fixed-point iteration, the least stable
scheme available. Newton does not merely stop helping in a flat region; it actively degenerates
into the worst possible mixer.

---

## Finding 2 — The problem it is being asked to solve is a cliff

`krylov_probe.py chargemap`. The charge map for Eu-N at r = 3.60 Å, evaluated directly:

```
    q_in(Eu)   q_out(Eu)  q_out-q_in  d q_out/d q_in
      -5.000      5.0001     10.0001       -0.000068
      -4.000      5.0000      9.0000       -0.000196
      -3.000      4.9995      7.9995       -0.001042
      -2.000      4.9887      6.9887       -0.071243
      -1.000      3.0080      4.0080       -0.021843
      -0.500      1.7564      2.2564      -35.017913
      -0.365     -2.8035     -2.4385       -1.390253
       0.000     -2.9633     -2.9633       -0.118653
       0.500     -2.9898     -3.4898       -0.021865
       1.000     -2.9963     -3.9963       -0.007591
       2.000     -3.0002     -5.0002       -0.001924
       3.000     -3.0014     -6.0014       -0.000765
       4.000     -3.0020     -7.0020       -0.000381
       5.000     -3.0023     -8.0023       -0.000218
```

Converged answer: **q_Eu = −0.466688**, which sits *inside* the near-vertical segment. Between
q_in = −0.50 and −0.365 the output swings 4.56 electrons; the slope reaches −35. Everywhere
else the map is flat, saturating at q_out ≈ +5.00 from one side and −3.00 from the other.

So the loop must find a fixed point on a cliff, flanked on both sides by plateaus where Newton
degenerates. That is the whole failure in one table.

---

## Finding 3 — The cliff is a degenerate manifold pinned at the Fermi level

`krylov_probe.py spectrum`. Eigenvalues, occupations and atomic-orbital character at the
converged charge:

```
  at the converged charge q_Eu = -0.4667,  mu = -2.4243 eV
  rank    E (eV)    occ   Eu s   Eu p   Eu d   Eu f    N s    N p
     2   -2.7923  0.986   0.00   0.07   0.10   0.03   0.00   0.80
     3   -2.4364  0.535   0.00   0.00   0.01   0.36   0.00   0.63  <== at mu
     4   -2.4364  0.535   0.00   0.00   0.01   0.36   0.00   0.63  <== at mu
     5   -2.4034  0.440   0.00   0.00   0.00   1.00   0.00   0.00  <== at mu
     6   -2.4034  0.440   0.00   0.00   0.00   1.00   0.00   0.00  <== at mu
     7   -2.4034  0.440   0.00   0.00   0.00   1.00   0.00   0.00  <== at mu
     8   -2.4034  0.440   0.00   0.00   0.00   1.00   0.00   0.00  <== at mu
     9   -2.3916  0.406   0.00   0.00   0.01   0.97   0.00   0.02  <== at mu
    10   -2.3853  0.389   0.00   0.00   0.01   0.65   0.00   0.35  <== at mu
    11   -2.3853  0.389   0.00   0.00   0.01   0.65   0.00   0.35  <== at mu
    12   -1.5544  0.000   0.00   0.00   1.00   0.00   0.00   0.00
```

Nine levels spanning **0.051 eV**, straddling the chemical potential at −2.4243 eV, every one
fractionally occupied between 0.389 and 0.535. At T_ELECTRONIC = 1000 K the thermal window is
kT = 0.0862 eV — wider than the entire manifold. Five of the nine are pure Eu 4f (weight 1.00);
the rest are 4f mixed with N 2p.

The physical mechanism: f orbitals are spatially contracted, so they barely hybridize with the
partner atom. That is exactly what preserves their degeneracy instead of splitting them into
bonding/antibonding pairs the way s and p states split. Seven near-degenerate levels then share
one partially-filled reservoir at the Fermi level, and shifting the Eu on-site potential by a
hair repartitions occupancy across the whole block at once. That is the −35 slope of Finding 2.

**Confirmation by removing the cause.** `krylov_probe.py smear` widens the Fermi window over
the manifold by raising the electronic temperature:

```
  Krylov ON:
    r (A)         T=1000K         T=2000K         T=4000K         T=8000K
     2.40     DIVERGE          OK  it=17       OK  it=16       OK  it=14
     2.50     DIVERGE          OK  it=19       OK  it=16       OK  it=14
     2.80     DIVERGE          OK  it=17       OK  it=15       OK  it=13
     3.10     DIVERGE          OK  it=14       OK  it=13       OK  it=13
     3.20     DIVERGE         DIVERGE          OK  it=14       OK  it=13
     3.30     DIVERGE         DIVERGE          OK  it=14       OK  it=13
     3.40     DIVERGE         DIVERGE          OK  it=14       OK  it=13
     3.50     DIVERGE         DIVERGE          OK  it=14       OK  it=13
     3.60     DIVERGE         DIVERGE          OK  it=14       OK  it=13

  Krylov OFF:
     2.40      OK  it=29       OK  it=29       OK  it=26       OK  it=23
     3.60      OK  it=28       OK  it=26       OK  it=24       OK  it=20
```

At 4000 K every previously-failing separation converges **with the accelerator on, in 13-16
passes against 20-26 without it**. Smooth the map and the accelerator does its job well.

> **This is a diagnostic, not a proposed fix.** See "What is not the fix" below.

---

## Finding 4 — The reproduction

`krylov_probe.py scan`. All 21 Eu-N separations, accelerator on versus off:

```
   r (A)                        Krylov ON                       Krylov OFF
    1.60   OK   it=12   E=   -0.410807     OK   it=13   E=   -0.410807
    1.70   OK   it=12   E=   -6.409461     OK   it=14   E=   -6.409461
    1.80   OK   it=12   E=   -9.439071     OK   it=14   E=   -9.439071
    1.90   OK   it=12   E=  -10.719673     OK   it=17   E=  -10.719673
    2.00   OK   it=13   E=  -10.997697     OK   it=19   E=  -10.997697
    2.10   OK   it=14   E=  -10.717228     OK   it=20   E=  -10.717228
    2.20   OK   it=15   E=  -10.129834     OK   it=27   E=  -10.129834
    2.30   OK   it=15   E=   -9.424637     OK   it=28   E=   -9.424637
    2.40   DIV  it=100  E=   20.206143     OK   it=29   E=   -8.696156
    2.50   DIV  it=100  E=   18.172684     OK   it=29   E=   -7.977710
    2.60   OK   it=18   E=   -7.283848     OK   it=29   E=   -7.283848
    2.70   OK   it=22   E=   -6.636423     OK   it=27   E=   -6.636423
    2.80   DIV  it=100  E=   29.601449     OK   it=28   E=   -6.059714
    2.90   OK   it=15   E=   -5.557098     OK   it=29   E=   -5.557098
    3.00   OK   it=15   E=   -5.090483     OK   it=27   E=   -5.090483
    3.10   DIV  it=100  E=  185.618523     OK   it=29   E=   -4.659414
    3.20   DIV  it=100  E=  188.335010     OK   it=29   E=   -4.275004
    3.30   DIV  it=100  E=  190.911366     OK   it=28   E=   -3.931544
    3.40   DIV  it=100  E=  193.369418     OK   it=28   E=   -3.625795
    3.50   DIV  it=100  E=  195.723861     OK   it=28   E=   -3.356250
    3.60   DIV  it=100  E=  197.984953     OK   it=28   E=   -3.122492

  diverged with Krylov on: 9/21;  with Krylov off: 0/21
```

Matches the 9-of-21 count recorded in `_krylov_params_for_f_interim` (`ESDriver.py:117-121`).
Failures land at +18 to +198 eV against correct answers of −3 to −11 eV, with q_Eu at ±5 — a
fully-ionized nonsense state, not slow convergence. Where both converge the energies are
identical to the digits printed.

Note also the 8 separations where the accelerator **works and is worth having**: at r = 2.20 it
converges in 15 passes against 27, at r = 2.30 in 15 against 28.

---

## Finding 5 — It is when the accelerator engages, not that it engages

`krylov_probe.py handover`. At r = 3.60 Å, varying `KRYLOV_START`. Anderson alone needs 28
passes; the handover residual is read off the Krylov-off run at that pass:

```
   KRYLOV_START  |Res| at handover    result  passes     E_tot (eV)
             10          3.330e+00   DIVERGE     100     197.984953
             12          3.527e-01        OK      15      -3.122492
             14          8.575e-01        OK      18      -3.122492
             16          5.774e-01        OK      20      -3.122492
             18          1.486e+00        OK      21      -3.122492
             20          1.814e-02        OK      23      -3.122492
             22          1.664e-02        OK      24      -3.122492
             24          9.588e-03        OK      26      -3.122492
             26          2.727e-03        OK      28      -3.122492
            off          0.000e+00        OK      28      -3.122492
```

Two things follow. First, the failure is entirely about the residual at handover — engaging at
pass 12 instead of 10 turns a divergence into a 15-pass convergence. Second, **the accelerator
is genuinely valuable**: `KRYLOV_START = 12` converges in 15 passes against Anderson's 28. A
repair that simply deletes it gives up a near-2× speedup on exactly the systems that need it.

---

## Finding 6 — The commented-out trust region is not sufficient

`krylov_probe.py trustreg`. `_xl_tools.py:664-668` carries a trust region, commented out, that
caps the step at 1.25× the plain linear-mixing step. Enabling it verbatim:

```
  diverged: plain Krylov 9/21, with trust region 4/21, off 0/21
```

It converts r = 2.40, 2.50, 2.80, 3.50 and 3.60 to convergence, and leaves r = 3.10, 3.20, 3.30
and 3.40 diverging (their energies improve from +185…+195 eV to +45…+65 eV, still garbage). A
step-length cap cannot detect that the linear model is *invalid*; it only makes an invalid step
shorter. **Uncommenting those five lines is not the repair.**

---

## Finding 7 — This is not f-specific, and the interim guard is too narrow

`krylov_probe.py mio` — ordinary mio-1-1 diatomics never reach the branch at all (it engages at
pass 11):

```
    system       Krylov OFF        Krylov ON  Krylov iters
       N-N         OK  it=1         OK  it=1             0
       C-O         OK  it=8         OK  it=8             0
       C-C         OK  it=1         OK  it=1             0
       O-O         OK  it=1         OK  it=1             0
       H-H         OK  it=1         OK  it=1             0
       N-H         OK  it=5         OK  it=5             0
```

`krylov_probe.py tutorial` — the pre-existing systems in `experiments/`, run with the
tutorial's own parameter values:

```
              system  atoms   orbs     Krylov OFF      Krylov ON  kry its      dE (eV)
           COORD.xyz     20     32       OK  it=9       OK  it=9        0     0.00e+00
    COORD_8WATER.xyz     24     48      OK  it=11      OK  it=11        1     0.00e+00
   COORD_ACETONE.xyz     10     22      OK  it=11      OK  it=11        1     0.00e+00
         m30_o60.xyz    270    720     DIV it=100     DIV it=100      713     7.74e+04
            C840.xyz    840   3360      OK  it=19      OK  it=19       12    -7.28e-12
```

Three things this settles.

**(a) The accelerator has been exercised, and works, on a large system.** C840 — 840 atoms,
3360 orbitals — runs Krylov across passes 11-19 for 12 iterations and reproduces the
Krylov-off energy to **7.3e-12 eV**. (`_scf.py:1845` still carries the developer's
`# KK = torch.load(".../tests/KK_C840.pt") # For testing purposes`, so C840 looks to have been
the original validation system.) It did not *accelerate* — 19 passes either way — but it is
correct.

**(b) But it has never been stressed in the SCF path.** The water box and acetone reach pass 11
and take exactly 1 Krylov iteration on an already-converged residual. The one test that sets
`KRYLOV_START` (`tests/test_scf.py:47`, value 5, CH4) is worse than untested: CH4 takes 6
passes, so `kernel_update_lr` is called once, with `|Res| = 7.3e-09`; `K0Res = 0.1 × 7.3e-09`
trips the `norm_dr < 1e-9` guard at `_xl_tools.py:566`, prints `zero norm_dr`, breaks at
`krylov_rank = 0` and returns the plain linear step. **Zero Krylov iterations ever execute**,
and the test asserts only `torch.isfinite(e_tot)`.

**(c) `m30_o60.xyz` fails with the accelerator OFF too — and it contains no f orbitals.**
A single non-SCF diagonalization shows why:

```
  m30_o60: 270 atoms, 720 orbitals
    Fermi level mu = -5.6137 eV,  kT = 0.0862 eV
    eigenvalues within 3kT of mu        : 85
    orbitals with fractional occupation : 94
       E =  -5.7691 eV   occ = 0.8585
       E =  -5.7452 eV   occ = 0.8214
       E =  -5.6905 eV   occ = 0.7091
       E =  -5.6428 eV   occ = 0.5837
       E =  -5.6205 eV   occ = 0.5197
```

85 levels inside one thermal window of the Fermi level, 94 fractionally occupied — the O₂ π\*
manifolds. **Structurally the same pathology as Eu's 4f shell, in a plain mio-1-1 system.**

Consequence for the interim guard: `_krylov_params_for_f_interim` (`ESDriver.py:109-171`) keys
on `const.n_orb[TYPE] == 16`, so it protects f systems and nothing else. The condition that
actually predicts trouble is *degenerate levels at the Fermi level*, which is measurable
(`sum(f * (1 - f))`, or a count of levels within a few kT of mu) and is not what is being
tested.

---

## Finding 8 — Where it is designed to be used, it works well

`krylov_probe.py md`. `MD.py:913` calls the same routine once per XL-BOMD step. Twelve steps on
COORD.xyz:

```
  step     |Res| in  |step| out   ratio   ETOT low-rank     ResErr    ETOT NoRank     ResErr
     0    7.694e-03   6.186e-03   0.804      -75.686205   0.00e+00     -75.686205   0.00e+00
     1    8.780e-03   7.525e-03   0.857      -75.686097   1.72e-03     -75.686097   1.72e-03
     4    2.926e-03   2.187e-03   0.747      -75.689107   6.99e-04     -75.688409   9.94e-03
     8    4.104e-03   3.534e-03   0.861      -75.696131   8.36e-04     -75.689984   6.64e-03
    11    2.011e-03   1.521e-03   0.756      -75.697715   4.55e-04     -75.693888   1.17e-02

  energy drift:                low-rank -0.011510 eV     NoRank -0.007683 eV
  mean XL-BOMD residual error: low-rank 9.069e-04      NoRank 7.195e-03
```

**8× lower residual error than bypassing it.** The reason is visible in the first column: the
residuals handed to it are 2e-3 to 9e-3, because the extended-Lagrangian dynamics keeps the
auxiliary charges near the ground state by construction. The output step is 0.75-0.89 of the
input — a modest correction, never an amplification. `MD.py` also asks for `KRYLOV_TOL_MD =
1e-2`, three orders looser than the SCF path's `KRYLOV_TOL = 1e-6`.

Contrast the SCF path at Eu-N r = 3.60 Å: `|Res| = 3.45` in, a step **4× larger** out.

**The accelerator is not broken. It is a near-solution method being called far from the
solution.**

---

## What is not the fix

**Raising `T_ELECTRONIC` is not a fix.** It is a physical parameter, and moving it computes a
different problem. `krylov_probe.py temp`, Krylov off at every point so all numbers are
converged answers:

```
                       kT (eV):       0.0862      0.1723      0.3447      0.6894
                         T (K):         1000         2000         4000         8000

  E_tot at r = 1.90 A     (eV):     -10.7197     -11.5082     -13.3200     -17.4711
  E_tot at r = 2.00 A     (eV):     -10.9977     -11.8053     -13.6352     -17.8856
  E_tot at r = 2.10 A     (eV):     -10.7172     -11.5328     -13.3734     -17.7593
  E_tot at r = 2.60 A     (eV):      -7.2838      -8.0871     -10.1200     -15.4124
  E_tot at r = 3.00 A     (eV):      -5.0905      -6.0831      -8.3579     -14.0899
  E_tot at r = 3.60 A     (eV):      -3.1225      -4.2291      -6.7451     -12.8838

        well depth min(E) (eV):     -10.9977     -11.8053     -13.6352     -17.8856
         r at that minimum (A):         2.00         2.00         2.00         2.00
        E(3.60) - E(min)  (eV):       7.8752       7.5762       6.8901       5.0017
  max |dE| vs the 1000 K curve:       0.0000       1.1066       3.6226       9.7613
```

Going to 4000 K moves total energies by up to **3.62 eV** and changes the Eu-N dissociation
energy from 7.875 eV to 6.890 eV. At 8000 K the shift is 9.76 eV and the well depth falls to
5.00 eV. The minimum stays at 2.00 Å, so it is not *arbitrary* — but a 1 eV change in binding
energy is not a convergence setting.

**Uncommenting the trust region is not the fix** — Finding 6, 9/21 → 4/21.

**Deleting the accelerator is not the fix** — Finding 5, it is worth a near-2× speedup, and
Finding 8, MD depends on it.

---

## What a repair would have to do

Not planned, not decided — recorded so the later phase starts from the evidence.

**Solver layer.** Three changes, in descending order of leverage, all supported above:

1. **Gate on residual magnitude, not pass count.** `_scf.py:690` tests `it > KRYLOV_START`.
   Finding 5 shows the residual is what decides the outcome. A gate like
   `ResNorm < KRYLOV_ENTRY_TOL` keeps the speedup and removes every failure in the scan.
2. **Reject a step that does not reduce the residual**, falling back to the Anderson step for
   that pass. This is what the trust region cannot do (Finding 6) — it caps length without
   testing validity.
3. **Detect the degenerate case explicitly.** `J ≈ -1` means the map is flat and Newton has no
   information. The Jacobian is already computed inside `kernel_update_lr`, so this is nearly
   free.

**Guard layer.** Replace the `n_orb == 16` test in `_krylov_params_for_f_interim` with a
measurement of Fermi-level degeneracy, so `m30_o60`-class systems are covered too (Finding 7c).

**Physics layer — needs a human ruling.** Eu-N is being solved closed-shell restricted, with
seven 4f levels fractionally occupied at ~0.44 each. Real Eu(II) is high-spin, S = 7/2, seven
*unpaired* f electrons. Forcing restricted pairing across a degenerate manifold is what creates
the near-discontinuous charge map in the first place. The physically correct treatment is
spin-polarized, and it is currently refused: `ESDriver.py:102` raises
`FSpinPolarizationUnsupportedError` for any atom with `n_orb == 16` when `UNRESTRICTED` is set,
per Phase 4 decision D-12. **The SCF instability may be the solver correctly reporting that the
model it has been handed is ill-conditioned, rather than a mixer defect.** Worth settling
before a phase is spent on the mixer.

---

## Separate defect noticed, not the cause

`_xl_tools.py:554-555` allocates the Krylov basis without a dtype:

```python
vi = torch.zeros(n_atoms, krylov_maxrank, device=S.device)
fi = torch.zeros(n_atoms, krylov_maxrank, device=S.device)
```

These inherit the global default dtype. A caller who builds float64 tensors without setting
`torch.set_default_dtype(torch.float64)` globally gets a float32 Krylov basis silently truncating
a float64 calculation. Every script in `experiments/diatomic_scans/` sets the global default, so
it is masked everywhere it is currently exercised. `dtype=S.dtype` on both lines is the whole
fix. Same pattern at `_xl_tools.py:749` and `:999` (`kernel_update_lr_os`,
`kernel_update_lr_batch`) and `sedacs/sedacs_interface.py:708`.

---

## Relationship to existing records

- **`ROADMAP.md`, Phase 6 scope note (2026-08-04)** says the cause "is **not** the f orbitals".
  Refined, not contradicted: there is no f-orbital *bug*, and the accelerator's arithmetic is
  correct — but the Eu 4f manifold is what makes the map non-linear, and the same pathology
  appears in O₂-containing systems with no f orbitals at all (Finding 7c).
- **`WINDOWS.md` entry 6** (open) records the shell-resolved `MAGNETIC_HUBBARD_LDEP` loop
  settling at only 12 of 21 Eu-N separations while the per-atom loop settles 21 of 21. That is a
  **different code path** — `use_krylov` is false whenever `shell_resolved` is true
  (`_scf.py:690`), so entry 6 is an Anderson-mixing failure, not this one. Findings 2 and 3
  almost certainly bear on it: the same manifold is at the Fermi level either way. Not
  investigated here.
- **`06-RESEARCH.md`, "Scope rulings" item 1** is the deferral this record serves.
