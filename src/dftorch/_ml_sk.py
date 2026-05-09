"""
ML-based Slater-Koster integral prediction module.

Provides ``ml_eval_channel`` — a drop-in replacement for the cubic-spline
look-up used inside ``Slater_Koster_Pair_SKF_vectorized``.  Each call
evaluates a single SK channel for a masked subset of neighbor pairs via
a forward pass through a trained ``SKGraphNet`` model.

Key unit conventions
--------------------
*  The ML model predicts integral values in **Hartree** with the cosine
   cutoff already applied inside ``forward()``.
*  ``Slater_Koster_Pair_SKF_vectorized`` expects values in **eV**
   (because ``coeffs_tensor`` is stored in eV).
   → ``ml_eval_channel`` multiplies by ``HA_TO_EV = 27.21138625``.

*  The model stores its own ``rcut_by_Z_pair`` dict and ``Z_to_idx``
   mapping so that **no SKF files are needed at inference time**.

*  ``r_norm = dR / pair_type_rcut`` is the normalised distance.
   The cosine cutoff ``½(1 + cos(π · clamp(r_norm, max=1)))`` is
   applied inside the model's ``forward()`` method.

Usage
-----
The integration is handled automatically: ``_slater_koster_pair.py``
calls ``ml_eval_channel`` inside its ``_get_val_dR`` helper when
``ml_ctx`` is not ``None``.  The caller sets up ``ml_ctx`` in
``_h0ands.py :: H0_and_S_vectorized``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.data import Batch
    from torch_geometric.nn import TransformerConv, global_add_pool

    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False


# ── Channel ordering (same as in _bond_integral.py) ──────────────────────
# Within each 10-channel block the layout is:
#   0: dd_sigma, 1: dd_pi, 2: dd_delta,
#   3: pd_sigma, 4: pd_pi,
#   5: pp_sigma, 6: pp_pi,
#   7: sd_sigma,
#   8: sp_sigma,
#   9: ss_sigma
# Block 0 = Hamiltonian, Block 1 = Overlap  →  total 20 channels.
#
# The ML model's training data uses the same ordering via
# ``parse_channel_name`` which maps (is_H, l1, l2, bond) → row in the
# all_rows table.  We enumerate channels as (l1, l2, bond) tuples
# and assign each the canonical index above.

_CHANNEL_MAP: dict[tuple[int, int, int], int] = {
    #  (l1, l2, bond): index inside a 10-channel block
    (2, 2, 0): 0,  # dd_sigma
    (2, 2, 1): 1,  # dd_pi
    (2, 2, 2): 2,  # dd_delta
    (1, 2, 0): 3,  # pd_sigma
    (1, 2, 1): 4,  # pd_pi
    (1, 1, 0): 5,  # pp_sigma
    (1, 1, 1): 6,  # pp_pi
    (0, 2, 0): 7,  # sd_sigma
    (0, 1, 0): 8,  # sp_sigma
    (0, 0, 0): 9,  # ss_sigma
}

# Inverse: index → (l1, l2, bond)
_INDEX_TO_CHANNEL = {v: k for k, v in _CHANNEL_MAP.items()}


# ── Model definition (must match the notebook exactly) ───────────────────
class SKGraphNet(nn.Module):
    """
    Dual-head GNN for Slater-Koster integrals (Hamiltonian + Overlap).

    Architecture
    ------------
    * Shared backbone: Embedding → Linear → N × TransformerConv (residual +
      LayerNorm + SiLU)
    * Raw normalised distance scalar as edge feature (no RBF needed for
      2-node graphs with a single unique distance)
    * Symmetric readout: ``global_add_pool`` over 2-node graphs
    * Two MLP heads: ``head_H`` (Hamiltonian) and ``head_S`` (Overlap)
    * Cosine cutoff applied inside ``forward()``

    The model is self-contained: ``rcut_by_Z_pair`` and ``Z_to_idx`` are
    stored as attributes so no external SKF files are needed.

    Parameters
    ----------
    n_species : int
        Number of distinct atomic species.
    embed_dim, hidden, n_conv, n_heads : int
        Architecture hyper-parameters.
    n_readout_layers : int
        Number of linear layers in each readout MLP head (≥ 1).
    """

    def __init__(
        self,
        n_species: int,
        embed_dim: int = 8,
        hidden: int = 64,
        n_conv: int = 3,
        n_heads: int = 4,
        n_readout_layers: int = 3,
    ) -> None:
        """Initialise the dual-head graph network.

        Parameters
        ----------
        n_species : int
            Number of supported atomic species.
        embed_dim : int, default 8
            Embedding dimension for atomic species.
        hidden : int, default 64
            Hidden feature dimension used throughout the network.
        n_conv : int, default 3
            Number of TransformerConv blocks.
        n_heads : int, default 4
            Number of attention heads per TransformerConv block.
        n_readout_layers : int, default 3
            Number of linear layers in each readout head.
        """
        super().__init__()
        if not _HAS_PYG:
            raise ImportError(
                "torch_geometric is required for SKGraphNet. "
                "Install it with: pip install torch_geometric"
            )
        self.embed = nn.Embedding(n_species, embed_dim)
        self.n_readout_layers = n_readout_layers

        node_in = embed_dim + 3  # embed(Z) + one_hot(l)
        edge_dim = 1 + 3  # r_norm(1) + bond_oh(3)

        self.node_proj = nn.Linear(node_in, hidden)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_conv):
            self.convs.append(
                TransformerConv(
                    hidden,
                    hidden // n_heads,
                    heads=n_heads,
                    edge_dim=edge_dim,
                    dropout=0.0,
                )
            )
            self.norms.append(nn.LayerNorm(hidden))

        def _make_head():
            layers: list[nn.Module] = []
            for _ in range(n_readout_layers - 1):
                layers += [nn.Linear(hidden, hidden), nn.SiLU()]
            layers.append(nn.Linear(hidden, 1))
            head = nn.Sequential(*layers)
            nn.init.zeros_(head[-1].bias)
            nn.init.normal_(head[-1].weight, std=0.01)
            return head

        self.head_H = _make_head()
        self.head_S = _make_head()

        # Cutoff & species metadata (set after training via set_cutoff_data)
        self._Z_to_idx: Dict[int, int] = {}
        self._rcut_by_Z_pair: Dict[tuple, float] = {}

    # ── Cutoff / species metadata ────────────────────────────────────────
    def set_cutoff_data(
        self,
        rcut_by_Z_pair: Dict[tuple, float],
        Z_to_idx: Dict[int, int],
    ) -> None:
        """Store per-Z-pair cutoffs and Z→idx mapping inside the model.

        Parameters
        ----------
        rcut_by_Z_pair : {(Z1, Z2): float, ...}
            Cutoff in mixed Å-units for every ordered pair of atomic
            numbers.  Symmetric pairs should both be present.
        Z_to_idx : {Z: int, ...}
            Atomic-number → embedding-index mapping.
        """
        self._rcut_by_Z_pair = {
            (int(k[0]), int(k[1])): float(v) for k, v in rcut_by_Z_pair.items()
        }
        self._Z_to_idx = {int(k): int(v) for k, v in Z_to_idx.items()}

    @property
    def Z_to_idx(self) -> Dict[int, int]:
        return self._Z_to_idx

    @property
    def rcut_by_Z_pair(self) -> Dict[tuple, float]:
        return self._rcut_by_Z_pair

    def get_rcut(self, Z1: int, Z2: int) -> float:
        """Look up the cutoff (mixed Å-units) for an ordered atom pair."""
        return self._rcut_by_Z_pair[(int(Z1), int(Z2))]

    @staticmethod
    def _cosine_cutoff(r_norm: torch.Tensor) -> torch.Tensor:
        """½(1 + cos(π · clamp(r_norm, max=1))).  Zero for r ≥ r_cut."""
        return 0.5 * (1.0 + torch.cos(math.pi * r_norm.clamp(max=1.0)))

    def forward(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(pred_H, pred_S)`` each of shape ``(B,)`` **with cosine
        cutoff already applied**."""
        x = torch.cat([self.embed(batch.z), batch.l_oh], dim=-1)
        x = self.node_proj(x)

        # Edge features: r_norm(1) + bond_oh(3)
        r_scalar = batch.edge_attr[:, :1]
        bond_oh = batch.edge_attr[:, 1:]
        edge_feat = torch.cat([r_scalar, bond_oh], dim=-1)

        for conv, norm in zip(self.convs, self.norms):
            x = x + conv(x, batch.edge_index, edge_feat)
            x = norm(x)
            x = F.silu(x)
        g = global_add_pool(x, batch.batch)

        # Cosine cutoff from r_norm (one value per graph = per pair)
        # edge_attr layout: first B edges are src→dst (one per graph),
        # next B edges are dst→src.  Take the first B entries.
        n_edges = r_scalar.shape[0]
        r_per_graph = r_scalar[: n_edges // 2].squeeze(-1)  # (B,)
        cos_cut = self._cosine_cutoff(r_per_graph)

        return self.head_H(g).squeeze(-1) * cos_cut, self.head_S(g).squeeze(
            -1
        ) * cos_cut

    def forward_bare(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(bare_H, bare_S)`` each of shape ``(B,)`` **without**
        cosine cutoff.  Useful for gradient computation via autograd
        where we need ``d(bare)/d(r_norm)``."""
        x = torch.cat([self.embed(batch.z), batch.l_oh], dim=-1)
        x = self.node_proj(x)

        r_scalar = batch.edge_attr[:, :1]
        bond_oh = batch.edge_attr[:, 1:]
        edge_feat = torch.cat([r_scalar, bond_oh], dim=-1)

        for conv, norm in zip(self.convs, self.norms):
            x = x + conv(x, batch.edge_index, edge_feat)
            x = norm(x)
            x = F.silu(x)
        g = global_add_pool(x, batch.batch)

        return self.head_H(g).squeeze(-1), self.head_S(g).squeeze(-1)


# ── Helper: build PyG batch ──────────────────────────────────────────────
def _build_pyg_batch(
    Z1_idx: torch.Tensor,
    Z2_idx: torch.Tensor,
    l1_oh: torch.Tensor,
    l2_oh: torch.Tensor,
    bond_oh: torch.Tensor,
    r_norm: torch.Tensor,
) -> "Batch":
    """Construct a PyG ``Batch`` of two-node graphs from flat tensors.

    Parameters
    ----------
    Z1_idx, Z2_idx : torch.Tensor
        Species embedding indices of shape ``(B,)``.
    l1_oh, l2_oh : torch.Tensor
        One-hot angular-momentum encodings of shape ``(B, 3)``.
    bond_oh : torch.Tensor
        One-hot bond-type encodings of shape ``(B, 3)``.
    r_norm : torch.Tensor
        Normalised distances of shape ``(B, 1)``.

    Returns
    -------
    Batch
        PyG batch containing ``B`` independent two-node graphs.
    """
    B = Z1_idx.shape[0]
    device = Z1_idx.device
    z = torch.stack([Z1_idx, Z2_idx], dim=1).reshape(-1)
    l_oh = torch.stack([l1_oh, l2_oh], dim=1).reshape(-1, 3)
    base = torch.arange(B, device=device) * 2
    src = torch.cat([base, base + 1])
    dst = torch.cat([base + 1, base])
    edge_index = torch.stack([src, dst])
    edge_feat = torch.cat([r_norm, bond_oh], dim=-1)
    edge_attr = edge_feat.repeat(2, 1)
    batch_vec = torch.arange(B, device=device).repeat_interleave(2)
    return Batch(
        z=z, l_oh=l_oh, edge_index=edge_index, edge_attr=edge_attr, batch=batch_vec
    )


# ── Load a trained model ─────────────────────────────────────────────────
def load_ml_sk_model(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> Tuple[SKGraphNet, dict, int]:
    """Load a saved SKGraphNet checkpoint.

    The returned model is fully self-contained: ``model.rcut_by_Z_pair``
    and ``model.Z_to_idx`` are populated from the checkpoint so **no SKF
    files are needed at inference time**.

    Parameters
    ----------
    path : str or Path
        Path to the ``.pt`` checkpoint file.
    device : torch.device or str
        Device to load the model onto.

    Returns
    -------
    model : SKGraphNet
        The model in ``eval()`` mode.
    Z_to_idx : dict
        Mapping from atomic number Z to model embedding index.
    n_species : int
        Number of species the model was trained on.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg = checkpoint["model_config"]
    n_species = checkpoint["n_species"]
    model = SKGraphNet(
        n_species,
        embed_dim=cfg["embed_dim"],
        hidden=cfg["hidden"],
        n_conv=cfg["n_conv"],
        n_heads=cfg["n_heads"],
        n_readout_layers=cfg.get("n_readout_layers", 3),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    # Restore cutoff data stored alongside the model
    Z_to_idx = checkpoint["Z_to_idx"]
    rcut_by_Z_pair = checkpoint.get("rcut_by_Z_pair", {})
    if rcut_by_Z_pair:
        # Convert string-keyed dict back to tuple-keyed
        rcut = {
            (tuple(k) if isinstance(k, list) else k): v
            for k, v in rcut_by_Z_pair.items()
        }
        model.set_cutoff_data(rcut, Z_to_idx)
    else:
        model._Z_to_idx = {int(k): int(v) for k, v in Z_to_idx.items()}

    model.eval()
    return model, Z_to_idx, n_species


# ── Constants ─────────────────────────────────────────────────────────────
HA_TO_EV = 27.21138625


# ── Simple prediction interface ───────────────────────────────────────────
@torch.no_grad()
def predict(
    model: SKGraphNet,
    Z1: int,
    Z2: int,
    l1: int,
    l2: int,
    bond: int,
    r_bohr: float,
    is_H: bool = True,
    *,
    r_cut: Optional[float] = None,
) -> float:
    """Predict a single SK integral value (bare, without cosine cutoff).

    The model stores ``Z_to_idx`` and ``rcut_by_Z_pair`` internally, so
    **no external SKF data is required**.

    Parameters
    ----------
    model : SKGraphNet (with cutoff data set)
    Z1, Z2 : atomic numbers (e.g. 1, 6, 7, 8)
    l1, l2 : angular momenta (0=s, 1=p, 2=d)
    bond   : bond type (0=σ, 1=π, 2=δ)
    r_bohr : interatomic distance in **Bohr**
    is_H   : True → Hamiltonian, False → Overlap
    r_cut  : override cutoff (mixed Å-units).
             If *None*, uses ``model.get_rcut(Z1, Z2)``.

    Returns
    -------
    float – bare integral value in **Hartree** (no cosine cutoff)
    """
    BOHR_TO_ANG = 0.52917721
    Z_to_idx = model.Z_to_idx
    if r_cut is None:
        r_cut = model.get_rcut(Z1, Z2)
    dev = next(model.parameters()).device
    dt = next(model.parameters()).dtype
    # r_cut is in Å (mixed units) — convert r_bohr → Å to match
    r_ang = r_bohr * BOHR_TO_ANG
    r_norm = torch.tensor([[r_ang / r_cut]], dtype=dt, device=dev)
    z1i = torch.tensor([Z_to_idx[Z1]], dtype=torch.long, device=dev)
    z2i = torch.tensor([Z_to_idx[Z2]], dtype=torch.long, device=dev)
    l1_oh = F.one_hot(torch.tensor([l1], device=dev), 3).to(dt)
    l2_oh = F.one_hot(torch.tensor([l2], device=dev), 3).to(dt)
    b_oh = F.one_hot(torch.tensor([bond], device=dev), 3).to(dt)
    batch = _build_pyg_batch(z1i, z2i, l1_oh, l2_oh, b_oh, r_norm).to(dev)
    pH, pS = model(batch)
    # Divide out the cosine cutoff applied inside forward()
    r_ratio = min(r_ang / r_cut, 1.0)
    cos_cut = 0.5 * (1.0 + math.cos(math.pi * r_ratio))
    cos_cut = max(cos_cut, 1e-8)
    result = pH.item() / cos_cut if is_H else pS.item() / cos_cut
    return result


# ── Helper: build pair-type cutoff table ──────────────────────────────────
def build_pair_type_rcut(
    coeffs_tensor: torch.Tensor,
    R_orb: torch.Tensor,
) -> torch.Tensor:
    """Return a 1-D tensor of effective cutoffs (one per pair type).

    DFTorch stores the spline radial grid ``R_orb`` in **Ångström** units,
    but the pair distances ``dR_mskd`` are in **Bohr**.  Because
    ``searchsorted(R_orb, dR_mskd)`` compares these mixed-unit arrays
    directly, the effective cutoff *in dR_mskd (Bohr) units* for each
    pair type is the **endpoint** of the last non-zero spline interval.

    The cubic spline coefficient at index *i* covers the interval
    ``R_orb[i] → R_orb[i+1]``.  So the cutoff is ``R_orb[last_nz + 1]``,
    i.e. ``(last_nz + 2) * dr`` where ``dr = R_orb[1] - R_orb[0]``.

    For a 3ob SKF with ``dr_Bohr = 0.02`` and ``npts = 550`` data rows
    (549 read + 1 zero-padded), the last non-zero interval is 548 and
    the cutoff is ``(548 + 2) * dr = 550 * 0.02 * 0.52917721`` in the
    mixed Å-units, corresponding to exactly 11.0 Bohr.

    Parameters
    ----------
    coeffs_tensor : Tensor, shape (n_pair_types, n_intervals, 20, 4)
    R_orb : Tensor, shape (n_intervals,) or (n_intervals + 1,)

    Returns
    -------
    rcut : Tensor, shape (n_pair_types,)  — effective cutoffs
    """
    n_pt = coeffs_tensor.shape[0]
    dr = (R_orb[1] - R_orb[0]).item()
    rcut_list = []
    for pt in range(n_pt):
        # Find last non-zero interval for this pair type
        nz = coeffs_tensor[pt].abs().sum(dim=(1, 2))  # (n_intervals,)
        last_nz = nz.nonzero(as_tuple=True)[0]
        if last_nz.numel() > 0:
            # Endpoint of the last non-zero interval
            rcut_list.append((last_nz[-1].item() + 2) * dr)
        else:
            rcut_list.append(R_orb[-1].item())
    return torch.tensor(rcut_list, dtype=R_orb.dtype, device=R_orb.device)


# ── Core: evaluate one SK channel lazily for masked pairs ─────────────────
def ml_eval_channel(
    ml_ctx: dict,
    mask: torch.Tensor | slice,
    channel: int,
    SH_shift: int,
    direction: str = "IJ",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate a single SK channel for the masked subset of pairs.

    This is the drop-in replacement for the cubic-spline look-up used
    inside ``Slater_Koster_Pair_SKF_vectorized``.  It is called once per
    channel from ``_get_val_dR`` (see ``_slater_koster_pair.py``).

    The model's ``forward()`` already applies the cosine cutoff, so this
    function divides it out and scales Hartree → eV to match spline
    convention.  Gradients ``dval`` are computed via ``torch.autograd``
    through the ``forward_bare()`` path.

    Parameters
    ----------
    ml_ctx : dict
        Context dictionary built by ``H0_and_S_vectorized`` containing:
        ``model``, ``TYPE``, ``neighbor_I``, ``neighbor_J``,
        ``dR_mskd`` (Å, mixed units).
    mask : slice or BoolTensor
        Which pairs to evaluate.
    channel : int  (0-9)
        SK channel index within the H or S block.
    SH_shift : int  (0 = H, 1 = S)
        Selects Hamiltonian or Overlap head of the model.
    direction : ``'IJ'`` or ``'JI'``
        Atom ordering for the pair (affects Z1/Z2 assignment).

    Returns
    -------
    val : Tensor, shape (N,)   — integral values in **eV**
    dval : Tensor, shape (N,)  — d(val)/d(r) in **eV/Å**
    """
    model = ml_ctx["model"]
    Z_to_idx = model.Z_to_idx
    rcut_dict = model.rcut_by_Z_pair
    TYPE = ml_ctx["TYPE"]
    nI = ml_ctx["neighbor_I"]
    nJ = ml_ctx["neighbor_J"]
    dR_mskd = ml_ctx["dR_mskd"]  # Å (mixed units, same as R_orb)

    dev = next(model.parameters()).device
    dtype = dR_mskd.dtype

    # Select the masked subset
    r_masked = dR_mskd[mask]
    N = r_masked.shape[0]
    if N == 0:
        z = torch.zeros(0, dtype=dtype, device=dR_mskd.device)
        return z, z

    # Atom types (always IJ for cutoff lookup, direction only affects Z1/Z2)
    Z_I = TYPE[nI[mask]]
    Z_J = TYPE[nJ[mask]]

    # Per-pair cutoff (looked up from model's rcut_by_Z_pair dict)
    r_cut = torch.tensor(
        [rcut_dict[(int(zi), int(zj))] for zi, zj in zip(Z_I, Z_J)],
        dtype=dtype,
        device=dev,
    )

    # Atom types in the requested direction
    if direction == "IJ":
        Z1, Z2 = Z_I, Z_J
    else:
        Z1, Z2 = Z_J, Z_I

    # Map atomic numbers to model embedding indices
    Z1_idx = torch.tensor([Z_to_idx[int(z)] for z in Z1], dtype=torch.long, device=dev)
    Z2_idx = torch.tensor([Z_to_idx[int(z)] for z in Z2], dtype=torch.long, device=dev)

    # Channel one-hot encoding
    l1v, l2v, bv = _INDEX_TO_CHANNEL[channel]
    l1_oh = F.one_hot(torch.full((N,), l1v, dtype=torch.long, device=dev), 3).to(dtype)
    l2_oh = F.one_hot(torch.full((N,), l2v, dtype=torch.long, device=dev), 3).to(dtype)
    bond_oh = F.one_hot(torch.full((N,), bv, dtype=torch.long, device=dev), 3).to(dtype)

    # ── Autograd-enabled forward pass for bare values + gradients ────────
    # r_norm requires grad so we can differentiate the bare prediction
    # w.r.t. normalised distance and then chain-rule to physical distance.
    r_norm = (r_masked.to(dev) / r_cut).unsqueeze(1)  # (N, 1)
    r_norm = r_norm.detach().requires_grad_(True)

    with torch.enable_grad():
        batch = _build_pyg_batch(Z1_idx, Z2_idx, l1_oh, l2_oh, bond_oh, r_norm)
        batch = batch.to(dev)
        bare_H, bare_S = model.forward_bare(batch)

        # Select H or S head (bare values in Hartree / raw SKF units)
        bare = bare_S if SH_shift == 1 else bare_H

        # Compute d(bare)/d(r_norm) via autograd
        (d_bare_d_rnorm,) = torch.autograd.grad(
            bare.sum(),
            r_norm,
            create_graph=False,
        )
        # d_bare_d_rnorm has shape (N, 1) — squeeze to (N,)
        d_bare_d_rnorm = d_bare_d_rnorm.squeeze(1)

    # Chain rule: d(bare)/d(r) = d(bare)/d(r_norm) / r_cut
    # Units: [Hartree / Å]
    d_bare_d_r = d_bare_d_rnorm / r_cut

    # bare is in Hartree, convert to eV
    val = (bare.detach() * HA_TO_EV).to(dtype).to(dR_mskd.device)
    dval = (d_bare_d_r.detach() * HA_TO_EV).to(dtype).to(dR_mskd.device)
    return val, dval
