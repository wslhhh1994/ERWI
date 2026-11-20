"""
SeismicMoE – Mixture-of-Experts for ERWI velocity inversion
------------------------------------------------------------
Pipeline:
  1. MoE feature extraction  — FNO + BiGRU + CNN, soft-gated            → (xn, H, L)
  2. Bottleneck compression  — residual H→H/2→H                         → (xn, H, L)
  3. Cross-trace attention   — segment-level axial attention across xn   → (xn, H, L)
                               well column injected as global token per window
  4. Freq-aware decoder      — blend low-freq trend + high-freq detail  → (xn, 2, L)

Input shape : (xn, 2, 5, L)  — [vp, vs] LSRTM images, full trace set with 5-trace window
Output shape: (xn, 2, 1, L)  — [vp, vs] prediction for every trace

well_profile : (2, L)   normalised Δvp/Δvs at the well trace
well_lsrtm   : (10, L)  normalised LSRTM input at the well trace

The well column is appended as a global token to every segment window, so every trace
can attend directly to well information regardless of lateral distance. No kriging weight
or exponential decay is applied — the attention mechanism learns the weighting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────
# 2D Rotary Positional Encoding (RoPE)
# ─────────────────────────────────────────────────────────────────

def apply_2d_rope(
    x: torch.Tensor,
    ix_arr: torch.Tensor,
    n_z_pairs: int = -1,
    base: float = 10000.0,
) -> torch.Tensor:
    """
    2D RoPE applied to (B, C, L).

    First n_z_pairs channel-pairs encode depth position (0..L-1, implicit).
    Remaining channel-pairs encode trace x-position (ix_arr, per sample).

    ix_arr  : (B,) long — trace index in the model space (0 .. n_x-1)
    n_z_pairs: -1 → C // 4  (e.g. 2 for C=10, leaving 3 pairs for x)
    base    : RoPE frequency base (default 10 000, same as original Transformer)
    """
    B, C, L = x.shape
    if n_z_pairs < 0:
        n_z_pairs = C // 4          # 2 for C=10
    n_x_pairs = C // 2 - n_z_pairs  # 3 for C=10
    z_end = n_z_pairs * 2

    device, dtype = x.device, x.dtype
    out = x.clone()

    # ── z-RoPE: channels [0 : z_end] encode depth position ────────
    if n_z_pairs > 0:
        d_z    = n_z_pairs * 2
        freq_z = 1.0 / (base ** (
            torch.arange(n_z_pairs, device=device, dtype=dtype) * 2.0 / d_z
        ))                                                   # (n_z_pairs,)
        z_pos   = torch.arange(L, device=device, dtype=dtype)
        theta_z = z_pos.unsqueeze(0) * freq_z.unsqueeze(1)  # (n_z_pairs, L)
        cos_z   = theta_z.cos().unsqueeze(0)                 # (1, n_z_pairs, L)
        sin_z   = theta_z.sin().unsqueeze(0)

        x1 = out[:, 0:z_end:2, :]                           # (B, n_z_pairs, L)
        x2 = out[:, 1:z_end:2, :]
        out[:, 0:z_end:2, :] = x1 * cos_z - x2 * sin_z
        out[:, 1:z_end:2, :] = x1 * sin_z + x2 * cos_z

    # ── x-RoPE: channels [z_end : C] encode trace x-position ──────
    if n_x_pairs > 0:
        d_x    = n_x_pairs * 2
        freq_x = 1.0 / (base ** (
            torch.arange(n_x_pairs, device=device, dtype=dtype) * 2.0 / d_x
        ))                                                   # (n_x_pairs,)
        theta_x = (
            ix_arr.to(dtype=dtype).unsqueeze(1) * freq_x.unsqueeze(0)
        )                                                    # (B, n_x_pairs)
        cos_x = theta_x.cos().unsqueeze(2)                  # (B, n_x_pairs, 1)
        sin_x = theta_x.sin().unsqueeze(2)

        x1 = out[:, z_end::2, :]                            # (B, n_x_pairs, L)
        x2 = out[:, z_end + 1::2, :]
        out[:, z_end::2, :]     = x1 * cos_x - x2 * sin_x
        out[:, z_end + 1::2, :] = x1 * sin_x + x2 * cos_x

    return out


# ─────────────────────────────────────────────────────────────────
# Expert 1 – FNO (Fourier Neural Operator)
# ─────────────────────────────────────────────────────────────────
class SpectralConv1d(nn.Module):
    """Single Fourier layer: multiply low-frequency modes by a learnable weight."""

    def __init__(self, channels: int, modes: int):
        super().__init__()
        self.modes = modes
        scale = 1.0 / channels
        self.weight = nn.Parameter(
            scale * torch.randn(channels, channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        xf = torch.fft.rfft(x)                              # (B, C, L//2+1)
        out = torch.zeros(B, C, L // 2 + 1, dtype=torch.cfloat, device=x.device)
        out[:, :, :self.modes] = torch.einsum(
            "bim,iom->bom", xf[:, :, :self.modes], self.weight
        )
        return torch.fft.irfft(out, n=L)                    # (B, C, L)


class FNOExpert(nn.Module):
    def __init__(self, in_ch: int, hidden: int, out_ch: int, modes: int = 12):
        super().__init__()
        self.proj_in  = nn.Conv1d(in_ch, hidden, 1)
        self.spec1    = SpectralConv1d(hidden, modes)
        self.mix1     = nn.Conv1d(hidden, hidden, 1)
        self.spec2    = SpectralConv1d(hidden, modes)
        self.mix2     = nn.Conv1d(hidden, hidden, 1)
        self.proj_out = nn.Sequential(nn.GELU(), nn.Conv1d(hidden, out_ch, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.proj_in(x))
        x = F.gelu(self.spec1(x) + self.mix1(x))
        x = self.spec2(x) + self.mix2(x)
        return self.proj_out(x)                             # (B, out_ch, L)


# ─────────────────────────────────────────────────────────────────
# Expert 2 – Bidirectional GRU
# ─────────────────────────────────────────────────────────────────
class RNNExpert(nn.Module):
    def __init__(self, in_ch: int, hidden: int, out_ch: int):
        super().__init__()
        self.proj_in  = nn.Linear(in_ch, hidden)
        self.gru      = nn.GRU(hidden, hidden, num_layers=1,
                               batch_first=True, bidirectional=True)
        self.proj_out = nn.Linear(hidden * 2, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1).contiguous()                # (B, L, C)
        x = F.gelu(self.proj_in(x))
        with torch.backends.cudnn.flags(enabled=False):    # 绕开 cuDNN GRU 内核，防止 CUDNN_STATUS_EXECUTION_FAILED
            x, _ = self.gru(x.contiguous())                # (B, L, 2H)
        return self.proj_out(x).permute(0, 2, 1)           # (B, out_ch, L)


# ─────────────────────────────────────────────────────────────────
# Expert 3 – CNN + Self-Attention along depth
# ─────────────────────────────────────────────────────────────────
class DepthAttention(nn.Module):
    def __init__(self, channels: int, heads: int = 2):
        super().__init__()
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt = x.permute(0, 2, 1)                            # (B, L, C)
        a, _ = self.attn(xt, xt, xt)
        return self.norm(xt + a).permute(0, 2, 1)          # (B, C, L)


class CNNExpert(nn.Module):
    def __init__(self, in_ch: int, hidden: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2), nn.GELU(),
        )
        self.attn  = DepthAttention(hidden, heads=2)
        self.conv2 = nn.Conv1d(hidden, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.attn(x)
        return self.conv2(x)                                # (B, out_ch, L)


# ─────────────────────────────────────────────────────────────────
# Soft Gating
# ─────────────────────────────────────────────────────────────────
class SoftGate(nn.Module):
    def __init__(self, in_ch: int, n_experts: int, pool_size: int = 8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(pool_size)
        self.mlp  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch * pool_size, 32), nn.GELU(),
            nn.Linear(32, n_experts),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.mlp(self.pool(x)), dim=-1)   # (B, n_experts)


# ─────────────────────────────────────────────────────────────────
# Bottleneck – residual feature compression
# ─────────────────────────────────────────────────────────────────
class Bottleneck(nn.Module):
    """Residual bottleneck: compress H → H/2 → H, then InstanceNorm."""

    def __init__(self, hidden: int):
        super().__init__()
        mid = max(hidden // 2, 1)
        self.compress = nn.Conv1d(hidden, mid, 1)
        self.expand   = nn.Conv1d(mid, hidden, 1)
        self.norm     = nn.InstanceNorm1d(hidden, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.expand(F.gelu(self.compress(x))))


# ─────────────────────────────────────────────────────────────────
# CrossTraceAttention – segment-level axial attention across traces
# ─────────────────────────────────────────────────────────────────
class CrossTraceAttention(nn.Module):
    """
    Axial attention across the trace (x) dimension with global well token injection.

    For each depth slice, traces within overlapping segment windows attend jointly.
    The well column is appended as a global token to every window, giving every
    trace direct access to well information regardless of lateral distance —
    no kriging weight or exponential decay is applied.

    Operates on (xn, H, L): permutes to (L, xn, H), segments xn, attends, permutes back.
    """
    _WELL_CH = 12   # 10 (lsrtm) + 2 (velocity) concatenated

    def __init__(self, hidden: int, heads: int = 4,
                 segment_size: int = 64, overlap: int = 16):
        super().__init__()
        self.S        = segment_size
        self.stride   = segment_size - overlap
        # Encodes joint [lsrtm‖velocity] well input → H-dim token per depth point
        self.well_enc = nn.Linear(self._WELL_CH, hidden)
        self.attn     = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm1    = nn.LayerNorm(hidden)
        self.ff       = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden)
        )
        self.norm2    = nn.LayerNorm(hidden)

    def forward(
        self,
        feat: torch.Tensor,
        well_profile: "torch.Tensor | None" = None,
        well_lsrtm:   "torch.Tensor | None" = None,
        gate: "torch.Tensor | None" = None,
    ) -> torch.Tensor:
        """
        feat         : (xn, H, L)
        well_profile : (2, L)   normalised Δvp/Δvs at well trace — velocity token
        well_lsrtm   : (10, L)  normalised LSRTM at well trace   — structure token
                       None → zero-pad the structure channels
        gate         : (xn,) float in [0, 1] — per-trace attention bias weight
                       = cosine_sim(trace_LSRTM, well_LSRTM) × exp(-dist/range)
                       Added as log(gate) to attention scores towards the well token,
                       so distant/dissimilar traces are naturally suppressed.
                       None → no bias (all traces attend to well equally).
        Returns      : (xn, H, L)  updated features
        """
        xn, H, L = feat.shape

        # Build per-depth well token: (L, H)
        well_z = None
        if well_profile is not None:
            if well_lsrtm is not None:
                well_in = torch.cat([well_lsrtm, well_profile], dim=0)   # (12, L)
            else:
                pad     = torch.zeros(10, L, device=feat.device, dtype=feat.dtype)
                well_in = torch.cat([pad, well_profile], dim=0)           # (12, L)
            well_z = self.well_enc(well_in.T)                             # (L, H)

        # (xn, H, L) → (L, xn, H): treat depth as batch, trace index as sequence
        fz    = feat.permute(2, 0, 1).contiguous()           # (L, xn, H)
        delta = torch.zeros_like(fz)                          # (L, xn, H)
        cnt   = torch.zeros(xn, device=feat.device, dtype=feat.dtype)

        for i in range(0, xn, self.stride):
            j   = min(i + self.S, xn)
            seg = fz[:, i:j, :]                               # (L, j-i, H)

            if well_z is not None:
                wt  = well_z.unsqueeze(1)                     # (L, 1, H)
                seq = torch.cat([seg, wt], dim=1)             # (L, j-i+1, H)

                # Additive attention bias: log(gate) on well-token interactions.
                # Adding log(g) to score before softmax ≡ multiplying unnorm. weight
                # by g, so gate≈1 (nearby+similar) → full constraint,
                # gate≈0 (far/dissimilar) → well token suppressed.
                if gate is not None:
                    n      = j - i
                    log_g  = torch.log(gate[i:j].clamp(min=1e-6))  # (n,)
                    bias   = torch.zeros(n + 1, n + 1,
                                         device=feat.device, dtype=feat.dtype)
                    bias[:n, n] = log_g   # trace → well column
                    bias[n, :n] = log_g   # well → trace row (symmetric)
                else:
                    bias = None
            else:
                seq  = seg
                bias = None

            # Self-attention: all trace tokens + well token attend to each other
            a, _ = self.attn(seq, seq, seq, attn_mask=bias)
            # Accumulate only the trace-slot outputs (discard well token position)
            delta[:, i:j, :] += a[:, :j - i, :]
            cnt[i:j] += 1

        # Average overlapping contributions, then apply transformer sub-layers
        delta = delta / cnt.view(1, xn, 1)
        fz    = self.norm1(fz + delta)
        fz    = self.norm2(fz + self.ff(fz))

        return fz.permute(1, 2, 0)                           # (xn, H, L)


# ─────────────────────────────────────────────────────────────────
# FreqAwareDecoder – separate low/high-freq output branches
# ─────────────────────────────────────────────────────────────────
class FreqAwareDecoder(nn.Module):
    """
    Produce a final 2-channel (vp, vs) output by blending:
      • Low-freq branch : pool → project → upsample  (smooth velocity trend)
      • High-freq branch: direct 1×1 projection      (layer-correlated detail)

    The learned `alpha` parameter controls the blend per model state.
    """

    def __init__(self, hidden: int, out_ch: int, pool_size: int = 16):
        super().__init__()
        self.pool      = nn.AdaptiveAvgPool1d(pool_size)
        self.low_proj  = nn.Conv1d(hidden, out_ch, 1)
        self.high_proj = nn.Conv1d(hidden, out_ch, 1)
        self.alpha     = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, L)
        _, _, L = x.shape
        low = F.interpolate(
            self.low_proj(self.pool(x)),
            size=L, mode='linear', align_corners=False,
        )                                                    # (B, out_ch, L)
        high = self.high_proj(x)                            # (B, out_ch, L)
        a = torch.sigmoid(self.alpha)
        return a * low + (1.0 - a) * high                   # (B, out_ch, L)


# ─────────────────────────────────────────────────────────────────
# SeismicMoE – main model
# ─────────────────────────────────────────────────────────────────
class SeismicMoE(nn.Module):
    """
    Input : (xn, 2, 5, L)  — full trace set; [vp, vs] LSRTM 5-trace window per trace
    Output: (xn, 2, 1, L)  — [vp, vs] prediction for every trace

    Architecture:
      InstanceNorm → 2D RoPE → MoE (FNO/GRU/CNN, soft-gated)
      → Bottleneck → CrossTraceAttention → FreqAwareDecoder

    well_profile : (2, L)  normalised Δvp/Δvs at well trace.
                   None disables well injection (backward-compatible).
    """
    _IN_CH  = 10   # 2 params × 5 traces
    _HIDDEN = 24
    _OUT_CH = 2    # vp, vs
    _MODES  = 24   # FNO kept frequency modes (L=312→157 bins, 24/157≈15%; L=175→88 bins, 24/88≈27%)
    _CHUNK  = 128  # traces per chunk for per-trace experts (controls DepthAttention peak memory)

    def __init__(self):
        super().__init__()
        C, H, O, M = self._IN_CH, self._HIDDEN, self._OUT_CH, self._MODES

        self.norm        = nn.InstanceNorm1d(C, affine=True)
        self.experts     = nn.ModuleList([
            FNOExpert(C, H, H, M),
            RNNExpert(C, H, H),
            CNNExpert(C, H, H),
        ])
        self.gate        = SoftGate(C, n_experts=3)
        self.bottleneck  = Bottleneck(H)
        self.cross_trace = CrossTraceAttention(H, heads=4, segment_size=64, overlap=16)
        self.decoder     = FreqAwareDecoder(H, O)

    def forward(
        self,
        x: torch.Tensor,
        ix: "torch.Tensor | None" = None,
        well_profile: "torch.Tensor | None" = None,
        well_lsrtm:   "torch.Tensor | None" = None,
        well_ix: "int | None" = None,
        krig_range_x: float = 60.0,
    ) -> torch.Tensor:
        """
        x            : (xn, 2, 5, L)  — full trace set (all xn traces together)
        ix           : (xn,) long     — trace x-index (0..n_x-1); None skips RoPE
        well_profile : (2, L)         — normalised [Δvp, Δvs] at well trace
        well_lsrtm   : (10, L)        — normalised LSRTM input at well; None → zero-pad
        well_ix      : int or None    — index of the well trace in x (0..xn-1)
                       When provided, a per-trace gate is computed:
                         sim_t  = cosine similarity of trace t vs well trace (LSRTM space)
                         idw_t  = exp(-|t - well_ix| / krig_range_x)
                         gate_t = sim_t × idw_t ∈ [0, 1]
                       gate is injected as log-scale attention bias towards the well token
                       inside CrossTraceAttention, so nearby+similar traces get more
                       well constraint while distant/dissimilar traces are suppressed.
        krig_range_x : exponential IDW decay range in traces (default 60)
        """
        B, _, T, L = x.shape                                 # B = xn (all traces)
        x = x.reshape(B, self._IN_CH, L)                    # (xn, 10, L)
        x = self.norm(x)
        if ix is not None:
            x = apply_2d_rope(x, ix)

        # ── Per-trace gate: LSRTM cosine similarity × inverse distance weight ──
        gate = None
        if well_ix is not None and well_profile is not None:
            x_flat    = x.reshape(B, -1)                                # (xn, 10*L)
            well_flat = x_flat[well_ix].unsqueeze(0).expand(B, -1)     # (xn, 10*L)
            sim  = F.cosine_similarity(x_flat, well_flat, dim=1)        # (xn,) ∈ [-1,1]
            sim  = (sim + 1.0) / 2.0                                    # → [0, 1]
            dist = torch.arange(B, device=x.device, dtype=x.dtype)
            idw  = torch.exp(-torch.abs(dist - well_ix) / krig_range_x) # (xn,)
            gate = (sim * idw).clamp(0.0, 1.0)                         # (xn,)

        # ── MoE: soft-weighted fusion of expert feature maps (chunked) ──
        # Per-trace experts have no cross-trace dependency; process in chunks
        # to cap DepthAttention's O(xn × L²) memory from ~362 MB to ~30 MB.
        feat_list = []
        for i in range(0, B, self._CHUNK):
            xi = x[i:i + self._CHUNK]
            wi = self.gate(xi)                               # (chunk, 3)
            oi = torch.stack([e(xi) for e in self.experts], dim=1)  # (chunk, 3, H, L)
            feat_list.append((wi[:, :, None, None] * oi).sum(dim=1))
        feat = torch.cat(feat_list, dim=0)                  # (xn, H, L)

        # ── Bottleneck: residual compression ──────────────────────────
        feat = self.bottleneck(feat)

        # ── Cross-trace attention: propagate well info across all traces
        feat = self.cross_trace(feat, well_profile, well_lsrtm, gate=gate)

        # ── Freq-aware decoder: low-freq trend + high-freq detail ─────
        out = self.decoder(feat)                             # (xn, 2, L)
        return out.unsqueeze(2)                              # (xn, 2, 1, L)


# ─────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────
def build_model(device: str = "cuda") -> tuple:
    model     = SeismicMoE().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    return model, optimizer, scheduler


def _w1_depth(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1D Wasserstein distance (CDF-L1) 沿深度方向.

    pred / target: (B, 2, 1, L)
    将速度剖面视为深度维上的质量分布，W1 衡量层位置误差，
    与 ERWI 的走时敏感性一致，弥补 RMSE/SSIM 仅考虑振幅误差的不足。
    """
    p = pred.squeeze(2)    # (B, 2, L)
    t = target.squeeze(2)  # (B, 2, L)
    # softmax 转概率分布，cumsum 得到 CDF
    p_cdf = torch.cumsum(torch.softmax(p, dim=-1), dim=-1)
    t_cdf = torch.cumsum(torch.softmax(t, dim=-1), dim=-1)
    return torch.mean(torch.abs(p_cdf - t_cdf))


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y_true: torch.Tensor,
    well_profile: "torch.Tensor | None" = None,
    well_lsrtm:   "torch.Tensor | None" = None,
    smooth_weight: float = 1e-3,
    w_w1: float = 0.5,
) -> float:
    """One gradient step on a full snapshot. Returns scalar loss value.

    Loss = RMSE + TV_smooth + w_w1 × W1_depth
    W1_depth (创新点5): 1D Wasserstein 沿深度方向，惩罚层位置误差。
    """
    model.train()
    optimizer.zero_grad()

    y_pred = model(x, well_profile=well_profile, well_lsrtm=well_lsrtm)  # (xn, 2, 1, L)

    loss = torch.sqrt(F.mse_loss(y_pred, y_true) + 1e-12)   # RMSE

    if smooth_weight > 0:
        diff = y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]
        loss = loss + smooth_weight * diff.abs().mean()

    if w_w1 > 0:
        loss = loss + w_w1 * _w1_depth(y_pred, y_true)

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()


# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    model, opt, _ = build_model(device)

    xn, L, C = 100, 175, 10
    x        = torch.randn(xn, 2, 5, L, device=device)
    y        = torch.randn(xn, 2, 1, L, device=device)
    ix       = torch.arange(xn, device=device)
    well_vel = torch.randn(2, L, device=device)
    well_fea = torch.randn(C, L, device=device)

    # forward: without well
    out_nw = model(x, ix=ix)
    assert out_nw.shape == (xn, 2, 1, L), out_nw.shape

    # forward: with well (velocity only)
    out_vel = model(x, ix=ix, well_profile=well_vel)
    assert out_vel.shape == (xn, 2, 1, L), out_vel.shape

    # forward: with well lsrtm + velocity
    out_kv = model(x, ix=ix, well_profile=well_vel, well_lsrtm=well_fea)
    assert out_kv.shape == (xn, 2, 1, L), out_kv.shape

    # training step
    loss_nw = train_step(model, opt, x, y)
    loss_w  = train_step(model, opt, x, y, well_profile=well_vel, well_lsrtm=well_fea)

    n = sum(p.numel() for p in model.parameters())
    print(f"Output shape       : {out_nw.shape}")
    print(f"Loss (no well)     : {loss_nw:.6f}")
    print(f"Loss (with well)   : {loss_w:.6f}")
    print(f"Params             : {n:,}")

    with torch.no_grad():
        xf = x.reshape(xn, 10, L)
        xf = model.norm(xf)
        w  = model.gate(xf)
    print(f"Gate               : FNO={w[0,0]:.3f}  GRU={w[0,1]:.3f}  CNN={w[0,2]:.3f}")
    print(f"Decoder alpha      : {torch.sigmoid(model.decoder.alpha).item():.3f}")
