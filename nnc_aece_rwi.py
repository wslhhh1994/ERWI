    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GO-ERWI: Gradient-Optimized Elastic Reflection Waveform Inversion
基于 AECE (声-弹耦合波动方程) 的弹性反射波形反演

算法结构 (Algorithm 1):
  外层循环 (ERWI)  : 更新背景模型 α_b = Vp_b², β_b = Vs_b²
  内层循环 (ELSRTM): 更新扰动模型 α_p = δ(Vp²), β_p = δ(Vs²)

Born 反偏移方程:
  L(m_b) u_p = C u_b
  扰动算子 C 中, 对各应力分量的 Born 源 (vp_p=ρδVp², vs_p=ρδVs²):
    f_p     = -(vp_p - vs_p)(∂vxb/∂x + ∂vzb/∂z)
    f_τs    =  vs_p          (∂vxb/∂x - ∂vzb/∂z)
    f_τxz   =  vs_p          (∂vxb/∂z + ∂vzb/∂x)

梯度公式 (AECE, 以 Vp²/Vs² 空间表达):
  内层 g_vp_p (∂J/∂vp_p):
      ∑∫  (∂vxb/∂x + ∂vzb/∂z) × ψ_p    dt
  内层 g_vs_p (∂J/∂vs_p, 优化版, 舍弃 Term 1):
      ∑∫ [(∂vxb/∂x - ∂vzb/∂z) × ψ_τs +
           (∂vxb/∂z + ∂vzb/∂x) × ψ_τxz] dt
  外层 g_la_b = ∂E/∂α_b: ρ[(div u_r)×p̃_b + (div u_b)×p̃_r]
  外层 g_me_b = ∂E/∂β_b (Term1丢弃): ρ[(shear u_r)×Ψ_τ_b + (shear u_b)×Ψ_τ_r]
  Ψ_r 满足 L^T(m_b)Ψ_r=C^T(m_p)Ψ_b, 与 Ψ_b 同步在时间反向循环中传播.
  链式法则: grad_vp=2Vp×g_α, grad_vs=2Vs×g_β

  ψ_p, ψ_τs, ψ_τxz 为 Ψ_b 的伴随压力/应力分量.
  伴随源: 压力残差 δp = p_obs - p_cal 注入 rpwave (拖缆数据).

差分格式 (与 c_aece_fwi.py 完全一致):
  核 [am[N-1],..,am[0],-am[0],..,-am[N-1]] 卷积后:
    Pattern A: [s:ex, s:ez]         ← vx_x, vz_z, txz_z, txz_x
    Pattern B: 偏移 +1              ← vx_z=[s:ex,s+1:ez+1], vz_x=[s+1:ex+1,s:ez]
                                      txx_x=[s+1:ex+1,s:ez], p_x=[s+1:ex+1,s:ez]
                                      txx_z=[s:ex,s+1:ez+1], p_z=[s:ex,s+1:ez+1]
  gradfun / Born 源使用 stress_cal 相同模式:
    dvxx = PatA-x, dvzz = PatA-z, dvxz = PatB-z, dvzx = PatB-x
"""

import numpy as np
import cupy as cp
from cupyx.scipy.signal import convolve2d
import scipy.signal
import scipy.ndimage
import math
import os
import time
from mpi4py import MPI

# ================================================================
# MPI + 多卡初始化
# ================================================================
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3,4,5"
processes_per_gpu = 5

import torch
torch.manual_seed(30)
np.random.seed(30)
import torch.nn.functional as _TF
from seismic_moe import SeismicMoE

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    cp_device_id = rank // processes_per_gpu
    if cp_device_id >= num_gpus:
        print(f"Rank {rank}: GPU {cp_device_id} 超出范围", flush=True)
        comm.Abort(1)
    cp.cuda.Device(cp_device_id).use()
    torch.cuda.set_device(cp_device_id)
    print(f"Rank {rank}: 使用 GPU {cp_device_id}/{num_gpus}", flush=True)
else:
    print(f"Rank {rank}: 无 CUDA 设备!", flush=True)
    comm.Abort(1)

# ================================================================
# ERWI 配置
# ================================================================
N_INNER_PER_CYCLE  = 3      # 每轮循环中内层 ELSRTM 次数
N_OUTER_PER_CYCLE  = 5      # 每轮循环中外层 ERWI 次数
CYCLE_LEN          = N_INNER_PER_CYCLE + N_OUTER_PER_CYCLE  # 一个完整循环长度: 3内+5外=8
DEPTH_WEIGHT_POWER = 3.0    # 深度加权指数
GRAD_SMOOTH_SIGMA  = 1.0    # 梯度 Gaussian 平滑 sigma (格点)
GRAD_CLIP_PERC     = 97     # 梯度截断百分位
lambda_tv          = 0.0001 # TV 正则化权重

X_start=70

# ================================================================
# 结构引导协克里金配置
# USE_WELL=False 时退化为纯结构引导各向异性平滑（无测井约束）
# ================================================================
USE_WELL      = False   # 是否有测井约束
SGK_SIGMA_ALONG = 3.0   # 沿层方向平滑强度（格点数，约 240m）
SGK_SIGMA_CROSS = 1.5    # 跨层方向平滑强度（格点数，保边界）
KRIG_RANGE_X    = 60     # 协克里金 x 方向相关距离（格点，约 1200m）

# 偏移距滚动策略 (随背景更新次数逐渐扩大使用的偏移距)
OFFSET_RAMP_START = 70    # 初始最大偏移距 (格点, 20*20m=400m)
OFFSET_RAMP_END   = 99    # 最终最大偏移距 (格点, NEAR_OFFSET+NR-1 = 10+89 = 99 格点)
OFFSET_RAMP_OUTER = 500     # 达到最大偏移距所需外层迭代次数

# ================================================================
# 全局参数
# ================================================================
PI = math.pi
N  = 4          # 差分半阶数 (4 阶 8 点)
FM = 8         # 主频 Hz
L  = 30
pml   = L
bound = 15
dx = 25
dz = 25
dt = 0.002
Tn    = 3000
vpmax = 4600
xn = 450
zn = 175
Xn = xn + 2 * pml
Zn = zn + 2 * pml

s  = N
ex = Xn - N
ez = Zn - N

# ================================================================
# 边界波场存储常量
# ================================================================
NN      = pml + N
n_top   = N * (Zn - 2 * pml)
n_bot   = n_top
n_left  = (Xn - 2 * NN) * N
n_right = n_left
n_bnd   = n_top + n_bot + n_left + n_right

# ================================================================
# 差分系数 (Taylor / Holberg 优化系数)
# ================================================================
am = np.zeros(N, dtype=np.float64)
for m in range(1, N + 1):
    sx1, sx2 = 1.0, 1.0
    for i in range(1, m):
        sx1 *= (2 * i - 1) ** 2
        sx2 *= abs((2 * m - 1) ** 2 - (2 * i - 1) ** 2)
    for i in range(m + 1, N + 1):
        sx1 *= (2 * i - 1) ** 2
        sx2 *= abs((2 * m - 1) ** 2 - (2 * i - 1) ** 2)
    am[m - 1] = (1 if m % 2 == 1 else -1) * sx1 / (sx2 * (2 * m - 1))
am = am.astype(np.float32)

# 卷积核 [am[N-1],...,am[0],-am[0],...,-am[N-1]]
am_list = list(am[::-1]) + list(-am)
ker_dx  = cp.array(am_list, dtype=cp.float32).reshape(2 * N, 1)   # x 方向
ker_dz  = cp.array(am_list, dtype=cp.float32).reshape(1, 2 * N)   # z 方向

# ================================================================
# 辅助函数
# ================================================================
def ricker(dt, favg, Tn):
    t = np.arange(Tn) * dt
    w = 100.0 * (1 - 2 * (PI * favg * (t - 1.1)) ** 2) * \
        np.exp(-(PI * favg * (t - 1 / favg)) ** 2)
    return w.astype(np.float32)


def ext_model(vp0, NX, NZ, pml):
    """将模型外推到 PML 边界"""
    V = np.zeros([NX + 2 * pml, NZ + 2 * pml], dtype=np.float32)
    V[pml:pml + NX, pml:pml + NZ] = vp0
    V[pml:-pml, :pml]       = V[pml:pml + NX, pml:pml + 1]
    V[pml:-pml, NZ + pml:]  = V[pml:pml + NX, NZ + pml - 1:NZ + pml]
    V[:pml, :]  = V[pml:pml + 1, :]
    V[NX + pml:, :] = V[NX + pml - 1:NX + pml, :]
    return V


def pml_coef(Xn, Zn, Re, pml, dx, dz, dt):
    """CPML 衰减系数"""
    PMLX = dx * pml;  PMLZ = dz * pml;  NN_ = pml + N
    PMLax = np.zeros([Xn, Zn], dtype=np.float32)
    PMLaz = np.zeros([Xn, Zn], dtype=np.float32)
    PMLbx = np.zeros([Xn, Zn], dtype=np.float32)
    PMLbz = np.zeros([Xn, Zn], dtype=np.float32)
    for i in range(N, Xn - N):
        lx = (NN_ - i) * dx if i < NN_ else ((NN_ + i - Xn + 1) * dx if i >= Xn - NN_ else 0.0)
        if lx > 0:
            PML_dx = 1.5 * vpmax * Re * lx ** 2 / PMLX ** 3
            ta = PI * FM * (1 - lx / PMLX)
            bx = np.exp(-(ta + PML_dx) * dt)
            ax = (bx - 1) * PML_dx / (ta + PML_dx)
        else:
            bx = ax = 0.0
        for j in range(N, Zn - N):
            if lx > 0: PMLbx[i, j] = bx;  PMLax[i, j] = ax
            lz_top = (NN_ - j) * dz if j < NN_ else 0.0
            lz_bot = (j - Zn + NN_ + 1) * dz if j >= Zn - NN_ else 0.0
            lz = max(lz_top, lz_bot)
            if lz > 0:
                PML_dz = 1.5 * vpmax * Re * lz ** 2 / PMLZ ** 3
                ta = PI * FM * (1 - lz / PMLZ)
                PMLbz[i, j] = np.exp(-(ta + PML_dz) * dt)
                PMLaz[i, j] = (PMLbz[i, j] - 1) * PML_dz / (ta + PML_dz)
    return PMLax, PMLaz, PMLbx, PMLbz


# ================================================================
# 边界条带存取辅助
# ================================================================
def _save_bnd(field, buf):
    o = 0
    buf[o:o + n_top]   = field[pml:NN,        pml:Zn - pml].ravel();  o += n_top
    buf[o:o + n_bot]   = field[Xn-NN:Xn-pml,  pml:Zn - pml].ravel();  o += n_bot
    buf[o:o + n_left]  = field[NN:Xn - NN,    pml:NN].ravel();         o += n_left
    buf[o:o + n_right] = field[NN:Xn - NN,    Zn-NN:Zn-pml].ravel()


def _restore_bnd(field, buf):
    o = 0
    field[pml:NN,        pml:Zn - pml]  = buf[o:o+n_top].reshape(N, Zn-2*pml);    o += n_top
    field[Xn-NN:Xn-pml,  pml:Zn - pml]  = buf[o:o+n_bot].reshape(N, Zn-2*pml);    o += n_bot
    field[NN:Xn - NN,    pml:NN]         = buf[o:o+n_left].reshape(Xn-2*NN, N);    o += n_left
    field[NN:Xn - NN,    Zn-NN:Zn-pml]  = buf[o:o+n_right].reshape(Xn-2*NN, N)


def _zero_bnd(field):
    field[pml:NN,        pml:Zn - pml] = 0
    field[Xn-NN:Xn-pml,  pml:Zn - pml] = 0
    field[NN:Xn - NN,    pml:NN]        = 0
    field[NN:Xn - NN,    Zn-NN:Zn-pml] = 0


# ================================================================
# MoE 辅助函数
# ================================================================
def _mirror5(ix, n):
    """以 ix 为中心的 5 点镜像填充索引."""
    idx = []
    for o in (-2, -1, 0, 1, 2):
        i = ix + o
        if i < 0:     i = -i
        elif i >= n:  i = 2 * n - 2 - i
        idx.append(max(0, min(i, n - 1)))
    return idx


def _build_moe_snapshot(vp_norm, vs_norm):
    """
    vp_norm, vs_norm: (xn, zn) 已归一化 float32 数组
    返回 X: (xn, 2, 5, zn)，5 道维度使用镜像填充
    """
    n_x, n_z = vp_norm.shape
    X = np.empty((n_x, 2, 5, n_z), dtype=np.float32)
    for ix in range(n_x):
        idx = _mirror5(ix, n_x)
        X[ix, 0] = vp_norm[idx]
        X[ix, 1] = vs_norm[idx]
    return X

def laplacian_filter(img):
    """∇²img ≈ img - smooth(img), 高通滤波压制直流"""
    smooth = scipy.ndimage.gaussian_filter(img, sigma=5)

    return (img - smooth).astype(np.float32)

# ================================================================
# 结构引导协克里金（Plan A + Plan 1）
# ================================================================

def compute_structure_tensor_dip(lsrtm_field, sigma_pre=1.5, sigma_post=3.0,
                                  median_size=3):
    """
    从 LSRTM 偏移图像计算局部地质倾角场（结构张量法）。

    lsrtm_field : (xn, zn) 偏移图像（绝对值或振幅，内域坐标）
    sigma_pre   : 高斯预平滑 sigma，抑制高频随机噪声
    sigma_post  : 结构张量平滑 sigma，扩大局部平均邻域
    median_size : 中值滤波核尺寸（奇数），在高斯预平滑之前先压制脉冲噪声
                  和 LSRTM 散射旁瓣（LSRTM 噪声主要是随机脉冲型）
                  设为 0 或 1 时跳过，默认 3（3×3 窗口）

    噪声抑制流程（双重保护）:
      abs(LSRTM) → 中值滤波(3×3) → 高斯平滑(sigma_pre) → 结构张量

    Returns:
        tx, tz    : (xn, zn) float32，沿层方向单位向量的 x / z 分量
                   水平层: tx≈±1, tz≈0；垂直层: tx≈0, tz≈±1
        coherence : (xn, zn) float32，各向异性相干性 ∈ [0,1]
                   0 = 各向同性（无结构），1 = 强层状结构
    """
    img = np.abs(lsrtm_field).astype(np.float64)

    # 全零图像（初期 LSRTM 未收敛）：直接返回水平方向 + 零相干性
    if img.max() < 1e-12:
        tx  = np.ones(lsrtm_field.shape, dtype=np.float32)
        tz  = np.zeros(lsrtm_field.shape, dtype=np.float32)
        coh = np.zeros(lsrtm_field.shape, dtype=np.float32)
        return tx, tz, coh

    # 中值滤波：压制 LSRTM 散射噪声和脉冲型旁瓣
    # 保边界效果优于高斯，不会把反射界面模糊成斜坡
    if median_size and median_size > 1:
        img = scipy.ndimage.median_filter(img, size=median_size)

    if sigma_pre > 0:
        img = scipy.ndimage.gaussian_filter(img, sigma=sigma_pre)

    gx = np.gradient(img, axis=0)
    gz = np.gradient(img, axis=1)

    # 结构张量分量（局部平均）
    Jxx = scipy.ndimage.gaussian_filter(gx * gx, sigma=sigma_post)
    Jxz = scipy.ndimage.gaussian_filter(gx * gz, sigma=sigma_post)
    Jzz = scipy.ndimage.gaussian_filter(gz * gz, sigma=sigma_post)

    # 特征值（解析公式）
    T   = Jxx + Jzz
    D   = np.sqrt(np.clip(0.25 * (Jxx - Jzz) ** 2 + Jxz ** 2, 0, None))
    lam_max = 0.5 * T + D   # 最大特征值 → 法向（跨层）
    lam_min = 0.5 * T - D   # 最小特征值 → 切向（沿层）

    # 最大特征值对应特征向量 = 法向量 n = (Jxz, lam_max-Jxx) normalized
    nx = Jxz.copy()
    nz = lam_max - Jxx
    n_norm = np.sqrt(nx ** 2 + nz ** 2) + 1e-12
    nx /= n_norm
    nz /= n_norm

    # 沿层切向量 t = 法向量旋转 90°: t = (-nz, nx)
    tx = -nz
    tz =  nx

    # 退化区域（无方向性）：tx/tz 趋零 → 补充为水平方向
    t_norm = np.sqrt(tx ** 2 + tz ** 2) + 1e-12
    tx /= t_norm
    tz /= t_norm

    # 各向异性相干性
    coherence = np.clip(
        (lam_max - lam_min) / (lam_max + lam_min + 1e-12), 0, 1)

    return tx.astype(np.float32), tz.astype(np.float32), coherence.astype(np.float32)


def structure_guided_filter(field, tx, tz, sigma_along=12.0, sigma_cross=1.5):
    """
    结构引导各向异性滤波（等价于各向异性克里金平滑）。

    沿层方向（tx, tz）施加宽高斯（sigma_along），保留层内连续性；
    跨层方向施加窄高斯（sigma_cross），维持界面锐利度。

    field      : (xn, zn) 待平滑场（ERWI Δvp/Δvs 或梯度）
    tx, tz     : (xn, zn) 沿层单位向量（由 compute_structure_tensor_dip 返回）
    sigma_along: 沿层方向高斯 sigma（格点）
    sigma_cross: 跨层方向高斯 sigma（格点）

    使用 map_coordinates 双线性插值逐步偏移采样，支持逐点变化的倾角。
    """
    xn, zn  = field.shape
    k_half  = int(np.ceil(2.5 * sigma_along))
    offsets = np.arange(-k_half, k_half + 1, dtype=np.float64)
    kernel  = np.exp(-0.5 * (offsets / sigma_along) ** 2)
    kernel /= kernel.sum()

    # 基准网格坐标（xn, zn）
    x_base = np.arange(xn, dtype=np.float32)[:, np.newaxis] * np.ones((1, zn), dtype=np.float32)
    z_base = np.ones((xn, 1), dtype=np.float32) * np.arange(zn, dtype=np.float32)[np.newaxis, :]

    result = np.zeros((xn, zn), dtype=np.float64)

    for k, w in zip(offsets, kernel):
        src_x = np.clip(x_base + k * tx, 0, xn - 1).ravel()
        src_z = np.clip(z_base + k * tz, 0, zn - 1).ravel()
        vals  = scipy.ndimage.map_coordinates(
            field.astype(np.float64),
            [src_x, src_z],
            order=1, mode='nearest')
        result += w * vals.reshape(xn, zn)

    # 跨层方向小 sigma 等效平滑（各向同性部分，抑制残余噪声）
    if sigma_cross > 0:
        result = scipy.ndimage.gaussian_filter(result, sigma=sigma_cross)

    return result.astype(np.float32)


def build_cokriging_pseudo_labels(
    dvp_erwi, dvs_erwi,
    lsrtm_vp_p, lsrtm_vs_p,
    tx, tz,
    well_ix=None,
    dvp_well=None,
    dvs_well=None,
    sigma_along=12.0,
    sigma_cross=1.5,
    krig_range_x=60,
):
    """
    结构引导协克里金伪标签构建（Plan A + Plan 1 联合，无测井时自动退化）。

    ─── Plan A（始终执行）─────────────────────────────────────────────
    用 LSRTM 提取局部倾角场，对 ERWI Δvp/Δvs 做各向异性扩散平滑：
      · 沿层方向 sigma_along（大）→ 保持层内连续性
      · 跨层方向 sigma_cross（小）→ 维持速度界面锐度

    ─── Plan 1（有测井时叠加）─────────────────────────────────────────
    并置协克里金（Collocated Co-Kriging, Markov 近似）：
      · 校正量 = 测井真实值 - 平滑后 ERWI 在测井位置的值
      · 指数衰减权重 w(x) = exp(-|x - well_ix| / krig_range_x) 向两侧传播
      · 这等价于单点条件化的简单克里金，在测井处严格吻合，远端退回平滑值

    Args:
        dvp_erwi, dvs_erwi     : (xn, zn)  ERWI 累积速度偏移
        lsrtm_vp_p, lsrtm_vs_p: (xn, zn)  LSRTM 偏移图（内域，仅用于早期零场检测）
        tx, tz                 : (xn, zn)  沿层方向向量（由 compute_structure_tensor_dip 得到）
        well_ix                : int or None  测井 x 道索引；None = 无测井
        dvp_well, dvs_well     : (zn,) or None  测井处真实 Δvp / Δvs
        sigma_along, sigma_cross, krig_range_x: 见上

    Returns:
        dvp_krig, dvs_krig : (xn, zn) float32，克里金伪标签
    """
    # ── Plan A: 结构引导各向异性平滑 ─────────────────────────────────
    dvp_smooth = structure_guided_filter(
        dvp_erwi, tx, tz, sigma_along=sigma_along, sigma_cross=sigma_cross)
    dvs_smooth = structure_guided_filter(
        dvs_erwi, tx, tz, sigma_along=sigma_along, sigma_cross=sigma_cross)

    if well_ix is None:
        return dvp_smooth, dvs_smooth

    # ── Plan 1: 并置协克里金测井校正 ──────────────────────────────────
    # 指数衰减克里金权重 w(x) = exp(-|x - well_ix| / range_x)
    # 等价于指数型变差函数的单点 Simple Kriging 解析解
    x_arr  = np.arange(dvp_smooth.shape[0], dtype=np.float32)
    krig_w = np.exp(-np.abs(x_arr - well_ix) / max(float(krig_range_x), 1.0))
    krig_w = krig_w[:, np.newaxis]                                # (xn, 1) → broadcast (xn, zn)

    # 测井处平滑值与真实值的残差（沿全深度传播）
    corr_vp = dvp_well - dvp_smooth[well_ix, :]                  # (zn,)
    corr_vs = dvs_well - dvs_smooth[well_ix, :]

    dvp_krig = dvp_smooth + krig_w * corr_vp[np.newaxis, :]      # (xn, zn)
    dvs_krig = dvs_smooth + krig_w * corr_vs[np.newaxis, :]

    return dvp_krig.astype(np.float32), dvs_krig.astype(np.float32)


def _ssim1d(pred, target, win=11, sigma=1.5, C1=1e-4, C2=9e-4):
    """1-D SSIM. pred/target: (N, 1, L), 数据已归一化至 [0,1] 量级."""
    g = torch.exp(-0.5 * (torch.arange(win, device=pred.device,
                 dtype=pred.dtype) - win // 2) ** 2 / sigma ** 2)
    g = (g / g.sum()).view(1, 1, win)
    p = win // 2
    mx = _TF.conv1d(pred,   g, padding=p)
    my = _TF.conv1d(target, g, padding=p)
    sx2 = _TF.conv1d(pred   ** 2, g, padding=p) - mx ** 2
    sy2 = _TF.conv1d(target ** 2, g, padding=p) - my ** 2
    sxy = _TF.conv1d(pred * target, g, padding=p) - mx * my
    num = (2 * mx * my + C1) * (2 * sxy + C2)
    den = (mx ** 2 + my ** 2 + C1) * (sx2 + sy2 + C2)
    return (num / den.clamp(min=1e-8)).mean()


def _moe_loss(pred, target):
    """RMSE + (1 − SSIM). pred/target: (B, 2, 1, L)."""
    rmse = torch.sqrt(_TF.mse_loss(pred, target) + 1e-12)
    x = pred.squeeze(2).reshape(-1, 1, pred.shape[-1])
    y = target.squeeze(2).reshape(-1, 1, target.shape[-1])
    return rmse + (1.0 - _ssim1d(x, y))


def _make_ricker_depth_kernel(half=12, peak=0.06, device='cpu', dtype=torch.float32):
    """
    深度域 Ricker 子波核（用于反射系数 → 合成像褶积）。

    half : 核半径（格点），核总长 2*half+1
    peak : 控制主波长，单位 格点^-1；主波长 ≈ 1/peak 个格点。
           默认 peak=0.06 → 主波长≈17 格点（dz=25m 时约 425m，
           对应 vp≈3000m/s、FM=8Hz 的反射层间距）。
           若层间距更细可增大 peak（如 0.08）；更粗可减小（如 0.04）。
    返回 : (1, 1, 2*half+1) max-abs 归一化核
    """
    x = torch.arange(-half, half + 1, dtype=dtype, device=device)
    u = (math.pi * peak * x) ** 2
    w = (1.0 - 2.0 * u) * torch.exp(-u)
    w = w - w.mean()                           # 去直流分量
    w = w / (w.abs().max() + 1e-12)            # max-abs 归一
    return w.view(1, 1, -1)                    # (1, 1, K)


def _impedance_synth(v_abs, wavelet):
    """
    由绝对速度（阻抗 ∝ v，忽略密度）合成深度域像。

    v_abs   : (N, L) 绝对速度张量（GPU，可微）
    wavelet : (1, 1, K) 深度域 Ricker 核（GPU）
    返回    : (N, 1, L) 合成像（与 LSRTM 输出形状一致）
    """
    # 反射系数 r_i = (v_{i+1} - v_i) / (v_{i+1} + v_i)
    r = (v_abs[:, 1:] - v_abs[:, :-1]) / (v_abs[:, 1:] + v_abs[:, :-1] + 1e-6)
    r = _TF.pad(r, (0, 1))                    # (N, L)，末端补零
    r = r.unsqueeze(1)                         # (N, 1, L)
    K = wavelet.shape[-1]
    syn = _TF.conv1d(r, wavelet, padding=K // 2)
    return syn[..., :v_abs.shape[-1]]          # 截断至 L，保证形状精确


def _struct_ssim_loss(syn, lsrtm):
    """
    结构一致性损失：合成像 vs LSRTM 输入，幅度无关 + 极性不敏感。

    syn, lsrtm : (N, 1, L)（GPU）
    - 逐道 max-abs 归一，消除幅度尺度差异
    - 取 1 − max(SSIM(syn, lsrtm), SSIM(syn, −lsrtm))，消除极性约定差异
    返回标量损失（∈ [0, 2]）。
    """
    s = syn   / (syn.abs().amax(dim=-1, keepdim=True)   + 1e-8)
    l = lsrtm / (lsrtm.abs().amax(dim=-1, keepdim=True) + 1e-8)
    ssim_pos = _ssim1d(s,  l)
    ssim_neg = _ssim1d(s, -l)
    return 1.0 - torch.stack([ssim_pos, ssim_neg]).max()


def _moe_train_outer(model, optimizer, X_snaps, Y_snaps, ix_snaps, device,
                     well_ix=90, w_well=2.0, w_pseudo=1.0,
                     val_frac=0.2, patience=10, max_epochs=200,
                     well_profile=None, well_lsrtm=None, well_Y_true=None,
                     krig_range_x=60.0,
                     vp_init_t=None, vs_init_t=None,
                     sc_dvp=1.0, sc_dvs=1.0,
                     w_struct=0.0, ricker_half=12, ricker_peak=0.06):
    """
    在快照列表上训练 MoE，以完整快照（全部 xn 道）为处理单元，支持跨道注意力。

    X_snaps   : list of (xn, 2, 5, zn) CPU tensors — 每个外层 ERWI 迭代的 LSRTM 快照
    Y_snaps   : list of (xn, 2, 1, zn) CPU tensors — 对应 ERWI 伪标签
    ix_snaps  : list of (xn,) long CPU tensors      — 道号索引（0..xn-1）
    well_ix   : 测井道 x 索引（默认 90）
    w_well    : 真实测井损失基准权重（EMA 动态调整后生效）
    w_pseudo  : 伪标签损失权重

    well_profile : (2, zn) GPU tensor — 真实测井速度剖面
    well_lsrtm   : (10, zn) GPU tensor — 测井处 LSRTM 输入展平
    well_Y_true  : (2, zn) CPU tensor  — 真实测井标签，用于额外测井损失
                   None 时仅受伪标签约束

    损失策略：
      ① 伪标签损失 — 整个快照（含 well_ix 道）的 ERWI 伪标签
      ② 真实测井损失 — well_ix 道与 well_Y_true 的额外约束
      ③ EMA 动态平衡 — 跟踪两路损失幅度，调整有效测井权重
      ④ 结构一致性损失（可选）— 将输出速度转化为阻抗-褶积合成像，
           与 LSRTM 输入的中心道比较结构相似性（SSIM，幅度/极性不敏感）

    vp_init_t  : (xn, zn) GPU float tensor — 初始速度场（还原绝对速度用）
    vs_init_t  : (xn, zn) GPU float tensor
    sc_dvp/sc_dvs : float — 速度差归一化尺度（与推理还原口径一致）
    w_struct   : 结构损失权重（默认 0.0 关闭；建议 ≤ 1.0，远小于 w_pseudo）
    ricker_half/peak : 深度域 Ricker 核参数（peak 控制主波长，half 控制宽度）
    """
    n_snaps = len(X_snaps)

    # 快照切分：末尾 val_frac 作验证；只有 1 个快照时训练验证共用
    if n_snaps > 1:
        n_val   = max(1, int(n_snaps * val_frac))
        tr_idx  = list(range(n_snaps - n_val))
        val_idx = list(range(n_snaps - n_val, n_snaps))
    else:
        tr_idx = val_idx = [0]

    # 真实测井标签: (2, zn) → (1, 2, 1, zn)，常驻 GPU
    well_Y_d = None
    if well_Y_true is not None:
        well_Y_d = well_Y_true.unsqueeze(0).unsqueeze(2).to(device)  # (1, 2, 1, zn)

    # EMA 跟踪两路损失幅度，用于动态平衡
    ema_p, ema_w = None, None
    ema_alpha    = 0.9

    best_val, best_state, no_imp = float('inf'), None, 0

    # 深度域 Ricker 核（epoch 循环前一次性构建，全程复用）
    _wavelet = None
    if w_struct > 0.0 and vp_init_t is not None:
        _wavelet = _make_ricker_depth_kernel(
            half=ricker_half, peak=ricker_peak,
            device=device, dtype=torch.float32)

    for epoch in range(max_epochs):
        model.train()
        for si in tr_idx:
            X_s  = X_snaps[si].to(device)    # (xn, 2, 5, zn)
            Y_s  = Y_snaps[si].to(device)    # (xn, 2, 1, zn)
            ix_s = ix_snaps[si].to(device)   # (xn,)

            pred = model(X_s, ix_s, well_profile=well_profile, well_lsrtm=well_lsrtm,
                         well_ix=well_ix, krig_range_x=krig_range_x)

            # ① 伪标签损失：全部道（含 well_ix，trade-off 物理模型输出）
            L_p = _moe_loss(pred, Y_s)

            # ② 真实测井损失：仅 well_ix 道
            L_w = torch.zeros(1, device=device)
            if well_Y_d is not None and well_ix is not None:
                L_w = _moe_loss(pred[well_ix:well_ix + 1], well_Y_d)

            # ③ EMA 动态平衡：调整 w_well 使两路梯度贡献均衡
            with torch.no_grad():
                p_v = L_p.item()
                w_v = max(L_w.item(), 1e-8)
                ema_p = p_v if ema_p is None else ema_alpha * ema_p + (1 - ema_alpha) * p_v
                ema_w = w_v if ema_w is None else ema_alpha * ema_w + (1 - ema_alpha) * w_v
                eff_w = float(np.clip(
                    w_well * ema_p / (ema_w + 1e-8), w_well * 0.1, w_well * 10.0))

            # ④ 结构一致性损失：阻抗-褶积合成像 vs LSRTM 中心道
            L_struct = torch.zeros(1, device=device)
            if _wavelet is not None:
                _dvp_pred = pred[:, 0, 0, :] * sc_dvp          # (xn, zn)
                _dvs_pred = pred[:, 1, 0, :] * sc_dvs
                _vp_abs   = vp_init_t + _dvp_pred               # 绝对速度
                _vs_abs   = vs_init_t + _dvs_pred
                _syn_vp   = _impedance_synth(_vp_abs, _wavelet)  # (xn, 1, zn)
                _syn_vs   = _impedance_synth(_vs_abs, _wavelet)
                _lsr_vp   = X_s[:, 0, 2, :].unsqueeze(1)        # (xn, 1, zn) 中心道 vp LSRTM
                _lsr_vs   = X_s[:, 1, 2, :].unsqueeze(1)        # (xn, 1, zn) 中心道 vs LSRTM
                L_struct  = (_struct_ssim_loss(_syn_vp, _lsr_vp)
                             + _struct_ssim_loss(_syn_vs, _lsr_vs))

            loss = w_pseudo * L_p + eff_w * L_w + w_struct * L_struct

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # ── 验证 + early-stop ─────────────────────────────────────────
        model.eval()
        v_p, v_w, v_struct, n_b = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for si in val_idx:
                X_s  = X_snaps[si].to(device)
                Y_s  = Y_snaps[si].to(device)
                ix_s = ix_snaps[si].to(device)
                pred = model(X_s, ix_s, well_profile=well_profile, well_lsrtm=well_lsrtm,
                             well_ix=well_ix, krig_range_x=krig_range_x)
                v_p += _moe_loss(pred, Y_s).item()
                if well_Y_d is not None and well_ix is not None:
                    v_w += _moe_loss(pred[well_ix:well_ix + 1], well_Y_d).item()
                if _wavelet is not None:
                    _dvp_pred = pred[:, 0, 0, :] * sc_dvp
                    _dvs_pred = pred[:, 1, 0, :] * sc_dvs
                    _vp_abs   = vp_init_t + _dvp_pred
                    _vs_abs   = vs_init_t + _dvs_pred
                    _syn_vp   = _impedance_synth(_vp_abs, _wavelet)
                    _syn_vs   = _impedance_synth(_vs_abs, _wavelet)
                    _lsr_vp   = X_s[:, 0, 2, :].unsqueeze(1)
                    _lsr_vs   = X_s[:, 1, 2, :].unsqueeze(1)
                    v_struct  += (_struct_ssim_loss(_syn_vp, _lsr_vp)
                                  + _struct_ssim_loss(_syn_vs, _lsr_vs)).item()
                n_b += 1
        v = (w_pseudo * v_p / max(n_b, 1)
             + w_well * v_w / max(n_b, 1)
             + w_struct * v_struct / max(n_b, 1))

        if v < best_val - 1e-6:
            best_val   = v
            best_state = {k: wt.clone() for k, wt in model.state_dict().items()}
            no_imp     = 0
        else:
            no_imp += 1
        if no_imp >= patience:
            print(f"    MoE early-stop @ epoch {epoch + 1}, val={best_val:.5f}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val


# ================================================================
# 模型加载 (背景模型)
# ================================================================
_raw = np.fromfile('overthrust_450_3_175_marine_iso_3.bin',
                   dtype=np.float32).reshape([xn, 3, zn])
_raw_init = np.fromfile('overthrust_450_3_175_marine_linear.bin',
                        dtype=np.float32).reshape([xn, 3, zn])

# vp = _raw[:, 0, :].copy()
# vs = _raw[:, 1, :].copy()
for _i in range(3):
    _raw_init[:, _i, :bound] = _raw[:, _i, :bound]
    
_raw_init[:, 0, bound:]=scipy.ndimage.gaussian_filter(_raw[:, 0, bound:],sigma=10)
_raw_init[:, 1, bound:]=scipy.ndimage.gaussian_filter(_raw[:, 1, bound:],sigma=10)

# 浅层用初始模型 (仅替换 vp/vs, 密度保持真实模型)

# vp = _raw[:, 0, :]
# vs = _raw[:, 1, :]
vp = _raw_init[:, 0, :].copy()
vs = _raw_init[:, 1, :].copy()

den = _raw[:, 2, :].copy()   # 与数据生成程序一致: 直接读文件第3列
# 扩展到 PML 域
vp  = ext_model(vp,  xn, zn, pml)
vs  = ext_model(vs,  xn, zn, pml)
den = ext_model(den, xn, zn, pml)

# 背景 Lamé 参数
me = (vs * vs * den).astype(np.float32)           # μ_b = ρ Vs²
la = (vp * vp * den - 2 * me).astype(np.float32)  # λ_b = ρ(Vp²-2Vs²)

# 保存初始模型供 MoE 计算 Δvp/Δvs
vp_init = vp.copy()
vs_init = vs.copy()

# ---------------------------------------------------------------
# 扰动模型: vp_p = δ(Vp²),  vs_p = δ(Vs²)  初始化为零
# ---------------------------------------------------------------
vp_p = np.zeros([Xn, Zn], dtype=np.float32)
vs_p = np.zeros([Xn, Zn], dtype=np.float32)

# ================================================================
# PML 系数
# ================================================================
Re_pml = 6
PMLax, PMLaz, PMLbx, PMLbz = pml_coef(Xn, Zn, Re_pml, pml, dx, dz, dt)
PMLax_cu = cp.asarray(PMLax);  PMLaz_cu = cp.asarray(PMLaz)
PMLbx_cu = cp.asarray(PMLbx);  PMLbz_cu = cp.asarray(PMLbz)

# ================================================================
# 震源 / 接收器几何
# ================================================================

PML = pml
# ── 拖缆观测系统参数 (与 make_ae_data_gpu_tower.py 一致) ──────────
NS = 50;   delta_s = 7      # 炮数, 炮点间距 (格点)
NR = 90;   NEAR_OFFSET = 10  # 道数, 最近偏移距 (格点)
sz_list = [int(pml + 1)] * NS
sx_list = [int(pml + 100 + i * delta_s) for i in range(NS)]  # sx[0]=pml+100

REC_DEPTH = pml + 1
rx_list = [[sx_list[shot] - NEAR_OFFSET - i for i in range(NR)] for shot in range(NS)]
rz_list = [[REC_DEPTH] * NR for _ in range(NS)]  # 2D (NS, NR)

if rank == 0:
    print(f"[观测系统] 拖缆: {NS}炮 × {NR}道, 近偏={NEAR_OFFSET*dx:.0f}m, "
          f"远偏={(NEAR_OFFSET+NR-1)*dx:.0f}m, 炮间距={delta_s*dx:.0f}m, "
          f"炮点 x=[{sx_list[0]},{sx_list[-1]}]", flush=True)

rx_cu = cp.array(rx_list, dtype=cp.int32)   # shape (NS, NR)
rz_cu = cp.array(rz_list, dtype=cp.int32)   # shape (NS, NR)

# ================================================================
# 输出目录
# ================================================================
totalrank  = size
DATA_DIR   = "ae_record_overthrust_50_90_f8_dx25_Tn3000_P_iso_cut"
save_dir   = 'nn_erwi_iso_result_over_pro2_sm/'
result_dir = save_dir + 'models/'
grad_dir   = save_dir + 'grads/'
res_dir    = save_dir + 'residuals/'
if rank == 0:
    for d in [result_dir, grad_dir, res_dir]:
        os.makedirs(d, exist_ok=True)
comm.Barrier()

# ================================================================
# 分频参数
# ================================================================
fenpin      = 1
N_filter    = 4
f_list      = list(range(8, int(2 * FM + 1), 2))
iter_f_list = [24 * i for i in range(len(f_list) + 1)]
# f_list      = list(range(4, int(2 * FM + 1), 1))
# iter_f_list = [8 * i for i in range(len(f_list) + 1)]
# ================================================================
# 优化器状态
# ================================================================
# 背景模型 CG (FR/PR+) 方向
vp_p = np.zeros([Xn, Zn], dtype=np.float32)
vs_p = np.zeros([Xn, Zn], dtype=np.float32)
cg_d_vp     = np.zeros([Xn, Zn], dtype=np.float64)
cg_d_vs     = np.zeros([Xn, Zn], dtype=np.float64)
cg_g_vp_pre = np.zeros([Xn, Zn], dtype=np.float64)
cg_g_vs_pre = np.zeros([Xn, Zn], dtype=np.float64)

# 扰动模型 CG 方向
cg_d_vp_p    = np.zeros([Xn, Zn], dtype=np.float64)
cg_d_vs_p    = np.zeros([Xn, Zn], dtype=np.float64)
cg_g_vp_p_pre= np.zeros([Xn, Zn], dtype=np.float64)
cg_g_vs_p_pre= np.zeros([Xn, Zn], dtype=np.float64)

BB_TARGET_MAX_VP = 100.0   # 背景 Vp 每迭代最大更新幅度 (m/s)
BB_TARGET_MAX_VS = 60.0   # 背景 Vs 每迭代最大更新幅度 (m/s)
# 扰动模型无物理单位约束, 使用归一化步长
STEP_P = 0.5   # 扰动模型更新步长系数 (相对于归一化梯度)

obj_f_list = []

# ================================================================
# MoE 初始化 (rank 0 独占，其余 rank 跳过)
# ================================================================
if rank == 0:
    _moe_device     = torch.device(f'cuda:{cp_device_id}')
    _moe_model      = SeismicMoE().to(_moe_device)
    _moe_optim      = torch.optim.AdamW(_moe_model.parameters(),
                                        lr=3e-4, weight_decay=1e-4)
    _moe_X_buf      = []   # list of torch.Tensor (xn, 2, 5, zn)
    _moe_Y_buf      = []   # list of torch.Tensor (xn, 2, 1, zn)
    _moe_ix_buf     = []   # list of torch.Tensor (xn,) long — x-positions
    _vp_init_inner  = vp_init[pml+X_start:-pml, pml:-pml].copy()   # (xn, zn)
    _vs_init_inner  = vs_init[pml+X_start:-pml, pml:-pml].copy()
    if USE_WELL:
        # 有测井：第 280+100 道使用真实模型偏移量作为有标签数据
        _well_ix      = 330-X_start
        _dvp_well_all = (_raw[X_start:, 0, :] - _vp_init_inner).astype(np.float32)  # (xn, zn)
        _dvs_well_all = (_raw[X_start:, 1, :] - _vs_init_inner).astype(np.float32)
    else:
        # 无测井：_well_ix / _dvp_well_all / _dvs_well_all 全部置 None
        # build_cokriging_pseudo_labels 收到 None 后自动退化为纯结构引导平滑
        _well_ix      = None
        _dvp_well_all = None
        _dvs_well_all = None

# ================================================================
# 主反演循环 (ERWI 双重循环)
# ================================================================
# 循环结构 (每 CYCLE_LEN=8 次为一轮):
#   iter % CYCLE_LEN in [0,1,2]   → 内层 ELSRTM (仅更新扰动)
#   iter % CYCLE_LEN in [3,4,5,6,7] → 外层 ERWI (背景 + 扰动)
# ================================================================
for iter_i in range(0, 100):

    is_outer = (iter_i % CYCLE_LEN >= N_INNER_PER_CYCLE)
    
    if iter_i % CYCLE_LEN==0:
        # 背景模型 CG (FR/PR+) 方向
        vp_p = np.zeros([Xn, Zn], dtype=np.float32)
        vs_p = np.zeros([Xn, Zn], dtype=np.float32)
        cg_d_vp     = np.zeros([Xn, Zn], dtype=np.float64)
        cg_d_vs     = np.zeros([Xn, Zn], dtype=np.float64)
        cg_g_vp_pre = np.zeros([Xn, Zn], dtype=np.float64)
        cg_g_vs_pre = np.zeros([Xn, Zn], dtype=np.float64)

        # 扰动模型 CG 方向
        cg_d_vp_p    = np.zeros([Xn, Zn], dtype=np.float64)
        cg_d_vs_p    = np.zeros([Xn, Zn], dtype=np.float64)
        cg_g_vp_p_pre= np.zeros([Xn, Zn], dtype=np.float64)
        cg_g_vs_p_pre= np.zeros([Xn, Zn], dtype=np.float64)
    # 偏移距滚动: 随已完成的背景更新次数线性扩大最大偏移距
    # 在第 iter_i 次迭代前, 已完成的外层更新数:
    #   完整周期内 = (iter_i // CYCLE_LEN) * N_OUTER_PER_CYCLE
    #   当前周期内 = max(0, iter_i % CYCLE_LEN - N_INNER_PER_CYCLE)
    _outer_done = (iter_i // CYCLE_LEN) * N_OUTER_PER_CYCLE + max(0, iter_i % CYCLE_LEN - N_INNER_PER_CYCLE)
    if is_outer:
        cur_max_offset = OFFSET_RAMP_END   # ERWI 始终使用全偏移距
    else:
        cur_max_offset = min(
            OFFSET_RAMP_START + (OFFSET_RAMP_END - OFFSET_RAMP_START) * _outer_done // max(OFFSET_RAMP_OUTER, 1),
            OFFSET_RAMP_END
        )

    # ────────────────────────────────────────────────
    # 频率段切换: 重置优化状态 + 清零扰动模型
    # ────────────────────────────────────────────────
    if fenpin == 1:
        for i_for_f in range(1, len(iter_f_list)):
            if iter_i - iter_f_list[i_for_f - 1] == 0:
                cg_d_vp[:]       = 0.0;  cg_d_vs[:]       = 0.0
                cg_g_vp_pre[:]   = 0.0;  cg_g_vs_pre[:]   = 0.0
                cg_d_vp_p[:]     = 0.0;  cg_d_vs_p[:]     = 0.0
                cg_g_vp_p_pre[:] = 0.0;  cg_g_vs_p_pre[:] = 0.0
                # 频率段切换时重置扰动模型 (新频段重新成像)
                vp_p[:] = 0.0;  vs_p[:] = 0.0
                if rank == 0:
                    print(f"  频率段切换 → 重置扰动模型, f={f_list[i_for_f-1]} Hz", flush=True)

    # ────────────────────────────────────────────────
    # 分频子波
    # ────────────────────────────────────────────────
    wave = ricker(dt, FM, Tn)
    if fenpin == 1 and is_outer:
        for i_for_f in range(1, len(iter_f_list)):
            if iter_f_list[i_for_f - 1] <= iter_i < iter_f_list[i_for_f]:
                aa, bb_c = scipy.signal.butter(N_filter,
                                               2 * f_list[i_for_f - 1] / (1 / dt),
                                               btype='lowpass', output='ba')
                wave = scipy.signal.filtfilt(aa, bb_c, wave).astype(np.float32)
                break

    if rank == 0:
        mode_str = '外层ERWI (背景更新)' if is_outer else '内层ELSRTM (扰动更新)'
        print(f"======= iter {iter_i}  {mode_str}  max_offset={cur_max_offset * dx:.0f}m =======", flush=True)

    # ────────────────────────────────────────────────
    # GPU 模型上传
    # ────────────────────────────────────────────────
    la_cu    = cp.asarray(la)       # 背景 λ_b
    me_cu    = cp.asarray(me)       # 背景 μ_b
    vp_p_cu  = cp.asarray(vp_p)     # 扰动 δ(Vp²)
    vs_p_cu  = cp.asarray(vs_p)     # 扰动 δ(Vs²)
    den_cu   = cp.asarray(den)
    vp_cu    = cp.asarray(vp)
    vs_cu    = cp.asarray(vs)
    wave_cu  = cp.asarray(wave)

    # ────────────────────────────────────────────────
    # 跨炮梯度累加器
    # ────────────────────────────────────────────────
    # 内层 ELSRTM: 扰动梯度 (AECE 公式, Term 1 丢弃)
    d_grad_vp_p_cu  = cp.zeros([Xn, Zn], dtype=cp.float32)
    d_grad_vs_p_cu  = cp.zeros([Xn, Zn], dtype=cp.float32)
    d_g_vp_p_ws_cu  = cp.zeros([Xn, Zn], dtype=cp.float32)  # 照明预调和
    d_g_vs_p_ws_cu  = cp.zeros([Xn, Zn], dtype=cp.float32)

    # 外层 ERWI: 背景梯度
    if is_outer:
        d_grad_la_b_cu = cp.zeros([Xn, Zn], dtype=cp.float32)
        d_grad_me_b_cu = cp.zeros([Xn, Zn], dtype=cp.float32)
        d_g_la_b_ws_cu = cp.zeros([Xn, Zn], dtype=cp.float32)
        d_g_me_b_ws_cu = cp.zeros([Xn, Zn], dtype=cp.float32)

    obj_f = 0.0

    # ============================================================
    # 炮点循环 (MPI 轮转分配)
    # ============================================================
    for si in range(int(np.ceil((len(sx_list) + 1) / totalrank))):
        s_sum = rank + si * totalrank
        if s_sum >= len(sx_list):
            continue

        sx_s = sx_list[s_sum]
        sz_s = sz_list[s_sum]

        # ──────────────────────────────────────────────
        # 背景波场 u_b = (vxb, vzb, pwaveb, txxb, txzb)
        # ──────────────────────────────────────────────
        vxb    = cp.zeros([Xn, Zn], dtype=cp.float32)
        vzb    = cp.zeros([Xn, Zn], dtype=cp.float32)
        pwaveb = cp.zeros([Xn, Zn], dtype=cp.float32)
        txxb   = cp.zeros([Xn, Zn], dtype=cp.float32)
        txzb   = cp.zeros([Xn, Zn], dtype=cp.float32)
        # CPML 辅助变量 (背景场)
        pv1b = cp.zeros([Xn,Zn],dtype=cp.float32); pv2b = cp.zeros([Xn,Zn],dtype=cp.float32)
        pv3b = cp.zeros([Xn,Zn],dtype=cp.float32); pv4b = cp.zeros([Xn,Zn],dtype=cp.float32)
        pt1b = cp.zeros([Xn,Zn],dtype=cp.float32); pt2b = cp.zeros([Xn,Zn],dtype=cp.float32)
        pt3b = cp.zeros([Xn,Zn],dtype=cp.float32); pt4b = cp.zeros([Xn,Zn],dtype=cp.float32)
        pt5b = cp.zeros([Xn,Zn],dtype=cp.float32); pt6b = cp.zeros([Xn,Zn],dtype=cp.float32)

        # ──────────────────────────────────────────────
        # 扰动波场 u_p = (vxp, vzp, pwavep, txxp, txzp)
        # Born 散射波场: L(m_b) u_p = C u_b
        # ──────────────────────────────────────────────
        vxp    = cp.zeros([Xn, Zn], dtype=cp.float32)
        vzp    = cp.zeros([Xn, Zn], dtype=cp.float32)
        pwavep = cp.zeros([Xn, Zn], dtype=cp.float32)
        txxp   = cp.zeros([Xn, Zn], dtype=cp.float32)
        txzp   = cp.zeros([Xn, Zn], dtype=cp.float32)
        # CPML 辅助变量 (扰动场)
        pv1p = cp.zeros([Xn,Zn],dtype=cp.float32); pv2p = cp.zeros([Xn,Zn],dtype=cp.float32)
        pv3p = cp.zeros([Xn,Zn],dtype=cp.float32); pv4p = cp.zeros([Xn,Zn],dtype=cp.float32)
        pt1p = cp.zeros([Xn,Zn],dtype=cp.float32); pt2p = cp.zeros([Xn,Zn],dtype=cp.float32)
        pt3p = cp.zeros([Xn,Zn],dtype=cp.float32); pt4p = cp.zeros([Xn,Zn],dtype=cp.float32)
        pt5p = cp.zeros([Xn,Zn],dtype=cp.float32); pt6p = cp.zeros([Xn,Zn],dtype=cp.float32)

        # 边界波场存储 (≈282 MB × 2 = 564 MB)
        saved_bnd_b = cp.zeros([Tn, 5, n_bnd], dtype=cp.float32)
        saved_bnd_p = cp.zeros([Tn, 5, n_bnd], dtype=cp.float32)

        # 压力 + 速度记录
        record_pb  = cp.zeros([NR, Tn], dtype=cp.float32)
        record_pp  = cp.zeros([NR, Tn], dtype=cp.float32)
        record_vxb = cp.zeros([NR, Tn], dtype=cp.float32)
        record_vxp = cp.zeros([NR, Tn], dtype=cp.float32)
        record_vzb = cp.zeros([NR, Tn], dtype=cp.float32)
        record_vzp = cp.zeros([NR, Tn], dtype=cp.float32)

        _rx = rx_cu[s_sum]   # 本炮接收器 x 坐标 (NR,)
        _rz = rz_cu[s_sum]   # 本炮接收器 z 坐标 (NR,)

        # ============================================================
        # 正演传播: u_b (背景) + u_p (Born 反偏移)
        # ============================================================
        for k in range(Tn):
            # ════════════════════════════════════════════════════
            # 背景场 stress_cal: 速度 → 应力
            # ════════════════════════════════════════════════════
            vxb_x = convolve2d(vxb, ker_dx, mode='same')[s:ex,   s:ez]       / dx  # PatA-x
            vzb_z = convolve2d(vzb, ker_dz, mode='same')[s:ex,   s:ez]       / dz  # PatA-z
            vxb_z = convolve2d(vxb, ker_dz, mode='same')[s:ex,   s+1:ez+1]   / dz  # PatB-z
            vzb_x = convolve2d(vzb, ker_dx, mode='same')[s+1:ex+1, s:ez]     / dx  # PatB-x

            pv1b[s:ex,s:ez] = PMLbx_cu[s:ex,s:ez]*pv1b[s:ex,s:ez] + PMLax_cu[s:ex,s:ez]*vxb_x
            pv2b[s:ex,s:ez] = PMLbz_cu[s:ex,s:ez]*pv2b[s:ex,s:ez] + PMLaz_cu[s:ex,s:ez]*vzb_z
            pv3b[s:ex,s:ez] = PMLbz_cu[s:ex,s:ez]*pv3b[s:ex,s:ez] + PMLaz_cu[s:ex,s:ez]*vxb_z
            pv4b[s:ex,s:ez] = PMLbx_cu[s:ex,s:ez]*pv4b[s:ex,s:ez] + PMLax_cu[s:ex,s:ez]*vzb_x

            # CPML 校正导数 (用于应力更新 + Born 源)
            Dvxb_x = vxb_x + pv1b[s:ex,s:ez]
            Dvzb_z = vzb_z + pv2b[s:ex,s:ez]
            Dvxb_z = vxb_z + pv3b[s:ex,s:ez]
            Dvzb_x = vzb_x + pv4b[s:ex,s:ez]

            # 背景应力更新 (AECE)
            pwaveb[s:ex,s:ez] -= dt * (la_cu[s:ex,s:ez] + me_cu[s:ex,s:ez]) * (Dvxb_x + Dvzb_z)
            txxb[s:ex,s:ez]   += dt * me_cu[s:ex,s:ez] * (Dvxb_x - Dvzb_z)
            txzb[s:ex,s:ez]   += dt * me_cu[s:ex,s:ez] * (Dvxb_z + Dvzb_x)

            # 震源注入 (纵波点源 → 背景压力场)
            pwaveb[sx_s, sz_s] += wave_cu[k]

            # ════════════════════════════════════════════════════
            # 扰动场 stress_cal: Born 反偏移
            # L(m_b) u_p 的应力步 + C u_b Born 源
            # Born 源使用背景场的 CPML 校正导数 Dvxb_x, Dvzb_z, Dvxb_z, Dvzb_x
            # ════════════════════════════════════════════════════
            vxp_x = convolve2d(vxp, ker_dx, mode='same')[s:ex,   s:ez]     / dx  # PatA-x
            vzp_z = convolve2d(vzp, ker_dz, mode='same')[s:ex,   s:ez]     / dz  # PatA-z
            vxp_z = convolve2d(vxp, ker_dz, mode='same')[s:ex,   s+1:ez+1] / dz  # PatB-z
            vzp_x = convolve2d(vzp, ker_dx, mode='same')[s+1:ex+1, s:ez]   / dx  # PatB-x

            pv1p[s:ex,s:ez] = PMLbx_cu[s:ex,s:ez]*pv1p[s:ex,s:ez] + PMLax_cu[s:ex,s:ez]*vxp_x
            pv2p[s:ex,s:ez] = PMLbz_cu[s:ex,s:ez]*pv2p[s:ex,s:ez] + PMLaz_cu[s:ex,s:ez]*vzp_z
            pv3p[s:ex,s:ez] = PMLbz_cu[s:ex,s:ez]*pv3p[s:ex,s:ez] + PMLaz_cu[s:ex,s:ez]*vxp_z
            pv4p[s:ex,s:ez] = PMLbx_cu[s:ex,s:ez]*pv4p[s:ex,s:ez] + PMLax_cu[s:ex,s:ez]*vzp_x

            Dvxp_x = vxp_x + pv1p[s:ex,s:ez]
            Dvzp_z = vzp_z + pv2p[s:ex,s:ez]
            Dvxp_z = vxp_z + pv3p[s:ex,s:ez]
            Dvzp_x = vzp_x + pv4p[s:ex,s:ez]

            # 扰动应力 = 背景算子作用于 u_p + Born 源 C·u_b
            # Born 源 (C 算子各行, 对应 p, τs, τxz):
            #   f_p   = -ρ(vp_p-vs_p)(Dvxb_x+Dvzb_z)
            #   f_τs  =  ρvs_p       (Dvxb_x-Dvzb_z)
            #   f_τxz =  ρvs_p       (Dvxb_z+Dvzb_x)
            pwavep[s:ex,s:ez] -= dt * (la_cu[s:ex,s:ez] + me_cu[s:ex,s:ez]) * (Dvxp_x + Dvzp_z)
            pwavep[s:ex,s:ez] -= dt * den_cu[s:ex,s:ez] * (vp_p_cu[s:ex,s:ez] - vs_p_cu[s:ex,s:ez]) * (Dvxb_x + Dvzb_z)
            txxp[s:ex,s:ez]   += dt * me_cu[s:ex,s:ez] * (Dvxp_x - Dvzp_z)
            txxp[s:ex,s:ez]   += dt * den_cu[s:ex,s:ez] * vs_p_cu[s:ex,s:ez] * (Dvxb_x - Dvzb_z)
            txzp[s:ex,s:ez]   += dt * me_cu[s:ex,s:ez] * (Dvxp_z + Dvzp_x)
            txzp[s:ex,s:ez]   += dt * den_cu[s:ex,s:ez] * vs_p_cu[s:ex,s:ez] * (Dvxb_z + Dvzb_x)

            # ════════════════════════════════════════════════════
            # 背景场 velocity_cal: 应力 → 速度
            # ════════════════════════════════════════════════════
            txzb_z = convolve2d(txzb,   ker_dz, mode='same')[s:ex,   s:ez]       / dz  # PatA-z
            txxb_x = convolve2d(txxb,   ker_dx, mode='same')[s+1:ex+1, s:ez]     / dx  # PatB-x
            pb_x   = convolve2d(pwaveb, ker_dx, mode='same')[s+1:ex+1, s:ez]     / dx  # PatB-x
            txzb_x = convolve2d(txzb,   ker_dx, mode='same')[s:ex,   s:ez]       / dx  # PatA-x
            txxb_z = convolve2d(txxb,   ker_dz, mode='same')[s:ex,   s+1:ez+1]   / dz  # PatB-z
            pb_z   = convolve2d(pwaveb, ker_dz, mode='same')[s:ex,   s+1:ez+1]   / dz  # PatB-z

            pt1b[s:ex,s:ez] = PMLbz_cu[s:ex,s:ez]*pt1b[s:ex,s:ez] + PMLaz_cu[s:ex,s:ez]*txzb_z
            pt2b[s:ex,s:ez] = PMLbx_cu[s:ex,s:ez]*pt2b[s:ex,s:ez] + PMLax_cu[s:ex,s:ez]*txxb_x
            pt3b[s:ex,s:ez] = PMLbx_cu[s:ex,s:ez]*pt3b[s:ex,s:ez] + PMLax_cu[s:ex,s:ez]*pb_x
            pt4b[s:ex,s:ez] = PMLbx_cu[s:ex,s:ez]*pt4b[s:ex,s:ez] + PMLax_cu[s:ex,s:ez]*txzb_x
            pt5b[s:ex,s:ez] = PMLbz_cu[s:ex,s:ez]*pt5b[s:ex,s:ez] + PMLaz_cu[s:ex,s:ez]*txxb_z
            pt6b[s:ex,s:ez] = PMLbz_cu[s:ex,s:ez]*pt6b[s:ex,s:ez] + PMLaz_cu[s:ex,s:ez]*pb_z

            vxb[s:ex,s:ez] += (dt / den_cu[s:ex,s:ez]) * (
                (txzb_z + pt1b[s:ex,s:ez]) + (txxb_x + pt2b[s:ex,s:ez]) - (pb_x + pt3b[s:ex,s:ez]))
            vzb[s:ex,s:ez] += (dt / den_cu[s:ex,s:ez]) * (
                (txzb_x + pt4b[s:ex,s:ez]) - (txxb_z + pt5b[s:ex,s:ez]) - (pb_z + pt6b[s:ex,s:ez]))

            # ════════════════════════════════════════════════════
            # 扰动场 velocity_cal: 背景算子 (无 Born 源)
            # ════════════════════════════════════════════════════
            txzp_z = convolve2d(txzp,   ker_dz, mode='same')[s:ex,   s:ez]       / dz
            txxp_x = convolve2d(txxp,   ker_dx, mode='same')[s+1:ex+1, s:ez]     / dx
            pp_x   = convolve2d(pwavep, ker_dx, mode='same')[s+1:ex+1, s:ez]     / dx
            txzp_x = convolve2d(txzp,   ker_dx, mode='same')[s:ex,   s:ez]       / dx
            txxp_z = convolve2d(txxp,   ker_dz, mode='same')[s:ex,   s+1:ez+1]   / dz
            pp_z   = convolve2d(pwavep, ker_dz, mode='same')[s:ex,   s+1:ez+1]   / dz

            pt1p[s:ex,s:ez] = PMLbz_cu[s:ex,s:ez]*pt1p[s:ex,s:ez] + PMLaz_cu[s:ex,s:ez]*txzp_z
            pt2p[s:ex,s:ez] = PMLbx_cu[s:ex,s:ez]*pt2p[s:ex,s:ez] + PMLax_cu[s:ex,s:ez]*txxp_x
            pt3p[s:ex,s:ez] = PMLbx_cu[s:ex,s:ez]*pt3p[s:ex,s:ez] + PMLax_cu[s:ex,s:ez]*pp_x
            pt4p[s:ex,s:ez] = PMLbx_cu[s:ex,s:ez]*pt4p[s:ex,s:ez] + PMLax_cu[s:ex,s:ez]*txzp_x
            pt5p[s:ex,s:ez] = PMLbz_cu[s:ex,s:ez]*pt5p[s:ex,s:ez] + PMLaz_cu[s:ex,s:ez]*txxp_z
            pt6p[s:ex,s:ez] = PMLbz_cu[s:ex,s:ez]*pt6p[s:ex,s:ez] + PMLaz_cu[s:ex,s:ez]*pp_z

            vxp[s:ex,s:ez] += (dt / den_cu[s:ex,s:ez]) * (
                (txzp_z + pt1p[s:ex,s:ez]) + (txxp_x + pt2p[s:ex,s:ez]) - (pp_x + pt3p[s:ex,s:ez]))
            vzp[s:ex,s:ez] += (dt / den_cu[s:ex,s:ez]) * (
                (txzp_x + pt4p[s:ex,s:ez]) - (txxp_z + pt5p[s:ex,s:ez]) - (pp_z + pt6p[s:ex,s:ez]))

            # 记录总压力 + 总速度 = 背景 + 扰动
            record_pb[:, k]  = pwaveb[_rx, _rz]
            record_pp[:, k]  = pwavep[_rx, _rz]
            record_vxb[:, k] = vxb[_rx, _rz]
            record_vxp[:, k] = vxp[_rx, _rz]
            record_vzb[:, k] = vzb[_rx, _rz]
            record_vzp[:, k] = vzp[_rx, _rz]

            # 边界存储 (5 个变量)
            _save_bnd(vxb,    saved_bnd_b[k, 0]);  _save_bnd(vzb,    saved_bnd_b[k, 1])
            _save_bnd(txxb,   saved_bnd_b[k, 2]);  _save_bnd(txzb,   saved_bnd_b[k, 3])
            _save_bnd(pwaveb, saved_bnd_b[k, 4])
            _save_bnd(vxp,    saved_bnd_p[k, 0]);  _save_bnd(vzp,    saved_bnd_p[k, 1])
            _save_bnd(txxp,   saved_bnd_p[k, 2]);  _save_bnd(txzp,   saved_bnd_p[k, 3])
            _save_bnd(pwavep, saved_bnd_p[k, 4])

        # ─────────────────────────────────────────
        # 压力数据加载与残差计算
        # 伴随源: δp = p_obs - p_cal (拖缆压力数据)
        # ─────────────────────────────────────────
#         if is_outer:
#             p_cal_np = (record_pb + record_pp).get()   # 模拟总压力
#         else:
#             p_cal_np = ( record_pp).get()   # 模拟总压力
        p_cal_np = record_pp.get()   # 模拟总压力

        try:
            p_obs_raw = np.fromfile(f"{DATA_DIR}/{s_sum}_pwave_cut.bin",
                                    dtype=np.float32).reshape([NR, Tn])
        except FileNotFoundError:
            if rank == 0 and iter_i == 0 and si == 0:
                print(f"\n{'='*60}\n[严重错误] 数据目录不存在: {DATA_DIR}/\n"
                      f"  反演结果将全为零! 请先生成拖缆观测数据:\n"
                      f"  正演几何:\n"
                      f"    - NS={NS} 炮, 炮点 x=[{sx_list[0]},{sx_list[-1]}], z={sz_list[0]}\n"
                      f"    - NR={NR} 道/炮, 近偏={NEAR_OFFSET}格点, 远偏={NEAR_OFFSET+NR-1}格点\n"
                      f"    - 接收器在震源左侧: rx[j][i] = sx[j] - {NEAR_OFFSET} - i\n"
                      f"    - 接收器深度 z={REC_DEPTH} (格点)\n"
                      f"  数据文件格式: [NR={NR}, Tn={Tn}] float32\n"
                      f"{'='*60}\n", flush=True)
            p_obs_raw = record_pp.get().copy()

        # 分频滤波
        if fenpin == 1 and is_outer:
            for i_for_f in range(1, len(iter_f_list)):
                if iter_f_list[i_for_f - 1] <= iter_i < iter_f_list[i_for_f]:
                    aa, bb_c = scipy.signal.butter(N_filter,
                                                   2 * f_list[i_for_f - 1] / (1 / dt),
                                                   btype='lowpass', output='ba')
                    p_obs_raw = scipy.signal.filtfilt(aa, bb_c, p_obs_raw).astype(np.float32)
                    p_cal_np  = scipy.signal.filtfilt(aa, bb_c, p_cal_np).astype(np.float32)
                    break

        # p_obs_raw = p_obs_raw  #/ max(float(np.abs(p_obs_raw).max()), 1e-10)
        # p_cal_np = p_cal_np   #/ max(float(np.abs(p_cal_np).max()),  1e-10)

        if is_outer:
            # 外层 ERWI: per-trace 归一化互相关目标函数, 走时匹配更精准
            # J = sum_r (1 - rho_r), rho_r = <d_obs_r, d_cal_r> / (||d_obs_r|| * ||d_cal_r||)
            # 伴随源: d(J)/d(d_cal_r) = d_obs_r/(No*Nc) - C_r*d_cal_r/(No*Nc^2)
            No_p = np.sqrt(np.sum(p_obs_raw**2, axis=1, keepdims=True)).clip(1e-10)
            Nc_p = np.sqrt(np.sum(p_cal_np**2,  axis=1, keepdims=True)).clip(1e-10)
            C_p  = np.sum(p_obs_raw * p_cal_np, axis=1, keepdims=True)
            rho_p = (C_p / (No_p * Nc_p)).astype(np.float32)
            residual_p = np.nan_to_num(
                (p_obs_raw / (No_p * Nc_p) - C_p * p_cal_np / (No_p * Nc_p**2)).astype(np.float32))
            obj_f += float(np.sum(1.0 - rho_p))
        else:
            # 内层 ELSRTM: L2 目标函数
            residual_p = np.nan_to_num((p_obs_raw - p_cal_np).astype(np.float32))
            obj_f += 0.5 * float(np.sum(residual_p ** 2))

        residual_p.tofile(res_dir + "{}_p.bin".format(s_sum))
        res_p_cu = cp.asarray(residual_p)   # [NR, Tn]

        # ─────────────────────────────────────────
        # VX/VZ 数据加载与残差计算
        # 伴随源: δvx/δvz → rvx/rvz (OBC 或陆地数据)
        # 注: 若仅有拖缆压力数据, 可将下方 USE_VEL_DATA 置 False
        # ─────────────────────────────────────────
        USE_VEL_DATA = True
        
        
#         vx_cal_np = (record_vxb + record_vxp).get()
#         vz_cal_np = (record_vzb + record_vzp).get()
        vx_cal_np = ( record_vxp).get()
        vz_cal_np = ( record_vzp).get()
#         if is_outer:

#             vx_cal_np = (record_vxb + record_vxp).get()
#             vz_cal_np = (record_vzb + record_vzp).get()
#         else:
#             vx_cal_np = ( record_vxp).get()
#             vz_cal_np = ( record_vzp).get()
        try:
            vx_obs_raw = np.fromfile(f"{DATA_DIR}/{s_sum}_vx_cut.bin",
                                     dtype=np.float32).reshape([NR, Tn])
            vz_obs_raw = np.fromfile(f"{DATA_DIR}/{s_sum}_vz_cut.bin",
                                     dtype=np.float32).reshape([NR, Tn])
        except FileNotFoundError:
            USE_VEL_DATA = False
            vx_obs_raw = np.zeros([NR, Tn], dtype=np.float32)
            vz_obs_raw = np.zeros([NR, Tn], dtype=np.float32)

        if USE_VEL_DATA:
            # 分频滤波
            if fenpin == 1 and is_outer:
                for i_for_f in range(1, len(iter_f_list)):
                    if iter_f_list[i_for_f - 1] <= iter_i < iter_f_list[i_for_f]:
                        aa, bb_c = scipy.signal.butter(N_filter,
                                                       2 * f_list[i_for_f - 1] / (1 / dt),
                                                       btype='lowpass', output='ba')
                        vx_obs_raw = scipy.signal.filtfilt(aa, bb_c, vx_obs_raw).astype(np.float32)
                        vz_obs_raw = scipy.signal.filtfilt(aa, bb_c, vz_obs_raw).astype(np.float32)
                        vx_cal_np  = scipy.signal.filtfilt(aa, bb_c, vx_cal_np).astype(np.float32)
                        vz_cal_np  = scipy.signal.filtfilt(aa, bb_c, vz_cal_np).astype(np.float32)
                        break

            if is_outer:
                # 外层 ERWI: per-trace 互相关目标函数
                No_vx = np.sqrt(np.sum(vx_obs_raw**2, axis=1, keepdims=True)).clip(1e-10)
                Nc_vx = np.sqrt(np.sum(vx_cal_np**2,  axis=1, keepdims=True)).clip(1e-10)
                C_vx  = np.sum(vx_obs_raw * vx_cal_np, axis=1, keepdims=True)
                No_vz = np.sqrt(np.sum(vz_obs_raw**2, axis=1, keepdims=True)).clip(1e-10)
                Nc_vz = np.sqrt(np.sum(vz_cal_np**2,  axis=1, keepdims=True)).clip(1e-10)
                C_vz  = np.sum(vz_obs_raw * vz_cal_np, axis=1, keepdims=True)
                rho_vx = C_vx / (No_vx * Nc_vx)
                rho_vz = C_vz / (No_vz * Nc_vz)
                residual_vx = np.nan_to_num(
                    (vx_obs_raw / (No_vx * Nc_vx) - C_vx * vx_cal_np / (No_vx * Nc_vx**2)).astype(np.float32))
                residual_vz = np.nan_to_num(
                    (vz_obs_raw / (No_vz * Nc_vz) - C_vz * vz_cal_np / (No_vz * Nc_vz**2)).astype(np.float32))
                obj_f += float(np.sum(1.0 - rho_vx) + np.sum(1.0 - rho_vz))
            else:
                # 内层 ELSRTM: 整体归一化 L2 目标函数
                vx_obs_norm = vx_obs_raw / max(float(np.abs(vx_obs_raw).max()), 1e-10)
                vz_obs_norm = vz_obs_raw / max(float(np.abs(vz_obs_raw).max()), 1e-10)
                vx_cal_norm = vx_cal_np  / max(float(np.abs(vx_cal_np).max()),  1e-10)
                vz_cal_norm = vz_cal_np  / max(float(np.abs(vz_cal_np).max()),  1e-10)
                residual_vx = np.nan_to_num((vx_obs_norm - vx_cal_norm).astype(np.float32))
                residual_vz = np.nan_to_num((vz_obs_norm - vz_cal_norm).astype(np.float32))
                obj_f += 0.5 * float(np.sum(residual_vx ** 2) + np.sum(residual_vz ** 2))

            # residual_vx.tofile(res_dir + "{}_vx.bin".format(s_sum))
            # residual_vz.tofile(res_dir + "{}_vz.bin".format(s_sum))

            res_vx_cu = cp.asarray(residual_vx)
            res_vz_cu = cp.asarray(residual_vz)
        else:
            res_vx_cu = None
            res_vz_cu = None

        # ─────────────────────────────────────────
        # 偏移距滚动窗口: 屏蔽超出当前最大偏移距的道
        # ─────────────────────────────────────────
        _off_grd  = np.arange(NEAR_OFFSET, NEAR_OFFSET + NR, dtype=np.float32)
        _off_mask = cp.asarray((_off_grd <= cur_max_offset).astype(np.float32))  # [NR]
        res_p_cu  *= _off_mask[:, cp.newaxis]
        if res_vx_cu is not None:
            res_vx_cu *= _off_mask[:, cp.newaxis]
            res_vz_cu *= _off_mask[:, cp.newaxis]

        # ─────────────────────────────────────────────────────────
        # 伴随传播 Ψ_b + 正演波场逆时重构 u_b (和 u_p, 若外层)
        # ─────────────────────────────────────────────────────────
        # 伴随波场 Ψ_b = (rvx, rvz, rpwave, rtxx, rtxz)
        rvx    = cp.zeros([Xn, Zn], dtype=cp.float32)
        rvz    = cp.zeros([Xn, Zn], dtype=cp.float32)
        rpwave = cp.zeros([Xn, Zn], dtype=cp.float32)
        rtxx   = cp.zeros([Xn, Zn], dtype=cp.float32)
        rtxz   = cp.zeros([Xn, Zn], dtype=cp.float32)

        # 伴随 CPML 辅助变量
        pv1 = cp.zeros([Xn,Zn],dtype=cp.float32); pv2 = cp.zeros([Xn,Zn],dtype=cp.float32)
        pv3 = cp.zeros([Xn,Zn],dtype=cp.float32); pv4 = cp.zeros([Xn,Zn],dtype=cp.float32)
        pt1 = cp.zeros([Xn,Zn],dtype=cp.float32); pt2 = cp.zeros([Xn,Zn],dtype=cp.float32)
        pt3 = cp.zeros([Xn,Zn],dtype=cp.float32); pt4 = cp.zeros([Xn,Zn],dtype=cp.float32)
        pt5 = cp.zeros([Xn,Zn],dtype=cp.float32); pt6 = cp.zeros([Xn,Zn],dtype=cp.float32)

        # 重构缓冲 (从正演末尾状态初始化)
        rec_vxb    = vxb.copy();    rec_vzb    = vzb.copy()
        rec_txxb   = txxb.copy();   rec_txzb   = txzb.copy()
        rec_pwaveb = pwaveb.copy()
        if is_outer:
            rec_vxp    = vxp.copy();    rec_vzp    = vzp.copy()
            rec_txxp   = txxp.copy();   rec_txzp   = txzp.copy()
            rec_pwavep = pwavep.copy()
            # 反射伴随波场 Ψ_r: L^T(m_b) Ψ_r = C^T(m_p) Ψ_b
            rvx_r    = cp.zeros([Xn, Zn], dtype=cp.float32)
            rvz_r    = cp.zeros([Xn, Zn], dtype=cp.float32)
            rpwave_r = cp.zeros([Xn, Zn], dtype=cp.float32)
            rtxx_r   = cp.zeros([Xn, Zn], dtype=cp.float32)
            rtxz_r   = cp.zeros([Xn, Zn], dtype=cp.float32)
            pv1_r = cp.zeros([Xn,Zn],dtype=cp.float32); pv2_r = cp.zeros([Xn,Zn],dtype=cp.float32)
            pv3_r = cp.zeros([Xn,Zn],dtype=cp.float32); pv4_r = cp.zeros([Xn,Zn],dtype=cp.float32)
            pt1_r = cp.zeros([Xn,Zn],dtype=cp.float32); pt2_r = cp.zeros([Xn,Zn],dtype=cp.float32)
            pt3_r = cp.zeros([Xn,Zn],dtype=cp.float32); pt4_r = cp.zeros([Xn,Zn],dtype=cp.float32)
            pt5_r = cp.zeros([Xn,Zn],dtype=cp.float32); pt6_r = cp.zeros([Xn,Zn],dtype=cp.float32)

        for k in range(Tn - 1, -1, -1):
            # ════════════════════════════════════════════════════
            # 重构步骤 A: 逆推 u_b 速度 (undo velocity_cal)
            # 内部区域 CPML=0, 时间反演精确; 边界用存储值修正
            # ════════════════════════════════════════════════════
            txzb_z_uc = convolve2d(rec_txzb,   ker_dz, mode='same')[s:ex,   s:ez]     / dz
            txxb_x_uc = convolve2d(rec_txxb,   ker_dx, mode='same')[s+1:ex+1, s:ez]   / dx
            pb_x_uc   = convolve2d(rec_pwaveb, ker_dx, mode='same')[s+1:ex+1, s:ez]   / dx
            txzb_x_uc = convolve2d(rec_txzb,   ker_dx, mode='same')[s:ex,   s:ez]     / dx
            txxb_z_uc = convolve2d(rec_txxb,   ker_dz, mode='same')[s:ex,   s+1:ez+1] / dz
            pb_z_uc   = convolve2d(rec_pwaveb, ker_dz, mode='same')[s:ex,   s+1:ez+1] / dz

            rec_vxb[s:ex,s:ez] -= (dt / den_cu[s:ex,s:ez]) * (txzb_z_uc + txxb_x_uc - pb_x_uc)
            rec_vzb[s:ex,s:ez] -= (dt / den_cu[s:ex,s:ez]) * (txzb_x_uc - txxb_z_uc - pb_z_uc)

            if k > 0:
                _restore_bnd(rec_vxb, saved_bnd_b[k-1, 0])
                _restore_bnd(rec_vzb, saved_bnd_b[k-1, 1])
            else:
                _zero_bnd(rec_vxb);  _zero_bnd(rec_vzb)

            # ════════════════════════════════════════════════════
            # 伴随传播 Ψ_b: stress_cal (符号取反)
            # AECE 自伴随: 同一套方程, 时间反向
            # ════════════════════════════════════════════════════
            vx_x_r = convolve2d(rvx, ker_dx, mode='same')[s:ex,   s:ez]     / dx
            vz_z_r = convolve2d(rvz, ker_dz, mode='same')[s:ex,   s:ez]     / dz
            vx_z_r = convolve2d(rvx, ker_dz, mode='same')[s:ex,   s+1:ez+1] / dz
            vz_x_r = convolve2d(rvz, ker_dx, mode='same')[s+1:ex+1, s:ez]   / dx

            pv1[s:ex,s:ez] = PMLbx_cu[s:ex,s:ez]*pv1[s:ex,s:ez] + PMLax_cu[s:ex,s:ez]*vx_x_r
            pv2[s:ex,s:ez] = PMLbz_cu[s:ex,s:ez]*pv2[s:ex,s:ez] + PMLaz_cu[s:ex,s:ez]*vz_z_r
            pv3[s:ex,s:ez] = PMLbz_cu[s:ex,s:ez]*pv3[s:ex,s:ez] + PMLaz_cu[s:ex,s:ez]*vx_z_r
            pv4[s:ex,s:ez] = PMLbx_cu[s:ex,s:ez]*pv4[s:ex,s:ez] + PMLax_cu[s:ex,s:ez]*vz_x_r

            Dvx_x_r = vx_x_r + pv1[s:ex,s:ez]
            Dvz_z_r = vz_z_r + pv2[s:ex,s:ez]
            Dvx_z_r = vx_z_r + pv3[s:ex,s:ez]
            Dvz_x_r = vz_x_r + pv4[s:ex,s:ez]

            # 伴随应力更新 (正演符号取反: -= → +=)
            rpwave[s:ex,s:ez] += dt * (la_cu[s:ex,s:ez] + me_cu[s:ex,s:ez]) * (Dvx_x_r + Dvz_z_r)
            rtxx[s:ex,s:ez]   -= dt * me_cu[s:ex,s:ez] * (Dvx_x_r - Dvz_z_r)
            rtxz[s:ex,s:ez]   -= dt * me_cu[s:ex,s:ez] * (Dvx_z_r + Dvz_x_r)

            # ════════════════════════════════════════════════════
            # 伴随源注入: 压力残差 δp → rpwave (拖缆压力数据)
            # 区别于 EFWI 的 rvx/rvz 速度伴随源
            # ════════════════════════════════════════════════════
            rpwave[_rx, _rz] += res_p_cu[:, k]

            # ════════════════════════════════════════════════════
            # 伴随传播 Ψ_b: velocity_cal (符号取反)
            # ════════════════════════════════════════════════════
            txz_z_r = convolve2d(rtxz,   ker_dz, mode='same')[s:ex,   s:ez]     / dz
            txx_x_r = convolve2d(rtxx,   ker_dx, mode='same')[s+1:ex+1, s:ez]   / dx
            rp_x_r  = convolve2d(rpwave, ker_dx, mode='same')[s+1:ex+1, s:ez]   / dx
            txz_x_r = convolve2d(rtxz,   ker_dx, mode='same')[s:ex,   s:ez]     / dx
            txx_z_r = convolve2d(rtxx,   ker_dz, mode='same')[s:ex,   s+1:ez+1] / dz
            rp_z_r  = convolve2d(rpwave, ker_dz, mode='same')[s:ex,   s+1:ez+1] / dz

            pt1[s:ex,s:ez] = PMLbz_cu[s:ex,s:ez]*pt1[s:ex,s:ez] + PMLaz_cu[s:ex,s:ez]*txz_z_r
            pt2[s:ex,s:ez] = PMLbx_cu[s:ex,s:ez]*pt2[s:ex,s:ez] + PMLax_cu[s:ex,s:ez]*txx_x_r
            pt3[s:ex,s:ez] = PMLbx_cu[s:ex,s:ez]*pt3[s:ex,s:ez] + PMLax_cu[s:ex,s:ez]*rp_x_r
            pt4[s:ex,s:ez] = PMLbx_cu[s:ex,s:ez]*pt4[s:ex,s:ez] + PMLax_cu[s:ex,s:ez]*txz_x_r
            pt5[s:ex,s:ez] = PMLbz_cu[s:ex,s:ez]*pt5[s:ex,s:ez] + PMLaz_cu[s:ex,s:ez]*txx_z_r
            pt6[s:ex,s:ez] = PMLbz_cu[s:ex,s:ez]*pt6[s:ex,s:ez] + PMLaz_cu[s:ex,s:ez]*rp_z_r

            rvx[s:ex,s:ez] -= (dt / den_cu[s:ex,s:ez]) * (
                (txz_z_r + pt1[s:ex,s:ez]) + (txx_x_r + pt2[s:ex,s:ez]) - (rp_x_r + pt3[s:ex,s:ez]))
            rvz[s:ex,s:ez] -= (dt / den_cu[s:ex,s:ez]) * (
                (txz_x_r + pt4[s:ex,s:ez]) - (txx_z_r + pt5[s:ex,s:ez]) - (rp_z_r + pt6[s:ex,s:ez]))

            # ════════════════════════════════════════════════════
            # VX/VZ 伴随源注入 (速度更新之后)
            # AECE 在能量内积下自伴随: 速度残差直接注入 rvx/rvz,
            # 与压力残差注入 rpwave 物理独立, 梯度贡献叠加.
            # ════════════════════════════════════════════════════
            if res_vx_cu is not None:
                rvx[_rx, _rz] += res_vx_cu[:, k]
                rvz[_rx, _rz] += res_vz_cu[:, k]

            # ════════════════════════════════════════════════════
            # 反射伴随传播 Ψ_r (仅外层 ERWI)
            # L^T(m_b) Ψ_r = C^T(m_p) Ψ_b
            # 等效体力: eff_p=-(vp_p-vs_p)ψ_p, eff_ts=-vs_p ψ_τs, eff_txz=-vs_p ψ_τxz
            # ════════════════════════════════════════════════════
            if is_outer:
                vx_x_r2 = convolve2d(rvx_r, ker_dx, mode='same')[s:ex,     s:ez]     / dx
                vz_z_r2 = convolve2d(rvz_r, ker_dz, mode='same')[s:ex,     s:ez]     / dz
                vx_z_r2 = convolve2d(rvx_r, ker_dz, mode='same')[s:ex,   s+1:ez+1]   / dz
                vz_x_r2 = convolve2d(rvz_r, ker_dx, mode='same')[s+1:ex+1, s:ez]     / dx

                pv1_r[s:ex,s:ez] = PMLbx_cu[s:ex,s:ez]*pv1_r[s:ex,s:ez] + PMLax_cu[s:ex,s:ez]*vx_x_r2
                pv2_r[s:ex,s:ez] = PMLbz_cu[s:ex,s:ez]*pv2_r[s:ex,s:ez] + PMLaz_cu[s:ex,s:ez]*vz_z_r2
                pv3_r[s:ex,s:ez] = PMLbz_cu[s:ex,s:ez]*pv3_r[s:ex,s:ez] + PMLaz_cu[s:ex,s:ez]*vx_z_r2
                pv4_r[s:ex,s:ez] = PMLbx_cu[s:ex,s:ez]*pv4_r[s:ex,s:ez] + PMLax_cu[s:ex,s:ez]*vz_x_r2

                Dvx_x_r2 = vx_x_r2 + pv1_r[s:ex,s:ez]
                Dvz_z_r2 = vz_z_r2 + pv2_r[s:ex,s:ez]
                Dvx_z_r2 = vx_z_r2 + pv3_r[s:ex,s:ez]
                Dvz_x_r2 = vz_x_r2 + pv4_r[s:ex,s:ez]

                rpwave_r[s:ex,s:ez] += dt * (la_cu[s:ex,s:ez] + me_cu[s:ex,s:ez]) * (Dvx_x_r2 + Dvz_z_r2)
                rtxx_r[s:ex,s:ez]   -= dt * me_cu[s:ex,s:ez] * (Dvx_x_r2 - Dvz_z_r2)
                rtxz_r[s:ex,s:ez]   -= dt * me_cu[s:ex,s:ez] * (Dvx_z_r2 + Dvz_x_r2)

                txz_z_r2 = convolve2d(rtxz_r,   ker_dz, mode='same')[s:ex,     s:ez]   / dz
                txx_x_r2 = convolve2d(rtxx_r,   ker_dx, mode='same')[s+1:ex+1, s:ez]   / dx
                rp_x_r2  = convolve2d(rpwave_r, ker_dx, mode='same')[s+1:ex+1, s:ez]   / dx
                txz_x_r2 = convolve2d(rtxz_r,   ker_dx, mode='same')[s:ex,     s:ez]   / dx
                txx_z_r2 = convolve2d(rtxx_r,   ker_dz, mode='same')[s:ex, s+1:ez+1]   / dz
                rp_z_r2  = convolve2d(rpwave_r, ker_dz, mode='same')[s:ex, s+1:ez+1]   / dz

                pt1_r[s:ex,s:ez] = PMLbz_cu[s:ex,s:ez]*pt1_r[s:ex,s:ez] + PMLaz_cu[s:ex,s:ez]*txz_z_r2
                pt2_r[s:ex,s:ez] = PMLbx_cu[s:ex,s:ez]*pt2_r[s:ex,s:ez] + PMLax_cu[s:ex,s:ez]*txx_x_r2
                pt3_r[s:ex,s:ez] = PMLbx_cu[s:ex,s:ez]*pt3_r[s:ex,s:ez] + PMLax_cu[s:ex,s:ez]*rp_x_r2
                pt4_r[s:ex,s:ez] = PMLbx_cu[s:ex,s:ez]*pt4_r[s:ex,s:ez] + PMLax_cu[s:ex,s:ez]*txz_x_r2
                pt5_r[s:ex,s:ez] = PMLbz_cu[s:ex,s:ez]*pt5_r[s:ex,s:ez] + PMLaz_cu[s:ex,s:ez]*txx_z_r2
                pt6_r[s:ex,s:ez] = PMLbz_cu[s:ex,s:ez]*pt6_r[s:ex,s:ez] + PMLaz_cu[s:ex,s:ez]*rp_z_r2

                rvx_r[s:ex,s:ez] -= (dt / den_cu[s:ex,s:ez]) * (
                    (txz_z_r2 + pt1_r[s:ex,s:ez]) + (txx_x_r2 + pt2_r[s:ex,s:ez]) - (rp_x_r2 + pt3_r[s:ex,s:ez]))
                rvz_r[s:ex,s:ez] -= (dt / den_cu[s:ex,s:ez]) * (
                    (txz_x_r2 + pt4_r[s:ex,s:ez]) - (txx_z_r2 + pt5_r[s:ex,s:ez]) - (rp_z_r2 + pt6_r[s:ex,s:ez]))

                eff_p   = -(vp_p_cu - vs_p_cu) * rpwave
                eff_ts  = -vs_p_cu * rtxx
                eff_txz = -vs_p_cu * rtxz

                eff_ts_x  = convolve2d(eff_ts,  ker_dx, mode='same')[s+1:ex+1, s:ez]   / dx
                eff_p_x   = convolve2d(eff_p,   ker_dx, mode='same')[s+1:ex+1, s:ez]   / dx
                eff_txz_z = convolve2d(eff_txz, ker_dz, mode='same')[s:ex,     s:ez]   / dz
                eff_txz_x = convolve2d(eff_txz, ker_dx, mode='same')[s:ex,     s:ez]   / dx
                eff_ts_z  = convolve2d(eff_ts,  ker_dz, mode='same')[s:ex, s+1:ez+1]   / dz
                eff_p_z   = convolve2d(eff_p,   ker_dz, mode='same')[s:ex, s+1:ez+1]   / dz

                rvx_r[s:ex,s:ez] -= (dt / den_cu[s:ex,s:ez]) * (eff_txz_z + eff_ts_x - eff_p_x)
                rvz_r[s:ex,s:ez] -= (dt / den_cu[s:ex,s:ez]) * (eff_txz_x - eff_ts_z - eff_p_z)

            # ════════════════════════════════════════════════════
            # gradfun: 背景场空间导数 (与 stress_cal 同模式)
            # 用于: (1) 梯度累加  (2) 重构步骤 B 的 undo_stress
            # ════════════════════════════════════════════════════
            dvxb_x = convolve2d(rec_vxb, ker_dx, mode='same')[s:ex,   s:ez]     / dx  # PatA-x
            dvzb_z = convolve2d(rec_vzb, ker_dz, mode='same')[s:ex,   s:ez]     / dz  # PatA-z
            dvxb_z = convolve2d(rec_vxb, ker_dz, mode='same')[s:ex,   s+1:ez+1] / dz  # PatB-z
            dvzb_x = convolve2d(rec_vzb, ker_dx, mode='same')[s+1:ex+1, s:ez]   / dx  # PatB-x

            rp  = rpwave[s:ex,s:ez]   # 伴随压力  ψ_p
            rt  = rtxx[s:ex,s:ez]     # 伴随 τs   ψ_τs
            rtz = rtxz[s:ex,s:ez]     # 伴随 τxz  ψ_τxz

            # ════════════════════════════════════════════════════
            # 内层 ELSRTM 梯度: ∂J/∂vp_p, ∂J/∂vs_p (AECE, Vp²/Vs² 空间)
            # ────────────────────────────────────────────────────
            # g_vp_p = ∑∫ (∂vxb/∂x + ∂vzb/∂z) × ψ_p dt
            #   Born 源 -ρ(vp_p-vs_p)(div vxb) 对 vp_p 的灵敏度:
            #   ∂f_p/∂vp_p = -ρ(dvxb_x+dvzb_z) → 除以 ρ 后净结果 = +(dvxb_x+dvzb_z)×ψ_p
            #
            # g_vs_p (优化版, 舍弃 Term 1 = (dvxb_x+dvzb_z)×ψ_p):
            #   Term 2: (dvxb_x - dvzb_z) × ψ_τs   (剪切法向应变与伴随 τs 互相关)
            #   Term 3: (dvxb_z + dvzb_x) × ψ_τxz  (剪切应变与伴随 τxz 互相关)
            #   舍弃 Term 1 消除 P 波对 S 波参数的串扰
            # ════════════════════════════════════════════════════
            d_grad_vp_p_cu[s:ex,s:ez] += (dvxb_x + dvzb_z) * rp

            d_grad_vs_p_cu[s:ex,s:ez] += (-(dvxb_x - dvzb_z) * rt - (dvxb_z + dvzb_x) * rtz)
            # Term 1 (舍弃): += (dvxb_x + dvzb_z) * rp  ← 不包含此项

            # 预调和权重 (背景波场能量作为照明)
            d_g_vp_p_ws_cu[s:ex,s:ez] += (dvxb_x + dvzb_z) ** 2
            d_g_vs_p_ws_cu[s:ex,s:ez] += (dvxb_x - dvzb_z) ** 2 + (dvxb_z + dvzb_x) ** 2

            # ════════════════════════════════════════════════════
            # 外层 ERWI: 重构 u_p + 累加背景梯度
            # ════════════════════════════════════════════════════
            if is_outer:
                # 逆推 u_p 速度
                txzp_z_uc = convolve2d(rec_txzp,   ker_dz, mode='same')[s:ex,   s:ez]     / dz
                txxp_x_uc = convolve2d(rec_txxp,   ker_dx, mode='same')[s+1:ex+1, s:ez]   / dx
                pp_x_uc   = convolve2d(rec_pwavep, ker_dx, mode='same')[s+1:ex+1, s:ez]   / dx
                txzp_x_uc = convolve2d(rec_txzp,   ker_dx, mode='same')[s:ex,   s:ez]     / dx
                txxp_z_uc = convolve2d(rec_txxp,   ker_dz, mode='same')[s:ex,   s+1:ez+1] / dz
                pp_z_uc   = convolve2d(rec_pwavep, ker_dz, mode='same')[s:ex,   s+1:ez+1] / dz

                rec_vxp[s:ex,s:ez] -= (dt / den_cu[s:ex,s:ez]) * (txzp_z_uc + txxp_x_uc - pp_x_uc)
                rec_vzp[s:ex,s:ez] -= (dt / den_cu[s:ex,s:ez]) * (txzp_x_uc - txxp_z_uc - pp_z_uc)

                if k > 0:
                    _restore_bnd(rec_vxp, saved_bnd_p[k-1, 0])
                    _restore_bnd(rec_vzp, saved_bnd_p[k-1, 1])
                else:
                    _zero_bnd(rec_vxp);  _zero_bnd(rec_vzp)

                # 扰动场空间导数 (同模式 PatA/PatB)
                dvxp_x = convolve2d(rec_vxp, ker_dx, mode='same')[s:ex,   s:ez]     / dx
                dvzp_z = convolve2d(rec_vzp, ker_dz, mode='same')[s:ex,   s:ez]     / dz
                dvxp_z = convolve2d(rec_vxp, ker_dz, mode='same')[s:ex,   s+1:ez+1] / dz
                dvzp_x = convolve2d(rec_vzp, ker_dx, mode='same')[s+1:ex+1, s:ez]   / dx

                # 完整背景梯度, 直接对 α_b=Vp², β_b=Vs² 求偏导 (PNG 公式):
                # ∂E/∂α_b = ρ[(div u_r)×p̃_b + (div u_b)×p̃_r]
                # ∂E/∂β_b (Term1 丢弃) = ρ[(shear u_r)×Ψ_τ_b + (shear u_b)×Ψ_τ_r]
                # 乘以 den_cu 使 d_grad_la_b = ∂E/∂α_b, d_grad_me_b = ∂E/∂β_b
                # 链式法则简化: grad_vp = 2Vp×g_α, grad_vs = 2Vs×g_β (无需 -2g_la 修正)
                rp_r  = rpwave_r[s:ex, s:ez]
                rt_r  = rtxx_r[s:ex, s:ez]
                rtz_r = rtxz_r[s:ex, s:ez]
                _den  = den_cu[s:ex, s:ez]

                d_grad_la_b_cu[s:ex,s:ez] -= _den * ((dvxp_x + dvzp_z) * rp \
                                                    + (dvxb_x + dvzb_z) * rp_r)

                d_grad_me_b_cu[s:ex,s:ez] += _den * ((dvxp_x - dvzp_z) * rt  + (dvxp_z + dvzp_x) * rtz \
                                                    + (dvxb_x - dvzb_z) * rt_r + (dvxb_z + dvzb_x) * rtz_r)

                d_g_la_b_ws_cu[s:ex,s:ez] += (dvxp_x + dvzp_z) ** 2 + (dvxb_x + dvzb_z) ** 2
                d_g_me_b_ws_cu[s:ex,s:ez] += (dvxp_x - dvzp_z) ** 2 + (dvxp_z + dvzp_x) ** 2 \
                                           + (dvxb_x - dvzb_z) ** 2 + (dvxb_z + dvzb_x) ** 2

            # ════════════════════════════════════════════════════
            # 重构步骤 B: 逆推应力 (undo stress_cal)
            # 利用已计算的 dvxb_x 等, 无需重复卷积
            # ════════════════════════════════════════════════════
            if k > 0:
                # u_b 应力逆推
                rec_pwaveb[s:ex,s:ez] += dt * (la_cu[s:ex,s:ez] + me_cu[s:ex,s:ez]) * (dvxb_x + dvzb_z)
                rec_txxb[s:ex,s:ez]   -= dt * me_cu[s:ex,s:ez] * (dvxb_x - dvzb_z)
                rec_txzb[s:ex,s:ez]   -= dt * me_cu[s:ex,s:ez] * (dvxb_z + dvzb_x)
                _restore_bnd(rec_txxb,   saved_bnd_b[k-1, 2])
                _restore_bnd(rec_txzb,   saved_bnd_b[k-1, 3])
                _restore_bnd(rec_pwaveb, saved_bnd_b[k-1, 4])

                # u_p 应力逆推 (背景算子逆推 + Born 源逆推)
                if is_outer:
                    rec_pwavep[s:ex,s:ez] += dt*(la_cu[s:ex,s:ez]+me_cu[s:ex,s:ez])*(dvxp_x+dvzp_z)
                    rec_pwavep[s:ex,s:ez] += dt*den_cu[s:ex,s:ez]*(vp_p_cu[s:ex,s:ez]-vs_p_cu[s:ex,s:ez])*(dvxb_x+dvzb_z)
                    rec_txxp[s:ex,s:ez]   -= dt*me_cu[s:ex,s:ez]*(dvxp_x - dvzp_z)
                    rec_txxp[s:ex,s:ez]   -= dt*den_cu[s:ex,s:ez]*vs_p_cu[s:ex,s:ez]*(dvxb_x - dvzb_z)
                    rec_txzp[s:ex,s:ez]   -= dt*me_cu[s:ex,s:ez]*(dvxp_z + dvzp_x)
                    rec_txzp[s:ex,s:ez]   -= dt*den_cu[s:ex,s:ez]*vs_p_cu[s:ex,s:ez]*(dvxb_z + dvzb_x)
                    _restore_bnd(rec_txxp,   saved_bnd_p[k-1, 2])
                    _restore_bnd(rec_txzp,   saved_bnd_p[k-1, 3])
                    _restore_bnd(rec_pwavep, saved_bnd_p[k-1, 4])

        # 释放内存
        del saved_bnd_b, saved_bnd_p
        del rec_vxb, rec_vzb, rec_txxb, rec_txzb, rec_pwaveb
        if is_outer:
            del rec_vxp, rec_vzp, rec_txxp, rec_txzp, rec_pwavep

    # ============================================================
    # MPI Reduce: 汇总各 rank 的梯度
    # ============================================================
    comm.Barrier()
    d_grad_vp_p  = np.nan_to_num(d_grad_vp_p_cu.get())
    d_grad_vs_p  = np.nan_to_num(d_grad_vs_p_cu.get())
    d_g_vp_p_ws  = np.nan_to_num(d_g_vp_p_ws_cu.get())
    d_g_vs_p_ws  = np.nan_to_num(d_g_vs_p_ws_cu.get())

    d_grad_vp_p  = comm.reduce(d_grad_vp_p,  root=0, op=MPI.SUM); comm.Barrier()
    d_grad_vs_p  = comm.reduce(d_grad_vs_p,  root=0, op=MPI.SUM); comm.Barrier()
    d_g_vp_p_ws  = comm.reduce(d_g_vp_p_ws,  root=0, op=MPI.SUM); comm.Barrier()
    d_g_vs_p_ws  = comm.reduce(d_g_vs_p_ws,  root=0, op=MPI.SUM); comm.Barrier()
    obj_f        = comm.reduce(obj_f,        root=0, op=MPI.SUM); comm.Barrier()

    if is_outer:
        d_grad_la_b = np.nan_to_num(d_grad_la_b_cu.get())
        d_grad_me_b = np.nan_to_num(d_grad_me_b_cu.get())
        d_g_la_b_ws = np.nan_to_num(d_g_la_b_ws_cu.get())
        d_g_me_b_ws = np.nan_to_num(d_g_me_b_ws_cu.get())
        d_grad_la_b = comm.reduce(d_grad_la_b, root=0, op=MPI.SUM); comm.Barrier()
        d_grad_me_b = comm.reduce(d_grad_me_b, root=0, op=MPI.SUM); comm.Barrier()
        d_g_la_b_ws = comm.reduce(d_g_la_b_ws, root=0, op=MPI.SUM); comm.Barrier()
        d_g_me_b_ws = comm.reduce(d_g_me_b_ws, root=0, op=MPI.SUM); comm.Barrier()

    # ============================================================
    # 梯度处理 + 模型更新 (rank 0)
    # ============================================================
    if rank == 0:
        obj_f_list.append(obj_f)
        print(f"  obj = {obj_f:.6e}", flush=True)

        # 分频平滑 sigma
        if fenpin == 1:
            _sigma = 5.0 if iter_i <= iter_f_list[3] else float(GRAD_SMOOTH_SIGMA)
        else:
            _sigma = float(GRAD_SMOOTH_SIGMA)

        _sl   = (slice(pml, -pml), slice(pml + bound, -pml))   # 有效内部区域
        _sl_f = (slice(pml, -pml), slice(pml, -pml))

        # ──────────────────────────────────────────────────────
        # A. 扰动模型更新 (内层 ELSRTM: 每次迭代均执行)
        # ──────────────────────────────────────────────────────
        # 预调和 (除以背景照明能量)
        _ws_max = max(np.abs(d_g_vp_p_ws[_sl]).max() * 1e-3, 1e-12)
        d_g_vp_p_ws[_sl_f] += _ws_max
        d_grad_vp_p = np.nan_to_num(d_grad_vp_p / d_g_vp_p_ws)

        _ws_max = max(np.abs(d_g_vs_p_ws[_sl]).max() * 1e-3, 1e-12)
        d_g_vs_p_ws[_sl_f] += _ws_max
        d_grad_vs_p = np.nan_to_num(d_grad_vs_p / d_g_vs_p_ws)

        # Gaussian 平滑
#         d_grad_vp_p[_sl] = scipy.ndimage.gaussian_filter(d_grad_vp_p[_sl], sigma=_sigma)
#         d_grad_vs_p[_sl] = scipy.ndimage.gaussian_filter(d_grad_vs_p[_sl], sigma=_sigma)

        # 截断 + 归一化
        for _g, _name in [(d_grad_vp_p, 'vp_p'), (d_grad_vs_p, 'vs_p')]:
            _, gmax = np.percentile(np.abs(_g[_sl]), [5, GRAD_CLIP_PERC])
            _g[_sl] = np.clip(_g[_sl], -gmax, gmax)
            _g[_sl] /= max(gmax, 1e-30)

        # CG 方向 (Polak-Ribière + β≥0)
        _g_vp_p = np.nan_to_num(-d_grad_vp_p[_sl])
        _g_vs_p = np.nan_to_num(-d_grad_vs_p[_sl])

        _n2_vpp = float(np.sum(cg_g_vp_p_pre[_sl] ** 2))
        _n2_vsp = float(np.sum(cg_g_vs_p_pre[_sl] ** 2))
        _beta_vpp = (max(float(np.dot(_g_vp_p.ravel(),
                                      (_g_vp_p - cg_g_vp_p_pre[_sl]).ravel()))
                         / (_n2_vpp + 1e-10), 0.0) if _n2_vpp > 1e-12 else 0.0)
        _beta_vsp = (max(float(np.dot(_g_vs_p.ravel(),
                                      (_g_vs_p - cg_g_vs_p_pre[_sl]).ravel()))
                         / (_n2_vsp + 1e-10), 0.0) if _n2_vsp > 1e-12 else 0.0)

        cg_d_vp_p[_sl] = -_g_vp_p + _beta_vpp * cg_d_vp_p[_sl]
        cg_d_vs_p[_sl] = -_g_vs_p + _beta_vsp * cg_d_vs_p[_sl]
        cg_g_vp_p_pre[_sl] = _g_vp_p.copy()
        cg_g_vs_p_pre[_sl] = _g_vs_p.copy()

        # 扰动模型更新: 直接更新 δ(Vp²) 和 δ(Vs²) (固定归一化步长)
        if not is_outer:
            vp_p[_sl] += STEP_P * cg_d_vp_p[_sl].astype(np.float32)
            vs_p[_sl] += STEP_P * cg_d_vs_p[_sl].astype(np.float32)

            # 物理约束: 扰动量不超过背景 Vp²/Vs² 的 50%
            _vp_max_p = np.abs(vp[_sl] ** 2).max() * 0.5
            _vs_max_p = np.abs(vs[_sl] ** 2).max() * 0.5
            vp_p[_sl] = np.clip(vp_p[_sl], -_vp_max_p, _vp_max_p)
            vs_p[_sl] = np.clip(vs_p[_sl], -_vs_max_p, _vs_max_p)

        # 扩展到 PML 域
        vp_p = ext_model(vp_p[pml:-pml, pml:-pml], xn, zn, pml)
        vs_p = ext_model(vs_p[pml:-pml, pml:-pml], xn, zn, pml)

        # 输出扰动模型 (δVp², δVs²) 及总模型 (开平方得 Vp/Vs)
        vp_p[pml:-pml, pml:-pml].astype(np.float32).tofile(
            result_dir + "vp_p_{}.bin".format(iter_i))
        vs_p[pml:-pml, pml:-pml].astype(np.float32).tofile(
            result_dir + "vs_p_{}.bin".format(iter_i))
        np.sqrt(np.clip(vp[pml:-pml, pml:-pml]**2 + vp_p[pml:-pml, pml:-pml], 1e-6, None)).astype(
            np.float32).tofile(result_dir + "vp_{}.bin".format(iter_i))
        np.sqrt(np.clip(vs[pml:-pml, pml:-pml]**2 + vs_p[pml:-pml, pml:-pml], 1e-6, None)).astype(
            np.float32).tofile(result_dir + "vs_{}.bin".format(iter_i))
        cg_d_vp_p[pml:-pml, pml:-pml].astype(np.float32).tofile(
            grad_dir + "grad_vp_p_{}.bin".format(iter_i))
        cg_d_vs_p[pml:-pml, pml:-pml].astype(np.float32).tofile(
            grad_dir + "grad_vs_p_{}.bin".format(iter_i))

        print(f"  ELSRTM: beta_vp_p={_beta_vpp:.4f}, beta_vs_p={_beta_vsp:.4f}", flush=True)

        # ──────────────────────────────────────────────────────
        # B. 背景模型更新 (外层 ERWI: 仅 is_outer 迭代执行)
        # ──────────────────────────────────────────────────────
        if is_outer:
            # 预调和
#             _ws_max = max(np.abs(d_g_la_b_ws[_sl]).max() * 1e-3, 1e-12)
#             d_g_la_b_ws[_sl_f] += _ws_max
#             d_grad_la_b = np.nan_to_num(d_grad_la_b / d_g_la_b_ws)

#             _ws_max = max(np.abs(d_g_me_b_ws[_sl]).max() * 1e-3, 1e-12)
#             d_g_me_b_ws[_sl_f] += _ws_max
#             d_grad_me_b = np.nan_to_num(d_grad_me_b / d_g_me_b_ws)

#             # 转换为 Vp/Vs 梯度 (链式法则: la=ρ(Vp²-2Vs²), me=ρVs²)
#             # ∂J/∂Vp = ∂la/∂Vp × ∂J/∂la = 2ρVp × g_la_b
#             # ∂J/∂Vs = ∂la/∂Vs × g_la_b + ∂me/∂Vs × g_me_b
#             #         = -4ρVs × g_la_b + 2ρVs × g_me_b
#             grad_vp = np.nan_to_num(2 * den * vp * d_grad_la_b)
#             # grad_vs = -np.nan_to_num(2 * den * vs * (d_grad_me_b - 2 * d_grad_la_b))
#             grad_vs = np.nan_to_num(2 * den * vs * (d_grad_me_b - 2 * d_grad_la_b))

# #             深度加权 (补偿深部照明不足)
#             _dw = (np.arange(zn) / max(zn, 1)) ** DEPTH_WEIGHT_POWER
#             for _ix in range(Xn):
#                 grad_vp[_ix, pml + bound:pml + zn] *= _dw[bound:]
#                 grad_vs[_ix, pml + bound:pml + zn] *= _dw[bound:]

#             # 结构引导梯度平滑：LSRTM 倾角驱动，沿层平滑 / 跨层保边界
#             # _sl z 方向从 pml+bound 起 → shape (xn, zn-bound)；
#             # 结构张量在完整内域 (xn, zn) 上计算，需截取对应 z 段传入
#             _lsrtm_for_st = np.abs(vp_p[pml:-pml, pml:-pml]) + np.abs(vs_p[pml:-pml, pml:-pml])
#             _st_tx, _st_tz, _ = compute_structure_tensor_dip(_lsrtm_for_st)
#             grad_vp[_sl] = structure_guided_filter(
#                 grad_vp[_sl], _st_tx[:, bound:], _st_tz[:, bound:],
#                 sigma_along=SGK_SIGMA_ALONG * 0.5, sigma_cross=SGK_SIGMA_CROSS)
#             grad_vs[_sl] = structure_guided_filter(
#                 grad_vs[_sl], _st_tx[:, bound:], _st_tz[:, bound:],
#                 sigma_along=SGK_SIGMA_ALONG * 0.5, sigma_cross=SGK_SIGMA_CROSS)

#             # 截断 + 归一化
#             for _g in [grad_vp, grad_vs]:
#                 _, gmax = np.percentile(np.abs(_g[_sl]), [5, GRAD_CLIP_PERC])
#                 if iter_i <= iter_f_list[1]:
#                     _g[_sl] = np.clip(_g[_sl], -gmax, gmax)
#                 _g[_sl] /= max(gmax, 1e-30)

#             # CG 方向 (Polak-Ribière +)
#             _g_vp = np.nan_to_num(-grad_vp[_sl])
#             _g_vs = np.nan_to_num(-grad_vs[_sl])

#             _n2_vp = float(np.sum(cg_g_vp_pre[_sl] ** 2))
#             _n2_vs = float(np.sum(cg_g_vs_pre[_sl] ** 2))
#             _beta_vp = (max(float(np.dot(_g_vp.ravel(),
#                                          (_g_vp - cg_g_vp_pre[_sl]).ravel()))
#                              / (_n2_vp + 1e-10), 0.0) if _n2_vp > 1e-12 else 0.0)
#             _beta_vs = (max(float(np.dot(_g_vs.ravel(),
#                                          (_g_vs - cg_g_vs_pre[_sl]).ravel()))
#                              / (_n2_vs + 1e-10), 0.0) if _n2_vs > 1e-12 else 0.0)

#             cg_d_vp[_sl] = -_g_vp + _beta_vp * cg_d_vp[_sl]
#             cg_d_vs[_sl] = -_g_vs + _beta_vs * cg_d_vs[_sl]
#             cg_g_vp_pre[_sl] = _g_vp.copy()
#             cg_g_vs_pre[_sl] = _g_vs.copy()

#             print(f"  ERWI: beta_vp={_beta_vp:.4f}, beta_vs={_beta_vs:.4f}", flush=True)

#             # GradNorm 步长 (防止更新过大)
#             _d_vp = cg_d_vp[pml:-pml, pml + bound:-pml].astype(np.float32)
#             _d_vs = cg_d_vs[pml:-pml, pml + bound:-pml].astype(np.float32)
#             _dmax_vp = max(float(np.abs(_d_vp).max()), 1e-12)
#             _dmax_vs = max(float(np.abs(_d_vs).max()), 1e-12)
#             _alpha_vp = min(BB_TARGET_MAX_VP / _dmax_vp, 120.0)
#             _alpha_vs = min(BB_TARGET_MAX_VS / _dmax_vs, 80.0)
#             _alpha_vp = max(_alpha_vp, 1.0)
#             _alpha_vs = max(_alpha_vs, 0.5)
#             print(f"  alpha_vp={_alpha_vp:.4f}, alpha_vs={_alpha_vs:.4f}", flush=True)

#             # 背景模型更新 + 物理约束
#             vp[pml:-pml, pml:-pml].astype(np.float32).tofile(
#                 result_dir + "vp0_{}.bin".format(iter_i))

#             vp[pml:-pml, pml + bound:-pml] = np.clip(
#                 vp[pml:-pml, pml + bound:-pml] + _alpha_vp * _d_vp, 1500.0, 6500.0)
#             vs[pml:-pml, pml + bound:-pml] = np.clip(
#                 vs[pml:-pml, pml + bound:-pml] + _alpha_vs * _d_vs, 10.0, 4300.0)

#             # 每 5 次外层更新做一次平滑
#             outer_count = _outer_done + 1  # 本次更新完成后的外层总次数
#             if outer_count % 5 == 0:
#                 vp[pml:-pml, pml + bound:-pml] = scipy.ndimage.gaussian_filter(
#                     vp[pml:-pml, pml + bound:-pml], sigma=2)
#                 vs[pml:-pml, pml + bound:-pml] = scipy.ndimage.gaussian_filter(
#                     vs[pml:-pml, pml + bound:-pml], sigma=2)

            _ws_max = max(np.abs(d_g_la_b_ws[_sl]).max() * 1e-3, 1e-12)
            d_g_la_b_ws[_sl_f] += _ws_max
            d_grad_la_b = np.nan_to_num(d_grad_la_b / d_g_la_b_ws)

            _ws_max = max(np.abs(d_g_me_b_ws[_sl]).max() * 1e-3, 1e-12)
            d_g_me_b_ws[_sl_f] += _ws_max
            d_grad_me_b = np.nan_to_num(d_grad_me_b / d_g_me_b_ws)

            # d_grad_la_b = ∂E/∂α_b (已含 ρ), d_grad_me_b = ∂E/∂β_b (已含 ρ, Term1 丢弃)
            # ∂E/∂Vp = 2Vp × ∂E/∂α_b
            # ∂E/∂Vs = 2Vs × ∂E/∂β_b
            grad_vp = np.nan_to_num(2 * vp * d_grad_la_b)
            grad_vs = np.nan_to_num(2 * vs * d_grad_me_b)

#             深度加权 (补偿深部照明不足)
#             _dw = (np.arange(zn) / max(zn, 1)) ** DEPTH_WEIGHT_POWER
#             for _ix in range(Xn):
#                 grad_vp[_ix, pml + bound:pml + zn] *= _dw[bound:]
#                 grad_vs[_ix, pml + bound:pml + zn] *= _dw[bound:]

#             Gaussian 平滑
#             grad_vp[_sl] = scipy.ndimage.gaussian_filter(grad_vp[_sl], sigma=_sigma)
#             grad_vs[_sl] = scipy.ndimage.gaussian_filter(grad_vs[_sl], sigma=_sigma)

            if iter_i < 21:
                _lsrtm_for_st = np.abs(vp_p[pml:-pml, pml:-pml]) + np.abs(vs_p[pml:-pml, pml:-pml])
                _st_tx, _st_tz, _ = compute_structure_tensor_dip(_lsrtm_for_st)
                grad_vp[_sl] = structure_guided_filter(
                    grad_vp[_sl], _st_tx[:, bound:], _st_tz[:, bound:],
                    sigma_along=SGK_SIGMA_ALONG * 0.5, sigma_cross=SGK_SIGMA_CROSS)
                grad_vs[_sl] = structure_guided_filter(
                    grad_vs[_sl], _st_tx[:, bound:], _st_tz[:, bound:],
                    sigma_along=SGK_SIGMA_ALONG * 0.5, sigma_cross=SGK_SIGMA_CROSS)

            # 截断 + 归一化
            for _g in [grad_vp, grad_vs]:
                _, gmax = np.percentile(np.abs(_g[_sl]), [5, GRAD_CLIP_PERC])
                if iter_i <= iter_f_list[1]:
                    _g[_sl] = np.clip(_g[_sl], -gmax, gmax)
                _g[_sl] /= max(gmax, 1e-30)

            # CG 方向 (Polak-Ribière +)
            _g_vp = np.nan_to_num(-grad_vp[_sl])
            _g_vs = np.nan_to_num(-grad_vs[_sl])

            _n2_vp = float(np.sum(cg_g_vp_pre[_sl] ** 2))
            _n2_vs = float(np.sum(cg_g_vs_pre[_sl] ** 2))
            _beta_vp = (max(float(np.dot(_g_vp.ravel(),
                                         (_g_vp - cg_g_vp_pre[_sl]).ravel()))
                             / (_n2_vp + 1e-10), 0.0) if _n2_vp > 1e-12 else 0.0)
            _beta_vs = (max(float(np.dot(_g_vs.ravel(),
                                         (_g_vs - cg_g_vs_pre[_sl]).ravel()))
                             / (_n2_vs + 1e-10), 0.0) if _n2_vs > 1e-12 else 0.0)

            cg_d_vp[_sl] = -_g_vp + _beta_vp * cg_d_vp[_sl]
            cg_d_vs[_sl] = -_g_vs + _beta_vs * cg_d_vs[_sl]
            cg_g_vp_pre[_sl] = _g_vp.copy()
            cg_g_vs_pre[_sl] = _g_vs.copy()

            print(f"  ERWI: beta_vp={_beta_vp:.4f}, beta_vs={_beta_vs:.4f}", flush=True)

            # GradNorm 步长 (防止更新过大)
            _d_vp = cg_d_vp[pml:-pml, pml + bound:-pml].astype(np.float32)
            _d_vs = cg_d_vs[pml:-pml, pml + bound:-pml].astype(np.float32)
            _dmax_vp = max(float(np.abs(_d_vp).max()), 1e-12)
            _dmax_vs = max(float(np.abs(_d_vs).max()), 1e-12)
            _alpha_vp = min(BB_TARGET_MAX_VP / _dmax_vp, 120.0)
            _alpha_vs = min(BB_TARGET_MAX_VS / _dmax_vs, 80.0)
            _alpha_vp = max(_alpha_vp, 1.0)
            _alpha_vs = max(_alpha_vs, 0.5)
            print(f"  alpha_vp={_alpha_vp:.4f}, alpha_vs={_alpha_vs:.4f}", flush=True)

            # 背景模型更新 + 物理约束
            vp[pml:-pml, pml:-pml].astype(np.float32).tofile(
                result_dir + "vp0_{}.bin".format(iter_i))

            vp[pml:-pml, pml + bound:-pml] = np.clip(
                vp[pml:-pml, pml + bound:-pml] + _alpha_vp * _d_vp, 1500.0, 4500.0)
            vs[pml:-pml, pml + bound:-pml] = np.clip(
                vs[pml:-pml, pml + bound:-pml] + _alpha_vs * _d_vs, 10.0, 3300.0)

            # 每 5 次外层更新做一次平滑
            outer_count = _outer_done + 1  # 本次更新完成后的外层总次数

            # 重新扩展 + 重算 Lamé 参数
            vp  = ext_model(vp[pml:-pml, pml:-pml],  xn, zn, pml)
            vs  = ext_model(vs[pml:-pml, pml:-pml],  xn, zn, pml)
            den = ext_model(den[pml:-pml, pml:-pml], xn, zn, pml)
            me  = (vs * vs * den).astype(np.float32)
            la  = (vp * vp * den - 2 * me).astype(np.float32)

            # 背景模型更新后重置扰动模型 (可选: 保留扰动模型继续迭代)
            # vp_p[:] = 0.0;  vs_p[:] = 0.0  # 若需要重新成像则取消注释

            # ────────────────────────────────────────────────────────
            # MoE: 在线训练 + 推理，替换 ERWI 背景更新结果
            # 流程: 1) ERWI 更新后提取 (vp_p,vs_p)→X 和 Δvp/Δvs→Y
            #        2) 追加缓冲区，在全量数据上训练 MoE (early-stop)
            #        3) 推理当前 snapshot，MoE 输出 + 初始模型替换背景
            # ────────────────────────────────────────────────────────
#             _vp_p_in = vp_p[pml:-pml, pml:-pml]                   # (xn, zn)
#             _vs_p_in = vs_p[pml:-pml, pml:-pml]
            
            if iter_i % 3==0:
                
#                 _vp_p_in=laplacian_filter(vp_p[pml:-pml, pml:-pml])
#                 _vs_p_in=laplacian_filter(vs_p[pml:-pml, pml:-pml])

#                 _vp_p_in=np.fromfile(result_dir + "vp_p_{}.bin".format(int(0//8)*8+2),dtype=np.float32).reshape([xn, zn])
#                 _vs_p_in=np.fromfile(result_dir + "vs_p_{}.bin".format(int(0//8)*8+2),dtype=np.float32).reshape([xn, zn])                
                
                _vp_p_in=np.fromfile(result_dir + "vp_p_{}.bin".format(int(iter_i//8)*8+2),dtype=np.float32).reshape([xn, zn])
                _vs_p_in=np.fromfile(result_dir + "vs_p_{}.bin".format(int(iter_i//8)*8+2),dtype=np.float32).reshape([xn, zn])
                
#                 _vp_p_in=laplacian_filter(_vp_p_in)
#                 _vs_p_in=laplacian_filter(_vs_p_in)
                

                _dvp     = vp[pml+X_start:-pml, pml:-pml] - _vp_init_inner    # ERWI 累积 Δvp
                _dvs     = vs[pml+X_start:-pml, pml:-pml] - _vs_init_inner

                # ── 归一化尺度 ───────────────────────────────────────────────
                _sc_vp_p = float(np.abs(_vp_p_in).max()) + 1e-10
                _sc_vs_p = float(np.abs(_vs_p_in).max()) + 1e-10
                if USE_WELL:
                    # 取 ERWI 与测井真实偏移的较大值，保证测井标签不溢出 [-1,1]
                    _sc_dvp = max(float(np.abs(_dvp).max()),
                                  float(np.abs(_dvp_well_all[_well_ix]).max())) + 1e-10
                    _sc_dvs = max(float(np.abs(_dvs).max()),
                                  float(np.abs(_dvs_well_all[_well_ix]).max())) + 1e-10
                else:
                    _sc_dvp = float(np.abs(_dvp).max()) + 1e-10
                    _sc_dvs = float(np.abs(_dvs).max()) + 1e-10

                # ── 构建 5 道滑动窗口快照 (镜像填充) ─────────────────────────
                _X_snap = _build_moe_snapshot(
                    _vp_p_in[X_start:] / _sc_vp_p, _vs_p_in[X_start:] / _sc_vs_p)        # (xn, 2, 5, zn)
                # Y_snap 直接使用 RWI 原始反演结果作为伪标签
                _Y_snap = np.stack(
                    [_dvp / _sc_dvp, _dvs / _sc_dvs], axis=1
                )[:, :, np.newaxis, :]                                  # (xn, 2, 1, zn)

                # ── 测井相关张量（仅 USE_WELL=True 时有意义）──────────────────
                if USE_WELL:
                    _well_true_np = np.stack(
                        [_dvp_well_all[_well_ix] / _sc_dvp,
                         _dvs_well_all[_well_ix] / _sc_dvs], axis=0)   # (2, zn)
                    _well_prof   = torch.from_numpy(_well_true_np.copy()).float().to(_moe_device)
                    _well_Y_true = torch.from_numpy(_well_true_np.copy()).float()
                    # 测井处 LSRTM 输入展平（cross-attn Keys）
                    _well_lsrtm  = torch.from_numpy(
                        _X_snap[_well_ix].reshape(-1, zn).copy()
                    ).float().to(_moe_device)                            # (10, zn)
                else:
                    _well_prof   = None
                    _well_Y_true = None
                    _well_lsrtm  = None

                # x-道号位置索引 (0..xn-1)，用于 2D RoPE
                _ix_snap = torch.arange(xn-X_start, dtype=torch.long)


                # 每次外层迭代用当前快照重新训练（不累积历史快照）
                _moe_X_buf = [torch.from_numpy(_X_snap)]   # (xn, 2, 5, zn)
                _moe_Y_buf = [torch.from_numpy(_Y_snap)]   # (xn, 2, 1, zn)
                _moe_ix_buf = [_ix_snap]                    # (xn,) long

                _moe_device     = torch.device(f'cuda:{cp_device_id}')
                _moe_model      = SeismicMoE().to(_moe_device)
                _moe_optim      = torch.optim.AdamW(_moe_model.parameters(),
                                            lr=3e-4, weight_decay=1e-4)
                _best_v = _moe_train_outer(
                    _moe_model, _moe_optim, _moe_X_buf, _moe_Y_buf, _moe_ix_buf,
                    device=_moe_device, well_ix=_well_ix,
                    w_well=1.0, w_pseudo=10.0,
                    patience=500, max_epochs=500,
                    well_profile=_well_prof,
                    well_lsrtm=_well_lsrtm,
                    well_Y_true=_well_Y_true,
                    krig_range_x=KRIG_RANGE_X,
                    vp_init_t=torch.from_numpy(_vp_init_inner).float().to(_moe_device),
                    vs_init_t=torch.from_numpy(_vs_init_inner).float().to(_moe_device),
                    sc_dvp=_sc_dvp, sc_dvs=_sc_dvs,
                    w_struct=0.5, ricker_half=12, ricker_peak=0.06,
                )
                print(f"  MoE train  val={_best_v:.5f}  "
                      f"xn={_X_snap.shape[0]}", flush=True)

                # 推理: MoE 输出替换背景模型（传入当前 snapshot 的 x 位置和测井剖面）
                
                
                _moe_model.eval()
                
                result_list=[]
                with torch.no_grad():
                    for _ in range(10):
                        _Y_pred = _moe_model(
                            torch.from_numpy(_X_snap).to(_moe_device),
                            _ix_snap.to(_moe_device),
                            well_profile=_well_prof,
                            well_lsrtm=_well_lsrtm,
                            well_ix=_well_ix,
                            krig_range_x=KRIG_RANGE_X,
                        ).cpu().numpy()                                     # (xn, 2, 1, zn)
                        result_list.append(_Y_pred)
                result_list=np.array(result_list)
                result_list=np.mean(result_list,axis=0)
                _Y_pred=result_list
                _new_dvp = _Y_pred[:, 0, 0, :] * _sc_dvp              # (xn, zn)
                _new_dvs = _Y_pred[:, 1, 0, :] * _sc_dvs

                # 还原背景模型：浅层保持初始值，bound 以下使用 MoE 结果
                _new_vp = vp[pml:-pml, pml:-pml].copy()
                _new_vs = vs[pml:-pml, pml:-pml].copy()
                
                _new_vp[X_start:, bound:] = np.clip(
                    _vp_init_inner[:, bound:] + _new_dvp[:, bound:], 1500.0, 5000.0)
                _new_vs[X_start:, bound:] = np.clip(
                    _vs_init_inner[:, bound:] + _new_dvs[:, bound:], 10.0, 4000.0)
                
                
                _new_vp[:X_start, bound:]=vp[pml:pml+X_start, pml+bound:-pml]
                _new_vs[:X_start, bound:]=vs[pml:pml+X_start, pml+bound:-pml]

#                 _new_vp[:, bound:] = scipy.ndimage.gaussian_filter(
#                     _new_vp[:, bound:], sigma=1)
#                 _new_vs[:, bound:] = scipy.ndimage.gaussian_filter(
#                     _new_vs[:, bound:], sigma=1)
        
                

                vp  = ext_model(_new_vp, xn, zn, pml)
                vs  = ext_model(_new_vs, xn, zn, pml)
                me  = (vs * vs * den).astype(np.float32)
                la  = (vp * vp * den - 2 * me).astype(np.float32)

                print(f"  MoE推理: dvp_max={float(np.abs(_new_dvp[:, bound:]).max()):.1f}  "
                      f"dvs_max={float(np.abs(_new_dvs[:, bound:]).max()):.1f} m/s", flush=True)

                # 保存 MoE 权重
                torch.save(_moe_model.state_dict(),
                           result_dir + f"moe_weights.pth")

            # 保存背景模型
            vp[pml:-pml,  pml:-pml].astype(np.float32).tofile(
                result_dir + "vp_{}.bin".format(iter_i))
            vs[pml:-pml,  pml:-pml].astype(np.float32).tofile(
                result_dir + "vs_{}.bin".format(iter_i))
            den[pml:-pml, pml:-pml].astype(np.float32).tofile(
                result_dir + "den_{}.bin".format(iter_i))
            cg_d_vp[pml:-pml, pml:-pml].astype(np.float32).tofile(
                grad_dir + "grad_vp_{}.bin".format(iter_i))
            cg_d_vs[pml:-pml, pml:-pml].astype(np.float32).tofile(
                grad_dir + "grad_vs_{}.bin".format(iter_i))

        # 目标函数记录
        np.savetxt(result_dir + "obj_f.txt", np.array(obj_f_list))

    # ============================================================
    # Broadcast 更新后的模型到所有 rank
    # ============================================================
    comm.Barrier()
    comm.Bcast(vp,   root=0);  comm.Barrier()
    comm.Bcast(vs,   root=0);  comm.Barrier()
    comm.Bcast(den,  root=0);  comm.Barrier()
    comm.Bcast(la,   root=0);  comm.Barrier()
    comm.Bcast(me,   root=0);  comm.Barrier()
    comm.Bcast(vp_p, root=0);  comm.Barrier()
    comm.Bcast(vs_p, root=0);  comm.Barrier()
