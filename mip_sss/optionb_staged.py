"""Option B as the M4 CUDA kernels compute it, stage by stage, with the hand-derived backward.

The CUDA implementation transliterates these functions; tests/test_optionb_staged.py checks the
manual gradients against autograd, so the derivatives are verified before any CUDA runs.

Per component (all float64 here; float32 in CUDA):
  Σ3 (3x3), σ (3D filter std), T (2x3, = J·W, the projection), ν, s (2D filter variance), o.
  Σ2 = T Σ3 Tᵀ, P = σ² T Tᵀ, Σ2f = Σ2 + P, covp = Σ2f + sI.
  D, m, tr: det, sum of principal 2x2 minors, trace of Σ3; det(Σ3/λ + σ²I) = D/λ³ + σ²m/λ² + σ⁴tr/λ + σ⁶.
  log hp = ½(log D − log det3f + log|Σ2f| − log|covp|)          (control peak; Option A in 3D+2D)
  nodes k (adaptive_tail, K = 12): λ_k, w_k as mip_sss.design.QUAD.
    cov2_k = Σ2/λ_k + P, filt2_k = cov2_k + sI,
    log g0_k = log w_k + ½(log D − 3 log λ_k − log det3_k) + ½(log|cov2_k| − log|filt2_k|).
Per pixel, offset d = xy − pixel:
  t-term τ = hp (1 + q'/ν)^(−(ν+2)/2),  q' = dᵀ covp⁻¹ d
  T = τ + Σ_k [ g0_k exp(−½ dᵀ filt2_k⁻¹ d) − w_k hp exp(−½ λ_k q') ]   (Option B)
  T = τ                                                                  (fast path)
Backward accumulators per component (what CUDA's render pass adds with atomics):
  G_S (dT/dΣ2), G_P (dT/dP) as symmetric 2x2 matrices, g_inv (D·dT/dD, dT/dm, dT/dtr), dT/dν, dT/dd.
Float32-safe forms (M4, NOTES.md 43): the D gradient is carried as D·∂/∂D, which is bounded, and the
chain applies Σ3⁻¹ = adj/D; the 3D terms go through e = det3f − D and L3_k = λ³det3_k = D + e3_k, so
they are exactly zero at σ = 0, where D is not used at all (c3 ≡ 1, and cov2_k = Σ2/λ_k exactly).
"""
import math

import torch

from .fastpath_table import TABLE

K = 12
LEFT = RIGHT = 3.0
LOG_TAU = math.log(1 / 255)


def sym_inv(c):
    det = c[0, 0] * c[1, 1] - c[0, 1] * c[1, 0]
    return torch.stack([torch.stack([c[1, 1], -c[0, 1]]), torch.stack([-c[1, 0], c[0, 0]])]) / det


def logdet2(c):
    return torch.log(c[0, 0] * c[1, 1] - c[0, 1] * c[1, 0])


def invariants(S3):
    D = torch.det(S3)
    m = (S3[0, 0] * S3[1, 1] - S3[0, 1] ** 2 + S3[0, 0] * S3[2, 2] - S3[0, 2] ** 2
         + S3[1, 1] * S3[2, 2] - S3[1, 2] ** 2)
    return D, m, torch.trace(S3)


def nodes_with_derivs(nu):
    """z_k, λ_k, log w_k and their ν-derivatives, exactly as CUDA computes them."""
    a, b = (nu + 2) / 2, nu / 2
    psi1 = torch.polygamma(1, a)
    c = torch.digamma(a) - torch.log(b)
    w = torch.sqrt(psi1)
    lo, hi = c - LEFT * w + LOG_TAU / a, c + RIGHT * w
    t = torch.linspace(0, 1, K, dtype=nu.dtype)
    z = lo + (hi - lo) * t
    lam = torch.exp(z)
    f = a * z - b * lam
    logw = f - torch.logsumexp(f, 0)
    wk = torch.exp(logw)
    dc = 0.5 * psi1 - 1 / nu
    dw = torch.polygamma(2, a) / (4 * w)
    dlo = dc - LEFT * dw - LOG_TAU / (2 * a * a)
    dhi = dc + RIGHT * dw
    dz = dlo + (dhi - dlo) * t
    df = 0.5 * z + a * dz - 0.5 * lam - b * lam * dz
    dlogw = df - (wk * df).sum()
    return {"z": z, "lam": lam, "logw": logw, "w": wk, "dlam": lam * dz, "dlogw": dlogw}


def fast_threshold(nu):
    rho = math.inf
    for nu_i, r in TABLE:
        if float(nu) >= nu_i:
            rho = r
    return rho


def preprocess(S3, sigma, T, nu, s):
    """Per-component constants, and the fast-path decision."""
    I2 = torch.eye(2, dtype=S3.dtype)
    S2 = T @ S3 @ T.T
    P = sigma ** 2 * (T @ T.T)
    S2f = S2 + P
    covp = S2f + s * I2
    D, m, tr = invariants(S3)
    s2, s4, s6 = sigma ** 2, sigma ** 4, sigma ** 6
    e = s2 * m + s4 * tr + s6
    det3f = D + e
    on3 = float(sigma) > 0
    loghp = 0.5 * (logdet2(S2f) - logdet2(covp))
    if on3:
        loghp = loghp + 0.5 * (torch.log(D) - torch.log(det3f))
    # ∂log hp/∂Σ2f (= ∂/∂Σ2 = ∂/∂P), (D·∂/∂D, ∂/∂m, ∂/∂tr) of log hp
    Lh = 0.5 * (sym_inv(S2f) - sym_inv(covp))
    hinv = torch.stack([0.5 * e / det3f, -0.5 * s2 / det3f, -0.5 * s4 / det3f])
    rho = fast_threshold(nu)
    lam3_min = torch.linalg.eigvalsh(S3)[0]
    lam2_min = torch.linalg.eigvalsh(S2f)[0]
    ok3 = sigma == 0 or bool(lam3_min >= rho * sigma ** 2)
    ok2 = s == 0 or bool(lam2_min >= rho * s)
    out = {"S2": S2, "P": P, "covp": covp, "Qp": sym_inv(covp), "loghp": loghp, "Lh": Lh,
           "hinv": hinv, "fast": ok3 and ok2, "D": D, "m": m, "tr": tr}
    if out["fast"]:
        return out
    nd = nodes_with_derivs(nu)
    nodes = []
    for k in range(K):
        lam = nd["lam"][k]
        cov2 = S2 / lam + P
        filt2 = cov2 + s * I2
        if on3:
            # L3 = λ³·det(Σ3/λ + σ²I) = D + e3, c3 = sqrt(D / L3)
            e3 = lam * s2 * m + lam ** 2 * s4 * tr + lam ** 3 * s6
            L3 = D + e3
            logc3 = 0.5 * (torch.log(D) - torch.log(L3))
            # (D·∂/∂D, ∂/∂m, ∂/∂tr) and ∂/∂λ of log c3
            c3inv = torch.stack([0.5 * e3 / L3, -0.5 * s2 * lam / L3, -0.5 * s4 * lam ** 2 / L3])
            dlc3 = -0.5 * (s2 * m + 2 * lam * s4 * tr + 3 * lam ** 2 * s6) / L3
            icov2 = sym_inv(cov2)
            logdet_cov2 = logdet2(cov2)
        else:
            logc3 = torch.zeros((), dtype=S3.dtype)
            c3inv = torch.zeros(3, dtype=S3.dtype)
            dlc3 = torch.zeros((), dtype=S3.dtype)
            icov2 = lam * sym_inv(S2)
            logdet_cov2 = logdet2(S2) - 2 * torch.log(lam)
        L2 = 0.5 * (icov2 - sym_inv(filt2))  # ∂log c2/∂cov2
        logc2 = 0.5 * (logdet_cov2 - logdet2(filt2))
        dlc2 = -(L2 * S2).sum() / lam ** 2  # tr(L2 · ∂cov2/∂λ), ∂cov2/∂λ = −Σ2/λ²
        nodes.append({
            "lam": lam, "w": nd["w"][k], "dlogw": nd["dlogw"][k], "dlam": nd["dlam"][k],
            "Q": sym_inv(filt2), "logg0": nd["logw"][k] + logc3 + logc2, "L2": L2,
            "c3inv": c3inv,
            # per-pixel ν term: g·(A_k) − ½ g (uᵀΣ2u) B_k
            "A": nd["dlogw"][k] + (dlc3 + dlc2) * nd["dlam"][k],
            "B": nd["dlam"][k] / lam ** 2,
        })
    out["nodes"] = nodes
    return out


def pixel(d, pre, nu, dLdT=None):
    """T at offsets d (n, 2) and, with dLdT (n,), the backward accumulators summed over pixels."""
    up = d @ pre["Qp"]
    qp = (up * d).sum(1)
    hp = torch.exp(pre["loghp"])
    base = 1 + qp / nu
    Gt = torch.exp(-(nu + 2) / 2 * torch.log(base))
    tau = hp * Gt
    T = tau.clone()
    if dLdT is None and pre["fast"]:
        return T
    if dLdT is None:
        for nd in pre["nodes"]:
            u = d @ nd["Q"]
            T = T + torch.exp(nd["logg0"] - 0.5 * (u * d).sum(1)) - nd["w"] * hp * torch.exp(-0.5 * nd["lam"] * qp)
        return T

    S2 = pre["S2"]
    G_S = torch.zeros(2, 2, dtype=d.dtype)
    G_P = torch.zeros(2, 2, dtype=d.dtype)
    g_inv = torch.zeros(3, dtype=d.dtype)
    dnu = torch.zeros((), dtype=d.dtype)
    dd = torch.zeros_like(d)
    # t-term
    tau_q = tau * (-(nu + 2) / 2) / nu / base
    dtau_nu = tau * (-0.5 * torch.log(base) + (nu + 2) / 2 * qp / nu ** 2 / base)
    Cq = tau_q.clone()
    Clh = tau.clone()
    dnu_pix = dtau_nu.clone()
    if not pre["fast"]:
        for nd in pre["nodes"]:
            u = d @ nd["Q"]
            q = (u * d).sum(1)
            g = torch.exp(nd["logg0"] - 0.5 * q)
            H = nd["w"] * hp * torch.exp(-0.5 * nd["lam"] * qp)
            T = T + g - H
            gw = dLdT * g
            Gn = 0.5 * torch.einsum("n,ni,nj->ij", gw, u, u) + gw.sum() * nd["L2"]  # dT/dcov2_k
            G_S = G_S + Gn / nd["lam"]
            G_P = G_P + Gn
            g_inv = g_inv + gw.sum() * nd["c3inv"]
            uSu = ((u @ S2) * u).sum(1)
            dnu = dnu + (gw * (nd["A"] - 0.5 * uSu * nd["B"])).sum()
            dd = dd - gw[:, None] * u
            Cq = Cq + 0.5 * nd["lam"] * H
            Clh = Clh - H
            dnu_pix = dnu_pix - H * (nd["dlogw"] - 0.5 * qp * nd["dlam"])
    # control: q' via covp = Σ2 + P + sI, log hp via Σ2f = Σ2 + P and (D, m, tr)
    Gq = -torch.einsum("n,ni,nj->ij", dLdT * Cq, up, up)
    Gl = (dLdT * Clh).sum() * pre["Lh"]
    G_S = G_S + Gq + Gl
    G_P = G_P + Gq + Gl
    g_inv = g_inv + (dLdT * Clh).sum() * pre["hinv"]
    dnu = dnu + (dLdT * dnu_pix).sum()
    dd = dd + 2 * (dLdT * Cq)[:, None] * up
    return T, {"G_S": G_S, "G_P": G_P, "g_inv": g_inv, "dnu": dnu, "dd": dd}


def chain(S3, sigma, T, acc):
    """dL/dΣ3 (3x3, entries independent) and dL/dT (2x3) from the accumulators."""
    G_S, G_P, (gD, gm, gt) = acc["G_S"], acc["G_P"], acc["g_inv"]
    I3 = torch.eye(3, dtype=S3.dtype)
    dS3 = T.T @ G_S @ T
    if float(sigma) > 0:  # gD is D·dL/dD, and ∂D/∂Σ3 / D = Σ3⁻¹; at σ = 0 all three are zero
        dS3 = dS3 + gD * torch.linalg.inv(S3) + gm * (torch.trace(S3) * I3 - S3) + gt * I3
    # Σ2 = TΣ3Tᵀ → ∂/∂T = (G + Gᵀ) T Σ3; P = σ²TTᵀ → σ²(G + Gᵀ) T
    dT = (G_S + G_S.T) @ T @ S3 + sigma ** 2 * (G_P + G_P.T) @ T
    return dS3, dT


def direct(S3, sigma, T, nu, s, d):
    """T(d) straight from the definitions, for autograd (same node rule and mode decision)."""
    pre = preprocess(S3.detach(), sigma, T.detach(), nu.detach(), s)
    I2 = torch.eye(2, dtype=S3.dtype)
    S2 = T @ S3 @ T.T
    P = sigma ** 2 * (T @ T.T)
    S2f = S2 + P
    covp = S2f + s * I2
    D, m, tr = invariants(S3)
    det3f = D + sigma ** 2 * m + sigma ** 4 * tr + sigma ** 6
    on3 = float(sigma) > 0  # at σ = 0, c3 ≡ 1 and D cancels from every term
    loghp = 0.5 * (logdet2(S2f) - logdet2(covp))
    if on3:
        loghp = loghp + 0.5 * (torch.log(D) - torch.log(det3f))
    hp = torch.exp(loghp)
    qp = ((d @ sym_inv(covp)) * d).sum(1)
    out = hp * torch.exp(-(nu + 2) / 2 * torch.log1p(qp / nu))
    if pre["fast"]:
        return out
    nd = nodes_with_derivs(nu)
    for k in range(K):
        lam, logw = nd["lam"][k], nd["logw"][k]
        det3 = D / lam ** 3 + sigma ** 2 * m / lam ** 2 + sigma ** 4 * tr / lam + sigma ** 6
        cov2 = S2 / lam + P
        filt2 = cov2 + s * I2
        logc3 = 0.5 * (torch.log(D) - 3 * torch.log(lam) - torch.log(det3)) if on3 else 0.0
        logg0 = logw + logc3 + 0.5 * (logdet2(cov2) - logdet2(filt2))
        q = ((d @ sym_inv(filt2)) * d).sum(1)
        out = out + torch.exp(logg0 - 0.5 * q) - torch.exp(logw) * hp * torch.exp(-0.5 * lam * qp)
    return out
