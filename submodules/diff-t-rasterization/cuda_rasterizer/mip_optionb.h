/*
 * Mip-SSS Option B (milestone M4): scale-mixture quadrature with the Option A control variate,
 * the nu-aware fast path and the per-node 3D smoothing filter.
 *
 * This file transliterates mip_sss/optionb_staged.py (Python, float64), whose hand-derived
 * backward is checked against autograd in tests/test_optionb_staged.py. Names follow it:
 *   Sigma2 = T Sigma3 T^T, P = sigma^2 T T^T, Sigma2f = Sigma2 + P, covp = Sigma2f + sI,
 *   log hp = 1/2 (log D - log det3f + log|Sigma2f| - log|covp|),
 *   node k: cov2_k = Sigma2/lam_k + P, filt2_k = cov2_k + sI,
 *           log g0_k = log w_k + log c3_k + log c2_k.
 * Symmetric 2x2 matrices are stored as (m00, m01, m11).
 */
#pragma once

#define OPTB_K 12
#define OPTB_NODE 16
#define OPTB_HEAD 11
#define OPTB_STRIDE (OPTB_HEAD + OPTB_K * OPTB_NODE)
#define OPTB_ACC 9  // per-component backward accumulators: G_S (3), G_P (3), g_inv (3)

// header offsets
#define OB_LOGHP 0
#define OB_LH 1
#define OB_HINV 4
#define OB_S2 7
#define OB_HP 10
// node offsets
#define ON_Q 0
#define ON_LOGG0 3
#define ON_LAM 4
#define ON_W 5
#define ON_L2 6
#define ON_C3INV 9
#define ON_A 12
#define ON_B 13
#define ON_DLOGW 14
#define ON_DLAM 15

// mip_sss/fastpath_table.py (nu_i, rho_i): Option A when lambda_min / s >= rho(nu)
#define OPTB_TABLE_N 27
__constant__ float OPTB_TABLE_NU[OPTB_TABLE_N] = {
	1.0f, 1.25f, 1.5f, 2.0f, 2.5f, 3.0f, 4.0f, 5.0f, 6.0f, 8.0f, 10.0f, 12.0f, 16.0f, 20.0f,
	25.0f, 30.0f, 40.0f, 50.0f, 70.0f, 100.0f, 150.0f, 200.0f, 300.0f, 400.0f, 500.0f, 700.0f,
	1000.0f };
__constant__ float OPTB_TABLE_RHO[OPTB_TABLE_N] = {
	3162.27766f, 1778.27941f, 1778.27941f, 1778.27941f, 1000.0f, 1000.0f, 562.341325f,
	562.341325f, 562.341325f, 316.227766f, 316.227766f, 316.227766f, 177.827941f, 177.827941f,
	100.0f, 100.0f, 56.234133f, 56.234133f, 31.622777f, 31.622777f, 17.782794f, 10.0f,
	5.623413f, 0.0f, 0.0f, 0.0f, 0.0f };
#define OPTB_LEFT 3.0f
#define OPTB_RIGHT 3.0f
#define OPTB_LOG_TAU (-5.541263545158426f)  // log(1/255)
#define OPTB_ALPHA_MIN (1.0f / 255.0f)

__device__ __forceinline__ float optb_threshold(float nu)
{
	float rho = 1e30f;
	for (int i = 0; i < OPTB_TABLE_N; i++)
		if (nu >= OPTB_TABLE_NU[i])
			rho = OPTB_TABLE_RHO[i];
	return rho;
}

// digamma, trigamma, tetragamma for x >= 1: recurrence up to 6, then asymptotic series
__device__ __forceinline__ void optb_polygammas(float x, float& psi0, float& psi1, float& psi2)
{
	float s0 = 0.f, s1 = 0.f, s2 = 0.f;
	while (x < 6.f)
	{
		const float ix = 1.f / x;
		s0 -= ix;
		s1 += ix * ix;
		s2 -= 2.f * ix * ix * ix;
		x += 1.f;
	}
	const float ix = 1.f / x, ix2 = ix * ix;
	psi0 = s0 + logf(x) - 0.5f * ix
		- ix2 * (1.f / 12.f - ix2 * (1.f / 120.f - ix2 * (1.f / 252.f - ix2 * (1.f / 240.f))));
	psi1 = s1 + ix + 0.5f * ix2
		+ ix2 * ix * (1.f / 6.f - ix2 * (1.f / 30.f - ix2 * (1.f / 42.f - ix2 * (1.f / 30.f))));
	psi2 = s2 - ix2 - ix2 * ix
		- ix2 * ix2 * (0.5f - ix2 * (1.f / 6.f - ix2 * (1.f / 6.f - ix2 * 0.3f)));
}

// Nodes of the adaptive_tail rule (mip_sss.design.QUAD) and their nu-derivatives
__device__ inline void optb_nodes(float nu, float* lam, float* logw, float* w, float* dlogw, float* dlam)
{
	const float a = 0.5f * (nu + 2.f), b = 0.5f * nu;
	float psi0, psi1, psi2;
	optb_polygammas(a, psi0, psi1, psi2);
	const float c = psi0 - logf(b);
	const float wd = sqrtf(psi1);
	const float lo = c - OPTB_LEFT * wd + OPTB_LOG_TAU / a, hi = c + OPTB_RIGHT * wd;
	const float dc = 0.5f * psi1 - 1.f / nu;
	const float dw = psi2 / (4.f * wd);
	const float dlo = dc - OPTB_LEFT * dw - OPTB_LOG_TAU / (2.f * a * a), dhi = dc + OPTB_RIGHT * dw;
	float f[OPTB_K], df[OPTB_K], fmax = -1e30f;
	for (int k = 0; k < OPTB_K; k++)
	{
		const float t = (float)k / (float)(OPTB_K - 1);
		const float z = lo + (hi - lo) * t;
		const float dz = dlo + (dhi - dlo) * t;
		lam[k] = expf(z);
		f[k] = a * z - b * lam[k];
		df[k] = 0.5f * z + a * dz - 0.5f * lam[k] - b * lam[k] * dz;
		dlam[k] = lam[k] * dz;
		fmax = fmaxf(fmax, f[k]);
	}
	float sum = 0.f;
	for (int k = 0; k < OPTB_K; k++)
		sum += expf(f[k] - fmax);
	const float lse = fmax + logf(sum);
	float mean_df = 0.f;
	for (int k = 0; k < OPTB_K; k++)
	{
		logw[k] = f[k] - lse;
		w[k] = expf(logw[k]);
		mean_df += w[k] * df[k];
	}
	for (int k = 0; k < OPTB_K; k++)
		dlogw[k] = df[k] - mean_df;
}

__device__ __forceinline__ float optb_det2(const float m[3]) { return m[0] * m[2] - m[1] * m[1]; }

__device__ __forceinline__ void optb_inv2(const float m[3], float out[3])
{
	const float id = 1.f / optb_det2(m);
	out[0] = m[2] * id;
	out[1] = -m[1] * id;
	out[2] = m[0] * id;
}

__device__ __forceinline__ float optb_lmax2(const float m[3])
{
	const float mid = 0.5f * (m[0] + m[2]);
	return mid + sqrtf(fmaxf(0.f, mid * mid - optb_det2(m)));
}

__device__ __forceinline__ float optb_lmin2(const float m[3])
{
	const float mid = 0.5f * (m[0] + m[2]);
	return mid - sqrtf(fmaxf(0.f, mid * mid - optb_det2(m)));
}

// smallest eigenvalue of a symmetric 3x3 (cov6: xx, xy, xz, yy, yz, zz), trigonometric method
__device__ inline float optb_lmin3(const float* c)
{
	const float p1 = c[1] * c[1] + c[2] * c[2] + c[4] * c[4];
	const float q = (c[0] + c[3] + c[5]) / 3.f;
	if (p1 <= 0.f)
		return fminf(c[0], fminf(c[3], c[5]));
	const float p2 = (c[0] - q) * (c[0] - q) + (c[3] - q) * (c[3] - q) + (c[5] - q) * (c[5] - q) + 2.f * p1;
	const float p = sqrtf(p2 / 6.f);
	const float b0 = (c[0] - q) / p, b3 = (c[3] - q) / p, b5 = (c[5] - q) / p;
	const float b1 = c[1] / p, b2 = c[2] / p, b4 = c[4] / p;
	const float r = 0.5f * (b0 * (b3 * b5 - b4 * b4) - b1 * (b1 * b5 - b4 * b2) + b2 * (b1 * b4 - b3 * b2));
	const float phi = acosf(fminf(1.f, fmaxf(-1.f, r))) / 3.f;
	return q + 2.f * p * cosf(phi + 2.0943951023931953f);
}

// det, sum of principal 2x2 minors, trace of a symmetric 3x3 given as cov6
__device__ __forceinline__ void optb_invariants(const float* c, float& D, float& m, float& tr)
{
	D = c[0] * (c[3] * c[5] - c[4] * c[4]) - c[1] * (c[1] * c[5] - c[4] * c[2]) + c[2] * (c[1] * c[4] - c[3] * c[2]);
	m = c[0] * c[3] - c[1] * c[1] + c[0] * c[5] - c[2] * c[2] + c[3] * c[5] - c[4] * c[4];
	tr = c[0] + c[3] + c[5];
}

// Projection T = J W (2x3, row-major) with computeCov2D's screen-space clamp; also t and the
// clamp masks for the backward pass.
__device__ inline void optb_projection(const float3& mean, float fx, float fy, float tan_fovx, float tan_fovy,
	const float* view, float Tm[6], float3& t, float& xmul, float& ymul)
{
	t = transformPoint4x3(mean, view);
	const float limx = 1.3f * tan_fovx, limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z, tytz = t.y / t.z;
	xmul = (txtz < -limx || txtz > limx) ? 0.f : 1.f;
	ymul = (tytz < -limy || tytz > limy) ? 0.f : 1.f;
	t.x = fminf(limx, fmaxf(-limx, txtz)) * t.z;
	t.y = fminf(limy, fmaxf(-limy, tytz)) * t.z;
	const float J00 = fx / t.z, J02 = -fx * t.x / (t.z * t.z);
	const float J11 = fy / t.z, J12 = -fy * t.y / (t.z * t.z);
	// world-to-camera rotation W[i][j] = view[4j + i]
	for (int j = 0; j < 3; j++)
	{
		Tm[j] = J00 * view[4 * j + 0] + J02 * view[4 * j + 2];
		Tm[3 + j] = J11 * view[4 * j + 1] + J12 * view[4 * j + 2];
	}
}

// Sigma2 = T Sigma3 T^T and T T^T as (00, 01, 11)
__device__ inline void optb_project(const float Tm[6], const float* c, float S2[3], float TT[3])
{
	const float S[3][3] = { { c[0], c[1], c[2] }, { c[1], c[3], c[4] }, { c[2], c[4], c[5] } };
	float TS[2][3];
	for (int r = 0; r < 2; r++)
		for (int j = 0; j < 3; j++)
			TS[r][j] = Tm[3 * r] * S[0][j] + Tm[3 * r + 1] * S[1][j] + Tm[3 * r + 2] * S[2][j];
	S2[0] = TS[0][0] * Tm[0] + TS[0][1] * Tm[1] + TS[0][2] * Tm[2];
	S2[1] = TS[0][0] * Tm[3] + TS[0][1] * Tm[4] + TS[0][2] * Tm[5];
	S2[2] = TS[1][0] * Tm[3] + TS[1][1] * Tm[4] + TS[1][2] * Tm[5];
	TT[0] = Tm[0] * Tm[0] + Tm[1] * Tm[1] + Tm[2] * Tm[2];
	TT[1] = Tm[0] * Tm[3] + Tm[1] * Tm[4] + Tm[2] * Tm[5];
	TT[2] = Tm[3] * Tm[3] + Tm[4] * Tm[4] + Tm[5] * Tm[5];
}

// Control-variate quantities shared by both modes (preprocess.* in optionb_staged.py)
struct OptbComp
{
	float S2[3], P[3], S2f[3], covp[3];
	float D, m, tr, det3f, loghp;
	bool fast;
};

__device__ inline OptbComp optb_component(const float Tm[6], const float* cov6, float sigma, float nu, float s)
{
	OptbComp o;
	float TT[3];
	optb_project(Tm, cov6, o.S2, TT);
	const float s2 = sigma * sigma, s4 = s2 * s2, s6 = s4 * s2;
	for (int i = 0; i < 3; i++)
	{
		o.P[i] = s2 * TT[i];
		o.S2f[i] = o.S2[i] + o.P[i];
		o.covp[i] = o.S2f[i];
	}
	o.covp[0] += s;
	o.covp[2] += s;
	optb_invariants(cov6, o.D, o.m, o.tr);
	o.det3f = o.D + s2 * o.m + s4 * o.tr + s6;
	o.loghp = 0.5f * (logf(o.D) - logf(o.det3f) + logf(optb_det2(o.S2f)) - logf(optb_det2(o.covp)));
	const float rho = optb_threshold(nu);
	const bool ok3 = (sigma == 0.f) || optb_lmin3(cov6) >= rho * s2;
	const bool ok2 = (s == 0.f) || optb_lmin2(o.S2f) >= rho * s;
	o.fast = ok3 && ok2;
	return o;
}

// Fill one Option B slot (header + nodes). Returns the alpha-safe radius bound (pixels).
__device__ inline float optb_fill(const OptbComp& c, float sigma, float nu, float s, float opacity, float* ob)
{
	const float s2 = sigma * sigma, s4 = s2 * s2, s6 = s4 * s2;
	const float hp = expf(c.loghp);
	float iS2f[3], icovp[3];
	optb_inv2(c.S2f, iS2f);
	optb_inv2(c.covp, icovp);
	ob[OB_LOGHP] = c.loghp;
	ob[OB_HP] = hp;
	for (int i = 0; i < 3; i++)
	{
		ob[OB_LH + i] = 0.5f * (iS2f[i] - icovp[i]);
		ob[OB_S2 + i] = c.S2[i];
	}
	ob[OB_HINV + 0] = 0.5f * (1.f / c.D - 1.f / c.det3f);
	ob[OB_HINV + 1] = -0.5f * s2 / c.det3f;
	ob[OB_HINV + 2] = -0.5f * s4 / c.det3f;

	float lam[OPTB_K], logw[OPTB_K], w[OPTB_K], dlogw[OPTB_K], dlam[OPTB_K];
	optb_nodes(nu, lam, logw, w, dlogw, dlam);
	const float absop = fabsf(opacity);
	const float nterms = 2.f * OPTB_K + 1.f;
	const float lmax_covp = optb_lmax2(c.covp);
	// t-term radius (closed form), then each Gaussian term's
	const float ratio_t = fmaxf(1.f, absop * hp * nterms / OPTB_ALPHA_MIN);
	float r2 = nu * lmax_covp * (powf(ratio_t, 2.f / (nu + 2.f)) - 1.f);
	for (int k = 0; k < OPTB_K; k++)
	{
		float* nd = ob + OPTB_HEAD + k * OPTB_NODE;
		const float l = lam[k];
		const float det3 = c.D / (l * l * l) + s2 * c.m / (l * l) + s4 * c.tr / l + s6;
		float cov2[3], filt2[3], icov2[3], ifilt2[3];
		for (int i = 0; i < 3; i++)
			cov2[i] = c.S2[i] / l + c.P[i];
		filt2[0] = cov2[0] + s;
		filt2[1] = cov2[1];
		filt2[2] = cov2[2] + s;
		optb_inv2(cov2, icov2);
		optb_inv2(filt2, ifilt2);
		const float logc3 = 0.5f * (logf(c.D) - 3.f * logf(l) - logf(det3));
		const float logc2 = 0.5f * (logf(optb_det2(cov2)) - logf(optb_det2(filt2)));
		float L2[3];
		for (int i = 0; i < 3; i++)
		{
			L2[i] = 0.5f * (icov2[i] - ifilt2[i]);
			nd[ON_Q + i] = ifilt2[i];
			nd[ON_L2 + i] = L2[i];
		}
		nd[ON_LOGG0] = logw[k] + logc3 + logc2;
		nd[ON_LAM] = l;
		nd[ON_W] = w[k];
		nd[ON_C3INV + 0] = 0.5f * (1.f / c.D - 1.f / (l * l * l) / det3);
		nd[ON_C3INV + 1] = -0.5f * s2 / (l * l) / det3;
		nd[ON_C3INV + 2] = -0.5f * s4 / l / det3;
		const float ddet3 = -3.f * c.D / (l * l * l * l) - 2.f * s2 * c.m / (l * l * l) - s4 * c.tr / (l * l);
		const float dlc3 = 0.5f * (-3.f / l - ddet3 / det3);
		// tr(L2 S2) with symmetric storage: off-diagonal counted twice
		const float dlc2 = -(L2[0] * c.S2[0] + 2.f * L2[1] * c.S2[1] + L2[2] * c.S2[2]) / (l * l);
		nd[ON_A] = dlogw[k] + (dlc3 + dlc2) * dlam[k];
		nd[ON_B] = dlam[k] / (l * l);
		nd[ON_DLOGW] = dlogw[k];
		nd[ON_DLAM] = dlam[k];
		// radius terms: g_k with variance lmax(filt2_k), h_k with lmax(covp)/lam_k
		const float rg = absop * expf(nd[ON_LOGG0]) * nterms / OPTB_ALPHA_MIN;
		if (rg > 1.f)
			r2 = fmaxf(r2, 2.f * optb_lmax2(filt2) * logf(rg));
		const float rh = absop * w[k] * hp * nterms / OPTB_ALPHA_MIN;
		if (rh > 1.f)
			r2 = fmaxf(r2, 2.f * lmax_covp / l * logf(rh));
	}
	return sqrtf(fmaxf(r2, 0.f));
}

// Per-pixel value T(d) for an Option B component; Qp is the conic of covp.
__device__ __forceinline__ float optb_value(const float* ob, float4 con_o, float nu, float2 d)
{
	const float qp = con_o.x * d.x * d.x + con_o.z * d.y * d.y + 2.f * con_o.y * d.x * d.y;
	const float hp = ob[OB_HP];
	float T = hp * powf(1.f + qp / nu, -(nu + 2.f) / 2.f);
	for (int k = 0; k < OPTB_K; k++)
	{
		const float* nd = ob + OPTB_HEAD + k * OPTB_NODE;
		const float q = nd[ON_Q] * d.x * d.x + nd[ON_Q + 2] * d.y * d.y + 2.f * nd[ON_Q + 1] * d.x * d.y;
		T += expf(nd[ON_LOGG0] - 0.5f * q) - nd[ON_W] * hp * expf(-0.5f * nd[ON_LAM] * qp);
	}
	return T;
}

// Per-pixel backward for an Option B component: accumulators (G_S, G_P, g_inv) and dT/dnu, dT/dd,
// all multiplied by dLdT (pixel() in optionb_staged.py).
__device__ inline void optb_pixel_backward(const float* ob, float4 con_o, float nu, float2 d, float dLdT,
	float acc[OPTB_ACC], float& dnu, float2& dd)
{
	const float upx = con_o.x * d.x + con_o.y * d.y, upy = con_o.y * d.x + con_o.z * d.y;
	const float qp = upx * d.x + upy * d.y;
	const float hp = ob[OB_HP];
	const float base = 1.f + qp / nu;
	const float tau = hp * powf(base, -(nu + 2.f) / 2.f);
	float Cq = tau * (-(nu + 2.f) / 2.f) / nu / base;
	float Clh = tau;
	float dnu_pix = tau * (-0.5f * logf(base) + (nu + 2.f) / 2.f * qp / (nu * nu) / base);
	const float* S2 = ob + OB_S2;
	for (int i = 0; i < OPTB_ACC; i++)
		acc[i] = 0.f;
	dnu = 0.f;
	dd = { 0.f, 0.f };
	for (int k = 0; k < OPTB_K; k++)
	{
		const float* nd = ob + OPTB_HEAD + k * OPTB_NODE;
		const float ux = nd[ON_Q] * d.x + nd[ON_Q + 1] * d.y, uy = nd[ON_Q + 1] * d.x + nd[ON_Q + 2] * d.y;
		const float q = ux * d.x + uy * d.y;
		const float g = expf(nd[ON_LOGG0] - 0.5f * q);
		const float l = nd[ON_LAM];
		const float H = nd[ON_W] * hp * expf(-0.5f * l * qp);
		const float gw = dLdT * g;
		// dT/dcov2_k = 1/2 g u u^T + g L2
		const float Gn0 = 0.5f * gw * ux * ux + gw * nd[ON_L2];
		const float Gn1 = 0.5f * gw * ux * uy + gw * nd[ON_L2 + 1];
		const float Gn2 = 0.5f * gw * uy * uy + gw * nd[ON_L2 + 2];
		acc[0] += Gn0 / l; acc[1] += Gn1 / l; acc[2] += Gn2 / l;
		acc[3] += Gn0; acc[4] += Gn1; acc[5] += Gn2;
		acc[6] += gw * nd[ON_C3INV]; acc[7] += gw * nd[ON_C3INV + 1]; acc[8] += gw * nd[ON_C3INV + 2];
		const float uSu = ux * ux * S2[0] + 2.f * ux * uy * S2[1] + uy * uy * S2[2];
		dnu += gw * (nd[ON_A] - 0.5f * uSu * nd[ON_B]);
		dd.x -= gw * ux;
		dd.y -= gw * uy;
		Cq += 0.5f * l * H;
		Clh -= H;
		dnu_pix -= H * (nd[ON_DLOGW] - 0.5f * qp * nd[ON_DLAM]);
	}
	// control: covp (q') and Sigma2f (log hp) both receive equal G_S and G_P contributions
	const float cq = dLdT * Cq, cl = dLdT * Clh;
	const float Gq0 = -cq * upx * upx + cl * ob[OB_LH];
	const float Gq1 = -cq * upx * upy + cl * ob[OB_LH + 1];
	const float Gq2 = -cq * upy * upy + cl * ob[OB_LH + 2];
	acc[0] += Gq0; acc[1] += Gq1; acc[2] += Gq2;
	acc[3] += Gq0; acc[4] += Gq1; acc[5] += Gq2;
	acc[6] += cl * ob[OB_HINV]; acc[7] += cl * ob[OB_HINV + 1]; acc[8] += cl * ob[OB_HINV + 2];
	dnu += dLdT * dnu_pix;
	dd.x += 2.f * cq * upx;
	dd.y += 2.f * cq * upy;
}

// chain(): accumulators (+ Sigma3, sigma, T) -> dL/dcov6 (SSS convention: off-diagonals doubled)
// and dL/dT (2x3)
__device__ inline void optb_chain(const float* c, float sigma, const float Tm[6], const float acc[OPTB_ACC],
	float dcov6[6], float dT[6])
{
	// G as full symmetric matrices
	const float GS[2][2] = { { acc[0], acc[1] }, { acc[1], acc[2] } };
	const float GP[2][2] = { { acc[3], acc[4] }, { acc[4], acc[5] } };
	const float S[3][3] = { { c[0], c[1], c[2] }, { c[1], c[3], c[4] }, { c[2], c[4], c[5] } };
	const float T[2][3] = { { Tm[0], Tm[1], Tm[2] }, { Tm[3], Tm[4], Tm[5] } };
	float dS[3][3];
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
		{
			float v = 0.f;
			for (int a = 0; a < 2; a++)
				for (int b = 0; b < 2; b++)
					v += T[a][i] * GS[a][b] * T[b][j];
			dS[i][j] = v;
		}
	// invariants: dD/dSigma = adjugate, dm/dSigma = tr I - Sigma, dtr/dSigma = I
	const float adj[3][3] = {
		{ S[1][1] * S[2][2] - S[1][2] * S[2][1], S[0][2] * S[2][1] - S[0][1] * S[2][2], S[0][1] * S[1][2] - S[0][2] * S[1][1] },
		{ S[1][2] * S[2][0] - S[1][0] * S[2][2], S[0][0] * S[2][2] - S[0][2] * S[2][0], S[0][2] * S[1][0] - S[0][0] * S[1][2] },
		{ S[1][0] * S[2][1] - S[1][1] * S[2][0], S[0][1] * S[2][0] - S[0][0] * S[2][1], S[0][0] * S[1][1] - S[0][1] * S[1][0] } };
	const float tr = S[0][0] + S[1][1] + S[2][2];
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			dS[i][j] += acc[6] * adj[i][j] + acc[7] * ((i == j ? tr : 0.f) - S[i][j]) + (i == j ? acc[8] : 0.f);
	dcov6[0] = dS[0][0];
	dcov6[1] = dS[0][1] + dS[1][0];
	dcov6[2] = dS[0][2] + dS[2][0];
	dcov6[3] = dS[1][1];
	dcov6[4] = dS[1][2] + dS[2][1];
	dcov6[5] = dS[2][2];
	// dT = (GS + GS^T) T Sigma3 + sigma^2 (GP + GP^T) T
	const float s2 = sigma * sigma;
	for (int a = 0; a < 2; a++)
		for (int j = 0; j < 3; j++)
		{
			float v = 0.f;
			for (int b = 0; b < 2; b++)
			{
				float TS = 0.f;
				for (int k = 0; k < 3; k++)
					TS += T[b][k] * S[k][j];
				v += 2.f * GS[a][b] * TS + 2.f * s2 * GP[a][b] * T[b][j];
			}
			dT[3 * a + j] = v;
		}
}
