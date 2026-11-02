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
 * Float32-safe forms (NOTES.md 43): the D gradient is carried as D·d/dD and chained with Sigma3^-1;
 * the 3D terms use e = det3f - D and L3_k = lam^3 det3_k = D + e3_k, so they vanish exactly at
 * sigma = 0, where D is not used (c3 = 1, cov2_k = Sigma2/lam_k exactly).
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

// Cancellation-free 2x2 determinants (NOTES.md 44). a*c - b*b cancels catastrophically for the
// nearly rank-1 footprints of needle and flat components (Sigma2/lam_k reaches ~1e7 px^2), so every
// determinant is built from non-negative pieces: det(A + B) = det A + det B + tr(adj(A) B) and
// det(A + sI) = det A + s tr A + s^2; inverses and eigenvalues then use the given determinant.
__device__ __forceinline__ float optb_cross2(const float A[3], const float B[3])
{
	return fmaxf(0.f, A[0] * B[2] + A[2] * B[0] - 2.f * A[1] * B[1]);  // tr(adj(A) B) >= 0 for PSD A, B
}

// Entry by entry, as M2's c_0/det_0: for a tiny component det is denormal and 1/det alone overflows
// (then 0 * inf = NaN when no pixel touched it), while every m_ij/det stays finite (NOTES.md 45).
__device__ __forceinline__ void optb_inv2d(const float m[3], float det, float out[3])
{
	out[0] = m[2] / det;
	out[1] = -m[1] / det;
	out[2] = m[0] / det;
}

__device__ __forceinline__ float optb_lmax2d(const float m[3], float det)
{
	const float mid = 0.5f * (m[0] + m[2]);
	return mid + sqrtf(fmaxf(0.f, mid * mid - det));
}

__device__ __forceinline__ float optb_lmin2d(const float m[3], float det)
{
	const float mid = 0.5f * (m[0] + m[2]);
	return det / (mid + sqrtf(fmaxf(0.f, mid * mid - det)));  // det / lmax: no cancellation
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
	float D, m, tr, e, det3f, loghp;
	// det Sigma2 (clamped >= 0), tr(adj(Sigma2) P), det P, det Sigma2f, det covp
	float detS2, crossP, detP, detS2f, detcovp;
	bool fast;
};

// keep: det Sigma2 and D as preprocessCUDA computed them (NOTES.md 47). The other kernels pass them so a
// component drawn in the forward pass never meets a rounded-to-zero determinant in a later kernel.
__device__ inline OptbComp optb_component(const float Tm[6], const float* cov6, float sigma, float nu, float s,
	const float* keep = nullptr)
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
	if (keep != nullptr)
		o.D = keep[1];
	o.e = s2 * o.m + s4 * o.tr + s6;
	o.det3f = o.D + o.e;
	// det Sigma2 as a*c - b*b clamped at 0 (more accurate in float32 than Cauchy-Binet for ordinary
	// footprints; a rounded 0 then skips the component at sigma = 0, as M2 does); det(T T^T) = |T_0 x T_1|^2
	o.detS2 = keep != nullptr ? keep[0] : fmaxf(0.f, o.S2[0] * o.S2[2] - o.S2[1] * o.S2[1]);
	const float n0 = Tm[1] * Tm[5] - Tm[2] * Tm[4], n1 = Tm[2] * Tm[3] - Tm[0] * Tm[5], n2 = Tm[0] * Tm[4] - Tm[1] * Tm[3];
	o.detP = s4 * (n0 * n0 + n1 * n1 + n2 * n2);
	o.crossP = optb_cross2(o.S2, o.P);
	o.detS2f = o.detS2 + o.crossP + o.detP;
	o.detcovp = o.detS2f + s * (o.S2f[0] + o.S2f[2]) + s * s;
	o.loghp = 0.5f * (logf(o.detS2f) - logf(o.detcovp));
	if (sigma > 0.f)
		o.loghp += 0.5f * (logf(o.D) - logf(o.det3f));
	const float rho = optb_threshold(nu);
	const bool ok3 = (sigma == 0.f) || rho <= 0.f || optb_lmin3(cov6) >= rho * s2;
	const bool ok2 = (s == 0.f) || rho <= 0.f || optb_lmin2d(o.S2f, o.detS2f) >= rho * s;
	o.fast = ok3 && ok2;
	return o;
}

// Fill one Option B slot (header + nodes). Returns the alpha-safe radius bound (pixels).
__device__ inline float optb_fill(const OptbComp& c, float sigma, float nu, float s, float opacity, float* ob)
{
	const float s2 = sigma * sigma, s4 = s2 * s2, s6 = s4 * s2;
	const float hp = expf(c.loghp);
	float iS2f[3], icovp[3];
	optb_inv2d(c.S2f, c.detS2f, iS2f);
	optb_inv2d(c.covp, c.detcovp, icovp);
	ob[OB_LOGHP] = c.loghp;
	ob[OB_HP] = hp;
	for (int i = 0; i < 3; i++)
	{
		ob[OB_LH + i] = 0.5f * (iS2f[i] - icovp[i]);
		ob[OB_S2 + i] = c.S2[i];
	}
	// D·d(log hp)/dD, d/dm, d/dtr; all zero at sigma = 0 (D may then be 0 for a flat component)
	const bool on3 = sigma > 0.f;
	ob[OB_HINV + 0] = on3 ? 0.5f * c.e / c.det3f : 0.f;
	ob[OB_HINV + 1] = on3 ? -0.5f * s2 / c.det3f : 0.f;
	ob[OB_HINV + 2] = on3 ? -0.5f * s4 / c.det3f : 0.f;

	float lam[OPTB_K], logw[OPTB_K], w[OPTB_K], dlogw[OPTB_K], dlam[OPTB_K];
	optb_nodes(nu, lam, logw, w, dlogw, dlam);
	const float absop = fabsf(opacity);
	const float nterms = 2.f * OPTB_K + 1.f;
	const float lmax_covp = optb_lmax2d(c.covp, c.detcovp);
	// t-term radius (closed form), then each Gaussian term's
	const float ratio_t = fmaxf(1.f, absop * hp * nterms / OPTB_ALPHA_MIN);
	float r2 = nu * lmax_covp * (powf(ratio_t, 2.f / (nu + 2.f)) - 1.f);
	for (int k = 0; k < OPTB_K; k++)
	{
		float* nd = ob + OPTB_HEAD + k * OPTB_NODE;
		const float l = lam[k];
		float cov2[3], filt2[3], icov2[3], ifilt2[3];
		for (int i = 0; i < 3; i++)
			cov2[i] = c.S2[i] / l + c.P[i];
		filt2[0] = cov2[0] + s;
		filt2[1] = cov2[1];
		filt2[2] = cov2[2] + s;
		// det cov2 = det(Sigma2)/lam^2 + tr(adj(Sigma2) P)/lam + det P; det filt2 = det cov2 + s tr cov2 + s^2
		const float detcov2 = c.detS2 / (l * l) + c.crossP / l + c.detP;
		const float detfilt2 = detcov2 + s * (cov2[0] + cov2[2]) + s * s;
		optb_inv2d(cov2, detcov2, icov2);
		optb_inv2d(filt2, detfilt2, ifilt2);
		float logc3 = 0.f, dlc3 = 0.f;
		nd[ON_C3INV + 0] = nd[ON_C3INV + 1] = nd[ON_C3INV + 2] = 0.f;
		if (on3)
		{
			// L3 = lam^3 det(Sigma3/lam + sigma^2 I) = D + e3; c3 = sqrt(D / L3)
			const float e3 = l * s2 * c.m + l * l * s4 * c.tr + l * l * l * s6;
			const float L3 = c.D + e3;
			logc3 = 0.5f * (logf(c.D) - logf(L3));
			nd[ON_C3INV + 0] = 0.5f * e3 / L3;  // D·d(log c3)/dD
			nd[ON_C3INV + 1] = -0.5f * s2 * l / L3;
			nd[ON_C3INV + 2] = -0.5f * s4 * l * l / L3;
			dlc3 = -0.5f * (s2 * c.m + 2.f * l * s4 * c.tr + 3.f * l * l * s6) / L3;
		}
		const float logc2 = 0.5f * (logf(detcov2) - logf(detfilt2));
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
		// d(log c2)/dlam from the determinants as computed (= -tr(L2 Sigma2)/lam^2 in exact arithmetic)
		const float ddetcov2 = -2.f * c.detS2 / (l * l * l) - c.crossP / (l * l);
		const float ddetfilt2 = ddetcov2 - s * (c.S2[0] + c.S2[2]) / (l * l);
		const float dlc2 = 0.5f * (ddetcov2 / detcov2 - ddetfilt2 / detfilt2);
		nd[ON_A] = dlogw[k] + (dlc3 + dlc2) * dlam[k];
		nd[ON_B] = dlam[k] / (l * l);
		nd[ON_DLOGW] = dlogw[k];
		nd[ON_DLAM] = dlam[k];
		// radius terms: g_k with variance lmax(filt2_k), h_k with lmax(covp)/lam_k
		const float rg = absop * expf(nd[ON_LOGG0]) * nterms / OPTB_ALPHA_MIN;
		if (rg > 1.f)
			r2 = fmaxf(r2, 2.f * optb_lmax2d(filt2, detfilt2) * logf(rg));
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
__device__ inline void optb_chain(const float* c, float sigma, float D, const float Tm[6], const float acc[OPTB_ACC],
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
	// invariants (sigma > 0 only; at sigma = 0 they do not enter): acc[6] is D·dL/dD and
	// (dD/dSigma)/D = adj/D = Sigma^-1; dm/dSigma = tr I - Sigma, dtr/dSigma = I
	if (sigma > 0.f)
	{
		const float adj[3][3] = {
			{ S[1][1] * S[2][2] - S[1][2] * S[2][1], S[0][2] * S[2][1] - S[0][1] * S[2][2], S[0][1] * S[1][2] - S[0][2] * S[1][1] },
			{ S[1][2] * S[2][0] - S[1][0] * S[2][2], S[0][0] * S[2][2] - S[0][2] * S[2][0], S[0][2] * S[1][0] - S[0][0] * S[1][2] },
			{ S[1][0] * S[2][1] - S[1][1] * S[2][0], S[0][1] * S[2][0] - S[0][0] * S[2][1], S[0][0] * S[1][1] - S[0][1] * S[1][0] } };
		// adj/D entry by entry: 1/D alone can overflow for a denormal D while every ratio is finite.
		// D is the forward pass's (OptbComp.D), positive whenever sigma > 0 and the component is drawn.
		const float tr = S[0][0] + S[1][1] + S[2][2];
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				dS[i][j] += acc[6] * (adj[i][j] / D) + acc[7] * ((i == j ? tr : 0.f) - S[i][j]) + (i == j ? acc[8] : 0.f);
	}
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

// Per-component backward of Option B mode (computeCovOptionB): a fast-path component's conic and
// effective-opacity (o·hp) gradients, or an Option B component's render-pass accumulators, chained
// to dL/dcov6 (SSS convention) and dL/dt (camera-space mean, before the view rotation).
__device__ inline void optb_cov_backward(const OptbComp& oc, const float* cov3D, float sigma, const float Tm[6],
	const float3& t, float xmul, float ymul, float h_x, float h_y, const float* view, const float* dL_dconic4,
	bool from_B, const float* accB, float opacity, float& dL_dopacity, float dcov6[6], float3& dL_dt)
{
	// A component no pixel used (alpha < 1/255 everywhere, e.g. a tiny one kept by SSS's >= 2 px
	// radius rule) has zero gradients exactly. Return them without evaluating its constants: at
	// sigma = 0 their determinants can underflow and 0 * inf would give NaN (NOTES.md 45).
	bool used = false;
	if (from_B)
		for (int i = 0; i < OPTB_ACC; i++)
			used |= accB[i] != 0.f;
	else
		used = dL_dconic4[0] != 0.f || dL_dconic4[1] != 0.f || dL_dconic4[3] != 0.f || dL_dopacity != 0.f;
	if (!used)
	{
		for (int i = 0; i < 6; i++)
			dcov6[i] = 0.f;
		dL_dt = { 0.f, 0.f, 0.f };
		return;
	}
	float acc[OPTB_ACC];
	if (!from_B)
	{
		// conic gradient w.r.t. covp, with SSS's formula (including its 1e-7 damping)
		const float a = oc.covp[0], b = oc.covp[1], c = oc.covp[2];
		const float3 dL_dconic = { dL_dconic4[0], dL_dconic4[1], dL_dconic4[3] };
		const float denom = oc.detcovp;  // the cancellation-free det of covp (SSS: a * c - b * b)
		const float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);
		const float dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
		const float dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
		const float dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);
		// effective opacity o·hp
		const float hp = expf(oc.loghp);
		const float dL_dopeff = dL_dopacity;
		const float dlh = dL_dopeff * opacity * hp;
		dL_dopacity = dL_dopeff * hp;
		float iS2f[3], icovp[3];
		optb_inv2d(oc.S2f, oc.detS2f, iS2f);
		optb_inv2d(oc.covp, oc.detcovp, icovp);
		const float s2 = sigma * sigma;
		const float G[3] = { dL_da + dlh * 0.5f * (iS2f[0] - icovp[0]),
			0.5f * dL_db + dlh * 0.5f * (iS2f[1] - icovp[1]),
			dL_dc + dlh * 0.5f * (iS2f[2] - icovp[2]) };
		for (int i = 0; i < 3; i++)
		{
			acc[i] = G[i];
			acc[3 + i] = G[i];
		}
		// D·dL/dD, dL/dm, dL/dtr (mip_optionb.h); zero at sigma = 0, where D may be 0
		const bool on3 = sigma > 0.f;
		acc[6] = on3 ? dlh * 0.5f * oc.e / oc.det3f : 0.f;
		acc[7] = on3 ? dlh * -0.5f * s2 / oc.det3f : 0.f;
		acc[8] = on3 ? dlh * -0.5f * s2 * s2 / oc.det3f : 0.f;
	}
	else
	{
		for (int i = 0; i < OPTB_ACC; i++)
			acc[i] = accB[i];
	}
	float dT[6];
	optb_chain(cov3D, sigma, oc.D, Tm, acc, dcov6, dT);
	// T = J W (W[i][j] = view[4j + i]): gradients for J's non-zero entries, then t, as computeCov2DCUDA
	float dJ00 = 0.f, dJ02 = 0.f, dJ11 = 0.f, dJ12 = 0.f;
	for (int j = 0; j < 3; j++)
	{
		dJ00 += dT[j] * view[4 * j + 0];
		dJ02 += dT[j] * view[4 * j + 2];
		dJ11 += dT[3 + j] * view[4 * j + 1];
		dJ12 += dT[3 + j] * view[4 * j + 2];
	}
	const float tz = 1.f / t.z, tz2 = tz * tz, tz3 = tz2 * tz;
	const float dL_dtx = xmul * -h_x * tz2 * dJ02;
	const float dL_dty = ymul * -h_y * tz2 * dJ12;
	const float dL_dtz = -h_x * tz2 * dJ00 - h_y * tz2 * dJ11 + (2 * h_x * t.x) * tz3 * dJ02 + (2 * h_y * t.y) * tz3 * dJ12;
	dL_dt = { dL_dtx, dL_dty, dL_dtz };
}
