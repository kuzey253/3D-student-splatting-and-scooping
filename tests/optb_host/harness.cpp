// One Option B component through mip_optionb.h as preprocess + render + backward use it.
#include "stub.h"
#include "mip_optionb.h"
extern "C" void run(const float* Tm, const float* cov6, float sigma, float nu, float s, float opacity, int n,
                    const float* d, const float* dLdT, float* outT, float* acc, float* dnu, float* dd,
                    float* dcov6, float* dT, int* fast, float* loghp, float* radius)
{
    OptbComp c = optb_component(Tm, cov6, sigma, nu, s);
    *fast = c.fast; *loghp = c.loghp;
    static float ob[OPTB_STRIDE];
    *radius = optb_fill(c, sigma, nu, s, opacity, ob);
    float ic[3]; optb_inv2d(c.covp, c.detcovp, ic);
    float4 con = { ic[0], ic[1], ic[2], opacity };
    for (int i = 0; i < OPTB_ACC; i++) acc[i] = 0.f;
    *dnu = 0.f;
    for (int p = 0; p < n; p++) {
        float2 dp = { d[2*p], d[2*p+1] };
        outT[p] = optb_value(ob, con, nu, dp);
        float a[OPTB_ACC], dn; float2 ddp;
        optb_pixel_backward(ob, con, nu, dp, dLdT[p], a, dn, ddp);
        for (int i = 0; i < OPTB_ACC; i++) acc[i] += a[i];
        *dnu += dn; dd[2*p] = ddp.x; dd[2*p+1] = ddp.y;
    }
    optb_chain(cov6, sigma, c.D, Tm, acc, dcov6, dT);
}

// The whole per-component path of Option B mode, as forward.cu / backward.cu run it: projection
// (identity view), mode decision, slot fill, per-pixel values and backward (pixel offsets are given
// as fractions of the alpha-safe radius), then optb_cov_backward. Returns 0 if preprocess skips it.
extern "C" int run_full(const float* mean, const float* cov6, float sigma, float nu, float s, float opacity,
                        float focal, float tan_fov, int n, const float* dfrac, const float* dLdT,
                        const float* dL_dconic4, float dL_dop_in,
                        float* outT, float* dnu_out, float* dd, float* dcov6, float* dL_dt, float* dL_dop,
                        int* fast, float* radius)
{
    const float view[16] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
    float Tm[6], xmul, ymul;
    float3 t;
    optb_projection({ mean[0], mean[1], mean[2] }, focal, focal, tan_fov, tan_fov, view, Tm, t, xmul, ymul);
    const OptbComp c = optb_component(Tm, cov6, sigma, nu, s);
    if ((sigma > 0.f && !(c.D > 0.f)) || !std::isfinite(c.loghp))
        return 0;
    *fast = c.fast;
    float ic[3];
    optb_inv2d(c.covp, c.detcovp, ic);
    const float4 con = { ic[0], ic[1], ic[2], c.fast ? opacity * expf(c.loghp) : opacity };
    float acc[OPTB_ACC] = { 0 };
    *dnu_out = 0.f;
    *radius = 0.f;
    if (!c.fast)
    {
        static float ob[OPTB_STRIDE];
        *radius = optb_fill(c, sigma, nu, s, opacity, ob);
        for (int p = 0; p < n; p++)
        {
            const float2 dp = { dfrac[2 * p] * *radius, dfrac[2 * p + 1] * *radius };
            outT[p] = optb_value(ob, con, nu, dp);
            dd[2 * p] = dd[2 * p + 1] = 0.f;
            // the render passes only pixels with |alpha| >= 1/255 to the backward
            if (!(fabsf(con.w * outT[p]) >= 1.f / 255.f))
                continue;
            float a[OPTB_ACC], dn;
            float2 ddp;
            optb_pixel_backward(ob, con, nu, dp, con.w * dLdT[p], a, dn, ddp);
            for (int i = 0; i < OPTB_ACC; i++)
                acc[i] += a[i];
            *dnu_out += dn;
            dd[2 * p] = ddp.x;
            dd[2 * p + 1] = ddp.y;
        }
    }
    // a fast-path component no pixel can use (|o hp| < 1/255 at its peak) receives no gradients
    const bool seen = c.fast ? fabsf(con.w) >= 1.f / 255.f : true;
    const float zero4[4] = { 0.f, 0.f, 0.f, 0.f };
    float op = seen ? dL_dop_in : 0.f;
    float3 dt;
    optb_cov_backward(c, cov6, sigma, Tm, t, xmul, ymul, focal, focal, view, seen ? dL_dconic4 : zero4, !c.fast, acc,
                      opacity, op, dcov6, dt);
    dL_dt[0] = dt.x; dL_dt[1] = dt.y; dL_dt[2] = dt.z;
    *dL_dop = op;
    return 1;
}

// Replay of one captured component's per-component backward (scripts/m4_nan_replay.py): camera-space
// mean and covariance, the camera's intrinsics, and the incoming gradients as the GPU had them.
// keep (nullable): det Sigma2 and D from the forward pass, as backward.cu passes them.
extern "C" int run_cov_backward(const float* mean_cam, const float* cov6_cam, float sigma, float nu, float s,
                                float opacity, float fx, float fy, float tanx, float tany,
                                const float* dL_dconic4, float dL_dop_in, const float* accB,
                                float* dcov6, float* dL_dt, float* dL_dop, int* fast, float* loghp,
                                const float* keep)
{
    const float view[16] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
    float Tm[6], xmul, ymul;
    float3 t;
    optb_projection({ mean_cam[0], mean_cam[1], mean_cam[2] }, fx, fy, tanx, tany, view, Tm, t, xmul, ymul);
    const OptbComp c = optb_component(Tm, cov6_cam, sigma, nu, s, keep);
    *fast = c.fast;
    *loghp = c.loghp;
    // without keep, apply preprocess's cull; with it, run as computeCovOptionB does (no cull there)
    if (keep == nullptr && ((sigma > 0.f && !(c.D > 0.f)) || !std::isfinite(c.loghp)))
        return 0;
    float op = dL_dop_in;
    float3 dt;
    optb_cov_backward(c, cov6_cam, sigma, Tm, t, xmul, ymul, fx, fy, view, dL_dconic4, !c.fast, accB, opacity, op,
                      dcov6, dt);
    dL_dt[0] = dt.x; dL_dt[1] = dt.y; dL_dt[2] = dt.z;
    *dL_dop = op;
    return 1;
}
