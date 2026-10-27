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
    float ic[3]; optb_inv2(c.covp, ic);
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
    optb_chain(cov6, sigma, Tm, acc, dcov6, dT);
}
