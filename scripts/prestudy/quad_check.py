"""Quick quadrature-error check for Option B (isotropic 2D, d = 2).

T_mip(r) = sum_k w_k * g_k(r),  g_k = (sig2/l_k)/(sig2/l_k + s) * exp(-r^2 / (2 (sig2/l_k + s)))
w_k ∝ Gamma(l_k; nu/2, nu/2) * l_k (log-grid Jacobian) * l_k^(d/2)

Reference: same integrand integrated over z = log(lambda) with a very fine trapezoid
(spectrally accurate for this smooth, fast-decaying integrand); checked against the
closed-form t at s = 0.
"""
import numpy as np
from scipy.special import gammaln, digamma, polygamma

D = 2
NUS = [1, 1.5, 2, 4, 8, 16, 30]
KS = [4, 8, 16, 32]
SIG2S = [0.01, 0.1, 1.0, 10.0]   # component variance in px^2 (sub-pixel .. large)
SS = [0.0, 0.1]                   # 2D filter variance in px^2
CORE, TAIL = 1e-3, 1e-6


def log_weights(z, nu):
    lam = np.exp(z)
    a = nu / 2
    log_gamma_pdf = a * np.log(a) - gammaln(a) + (a - 1) * z - a * lam
    return log_gamma_pdf + z + (D / 2) * z          # Jacobian + peak-1 factor


def t_mip(r2, z, nu, sig2, s):
    lw = log_weights(z, nu)
    w = np.exp(lw - lw.max())
    w /= w.sum()
    var = sig2 * np.exp(-z) + s                       # per-node filtered variance
    peak = (sig2 * np.exp(-z)) / var                  # sqrt(|Σ/λ|/|A|) for isotropic 2D
    g = peak[None, :] * np.exp(-0.5 * r2[:, None] / var[None, :])
    return g @ w


def reference(r2, nu, sig2, s):
    z = np.linspace(-45, 10, 40001)
    return t_mip(r2, z, nu, sig2, s)


def radii(nu, sig2, s):
    # radius where the s=0 kernel hits ~1e-7, widened by the filter
    delta = nu * (1e7 ** (2 / (nu + 2)) - 1)
    rmax = np.sqrt(delta * sig2 + 30 * s) + 1e-9
    return np.concatenate([[0.0], np.geomspace(1e-4, rmax, 600)]) ** 2


def grids(nu, K):
    out = {}
    out["fixed[-12,3]"] = np.linspace(-12, 3, K)
    # README adaptive: centre/scale from Gamma(nu/2) in log-space
    c, w = digamma(nu / 2) - np.log(nu / 2), np.sqrt(polygamma(1, nu / 2))
    out["readme-adaptive"] = np.linspace(c - 6 * w, c + 3 * w, K)
    # weight-matched adaptive: the w_k follow log-Gamma((nu+d)/2, nu/2)
    a2 = (nu + D) / 2
    c, w = digamma(a2) - np.log(nu / 2), np.sqrt(polygamma(1, a2))
    shift = np.log(1 / CORE) / a2                     # left shift of integrand at the core edge
    out["matched-adaptive"] = np.linspace(c - 4 * w - shift, c + 4 * w, K)
    return out


def main():
    # sanity: reference vs closed form at s = 0
    worst = 0.0
    for nu in NUS:
        r2 = radii(nu, 1.0, 0.0)
        ref = reference(r2, nu, 1.0, 0.0)
        exact = (1 + r2 / nu) ** (-(nu + 2) / 2)
        m = exact > TAIL
        worst = max(worst, np.max(np.abs(ref[m] - exact[m]) / exact[m]))
    print(f"reference vs closed form (s=0): max rel err = {worst:.2e}\n")

    refs = {}
    for nu in NUS:
        for sig2 in SIG2S:
            for s in SS:
                r2 = radii(nu, sig2, s)
                refs[nu, sig2, s] = (r2, reference(r2, nu, sig2, s))

    names = list(grids(1, 4).keys())
    for name in names:
        print(f"== grid: {name} ==  max rel error, core (T>={CORE}) / tail ({TAIL}<=T<{CORE}),"
              f" worst over sig2 in {SIG2S}, s in {SS}")
        print("      columns per K: core rel / tail rel / max abs / mass rel")
        print("nu    " + "".join(f"K={K:<38}" for K in KS))
        for nu in NUS:
            row = f"{nu:<6}"
            for K in KS:
                z = grids(nu, K)[name]
                lw = log_weights(z, nu)
                w = np.exp(lw - lw.max())
                mass = np.sum(w * np.exp(-z)) / np.sum(w)     # ∫T_mip / (2π|Σ|^½); exact = 1
                ce = te = ae = 0.0
                for sig2 in SIG2S:
                    for s in SS:
                        r2, ref = refs[nu, sig2, s]
                        approx = t_mip(r2, z, nu, sig2, s)
                        rel = np.abs(approx - ref) / ref
                        core = ref >= CORE
                        tail = (ref >= TAIL) & ~core
                        ce = max(ce, rel[core].max())
                        ae = max(ae, np.abs(approx - ref).max())
                        if tail.any():
                            te = max(te, rel[tail].max())
                row += f"{ce:7.1e}/{te:7.1e}/{ae:7.1e}/{abs(mass - 1):7.1e}  "
            print(row)
        print()


if __name__ == "__main__":
    main()
