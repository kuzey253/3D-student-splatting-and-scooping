"""Max abs error of the closed-form shortcuts vs the exact filtered t (isotropic 2D, d = 2).

- Option A:  t with Σ + sI, opacity × sqrt(|Σ|/|Σ'|)
- Option A': t with Σ + s(ν+2)/ν·I (matches the t's core curvature), same normalization
- Gaussian:  the README's large-ν fallback, Mip-Splatting's G_mip on Σ

Reference = exact λ-integral from quad_check.py. Errors are in alpha units at |o| = 1;
one 8-bit step is 1/255 ≈ 3.9e-3.
"""
import numpy as np
from quad_check import reference, radii

NUS = [1, 2, 4, 8, 16, 30, 50, 100, 300, 1000, 10000]
SIG2S = [0.01, 0.1, 1.0, 10.0]
S = 0.1


def t_kernel(r2, nu, var, sig2):
    return (sig2 / var) * (1 + r2 / (nu * var)) ** (-(nu + 2) / 2)


def main():
    refs = {(nu, sig2): (radii(min(nu, 100), sig2, S), None) for nu in NUS for sig2 in SIG2S}
    refs = {k: (r2, reference(r2, k[0], k[1], S)) for k, (r2, _) in refs.items()}

    shortcuts = {
        "Option A": lambda r2, nu, sig2: t_kernel(r2, nu, sig2 + S, sig2),
        "Option A'": lambda r2, nu, sig2: t_kernel(r2, nu, sig2 + S * (nu + 2) / nu, sig2),
        "Gaussian fallback": lambda r2, nu, sig2: (sig2 / (sig2 + S)) * np.exp(-0.5 * r2 / (sig2 + S)),
    }
    for name, fn in shortcuts.items():
        print(f"== {name} ==  s = {S} px^2, max abs error over r, columns sig2 = {SIG2S} px^2")
        for nu in NUS:
            row = "".join(f"{np.abs(fn(r2, nu, sig2) - ref).max():9.1e}"
                          for sig2 in SIG2S for r2, ref in [refs[nu, sig2]])
            print(f"nu={nu:<7}{row}")
        print()


if __name__ == "__main__":
    main()
