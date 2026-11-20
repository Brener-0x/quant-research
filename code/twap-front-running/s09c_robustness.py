"""Section 9. Robustness of the headline t statistics to autocorrelated days.

Daily series rebuilt from s08h (in sample) and s09b (out of sample). For each series: the plain daily t (as in the
note), a Newey-West t (Bartlett kernel, L = 4, 7 and 14 days), autocorrelations 1 to 5, and a circular block bootstrap
(blocks of 5, 10 and 20 days, 20,000 draws): 95 % interval of the mean and share of draws <= 0.
    python s09c_robustness.py
"""
import numpy as np, pandas as pd
from common import show
from s08d_impact import frozen_curves
from s08h_wake_v1 import B, K0, S0, in_sample, simulate
from s09b_oos import DAYS as OOS, oos_paths

COL = "A2 +2000"
RNG = np.random.default_rng(20261120)


def t_newey_west(x, L):
    x = np.asarray(x, float); T = len(x); e = x - x.mean()
    s = e @ e / T
    for k in range(1, L + 1):
        s += 2 * (1 - k / (L + 1)) * (e[k:] @ e[:-k]) / T
    return x.mean() / np.sqrt(s / T)


def block_bootstrap(x, b, n=20_000):
    x = np.asarray(x, float); T = len(x); k = int(np.ceil(T / b))
    d = RNG.integers(0, T, size=(n, k))
    idx = (d[:, :, None] + np.arange(b)[None, None, :]).reshape(n, -1)[:, :T] % T
    m = x[idx].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5), (m <= 0).mean()


def row(name, x):
    x = np.asarray(x, float); T = len(x)
    r = {"series": name, "days": T, "mean": x.mean(), "plain t": x.mean() / x.std(ddof=1) * np.sqrt(T)}
    for L in (4, 7, 14):
        r[f"NW t L={L}"] = t_newey_west(x, L)
    e = x - x.mean()
    for k in range(1, 6):
        r[f"ac{k}"] = (e[k:] @ e[:-k]) / (e @ e)
    for b in (5, 10, 20):
        lo, hi, p = block_bootstrap(x, b)
        r[f"95% CI b={b}"] = f"[{lo:.2f} ; {hi:.2f}]"; r[f"p<=0 b={b}"] = p
    return r


if __name__ == "__main__":
    Ra = in_sample(); Ra = Ra[Ra["px entry"].notna() & Ra[COL].notna()]
    Ro = oos_paths(); Ro = Ro[Ro["px entry"].notna() & Ro[COL].notna()]
    da = sorted(Ra.day.unique()); Zs, edges = frozen_curves()
    S = {"edge in sample (bp per position, daily mean)": Ra.groupby("day")[COL].mean().reindex(da).dropna(),
         "edge out of sample (bp per position, daily mean)": Ro.groupby("day")[COL].mean().reindex(OOS).dropna()}
    for name, R, dx in (("in sample", Ra, da), ("out of sample", Ro, OOS)):
        P = simulate(R, 0.01 * 25_000 / B, S0, K0, COL, Zs, edges, per_coin=2)
        S[f"Wake v2 {name}, 25k, 1 % (dollars per day)"] = P[P.taken].groupby("day")["pnl $"].sum().reindex(dx, fill_value=0)
    show(); pd.options.display.float_format = "{:.3f}".format
    print(pd.DataFrame([row(k, v.to_numpy()) for k, v in S.items()]).T.to_string(header=False))
