"""Section 10. Do the followers CAUSE the decline, or are these two trends that happen to coincide?

Table: work/decay.parquet (signals of the ten months). Always within the same month (month fixed effects), to remove
the common trend. Three tests, all written before the computation:
  T1 coin x month cross-section: is the total price response (last price before the first slice -> exit) lower on the
     coins that are followed more, at equal month and equal weight? Coin-month cells (n >= 5), weight = n, errors
     clustered by coin.
  T2 lagged: does a coin's follower intensity in month m-1 predict its response in month m (consecutive months only)?
  T3 signal by signal: total bp ~ follower volume within 2 s / vol30 + weight quartile + month fixed effect; clustered
     by day.
Criterion: a causal follower effect gives a NEGATIVE coefficient on the total response, |t| >= 3 in T1 or T3, and the
same sign in T2. Otherwise the decline is common to the whole market, not located where the followers are.
    python s10b_cross_section.py
"""
import numpy as np, pandas as pd
from common import WORK


def ols_cluster(y, X, w, g):
    """Weighted least squares with clustered standard errors (CR1). Returns coefficients and standard errors."""
    W = np.sqrt(w)[:, None]; Xw, yw = X * W, y * W[:, 0]
    XtX = np.linalg.pinv(Xw.T @ Xw); b = XtX @ Xw.T @ yw; e = yw - Xw @ b
    meat = np.zeros((X.shape[1], X.shape[1])); G = pd.factorize(g)[0]
    for k in range(G.max() + 1):
        s = Xw[G == k].T @ e[G == k]; meat += np.outer(s, s)
    n, p, m = len(y), X.shape[1], G.max() + 1
    V = XtX @ meat @ XtX * (m / (m - 1)) * ((n - 1) / (n - p))
    return b, np.sqrt(np.diag(V))


def demean(df, cols, by, w):
    out = df.copy()
    for c in cols:
        mu = df.groupby(by).apply(lambda d: np.average(d[c], weights=d[w]), include_groups=False)
        out[c] = df[c] - df[by].map(mu)
    return out


if __name__ == "__main__":
    M = pd.read_parquet(WORK / "decay.parquet")
    M = M[M["total bp"].notna() & M["followers 2 s / vol30"].notna()].copy()
    M["followed"] = M["followed within 2 s"].astype(float)
    M["lw"] = np.log(M["widened weight %"].clip(lower=0.1))
    print(f"Usable signals: {len(M):,} over ten months")
    Z = M.groupby(["coin", "month"]).agg(n=("sid", "size"), total=("total bp", "mean"), net=("net bp", "mean"), followed=("followed", "mean"),
                                         vfol=("followers 2 s / vol30", "mean"), lw=("lw", "mean")).reset_index()
    Z = Z[Z.n >= 5].copy()
    print(f"\nT1: {len(Z)} coin-month cells (n >= 5), {Z.coin.nunique()} coins")
    for yv in ("total", "net"):
        for xv in ("followed", "vfol"):
            D = demean(Z, [yv, xv, "lw"], "month", "n")
            b, se = ols_cluster(D[yv].to_numpy(), D[[xv, "lw"]].to_numpy(), D.n.to_numpy(float), D.coin.to_numpy())
            print(f"  {yv} bp ~ {xv} + log weight | month: {xv} = {b[0]:+.2f} (t {b[0] / se[0]:+.2f})")
    CONS = {"202511": "202510", "202605": "202604", "202606": "202605", "202607": "202606", "202608": "202607", "202609": "202608", "202610": "202609"}
    L = Z[Z.month.isin(CONS)].copy(); L["prev"] = L.month.map(CONS)
    L = L.merge(Z[["coin", "month", "followed", "vfol"]].rename(columns={"month": "prev", "followed": "followed_prev", "vfol": "vfol_prev"}), on=["coin", "prev"])
    print(f"\nT2: {len(L)} cells with the same coin in the previous month")
    for xv in ("followed_prev", "vfol_prev"):
        D = demean(L, ["total", xv, "lw"], "month", "n")
        b, se = ols_cluster(D["total"].to_numpy(), D[[xv, "lw"]].to_numpy(), D.n.to_numpy(float), D.coin.to_numpy())
        print(f"  total bp (m) ~ {xv} (m-1) + log weight | month: {b[0]:+.2f} (t {b[0] / se[0]:+.2f})")
    M["q weight"] = M.groupby("month")["widened weight %"].transform(lambda s: pd.qcut(s.rank(method="first"), 4, labels=False))
    M["cell"] = M.month + "_" + M["q weight"].astype(str)
    for xv in ("followed", "followers 2 s / vol30"):
        x = M[xv].clip(upper=M[xv].quantile(.99)) if xv != "followed" else M[xv]
        D = demean(pd.DataFrame({"y": M["total bp"], "x": x, "lw": M.lw, "cell": M.cell, "w": 1.0, "day": M.day}), ["y", "x", "lw"], "cell", "w")
        b, se = ols_cluster(D.y.to_numpy(), D[["x", "lw"]].to_numpy(), D.w.to_numpy(), D.day.to_numpy())
        print(f"\nT3 signal by signal: total bp ~ {xv} + log weight | month x weight quartile: {b[0]:+.2f} (t {b[0] / se[0]:+.2f}), clustered by day")
    D = pd.DataFrame({"y": M["before me bp"], "x": M.followed, "lw": M.lw, "cell": M.cell, "w": 1.0, "day": M.day}).dropna()
    D = demean(D, ["y", "x", "lw"], "cell", "w"); b, se = ols_cluster(D.y.to_numpy(), D[["x", "lw"]].to_numpy(), D.w.to_numpy(), D.day.to_numpy())
    print(f"Consistency check: taken before me bp ~ followed: {b[0]:+.2f} (t {b[0] / se[0]:+.2f})")
