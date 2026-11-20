"""Section 4. The drift without look-ahead.

Pre-registered: every TWAP with >= 2 slices, no size filter; main entry = second slice + 1 s (comparisons: first slice
+ 1 s, before the first slice). Classes known at entry: size of the first two slices x the wallet's past (earlier days)
x reference class. Controls: random time and side; momentum (sign of the previous 30 minutes).
Criterion: a class >= 9 bp at 30 min, daily t >= 3, random control within 2 bp of zero, and >= 9 bp above momentum.
    python s04b_controls.py   -> work/first_slices.parquet, work/controls.parquet
"""
import numpy as np, pandas as pd
from common import DAYS, WORK, price_at, series2, show, t_day
from s04a_drift import HORIZONS

LARGE = 100_000
rng = np.random.default_rng(35)
SIZES = [0, 100, 1_000, 10_000, np.inf]; SIZE_NAMES = ["< 100", "100-1,000", "1,000-10,000", ">= 10,000"]


def first_slices():
    """Time and notional of the first two slices of every TWAP."""
    f = WORK / "first_slices.parquet"
    if f.exists():
        return pd.read_parquet(f)
    L = []
    for d in DAYS:
        D = pd.read_parquet(WORK / "twap_fills" / f"{d}.parquet", columns=["user", "twapId", "t", "usd"])
        g = D.groupby(["user", "twapId", "t"], sort=False).usd.sum().reset_index().sort_values(["user", "twapId", "t"])
        g["k"] = g.groupby(["user", "twapId"]).cumcount(); L.append(g[g.k < 2].drop(columns="k"))
    A = pd.concat(L).sort_values(["user", "twapId", "t"]); A["k"] = A.groupby(["user", "twapId"]).cumcount(); A = A[A.k < 2]
    P = A.pivot_table(index=["user", "twapId"], columns="k", values=["t", "usd"]).reset_index()
    P.columns = ["user", "twapId", "t1", "t2", "usd1", "usd2"]; P.to_parquet(f, index=False); return P


if __name__ == "__main__":
    P = first_slices()
    T = pd.read_parquet(WORK / "twaps.parquet", columns=["user", "twapId", "coin", "usd", "net_side", "reference", "day0", "end"])
    T = T.merge(P, on=["user", "twapId"]); T = T[T.t2.notna()].reset_index(drop=True)
    T["class"] = np.where(T.reference == "external reference", "external reference", "no external reference")
    T["slice size"] = pd.cut((T.usd1 + T.usd2) / 2, SIZES, labels=SIZE_NAMES, right=False)
    # the wallet's past: strictly earlier days
    Jd = T.assign(large=T.usd >= LARGE).groupby(["user", "day0"]).agg(n=("usd", "size"), g=("large", "sum")).reset_index().sort_values(["user", "day0"])
    Jd["n_before"] = Jd.groupby("user").n.cumsum() - Jd.n; Jd["g_before"] = Jd.groupby("user").g.cumsum() - Jd.g
    T = T.merge(Jd[["user", "day0", "n_before", "g_before"]], on=["user", "day0"])
    T["past"] = np.select([T.n_before == 0, T.g_before == 0], ["new", "no large"], "with large")
    L = []
    for d in DAYS:
        Td = T[T.day0 == d]
        if len(Td) == 0:
            continue
        S = series2(d); d0 = int(pd.Timestamp(d).value // 1_000_000)
        for coin, G in Td.groupby("coin"):
            s = S.get(coin)
            if s is None:
                continue
            sg, t1, t2 = G.net_side.to_numpy(), G.t1.to_numpy().astype(np.int64), G.t2.to_numpy().astype(np.int64)
            D = G[["user", "twapId", "coin", "day0", "class", "slice size", "past", "usd", "end"]].rename(columns={"day0": "day"}).copy()
            D["t2"] = t2
            for name, te in (("2nd slice", t2 + 1000), ("1st slice", t1 + 1000), ("before", t1 - 1000)):
                p0 = price_at(s, te)
                for h, dt in HORIZONS.items():
                    D[f"{name} {h}"] = sg * np.log(price_at(s, te + dt) / p0) * 1e4
            D["30 min before the TWAP"] = sg * np.log(price_at(s, t1 - 1000) / price_at(s, t1 - 1000 - 1_800_000)) * 1e4
            ta = d0 + rng.integers(1_860_000, 86_400_000 - 3_600_000, size=len(G))
            pa = price_at(s, ta); pm = np.sign(np.log(pa / price_at(s, ta - 1_800_000))); sa = rng.choice([-1, 1], size=len(G))
            for h, dt in HORIZONS.items():
                r = np.log(price_at(s, ta + dt) / pa) * 1e4
                D[f"random {h}"] = sa * r; D[f"momentum {h}"] = np.where(pm != 0, pm * r, np.nan)
            L.append(D)
    X = pd.concat(L, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    num = [c for c in X.columns if any(c.endswith(h) for h in HORIZONS) or c == "30 min before the TWAP"]; X[num] = X[num].clip(-1000, 1000)
    X.to_parquet(WORK / "controls.parquet", index=False)
    show()

    def summary(g):
        return pd.Series({"n": len(g), "1st + 1 s -> 30 min": g["1st slice 30 min"].mean(), "2nd + 1 s -> 30 min": g["2nd slice 30 min"].mean(),
                          "t (2nd, 30 min)": t_day(g, "2nd slice 30 min"), "random 30 min": g["random 30 min"].mean(),
                          "momentum 30 min": g["momentum 30 min"].mean(), "above momentum": g["2nd slice 30 min"].mean() - g["momentum 30 min"].mean(),
                          "30 min before the TWAP": g["30 min before the TWAP"].mean()})
    print("By reference class (every TWAP with >= 2 slices):\n", X.groupby("class").apply(summary, include_groups=False).to_string())
    print("\nSame as s04a (TWAPs >= 100,000 dollars, size known after the fact):\n", X[X.usd >= LARGE].groupby("class").apply(summary, include_groups=False).to_string())
    B = X.groupby(["class", "slice size", "past"], observed=True).apply(summary, include_groups=False)
    B["criterion"] = (B["2nd + 1 s -> 30 min"] >= 9) & (B["t (2nd, 30 min)"] >= 3) & (B["random 30 min"].abs() < 2) & (B["above momentum"] >= 9)
    print("\nThe 24 classes (size x past x reference):\n", B.to_string())
    print(f"\nCriterion: {int(B.criterion.sum())} class(es) out of {len(B)} pass")
