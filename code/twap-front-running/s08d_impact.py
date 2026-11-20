"""Section 8. My own impact and the size. Entry E2, exit A without look-ahead (B as fallback).

There is no historical order book, so I measure what a taker order of IMPOSED size pays: a TWAP slice (the algorithm
does not size its slice to the book of the moment, unlike other takers).
1. Curve: for every slice (one taker oid) on the coins of s08b, slippage beyond the best level = side x ln(average price
   paid / best level touched) in bp; x = slice dollars / Hyperliquid volume per 30 s over the previous hour. The mean
   per class of x is the expected cost of an order of that relative size; one curve per liquidity tercile.
2. Model: each signal pays that cost on entry and exit, x = my size / the signal's volume per 30 s. Sizing rules:
   fixed dollars or a share of the announced size, capped at x <= 1, 3 or 10.
    python s08d_impact.py   -> work/slippage.parquet, work/sizes.parquet
"""
import numpy as np, pandas as pd
from common import DAYS, WORK, show
from s07c_reasons import VERIFY
from s08a_entries import day_before

X_EDGES = [0, 0.03, 0.1, 0.3, 1, 3, 10, 30, np.inf]
FIXED = [250, 500, 1000, 2000, 5000, 10_000, 20_000]
CAPS = [1, 3, 10]
SHARES = [0.01, 0.02, 0.05, 0.1, 0.25]
EXITS = {"A without look-ahead": "A2 +2000", "B": "B +2000"}
TERCILES = ["illiquid", "mid", "liquid"]


def slippages():
    f = WORK / "slippage.parquet"
    if f.exists():
        return pd.read_parquet(f)
    CO = set(pd.read_parquet(WORK / "e2.parquet", columns=["coin"]).coin)
    L = []
    for d in DAYS:
        F = pd.read_parquet(WORK / "twap_fills" / f"{d}.parquet", columns=["user", "coin", "oid", "t", "px", "sz", "side", "usd", "taker"])
        F = F[F.taker & F.coin.isin(CO)]
        O = F.groupby(["user", "coin", "oid"], sort=False).agg(t=("t", "min"), side=("side", "first"), usd=("usd", "sum"), sz=("sz", "sum"),
                                                                pmin=("px", "min"), pmax=("px", "max"), levels=("px", "nunique")).reset_index()
        best = np.where(O.side > 0, O.pmin, O.pmax); vwap = O.usd / O.sz
        O["slippage"] = O.side * np.log(vwap / best) * 1e4
        B = pd.read_parquet(WORK / "bars5s" / f"{d}.parquet", columns=["coin", "t5", "notional"]); B["coin"] = B.coin.astype(str)
        v = day_before(d)
        if v is not None:
            Bp = pd.read_parquet(WORK / "bars5s" / f"{v}.parquet", columns=["coin", "t5", "notional"]); Bp["coin"] = Bp.coin.astype(str); B = pd.concat([Bp, B])
        G = {c: (g.t5.to_numpy(), np.concatenate([[0], np.cumsum(g.notional.to_numpy())])) for c, g in B.sort_values("t5").groupby("coin")}
        O["vol30"] = np.nan
        for c, idx in O.groupby("coin").indices.items():
            g = G.get(c)
            if g is None:
                continue
            t1 = O.t.to_numpy()[idx]; a, b = np.searchsorted(g[0], t1 - 3_600_000, side="left"), np.searchsorted(g[0], t1, side="left")
            O.iloc[idx, O.columns.get_loc("vol30")] = (g[1][b] - g[1][a]) / 120
        O["day"] = d
        L.append(O[["coin", "t", "usd", "levels", "slippage", "vol30", "day"]].astype({"usd": "float32", "slippage": "float32", "vol30": "float32", "levels": "int16"}))
    D = pd.concat(L, ignore_index=True); D.to_parquet(f, index=False); return D


def curve(D):
    D = D[(D.vol30 > 0) & D.slippage.notna()].copy(); D["x"] = D.usd / D.vol30
    D["class"] = pd.cut(D.x, X_EDGES, right=False)
    Z = D.groupby("class", observed=True).agg(n=("x", "size"), x_med=("x", "median"), mean=("slippage", "mean"), median=("slippage", "median"))
    return D, Z


def cost(x, Z):
    """Expected cost (bp) of a taker order of relative size x: linear interpolation in log x of the class means;
    beyond the last class, the last mean."""
    lx = np.log(np.clip(Z.x_med.to_numpy(), 1e-6, None)); y = Z["mean"].to_numpy()
    return np.interp(np.log(np.clip(x, 1e-6, None)), lx, y)


def tercile_curves(D, edges):
    """One curve per liquidity tercile (Hyperliquid volume per 30 s): classes of x with at least 200 slices."""
    D = D.assign(tercile=np.digitize(D.vol30, edges))
    Zs = {}
    for t_, g in D.groupby("tercile"):
        z = g.groupby("class", observed=True).agg(n=("x", "size"), x_med=("x", "median"), mean=("slippage", "mean"))
        Zs[t_] = z[z.n >= 200]
    return Zs


def tercile_cost(x, v30, Zs, edges):
    t_ = np.digitize(v30, edges); c = np.zeros(len(x)); xmax = np.zeros(len(x))
    for k, z in Zs.items():
        m = t_ == k
        c[m] = cost(x[m], z); xmax[m] = z.x_med.max()
    return c, xmax


def frozen_curves():
    """The impact curves frozen on the 150 in-sample days (used unchanged out of sample)."""
    D, _ = curve(slippages()); edges = D.vol30.quantile([1 / 3, 2 / 3]).to_numpy(); return tercile_curves(D, edges), edges


if __name__ == "__main__":
    D, Z = curve(slippages())
    show()
    print(f"TWAP slices (taker orders of imposed size), coins of s08b, 150 days: {len(D):,}")
    edges = D.vol30.quantile([1 / 3, 2 / 3]).to_numpy()
    D["liquidity tercile"] = pd.Categorical.from_codes(np.digitize(D.vol30, edges), TERCILES)
    print(f"Slippage beyond the best level (bp per side) by x and liquidity tercile (volume per 30 s < {edges[0]:,.0f}, {edges[0]:,.0f}-{edges[1]:,.0f}, > {edges[1]:,.0f} dollars):\n",
          D.pivot_table(index="class", columns="liquidity tercile", values="slippage", aggfunc="mean", observed=True).to_string())
    Zs = tercile_curves(D, edges)
    K = pd.read_parquet(WORK / "e2.parquet")
    K = K.merge(pd.read_parquet(WORK / "announcements.parquet", columns=["user", "twapId", "announced usd"]), on=["user", "twapId"], how="left")
    K["t entry"] = K.t1 + 2000
    K["t exit"] = np.maximum(np.where(K["t exit A2"].notna(), K["t exit A2"], K.end + 35_000), K["t entry"])
    v30 = K.vol30.to_numpy().astype(float)
    order = np.argsort(np.concatenate([K["t entry"].to_numpy(), K["t exit"].to_numpy()]), kind="stable")
    rules = [("fixed", f"{n:,} $", np.full(len(K), float(n))) for n in FIXED] + [("announced", f"{p:.0%} of announced", p * K["announced usd"].fillna(0).to_numpy()) for p in SHARES]
    rows = []
    for fam, name, N0 in rules:
        for xm in CAPS:
            N = np.minimum(N0, xm * v30); x = np.where(v30 > 0, N / np.where(v30 > 0, v30, 1), 0); c, _ = tercile_cost(x, v30, Zs, edges)
            for so, col in EXITS.items():
                net = K[col].to_numpy() - 2 * c; m = (N > 0) & ~np.isnan(net)
                g = np.where(m, N * net / 1e4, 0.0)
                day = pd.Series(g).groupby(K.day.to_numpy()).sum().reindex(sorted(K.day.unique()), fill_value=0)
                cum = day.cumsum(); sgn = np.concatenate([np.where(m, N, 0), -np.where(m, N, 0)])[order]
                top = np.sort(g[m])[::-1]; k1 = max(1, int(m.sum() * 0.01))
                rows.append({"family": fam, "rule": name, "cap x": xm, "exit": so, "positions": int(m.sum()), "mean size $": N[m].mean(),
                             "round-trip cost bp": 2 * np.average(c[m], weights=N[m]), "net bp (size-weighted)": np.average(net[m], weights=N[m]),
                             "profit per day $": day.mean(), "verification per day $": day[[d_[:6] in VERIFY for d_ in day.index]].mean(),
                             "days > 0 %": 100 * (day > 0).mean(), "largest drawdown $": (cum - cum.cummax()).min(), "max capital $": np.cumsum(sgn).max(),
                             "share of top 1 % positions": top[:k1].sum() / g[m].sum() if g[m].sum() > 0 else np.nan,
                             "daily t": day.mean() / day.std() * np.sqrt(len(day))})
    R = pd.DataFrame(rows); R.to_parquet(WORK / "sizes.parquet", index=False)
    print("\nExit A without look-ahead, cap x <= 3:\n", R[(R.exit == "A without look-ahead") & (R["cap x"] == 3)].drop(columns=["exit", "cap x", "family"]).to_string(index=False))
