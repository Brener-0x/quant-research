"""Section 5. Does the weight estimated at entry isolate profitable TWAPs, beyond momentum?

Pre-registered: TWAPs of s04b (>= 2 slices); non-TWAP prices of s04a; coin volume from the 5-second bars.
Weight = mean of the first two slices / the coin's volume per 30 s over the previous hour. Momentum control matched on
the coin and on the size of the move over the previous 30 minutes. Criterion (external reference): a weight class
>= 10 bp at 30 min, surplus over the control >= 10 bp, daily t >= 3 on the surplus.
    python s05a_weight.py   -> work/weight.parquet, work/momentum_controls.parquet
"""
import numpy as np, pandas as pd
from common import DAYS, WORK, price_at, series2, show, t_day

WEIGHTS = [0, 1, 5, 20, np.inf]; WEIGHT_NAMES = ["< 1 %", "1-5 %", "5-20 %", ">= 20 %"]
MOVES = [0, 5, 10, 20, 40, np.inf]; MOVE_NAMES = ["< 5", "5-10", "10-20", "20-40", ">= 40"]
D30, N_CONTROL = 1_800_000, 40
rng = np.random.default_rng(37)


def vol_per_30s(day, X, idx):
    """Coin volume per 30 s over the hour before t1 (5-second bars, previous listed day included)."""
    B = pd.read_parquet(WORK / "bars5s" / f"{day}.parquet", columns=["coin", "t5", "notional"]); B["coin"] = B.coin.astype(str)
    i_ = DAYS.index(day)
    if i_ > 0:
        Bp = pd.read_parquet(WORK / "bars5s" / f"{DAYS[i_ - 1]}.parquet", columns=["coin", "t5", "notional"]); Bp["coin"] = Bp.coin.astype(str); B = pd.concat([Bp, B])
    G = {c: (g.t5.to_numpy(), np.concatenate([[0], np.cumsum(g.notional.to_numpy())])) for c, g in B.sort_values("t5").groupby("coin")}
    out = {}
    for c, sub in X.iloc[idx].groupby("coin"):
        g = G.get(c)
        if g is None:
            continue
        t1 = sub.t1.to_numpy()
        a, b = np.searchsorted(g[0], t1 - 3_600_000, side="left"), np.searchsorted(g[0], t1, side="left")
        out.update(zip(sub.index.to_numpy(), (g[1][b] - g[1][a]) / 120))
    return out


if __name__ == "__main__":
    X = pd.read_parquet(WORK / "controls.parquet", columns=["user", "twapId", "coin", "day", "class", "past", "slice size", "usd", "t2",
                                                            "2nd slice 30 min", "1st slice 30 min", "30 min before the TWAP"])
    P = pd.read_parquet(WORK / "first_slices.parquet"); X = X.merge(P[["user", "twapId", "t1", "usd1", "usd2"]], on=["user", "twapId"])
    X["slice"] = (X.usd1 + X.usd2) / 2
    vol, CTRL = np.full(len(X), np.nan), []
    for d in DAYS:
        idx = np.flatnonzero(X.day.to_numpy() == d)
        if len(idx) == 0:
            continue
        for i, v in vol_per_30s(d, X, idx).items():
            vol[i] = v
        S = series2(d); d0 = int(pd.Timestamp(d).value // 1_000_000)
        for c in X.iloc[idx].coin.unique():
            s = S.get(c)
            if s is None:
                continue
            ta = d0 + rng.integers(D30 + 60_000, 86_400_000 - D30, size=N_CONTROL); p0 = price_at(s, ta)
            m = np.log(p0 / price_at(s, ta - D30)) * 1e4; post = np.log(price_at(s, ta + D30) / p0) * 1e4
            CTRL.append(pd.DataFrame({"coin": c, "day": d, "m": m, "post": post}))
    X["vol30"] = vol; X["weight %"] = 100 * X.slice / X.vol30
    X["weight class"] = pd.cut(X["weight %"], WEIGHTS, labels=WEIGHT_NAMES, right=False)
    W = pd.concat(CTRL, ignore_index=True).replace([np.inf, -np.inf], np.nan).dropna(); W = W[(W.m.abs() < 1000) & (W.post.abs() < 1000)]
    W["move"] = pd.cut(W.m.abs(), MOVES, labels=MOVE_NAMES, right=False); W["mom"] = W.post * np.sign(W.m)
    W.to_parquet(WORK / "momentum_controls.parquet", index=False)
    W["class"] = W.coin.map(X.drop_duplicates("coin").set_index("coin")["class"])
    Ec = W.groupby(["coin", "move"], observed=True).mom.agg(["mean", "size"]); Ek = W.groupby(["class", "move"], observed=True).mom.mean()
    X["move"] = pd.cut(X["30 min before the TWAP"].abs(), MOVES, labels=MOVE_NAMES, right=False)
    ec = Ec.reindex(pd.MultiIndex.from_arrays([X.coin, X.move])); ek = Ek.reindex(pd.MultiIndex.from_arrays([X["class"], X.move])).to_numpy()
    X["expected (matched momentum)"] = np.where(ec["size"].to_numpy() >= 30, ec["mean"].to_numpy(), ek) * np.sign(X["30 min before the TWAP"].to_numpy())
    for e in ("2nd slice", "1st slice"):
        X[f"surplus {e}"] = X[f"{e} 30 min"] - X["expected (matched momentum)"]
    X.to_parquet(WORK / "weight.parquet", index=False)
    show()
    print("Momentum control (random times): continuation at 30 min in the direction of the previous move (bp)\n",
          W.groupby(["class", "move"], observed=True).mom.agg(["size", "mean", "median"]).to_string())

    def summary(g):
        return pd.Series({"n": len(g), "2nd + 1 s -> 30 min": g["2nd slice 30 min"].mean(), "t": t_day(g, "2nd slice 30 min"),
                          "expected momentum": g["expected (matched momentum)"].mean(), "surplus (2nd)": g["surplus 2nd slice"].mean(),
                          "t surplus": t_day(g, "surplus 2nd slice"), "1st + 1 s -> 30 min": g["1st slice 30 min"].mean(),
                          "surplus (1st)": g["surplus 1st slice"].mean()})
    print("\nBy class and weight at entry:\n", X.groupby(["class", "weight class"], observed=True).apply(summary, include_groups=False).to_string())
    B = X[X["class"] == "external reference"].groupby("weight class", observed=True).apply(summary, include_groups=False)
    ok = B[(B["2nd + 1 s -> 30 min"] >= 10) & (B["surplus (2nd)"] >= 10) & (B["t surplus"] >= 3)]
    print(f"\nCriterion (external reference): {len(ok)} weight class(es) out of {len(B)} pass")
