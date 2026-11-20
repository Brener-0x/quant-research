"""Section 7. Do several same-side TWAPs pushing together raise the gain?

Pre-registered. Base: the set of s07c after R1a, R1b and R2. Cumulative pressure % = (first slice + other active
same-side TWAPs - opposite side) / global volume (Hyperliquid + Binance perp + spot) per 30 s. Herd = other wallets on
the same side / my first slice. Thresholds chosen on the selection months, checked on the verification months.
    python s07d_cumulative_pressure.py   -> work/core_set.parquet (the set used from Section 8 on)
"""
import numpy as np, pandas as pd
from common import WORK, show, t_day
from s07c_reasons import BACKED, P, SELECT, VERIFY, test_reason

R1B_THRESHOLD = 182544674.72                       # frozen in s07c (Binance volume per day, dollars)
GRID_Q = [10, 20, 30, 40, 50]; GRID_X = [0, 0.25, 0.5, 1, 2]

if __name__ == "__main__":
    M = pd.read_parquet(WORK / "reasons.parquet")
    M = M[(M["widened weight %"] >= 5) & (M["Binance volume per day"] < R1B_THRESHOLD) & ~M.coin.isin(BACKED)].reset_index(drop=True)
    T = pd.read_parquet(WORK / "twaps.parquet", columns=["user", "twapId", "coin", "start", "end", "usd", "slices", "net_side"])
    T["flow"] = T.usd / T.slices.clip(lower=1)
    cols = ["same side, same wallet", "same side, other wallets", "opposite, same wallet", "opposite, other wallets"]
    for c in cols:
        M[c] = 0.0
    for coin, idx in M.groupby("coin").indices.items():
        G = T[T.coin == coin]; gd, ge, fl, sn, gid, gu = (G.start.to_numpy().astype(np.int64), G.end.to_numpy().astype(np.int64), G.flow.to_numpy(),
                                                          G.net_side.to_numpy(), G.twapId.to_numpy(), G.user.to_numpy())
        out = np.zeros((len(idx), 4))
        for j, k in enumerate(idx):
            t1, own, u, sg = int(M.t1.iat[k]), M.twapId.iat[k], M.user.iat[k], M.net_side.iat[k]
            a = (gd < t1) & (ge > t1) & (gid != own); mw = gu == u; ms = sn == sg
            out[j] = [fl[a & ms & mw].sum(), fl[a & ms & ~mw].sum(), fl[a & ~ms & mw].sum(), fl[a & ~ms & ~mw].sum()]
        for i, c in enumerate(cols):
            M.iloc[idx, M.columns.get_loc(c)] = out[:, i]
    v30 = (M["HL volume 1h"] + M["Binance volume 1h"] + M["spot volume 1h"]) / 120
    M["cumulative pressure %"] = 100 * (M.usd1 + M[cols[0]] + M[cols[1]] - M[cols[2]] - M[cols[3]]) / v30
    M["herd"] = M[cols[1]] / M.usd1
    M.to_parquet(WORK / "core_set.parquet", index=False)
    show()
    print(f"Core set: {len(M):,} TWAPs; {M[P].mean():.2f} bp (t {t_day(M, P):.2f})")
    Ch = M[M.month.isin(SELECT)]
    _, sp, okp = test_reason(M, "cumulative pressure %", "Cumulative pressure: the weakest are dropped",
                             [(f"{q} %", Ch["cumulative pressure %"].quantile(q / 100)) for q in GRID_Q], lambda v, s: v >= s)
    _, st, okt = test_reason(M, "herd", "Herd (other same-side wallets / my first slice): kept if > x",
                             [(f"> {x:g}", x) for x in GRID_X], lambda v, s: v > s)
    M["quintile"] = pd.qcut(M["cumulative pressure %"], 5, labels=["Q1 (weak)", "Q2", "Q3", "Q4", "Q5 (strong)"])
    for name, sel in (("selection", SELECT), ("verification", VERIFY)):
        g = M[M.month.isin(sel)]
        print(f"\nBy quintile of cumulative pressure ({name}):\n", g.groupby("quintile", observed=True).agg(n=(P, "size"), net=(P, "mean"),
              t=(P, lambda x: t_day(g.loc[x.index], P))).to_string())
    print(f"\nShare of TWAPs where the cumulative pressure equals the own weight: {(M[cols].sum(axis=1) == 0).mean():.1%}")
