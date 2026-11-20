"""Section 8. Wake, first version: the frozen model, position by position, over the in-sample days (exploratory).

Signal: E2 without look-ahead (frozen rules at the first slice, the API confirms, entry + 2 s). Exit: A without
look-ahead (announced end, or cancellation + 2 s; else B), B as fallback. Size: fixed N, capped at 3 x the Hyperliquid
volume per 30 s over the previous hour. Costs: 4.5 bp fee per side (already in s08b) + slippage by liquidity tercile
(s08d), on entry and exit. No stop.
    python s08g_wake_v0.py   -> work/wake_v0_positions.parquet
"""
import numpy as np, pandas as pd
from common import WORK, show
from s08d_impact import frozen_curves, tercile_cost

NS = [1000, 2000, 5000]; CAP = 3
EXITS = {"A without look-ahead": "A2 +2000", "B": "B +2000"}
BOUND = 2000          # s08a bounds each result at +/- 2,000 bp

if __name__ == "__main__":
    Zs, edges = frozen_curves()
    K = pd.read_parquet(WORK / "e2.parquet")
    K["t entry"] = K.t1 + 2000
    K["t exit"] = np.maximum(np.where(K["t exit A2"].notna(), K["t exit A2"], K.end + 35_000), K["t entry"])
    v30 = K.vol30.to_numpy().astype(float)
    L = []
    for N in NS:
        n = np.minimum(N, CAP * v30); x = n / np.where(v30 > 0, v30, 1); c, _ = tercile_cost(x, v30, Zs, edges)
        for so, col in EXITS.items():
            gross = K[col].to_numpy()
            L.append(pd.DataFrame({"N": N, "exit": so, "user": K.user, "twapId": K.twapId, "coin": K.coin, "day": K.day, "month": K.month,
                                   "t entry": K["t entry"], "t exit": K["t exit"], "size $": n, "net of fees bp": gross, "slippage bp": 2 * c,
                                   "net bp": gross - 2 * c, "bounded": np.abs(gross) >= BOUND, "pnl $": n * (gross - 2 * c) / 1e4}))
    P = pd.concat(L, ignore_index=True); P = P[P["net bp"].notna() & (P["size $"] > 0)]
    P["minutes"] = (P["t exit"] - P["t entry"]) / 60_000
    P.to_parquet(WORK / "wake_v0_positions.parquet", index=False)
    show()
    days = sorted(K.day.unique())
    rows = []
    for (N, so), g in P.groupby(["N", "exit"], sort=False):
        d = g.groupby("day")["pnl $"].sum().reindex(days, fill_value=0); cum = d.cumsum(); dd = cum - cum.cummax()
        gp, gm = g[g["pnl $"] > 0]["pnl $"], g[g["pnl $"] <= 0]["pnl $"]
        rows.append({"N": N, "exit": so, "positions": len(g), "total $": g["pnl $"].sum(), "per day $": d.mean(), "winners %": 100 * (g["pnl $"] > 0).mean(),
                     "mean win $": gp.mean(), "mean loss $": gm.mean(), "days > 0 %": 100 * (d > 0).mean(), "largest drawdown $": dd.min(),
                     "worst position bp": g["net bp"].min(), "daily t": d.mean() / d.std() * np.sqrt(len(d))})
    print(f"Wake v0, {len(days)} tradable days of the 150 in-sample days (the 7-day wallet count needs a week of history after each gap):\n"
          + pd.DataFrame(rows).to_string(index=False))
    g = P[(P.N == 2000) & (P.exit == "A without look-ahead")]
    print(f"\nShare of profit from the 10 best positions: {g.nlargest(10, 'pnl $')['pnl $'].sum() / g['pnl $'].sum():.1%}")
    print("10 worst positions:\n" + g.nsmallest(10, "pnl $")[["day", "coin", "size $", "minutes", "net bp", "pnl $"]].to_string(index=False))
