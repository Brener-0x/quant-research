"""Section 7. Do TWAPs of wallets that launch many of them push the price less?

Pre-registered. Frequency = TWAPs launched by the wallet (all coins) in the 7 days before the start; only TWAPs with
7 days of data before them enter the criterion. Main gain: s06b (taker at the end); s06d alongside.
Criterion (perps, weight >= 20 %): for S = 10, 30, 100, kept (<= S) net > all, t >= 3, kept minus dropped t >= 3,
kept > 0 without the top 3 coins.
    python s07a_wallet_frequency.py   -> work/wallet_frequency.parquet
"""
import numpy as np, pandas as pd
from common import DAYS, WORK, show, t_day

WEEK = 7 * 86_400_000
EDGES = [-1, 0, 3, 10, 30, 100, 300, np.inf]; NAMES = ["0", "1-3", "4-10", "11-30", "31-100", "101-300", "> 300"]
THRESHOLDS = [3, 10, 30, 100, 300]; PLATEAU = [10, 30, 100]


def t_diff(a, b, c):
    """Daily t of the difference of daily means (a - b)."""
    d = (a.groupby("day")[c].mean() - b.groupby("day")[c].mean()).dropna()
    return d.mean() / d.std(ddof=1) * np.sqrt(len(d)) if len(d) > 2 else np.nan


if __name__ == "__main__":
    T = pd.read_parquet(WORK / "twaps.parquet", columns=["user", "twapId", "start"]).sort_values(["user", "start"])
    M = pd.read_parquet(WORK / "announcement_entry.parquet")
    M = M[M["net announcement"].notna() & M["net s06b"].notna()].merge(T, on=["user", "twapId"], how="left")
    freq = np.full(len(M), -1, dtype=np.int64)
    pos = {u: g.start.to_numpy() for u, g in T.groupby("user", sort=False)}
    for u, idx in M.groupby("user").indices.items():
        d = pos[u]; t0 = M.start.to_numpy()[idx].astype(np.int64)
        freq[idx] = np.searchsorted(d, t0, side="left") - np.searchsorted(d, t0 - WEEK, side="left")
    M["frequency 7d"] = freq
    ds = set(DAYS)
    M["full history"] = [all((pd.Timestamp(d) - pd.Timedelta(days=k)).strftime("%Y%m%d") in ds for k in range(1, 8)) for d in M.day]
    M["frequency class"] = pd.cut(M["frequency 7d"], EDGES, labels=NAMES)
    M.to_parquet(WORK / "wallet_frequency.parquet", index=False)
    show()
    P = M[(M.type == "perp") & (M["weight class"] == ">= 20 %")]; g = P[P["full history"]]
    print(f"Perps, weight >= 20 %: {len(P):,} TWAPs; with 7 days of history: {len(g):,}")
    print("\nBy frequency class:\n", g.groupby("frequency class", observed=True).agg(n=("twapId", "size"), net=("net s06b", "mean"),
          t=("net s06b", lambda x: t_day(g.loc[x.index], "net s06b"))).to_string())
    rows = []
    for S in THRESHOLDS:
        k, e = g[g["frequency 7d"] <= S], g[g["frequency 7d"] > S]
        pc = k.groupby("coin")["net s06b"].sum().sort_values(ascending=False)
        rows.append({"S": S, "kept": len(k), "net kept": k["net s06b"].mean(), "t kept": t_day(k, "net s06b"), "net dropped": e["net s06b"].mean(),
                     "t diff": t_diff(k, e, "net s06b"), "kept without top 3": k[~k.coin.isin(pc.index[:3])]["net s06b"].mean()})
    R = pd.DataFrame(rows).set_index("S")
    print(f"\nAll (full history): {g['net s06b'].mean():.2f} (t {t_day(g, 'net s06b'):.2f})\nBy threshold S (kept = frequency <= S):\n", R.to_string())
    print("\nSame duration, net by frequency:\n", g.pivot_table(index="duration class", columns="frequency class", values="net s06b", aggfunc="mean", observed=True).to_string())
    ok = all(R.loc[S, "net kept"] > g["net s06b"].mean() and R.loc[S, "t kept"] >= 3 and R.loc[S, "t diff"] >= 3 and R.loc[S, "kept without top 3"] > 0 for S in PLATEAU)
    print("\nCriterion (S = 10, 30, 100): " + ("passed" if ok else "failed"))
