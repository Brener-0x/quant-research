"""Section 10. Why does the edge of the frozen signal fall month after month? Observation only, nothing is tuned.

Population: E2 signals of the ten months (in sample: work/wake_paths; out of sample: work/oos/paths), exit A without
look-ahead, entry at the first slice + 2 s. Four hypotheses measured month by month:
  H1 competition: part of the move taken BEFORE my entry (last price before the first slice -> my entry price); same-side
     taker volume of other wallets (TWAP slices excluded) within 2 s and from 2 to 30 s after the first slice, over the
     volume per 30 s; recurring "follower" wallets;
  H2 absorption: total move (last price before -> exit) by class of weight;
  H3 activity: |move| over the 30 minutes before the first slice (unsigned); edge relative to that activity;
  H4 composition: weight, duration, size, share of cancelled TWAPs.
    python s10a_decay.py   -> work/decay.parquet, work/followers.parquet
"""
import numpy as np, pandas as pd
from common import TAKER_FEE, WORK, show, t_day

SOURCES = {"in sample": (WORK / "wake_paths.parquet", WORK / "orders", WORK / "twap_fills"),
           "out of sample": (WORK / "oos" / "paths.parquet", WORK / "oos" / "orders", WORK / "oos" / "twap_fills")}


def signal_id(period, user, twap_id):
    return period[:1] + user.str[-8:] + twap_id.astype(str)


def measure():
    f, fs = WORK / "decay.parquet", WORK / "followers.parquet"
    if f.exists() and fs.exists():
        return pd.read_parquet(f), pd.read_parquet(fs)
    L, S = [], []
    for per, (fc, dord, dtw) in SOURCES.items():
        R = pd.read_parquet(fc); R = R[R["px entry"].notna() & R["A2 +2000"].notna()].copy()
        R["sid"] = signal_id(per, R.user, R.twapId)
        for d in sorted(R.day.unique()):
            Sd = R[R.day == d]
            nx = (pd.Timestamp(d) + pd.Timedelta(days=1)).strftime("%Y%m%d"); pv = (pd.Timestamp(d) - pd.Timedelta(days=1)).strftime("%Y%m%d")
            fl = [x for x in (pv, d, nx) if (dord / f"{x}.parquet").exists()]
            O = pd.concat([pd.read_parquet(dord / f"{x}.parquet", columns=["user", "coin", "oid", "t", "side", "usd", "px"]) for x in fl])
            O["coin"] = O.coin.astype(str); O["user"] = O.user.astype(str); O = O[O.coin.isin(set(Sd.coin))]
            tw = set(pd.concat([pd.read_parquet(dtw / f"{x}.parquet", columns=["oid"]) for x in fl if (dtw / f"{x}.parquet").exists()]).oid)
            O["twap"] = O.oid.isin(tw)
            for coin, G in Sd.groupby("coin"):
                Oc = O[O.coin == coin].sort_values("t"); t, px, se, us, tv = Oc.t.to_numpy(), Oc.px.to_numpy(), Oc.side.to_numpy(), Oc.usd.to_numpy(), Oc.twap.to_numpy()
                uu = Oc.user.to_numpy()
                for r in G.to_dict("records"):
                    sg, t1, pe, v30 = int(r["side"]), int(r["t1"]), float(r["px entry"]), float(r["vol30"])
                    i0 = np.searchsorted(t, t1, side="left") - 1
                    p_bef = px[i0] if i0 >= 0 and t1 - t[i0] <= 300_000 else np.nan
                    i30 = np.searchsorted(t, t1 - 1_800_000, side="left") - 1
                    p_30 = px[i30] if i30 >= 0 and t1 - 1_800_000 - t[i30] <= 300_000 else np.nan
                    net = float(r["A2 +2000"]); p_exit = pe * np.exp(sg * (net + 2 * TAKER_FEE) / 1e4)
                    a, b, c = np.searchsorted(t, t1, side="right"), np.searchsorted(t, t1 + 2000, side="right"), np.searchsorted(t, t1 + 30_000, side="right")
                    others = (uu != r["user"]) & ~tv
                    m2 = np.zeros(len(t), bool); m2[a:b] = True; m2 &= others & (se == sg)
                    m30 = np.zeros(len(t), bool); m30[b:c] = True; m30 &= others & (se == sg)
                    o2 = np.zeros(len(t), bool); o2[a:b] = True; o2 &= others & (se == -sg)
                    canc = r.get("cancelled"); canc = bool(canc) if canc == canc and canc is not None else False
                    L.append({"sid": r["sid"], "period": per, "day": d, "month": d[:6], "coin": coin, "side": sg, "net bp": net,
                              "before me bp": sg * np.log(pe / p_bef) * 1e4 if p_bef == p_bef else np.nan,
                              "total bp": sg * np.log(p_exit / p_bef) * 1e4 if p_bef == p_bef else np.nan,
                              "activity 30 min bp": abs(np.log(p_bef / p_30)) * 1e4 if p_bef == p_bef and p_30 == p_30 else np.nan,
                              "followers 2 s / vol30": us[m2].sum() / v30 if v30 > 0 else np.nan, "followers 2-30 s / vol30": us[m30].sum() / v30 if v30 > 0 else np.nan,
                              "against 2 s / vol30": us[o2].sum() / v30 if v30 > 0 else np.nan, "followed within 2 s": bool(m2.any()),
                              "widened weight %": r["widened weight %"], "vol30": v30, "usd1": r["usd1"], "minutes": (r["end"] - t1) / 60_000, "cancelled": canc})
                    for w in set(uu[m2]):
                        S.append({"sid": r["sid"], "month": d[:6], "wallet": w, "usd": us[m2 & (uu == w)].sum()})
    M = pd.DataFrame(L).replace([np.inf, -np.inf], np.nan); Sv = pd.DataFrame(S)
    M.to_parquet(f, index=False); Sv.to_parquet(fs, index=False); return M, Sv


def top_followers(Sv, n=20):
    return Sv.groupby("wallet").sid.nunique().sort_values(ascending=False).head(n).index


if __name__ == "__main__":
    M, Sv = measure()
    show()
    g = M.groupby("month")
    T = pd.DataFrame({"n": g.size(), "net edge bp": g["net bp"].mean(), "daily t": g.apply(lambda d: t_day(d, "net bp"), include_groups=False),
                      "total move bp": g["total bp"].mean(), "taken before me bp": g["before me bp"].mean(),
                      "followed within 2 s %": 100 * g["followed within 2 s"].mean(), "followers 2 s (mean)": g["followers 2 s / vol30"].mean(),
                      "activity 30 min (median)": g["activity 30 min bp"].median(), "edge / activity": g["net bp"].mean() / g["activity 30 min bp"].median(),
                      "widened weight (median)": g["widened weight %"].median(), "minutes (median)": g["minutes"].median(), "cancelled %": 100 * g["cancelled"].mean()})
    print(f"{len(M):,} signals over ten months\n" + T.T.to_string())
    M["weight quartile"] = pd.qcut(M["widened weight %"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    print("\nTotal move (bp) by quartile of widened weight (H2):\n" + M.pivot_table(index="month", columns="weight quartile", values="total bp", aggfunc="mean", observed=True).to_string())
    ns = M.groupby("month").size(); top = top_followers(Sv)
    first = Sv.groupby("wallet").month.min()
    print("\nFirst month of the 20 most frequent followers:", first.reindex(top).value_counts().sort_index().to_dict())
    print("Share of signals touched by one of them within 2 s (%):", (Sv[Sv.wallet.isin(top)].groupby("month").sid.nunique() / ns * 100).round(1).to_dict())
    print("\nSpeed ceiling (entering at the last price before the first slice, net of fees):", (g["total bp"].mean() - 2 * TAKER_FEE).round(2).to_dict())
    print("\nNet edge (bp) with and without a follower within 2 s:\n" + M.pivot_table(index="followed within 2 s", columns="month", values="net bp", aggfunc="mean").to_string())
    # 95 % confidence interval of the monthly edge (Figure 9): standard error = edge / daily t
    ci = (T["net edge bp"] - 1.96 * T["net edge bp"] / T["daily t"], T["net edge bp"] + 1.96 * T["net edge bp"] / T["daily t"])
    print("\n95 % CI of the monthly edge:", {m: (round(a, 1), round(b, 1)) for m, a, b in zip(T.index, *ci)})
