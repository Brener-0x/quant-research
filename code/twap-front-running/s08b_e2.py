"""Section 8. E2 without look-ahead.

The core set (5,077 TWAPs) still holds filters unknown at entry: at least two slices (s04b; a TWAP cancelled after its
first slice drops out), an announcement found in the API after the fact (s06d), an exit price available (s06b).
Here: EVERY new TWAP (s03 table, first slice of s04b) that passes the frozen rules at the time of its first slice, with
the exact 7-day count (what the API returns), whatever happens next. Entry + 1 s and + 2 s; exits A (end known, at
the earliest + 2 s) and B (35 s without a slice in the public feed). Announcement fields joined from s06c.
    python s08b_e2.py   -> work/e2.parquet
"""
import numpy as np, pandas as pd
from common import DAYS, TAKER_FEE, WORK, first_after, show, t_day
from s07a_wallet_frequency import WEEK
from s07b_global_weight import KL, symbols
from s07c_reasons import BACKED, SPOT, VERIFY, hour_volume, klines, spot_of, week_volume
from s07d_cumulative_pressure import R1B_THRESHOLD
from s08a_entries import ENTRY_WIN, EXIT_WIN, GAP, day_after, day_before, full_history, public_slices

if __name__ == "__main__":
    T = pd.read_parquet(WORK / "twaps.parquet", columns=["user", "twapId", "coin", "start", "end", "slices", "usd", "net_side"]).sort_values(["user", "start"])
    d = T.start.to_numpy().astype(np.int64); n7 = np.zeros(len(T), dtype=np.int64)
    for u, i in T.groupby("user", sort=False).indices.items():
        x = d[i]; n7[i] = np.searchsorted(x, x, side="left") - np.searchsorted(x, x - WEEK, side="left")
    T["n7"] = n7
    T = T.merge(pd.read_parquet(WORK / "first_slices.parquet", columns=["user", "twapId", "t1", "usd1"]), on=["user", "twapId"])
    SY = symbols(); CO = {c for c, s in SY.items() if c not in BACKED}
    T["day"] = pd.to_datetime(T.t1, unit="ms").dt.strftime("%Y%m%d")
    FH = {d_: full_history(d_) for d_ in DAYS}
    K = T[T.coin.isin(CO) & T.day.map(FH).fillna(False).astype(bool) & (T.n7 <= 30)].copy().reset_index(drop=True)
    K["t1"] = K.t1.astype(np.int64); K["side"] = K.net_side.astype(np.int64); K["vol30"] = np.nan
    for d_ in DAYS:
        idx = np.flatnonzero(K.day.to_numpy() == d_)
        if len(idx) == 0:
            continue
        B = pd.read_parquet(WORK / "bars5s" / f"{d_}.parquet", columns=["coin", "t5", "notional"]); B["coin"] = B.coin.astype(str)
        v = day_before(d_)
        if v is not None:
            Bp = pd.read_parquet(WORK / "bars5s" / f"{v}.parquet", columns=["coin", "t5", "notional"]); Bp["coin"] = Bp.coin.astype(str); B = pd.concat([Bp, B])
        G = {c: (g.t5.to_numpy(), np.concatenate([[0], np.cumsum(g.notional.to_numpy())])) for c, g in B.sort_values("t5").groupby("coin")}
        for c, sub in K.iloc[idx].groupby("coin"):
            g = G.get(c)
            if g is None:
                continue
            t1 = sub.t1.to_numpy(); a, b = np.searchsorted(g[0], t1 - 3_600_000, side="left"), np.searchsorted(g[0], t1, side="left")
            K.loc[sub.index, "vol30"] = (g[1][b] - g[1][a]) / 120
    K["bin1h"] = 0.0; K["spot1h"] = 0.0; K["binday"] = 0.0
    for c, idx in K.groupby("coin").indices.items():
        t1 = K.t1.to_numpy()[idx]; s = SY[c]
        tp, cp = klines(KL, s); ts, cs = klines(SPOT, spot_of(s))
        K.iloc[idx, K.columns.get_loc("bin1h")] = hour_volume(tp, cp, t1); K.iloc[idx, K.columns.get_loc("spot1h")] = hour_volume(ts, cs, t1)
        K.iloc[idx, K.columns.get_loc("binday")] = week_volume(tp, cp, t1) + week_volume(ts, cs, t1)
    K["HL weight %"] = 100 * K.usd1 / K.vol30
    K["widened weight %"] = 100 * K.usd1 / ((K.vol30 * 120 + K.bin1h + K.spot1h) / 120)
    K = K[(K["HL weight %"] >= 20) & (K["widened weight %"] >= 5) & (K.binday < R1B_THRESHOLD)].reset_index(drop=True)
    P = pd.read_parquet(WORK / "core_set.parquet", columns=["user", "twapId"]); P["core"] = True
    K = K.merge(P, on=["user", "twapId"], how="left"); K["core"] = K.core.fillna(False).astype(bool)
    L = []
    for d_ in DAYS:
        Sd = K[K.day == d_]
        if len(Sd) == 0:
            continue
        dd_ = [d_] + ([day_after(d_)] if day_after(d_) else [])
        Od = pd.concat([pd.read_parquet(WORK / "orders" / f"{x}.parquet", columns=["coin", "t", "side", "px"]) for x in dd_]); Od = Od[Od.coin.isin(set(Sd.coin))]
        Tr = pd.concat([public_slices(x) for x in dd_]); Tr = Tr.merge(Sd[["user", "coin", "side"]].drop_duplicates(), on=["user", "coin", "side"])
        TR = {k: np.sort(g.t.to_numpy().astype(np.int64)) for k, g in Tr.groupby(["user", "coin", "side"])}
        for coin, G in Sd.groupby("coin"):
            Oc = Od[Od.coin == coin].sort_values("t"); book = {s: (Oc[Oc.side == s].t.to_numpy(), Oc[Oc.side == s].px.to_numpy()) for s in (1, -1)}
            for r in G.itertuples(index=False):
                sg, t1, end = int(r.side), int(r.t1), int(r.end)
                res = {"user": r.user, "twapId": r.twapId}
                tt = TR.get((r.user, coin, sg), np.array([t1])); tt = tt[tt >= t1]
                k = np.flatnonzero(np.diff(tt) > GAP); end_seen = (tt[k[0]] if len(k) else tt[-1]) + GAP
                _, pA = first_after(*book[-sg], np.array([max(end, t1 + 2000)]), np.array([max(end, t1 + 2000) + EXIT_WIN]))
                _, pB = first_after(*book[-sg], np.array([max(end_seen, t1 + 2000)]), np.array([max(end_seen, t1 + 2000) + EXIT_WIN]))
                for dl in (1000, 2000):
                    _, pe = first_after(*book[sg], np.array([t1 + dl]), np.array([t1 + dl + ENTRY_WIN]))
                    res[f"A +{dl}"] = sg * np.log(pA[0] / pe[0]) * 1e4 - 2 * TAKER_FEE
                    res[f"B +{dl}"] = sg * np.log(pB[0] / pe[0]) * 1e4 - 2 * TAKER_FEE
                L.append(res)
    R = pd.DataFrame(L).replace([np.inf, -np.inf], np.nan)
    num = [c for c in R.columns if c[:3] in ("A +", "B +")]; R[num] = R[num].clip(-2000, 2000)
    K = K.merge(R, on=["user", "twapId"], how="left"); K["month"] = K.day.str[:6]
    # announcement fields (s06c): send time, duration, status; no announcement found -> treated as cancelled (exit B)
    A = pd.read_parquet(WORK / "announcements.parquet", columns=["user", "twapId", "t_sent", "minutes", "status", "cancelled"])
    K = K.merge(A, on=["user", "twapId"], how="left", indicator=True)
    K["found"] = K.pop("_merge") == "both"; K["cancelled2"] = K.cancelled.where(K.found, True).astype(bool)
    for dl in ("1000", "2000"):
        K[f"Ah +{dl}"] = np.where(K.cancelled2, K[f"B +{dl}"], K[f"A +{dl}"])
    K.to_parquet(WORK / "e2.parquet", index=False)
    show()
    K["class"] = np.select([K.core, K.slices < 2], ["core set", "outside: a single slice (cancelled)"], "outside: other")
    rows = []
    for cl, g in list(K.groupby("class")) + [("ALL (E2 without look-ahead)", K)]:
        rows.append({"class": cl, "n": len(g), "A +2000": g["A +2000"].mean(), "t A": t_day(g, "A +2000"), "B +2000": g["B +2000"].mean(), "t B": t_day(g, "B +2000")})
    print(f"E2 without look-ahead: {len(K):,} TWAPs, of which core set {K.core.sum():,}\n" + pd.DataFrame(rows).to_string(index=False))
