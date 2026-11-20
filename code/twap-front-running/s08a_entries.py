"""Section 8. Model, step 1: three entries (exploratory, 150 in-sample days, rules of Section 7 unchanged).

E1: signal read from the public trade feed alone (a slice = a trade with a null hash, s08e): a "new TWAP" is a slice of a
    (wallet, coin, side) with no slice in the previous G seconds; the wallet's 7-day count is kept by me from the same
    detected starts; entry at the first slice + 1 s. No API needed.
E2: signal confirmed by the API (twapHistory, answers in ~1.2 s): the exact core set of s07d; entry at first slice + 2 s.
E3: half at + 1 s on the E1 signal; at + 2 s the API confirms (core set) and the other half enters, otherwise the first
    half leaves as a taker at + 2 s. Core-set TWAPs that E1 missed enter at + 2 s for half.
Exits: A = right after the end of the TWAP (duration from the API); B = 35 s without a slice of the (wallet, coin, side)
in the public feed. Prices actually paid, 4.5 bp per side, zero size (my impact comes in s08d).
    python s08a_entries.py   -> work/starts/<day>.parquet, work/signals_e.parquet, work/candidates.parquet
"""
import numpy as np, pandas as pd
from common import DAYS, TAKER_FEE, WORK, first_after, show, t_day, work
from s07a_wallet_frequency import WEEK
from s07b_global_weight import KL, symbols
from s07c_reasons import BACKED, SPOT, VERIFY, hour_volume, klines, spot_of, week_volume
from s07d_cumulative_pressure import R1B_THRESHOLD

STARTS = work("starts")
GS = [60, 120, 180, 300]                  # gap (s) that makes a "new TWAP" in the public feed
ENTRY_WIN, EXIT_WIN, GAP = 60_000, 600_000, 35_000
DELAYS = [1000, 1500, 2000, 3000]         # entry after the first slice (ms)
DS = set(DAYS)


def day_before(d):
    p = (pd.Timestamp(d) - pd.Timedelta(days=1)).strftime("%Y%m%d"); return p if p in DS else None


def day_after(d):
    p = (pd.Timestamp(d) + pd.Timedelta(days=1)).strftime("%Y%m%d"); return p if p in DS else None


def full_history(d):
    return all((pd.Timestamp(d) - pd.Timedelta(days=k)).strftime("%Y%m%d") in DS for k in range(1, 8))


def public_slices(d, t_min=None):
    """Public slices of a day: (wallet, coin, side, time), dollars, main twapId (ground truth, not used by E1)."""
    F = pd.read_parquet(WORK / "twap_fills" / f"{d}.parquet", columns=["user", "twapId", "coin", "t", "side", "usd", "taker"])
    F = F[F.taker]
    if t_min is not None:
        F = F[F.t >= t_min]
    F = F.sort_values("usd", ascending=False)
    return F.groupby(["user", "coin", "side", "t"], sort=False).agg(usd=("usd", "sum"), twapId=("twapId", "first")).reset_index()


def detect(d):
    f = STARTS / f"{d}.parquet"
    if f.exists():
        return
    S = public_slices(d); v = day_before(d); d0 = int(pd.Timestamp(d).value // 1_000_000)
    if v is not None:
        S = pd.concat([public_slices(v, d0 - 600_000).assign(prev=True), S.assign(prev=False)])
    else:
        S["prev"] = False
    S = S.sort_values(["user", "coin", "side", "t"]).reset_index(drop=True)
    same = (S.user == S.user.shift()) & (S.coin == S.coin.shift()) & (S.side == S.side.shift())
    S["gap"] = np.where(same, S.t - S.t.shift(), np.inf)
    S = S[~S.prev & (S.gap > GS[0] * 1000)].drop(columns="prev")
    S.to_parquet(f, index=False)


if __name__ == "__main__":
    for d in DAYS:                                            # 1. starts detected in the public feed, all coins
        detect(d)
    D = pd.concat([pd.read_parquet(STARTS / f"{d}.parquet").assign(day=d) for d in DAYS], ignore_index=True)
    T = pd.read_parquet(WORK / "twaps.parquet", columns=["user", "twapId", "start", "end"])
    D = D.merge(T, on=["user", "twapId"], how="left"); D["true"] = D.t == D.start
    for g in GS:
        D[f"n7 G{g}"] = -1
        m = D.gap > g * 1000; Dg = D[m].sort_values(["user", "t"])
        n7 = np.empty(len(Dg), dtype=np.int64); tt = Dg.t.to_numpy()
        for u, idx in Dg.groupby("user", sort=False).indices.items():
            x = tt[idx]; n7[idx] = np.searchsorted(x, x, side="left") - np.searchsorted(x, x - WEEK, side="left")
        D.loc[Dg.index, f"n7 G{g}"] = n7
    # 2. the core set (E2)
    P = pd.read_parquet(WORK / "core_set.parquet", columns=["user", "twapId", "coin", "day", "month", "t1", "net_side", "usd1", "net s06b"])
    assert len(P) == 5077 and round(P["net s06b"].mean(), 2) == 38.29
    core = set(zip(P.user, P.twapId))
    # 3. E1 candidates: possible coins (referenced perps with Binance klines), Section 7 rules read from the public feed
    SY = symbols(); CO = {c for c, s in SY.items() if c not in BACKED}
    FH = {d: full_history(d) for d in DAYS}; K = D[D.coin.isin(CO) & D.day.map(FH)].copy()
    K["t"] = K.t.astype(np.int64); K["vol30"] = np.nan; K["bin1h"] = 0.0; K["spot1h"] = 0.0; K["binday"] = 0.0
    for d in DAYS:
        idx = np.flatnonzero(K.day.to_numpy() == d)
        if len(idx) == 0:
            continue
        B = pd.read_parquet(WORK / "bars5s" / f"{d}.parquet", columns=["coin", "t5", "notional"]); B["coin"] = B.coin.astype(str)
        v = day_before(d)
        if v is not None:
            Bp = pd.read_parquet(WORK / "bars5s" / f"{v}.parquet", columns=["coin", "t5", "notional"]); Bp["coin"] = Bp.coin.astype(str); B = pd.concat([Bp, B])
        G = {c: (g.t5.to_numpy(), np.concatenate([[0], np.cumsum(g.notional.to_numpy())])) for c, g in B.sort_values("t5").groupby("coin")}
        for c, sub in K.iloc[idx].groupby("coin"):
            g = G.get(c)
            if g is None:
                continue
            t1 = sub.t.to_numpy(); a, b = np.searchsorted(g[0], t1 - 3_600_000, side="left"), np.searchsorted(g[0], t1, side="left")
            K.loc[sub.index, "vol30"] = (g[1][b] - g[1][a]) / 120
    for c, idx in K.groupby("coin").indices.items():
        t1 = K.t.to_numpy()[idx]; s = SY[c]
        tp, cp = klines(KL, s); ts, cs = klines(SPOT, spot_of(s))
        K.iloc[idx, K.columns.get_loc("bin1h")] = hour_volume(tp, cp, t1); K.iloc[idx, K.columns.get_loc("spot1h")] = hour_volume(ts, cs, t1)
        K.iloc[idx, K.columns.get_loc("binday")] = week_volume(tp, cp, t1) + week_volume(ts, cs, t1)
    K["HL weight %"] = 100 * K.usd / K.vol30
    K["widened weight %"] = 100 * K.usd / ((K.vol30 * 120 + K.bin1h + K.spot1h) / 120)
    K["heavy"] = (K["HL weight %"] >= 20) & (K["widened weight %"] >= 5) & (K.binday < R1B_THRESHOLD)
    K["core"] = pd.Series([(u, i) in core for u, i in zip(K.user, K.twapId)], index=K.index) & K["true"]
    # 4. signals: E1 (each G) and the core set; entry prices, exits A and public B
    E1 = {g: K[K.heavy & (K.gap > g * 1000) & (K[f"n7 G{g}"] <= 30)] for g in GS}
    keep = ["user", "coin", "side", "t", "twapId", "day", "end", "true", "core", "usd"]
    sig = pd.concat([E1[g][keep + ["vol30"]] for g in GS]).drop_duplicates(["user", "coin", "side", "t"])
    Pq = P.merge(T[["user", "twapId", "end"]], on=["user", "twapId"]).rename(columns={"t1": "t", "net_side": "side", "usd1": "usd"})
    Pq = Pq.assign(true=True, core=True)[keep]
    Pq["t"] = Pq.t.astype(np.int64); Pq["side"] = Pq.side.astype(np.int64)
    sig = pd.concat([Pq, sig]).drop_duplicates(["user", "coin", "side", "t"]).reset_index(drop=True)
    L = []
    for d in DAYS:
        Sd = sig[sig.day == d]
        if len(Sd) == 0:
            continue
        dd_ = [d] + ([day_after(d)] if day_after(d) else [])
        Od = pd.concat([pd.read_parquet(WORK / "orders" / f"{x}.parquet", columns=["coin", "t", "side", "px"]) for x in dd_]); Od = Od[Od.coin.isin(set(Sd.coin))]
        Tr = pd.concat([public_slices(x) for x in dd_]); Tr = Tr.merge(Sd[["user", "coin", "side"]].drop_duplicates(), on=["user", "coin", "side"])
        TR = {k: np.sort(g.t.to_numpy().astype(np.int64)) for k, g in Tr.groupby(["user", "coin", "side"])}
        for coin, G in Sd.groupby("coin"):
            Oc = Od[Od.coin == coin].sort_values("t"); book = {s: (Oc[Oc.side == s].t.to_numpy(), Oc[Oc.side == s].px.to_numpy()) for s in (1, -1)}
            for r in G.itertuples(index=False):
                sg, t1, end = int(r.side), int(r.t), int(r.end)
                res = {"user": r.user, "coin": coin, "side": sg, "t": t1, "twapId": r.twapId}
                pe = {}
                for dl in DELAYS:
                    _, p = first_after(*book[sg], np.array([t1 + dl]), np.array([t1 + dl + ENTRY_WIN])); pe[dl] = p[0]
                tt = TR.get((r.user, coin, sg), np.array([t1])); tt = tt[tt >= t1]
                k = np.flatnonzero(np.diff(tt) > GAP); end_seen = (tt[k[0]] if len(k) else tt[-1]) + GAP
                _, pA = first_after(*book[-sg], np.array([max(end, t1 + 2000)]), np.array([max(end, t1 + 2000) + EXIT_WIN]))
                _, pB = first_after(*book[-sg], np.array([max(end_seen, t1 + 2000)]), np.array([max(end_seen, t1 + 2000) + EXIT_WIN]))
                _, p2 = first_after(*book[-sg], np.array([t1 + 2000]), np.array([t1 + 2000 + ENTRY_WIN]))
                for dl in DELAYS:
                    res[f"A +{dl}"] = sg * np.log(pA[0] / pe[dl]) * 1e4 - 2 * TAKER_FEE
                    res[f"B +{dl}"] = sg * np.log(pB[0] / pe[dl]) * 1e4 - 2 * TAKER_FEE
                res["cut +2 s"] = sg * np.log(p2[0] / pe[1000]) * 1e4 - 2 * TAKER_FEE
                res["end seen - end (s)"] = (end_seen - end) / 1000
                L.append(res)
    R = pd.DataFrame(L).replace([np.inf, -np.inf], np.nan)
    num = [c for c in R.columns if c.startswith(("A +", "B +", "cut"))]; R[num] = R[num].clip(-2000, 2000)
    sig = sig.merge(R, on=["user", "coin", "side", "t", "twapId"], how="left"); sig["month"] = sig.day.str[:6]
    sig.to_parquet(WORK / "signals_e.parquet", index=False)
    K.drop(columns=[c for c in K.columns if c.startswith("n7")]).to_parquet(WORK / "candidates.parquet", index=False)
    show()
    key = lambda d_: set(zip(d_.user, d_.coin, d_.side, d_.t))
    Sg = sig.set_index(["user", "coin", "side", "t"]); kp = key(Pq); Pg = Sg.loc[list(kp)].reset_index(); nd = Pq.day.nunique()
    rows = []
    for dl in DELAYS:
        for x in ("A", "B"):
            c = f"{x} +{dl}"; rows.append({"entry": f"+ {dl / 1000:g} s", "exit": x, "n": Pg[c].notna().sum(), "net": Pg[c].mean(), "t": t_day(Pg, c)})
    print("Core set, net per signal by entry delay (zero size):\n", pd.DataFrame(rows).to_string(index=False))
    tab = []
    for x in ("A", "B"):
        e2 = Pg.assign(g=Pg[f"{x} +2000"])
        tab.append({"entry": "E2 (+ 2 s, API)", "G": "", "exit": x, "positions": len(e2), "net per position": e2.g.mean(), "t": t_day(e2, "g"), "bp per day": e2.g.sum() / nd})
        for g in GS:
            e = Sg.loc[list(key(E1[g]))].reset_index(); e1 = e.assign(g=e[f"{x} +1000"])
            tab.append({"entry": "E1 (+ 1 s, public)", "G": g, "exit": x, "positions": len(e1), "net per position": e1.g.mean(), "t": t_day(e1, "g"), "bp per day": e1.g.sum() / nd})
            h1 = 0.5 * np.where(e.core, e[f"{x} +1000"], e["cut +2 s"]); h2 = 0.5 * Pg[f"{x} +2000"]
            e3 = pd.concat([pd.DataFrame({"day": e.day, "g": h1}), pd.DataFrame({"day": Pg.day, "g": h2})]); e3d = e3.groupby("day").g.sum()
            tab.append({"entry": "E3 (half + 1 s, half + 2 s)", "G": g, "exit": x, "positions": len(e) + len(Pg), "net per position": np.nan,
                        "t": e3d.mean() / e3d.std() * np.sqrt(len(e3d)), "bp per day": e3.g.sum() / nd})
    print("\nE1, E2, E3 (bp per day = sum of bp per day, i.e. dollars a day for 10,000 dollars per position):\n", pd.DataFrame(tab).to_string(index=False))
