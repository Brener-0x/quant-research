"""Section 9. Wake (second version) OUT OF SAMPLE, 2026-06 to 2026-10, protocol and criterion written before reading.

Nothing is tuned here: frozen rules (Sections 7 and 8), impact curve and tercile bounds from the in-sample days, same
leverage table. Inputs: tables of s09a, free Binance klines, twapHistory.
Criterion: total profit > 0 and daily t >= 3 for 25,000 dollars, 1 % risk, two positions per coin, exit A.
    python s09b_oos.py --download   -> Binance klines 2026-05 to 2026-10, perp and spot
    python s09b_oos.py --api        -> work/oos/api/<wallet>.json
    python s09b_oos.py              -> work/oos/signals.parquet, work/oos/paths.parquet
"""
import json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
from common import DATA, OOS_DAYS, TAKER_FEE, WORK, coin_reference, first_after, show, t_day, work
from s06c_announcements import one as api_one
from s07a_wallet_frequency import WEEK
from s07b_global_weight import kline_file
from s07c_reasons import hour_volume, klines, spot_of, week_volume
from s07d_cumulative_pressure import R1B_THRESHOLD
from s08d_impact import frozen_curves
from s08h_wake_v1 import B, CAPITALS, K0, RISKS, S0, in_sample, paths, simulate, summary

O = work("oos"); KL, SPD, API = work("oos", "klines_perp"), work("oos", "klines_spot"), work("oos", "api")
MONTHS_K = ["2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10"]
DAYS = OOS_DAYS
START_OOS = int(pd.Timestamp("2026-06-01").value // 1_000_000); END_OOS = int(pd.Timestamp("2026-11-01").value // 1_000_000)
ENTRY_WIN, EXIT_WIN, GAP = 60_000, 600_000, 35_000
BACKED = {"PAXG"}


def day_after(d):
    p = (pd.Timestamp(d) + pd.Timedelta(days=1)).strftime("%Y%m%d"); return p if p in DAYS else None


def universe():
    """Perps with an external reference (data/coin_reference.csv) plus perps listed since, when a Binance USDT perp
    exists (checked with a HEAD request). The list obtained is frozen in data/universe_oos.csv."""
    f = DATA / "universe_oos.csv"
    if f.exists():
        U = pd.read_csv(f, keep_default_na=False); return dict(zip(U.coin, U.binance))
    U = coin_reference(); U = U[(U.type == "perp") & U.reference.str.startswith("external")]
    uni = {c: str(b or "").replace("partial:", "") for c, b in zip(U.coin, U.binance)}
    coins = set()
    for d in DAYS:
        coins |= set(pd.read_parquet(O / "orders" / f"{d}.parquet", columns=["coin"]).coin.astype(str).unique())
    known = set(coin_reference().coin); coins = {c for c in coins if c[:1].isalnum()}       # not the "#..." markets
    for c in sorted(coins - known):
        for s in ([f"1000{c[1:]}USDT"] if c.startswith("k") else []) + [f"{c}USDT"]:
            try:
                urllib.request.urlopen(urllib.request.Request(f"https://data.binance.vision/data/futures/um/monthly/klines/{s}/5m/{s}-5m-2026-08.zip", method="HEAD"), timeout=30)
                uni[c] = s; break
            except Exception:
                pass
    uni = {c: s for c, s in uni.items() if c in coins and c not in BACKED}
    pd.DataFrame(sorted(uni.items()), columns=["coin", "binance"]).to_csv(f, index=False); return uni


def twap_table():
    """TWAPs of the out-of-sample days, with their first slice and the wallet's 7-day count (in-sample starts included)."""
    f = O / "twaps.parquet"
    if f.exists():
        return pd.read_parquet(f)
    A, P = [], []
    for d in DAYS:
        D = pd.read_parquet(O / "twap_fills" / f"{d}.parquet", columns=["user", "twapId", "coin", "t", "usd", "side"])
        A.append(D.assign(one=1).groupby(["user", "twapId"]).agg(coin=("coin", "first"), start=("t", "min"), end=("t", "max"), slices=("t", "nunique"),
                                                                 usd=("usd", "sum"), side=("side", "sum"), n=("one", "sum")).reset_index())
        g = D.groupby(["user", "twapId", "t"], sort=False).usd.sum().reset_index().sort_values(["user", "twapId", "t"])
        P.append(g.groupby(["user", "twapId"]).head(1).rename(columns={"t": "t1", "usd": "usd1"}))
    A = pd.concat(A); T = A.groupby(["user", "twapId"]).agg(coin=("coin", "first"), start=("start", "min"), end=("end", "max"), slices=("slices", "sum"),
                                                            usd=("usd", "sum"), side=("side", "sum"), n=("n", "sum")).reset_index()
    T["net_side"] = np.sign(T.side / T.n)
    P = pd.concat(P).sort_values("t1").groupby(["user", "twapId"]).head(1)
    T = T.merge(P, on=["user", "twapId"], how="left")
    Ta = pd.read_parquet(WORK / "twaps.parquet", columns=["user", "twapId", "start"]).rename(columns={"start": "start in sample"})
    T = T.merge(Ta, on=["user", "twapId"], how="left"); T["new"] = T["start in sample"].isna() & (T.start >= START_OOS)
    Dd = pd.concat([Ta.rename(columns={"start in sample": "start"}), T[T["start in sample"].isna()][["user", "twapId", "start"]]]).drop_duplicates(["user", "twapId"])
    Dd = Dd.sort_values(["user", "start"]); pos = {u: g.start.to_numpy().astype(np.int64) for u, g in Dd.groupby("user", sort=False)}
    n7 = np.full(len(T), -1, dtype=np.int64); d0 = T.start.to_numpy().astype(np.int64)
    for u, idx in T.groupby("user").indices.items():
        x = pos[u]; n7[idx] = np.searchsorted(x, d0[idx], side="left") - np.searchsorted(x, d0[idx] - WEEK, side="left")
    T["n7"] = n7
    T.to_parquet(f, index=False); return T


def signals():
    """E2 without look-ahead, out of sample."""
    f = O / "signals.parquet"
    if f.exists():
        return pd.read_parquet(f)
    T = twap_table(); UNI = universe()
    K = T[T.new & T.coin.isin(UNI) & (T.n7 <= 30) & (T.t1 >= START_OOS + 3_600_000) & (T.net_side != 0)].copy().reset_index(drop=True)
    K["t1"] = K.t1.astype(np.int64); K["side"] = K.net_side.astype(np.int64); K["day"] = pd.to_datetime(K.t1, unit="ms").dt.strftime("%Y%m%d")
    K["vol30"] = np.nan
    for d in DAYS:
        idx = np.flatnonzero(K.day.to_numpy() == d)
        if len(idx) == 0:
            continue
        Bd = pd.concat([pd.read_parquet(O / "bars5s" / f"{x}.parquet", columns=["coin", "t5", "notional"]) for x in
                        [(pd.Timestamp(d) - pd.Timedelta(days=1)).strftime("%Y%m%d"), d] if (O / "bars5s" / f"{x}.parquet").exists()])
        Bd["coin"] = Bd.coin.astype(str)
        G = {c: (g.t5.to_numpy(), np.concatenate([[0], np.cumsum(g.notional.to_numpy())])) for c, g in Bd.sort_values("t5").groupby("coin")}
        for c, sub in K.iloc[idx].groupby("coin"):
            g = G.get(c)
            if g is None:
                continue
            t1 = sub.t1.to_numpy(); a, b = np.searchsorted(g[0], t1 - 3_600_000, side="left"), np.searchsorted(g[0], t1, side="left")
            K.loc[sub.index, "vol30"] = (g[1][b] - g[1][a]) / 120
    K["bin1h"] = 0.0; K["spot1h"] = 0.0; K["binday"] = 0.0
    for c, idx in K.groupby("coin").indices.items():
        t1 = K.t1.to_numpy()[idx]; s = UNI[c]
        tp, cp = klines(KL, s, MONTHS_K); ts, cs = klines(SPD, spot_of(s), MONTHS_K)
        K.iloc[idx, K.columns.get_loc("bin1h")] = hour_volume(tp, cp, t1); K.iloc[idx, K.columns.get_loc("spot1h")] = hour_volume(ts, cs, t1)
        K.iloc[idx, K.columns.get_loc("binday")] = week_volume(tp, cp, t1) + week_volume(ts, cs, t1)
    K["HL weight %"] = 100 * K.usd1 / K.vol30
    K["widened weight %"] = 100 * K.usd1 / ((K.vol30 * 120 + K.bin1h + K.spot1h) / 120)
    K = K[(K["HL weight %"] >= 20) & (K["widened weight %"] >= 5) & (K.binday < R1B_THRESHOLD)].reset_index(drop=True)
    K["month"] = K.day.str[:6]
    K.to_parquet(f, index=False); return K


def announcements(K):
    L = []
    for u in K.user.unique():
        f = API / f"{u}.json"
        if not f.exists():
            continue
        for e in json.loads(f.read_text(encoding="utf-8"))["kept"]:
            st, s_ = e.get("state", {}), e.get("status", {})
            L.append({"user": u, "twapId": e["twapId"], "minutes": st.get("minutes"), "t_sent": st.get("timestamp"), "sz": float(st.get("sz", "nan")),
                      "exec_sz": float(st.get("executedSz", "nan")), "status": s_.get("status") if isinstance(s_, dict) else str(s_), "time": e.get("time")})
    A = pd.DataFrame(L)
    A["rank"] = A.status.map({"activated": 0, "finished": 2, "terminated": 2, "error": 2}).fillna(1)
    canc = A[A.status == "terminated"].groupby(["user", "twapId"]).time.max().mul(1000).rename("t cancel")
    F = A.sort_values(["twapId", "rank"]).groupby(["user", "twapId"]).agg(minutes=("minutes", "first"), t_sent=("t_sent", "first"), status=("status", "last"),
                                                                         sz=("sz", "first"), exec_sz=("exec_sz", "last")).reset_index()
    F = F.merge(canc, on=["user", "twapId"], how="left")
    F["cancelled"] = (F.status == "terminated") | (F.exec_sz / F.sz < 0.95); F["found"] = True
    return F


def oos_paths():
    """Entry price, exits A (without look-ahead) and B, and the 5 % stop, for every out-of-sample signal."""
    f = O / "paths.parquet"
    if f.exists():
        return pd.read_parquet(f)
    K = signals().merge(announcements(signals()), on=["user", "twapId"], how="left")
    K["found"] = K.found.fillna(False).astype(bool); K["cancelled"] = K.cancelled.fillna(False).astype(bool)
    K["announced end"] = K.t_sent + K.minutes * 60_000
    tsA = np.where(K.cancelled, K["t cancel"] + 2000, K["announced end"]); K["t exit A2"] = np.where(K.found, tsA, np.nan)
    K["t entry"] = K.t1 + 2000
    from s08h_wake_v1 import LEV
    L = []
    for d in DAYS:
        Sd = K[K.day == d]
        if len(Sd) == 0:
            continue
        dd_ = [d] + ([day_after(d)] if day_after(d) else [])
        Od = pd.concat([pd.read_parquet(O / "orders" / f"{x}.parquet", columns=["coin", "t", "side", "px"]) for x in dd_]); Od["coin"] = Od.coin.astype(str)
        Od = Od[Od.coin.isin(set(Sd.coin))]
        Tw = pd.concat([pd.read_parquet(O / "twap_fills" / f"{x}.parquet", columns=["user", "coin", "t", "side", "taker"]) for x in dd_])
        Tw = Tw[Tw.taker].merge(Sd[["user", "coin", "side"]].drop_duplicates(), on=["user", "coin", "side"])
        TR = {k: np.unique(g.t.to_numpy().astype(np.int64)) for k, g in Tw.groupby(["user", "coin", "side"])}
        for coin, G in Sd.groupby("coin"):
            Oc = Od[Od.coin == coin].sort_values("t"); ta, pa = Oc.t.to_numpy(), Oc.px.to_numpy()
            book = {s: (Oc[Oc.side == s].t.to_numpy(), Oc[Oc.side == s].px.to_numpy()) for s in (1, -1)}
            lev = LEV.get(coin, 3); M = B + 1 / (2 * lev)
            for r in G.to_dict("records"):
                sg, t1 = int(r["side"]), int(r["t1"])
                res = {"user": r["user"], "twapId": r["twapId"], "max leverage": lev, "margin M": M}
                te, pe = first_after(*book[sg], np.array([t1 + 2000]), np.array([t1 + 2000 + ENTRY_WIN])); te, pe = te[0], pe[0]
                res["px entry"] = pe
                tt = TR.get((r["user"], coin, sg), np.array([t1])); tt = tt[tt >= t1]
                k = np.flatnonzero(np.diff(tt) > GAP); end_seen = (tt[k[0]] if len(k) else tt[-1]) + GAP
                tA = r["t exit A2"]
                ts_ = {"B": max(end_seen, t1 + 2000), "A2": max(tA, t1 + 2000) if not np.isnan(tA) else np.nan}
                for x, tsx in ts_.items():
                    v = np.nan
                    if not np.isnan(pe) and not np.isnan(tsx):
                        _, p = first_after(*book[-sg], np.array([int(tsx)]), np.array([int(tsx) + EXIT_WIN]))
                        v = np.clip(sg * np.log(p[0] / pe) * 1e4 - 2 * TAKER_FEE, -2000, 2000)
                    res[f"{x} +2000"] = v
                res["t exit B"] = ts_["B"]
                if np.isnan(res["A2 +2000"]):
                    res["A2 +2000"] = res["B +2000"]
                ts = int(ts_["A2"]) if not np.isnan(ts_["A2"]) else int(ts_["B"])
                res["t exit"] = ts
                if not np.isnan(pe):
                    a, b = np.searchsorted(ta, te, side="right"), np.searchsorted(ta, ts, side="right")
                    mv = sg * np.log(pa[a:b] / pe); kk = np.flatnonzero(mv <= np.log(1 - S0))
                    if len(kk):
                        t_ = int(ta[a + kk[0]]); _, pf = first_after(*book[-sg], np.array([t_]), np.array([t_ + EXIT_WIN]))
                        g = sg * np.log(pf[0] / pe) * 1e4 if not np.isnan(pf[0]) else np.log(1 - S0) * 1e4
                        res[f"stop {S0:.0%} t"] = t_; res[f"stop {S0:.0%} bp"] = max(g, -M * 1e4) - 2 * TAKER_FEE
                    else:
                        res[f"stop {S0:.0%} t"] = np.nan; res[f"stop {S0:.0%} bp"] = np.nan
                L.append(res)
    R = K.merge(pd.DataFrame(L), on=["user", "twapId"], how="left")
    R.to_parquet(f, index=False); return R


if __name__ == "__main__":
    if "--download" in sys.argv:
        UNI = universe(); S = sorted({s for s in UNI.values() if s})
        jobs = [(KL, s, m, False) for s in S for m in MONTHS_K] + [(SPD, spot_of(s), m, True) for s in S for m in MONTHS_K]
        with ThreadPoolExecutor(4) as ex:
            list(ex.map(lambda j: kline_file(j[0], j[1], j[2], END_OOS, spot=j[3]), jobs))
        sys.exit()
    if "--api" in sys.argv:
        W = signals().user.value_counts().index.tolist()
        with ThreadPoolExecutor(2) as ex:
            keep = lambda e: e.get("twapId") is not None and START_OOS - WEEK <= (e.get("state", {}).get("timestamp") or 0) < END_OOS + 86_400_000
            list(ex.map(lambda u: api_one(u, API, keep), W))
        sys.exit()
    R = oos_paths(); Zs, edges = frozen_curves()
    show()
    ok = R["px entry"].notna(); g = R[ok]
    print(f"Out of sample: {len(DAYS)} days; signals {len(R):,}; with entry price {ok.sum():,}; announcement found {R.found.mean():.1%}; cancelled {R.cancelled.sum():,}")
    print("\nEdge at zero size (fees included, no impact):\n" + pd.DataFrame([{"exit": n, "n": g[c].notna().sum(), "net bp": g[c].mean(), "daily t": t_day(g, c)}
                                                                              for c, n in (("A2 +2000", "A without look-ahead"), ("B +2000", "B"))]).to_string(index=False))
    print("By month (net bp, exit A):", g.groupby("month")["A2 +2000"].agg(["size", "mean"]).round(2).to_dict("index"))
    known = set(in_sample().coin)                       # coins that had a signal in sample
    print(f"Coins with an in-sample signal {g[g.coin.isin(known)]['A2 +2000'].mean():.2f} bp; the others {g[~g.coin.isin(known)]['A2 +2000'].mean():.2f} bp")
    print("Completed / cancelled (net bp, exit A):", g.groupby("cancelled")["A2 +2000"].mean().round(2).to_dict())

    def grid(Rx, dx):
        rows = []
        for Cap in CAPITALS:
            for name, r in RISKS.items():
                b = summary(simulate(Rx, r * Cap / B, S0, K0, "A2 +2000", Zs, edges, per_coin=2), Cap, dx)
                rows.append({"capital $": Cap, "risk": name, **{k: b[k] for k in ("total $", "% per month", "worst month %", "largest drawdown %", "daily t")}})
        return pd.DataFrame(rows)
    Ro = R[ok & R["A2 +2000"].notna()].copy()
    To = grid(Ro, DAYS); print("\nWake v2 OUT OF SAMPLE (two per coin, exit A):\n" + To.to_string(index=False))
    Ra = in_sample(); Ti = grid(Ra[Ra["px entry"].notna() & Ra["A2 +2000"].notna()], sorted(Ra.day.unique()))
    print("\nSame model in sample:\n" + Ti.to_string(index=False))
    P = simulate(Ro, 0.01 * 25_000 / B, S0, K0, "A2 +2000", Zs, edges, per_coin=2); b = summary(P, 25_000, DAYS)
    crit = b["total $"] > 0 and b["daily t"] >= 3
    print(f"\nVERDICT (criterion written before reading): profit {b['total $']:,.0f} dollars, {b['% per month']:.2f} % a month, daily t {b['daily t']:.2f} -> "
          + ("EDGE CONFIRMED" if crit else "EDGE NOT CONFIRMED"))
