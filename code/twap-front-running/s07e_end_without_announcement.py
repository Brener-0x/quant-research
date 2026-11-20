"""Section 7. Without the announcement, what does detecting the end of the TWAP (35 s without a slice) cost?

Method fixed before the computation. Core set of s07d; entry as in s06b; exits: right after the last slice
(reference), live rule (35 s after the first slice followed by a gap > 35 s), last slice + 35, 60, 90 s. Base fees.
    python s07e_end_without_announcement.py   -> work/end_without_announcement.parquet
"""
import numpy as np, pandas as pd
from common import DAYS, TAKER_FEE, WORK, first_after, show, t_day
from s07c_reasons import SELECT, VERIFY

ENTRY_WIN, EXIT_WIN, GAP = 60_000, 600_000, 35_000
EXITS = {"reference (end known)": None, "live rule (gap > 35 s)": "live", "last + 35 s": 35_000, "last + 60 s": 60_000, "last + 90 s": 90_000}

if __name__ == "__main__":
    X = pd.read_parquet(WORK / "core_set.parquet", columns=["user", "twapId", "coin", "day", "month", "t1", "net_side", "net s06b"])
    X = X.merge(pd.read_parquet(WORK / "twaps.parquet", columns=["user", "twapId", "end"]), on=["user", "twapId"])
    L, gaps = [], []
    for d in DAYS:
        Xd = X[X.day == d]
        if len(Xd) == 0:
            continue
        i_ = DAYS.index(d); dd_ = [d] + ([DAYS[i_ + 1]] if i_ + 1 < len(DAYS) else [])
        Od = pd.concat([pd.read_parquet(WORK / "orders" / f"{x}.parquet", columns=["coin", "t", "side", "px"]) for x in dd_]); Od = Od[Od.coin.isin(set(Xd.coin))]
        Tw = pd.concat([pd.read_parquet(WORK / "twap_fills" / f"{x}.parquet", columns=["user", "twapId", "t"]) for x in dd_])
        Tw = Tw[Tw.twapId.isin(set(Xd.twapId))].drop_duplicates(["user", "twapId", "t"]).sort_values("t")
        TT = {k: g.t.to_numpy().astype(np.int64) for k, g in Tw.groupby(["user", "twapId"])}
        for coin, G in Xd.groupby("coin"):
            Oc = Od[Od.coin == coin].sort_values("t"); book = {s: (Oc[Oc.side == s].t.to_numpy(), Oc[Oc.side == s].px.to_numpy()) for s in (1, -1)}
            for r in G.to_dict("records"):
                sg, t1, end = int(r["net_side"]), int(r["t1"]), int(r["end"])
                _, pe = first_after(*book[sg], np.array([t1 + 1000]), np.array([t1 + 1000 + ENTRY_WIN])); pe = pe[0]
                tt = TT.get((r["user"], r["twapId"]), np.array([t1, end])); tt = tt[(tt >= t1) & (tt <= end)]
                dt = np.diff(tt); holes = np.where(dt > GAP)[0]
                gaps.append({"twapId": r["twapId"], "median gap (s)": np.median(dt) / 1000 if len(dt) else np.nan, "max gap (s)": dt.max() / 1000 if len(dt) else np.nan})
                res = {"user": r["user"], "twapId": r["twapId"], "coin": coin, "day": d, "month": r["month"], "net s06b": r["net s06b"], "entered": not np.isnan(pe)}
                if not np.isnan(pe):
                    for name, v in EXITS.items():
                        ts = end if v is None else ((tt[holes[0]] if len(holes) else end) + GAP if v == "live" else end + v)
                        _, px = first_after(*book[-sg], np.array([ts]), np.array([ts + EXIT_WIN]))
                        res[name] = sg * np.log(px[0] / pe) * 1e4 - 2 * TAKER_FEE
                L.append(res)
    M = pd.DataFrame(L).replace([np.inf, -np.inf], np.nan); E = pd.DataFrame(gaps)
    names = list(EXITS); M[names] = M[names].clip(-2000, 2000)
    for n in names[1:]:
        M[f"cost {n}"] = M[names[0]] - M[n]
    M.to_parquet(WORK / "end_without_announcement.parquet", index=False)
    show()
    g = M[M.entered & M[names[0]].notna()]
    print(f"Gaps between slices: median of medians {E['median gap (s)'].median():.2f} s; largest gap, quantiles {E['max gap (s)'].quantile([.5, .9, .99]).round(1).to_dict()}")
    rows = []
    for n in names:
        r = {"exit": n, "n": g[n].notna().sum(), "net": g[n].mean(), "t": t_day(g, n)}
        if n != names[0]:
            r["cost"] = g[f"cost {n}"].mean(); r["t cost"] = t_day(g, f"cost {n}"); r["cost verification"] = g[g.month.isin(VERIFY)][f"cost {n}"].mean()
        rows.append(r)
    print("\nBy exit (net bp, base fees):\n", pd.DataFrame(rows).to_string(index=False))
