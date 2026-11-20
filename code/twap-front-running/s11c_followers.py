"""Section 11. Do the followers earn? If they do, the edge changed hands; if not, competition destroyed it.

Population: every (signal, wallet) of work/followers.parquet (same-side taker within 2 s after the first slice, TWAP
slices and the TWAP's own wallet excluded). Entry = volume-weighted price of ITS same-side taker orders in (t1, t1 + 2 s];
fee = ITS own fee (fee column of the orders), counted twice (round trip at the same rate). Marked at the last non-TWAP
taker price (at most 5 minutes old): t1 + 30 s, t1 + 5 min, end of the TWAP (exit A of the paths), end + 5 min.
Their real exits are unknown: this is the result of a follower who leaves at those times, not its realised P&L.
Dollar-weighted, by month, for all followers and for the 20 most frequent (s10a).
    python s11c_followers.py [workers]   -> work/follower_pnl/<day>.parquet
"""
import sys
from multiprocessing import Pool
import numpy as np, pandas as pd
from common import WORK, work
from s10a_decay import SOURCES, signal_id, top_followers

OUT = work("follower_pnl")
VALID = 300_000
HORIZONS = ["30 s", "5 min", "TWAP end", "end + 5 min"]


def signals():
    L = []
    for per, (fc, _, _) in SOURCES.items():
        R = pd.read_parquet(fc, columns=["user", "twapId", "coin", "side", "t1", "t exit", "day", "px entry", "A2 +2000"])
        R = R[R["px entry"].notna() & R["A2 +2000"].notna()].copy()
        R["sid"] = signal_id(per, R.user, R.twapId); R["period"] = per; L.append(R)
    return pd.concat(L)


def one_day(args):
    d, per, S = args
    f = OUT / f"{d}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    _, dord, dtw = SOURCES[per]
    nx = (pd.Timestamp(d) + pd.Timedelta(days=1)).strftime("%Y%m%d")
    fl = [x for x in (d, nx) if (dord / f"{x}.parquet").exists()]
    O = pd.concat([pd.read_parquet(dord / f"{x}.parquet", columns=["user", "coin", "oid", "t", "side", "usd", "fee", "px"]) for x in fl])
    O["coin"] = O.coin.astype(str); O = O[O.coin.isin(set(S.coin))]; O["user"] = O.user.astype(str)
    tw = set(pd.concat([pd.read_parquet(dtw / f"{x}.parquet", columns=["oid"]) for x in fl if (dtw / f"{x}.parquet").exists()]).oid)
    O["twap"] = O.oid.isin(tw)
    rows = []
    for coin, G in S.groupby("coin"):
        Oc = O[O.coin == coin].sort_values("t"); Nt = Oc[~Oc.twap]
        tn, pn = Nt.t.to_numpy(), Nt.px.to_numpy()

        def price_at(h):
            i = np.searchsorted(tn, h, side="right") - 1
            return pn[i] if i >= 0 and h - tn[i] <= VALID else np.nan
        t, uu, se, us, fr, px, tv = (Oc[c].to_numpy() for c in ("t", "user", "side", "usd", "fee", "px", "twap"))
        for r in G.to_dict("records"):
            a, b = np.searchsorted(t, r["t1"], side="right"), np.searchsorted(t, r["t1"] + 2000, side="right")
            m = (uu[a:b] == r["wallet"]) & (se[a:b] == r["side"]) & ~tv[a:b]
            if not m.any():
                continue
            u_, p_, f_ = us[a:b][m], px[a:b][m], fr[a:b][m]
            pe = np.average(p_, weights=u_); usd = u_.sum()
            res = {"sid": r["sid"], "wallet": r["wallet"], "day": d, "month": d[:6], "period": per, "coin": coin, "usd": usd,
                   "fee bp (one side)": 1e4 * np.nansum(f_) / usd}
            for name, h in zip(HORIZONS, (r["t1"] + 30_000, r["t1"] + 300_000, int(r["t exit"]), int(r["t exit"]) + 300_000)):
                p = price_at(h); res[f"gross {name}"] = r["side"] * np.log(p / pe) * 1e4 if p == p else np.nan
            rows.append(res)
    D = pd.DataFrame(rows); D.to_parquet(f, index=False); return D


def wmean(d, c):
    d = d[d[c].notna()]; return np.average(d[c], weights=d.usd) if len(d) else np.nan


def t_day_w(d, c):
    d = d[d[c].notna()]; m = d.groupby("day").apply(lambda x: np.average(x[c], weights=x.usd), include_groups=False)
    return m.mean() / m.std() * np.sqrt(len(m)) if len(m) > 2 else np.nan


if __name__ == "__main__":
    Sv = pd.read_parquet(WORK / "followers.parquet"); top20 = top_followers(Sv)
    X = Sv.merge(signals()[["sid", "coin", "side", "t1", "t exit", "day", "period"]], on="sid")
    jobs = [(d, g.period.iloc[0], g) for d, g in X.groupby("day")]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 3) as p:
        D = pd.concat(p.map(one_day, jobs, chunksize=1))
    for h in HORIZONS:
        D[f"net {h}"] = D[f"gross {h}"] - 2 * D["fee bp (one side)"]
    D["top20"] = D.wallet.isin(top20)
    D["net end capped"] = D["net TWAP end"].clip(-500, 500); D["$ net"] = D.usd * D["net TWAP end"] / 1e4
    pd.set_option("display.width", 260); pd.options.display.float_format = "{:.2f}".format
    print(f"{len(D):,} follower entries ({D.wallet.nunique():,} wallets, {D.usd.sum() / 1e6:,.1f} M$); top 20: {D.top20.sum():,}")
    for name, d in (("ALL followers", D), ("20 most frequent followers", D[D.top20])):
        d = d[d["net TWAP end"].notna()]; g = d.groupby("month")
        T = pd.DataFrame({"entries": g.size(), "M$": g.usd.sum() / 1e6, "fee bp (one side)": g.apply(lambda x: wmean(x, "fee bp (one side)"), include_groups=False),
                          "net end, capped +/-500 ($-weighted)": g.apply(lambda x: wmean(x, "net end capped"), include_groups=False),
                          "net end median": g["net TWAP end"].median(), "winners %": g["net TWAP end"].apply(lambda s: 100 * (s > 0).mean()), "$ net": g["$ net"].sum()})
        T.loc["TOTAL"] = [len(d), d.usd.sum() / 1e6, wmean(d, "fee bp (one side)"), wmean(d, "net end capped"), d["net TWAP end"].median(),
                          100 * (d["net TWAP end"] > 0).mean(), d["$ net"].sum()]
        print(f"\n{name} (marked at the end of the TWAP, own fees):\n" + T.to_string())
        print(f"  daily t, net at the end ($-weighted, uncapped): {t_day_w(d, 'net TWAP end'):.2f}")
    W = D[D.top20].groupby("wallet")["$ net"].sum().sort_values(ascending=False)
    print(f"\nTop 20: largest single-wallet contribution {W.iloc[0]:,.0f} dollars of {W.sum():,.0f}")
