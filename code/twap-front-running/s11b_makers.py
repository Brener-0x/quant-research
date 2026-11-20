"""Section 11. Step 2: what does the maker facing TWAP slices earn?

Each maker fill of s11a (20 days: the 10th and 20th of ten months) is marked at the last non-TWAP taker price
(work/orders and work/oos/orders, TWAP oids excluded) at t + 30 s, t + 5 min and at the end of the TWAP (last slice of the
twapId, at least t + 30 s). A mark is valid if it is after the fill and at most 5 minutes older than the horizon.
  gross bp      = -twap_side x ln(P_h / px) x 1e4         (the maker is on the other side)
  maker net bp  = gross - maker_fee / usd x 1e4           (a negative fee is a rebate, it adds)
  my net bp     = gross - 1.5 bp (perp) or 4 bp (spot)    (lowest fee tier, no rebate)
Dollar-weighted means. Wake set = twapIds of the E2 signals; followers = the 20 most frequent of s10a.
    python s11b_makers.py   -> work/makers/<day>.parquet
"""
import numpy as np, pandas as pd
from common import IN_SAMPLE, WORK, work
from s10a_decay import top_followers

IN = WORK / "counterparties"; OUT = work("makers")
MY_FEE = {"perp": 1.5, "spot": 4.0}


def mark(d):
    f = OUT / f"{d}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    dord, dtw = (WORK / "orders", WORK / "twap_fills") if d[:6] in IN_SAMPLE else (WORK / "oos" / "orders", WORK / "oos" / "twap_fills")
    nx = (pd.Timestamp(d) + pd.Timedelta(days=1)).strftime("%Y%m%d")
    fl = [x for x in (d, nx) if (dord / f"{x}.parquet").exists()]
    D = pd.read_parquet(IN / f"{d}.parquet")
    TW = pd.concat([pd.read_parquet(dtw / f"{x}.parquet", columns=["twapId", "oid", "t"]) for x in fl if (dtw / f"{x}.parquet").exists()])
    end = TW.groupby("twapId").t.max()
    O = pd.concat([pd.read_parquet(dord / f"{x}.parquet", columns=["coin", "oid", "t", "px"]) for x in fl])
    O["coin"] = O.coin.astype(str); O = O[O.coin.isin(set(D.coin)) & ~O.oid.isin(set(TW.oid))]
    O = O.drop_duplicates(["coin", "oid", "t"]).sort_values("t")
    D["t_end"] = np.maximum(D.twapId.map(end).fillna(D.t).to_numpy(np.int64), D.t.to_numpy() + 30_000)
    for k in ("30 s", "5 min", "end"):
        D[k] = np.nan
    for coin, idx in D.groupby("coin").groups.items():
        Oc = O[O.coin == coin]
        if Oc.empty:
            continue
        t, px = Oc.t.to_numpy(), Oc.px.to_numpy(); G = D.loc[idx]; tf, pf, sg = G.t.to_numpy(), G.px.to_numpy(), G.twap_side.to_numpy()
        for k, th in (("30 s", tf + 30_000), ("5 min", tf + 300_000), ("end", G.t_end.to_numpy())):
            i = np.searchsorted(t, th, side="right") - 1; ok = i >= 0; ii = np.where(ok, i, 0)
            ok &= (t[ii] > tf) & (th - t[ii] <= 300_000)
            D.loc[idx, k] = np.where(ok, -sg * np.log(px[ii] / pf) * 1e4, np.nan)
    D["fee bp"] = D.maker_fee / D.usd * 1e4
    D["market"] = np.where(D.coin.str.startswith("@"), "spot", "perp")
    D["month"] = d[:6]
    D = D.drop(columns=["px", "sz", "maker_fee", "maker_oid", "tid"])
    D.to_parquet(f, index=False); return D


def wm(d, c):
    v = d[c].notna(); return (d.loc[v, c] * d.loc[v, "usd"]).sum() / d.loc[v, "usd"].sum() if v.any() else np.nan


def summary(d):
    me = d.market.map(MY_FEE)
    return {"fills": len(d), "M$": d.usd.sum() / 1e6, "gross 30 s": wm(d, "30 s"), "gross 5 min": wm(d, "5 min"), "gross end": wm(d, "end"),
            "maker fee bp": (d["fee bp"] * d.usd).sum() / d.usd.sum(), "maker net 5 min": wm(d.assign(x=d["5 min"] - d["fee bp"]), "x"),
            "my net 5 min": wm(d.assign(x=d["5 min"] - me), "x"), "my net end": wm(d.assign(x=d["end"] - me), "x")}


if __name__ == "__main__":
    days = sorted(p.stem for p in IN.glob("*.parquet"))
    D = pd.concat([mark(d) for d in days], ignore_index=True)
    wake = set(pd.read_parquet(WORK / "wake_paths.parquet", columns=["twapId"]).twapId) | set(pd.read_parquet(WORK / "oos" / "paths.parquet", columns=["twapId"]).twapId)
    D["wake"] = D.twapId.isin(wake)
    fol = top_followers(pd.read_parquet(WORK / "followers.parquet"))
    vol = D.groupby("maker").usd.sum().sort_values(ascending=False); top = list(vol.head(10).index)
    D["category"] = np.select([D.maker.isin(top), D.maker.isin(fol)], ["top 10 makers", "20 followers"], "others")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    me = D.market.map(MY_FEE); x = D.assign(x=D["5 min"] - me)
    m = x.groupby("day").apply(lambda g: wm(g, "x"), include_groups=False).dropna()
    print("All TWAP fills, by month:\n" + pd.DataFrame({k: summary(g) for k, g in D.groupby("month")}).T.round(2).to_string())
    print("\nTotal:\n" + pd.Series(summary(D)).round(2).to_string() + f"\ndaily t (my net at 5 min, 20 days): {m.mean() / m.std() * np.sqrt(len(m)):.2f}")
    print("\nWake set, by month:\n" + pd.DataFrame({k: summary(g) for k, g in D[D.wake].groupby("month")}).T.round(2).to_string())
    part = D.pivot_table(index="category", columns="month", values="usd", aggfunc="sum")
    print("\nShare of TWAP volume absorbed by category, %:\n" + (100 * part / part.sum()).round(1).to_string())
    rank = {w: f"maker #{i + 1}" for i, w in enumerate(vol.index)}
    pm = D[D.maker.isin(vol.head(5).index)].groupby(["maker", "month"]).apply(lambda g: wm(g.assign(x=g["5 min"] - g["fee bp"]), "x"), include_groups=False).unstack().round(1)
    pm.index = [rank[i] for i in pm.index]
    print("\nMaker net at 5 min of the 5 largest makers, by month (bp):\n" + pm.to_string())
    print("\nDistinct makers per month:\n" + D.groupby("month").maker.nunique().to_string())
