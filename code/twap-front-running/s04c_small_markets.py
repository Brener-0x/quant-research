"""Section 4. On markets without an external reference, does following the TWAP pay at prices actually paid?

Pre-registered: TWAPs without an external reference (s04b). Entry: best level of the first taker order on my side
within 60 s after the second slice + 1 s. Exit: best level of the first opposite taker order between entry + 30 and
+ 40 min. Fees: the coin's 90th percentile, both ways. Criterion (each of the two classes that passed in s04b):
net > 0, daily t >= 3, and > 0 without the coin that contributes most.
    python s04c_small_markets.py   -> work/small_orders/<day>.parquet, work/small_markets.parquet
"""
from multiprocessing import Pool
import numpy as np, pandas as pd
from common import DAYS, STABLES, WORK, fills, first_after, show, t_day, work

ORD = work("small_orders")
PASSED = [("100-1,000", "with large"), ("1,000-10,000", "with large")]
ENTRY_WIN, HOLD, EXIT_WIN = 60_000, 1_800_000, 600_000
rng = np.random.default_rng(36)


def coins():
    X = pd.read_parquet(WORK / "controls.parquet", columns=["coin", "class"])
    return sorted(X[X["class"] == "no external reference"].coin.unique())


def one_day(day):
    """One row per taker order: wallet, coin, oid, time, side, best level touched, notional, fee, TWAP child or not."""
    f = ORD / f"{day}.parquet"
    if f.exists():
        return day, -1
    CO = set(coins()); L = []
    for u, e in fills(day):
        if e["crossed"] and e["coin"] in CO:
            px, sz, fee = float(e["px"]), float(e["sz"]), float(e["fee"])
            L.append((u, e["coin"], e["oid"], e["time"], 1 if e["side"] == "B" else -1, px, px * sz,
                      fee if e["feeToken"] in STABLES else fee * px, e.get("twapId") is not None))
    D = pd.DataFrame(L, columns=["user", "coin", "oid", "t", "side", "px", "usd", "fee", "twap"])
    O = D.groupby(["user", "coin", "oid"], sort=False).agg(t=("t", "min"), side=("side", "first"), pmin=("px", "min"), pmax=("px", "max"),
                                                           usd=("usd", "sum"), fee=("fee", "sum"), twap=("twap", "any")).reset_index()
    O["px"] = np.where(O.side > 0, O.pmin, O.pmax)
    O.drop(columns=["pmin", "pmax"]).sort_values("t").to_parquet(f, index=False); return day, len(O)


if __name__ == "__main__":
    with Pool(6) as pool:
        for d, n in pool.imap_unordered(one_day, DAYS):
            print(d, n, flush=True)
    O = pd.concat([pd.read_parquet(ORD / f"{d}.parquet", columns=["coin", "usd", "fee"]) for d in DAYS])
    FEE = (O.fee / O.usd * 1e4).groupby(O.coin).quantile(.9)
    X = pd.read_parquet(WORK / "controls.parquet", columns=["user", "twapId", "coin", "day", "class", "slice size", "past", "t2"])
    T = pd.read_parquet(WORK / "twaps.parquet", columns=["user", "twapId", "net_side"])
    X = X[X["class"] == "no external reference"].merge(T, on=["user", "twapId"])
    L = []
    for d in DAYS:
        Xd = X[X.day == d]
        if len(Xd) == 0:
            continue
        i_ = DAYS.index(d); Od = pd.read_parquet(ORD / f"{d}.parquet")
        if i_ + 1 < len(DAYS):
            Od = pd.concat([Od, pd.read_parquet(ORD / f"{DAYS[i_ + 1]}.parquet")])
        d0 = int(pd.Timestamp(d).value // 1_000_000)
        for coin, G in Xd.groupby("coin"):
            Oc = Od[Od.coin == coin].sort_values("t")
            book = {s: (Oc[Oc.side == s].t.to_numpy(), Oc[Oc.side == s].px.to_numpy()) for s in (1, -1)}
            for who in ("TWAP", "random"):
                if who == "TWAP":
                    te0, sg = G.t2.to_numpy().astype(np.int64) + 1000, G.net_side.to_numpy().astype(int)
                else:
                    te0, sg = d0 + rng.integers(60_000, 86_400_000 - 3_600_000, size=len(G)), rng.choice([-1, 1], size=len(G))
                R = G[["user", "coin", "day", "slice size", "past"]].copy(); R["who"] = who
                pe, te, ps, opp = (np.full(len(G), np.nan) for _ in range(4))
                for s in (1, -1):
                    m = sg == s
                    if not m.any():
                        continue
                    a = book[s]; b = book[-s]
                    te[m], pe[m] = first_after(a[0], a[1], te0[m], te0[m] + ENTRY_WIN)
                    ok = ~np.isnan(te[m]); tes = np.where(ok, te[m], 0).astype(np.int64)
                    _, x = first_after(b[0], b[1], tes + HOLD, tes + HOLD + EXIT_WIN); ps[m] = np.where(ok, x, np.nan)
                    if len(b[0]):
                        i = np.searchsorted(b[0], te0[m], side="right") - 1; i2 = np.clip(i, 0, None)
                        opp[m] = np.where((i >= 0) & (b[0][i2] > te0[m] - ENTRY_WIN), b[1][i2], np.nan)
                R["entered"] = ~np.isnan(pe); R["exited"] = ~np.isnan(ps)
                R["gross"] = sg * np.log(ps / pe) * 1e4; R["fees"] = 2 * FEE.get(coin, np.nan)
                R["net"] = R.gross - R.fees; R["spread at entry"] = sg * np.log(pe / opp) * 1e4
                L.append(R)
    M = pd.concat(L, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    for c in ("gross", "net", "spread at entry"):
        M[c] = M[c].clip(-2000, 2000)
    M.to_parquet(WORK / "small_markets.parquet", index=False)
    show()
    for k in PASSED:
        g = M[(M.who == "TWAP") & (M["slice size"] == k[0]) & (M.past == k[1])]; e = g[g.entered & g.exited]
        pc = e.groupby("coin").agg(n=("net", "size"), net=("net", "mean"), total=("net", "sum")).sort_values("n", ascending=False)
        top = pc.total.idxmax(); without = e[e.coin != top]
        print(f"Class {k}: {len(e)} entered and exited; spread at entry (median) {e['spread at entry'].median():.2f} bp")
        print(pc.head(10).to_string())
        ok = e.net.mean() > 0 and t_day(e, "net") >= 3 and without.net.mean() > 0
        print(f"Criterion: net {e.net.mean():.2f} bp (t {t_day(e, 'net'):.2f}), without {top} {without.net.mean():.2f} bp -> " + ("passed" if ok else "failed") + "\n")
