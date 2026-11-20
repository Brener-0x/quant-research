"""Section 4. After the first slice of a large TWAP, does the price drift its way?

Pre-registered: TWAPs >= 100,000 dollars; price = median of non-TWAP taker trades per 10-second bar; entry at the
first fill + 1 s; horizons 5, 15, 30, 60 min; control: same coin, same day, random time and side.
Criterion per class (external reference / none): >= 9 bp at 30 min, daily t >= 3, control within 2 bp of zero.
Size and weight are known only after the fact here; Section 4 (s04b) removes that bias.
    python s04a_drift.py   -> work/bars10s/<day>.parquet, work/drift.parquet
"""
from multiprocessing import Pool
import numpy as np, pandas as pd
from common import DAYS, WORK, fills, price_at, series2, show, t_day, work

BARS = work("bars10s")
HORIZONS = {"5 min": 300_000, "15 min": 900_000, "30 min": 1_800_000, "60 min": 3_600_000}
LARGE = 100_000
rng = np.random.default_rng(34)


def one_day(day):
    f = BARS / f"{day}.parquet"
    if f.exists():
        return day, -1
    co, tt, pp, uu = [], [], [], []
    for _, e in fills(day):
        if e["crossed"] and e.get("twapId") is None:
            co.append(e["coin"]); tt.append(e["time"]); px = float(e["px"]); pp.append(px); uu.append(px * float(e["sz"]))
    D = pd.DataFrame({"coin": pd.Categorical(co), "t10": np.array(tt) // 10_000 * 10_000, "px": pp, "usd": uu})
    B = D.groupby(["coin", "t10"], observed=True).agg(px=("px", "median"), n=("px", "size"), usd=("usd", "sum")).reset_index()
    B.to_parquet(f, index=False); return day, len(B)


if __name__ == "__main__":
    with Pool(6) as pool:
        for d, n in pool.imap_unordered(one_day, DAYS):
            print(d, n, flush=True)
    T = pd.read_parquet(WORK / "twaps.parquet"); T = T[T.usd >= LARGE].reset_index(drop=True)
    T["class"] = np.where(T.reference == "external reference", "external reference", "no external reference")
    L = []
    for d in DAYS:
        Td = T[T.day0 == d]
        if len(Td) == 0:
            continue
        S = series2(d); d0 = int(pd.Timestamp(d).value // 1_000_000)
        for coin, G in Td.groupby("coin"):
            s = S.get(coin)
            if s is None:
                continue
            for who in ("TWAP", "random"):
                if who == "TWAP":
                    t1, sg = G.start.to_numpy() + 1000, G.net_side.to_numpy()
                else:
                    t1, sg = d0 + rng.integers(60_000, 86_400_000 - 3_600_000, size=len(G)), rng.choice([-1, 1], size=len(G))
                p0 = price_at(s, t1)
                D = pd.DataFrame({"who": who, "day": d, "coin": coin, "class": G["class"].to_numpy(), "usd": G.usd.to_numpy(),
                                  "share": G["volume share %"].to_numpy(), "user": G.user.to_numpy(), "sg": sg})
                for h, dt in HORIZONS.items():
                    D[h] = sg * np.log(price_at(s, t1 + dt) / p0) * 1e4
                    D[f"running {h}"] = G.end.to_numpy() > t1 + dt
                if who == "TWAP":
                    D["to its end"] = sg * np.log(price_at(s, G.end.to_numpy()) / p0) * 1e4
                L.append(D)
    X = pd.concat(L, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    hs = list(HORIZONS) + ["to its end"]; X[hs] = X[hs].clip(-1000, 1000)
    X["size"] = pd.cut(X.usd, [LARGE, 1e6, 1e7, np.inf], labels=["100k-1M", "1-10M", ">=10M"], right=False)
    X["weight"] = pd.qcut(X.share, 4, labels=["Q1 (light)", "Q2", "Q3", "Q4 (heavy)"])
    X.to_parquet(WORK / "drift.parquet", index=False)
    show(); Y = X[X.who == "TWAP"]

    def table(Z, by):
        R = []
        for k, g in Z.groupby(by, observed=True):
            r = {"n": len(g)} | {h: g[h].mean() for h in hs} | {"t 30 min": t_day(g, "30 min")}
            R.append(pd.Series(r, name=k))
        return pd.DataFrame(R)
    print("By class (bp in the TWAP's direction):\n", table(Y, "class").to_string())
    print("\nRandom control:\n", table(X[X.who == "random"].assign(**{"to its end": np.nan}), "class")[["n"] + list(HORIZONS) + ["t 30 min"]].to_string())
    print("\nBy class and weight (known after the fact):\n", table(Y, ["class", "weight"]).to_string())
    for cl in ("external reference", "no external reference"):
        g, w = Y[Y["class"] == cl], X[(X.who == "random") & (X["class"] == cl)]
        m, t, tm = g["30 min"].mean(), t_day(g, "30 min"), w["30 min"].mean()
        print(f"Criterion, {cl}: {m:.2f} bp at 30 min (>= 9), t {t:.2f} (>= 3), control {tm:.2f} (|.| < 2) -> "
              + ("passed" if m >= 9 and t >= 3 and abs(tm) < 2 else "not met"))
