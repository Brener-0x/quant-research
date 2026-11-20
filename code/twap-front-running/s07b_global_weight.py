"""Section 7. Does the TWAP's weight in the coin's global volume sort the winners better?

Pre-registered. Global weight = first slice / ((Hyperliquid volume + Binance USDT perp volume over the previous hour)
/ 120). Binance: free public monthly 5-minute klines (data.binance.vision), in-sample months only.
Criterion (perps, Hyperliquid weight >= 20 %): for S = 2, 5, 10 %, kept (global weight >= S) net > all, t >= 3,
kept minus dropped t >= 3, kept > 0 without the top 3 coins.
    python s07b_global_weight.py --download   -> work/klines_perp/<symbol>_<month>.parquet
    python s07b_global_weight.py              -> work/global_weight.parquet
"""
import io, sys, time, urllib.request, urllib.error, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
from common import WORK, coin_reference, show, t_day, work
from s07a_wallet_frequency import t_diff

KL = work("klines_perp")
MONTHS = ["2025-10", "2025-11", "2026-02", "2026-04", "2026-05"]          # never past 2026-05-31
THRESHOLDS = [1, 2, 5, 10, 20]; PLATEAU = [2, 5, 10]
SHARES = [0, .1, .25, .5, .75, 1.0001]; SHARE_NAMES = ["< 10 %", "10-25 %", "25-50 %", "50-75 %", ">= 75 %"]
P = "net s06b"


def symbols():
    U = coin_reference().set_index("coin").binance
    co = pd.read_parquet(WORK / "wallet_frequency.parquet", columns=["coin", "type"]).query("type == 'perp'").coin.unique()
    return {c: str(U.get(c, "") or "").replace("partial:", "") for c in co}


def kline_file(folder, s, m, end_ms, spot=False):
    """One monthly file of 5-minute klines: open time and quote volume (dollars)."""
    f = folder / f"{s}_{m}.parquet"
    if f.exists():
        return 0
    url = (f"https://data.binance.vision/data/spot/monthly/klines/{s}/5m/{s}-5m-{m}.zip" if spot
           else f"https://data.binance.vision/data/futures/um/monthly/klines/{s}/5m/{s}-5m-{m}.zip")
    for k in range(4):
        try:
            raw = urllib.request.urlopen(url, timeout=120).read(); break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                pd.DataFrame({"t": pd.Series(dtype="int64"), "usd": pd.Series(dtype="float64")}).to_parquet(f, index=False); return 0
            time.sleep(3 * (k + 1))
        except Exception:
            time.sleep(3 * (k + 1))
    else:
        return 0
    z = zipfile.ZipFile(io.BytesIO(raw)); D = pd.read_csv(z.open(z.namelist()[0]), header=None)
    if not str(D.iloc[0, 0]).isdigit():
        D = D.iloc[1:]
    t = D[0].astype("int64").to_numpy()
    if spot:
        t = np.where(t > 10 ** 14, t // 1000, t)                          # spot open times are in microseconds since 2025
    D = pd.DataFrame({"t": t, "usd": D[7].astype(float).to_numpy()}); D = D[D.t <= end_ms]
    D.to_parquet(f, index=False); return 1


if __name__ == "__main__":
    if "--download" in sys.argv:
        S = sorted({s for s in symbols().values() if s})
        with ThreadPoolExecutor(4) as ex:
            list(ex.map(lambda sm: kline_file(KL, sm[0], sm[1], 1780271999999), [(s, m) for s in S for m in MONTHS]))
        sys.exit()
    SY = symbols()
    M = pd.read_parquet(WORK / "wallet_frequency.parquet"); M = M[M.type == "perp"].copy()
    V = pd.read_parquet(WORK / "weight.parquet", columns=["user", "twapId", "t1", "usd1", "vol30"])
    M = M.merge(V, on=["user", "twapId"], how="left")
    M["HL volume 1h"] = M.vol30 * 120; M["Binance volume 1h"] = np.nan; M["no Binance"] = True
    for coin, idx in M.groupby("coin").indices.items():
        s = SY.get(coin, "")
        fs = [KL / f"{s}_{m}.parquet" for m in MONTHS] if s else []
        K = pd.concat([pd.read_parquet(f) for f in fs if f.exists()]) if fs else pd.DataFrame(columns=["t", "usd"])
        t1 = M.t1.to_numpy()[idx].astype(np.int64)
        if len(K) == 0:
            M.iloc[idx, M.columns.get_loc("Binance volume 1h")] = 0.0; continue
        K = K.sort_values("t"); tt = K.t.to_numpy(); cu = np.concatenate([[0.0], np.cumsum(K.usd.to_numpy())])
        last = (t1 // 300_000) * 300_000 - 300_000                         # open time of the last complete 5-minute kline
        hi = np.searchsorted(tt, last, side="right"); lo = np.searchsorted(tt, last - 11 * 300_000, side="left")
        n = hi - lo
        M.iloc[idx, M.columns.get_loc("Binance volume 1h")] = np.where(n == 12, cu[hi] - cu[lo], 0.0)
        M.iloc[idx, M.columns.get_loc("no Binance")] = n != 12
    M["HL share"] = M["HL volume 1h"] / (M["HL volume 1h"] + M["Binance volume 1h"])
    M["global weight %"] = 100 * M.usd1 / ((M["HL volume 1h"] + M["Binance volume 1h"]) / 120)
    M["HL share class"] = pd.cut(M["HL share"], SHARES, labels=SHARE_NAMES, right=False)
    M.to_parquet(WORK / "global_weight.parquet", index=False)
    show()
    g = M[M["weight class"] == ">= 20 %"]
    print(f"Perps, HL weight >= 20 %: {len(g):,} TWAPs; median HL share of (HL + Binance perp) volume {g['HL share'].median():.3f}")
    print("\nBy HL share:\n", g.groupby("HL share class", observed=True).agg(n=("twapId", "size"), net=(P, "mean"),
          t=(P, lambda x: t_day(g.loc[x.index], P)), coins=("coin", "nunique")).to_string())
    rows = []
    for S in THRESHOLDS:
        k, e = g[g["global weight %"] >= S], g[g["global weight %"] < S]
        pc = k.groupby("coin")[P].sum().sort_values(ascending=False)
        rows.append({"S %": S, "kept": len(k), "net kept": k[P].mean(), "t kept": t_day(k, P), "net dropped": e[P].mean(),
                     "t diff": t_diff(k, e, P), "kept without top 3": k[~k.coin.isin(pc.index[:3])][P].mean()})
    R = pd.DataFrame(rows).set_index("S %"); print("\nBy threshold S (kept = global weight >= S):\n", R.to_string())
    f = g[g["full history"] & (g["frequency 7d"] <= 30)]; fk = f[f["global weight %"] >= 5]
    print(f"\nWith the wallet filter (<= 30 TWAPs in 7 days) and global weight >= 5 %: {len(fk):,} TWAPs, {fk[P].mean():.2f} bp (t {t_day(fk, P):.2f})")
    ok = all(R.loc[S, "net kept"] > g[P].mean() and R.loc[S, "t kept"] >= 3 and R.loc[S, "t diff"] >= 3 and R.loc[S, "kept without top 3"] > 0 for S in PLATEAU)
    print("\nCriterion (S = 2, 5, 10 %): " + ("passed" if ok else "failed"))
