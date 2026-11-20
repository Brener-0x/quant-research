"""Section 2. 5-second taker volume bars, used for the weight of a TWAP in Hyperliquid volume.

In sample the bars cover a fixed list of coins (data/bar_coins.csv) inherited from an earlier study; with
data/coin_reference.csv this gives the 170 in-sample perps with an external reference. One record per trade (taker side).
    python s00_volume_bars.py   -> work/bars5s/<day>.parquet  (coin, t5, px, notional, n)
"""
from array import array
from multiprocessing import Pool
import numpy as np, pandas as pd
from common import DATA, DAYS, fills, work

OUT = work("bars5s")
BAR = 5_000
COINS = set(pd.read_csv(DATA / "bar_coins.csv").coin)


def one_day(day):
    f = OUT / f"{day}.parquet"
    if f.exists():
        return day, -1
    C, T, P, N = [], array("q"), array("d"), array("d")
    for _, e in fills(day):
        if e["crossed"] and e["coin"] in COINS:
            px = float(e["px"]); C.append(e["coin"]); T.append(e["time"]); P.append(px); N.append(px * float(e["sz"]))
    M = pd.DataFrame({"coin": pd.Categorical(C), "t": np.frombuffer(T, dtype=np.int64), "px": np.frombuffer(P), "notional": np.frombuffer(N)})
    M["t5"] = M.t // BAR * BAR
    M = M.sort_values(["coin", "t"], kind="stable")
    B = M.groupby(["coin", "t5"], observed=True, sort=False).agg(px=("px", "last"), notional=("notional", "sum"), n=("px", "size")).reset_index()
    B.to_parquet(f, index=False); return day, len(B)


if __name__ == "__main__":
    with Pool(5) as pool:
        for d, n in pool.imap_unordered(one_day, DAYS):
            print(d, n, flush=True)
