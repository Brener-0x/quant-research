"""Section 9. ONE read of the raw fills of 2026-06 to 2026-10 (never read before the model was frozen). It produces the
same tables the model uses in sample, in a separate folder (work/oos/):
  - twap_fills/<day> : every fill with a twapId (as work/twap_fills);
  - orders/<day>     : perp taker orders, one per (wallet, coin, oid), price = best level touched (as work/orders);
  - bars5s/<day>     : 5-second taker bars of all perps, one record per trade (as work/bars5s).
No result is computed here, only row counts.
    python s09a_oos_read.py [--workers N]
"""
import argparse
from array import array
from multiprocessing import Pool
import numpy as np, pandas as pd
from common import OUT_OF_SAMPLE, RAW, STABLES, coin_type, fills, work

OUT = work("oos")
for t in ("twap_fills", "orders", "bars5s"):
    work("oos", t)


def one_day(day):
    assert day[:6] in OUT_OF_SAMPLE
    if all((OUT / t / f"{day}.parquet").exists() for t in ("twap_fills", "orders", "bars5s")):
        return day, -1, -1, -1
    TW = []
    us, cs, oid, tt, sd = [], [], array("q"), array("q"), array("b"); P, U, F = array("d"), array("d"), array("d")
    for u, e in fills(day):
        c = e["coin"]
        if e.get("twapId") is not None:
            TW.append((u, e["twapId"], c, e["time"], float(e["px"]), float(e["sz"]), 1 if e["side"] == "B" else -1,
                       bool(e["crossed"]), e.get("dir", ""), float(e["fee"]), e["feeToken"], e["oid"]))
        if e["crossed"] and coin_type(c) == "perp":
            p, sz, fee = float(e["px"]), float(e["sz"]), float(e["fee"])
            us.append(u); cs.append(c); oid.append(e["oid"]); tt.append(e["time"]); sd.append(1 if e["side"] == "B" else -1)
            P.append(p); U.append(p * sz); F.append(fee if e["feeToken"] in STABLES else fee * p)
    D = pd.DataFrame(TW, columns=["user", "twapId", "coin", "t", "px", "sz", "side", "taker", "dir", "fee", "feeToken", "oid"]); del TW
    D["usd"] = D.px * D.sz; D["fee_usd"] = np.where(D.feeToken == "USDC", D.fee, np.nan)
    D.drop(columns=["fee", "feeToken"]).to_parquet(OUT / "twap_fills" / f"{day}.parquet", index=False); ntw = len(D); del D
    X = pd.DataFrame({"user": pd.Categorical(us), "coin": pd.Categorical(cs), "oid": np.frombuffer(oid, dtype=np.int64), "t": np.frombuffer(tt, dtype=np.int64),
                      "side": np.frombuffer(sd, dtype=np.int8), "px": np.frombuffer(P), "usd": np.frombuffer(U), "fee": np.frombuffer(F)}); del us, cs
    X["t5"] = X.t // 5000 * 5000
    B = X.sort_values(["coin", "t"], kind="stable").groupby(["coin", "t5"], observed=True, sort=False).agg(px=("px", "last"), notional=("usd", "sum"), n=("px", "size")).reset_index()
    B.to_parquet(OUT / "bars5s" / f"{day}.parquet", index=False); nb = len(B); del B
    O = X.groupby(["user", "coin", "oid"], sort=False, observed=True).agg(t=("t", "min"), side=("side", "first"), pmin=("px", "min"), pmax=("px", "max"),
                                                                         usd=("usd", "sum"), fee=("fee", "sum")).reset_index()
    O["px"] = np.where(O.side > 0, O.pmin, O.pmax)
    O.drop(columns=["pmin", "pmax"]).sort_values("t").to_parquet(OUT / "orders" / f"{day}.parquet", index=False)
    return day, ntw, len(O), nb


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=1); a = ap.parse_args()
    days = sorted(d.name for d in RAW.iterdir() if d.is_dir() and d.name[:6] in OUT_OF_SAMPLE)
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(one_day, days):
            print("{}: {} TWAP fills, {} orders, {} bars".format(*r), flush=True)
