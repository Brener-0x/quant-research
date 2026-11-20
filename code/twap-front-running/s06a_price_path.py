"""Section 6. After entering on a heavy TWAP, when does the price peak?

Method fixed before the computation: TWAPs of s05b (referenced coins, weight of the first slice) in the 5-20 % and
>= 20 % classes; non-TWAP prices of s04a. Path every 30 s up to + 120 min from the first and second slice + 1 s;
MFE, MAE, time of the MFE; moves measured in units of the TWAP's own duration.
    python s06a_price_path.py   -> work/paths.npz, work/path_summary.parquet
"""
import numpy as np, pandas as pd
from common import DAYS, WORK, coin_type, price_at, series2, show, t_day
from s05b_heavy_executable import universe

GRID = np.arange(0, 241) * 30_000
AFTER_END = {"end": 0, "end + 5 min": 300_000, "end + 15 min": 900_000, "end + 30 min": 1_800_000, "end + 60 min": 3_600_000}
FRACS = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3]

if __name__ == "__main__":
    X = universe(); X = X[X["weight class"].isin(["5-20 %", ">= 20 %"])]
    X = X.merge(pd.read_parquet(WORK / "twaps.parquet", columns=["user", "twapId", "end"]), on=["user", "twapId"]).reset_index(drop=True)
    X["type"] = X.coin.map(coin_type); X["minutes"] = (X.end - X.t1) / 60_000
    X["duration"] = pd.cut(X.minutes, [-1, 10, 30, 120, np.inf], labels=["<= 10 min", "10-30", "30-120", "> 120"])
    TR = {e: np.full((len(X), len(GRID)), np.nan, dtype=np.float32) for e in ("1st", "2nd")}
    AF = {k: np.full(len(X), np.nan) for k in AFTER_END}; FR = {f: np.full(len(X), np.nan) for f in FRACS}
    for d in DAYS:
        idx = np.flatnonzero(X.day.to_numpy() == d)
        if len(idx) == 0:
            continue
        S = series2(d)
        for c, sub in X.iloc[idx].groupby("coin"):
            s = S.get(c)
            if s is None:
                continue
            ii = sub.index.to_numpy(); sg = sub.net_side.to_numpy()[:, None]
            for e, col in (("1st", "t1"), ("2nd", "t2")):
                te = sub[col].to_numpy().astype(np.int64) + 1000
                p0 = price_at(s, te)[:, None]
                pt = price_at(s, (te[:, None] + GRID[None, :]).ravel()).reshape(len(te), -1)
                TR[e][ii] = (sg * np.log(pt / p0) * 1e4).astype(np.float32)
            te = sub.t1.to_numpy().astype(np.int64) + 1000; p0 = price_at(s, te); end = sub.end.to_numpy().astype(np.int64)
            for k, dt in AFTER_END.items():
                AF[k][ii] = sub.net_side.to_numpy() * np.log(price_at(s, end + dt) / p0) * 1e4
            for f in FRACS:
                FR[f][ii] = sub.net_side.to_numpy() * np.log(price_at(s, (te + f * (end - te)).astype(np.int64)) / p0) * 1e4
    for e in TR:
        TR[e] = np.clip(TR[e], -1000, 1000)
    np.savez_compressed(WORK / "paths.npz", **{f"path_{e}": v for e, v in TR.items()})
    R = X[["user", "twapId", "coin", "day", "weight class", "type", "duration", "minutes", "net_side"]].copy()
    for e in TR:
        v = np.where(np.isnan(TR[e][:, 1:]), -np.inf, TR[e][:, 1:]); w = np.where(np.isnan(TR[e][:, 1:]), np.inf, TR[e][:, 1:])
        R[f"MFE {e}"] = np.where(np.isfinite(v.max(1)), v.max(1), np.nan); R[f"MAE {e}"] = np.where(np.isfinite(w.min(1)), w.min(1), np.nan)
        R[f"MFE time {e} (min)"] = np.where(np.isfinite(v.max(1)), (v.argmax(1) + 1) / 2, np.nan)
    for k in AF:
        R[k] = np.clip(AF[k], -1000, 1000)
    for f in FRACS:
        R[f"{f:g} x duration"] = np.clip(FR[f], -1000, 1000)
    R.to_parquet(WORK / "path_summary.parquet", index=False)
    show()
    cols = list(AFTER_END) + [f"{f:g} x duration" for f in FRACS]
    print("From the first slice + 1 s, against the observed end of the TWAP (mean, bp):\n", R.groupby(["weight class", "type"], observed=True)[cols].mean().to_string())
    g = R[(R["weight class"] == ">= 20 %") & (R.type == "perp")]
    print("\n... by duration (weight >= 20 %, perps):\n", g.groupby("duration", observed=True)[["minutes"] + cols].mean().assign(n=g.groupby("duration", observed=True).size()).to_string())
    print(f"\nWeight >= 20 %, perps: MFE before the end of the TWAP in {(g['MFE time 1st (min)'] <= g['minutes']).mean():.1%} of cases; "
          f"daily t at the end {t_day(g, 'end'):.2f}")
