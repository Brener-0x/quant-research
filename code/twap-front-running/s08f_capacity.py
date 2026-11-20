"""Section 8. How often the rule fires and how many positions run at the same time (core set, 150 in-sample days).
    python s08f_capacity.py
"""
import numpy as np, pandas as pd
from common import WORK

if __name__ == "__main__":
    X = pd.read_parquet(WORK / "core_set.parquet", columns=["user", "twapId", "coin", "day", "month", "usd1", "vol30", "net s06b"])
    X = X.merge(pd.read_parquet(WORK / "twaps.parquet", columns=["user", "twapId", "start", "end", "usd"]), on=["user", "twapId"], how="left")
    q = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    print(f"{len(X):,} TWAPs, {X.day.nunique()} days, {X.coin.nunique()} coins, {X.user.nunique()} wallets")
    print("Executed size of the TWAP (dollars):", X.usd.quantile(q).round(0).to_dict())
    print("First slice (dollars):", X.usd1.quantile(q).round(0).to_dict())
    per_day = X.groupby("day").size()
    print(f"Signals per day: mean {per_day.mean():.1f}; quantiles {per_day.quantile([0, 0.1, 0.5, 0.9, 1]).to_dict()}")
    ev = np.concatenate([np.c_[X.start.values, np.ones(len(X))], np.c_[X.end.values, -np.ones(len(X))]])
    ev = ev[np.lexsort((ev[:, 1], ev[:, 0]))]
    open_ = np.cumsum(ev[:, 1])
    grid = np.arange(X.start.min(), X.end.max(), 60_000)
    running = open_[np.searchsorted(ev[:, 0], grid, side="right") - 1]
    active = pd.to_datetime(grid, unit="ms").strftime("%Y%m%d").isin(X.day.unique())
    r = running[active]
    print("TWAPs running at the same time (minutes of active days):", pd.Series(r).quantile([0.5, 0.9, 0.99, 1]).to_dict(), f"; share of minutes with at least one {np.mean(r > 0):.1%}")
