"""Shared paths, calendar and helpers.

Environment variables:
  HL_FILLS   folder of Hyperliquid node fills, one sub-folder per day (YYYYMMDD) holding the hourly
             node_fills_by_block files (*.lz4) as published in the hl-mainnet-node-data bucket.
  TWAP_WORK  folder for every derived table (default: ./work next to this file).
"""
import json, os
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RAW = Path(os.environ.get("HL_FILLS", HERE / "raw"))
WORK = Path(os.environ.get("TWAP_WORK", HERE / "work"))

IN_SAMPLE = ["202510", "202511", "202602", "202604", "202605"]
SELECTION, VERIFICATION = ["202510", "202511"], ["202602", "202604", "202605"]
OUT_OF_SAMPLE = ["202606", "202607", "202608", "202609", "202610"]
STABLES = {"USDC", "USDH", "USDT0", "USDE", "USDT", "USD"}
TAKER_FEE = 4.5                       # bp per side, perps, base tier
AGE = 300_000                         # a price older than 5 minutes is not used


def work(*parts):
    p = WORK.joinpath(*parts); p.mkdir(parents=True, exist_ok=True); return p


def _days(months, bars):
    if RAW.exists():
        d = sorted(p.name for p in RAW.iterdir() if p.is_dir() and p.name[:6] in months)
        if d:
            return d
    return sorted(f.stem for f in (WORK / bars).glob("*.parquet") if f.stem[:6] in months)


DAYS = _days(IN_SAMPLE, "bars5s")                   # 150 in-sample days
OOS_DAYS = _days(OUT_OF_SAMPLE, "oos/twap_fills")             # 153 out-of-sample days


def fills(day):
    """Every fill of one day, as (wallet, event) pairs, in file order."""
    import lz4.frame
    for f in sorted((RAW / day).glob("*.lz4"), key=lambda p: int(p.stem)):
        with lz4.frame.open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                for u, e in json.loads(line)["events"]:
                    yield u, e


def coin_reference():
    return pd.read_csv(DATA / "coin_reference.csv", keep_default_na=False)


def coin_type(c):
    return "HIP-3" if ":" in c else ("spot" if c.startswith("@") or "/" in c else "perp")


def next_day(d):
    return (pd.Timestamp(d) + pd.Timedelta(days=1)).strftime("%Y%m%d")


def prev_day(d):
    return (pd.Timestamp(d) - pd.Timedelta(days=1)).strftime("%Y%m%d")


def day_start_ms(d):
    return int(pd.Timestamp(d).value // 1_000_000)


def t_day(d, c):
    """t statistic of column c, with the day as the unit of inference."""
    m = d.groupby("day")[c].mean().dropna()
    return m.mean() / m.std() * np.sqrt(len(m)) if len(m) > 2 else np.nan


def price_at(S, t):
    """Last 10-second bar price at or before t, if it is at most 5 minutes old."""
    tf, px = S; i = np.searchsorted(tf, t, side="right") - 1; ok = i >= 0; i = np.clip(i, 0, None)
    return np.where(ok & (t - tf[i] <= AGE), px[i], np.nan)


def series(day, folder="bars10s"):
    """Non-TWAP taker price per coin: (bar end times, median price) from the 10-second bars of one day."""
    f = WORK / folder / f"{day}.parquet"
    if not f.exists():
        return {}
    B = pd.read_parquet(f); B["coin"] = B.coin.astype(str)
    return {c: (g.t10.to_numpy() + 10_000, g.px.to_numpy()) for c, g in B.sort_values("t10").groupby("coin")}


def series2(day, days=None, folder="bars10s"):
    """Same, with the next listed day appended (positions that run past midnight)."""
    days = DAYS if days is None else days
    S = series(day, folder); i = days.index(day); S2 = series(days[i + 1], folder) if i + 1 < len(days) else {}
    return {c: (np.concatenate([s[0], S2[c][0]]), np.concatenate([s[1], S2[c][1]])) if c in S2 else s for c, s in S.items()}


def first_after(tt, px, a, b):
    """First order in (a, b]: returns (time, price) or NaN."""
    if len(tt) == 0:
        return np.full(np.shape(a), np.nan), np.full(np.shape(a), np.nan)
    i = np.searchsorted(tt, a, side="right"); ok = (i < len(tt))
    i2 = np.clip(i, 0, len(tt) - 1); ok &= tt[i2] <= b
    return np.where(ok, tt[i2], np.nan), np.where(ok, px[i2], np.nan)


def show():
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
    pd.options.display.float_format = "{:.2f}".format
