"""Section 7. Do reasons written in advance remove the losing TWAPs?

Pre-registered. R1a widened global weight (Hyperliquid + Binance perp + Binance spot), threshold 5 % unchanged;
R1b the coin's Binance volume per day (7 days before); R2 backed by a real asset (PAXG, by definition); R3 share of the
coin's TWAPs launched by bots (7 days before); R4 counter-flow of other active TWAPs on the opposite side.
Thresholds chosen on 2025-10 and 2025-11 (middle of a plateau), checked on 2026-02, 2026-04, 2026-05: kept minus
dropped t >= 3 on the three thresholds of the chosen triplet.
    python s07c_reasons.py --download   -> work/klines_spot/<symbol>_<month>.parquet
    python s07c_reasons.py              -> work/reasons.parquet
"""
import re, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
from common import SELECTION, VERIFICATION, WORK, show, t_day, work
from s07a_wallet_frequency import WEEK, t_diff
from s07b_global_weight import KL, MONTHS, kline_file, symbols

SPOT = work("klines_spot")
END_IN_SAMPLE = 1780271999999
SELECT, VERIFY = set(SELECTION), set(VERIFICATION)
BACKED = {"PAXG"}                                   # R2: real-asset-backed tokens, by reading the 170 coins
GRID_Q = [10, 20, 30, 40, 50]; GRID_X = [2, 1, 0.5, 0.25, 0.1]
P = "net s06b"


def spot_of(perp):
    return re.sub(r"^(1000000|1000)", "", perp) if perp else ""


def klines(folder, s, months=MONTHS):
    fs = [folder / f"{s}_{m}.parquet" for m in months] if s else []
    K = pd.concat([pd.read_parquet(f) for f in fs if f.exists()]) if fs else pd.DataFrame({"t": [], "usd": []})
    K = K.sort_values("t"); return K.t.to_numpy().astype(np.int64), np.concatenate([[0.0], np.cumsum(K.usd.to_numpy().astype(float))])


def hour_volume(tt, cu, t1):
    """Volume of the 12 complete 5-minute klines before t1 (0 if incomplete or missing)."""
    if len(tt) == 0:
        return np.zeros(len(t1))
    last = (t1 // 300_000) * 300_000 - 300_000
    hi = np.searchsorted(tt, last, side="right"); lo = np.searchsorted(tt, last - 11 * 300_000, side="left")
    return np.where(hi - lo == 12, cu[hi] - cu[lo], 0.0)


def week_volume(tt, cu, t1):
    """Mean volume per day over the 7 days before t1."""
    if len(tt) == 0:
        return np.zeros(len(t1))
    return (cu[np.searchsorted(tt, t1, side="left")] - cu[np.searchsorted(tt, t1 - WEEK, side="left")]) / 7


def test_reason(B, var, name, thresholds, keep):
    """Choose the triplet on the selection months, check it on the verification months. keep(v, s) -> kept mask."""
    Ch, Ve = B[B.month.isin(SELECT)], B[B.month.isin(VERIFY)]
    rows = []
    for lab, s in thresholds:
        kc, ec = Ch[keep(Ch[var], s)], Ch[~keep(Ch[var], s)]; kv, ev = Ve[keep(Ve[var], s)], Ve[~keep(Ve[var], s)]
        pc = kv.groupby("coin")[P].sum().sort_values(ascending=False)
        rows.append({"threshold": lab, "value": s, "sel kept": len(kc), "sel net kept": kc[P].mean(), "sel net dropped": ec[P].mean(),
                     "sel gap": kc[P].mean() - ec[P].mean(), "ver kept": len(kv), "ver net kept": kv[P].mean(), "ver net dropped": ev[P].mean(),
                     "ver t diff": t_diff(kv, ev, P), "ver kept without top 3": kv[~kv.coin.isin(pc.index[:3])][P].mean()})
    R = pd.DataFrame(rows); mins = [R["sel gap"].iloc[i:i + 3].min() for i in range(len(R) - 2)]; i0 = int(np.nanargmax(mins))
    tri = R.iloc[i0:i0 + 3]
    ok = bool((tri["ver net kept"] > tri["ver net dropped"]).all() and (tri["ver t diff"] >= 3).all() and tri["ver kept without top 3"].iloc[1] > 0)
    print(f"\n=== {name} ===\n{R.to_string(index=False)}\nTriplet chosen on the selection months: {list(tri.threshold)}; kept {tri.threshold.iloc[1]} -> "
          + ("retained" if ok else "rejected"))
    return R, tri.value.iloc[1], ok


if __name__ == "__main__":
    if "--download" in sys.argv:
        S = sorted({spot_of(s) for s in symbols().values() if s})
        with ThreadPoolExecutor(4) as ex:
            list(ex.map(lambda sm: kline_file(SPOT, sm[0], sm[1], END_IN_SAMPLE, spot=True), [(s, m) for s in S for m in MONTHS]))
        sys.exit()
    SY = symbols()
    M = pd.read_parquet(WORK / "global_weight.parquet"); M = M[(M["weight class"] == ">= 20 %") & M["full history"] & (M["frequency 7d"] <= 30)].copy()
    M["month"] = M.day.str[:6]; M["t1"] = M.t1.astype(np.int64)
    T = pd.read_parquet(WORK / "twaps.parquet", columns=["user", "twapId", "coin", "start", "end", "usd", "slices", "net_side"]).sort_values(["user", "start"])
    d = T.start.to_numpy().astype(np.int64); fq = np.zeros(len(T), dtype=np.int64)
    for u, idx in T.groupby("user", sort=False).indices.items():
        x = d[idx]; fq[idx] = np.searchsorted(x, x, side="left") - np.searchsorted(x, x - WEEK, side="left")
    T["bot"] = fq > 30; T["flow 30s"] = T.usd / T.slices.clip(lower=1)
    M = M.merge(T[["user", "twapId", "net_side"]], on=["user", "twapId"], how="left").reset_index(drop=True)
    for c in ["spot volume 1h", "Binance volume per day", "bot share", "counter-flow", "same side"]:
        M[c] = np.nan
    for coin, idx in M.groupby("coin").indices.items():
        t1 = M.t1.to_numpy()[idx]; s = SY.get(coin, "")
        tp, cp = klines(KL, s); ts, cs = klines(SPOT, spot_of(s))
        M.iloc[idx, M.columns.get_loc("spot volume 1h")] = hour_volume(ts, cs, t1)
        M.iloc[idx, M.columns.get_loc("Binance volume per day")] = week_volume(tp, cp, t1) + week_volume(ts, cs, t1)
        G = T[T.coin == coin].sort_values("start"); gd = G.start.to_numpy().astype(np.int64); cr = np.concatenate([[0], np.cumsum(G.bot.to_numpy())])
        lo, hi = np.searchsorted(gd, t1 - WEEK, side="left"), np.searchsorted(gd, t1, side="left")
        M.iloc[idx, M.columns.get_loc("bot share")] = np.where(hi > lo, (cr[hi] - cr[lo]) / np.maximum(hi - lo, 1), np.nan)
        ge, fl, sn, gid = G.end.to_numpy().astype(np.int64), G["flow 30s"].to_numpy(), G.net_side.to_numpy(), G.twapId.to_numpy()
        sg = M.net_side.to_numpy()[idx]; own = M.twapId.to_numpy()[idx]; u1 = M.usd1.to_numpy()[idx]
        opp, same = np.zeros(len(idx)), np.zeros(len(idx))
        for k in range(len(idx)):
            a = (gd < t1[k]) & (ge > t1[k]) & (gid != own[k])
            opp[k] = fl[a & (sn != sg[k])].sum(); same[k] = fl[a & (sn == sg[k])].sum()
        M.iloc[idx, M.columns.get_loc("counter-flow")] = opp / u1; M.iloc[idx, M.columns.get_loc("same side")] = same / u1
    M["widened weight %"] = 100 * M.usd1 / ((M["HL volume 1h"] + M["Binance volume 1h"] + M["spot volume 1h"]) / 120)
    M.to_parquet(WORK / "reasons.parquet", index=False)
    show()
    B = M[M["global weight %"] >= 5]
    print(f"Set after the global weight (perp) >= 5 %: {len(B):,} TWAPs, {B[P].mean():.2f} bp (t {t_day(B, P):.2f})")
    # R1a
    removed = M[(M["global weight %"] >= 5) & (M["widened weight %"] < 5)]; kept = M[M["widened weight %"] >= 5]
    r1a = {}
    for name, sel in (("selection", SELECT), ("verification", VERIFY)):
        a, b = kept[kept.month.isin(sel)], removed[removed.month.isin(sel)]
        r1a[name] = (len(a), a[P].mean(), len(b), b[P].mean(), t_diff(a, b, P))
    ok1a = r1a["verification"][1] > r1a["verification"][3] and r1a["verification"][4] >= 3
    print("\n=== R1a: widened global weight (spot included), threshold 5 % ===")
    for k, v in r1a.items():
        print(f"{k}: kept {v[0]:,} ({v[1]:.2f}), removed by spot {v[2]:,} ({v[3]:.2f}), t diff {v[4]:.2f}")
    print("R1a -> " + ("retained" if ok1a else "rejected"))
    print(f"\n=== R2: real-asset-backed ({', '.join(BACKED)}), by definition: {B.coin.isin(BACKED).sum()} TWAPs in the set ===")
    Ch = B[B.month.isin(SELECT)]
    _, s1b, ok1b = test_reason(B, "Binance volume per day", "R1b: the coin's Binance volume per day (7 days before), the largest are dropped",
                               [(f"{q} %", Ch["Binance volume per day"].quantile(1 - q / 100)) for q in GRID_Q], lambda v, s: v < s)
    _, s3, ok3 = test_reason(B, "bot share", "R3: share of the coin's TWAPs sent by bots (7 days before), the most bot-heavy are dropped",
                             [(f"{q} %", Ch["bot share"].quantile(1 - q / 100)) for q in GRID_Q], lambda v, s: ~(v >= s))
    _, s4, ok4 = test_reason(B, "counter-flow", "R4: counter-flow of other active TWAPs (/ my first slice), the largest are dropped",
                             [(f">= {x:g}", x) for x in GRID_X], lambda v, s: v < s)
    F = M[M["widened weight %"] >= 5] if ok1a else B.copy()
    F = F[~F.coin.isin(BACKED)]
    if ok1b:
        F = F[F["Binance volume per day"] < s1b]
        print(f"\nR1b threshold: {s1b:,.0f} dollars a day")
    if ok3:
        F = F[~(F["bot share"] >= s3)]
    if ok4:
        F = F[F["counter-flow"] < s4]
    print(f"\nAfter the retained reasons: {len(F):,} TWAPs, {F[P].mean():.2f} bp (t {t_day(F, P):.2f}), median {F[P].median():.2f}")
    for name, sel in (("selection", SELECT), ("verification", VERIFY)):
        f = F[F.month.isin(sel)]; print(f"  {name}: {f[P].mean():.2f} bp (t {t_day(f, P):.2f}, {len(f):,} TWAPs)")
