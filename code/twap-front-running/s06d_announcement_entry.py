"""Section 6. Does entering at the announcement, before the first slice, add to the gain of s06b?

Pre-registered. Main entry: best level touched by the first slice; lower bound: its worst level. Exit as in s06b: first
opposite taker order within 10 min after the end. Base fee schedule. Compared one to one with s06b.
Competition: taker orders of other wallets on my side between the announcement (API, s06c) + 0.2 s and the first slice,
against random windows of the same length.
Criterion (perps, weight >= 20 %): net > 0, daily t >= 3, gain over s06b > 0 with t >= 3, > 0 without the top 3 coins.
    python s06d_announcement_entry.py   -> work/announcement_entry.parquet
"""
import json
import numpy as np, pandas as pd
from common import DAYS, WORK, coin_type, first_after, show, t_day
from s05b_heavy_executable import universe
from s06b_maker_exit import SCHEDULE

EXIT_WIN, BEFORE, DELAY = 600_000, 60_000, 200
DURATIONS = [0, 10, 30, 120, np.inf]; DURATION_NAMES = ["<= 10 min", "10-30 min", "30-120 min", "> 120 min"]
rng = np.random.default_rng(43)


def last_before(tt, px, t):
    """Price of the last order before t (at most 60 s old), else NaN."""
    k = np.searchsorted(tt, t, side="left") - 1
    ok = (k >= 0) & (t - tt[np.maximum(k, 0)] <= BEFORE)
    return np.where(ok, px[np.maximum(k, 0)], np.nan)


if __name__ == "__main__":
    X = universe().merge(pd.read_parquet(WORK / "twaps.parquet", columns=["user", "twapId", "end"]), on=["user", "twapId"])
    co = set(json.loads((WORK / "orders_coins.json").read_text())); X = X[X.coin.isin(co) & X["weight class"].isin(["5-20 %", ">= 20 %"])].reset_index(drop=True)
    X["type"] = X.coin.map(coin_type)
    mx = pd.read_parquet(WORK / "maker_exit.parquet", columns=["user", "twapId", "control", "entered", "taker at end"])
    mx = mx[mx.control != True].drop(columns=["control"]).rename(columns={"taker at end": "net s06b", "entered": "entered s06b"})
    X = X.merge(mx, on=["user", "twapId"], how="left")
    X = X.merge(pd.read_parquet(WORK / "announcements.parquet", columns=["user", "twapId", "t_sent"]), on=["user", "twapId"], how="left")
    f39 = pd.read_parquet(WORK / "heavy_executable.parquet", columns=["coin", "fees"]).groupby("coin").fees.first() / 2
    fees = {c: SCHEDULE.get(coin_type(c), (f39.get(c, 4.5), f39.get(c, 4.5) / 3))[0] for c in X.coin.unique()}
    L = []
    for d in DAYS:
        Xd = X[X.day == d]
        if len(Xd) == 0:
            continue
        i_ = DAYS.index(d); dd_ = [d] + ([DAYS[i_ + 1]] if i_ + 1 < len(DAYS) else [])
        Od = pd.concat([pd.read_parquet(WORK / "orders" / f"{x}.parquet", columns=["user", "coin", "t", "side", "px"]) for x in dd_])
        Od = Od[Od.coin.isin(set(Xd.coin))]; Od["user"] = Od.user.astype(str)
        Tw = pd.read_parquet(WORK / "twap_fills" / f"{d}.parquet", columns=["user", "twapId", "t", "px"])
        Tw = Tw[Tw.twapId.isin(set(Xd.twapId))].merge(Xd[["user", "twapId", "t1"]], on=["user", "twapId"])
        T1 = Tw[Tw.t == Tw.t1].groupby(["user", "twapId"]).px.agg(["min", "max"]).reset_index()
        Xd = Xd.merge(T1, on=["user", "twapId"], how="left")
        d0 = int(pd.Timestamp(d).value // 1_000_000)
        for coin, G in Xd.groupby("coin"):
            Oc = Od[Od.coin == coin].sort_values("t")
            book = {s: (Oc[Oc.side == s].t.to_numpy(), Oc[Oc.side == s].px.to_numpy(), Oc[Oc.side == s].user.to_numpy()) for s in (1, -1)}
            fp = fees[coin]
            R = G[["user", "twapId", "coin", "day", "type", "weight class", "weight1 %", "net s06b", "entered s06b", "t_sent"]].copy()
            sg = G.net_side.to_numpy().astype(int); t1 = G.t1.to_numpy().astype(np.int64); end = G.end.to_numpy().astype(np.int64)
            best = np.where(sg > 0, G["min"], G["max"]); worst = np.where(sg > 0, G["max"], G["min"])
            ps, pre, ncomp, nctl, mctl = (np.full(len(G), np.nan) for _ in range(5))
            ts_ = G.t_sent.to_numpy(); ta = d0 + rng.integers(60_000, 86_400_000 - 3_600_000, size=len(G))
            for s in (1, -1):
                m = sg == s
                if not m.any():
                    continue
                _, ps[m] = first_after(*book[-s][:2], end[m], end[m] + EXIT_WIN)
                tt, px, us = book[s]
                sent = np.where(np.isnan(ts_[m]), t1[m] - 1490, ts_[m]).astype(np.int64)
                pre[m] = last_before(tt, px, sent)
                for k in np.where(m)[0]:                      # competition (announcement known) and a random window
                    if np.isnan(ts_[k]):
                        continue
                    a, b = int(ts_[k]) + DELAY, int(t1[k])
                    if b <= a:
                        continue
                    lo, hi = np.searchsorted(tt, a), np.searchsorted(tt, b)
                    ncomp[k] = int((us[lo:hi] != G.user.iloc[k]).sum())
                    lo, hi = np.searchsorted(tt, ta[k]), np.searchsorted(tt, ta[k] + (b - a)); nctl[k] = hi - lo
                    p0 = last_before(tt, px, np.array([ta[k]]))[0]
                    _, p1 = first_after(tt, px, np.array([ta[k] + (b - a)]), np.array([ta[k] + (b - a) + BEFORE])); mctl[k] = s * np.log(p1[0] / p0) * 1e4
            R["net announcement"] = sg * np.log(ps / best) * 1e4 - 2 * fp
            R["net lower bound"] = sg * np.log(ps / worst) * 1e4 - 2 * fp
            R["move announcement -> 1st slice"] = sg * np.log(best / pre) * 1e4
            R["other takers before 1st slice"] = ncomp; R["control takers"] = nctl; R["control move"] = mctl
            R["minutes"] = (end - t1) / 60_000
            L.append(R)
    M = pd.concat(L, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    num = ["net announcement", "net lower bound", "net s06b", "move announcement -> 1st slice", "control move"]; M[num] = M[num].clip(-2000, 2000)
    M["gain"] = M["net announcement"] - M["net s06b"]
    M["duration class"] = pd.cut(M.minutes, DURATIONS, labels=DURATION_NAMES)
    M.to_parquet(WORK / "announcement_entry.parquet", index=False)
    show()
    V = M[M["net announcement"].notna() & M["net s06b"].notna()]
    g = V[(V.type == "perp") & (V["weight class"] == ">= 20 %")]
    pc = g.groupby("coin")["net announcement"].sum().sort_values(ascending=False); s3 = g[~g.coin.isin(pc.index[:3])]
    print("By duration:\n", g.groupby("duration class", observed=True).agg(n=("gain", "size"), announcement=("net announcement", "mean"),
          s06b=("net s06b", "mean"), gain=("gain", "mean")).to_string())
    k = M[M["other takers before 1st slice"].notna() & (M.type == "perp") & (M["weight class"] == ">= 20 %")]
    print(f"\nHeavy perps: another taker on my side between announcement and 1st slice in {(k['other takers before 1st slice'] > 0).mean():.1%} "
          f"of TWAPs, against {(k['control takers'] > 0).mean():.1%} in random windows")
    ok = g["net announcement"].mean() > 0 and t_day(g, "net announcement") >= 3 and g.gain.mean() > 0 and t_day(g, "gain") >= 3 and s3["net announcement"].mean() > 0
    print(f"Criterion: net {g['net announcement'].mean():.2f} bp (t {t_day(g, 'net announcement'):.2f}), gain {g.gain.mean():.2f} (t {t_day(g, 'gain'):.2f}), "
          f"without the top 3 {s3['net announcement'].mean():.2f} -> " + ("passed" if ok else "failed"))
