"""Section 6. Does selling back to the TWAP as a maker in its last quarter make following it pay?

Pre-registered (rule taken from s06a). Entry: first slice + 1 s at the price paid (s05b). Main exit: at 0.75 x the
duration, an order at the price of its last slice, filled at the next slice that strictly crosses it, else taker at
its end. Variants: optimistic fill (touched), windows at 0.5 and 0.9, plain taker at the end. Base fee schedule.
Criterion (perps, weight >= 20 %): net > 0, daily t >= 3, > 0 without the top 3 coins, and > 0 at 0.5 and 0.9.
    python s06b_maker_exit.py   -> work/maker_exit.parquet
"""
import json
import numpy as np, pandas as pd
from common import DAYS, WORK, coin_type, first_after, show, t_day
from s05b_heavy_executable import universe

ENTRY_WIN, EXIT_WIN = 60_000, 600_000
STARTS = [0.5, 0.75, 0.9]
SCHEDULE = {"perp": (4.5, 1.5), "spot": (7.0, 4.0)}      # (taker, maker) bp per side
rng = np.random.default_rng(41)

if __name__ == "__main__":
    X = universe().merge(pd.read_parquet(WORK / "twaps.parquet", columns=["user", "twapId", "end"]), on=["user", "twapId"])
    co = set(json.loads((WORK / "orders_coins.json").read_text())); X = X[X.coin.isin(co) & X["weight class"].isin(["5-20 %", ">= 20 %"])].reset_index(drop=True)
    X["type"] = X.coin.map(coin_type)
    f39 = pd.read_parquet(WORK / "heavy_executable.parquet", columns=["coin", "fees"]).groupby("coin").fees.first() / 2
    fees = {c: SCHEDULE.get(coin_type(c), (f39.get(c, 4.5), f39.get(c, 4.5) / 3)) for c in X.coin.unique()}
    L = []
    for d in DAYS:
        Xd = X[X.day == d]
        if len(Xd) == 0:
            continue
        i_ = DAYS.index(d); dd_ = [d] + ([DAYS[i_ + 1]] if i_ + 1 < len(DAYS) else [])
        Od = pd.concat([pd.read_parquet(WORK / "orders" / f"{x}.parquet", columns=["coin", "t", "side", "px"]) for x in dd_])
        Od = Od[Od.coin.isin(set(Xd.coin))]
        Tw = pd.concat([pd.read_parquet(WORK / "twap_fills" / f"{x}.parquet", columns=["user", "twapId", "t", "px"]) for x in dd_])
        Tw = Tw[Tw.twapId.isin(set(Xd.twapId))].groupby(["user", "twapId", "t"]).px.agg(["min", "max"]).reset_index()
        TW = {k: g for k, g in Tw.groupby(["user", "twapId"])}
        d0 = int(pd.Timestamp(d).value // 1_000_000)
        for coin, G in Xd.groupby("coin"):
            Oc = Od[Od.coin == coin].sort_values("t"); book = {s: (Oc[Oc.side == s].t.to_numpy(), Oc[Oc.side == s].px.to_numpy()) for s in (1, -1)}
            fp, fm = fees[coin]
            for r in G.itertuples(index=False):
                sg, t1, end = int(r.net_side), int(r.t1), int(r.end); dur = end - t1
                te, pe = first_after(*book[sg], np.array([t1 + 1000]), np.array([t1 + 1000 + ENTRY_WIN])); te, pe = te[0], pe[0]
                res = {"user": r.user, "twapId": r.twapId, "coin": coin, "day": d, "type": r.type, "minutes": dur / 60_000, "entered": not np.isnan(pe)}
                if not np.isnan(pe):
                    te = int(te)
                    _, px_end = first_after(*book[-sg], np.array([end]), np.array([end + EXIT_WIN])); px_end = px_end[0]
                    res["taker at end"] = sg * np.log(px_end / pe) * 1e4 - 2 * fp
                    g = TW.get((r.user, r.twapId))
                    for f in STARTS:
                        for mode in ("strict", "optimistic"):
                            ws = max(t1 + int(f * dur), te)
                            out, fr, filled = px_end, fp, False
                            if g is not None:
                                tt = g.t.to_numpy(); best = (g["min"] if sg > 0 else g["max"]).to_numpy(); worst = (g["max"] if sg > 0 else g["min"]).to_numpy()
                                k = np.searchsorted(tt, ws, side="right") - 1
                                if k >= 0:
                                    P = best[k]; nxt = np.arange(k + 1, len(tt))
                                    ok = (worst[nxt] > P if sg > 0 else worst[nxt] < P) if mode == "strict" else (worst[nxt] >= P if sg > 0 else worst[nxt] <= P)
                                    if ok.any():
                                        out, fr, filled = P, fm, True
                            res[f"maker {f:g} {mode}"] = sg * np.log(out / pe) * 1e4 - fp - fr
                            res[f"filled {f:g} {mode}"] = filled
                L.append(res)
            # control: random times, same duration, taker exit
            ta = d0 + rng.integers(60_000, 86_400_000 - 3_600_000, size=len(G)); sa = rng.choice([-1, 1], size=len(G)); dd = (G.end - G.t1).to_numpy()
            for k_ in range(len(G)):
                te, pe = first_after(*book[sa[k_]], np.array([ta[k_]]), np.array([ta[k_] + ENTRY_WIN])); te, pe = te[0], pe[0]
                v = np.nan
                if not np.isnan(pe):
                    _, ps = first_after(*book[-sa[k_]], np.array([int(te) + dd[k_]]), np.array([int(te) + dd[k_] + EXIT_WIN])); v = sa[k_] * np.log(ps[0] / pe) * 1e4 - 2 * fp
                L.append({"coin": coin, "day": d, "type": coin_type(coin), "control": True, "control net": v, "minutes": dd[k_] / 60_000})
    M = pd.DataFrame(L).replace([np.inf, -np.inf], np.nan)
    M = M.merge(X[["user", "twapId", "weight class", "weight1 %"]], on=["user", "twapId"], how="left")
    num = [c for c in M.columns if c.startswith(("maker", "taker", "control net"))]; M[num] = M[num].clip(-2000, 2000)
    M.to_parquet(WORK / "maker_exit.parquet", index=False)
    show()
    T = M[M.get("control") != True]
    g = T[(T.entered == True) & (T.type == "perp") & (T["weight class"] == ">= 20 %")]
    nets = ["taker at end"] + [f"maker {f:g} {m}" for f in STARTS for m in ("strict", "optimistic")]
    print(f"Perps, weight >= 20 %: {len(g)} TWAPs\n" + "\n".join(f"  {c}: {g[c].mean():.2f} bp (t {t_day(g, c):.2f})" for c in nets))
    pc = g.groupby("coin")["maker 0.75 strict"].sum().sort_values(ascending=False); top3 = list(pc.index[:3]); s3 = g[~g.coin.isin(top3)]
    m = g["maker 0.75 strict"]
    ok = m.mean() > 0 and t_day(g, "maker 0.75 strict") >= 3 and s3["maker 0.75 strict"].mean() > 0 and g["maker 0.5 strict"].mean() > 0 and g["maker 0.9 strict"].mean() > 0
    print(f"Criterion: net {m.mean():.2f} bp (t {t_day(g, 'maker 0.75 strict'):.2f}), without the top 3 {s3['maker 0.75 strict'].mean():.2f}, "
          f"0.5: {g['maker 0.5 strict'].mean():.2f}, 0.9: {g['maker 0.9 strict'].mean():.2f} -> " + ("passed" if ok else "failed"))
