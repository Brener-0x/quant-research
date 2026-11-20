"""Section 8. Exit A without look-ahead.

"Right after the last slice" is known only after the fact: for a cancelled TWAP nobody knows that time in advance.
Live: announced end (send time + duration from the API) if the TWAP completes; if it is cancelled, the cancellation is
read from the API (time field of twapHistory, to the second; live: the userTwapHistory websocket of the wallet):
exit at cancellation + 2 s. No announcement found: exit B (35 s without a slice). Base: s08b.
    python s08c_exit_a.py   -> work/e2.parquet (columns added)
"""
import json
import numpy as np, pandas as pd
from common import DAYS, TAKER_FEE, WORK, first_after, show, t_day
from s07c_reasons import SELECT, VERIFY
from s08a_entries import ENTRY_WIN, EXIT_WIN, day_after

LAT = 2000      # cancellation (to the second) -> my order

if __name__ == "__main__":
    K = pd.read_parquet(WORK / "e2.parquet")
    K = K[[c for c in K.columns if not c.startswith(("A2 ", "t cancel"))]]
    canc = {}
    for u in K.user.unique():
        f = WORK / "api" / f"{u}.json"
        if f.exists():
            for e in json.loads(f.read_text())["kept"]:
                if e.get("twapId") is not None and e["status"]["status"] == "terminated":
                    canc[(u, e["twapId"])] = e["time"] * 1000
    K["t cancel"] = [canc.get((u, i), np.nan) for u, i in zip(K.user, K.twapId)]
    K["announced end"] = K.t_sent + K.minutes * 60_000
    ts = np.where(K.cancelled.fillna(False).astype(bool), K["t cancel"] + LAT, K["announced end"])
    K["t exit A2"] = np.where(K.found, ts, np.nan)
    L = []
    for d in DAYS:
        Sd = K[(K.day == d) & K["t exit A2"].notna()]
        if len(Sd) == 0:
            continue
        dd_ = [d] + ([day_after(d)] if day_after(d) else [])
        Od = pd.concat([pd.read_parquet(WORK / "orders" / f"{x}.parquet", columns=["coin", "t", "side", "px"]) for x in dd_]); Od = Od[Od.coin.isin(set(Sd.coin))]
        for coin, G in Sd.groupby("coin"):
            Oc = Od[Od.coin == coin].sort_values("t"); book = {s: (Oc[Oc.side == s].t.to_numpy(), Oc[Oc.side == s].px.to_numpy()) for s in (1, -1)}
            for r in G.to_dict("records"):
                sg, t1 = int(r["side"]), int(r["t1"]); tsx = max(int(r["t exit A2"]), t1 + 2000)
                _, pS = first_after(*book[-sg], np.array([tsx]), np.array([tsx + EXIT_WIN]))
                res = {"user": r["user"], "twapId": r["twapId"]}
                for dl in (1000, 2000):
                    _, pe = first_after(*book[sg], np.array([t1 + dl]), np.array([t1 + dl + ENTRY_WIN]))
                    res[f"A2 +{dl}"] = np.clip(sg * np.log(pS[0] / pe[0]) * 1e4 - 2 * TAKER_FEE, -2000, 2000)
                L.append(res)
    R = pd.DataFrame(L).replace([np.inf, -np.inf], np.nan)
    K = K.merge(R, on=["user", "twapId"], how="left")
    for dl in ("1000", "2000"):
        K[f"A2 +{dl}"] = K[f"A2 +{dl}"].where(K["t exit A2"].notna(), K[f"B +{dl}"])
    K.to_parquet(WORK / "e2.parquet", index=False)
    show()
    an = K.cancelled.fillna(False).astype(bool) & K.found
    nd = K.day.nunique(); rows = []
    for c, name in (("A +2000", "A, last slice known after the fact"), ("A2 +2000", "A without look-ahead (announced end, or cancellation + 2 s)"),
                    ("Ah +2000", "A without following cancellations (cancelled -> B)"), ("B +2000", "B (35 s without a slice)")):
        g = K[K[c].notna()]
        rows.append({"exit (entry E2, + 2 s)": name, "n": len(g), "net": g[c].mean(), "t": t_day(g, c), "bp per day": g[c].sum() / nd,
                     "cancelled": g.loc[an.reindex(g.index), c].mean(), "completed": g.loc[~an.reindex(g.index), c].mean()})
    print(pd.DataFrame(rows).to_string(index=False))
    print("By month, exit A without look-ahead:", K.groupby("month")["A2 +2000"].agg(["size", "mean"]).round(2).to_dict("index"))
